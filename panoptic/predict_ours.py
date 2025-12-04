import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
sys.path.append(str(ROOT)) if str(ROOT) not in sys.path else None
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
    colorstr, cv2, increment_path, non_max_suppression, print_args,
    scale_boxes, scale_segments, strip_optimizer, xyxy2xywh
)
from utils.plots import colors
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode


def draw_semantic_mask(im0, semantic_logits, stuff_class_id, alpha):
    """绘制语义分割掩码（Stuff 类别）"""
    if semantic_logits is None:
        return im0

    H, W = im0.shape[:2]
    semantic_pred = torch.argmax(semantic_logits, dim=0).cpu().numpy().astype(np.uint8)
    semantic_resized = cv2.resize(semantic_pred, (W, H), interpolation=cv2.INTER_NEAREST)

    stuff_mask = (semantic_resized == stuff_class_id)
    if np.any(stuff_mask):
        color_layer = np.zeros_like(im0, dtype=np.uint8)
        color_layer[stuff_mask] = colors(stuff_class_id, True)
        im0 = cv2.addWeighted(im0, 1.0, color_layer, alpha, 0)
    return im0


def draw_instance_masks(im0, det, mask_proto, im_shape, names,
                        save_txt=False, save_conf=False,
                        line_thickness=3, hide_labels=False, hide_conf=False, alpha=0.5):
    """绘制实例掩码（Thing 类别）+ 边界框"""
    segments = []
    if not len(det):
        return im0, segments

    # 计算实例掩码 & 边界框映射
    masks = process_mask(mask_proto, det[:, 6:], det[:, :4], im_shape, upsample=True)
    det[:, :4] = scale_boxes(im_shape, det[:, :4], im0.shape).round()  # 映射到原图尺寸

    if save_txt:
        segments = [scale_segments(im_shape, seg, im0.shape, normalize=True)
                    for seg in reversed(masks2segments(masks))]

    masks_np = masks.permute(1, 2, 0).cpu().numpy()
    H, W = im0.shape[:2]

    # 绘制每个实例
    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
        c = int(cls)
        mask_resized = cv2.resize(masks_np[:, :, j], (W, H), interpolation=cv2.INTER_LINEAR)
        mask_bool = (mask_resized > 0.5)

        color_layer = np.zeros_like(im0, dtype=np.uint8)
        color_layer[mask_bool] = colors(c, True)
        im0 = cv2.addWeighted(im0, 1.0, color_layer, alpha, 0)

        # 绘制边界框
        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(im0, p1, p2, colors(c, True), thickness=line_thickness)

        # 绘制标签
        if not hide_labels:
            label = names[c] if hide_conf else f'{names[c]} {conf:.2f}'
            tf = max(line_thickness - 1, 1)
            w_text, h_text = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
            cv2.rectangle(im0, p1, (p1[0] + w_text, p1[1] - h_text - 3), colors(c, True), -1)
            cv2.putText(im0, label, (p1[0], p1[1] - 2), 0,
                        line_thickness / 3, [255, 255, 255], thickness=tf)
    return im0, segments


@smart_inference_mode()
def run(**kwargs):
    source = str(kwargs['source'])
    # 数据源类型判断
    suffix = Path(source).suffix[1:]
    is_file = suffix in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)

    save_img = not kwargs['nosave'] and not source.endswith('.txt')

    # 输出目录
    save_dir = increment_path(Path(kwargs['project']) / kwargs['name'], exist_ok=kwargs['exist_ok'])
    (save_dir / 'labels' if kwargs['save_txt'] else save_dir).mkdir(parents=True, exist_ok=True)

    # 模型加载
    device = select_device(kwargs['device'])
    model = DetectMultiBackend(kwargs['weights'], device=device, dnn=kwargs['dnn'],
                               data=kwargs['data'], fp16=kwargs['half'])
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(kwargs['imgsz'], s=stride)

    # 数据加载器
    bs = 1
    if webcam:
        kwargs['view_img'] = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=kwargs['vid_stride'])
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=kwargs['vid_stride'])
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 推理
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    seen, windows, dt_profiles = 0, [], (Profile(), Profile(), Profile())

    for idx, (path, im, im0s, vid_cap, s) in enumerate(dataset, start=1):
        if idx >= 500:
            break

        # 预处理
        with dt_profiles[0]:
            im = torch.from_numpy(im).to(model.device).float()
            if model.fp16:
                im = im.half()
            im /= 255
            if im.ndim == 3:
                im = im.unsqueeze(0)

        # 模型推理
        with dt_profiles[1]:
            visual_path = increment_path(save_dir / Path(path).stem, mkdir=True) if kwargs['visualize'] else False
            pred, panoptic_outs = model(im, augment=kwargs['augment'], visualize=visual_path)
            mask_proto, semantic_logits = panoptic_outs[2], panoptic_outs[3]

        # NMS
        with dt_profiles[2]:
            pred = non_max_suppression(pred, kwargs['conf_thres'], kwargs['iou_thres'],
                                       kwargs['classes'], kwargs['agnostic_nms'],
                                       kwargs['max_det'], nm=32)

        # 逐帧处理
        for i, det in enumerate(pred):
            seen += 1
            im0 = im0s.copy() if not webcam else im0s[i].copy()

            im0 = draw_semantic_mask(im0, semantic_logits[i], kwargs.get('STUFF_ID_TO_PLOT', 7), kwargs.get('ALPHA', 0.5))
            im0, segments = draw_instance_masks(im0, det, mask_proto[i], im.shape[2:], names,
                                                save_txt=kwargs['save_txt'],
                                                save_conf=kwargs['save_conf'],
                                                line_thickness=kwargs['line_thickness'],
                                                hide_labels=kwargs['hide_labels'],
                                                hide_conf=kwargs['hide_conf'],
                                                alpha=kwargs.get('ALPHA', 0.5))
            cv2.imwrite(save_dir / f"rack{idx}.png", im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt_profiles[1].dt * 1E3:.1f}ms")

    # 结果统计
    t = tuple(x.t / seen * 1E3 for x in dt_profiles)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS at shape {(1, 3, *imgsz)}" % t)
    if kwargs['save_txt'] or save_img:
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    if kwargs['update']:
        strip_optimizer(kwargs['weights'][0])

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'gelan-c-pan.pt')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--max-det', type=int, default=100)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--view-img', type=bool, default=False, choices=[True, False])
    parser.add_argument('--save-txt', type=bool, default=False, choices=[True, False])
    parser.add_argument('--save-conf', type=bool, default=False, choices=[True, False])
    parser.add_argument('--nosave', type=bool, default=False, choices=[True, False])
    parser.add_argument('--classes', nargs='+', type=int, default=None)
    parser.add_argument('--agnostic-nms', type=bool, default=False, choices=[True, False])
    parser.add_argument('--augment', type=bool, default=False, choices=[True, False])
    parser.add_argument('--visualize', type=bool, default=False, choices=[True, False])
    parser.add_argument('--update', type=bool, default=False, choices=[True, False])
    parser.add_argument('--project', type=str, default=ROOT / 'runs/predict-seg')
    parser.add_argument('--name', type=str, default='rack_result')
    parser.add_argument('--exist-ok', type=bool, default=False, choices=[True, False])
    parser.add_argument('--line-thickness', type=int, default=3)
    parser.add_argument('--hide-labels', type=bool, default=False, choices=[True, False])
    parser.add_argument('--hide-conf', type=bool, default=False, choices=[True, False])
    parser.add_argument('--half', type=bool, default=False, choices=[True, False])
    parser.add_argument('--dnn', type=bool, default=False, choices=[True, False])
    parser.add_argument('--vid-stride', type=int, default=1)
    parser.add_argument('--retina-masks', type=bool, default=False, choices=[True, False])

    opt = parser.parse_args()
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    main(parse_opt())