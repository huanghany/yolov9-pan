# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : predict_rack_line.py
# @Create      : 2025/12/8-20:01
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2025, Huayi Robot Inc. All Rights Reserved.
# @Description : 作物架离线推理（图片/视频输入）
import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from utils.mask_center_cal import MaskCenterEstimator

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

def draw_semantic_mask(im0, semantic_logits, stuff_class_id, alpha, im_shape=None):
    """绘制语义分割掩码（Stuff 类别），修复偏移"""
    if semantic_logits is None:
        return im0

    H, W = im0.shape[:2]
    semantic_pred = torch.argmax(semantic_logits, dim=0).cpu().numpy().astype(np.uint8)

    if im_shape is None:
        # 默认用原图尺寸直接拉伸，会有一定偏移（如果使用了 letterbox）
        semantic_resized = cv2.resize(semantic_pred, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        # 按模型输入尺寸先恢复，再映射到原图尺寸
        semantic_resized = cv2.resize(semantic_pred, (im_shape[1], im_shape[0]), interpolation=cv2.INTER_NEAREST)
        # letterbox 反变换到原图大小
        semantic_resized = cv2.resize(semantic_resized, (W, H), interpolation=cv2.INTER_NEAREST)

    stuff_mask = (semantic_resized == stuff_class_id)
    if np.any(stuff_mask):
        color_layer = np.zeros_like(im0, dtype=np.uint8)
        color_layer[stuff_mask] = colors(stuff_class_id, True)
        im0 = cv2.addWeighted(im0, 1.0, color_layer, alpha, 0)
    return im0


def draw_instance_masks(im0, det, mask_proto, im_shape, names,
                        save_txt=False, save_conf=False,
                        line_thickness=3, hide_labels=False, hide_conf=False, alpha=0.5):
    """
    绘制实例掩码（Thing 类别）+ 边界框
    修复坐标系不统一导致的 mask 偏移问题
    """
    segments = []
    if not len(det):
        return im0, segments

    # 先映射检测框到原图尺寸
    det[:, :4] = scale_boxes(im_shape, det[:, :4], im0.shape).round()

    # 在原图尺寸中生成mask（直接对齐原图）
    masks = process_mask(mask_proto, det[:, 6:], det[:, :4], im0.shape[:2], upsample=True)

    if save_txt:
        segments = [scale_segments(im_shape, seg, im0.shape, normalize=True)
                    for seg in reversed(masks2segments(masks))]

    # masks: [H, W, num_instances], 已是原图尺寸
    masks_np = masks.permute(1, 2, 0).cpu().numpy()
    H, W = im0.shape[:2]

    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
        c = int(cls)

        # 因为 process_mask 已生成原图尺寸，这里无需再次 resize
        mask_bool = masks_np[:, :, j] > 0.5
        # 绘制掩码颜色层
        color_layer = np.zeros_like(im0, dtype=np.uint8)
        color_layer[mask_bool] = colors(c, True)
        im0 = cv2.addWeighted(im0, 1.0, color_layer, alpha, 0)

        # 绘制边框
        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(im0, p1, p2, colors(c, True), thickness=line_thickness)

        # 绘制标签
        if not hide_labels:
            label = names[c] if hide_conf else f'{names[c]} {conf:.2f}'
            tf = max(line_thickness - 1, 1)
            w_text, h_text = cv2.getTextSize(label, fontFace=0,
                                             fontScale=line_thickness / 3,
                                             thickness=tf)[0]
            cv2.rectangle(im0, p1, (p1[0] + w_text, p1[1] - h_text - 3),
                          colors(c, True), -1)
            cv2.putText(im0, label, (p1[0], p1[1] - 2), 0,
                        line_thickness / 3, [255, 255, 255], thickness=tf)
    return im0, segments

def remove_edge_noise(mask_binary, crop_ratio=0.02):
    """
    去除左右边缘噪声区域。
    crop_ratio: 占图像宽度的比例，例如 0.02 表示左右各裁掉 2%
    """
    H, W = mask_binary.shape
    crop_px = int(W * crop_ratio)
    mask_binary[:, :crop_px] = 0
    mask_binary[:, -(crop_px):] = 0
    return mask_binary

@smart_inference_mode()
def run(**kwargs):
    source = str(kwargs['source'])
    suffix = Path(source).suffix[1:]
    is_file = suffix in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)

    save_img = not kwargs['nosave'] and not source.endswith('.txt')
    save_dir = increment_path(Path(kwargs['project']) / kwargs['name'], exist_ok=kwargs['exist_ok'])
    (save_dir / 'labels' if kwargs['save_txt'] else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(kwargs['device'])
    model = DetectMultiBackend(kwargs['weights'], device=device, dnn=kwargs['dnn'],
                               data=kwargs['data'], fp16=kwargs['half'])
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(kwargs['imgsz'], s=stride)

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

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    seen, windows, dt_profiles = 0, [], (Profile(), Profile(), Profile())
    center_estimator = MaskCenterEstimator(kernel_size=5, use_bbox=False, use_ema=True)
    save_csv = kwargs.get('save_csv', True)
    ref_pos = kwargs.get('ref_pos', 0.5)
    csv_results = []

    # === 新增: 视频保存 ===
    video_path = str(save_dir / 'inference_output.mp4')
    video_writer = None
    video_fps = 30  # 默认帧率，可根据实际视频修改

    for idx, (path, im, im0s, vid_cap, s) in enumerate(dataset, start=1):
        if idx ==50:
            break
        # 预处理
        with dt_profiles[0]:
            im = torch.from_numpy(im).to(model.device).float()
            if model.fp16:
                im = im.half()
            im /= 255
            if im.ndim == 3:
                im = im.unsqueeze(0)

        with dt_profiles[1]:
            visual_path = increment_path(save_dir / Path(path).stem, mkdir=True) if kwargs['visualize'] else False
            pred, panoptic_outs = model(im, augment=kwargs['augment'], visualize=visual_path)
            mask_proto, semantic_logits = panoptic_outs[2], panoptic_outs[3]

        with dt_profiles[2]:
            pred = non_max_suppression(pred, kwargs['conf_thres'], kwargs['iou_thres'],
                                       kwargs['classes'], kwargs['agnostic_nms'],
                                       kwargs['max_det'], nm=32)

        for i, det in enumerate(pred):
            seen += 1
            im0 = im0s.copy() if not webcam else im0s[i].copy()
            stuff_id = kwargs.get('STUFF_ID_TO_PLOT', 7)

            # === 原来的 mask + bbox 绘制 ===
            im0 = draw_semantic_mask(
                im0, semantic_logits[i], stuff_id,
                kwargs.get('ALPHA', 0.5), im.shape[2:]
            )
            im0, segments = draw_instance_masks(im0, det, mask_proto[i], im.shape[2:], names,
                                                save_txt=kwargs['save_txt'],
                                                save_conf=kwargs['save_conf'],
                                                line_thickness=kwargs['line_thickness'],
                                                hide_labels=kwargs['hide_labels'],
                                                hide_conf=kwargs['hide_conf'],
                                                alpha=kwargs.get('ALPHA', 0.5))

            # === 新的中心线计算 ===
            H, W = im0.shape[:2]
            sem_pred = torch.argmax(semantic_logits[i], dim=0)
            sem_resized = cv2.resize(sem_pred.cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            mask_binary = (sem_resized == stuff_id).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_binary).to(model.device)
            mask_binary = remove_edge_noise(mask_binary, crop_ratio=0.03)
            y_norm, dy_norm = center_estimator.compute(mask_tensor, ref_pos=ref_pos, return_cpu=True)
            y_center_px = int(y_norm[0] * (H - 1))
            ref_px      = int(ref_pos * (H - 1))
            cv2.line(im0, (0, y_center_px), (W - 1, y_center_px), (0, 255, 0), 2)
            cv2.line(im0, (0, ref_px), (W - 1, ref_px), (0, 0, 255), 1)

            # === 图片保存 (原有保留) ===
            if save_img:
                cv2.imwrite(str(save_dir / f"rack{idx}.png"), im0)

            # === 视频帧写入 ===
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (W, H))
            video_writer.write(im0)

            # === CSV 保存数据 ===
            if save_csv:
                csv_results.append({
                    'filename': Path(path).name,
                    'y_center_norm': float(y_norm[0]),
                    'dy_norm': float(dy_norm[0]),
                    'y_center_pixel': y_center_px,
                    'ref_pixel': ref_px,
                    'image_height': H,
                    'image_width': W
                })

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt_profiles[1].dt * 1E3:.1f}ms")

    # === 视频释放 ===
    if video_writer is not None:
        video_writer.release()
        LOGGER.info(f"[INFO] Video saved to {video_path}")

    # === CSV 保存 ===
    if save_csv and csv_results:
        import csv
        csv_path = save_dir / "semantic_centers.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
            writer.writeheader()
            writer.writerows(csv_results)
        LOGGER.info(f"[INFO] Center line data saved to {csv_path}")

    # === 时间统计 ===
    t = tuple(x.t / seen * 1E3 for x in dt_profiles)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS at shape {(1, 3, *imgsz)}" % t)

    if kwargs['save_txt'] or save_img:
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    if kwargs['update']:
        strip_optimizer(kwargs['weights'][0])

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/hhy/Project/yolov9-pan/weights/yolov9-pan-strawberry7cls_rackcls1_v2.pt')
    # parser.add_argument('--source', type=str, default='/home/huanghanyang/Project/yolov9/data/images')
    # parser.add_argument('--source', type=str, default='/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v1_coco/images/test')
    parser.add_argument('--source', type=str, default='/home/hhy/Project/yolov9-pan/data/images')
    # parser.add_argument('--source', type=str, default='/home/hhy/Datasets/strawberry/monit/20250321-180245_rack-10_right_layer-1_RGB.mp4')
    # parser.add_argument('--source', type=str, default='/home/huanghanyang/Datasets/rack_datasets/project-145/images_origin')
    parser.add_argument('--data', type=str, default='/home/huanghanyang/Project/yolov9/data/rack-v2.yaml')
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
    parser.add_argument('--agnostic-nms', type=bool, default=True, choices=[True, False])
    parser.add_argument('--augment', type=bool, default=False, choices=[True, False])
    parser.add_argument('--visualize', type=bool, default=False, choices=[True, False])
    parser.add_argument('--update', type=bool, default=False, choices=[True, False])
    parser.add_argument('--project', type=str, default=ROOT / 'runs/predict-seg')
    parser.add_argument('--name', type=str, default='rack_result_v2_')
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
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    main(parse_opt())