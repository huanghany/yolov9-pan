# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : predict_rack_online.py
# @Create      : 2025/12/8-19:51
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2025, ZenoAI Robot Inc. All Rights Reserved.
# @Description : 作物架实时推理

import argparse
import os
import sys
from pathlib import Path
import time

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
    LOGGER, Profile, check_file, check_img_size, check_imshow,
    colorstr, cv2, increment_path, non_max_suppression, print_args,
    scale_boxes, scale_segments, strip_optimizer
)
from utils.plots import colors
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode


def draw_semantic_mask(im0, semantic_logits, stuff_class_id, alpha, im_shape=None):
    if semantic_logits is None:
        return im0
    H, W = im0.shape[:2]
    semantic_pred = torch.argmax(semantic_logits, dim=0).cpu().numpy().astype(np.uint8)

    if im_shape is None:
        semantic_resized = cv2.resize(semantic_pred, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        semantic_resized = cv2.resize(semantic_pred, (im_shape[1], im_shape[0]), interpolation=cv2.INTER_NEAREST)
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
    segments = []
    if not len(det):
        return im0, segments
    det[:, :4] = scale_boxes(im_shape, det[:, :4], im0.shape).round()
    masks = process_mask(mask_proto, det[:, 6:], det[:, :4], im0.shape[:2], upsample=True)

    if save_txt:
        segments = [scale_segments(im_shape, seg, im0.shape, normalize=True)
                    for seg in reversed(masks2segments(masks))]
    masks_np = masks.permute(1, 2, 0).cpu().numpy()
    H, W = im0.shape[:2]
    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
        c = int(cls)
        mask_bool = masks_np[:, :, j] > 0.5
        color_layer = np.zeros_like(im0, dtype=np.uint8)
        color_layer[mask_bool] = colors(c, True)
        im0 = cv2.addWeighted(im0, 1.0, color_layer, alpha, 0)

        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(im0, p1, p2, colors(c, True), thickness=line_thickness)

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
    # snapshot_dir = save_dir / 'snapshots'
    snapshot_dir = save_dir
    snapshot_dir.mkdir(parents=True, exist_ok=True)

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
    save_video = kwargs.get('save_video', True)
    ref_pos = kwargs.get('ref_pos', 0.5)
    csv_results = []

    video_path = str(save_dir / 'inference_output.mp4')
    video_writer = None
    video_fps = 30

    for idx, (path, im, im0s, vid_cap, s) in enumerate(dataset, start=1):
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

            im0 = draw_semantic_mask(im0, semantic_logits[i], stuff_id,
                                     kwargs.get('ALPHA', 0.5), im.shape[2:])
            im0, segments = draw_instance_masks(im0, det, mask_proto[i], im.shape[2:], names,
                                                save_txt=kwargs['save_txt'],
                                                save_conf=kwargs['save_conf'],
                                                line_thickness=kwargs['line_thickness'],
                                                hide_labels=kwargs['hide_labels'],
                                                hide_conf=kwargs['hide_conf'],
                                                alpha=kwargs.get('ALPHA', 0.5))
            H, W = im0.shape[:2]
            sem_pred = torch.argmax(semantic_logits[i], dim=0)
            sem_resized = cv2.resize(sem_pred.cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            mask_binary = (sem_resized == stuff_id).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_binary).to(model.device)
            mask_binary = remove_edge_noise(mask_binary, crop_ratio=0.03)
            y_norm, dy_norm = center_estimator.compute(mask_tensor, ref_pos=ref_pos, return_cpu=True)
            y_center_px = int(y_norm[0] * (H - 1))
            ref_px = int(ref_pos * (H - 1))
            cv2.line(im0, (0, y_center_px), (W - 1, y_center_px), (0, 255, 0), 2)
            cv2.line(im0, (0, ref_px), (W - 1, ref_px), (0, 0, 255), 1)

            if save_img:
                cv2.imwrite(str(save_dir / f"rack{idx}.png"), im0)

            if save_video:
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (W, H))
                video_writer.write(im0)

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

            if kwargs['view_img']:
                cv2.imshow("Real-time Inference", im0)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    LOGGER.info("[INFO] Exit by user pressing 'q'")
                    if save_video and video_writer is not None:
                        video_writer.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('s'):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    snap_path = snapshot_dir / f"snapshot_{ts}.png"
                    cv2.imwrite(str(snap_path), im0)
                    LOGGER.info(f"[INFO] Snapshot saved: {snap_path}")

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt_profiles[1].dt * 1E3:.1f}ms")

    if save_video and video_writer is not None:
        video_writer.release()
        LOGGER.info(f"[INFO] Video saved to {video_path}")

    if save_csv and csv_results:
        import csv
        csv_path = save_dir / "semantic_centers.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
            writer.writeheader()
            writer.writerows(csv_results)
        LOGGER.info(f"[INFO] Center line data saved to {csv_path}")

    t = tuple(x.t / seen * 1E3 for x in dt_profiles)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS at shape {(1, 3, *imgsz)}" % t)

    if kwargs['save_txt'] or save_img:
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    if kwargs['update']:
        strip_optimizer(kwargs['weights'][0])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/hhy/Project/yolov9-pan/yolov9-pan-strawberry7cls_rackcls1_v2.pt')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--data', type=str, default='/home/huanghanyang/Project/yolov9/data/rack-v2.yaml')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.8)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--max-det', type=int, default=100)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--view-img', type=bool, default=True, choices=[True, False])
    parser.add_argument('--save-txt', type=bool, default=False, choices=[True, False])
    parser.add_argument('--save-csv', type=bool, default=False, choices=[True, False])
    parser.add_argument('--save-video', type=bool, default=False, choices=[True, False], help='whether to save inference video')
    parser.add_argument('--save-conf', type=bool, default=False, choices=[True, False])
    parser.add_argument('--nosave', type=bool, default=True, choices=[True, False])
    parser.add_argument('--classes', nargs='+', type=int, default=None)
    parser.add_argument('--agnostic-nms', type=bool, default=True, choices=[True, False])
    parser.add_argument('--augment', type=bool, default=False, choices=[True, False])
    parser.add_argument('--visualize', type=bool, default=False, choices=[True, False])
    parser.add_argument('--update', type=bool, default=False, choices=[True, False])
    parser.add_argument('--project', type=str, default=ROOT / 'runs/predict-seg')
    parser.add_argument('--name', type=str, default='rack_online')
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
    run(**vars(opt))


if __name__ == "__main__":
    main(parse_opt())