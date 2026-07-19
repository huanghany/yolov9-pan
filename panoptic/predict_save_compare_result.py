# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : predict_save_compare_result.py
# @Create      : 2026/4/17-16:41
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2025, ZenoAI Inc. All Rights Reserved.
# @Description : 保存两个模型推理结果，方便评估

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
sys.path.append(str(ROOT)) if str(ROOT) not in sys.path else None
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, colorstr, increment_path,
    non_max_suppression, print_args, scale_boxes, scale_segments
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


def draw_instance_masks(im0, det, mask_proto, im_shape, names, line_thickness=3, alpha=0.5):
    if not len(det):
        return im0
    det = det.clone()  # 避免修改原始预测
    det[:, :4] = scale_boxes(im_shape, det[:, :4], im0.shape).round()
    masks = process_mask(mask_proto, det[:, 6:], det[:, :4], im0.shape[:2], upsample=True)
    masks_np = masks.permute(1, 2, 0).cpu().numpy()

    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
        c = int(cls)
        mask_bool = masks_np[:, :, j] > 0.5
        color_layer = np.zeros_like(im0, dtype=np.uint8)
        color_layer[mask_bool] = colors(c, True)
        im0 = cv2.addWeighted(im0, 1.0, color_layer, alpha, 0)
        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(im0, p1, p2, colors(c, True), thickness=line_thickness)
        label = f'{names[c]} {conf:.2f}'
        tf = max(line_thickness - 1, 1)
        w_text, h_text = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
        cv2.rectangle(im0, p1, (p1[0] + w_text, p1[1] - h_text - 3), colors(c, True), -1)
        cv2.putText(im0, label, (p1[0], p1[1] - 2), 0, line_thickness / 3, [255, 255, 255], thickness=tf)
    return im0


@smart_inference_mode()
def run(**kwargs):
    source = str(kwargs['source'])
    weights_list = kwargs['weights']  # 应该是一个包含两个路径的列表
    if len(weights_list) < 2:
        LOGGER.error("Please provide at least two weight files for comparison.")
        return

    save_dir = increment_path(Path(kwargs['project']) / kwargs['name'], exist_ok=kwargs['exist_ok'])
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(kwargs['device'])

    # --- 加载两个模型 ---
    LOGGER.info(f'Loading Model 1: {weights_list[0]}')
    model1 = DetectMultiBackend(weights_list[0], device=device, dnn=kwargs['dnn'], data=kwargs['data'],
                                fp16=kwargs['half'])

    LOGGER.info(f'Loading Model 2: {weights_list[1]}')
    model2 = DetectMultiBackend(weights_list[1], device=device, dnn=kwargs['dnn'], data=kwargs['data'],
                                fp16=kwargs['half'])

    stride = max(model1.stride, model2.stride)
    names1, names2 = model1.names, model2.names
    imgsz = check_img_size(kwargs['imgsz'], s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)

    model1.warmup(imgsz=(1, 3, *imgsz))
    model2.warmup(imgsz=(1, 3, *imgsz))

    for idx, (path, im, im0s, vid_cap, s) in enumerate(dataset, start=1):
        im = torch.from_numpy(im).to(device).float()
        im /= 255
        if im.ndim == 3:
            im = im.unsqueeze(0)
        if kwargs['half']:
            im = im.half()

        # --- 模型 1 推理 ---
        pred1, proto1 = model1(im)[:2]  # 获取推理和泛化输出
        mask_proto1, sem1 = proto1[2], proto1[3]
        p1 = non_max_suppression(pred1, kwargs['conf_thres'], kwargs['iou_thres'], kwargs['classes'],
                                 kwargs['agnostic_nms'], max_det=kwargs['max_det'], nm=32)

        # --- 模型 2 推理 ---
        pred2, proto2 = model2(im)[:2]
        mask_proto2, sem2 = proto2[2], proto2[3]
        p2 = non_max_suppression(pred2, kwargs['conf_thres'], kwargs['iou_thres'], kwargs['classes'],
                                 kwargs['agnostic_nms'], max_det=kwargs['max_det'], nm=32)

        # --- 处理与绘图 ---
        im_out1 = im0s.copy()
        im_out2 = im0s.copy()

        # 绘制模型 1
        im_out1 = draw_semantic_mask(im_out1, sem1[0], kwargs.get('STUFF_ID_TO_PLOT', 7), kwargs.get('ALPHA', 0.5),
                                     im.shape[2:])
        im_out1 = draw_instance_masks(im_out1, p1[0], mask_proto1[0], im.shape[2:], names1, kwargs['line_thickness'])
        # cv2.putText(im_out1, f"Model A: {Path(weights_list[0]).stem}", (20, 40), 0, 1, (0, 255, 0), 2)
        cv2.putText(im_out1, f"Model A: v2", (20, 40), 0, 1, (0, 255, 0), 2)

        # 绘制模型 2
        im_out2 = draw_semantic_mask(im_out2, sem2[0], kwargs.get('STUFF_ID_TO_PLOT', 7), kwargs.get('ALPHA', 0.5),
                                     im.shape[2:])
        im_out2 = draw_instance_masks(im_out2, p2[0], mask_proto2[0], im.shape[2:], names2, kwargs['line_thickness'])
        # cv2.putText(im_out2, f"Model B: {Path(weights_list[1]).stem}", (20, 40), 0, 1, (0, 255, 0), 2)
        cv2.putText(im_out2, f"Model B: v4", (20, 40), 0, 1, (0, 255, 0), 2)

        # --- 左右拼接 ---
        combined_img = np.hstack((im_out1, im_out2))

        # 保存
        save_path = str(save_dir / f"compare_{idx:04d}_{Path(path).stem}.png")
        cv2.imwrite(save_path, combined_img)
        LOGGER.info(f"Saved comparison: {save_path}")


def parse_opt():
    parser = argparse.ArgumentParser()
    # 这里默认传入两个权重路径
    parser.add_argument('--weights', nargs='+', type=str, default=[
        '/home/huanghanyang/Project/yolov9/runs/train-pan/strawberry_rack_v2/weights/yolov9-pan-strawberry7cls_rackcls1_v2.pt',
        '/home/huanghanyang/Project/yolov9/runs/train-pan/strawberry_rack_v4/weights/yolov9-pan-strawberry7cls_rackcls1_v4_0.pt'
    ])
    parser.add_argument('--source', type=str, default='/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v4_0_cls1/images/test')
    parser.add_argument('--data', type=str, default='/home/huanghanyang/Project/yolov9/data/rack-v4.yaml')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--max-det', type=int, default=100)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--project', type=str, default=ROOT / 'runs/compare-seg')
    parser.add_argument('--name', type=str, default='compare_v2_v3')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--line-thickness', type=int, default=2)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int, default=None)
    parser.add_argument('--agnostic-nms', action='store_true')
    # 兼容性冗余参数
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--vid-stride', type=int, default=1)

    opt = parser.parse_args()
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
