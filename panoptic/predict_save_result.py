# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : predict_save_result.py
# @Create      : 2026/4/13-18:17
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2025, ZenoAI Inc. All Rights Reserved.
# @Description : 推理并保存结果（用于制作伪标签）


import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2

# 设置路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
sys.path.append(str(ROOT)) if str(ROOT) not in sys.path else None

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    check_file, check_img_size, increment_path, non_max_suppression,
    print_args, scale_boxes, scale_segments, LOGGER
)
from utils.plots import colors
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode


def save_yolo_poly_labels(path, segments, class_id):
    """将多边形段保存为 YOLO 格式文本"""
    with open(path, 'a') as f:
        for seg in segments:
            # seg 是归一化后的坐标 (n, 2)
            line = f"{class_id} " + " ".join([f"{x:.6f}" for x in seg.reshape(-1)])
            f.write(line + "\n")


def get_stuff_segments(semantic_logits, stuff_id, target_shape):
    """从语义分割结果中提取多边形并归一化"""
    semantic_pred = torch.argmax(semantic_logits, dim=0).cpu().numpy().astype(np.uint8)
    mask = (semantic_pred == stuff_id).astype(np.uint8) * 255
    if not np.any(mask):
        return []

    # 尺寸恢复到原图尺寸
    mask_resized = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    for contour in contours:
        if len(contour) < 3: continue
        contour = contour.reshape(-1, 2).astype(np.float32)
        contour[:, 0] /= target_shape[1]
        contour[:, 1] /= target_shape[0]
        segments.append(contour)
    return segments


def draw_semantic_mask(im0, semantic_logits, stuff_ids, alpha):
    """绘制语义分割掩码，增强区分度"""
    if semantic_logits is None:
        return im0

    H, W = im0.shape[:2]
    semantic_pred = torch.argmax(semantic_logits, dim=0).cpu().numpy().astype(np.uint8)
    semantic_resized = cv2.resize(semantic_pred, (W, H), interpolation=cv2.INTER_NEAREST)

    # 定义高对比度颜色字典 (B, G, R)
    # 你可以根据需要修改这里的颜色
    STUFF_COLORS = {
        7: (0, 255, 0),  # 鲜绿色 - 作物架
        8: (255, 0, 0),  # 鲜蓝色 - 背景
    }

    for sid in stuff_ids:
        mask = (semantic_resized == sid)
        if np.any(mask):
            # 1. 颜色层填充
            color = STUFF_COLORS.get(sid, (255, 255, 255))  # 找不到就用白色
            color_layer = np.zeros_like(im0, dtype=np.uint8)
            color_layer[mask] = color
            im0 = cv2.addWeighted(im0, 1.0, color_layer, alpha, 0)

            # 2. 绘制边缘轮廓 (让边界更清晰)
            mask_uint8 = (mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im0, contours, -1, (255, 255, 255), 1)  # 白色细边框

            # 3. 在区域中心写上 ID 标签
            # 找到最大的轮廓并在其中心标 ID
            if contours:
                max_cnt = max(contours, key=cv2.contourArea)
                M = cv2.moments(max_cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"ID:{sid}"
                    # 画个小黑底色让文字更清晰
                    cv2.putText(im0, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 4, cv2.LINE_AA)  # 黑色描边
                    cv2.putText(im0, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 1, cv2.LINE_AA)  # 白色字体

    return im0


def draw_instance_masks(im0, det, mask_proto, im_shape, names, alpha=0.5):
    """绘制实例分割掩码和边界框"""
    if not len(det):
        return im0
    # 拷贝一份以防影响原始 det 坐标转换
    det_copy = det.clone()
    det_copy[:, :4] = scale_boxes(im_shape, det_copy[:, :4], im0.shape).round()
    masks = process_mask(mask_proto, det_copy[:, 6:], det_copy[:, :4], im0.shape[:2], upsample=True)
    masks_np = masks.permute(1, 2, 0).cpu().numpy()

    for j, (*xyxy, conf, cls) in enumerate(reversed(det_copy[:, :6])):
        c = int(cls)
        mask_bool = masks_np[:, :, j] > 0.5
        color_layer = np.zeros_like(im0, dtype=np.uint8)
        color_layer[mask_bool] = colors(c, True)
        im0 = cv2.addWeighted(im0, 1.0, color_layer, alpha, 0)
        # 画框
        cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), colors(c, True), 2)
    return im0


@smart_inference_mode()
def run_pseudo_labeling(**kwargs):
    source, weights, imgsz = kwargs['source'], kwargs['weights'], kwargs['imgsz']

    # 目录准备
    save_root = Path(kwargs['save_dir'])
    label_dir = save_root / 'labels'
    visual_dir = save_root / 'visuals'
    label_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    # 模型加载
    device = select_device(kwargs['device'])
    model = DetectMultiBackend(weights, device=device, fp16=kwargs['half'])
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # 数据加载
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    model.warmup(imgsz=(1, 3, *imgsz))

    for path, im, im0s, vid_cap, s in dataset:
        img_name = Path(path).stem
        txt_path = label_dir / f"{img_name}.txt"
        if txt_path.exists(): os.remove(txt_path)

        # 预处理
        im = torch.from_numpy(im).to(device).float() / 255
        if im.ndim == 3: im = im.unsqueeze(0)
        if kwargs['half']: im = im.half()

        # 推理
        pred, panoptic_outs = model(im, augment=False)
        mask_proto, semantic_logits = panoptic_outs[2], panoptic_outs[3]

        # NMS
        pred = non_max_suppression(pred, kwargs['conf_thres'], kwargs['iou_thres'], nm=32)

        for i, det in enumerate(pred):
            im0 = im0s.copy()
            h_orig, w_orig = im0.shape[:2]

            # 1. 处理 Thing 类别并保存标签
            # if len(det):
            #     masks = process_mask(mask_proto[i], det[:, 6:], det[:, :4], im0.shape[:2], upsample=True)
            #     segments = masks2segments(masks)
            #     for j, seg in enumerate(segments):
            #         save_yolo_poly_labels(txt_path, [seg], int(det[j, 5]))

            # 2. 处理 Stuff 类别 (7: 作物架, 8: 背景) 并保存标签
            for stuff_id in [7, 8]:
                stuff_segs = get_stuff_segments(semantic_logits[i], stuff_id, (h_orig, w_orig))
                if stuff_segs:
                    save_yolo_poly_labels(txt_path, stuff_segs, stuff_id)

            # 3. 绘制结果图
            # 先画语义分割 (7和8)
            im0 = draw_semantic_mask(im0, semantic_logits[i], [7, 8], alpha=0.4)
            # 再画实例分割
            im0 = draw_instance_masks(im0, det, mask_proto[i], im.shape[2:], names, alpha=0.5)

            # 4. 保存结果图
            cv2.imwrite(str(visual_dir / f"{img_name}.jpg"), im0)

        print(f"Processed: {img_name} (Label & Visual saved)")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="/home/huanghanyang/Project/yolov9/runs/train-pan/strawberry_rack_v2/weights/yolov9-pan-strawberry7cls_rackcls1_v2.pt")
    parser.add_argument('--source', type=str, default="/home/huanghanyang/Datasets/strawberry/strawberry_datasets_v3/project-120/images")
    parser.add_argument('--save-dir', type=str, default='/home/huanghanyang/Datasets/strawberry/strawberry_datasets_v3/project-120/yolov9-pan-result')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='')
    parser.add_argument('--half', action='store_true')
    opt = parser.parse_args()
    if len(opt.imgsz) == 1: opt.imgsz *= 2
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run_pseudo_labeling(**vars(opt))
