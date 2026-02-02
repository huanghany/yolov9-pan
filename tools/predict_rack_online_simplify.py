# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : predict_rack_online.py
# @Description : 作物架实时推理 (FPS + 红色掩码 + 面积过滤)

import argparse
import os
import sys
from pathlib import Path
import time

import numpy as np
import torch
import cv2

# 假设 utils/mask_center_cal.py 在相应路径下
from utils.mask_center_cal import MaskCenterEstimator

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
sys.path.append(str(ROOT)) if str(ROOT) not in sys.path else None
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import LOGGER, check_file, check_img_size, check_imshow, non_max_suppression, print_args
from utils.plots import colors
from utils.torch_utils import select_device, smart_inference_mode


def draw_semantic_mask(im0, semantic_logits, stuff_class_id, alpha, im_shape=None):
    """绘制原始语义分割掩码 (作为底色参考)"""
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
        # 使用默认颜色绘制原始预测
        color_layer[stuff_mask] = colors(stuff_class_id, True)
        im0 = cv2.addWeighted(im0, 1.0, color_layer, alpha, 0)
    return im0


def remove_edge_noise(mask_binary, crop_ratio=0.02):
    """移除掩码左右边缘的噪声"""
    H, W = mask_binary.shape
    crop_px = int(W * crop_ratio)
    mask_binary[:, :crop_px] = 0
    mask_binary[:, -(crop_px):] = 0
    return mask_binary


@smart_inference_mode()
def run(**kwargs):
    source = str(kwargs['source'])
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')

    if is_url and is_file:
        source = check_file(source)

    device = select_device(kwargs['device'])
    model = DetectMultiBackend(kwargs['weights'], device=device, dnn=kwargs['dnn'],
                               data=kwargs['data'], fp16=kwargs['half'])
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(kwargs['imgsz'], s=stride)

    bs = 1
    if webcam:
        check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=kwargs['vid_stride'])
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=kwargs['vid_stride'])

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))

    # 业务工具类初始化
    center_estimator = MaskCenterEstimator(kernel_size=5, use_bbox=False, use_ema=True)
    ref_pos = kwargs.get('ref_pos', 0.5)
    prev_time = 0

    LOGGER.info(f"Starting inference... Ref Line: {ref_pos}. Press 'q' to exit.")

    for path, im, im0s, vid_cap, s in dataset:
        # FPS 计算
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # 预处理
        im = torch.from_numpy(im).to(model.device).float()
        if model.fp16:
            im = im.half()
        im /= 255
        if im.ndim == 3:
            im = im.unsqueeze(0)

        # 推理
        pred, panoptic_outs = model(im, augment=kwargs['augment'], visualize=False)
        mask_proto, semantic_logits = panoptic_outs[2], panoptic_outs[3]

        # NMS
        pred = non_max_suppression(pred, kwargs['conf_thres'], kwargs['iou_thres'],
                                   kwargs['classes'], kwargs['agnostic_nms'],
                                   kwargs['max_det'], nm=32)

        for i, det in enumerate(pred):
            im0 = im0s.copy() if not webcam else im0s[i].copy()
            H, W = im0.shape[:2]
            stuff_id = kwargs.get('STUFF_ID_TO_PLOT', 7)

            # 1. 绘制基础语义分割 (可选，这里保留作为底色，alpha设低一点)
            im0 = draw_semantic_mask(im0, semantic_logits[i], stuff_id,
                                     alpha=0.3, im_shape=im.shape[2:])

            # 2. 准备掩码数据
            sem_pred = torch.argmax(semantic_logits[i], dim=0)
            sem_resized = cv2.resize(sem_pred.cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

            # 生成二值掩码 (numpy)
            mask_binary_np = (sem_resized == stuff_id).astype(np.float32)

            # 边缘去噪
            mask_binary_np = remove_edge_noise(mask_binary_np, crop_ratio=0.2)

            # === 功能：用红色绘制去噪后的掩码 ===
            # 创建红色图层 (BGR: 0, 0, 255)
            red_overlay = np.zeros_like(im0, dtype=np.uint8)
            red_overlay[mask_binary_np == 1.0] = (0, 0, 255)
            # 叠加红色 (alpha=0.5)
            im0 = cv2.addWeighted(im0, 1.0, red_overlay, 0.5, 0)

            # 计算有效面积
            mask_area = np.sum(mask_binary_np)

            # 初始化显示文本
            offset_text = "N/A"
            offset_color = (128, 128, 128)  # 灰色

            # === 功能：只有面积 >= 10000 才计算中心线 ===
            if mask_area >= 10000:
                # 转为 Tensor
                mask_tensor = torch.from_numpy(mask_binary_np).to(model.device)

                # 计算中心
                y_norm, dy_norm = center_estimator.compute(mask_tensor, ref_pos=ref_pos, return_cpu=True)

                current_y_norm = float(y_norm[0])
                offset_norm = current_y_norm - ref_pos
                offset_px = int(offset_norm * (H - 1))
                y_center_px = int(current_y_norm * (H - 1))

                # 绘制绿色中心线
                cv2.line(im0, (0, y_center_px), (W - 1, y_center_px), (0, 255, 0), 2)

                # 更新文本信息
                status_text = "Tracking"
                offset_text = f"{offset_norm:.4f} ({offset_px} px)"
                offset_color = (0, 255, 255)  # 黄色

                # 终端打印
                print(f"\r[INFO] FPS: {fps:.1f} | Area: {int(mask_area)} | Offset: {offset_norm:+.4f}", end="")
            else:
                status_text = "Lost (Area < 1000)"
                # 终端打印
                print(f"\r[INFO] FPS: {fps:.1f} | Area: {int(mask_area)} | Target Lost", end="")

            # 绘制参考线 (始终绘制，方便参考)
            ref_px = int(ref_pos * (H - 1))
            cv2.line(im0, (0, ref_px), (W - 1, ref_px), (0, 0, 255), 1)

            # === UI 显示 ===
            # 左上角：偏差信息
            cv2.rectangle(im0, (5, 5), (420, 75), (0, 0, 0), -1)  # 背景框
            cv2.putText(im0, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(im0, f"Offset: {offset_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, offset_color, 2)

            # === 功能：右上角 FPS ===
            fps_text = f"FPS: {fps:.1f}"
            fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            # 绘制背景框让FPS更清晰
            cv2.rectangle(im0, (W - fps_size[0] - 20, 5), (W - 5, 45), (0, 0, 0), -1)
            cv2.putText(im0, fps_text, (W - fps_size[0] - 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 显示结果
            cv2.imshow("Real-time Inference", im0)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print()
                LOGGER.info("[INFO] Exit by user pressing 'q'")
                cv2.destroyAllWindows()
                return

    print()
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    # 核心路径与模型参数
    parser.add_argument('--weights', nargs='+', type=str,
                        # default='../yolov9-pan-strawberry7cls_rackcls1_v2.pt')
                        default='../yolov9-pan-strawberry7cls_rackcls1_v3_0.pt')
    parser.add_argument('--source', type=str, default='2', help='file/dir/URL/glob/screen/0(webcam)')  # left: 2 right: 6
    parser.add_argument('--data', type=str, default='../data/rack-v2.yaml')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=100, help='maximum detections per image')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 业务参数
    parser.add_argument('--ref-pos', type=float, default=0.5, help='Reference line vertical position (0.0-1.0)')

    # 可视化参数
    parser.add_argument('--classes', nargs='+', type=int, default=None,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', type=bool, default=True, help='class-agnostic NMS')
    parser.add_argument('--augment', type=bool, default=False, help='augmented inference')
    parser.add_argument('--line-thickness', type=int, default=3, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', type=bool, default=False, help='hide labels')
    parser.add_argument('--hide-conf', type=bool, default=False, help='hide confidences')
    parser.add_argument('--half', type=bool, default=False, help='use FP16 half-precision inference')
    parser.add_argument('--dnn', type=bool, default=False, help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')

    opt = parser.parse_args()
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    main(parse_opt())