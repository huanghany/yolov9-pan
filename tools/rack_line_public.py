# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : predict_rack_online.py
# @Description : 作物架实时推理 (ROS 发布版)

import argparse
import os
import sys
from pathlib import Path
import time

import numpy as np
# import torch
import cv2

# --- ROS Imports ---

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header

# 尝试导入自定义消息，防止在非ROS环境下直接报错崩溃
try:
    from dangkang_picking_msgs.msg import moveview_return

    # 假设 strawberry_info 类型也在某个包里，如果需要填充列表，需要导入
    # from dangkang_picking_msgs.msg import strawberry_info
    ROS_AVAILABLE = True
except ImportError:
    print("[WARNING] ROS custom messages not found. ROS publishing will be disabled.")
    ROS_AVAILABLE = False
# -------------------

# 假设 utils/mask_center_cal.py 在相应路径下
from utils.mask_center_cal import MaskCenterEstimator

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
sys.path.append(str(ROOT)) if str(ROOT) not in sys.path else None
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import LOGGER, check_file, check_img_size, check_imshow, non_max_suppression, print_args, scale_boxes
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
    # --- ROS Init ---
    ros_pub = None
    cv_bridge = None
    if ROS_AVAILABLE:
        try:
            rospy.init_node('predict_rack_online', anonymous=True)
            # 话题名: /picking/moveview_return
            ros_pub = rospy.Publisher('picking/moveview_return', moveview_return, queue_size=1)
            cv_bridge = CvBridge()
            LOGGER.info("[INFO] ROS Node initialized. Publishing to 'picking/moveview_return'")
        except Exception as e:
            LOGGER.error(f"[ERROR] Failed to init ROS node: {e}")
    # ----------------

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

    center_estimator = MaskCenterEstimator(kernel_size=5, use_bbox=False, use_ema=True)
    ref_pos = kwargs.get('ref_pos', 0.5)
    prev_time = 0

    LOGGER.info(f"Starting inference... Ref Line: {ref_pos}. Press 'q' to exit.")

    for path, im, im0s, vid_cap, s in dataset:
        # ROS Shutdown Check
        if rospy.is_shutdown():
            break

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        im = torch.from_numpy(im).to(model.device).float()
        if model.fp16:
            im = im.half()
        im /= 255
        if im.ndim == 3:
            im = im.unsqueeze(0)

        pred, panoptic_outs = model(im, augment=kwargs['augment'], visualize=False)
        mask_proto, semantic_logits = panoptic_outs[2], panoptic_outs[3]

        pred = non_max_suppression(pred, kwargs['conf_thres'], kwargs['iou_thres'],
                                   kwargs['classes'], kwargs['agnostic_nms'],
                                   kwargs['max_det'], nm=32)

        for i, det in enumerate(pred):
            im0 = im0s.copy() if not webcam else im0s[i].copy()
            H, W = im0.shape[:2]
            stuff_id = kwargs.get('STUFF_ID_TO_PLOT', 7)

            # 1. 基础语义分割底色
            im0 = draw_semantic_mask(im0, semantic_logits[i], stuff_id,
                                     alpha=0.3, im_shape=im.shape[2:])

            # 2. 准备掩码
            sem_pred = torch.argmax(semantic_logits[i], dim=0)
            sem_resized = cv2.resize(sem_pred.cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            mask_binary_np = (sem_resized == stuff_id).astype(np.float32)

            # 边缘去噪
            mask_binary_np = remove_edge_noise(mask_binary_np, crop_ratio=0.1)

            # 绘制红色去噪掩码
            red_overlay = np.zeros_like(im0, dtype=np.uint8)
            red_overlay[mask_binary_np == 1.0] = (0, 0, 255)
            im0 = cv2.addWeighted(im0, 1.0, red_overlay, 0.5, 0)

            # 计算
            mask_area = np.sum(mask_binary_np)

            # 变量初始化
            status_text = "Wait"
            offset_text = "N/A"
            offset_color = (128, 128, 128)

            # ROS 消息数据变量
            delta_pixel_val = 0.0
            delta_height_val = 0.0  # 暂时为0
            stop_flag_val = True  # 默认停止，检测到才走

            if mask_area >= 10000:
                mask_tensor = torch.from_numpy(mask_binary_np).to(model.device)
                y_norm, dy_norm = center_estimator.compute(mask_tensor, ref_pos=ref_pos, return_cpu=True)

                current_y_norm = float(y_norm[0])
                y_center_px = int(current_y_norm * (H - 1))
                ref_px = int(ref_pos * (H - 1))

                # === 偏差计算逻辑 ===
                # 规范：目标高于相机中心为正(+)。
                # 图像坐标系：Y越小越靠上。
                # 公式：Reference_Y (相机中心) - Target_Y (目标中心)
                delta_pixel_val = float(ref_px - y_center_px)

                # 停止标志：检测到有效道路/架子，不停止
                # stop_flag_val = False

                # 绘制绿色中心线
                cv2.line(im0, (0, y_center_px), (W - 1, y_center_px), (0, 255, 0), 2)

                status_text = "Tracking"
                offset_text = f"{delta_pixel_val:.1f} px"
                offset_color = (0, 255, 255)  # 黄色

                print(f"\r[INFO] FPS: {fps:.1f} | Area: {int(mask_area)} | Delta: {delta_pixel_val:+.1f} px", end="")
            else:
                status_text = "Lost (Area < 1000)"
                # 丢失目标，delta 保持 0.0，stop_flag 为 True
                print(f"\r[INFO] FPS: {fps:.1f} | Area: {int(mask_area)} | Target Lost", end="")

            # 绘制参考线
            ref_px_draw = int(ref_pos * (H - 1))
            cv2.line(im0, (0, ref_px_draw), (W - 1, ref_px_draw), (0, 0, 255), 1)

            # === OSD UI ===
            cv2.rectangle(im0, (5, 5), (420, 75), (0, 0, 0), -1)
            cv2.putText(im0, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(im0, f"Delta Pixel: {offset_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, offset_color, 2)

            fps_text = f"FPS: {fps:.1f}"
            fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.rectangle(im0, (W - fps_size[0] - 20, 5), (W - 5, 45), (0, 0, 0), -1)
            cv2.putText(im0, fps_text, (W - fps_size[0] - 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # === ROS 发布 ===
            if ros_pub is not None and ROS_AVAILABLE:
                try:
                    msg = moveview_return()
                    msg.header = Header()
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = "camera_link"  # 根据实际情况修改 TF frame

                    # 填充图像
                    msg.image_out = cv_bridge.cv2_to_imgmsg(im0, encoding="bgr8")

                    # 填充控制数据
                    msg.stop_flag = stop_flag_val
                    msg.delta_camera_pixel_height = delta_pixel_val
                    msg.delta_camera_height = delta_height_val  # 目前设为 0.0，需实际相机内参转换

                    # 填充空数据 (strawberries, cloud_out)
                    msg.strawberries = []
                    msg.cloud_out = PointCloud2()

                    ros_pub.publish(msg)
                except Exception as e:
                    # 降低错误打印频率，防止刷屏
                    if i % 30 == 0:
                        LOGGER.warning(f"Publish failed: {e}")

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
    # 核心参数
    parser.add_argument('--weights', nargs='+', type=str,
                        default='../yolov9-pan-strawberry7cls_rackcls1_v2.pt')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='../data/rack-v2.yaml')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=100, help='maximum detections per image')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 业务参数
    parser.add_argument('--ref-pos', type=float, default=0.5, help='Reference line vertical position (0.0-1.0)')

    # 其他参数
    parser.add_argument('--classes', nargs='+', type=int, default=None, help='filter by class')
    parser.add_argument('--agnostic-nms', type=bool, default=True, help='class-agnostic NMS')
    parser.add_argument('--augment', type=bool, default=False, help='augmented inference')
    parser.add_argument('--line-thickness', type=int, default=3, help='bounding box thickness')
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