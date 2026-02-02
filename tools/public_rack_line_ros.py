# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : predict_rack_online.py
# @Description : 作物架实时推理 (支持 ROS Topic 输入、裁剪及上下边界过滤)

import argparse
import os
os.environ['ROS_MASTER_URI'] = 'http://192.168.3.101:11311'
import sys
from pathlib import Path
import time
import threading

import numpy as np
import torch
import cv2

# --- ROS Imports ---
import rospy
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Header

# 尝试导入自定义消息
try:
    from dangkang_picking_msgs_1.msg import moveview_return
    ROS_AVAILABLE = True
except ImportError:
    print("[WARNING] ROS custom messages not found. ROS publishing will be disabled.")
    ROS_AVAILABLE = False
# -------------------

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
sys.path.append(str(ROOT)) if str(ROOT) not in sys.path else None
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import LOGGER, check_file, check_img_size, check_imshow, non_max_suppression, print_args, scale_boxes
from utils.plots import colors
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox  # 导入letterbox用于缩放
# 假设 utils/mask_center_cal.py 在相应路径下
from utils.mask_center_cal import MaskCenterEstimator


def numpy_to_ros_image(img_np, encoding="bgr8"):
    """手动将 OpenCV (Numpy) 图片转换为 ROS Image 消息 (发布用)"""
    msg = Image()
    msg.height = img_np.shape[0]
    msg.width = img_np.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = len(img_np[0]) * img_np.itemsize * img_np.shape[2]
    msg.data = img_np.tobytes()
    return msg


def ros_image_to_numpy(msg):
    """手动将 ROS Image 消息转换为 Numpy (接收用) - 替代 cv_bridge"""
    dtype_map = {
        "8UC1": np.uint8, "8UC3": np.uint8, "bgr8": np.uint8, "rgb8": np.uint8,
        "mono8": np.uint8
    }
    np_dtype = dtype_map.get(msg.encoding, np.uint8)
    channels = 3 if "8" in msg.encoding and "C3" in msg.encoding or "bgr" in msg.encoding or "rgb" in msg.encoding else 1

    # 从 buffer 读取
    img = np.frombuffer(msg.data, dtype=np_dtype)
    img = img.reshape((msg.height, msg.width, -1))  # reshape to (H, W, C)

    if msg.encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if channels == 1 and len(img.shape) == 3:
        img = img.squeeze(-1)

    return img


def draw_semantic_mask(im0, semantic_logits, stuff_class_id, alpha, im_shape=None):
    if semantic_logits is None: return im0
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


def apply_mask_roi(mask_binary, crop_x=0.0, crop_y_top=0.0, crop_y_bot=0.0):
    """
    对掩码进行 ROI 过滤，移除边缘噪声
    :param mask_binary: 二值掩码
    :param crop_x: 左右边缘裁剪比例 (0.0 - 0.5)
    :param crop_y_top: 顶部裁剪比例 (0.0 - 1.0)
    :param crop_y_bot: 底部裁剪比例 (0.0 - 1.0)
    """
    H, W = mask_binary.shape

    # 左右裁剪
    if crop_x > 0:
        crop_px_x = int(W * crop_x)
        mask_binary[:, :crop_px_x] = 0
        mask_binary[:, -(crop_px_x):] = 0

    # 顶部裁剪
    if crop_y_top > 0:
        crop_px_top = int(H * crop_y_top)
        mask_binary[:crop_px_top, :] = 0

    # 底部裁剪
    if crop_y_bot > 0:
        crop_px_bot = int(H * crop_y_bot)
        mask_binary[-crop_px_bot:, :] = 0

    return mask_binary


# === 自定义 ROS 数据加载器 ===
class LoadRosTopic:
    def __init__(self, topic, img_size=640, stride=32, auto=True):
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.topic = topic
        self.latest_msg = None
        self.lock = threading.Lock()
        self.new_data = False

        LOGGER.info(f"[INFO] Subscribing to ROS topic: {topic}")
        # 订阅话题
        rospy.Subscriber(topic, Image, self.callback, queue_size=1)

    def callback(self, msg):
        with self.lock:
            self.latest_msg = msg
            self.new_data = True

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if rospy.is_shutdown():
            raise StopIteration

        # 等待新数据
        while not self.new_data:
            if rospy.is_shutdown(): raise StopIteration
            time.sleep(0.01)

        with self.lock:
            msg = self.latest_msg
            self.new_data = False  # 重置标志

        # 1. Msg -> Numpy (BGR)
        im0 = ros_image_to_numpy(msg)

        # 2. Crop Left Half (裁剪左半部分)
        h, w = im0.shape[:2]
        im0 = im0[:, :w // 2]

        # 3. Resize & Pad (Letterbox)
        im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # 4. HWC to CHW, BGR to RGB
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        # 返回格式兼容 LoadImages: (path, im, im0, cap, s)
        return self.topic, im, im0, None, ''

    def __len__(self):
        return 0  # Infinite stream


# ===========================

@smart_inference_mode()
def run(**kwargs):
    # --- ROS Init ---
    ros_pub = None
    if ROS_AVAILABLE or kwargs['input_mode'] == 1:
        try:
            if not rospy.core.is_initialized():
                rospy.init_node('predict_rack_online', anonymous=True)

            if ROS_AVAILABLE:
                ros_pub = rospy.Publisher('arm_0/picking/moveview_return', moveview_return, queue_size=1)  # right
                # ros_pub = rospy.Publisher('arm_1/picking/moveview_return', moveview_return, queue_size=1)  # left
                LOGGER.info("[INFO] ROS Publisher initialized: 'arm_0/picking/moveview_return'")
        except Exception as e:
            LOGGER.error(f"[ERROR] Failed to init ROS node: {e}")
    # ----------------

    # 参数解析
    input_mode = kwargs.get('input_mode', 0)
    source = str(kwargs['source'])
    ros_topic = kwargs.get('ros_topic', 'perception_binocular_image_rect_pair')

    # 获取裁剪参数
    crop_top = kwargs.get('crop_top', 0.0)
    crop_bottom = kwargs.get('crop_bottom', 0.0)

    # 模型加载
    device = select_device(kwargs['device'])
    model = DetectMultiBackend(kwargs['weights'], device=device, dnn=kwargs['dnn'],
                               data=kwargs['data'], fp16=kwargs['half'])
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(kwargs['imgsz'], s=stride)

    # 数据集加载逻辑选择
    bs = 1
    dataset = None
    webcam = False

    if input_mode == 1:
        # === Mode 1: ROS Input ===
        LOGGER.info(f"[MODE] Using ROS Topic Input: {ros_topic}")
        dataset = LoadRosTopic(ros_topic, img_size=imgsz, stride=stride, auto=pt)
        webcam = False
    else:
        # === Mode 0: Default (Camera/File) ===
        LOGGER.info(f"[MODE] Using Source Input: {source}")
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')

        if is_url and is_file:
            source = check_file(source)

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

    LOGGER.info(f"Starting inference... Ref: {ref_pos}, CropTop: {crop_top}, CropBot: {crop_bottom}")

    for path, im, im0s, vid_cap, s in dataset:
        if rospy.is_shutdown():
            break

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
            if webcam:
                im0 = im0s[i].copy()
            else:
                im0 = im0s.copy()

            H, W = im0.shape[:2]
            stuff_id = kwargs.get('STUFF_ID_TO_PLOT', 7)

            # 1. 分割底色 (原始结果，不裁剪，用于参考)
            im0 = draw_semantic_mask(im0, semantic_logits[i], stuff_id, alpha=0.2, im_shape=im.shape[2:])

            # 2. 掩码计算与过滤
            sem_pred = torch.argmax(semantic_logits[i], dim=0)
            sem_resized = cv2.resize(sem_pred.cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            mask_binary_np = (sem_resized == stuff_id).astype(np.float32)

            # --- 关键修改：应用上下左右边界过滤 ---
            # crop_x=0.3 保持原有逻辑，新增 crop_y_top 和 crop_y_bot
            mask_binary_np = apply_mask_roi(mask_binary_np,
                                            crop_x=0.3,
                                            crop_y_top=crop_top,
                                            crop_y_bot=crop_bottom)
            # -----------------------------------

            # 3. 红色高亮 (仅高亮过滤后的有效区域)
            red_overlay = np.zeros_like(im0, dtype=np.uint8)
            red_overlay[mask_binary_np == 1.0] = (0, 0, 255)
            im0 = cv2.addWeighted(im0, 1.0, red_overlay, 0.5, 0)

            # 4. 业务逻辑
            mask_area = np.sum(mask_binary_np)
            status_text = "Wait"
            offset_text = "N/A"
            offset_color = (128, 128, 128)
            delta_pixel_val = 0.0
            delta_height_val = 0.0
            stop_flag_val = True

            if mask_area >= 2000:
                mask_tensor = torch.from_numpy(mask_binary_np).to(model.device)
                y_norm, dy_norm = center_estimator.compute(mask_tensor, ref_pos=ref_pos, return_cpu=True)

                current_y_norm = float(y_norm[0])
                y_center_px = int(current_y_norm * (H - 1))
                ref_px = int(ref_pos * (H - 1))

                delta_pixel_val = float(ref_px - y_center_px)

                cv2.line(im0, (0, y_center_px), (W - 1, y_center_px), (0, 255, 0), 2)
                status_text = "Tracking"
                offset_text = f"{delta_pixel_val:.1f} px"
                offset_color = (0, 255, 255)

                print(f"\r[INFO] FPS: {fps:.1f} | Area: {int(mask_area)} | Delta: {delta_pixel_val:+.1f} px", end="")
            else:
                status_text = "Lost (Area < 2000)"
                print(f"\r[INFO] FPS: {fps:.1f} | Area: {int(mask_area)} | Target Lost", end="")

            # 绘制参考线
            ref_px_draw = int(ref_pos * (H - 1))
            cv2.line(im0, (0, ref_px_draw), (W - 1, ref_px_draw), (0, 0, 255), 1)

            # 绘制上下过滤边界线 (可选可视化)
            if crop_top > 0:
                y_top_limit = int(H * crop_top)
                cv2.line(im0, (0, y_top_limit), (W - 1, y_top_limit), (0, 0, 0), 1)
            if crop_bottom > 0:
                y_bot_limit = int(H * (1 - crop_bottom))
                cv2.line(im0, (0, y_bot_limit), (W - 1, y_bot_limit), (0, 0, 0), 1)

            # OSD
            cv2.rectangle(im0, (5, 5), (420, 75), (0, 0, 0), -1)
            cv2.putText(im0, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(im0, f"Delta Pixel: {offset_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, offset_color, 2)

            fps_text = f"FPS: {fps:.1f}"
            fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.rectangle(im0, (W - fps_size[0] - 20, 5), (W - 5, 45), (0, 0, 0), -1)
            cv2.putText(im0, fps_text, (W - fps_size[0] - 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if abs(delta_pixel_val) < 5:
                delta_pixel_val = 0
            # ROS Publish
            if ros_pub is not None and ROS_AVAILABLE:
                try:
                    msg = moveview_return()
                    msg.header = Header()
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = "camera_link"

                    # msg.image_out = numpy_to_ros_image(im0, encoding="bgr8")
                    msg.stop_flag = stop_flag_val
                    msg.delta_camera_pixel_height = -delta_pixel_val
                    msg.delta_camera_height = delta_height_val
                    msg.strawberries = []
                    msg.cloud_out = PointCloud2()

                    ros_pub.publish(msg)
                except Exception as e:
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
    # === 新增参数 ===
    parser.add_argument('--input-mode', type=int, default=1,
                        help='Input mode: 0 for source(cam/file), 1 for ROS topic')
    parser.add_argument('--ros-topic', type=str, default='arm_0/perception_binocular_image_raw_pair',  # right
    # parser.add_argument('--ros-topic', type=str, default='arm_1/perception_binocular_image_raw_pair',  # left
                        help='ROS topic name for input_mode=1. Can be absolute or relative.')
    parser.add_argument('--crop-top', type=float, default=0.2,
                        help='Top margin crop ratio (0.0-1.0), mask in this area will be ignored.')
    parser.add_argument('--crop-bottom', type=float, default=0.2,
                        help='Bottom margin crop ratio (0.0-1.0), mask in this area will be ignored.')
    # =================

    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/zeno/Project/yolov9-pan/yolov9-pan-strawberry7cls_rackcls1_v2.pt')
                        # default='/home/zeno/Project/yolov9-pan/yolov9-pan-strawberry7cls_rackcls1_v3_0.pt')  # v3
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='../data/rack-v2.yaml')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=100, help='maximum detections per image')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--ref-pos', type=float, default=0.5, help='Reference line vertical position (0.0-1.0)')
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