# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : rack_line_topic_bridge.py
# @Create      : 2025/12/8-19:51
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2025, ZenoAI Robot Inc. All Rights Reserved.
# @Description : ROS 话题桥接推理（订阅图像并发布 moveview_return/image_out）

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

try:
    from dangkang_picking_msgs.msg import moveview_return

    ROS_MOVEVIEW_MSG_AVAILABLE = True
except ImportError:
    ROS_MOVEVIEW_MSG_AVAILABLE = False
    moveview_return = None

from tools.rack_line_utils import (
    Annotator,
    DetectMultiBackend,
    LOGGER,
    MaskCenterEstimator,
    LoadRosTopic,
    apply_mask_roi,
    check_img_size,
    check_roscore,
    colors,
    create_video_writer,
    draw_semantic_mask,
    non_max_suppression,
    numpy_to_ros_image,
    print_args,
    scale_boxes,
    select_device,
    smart_inference_mode,
)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


@smart_inference_mode()
def run(
    ros_topic: str,
    crop_top: float,
    crop_bottom: float,
    weights,
    imgsz,
    conf: float,
    iou: float,
    ref_pos: float,
    classes,
    device: str = "",
    crop_x: float = 0.3,
    save_video: bool = False,
    save_path: str = "rack_result.mp4",
    show_window: bool = False,
    ros_publish: bool = True,
    ros_publish_topic: str = "arm_0/picking/moveview_return",
    ros_publish_image: bool = True,
    show_det: bool = True,
):
    show_det = bool(show_det)

    if not check_roscore():
        LOGGER.error("[ERROR] roscore 未启动，请先运行 'roscore' 后再启动本脚本。")
        return
    if not rospy.core.is_initialized():
        rospy.init_node("rack_line_topic_bridge", anonymous=True)

    ros_pub = None
    if ros_publish or ros_publish_image:
        if not ROS_MOVEVIEW_MSG_AVAILABLE:
            LOGGER.warning(
                "[WARN] 未找到 dangkang_picking_msgs.msg.moveview_return，跳过结果发布。"
            )
        else:
            try:
                ros_pub = rospy.Publisher(ros_publish_topic, moveview_return, queue_size=1)
                LOGGER.info(f"[INFO] ROS Publisher initialized: '{ros_publish_topic}'")
            except Exception as e:
                LOGGER.error(f"[ERROR] 创建 ROS Publisher 失败: {e}")

    device_torch = select_device(device)
    model = DetectMultiBackend(weights, device=device_torch, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    LOGGER.info(f"[MODE] ROS 订阅: {ros_topic}")
    dataset = LoadRosTopic(ros_topic, img_size=imgsz, stride=stride, auto=pt)

    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))

    center_estimator = MaskCenterEstimator(kernel_size=5, use_bbox=False, use_ema=True)
    prev_time = 0.0
    video_writer = None

    LOGGER.info(
        f"Start | ref-pos={ref_pos}, crop-top={crop_top}, crop-bottom={crop_bottom}, crop-x={crop_x}, "
        f"save_video={save_video}, show_window={show_window}, ros_publish={ros_publish}"
    )

    stuff_id = int(classes[0]) if classes is not None and len(classes) > 0 else 7

    for path, im, im0s, vid_cap, s in dataset:
        if rospy.is_shutdown():
            break

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time != 0 else 0.0
        prev_time = curr_time

        im_tensor = torch.from_numpy(im).to(model.device).float()
        im_tensor /= 255.0
        if im_tensor.ndim == 3:
            im_tensor = im_tensor.unsqueeze(0)

        pred, panoptic_outs = model(im_tensor, augment=False, visualize=False)
        _, semantic_logits = panoptic_outs[2], panoptic_outs[3]

        pred = non_max_suppression(
            pred,
            conf_thres=conf,
            iou_thres=iou,
            classes=classes,
            agnostic=False,
            max_det=100,
            nm=32,
        )

        for i, det in enumerate(pred):
            im0 = im0s.copy()

            h, w = im0.shape[:2]

            im0 = draw_semantic_mask(
                im0,
                semantic_logits[i],
                stuff_class_id=stuff_id,
                alpha=0.2,
                im_shape=im_tensor.shape[2:],
            )

            sem_pred = torch.argmax(semantic_logits[i], dim=0)
            sem_resized = cv2.resize(
                sem_pred.cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
            mask_binary_np = (sem_resized == stuff_id).astype(np.float32)
            mask_binary_np = apply_mask_roi(
                mask_binary_np,
                crop_x=crop_x,
                crop_y_top=crop_top,
                crop_y_bot=crop_bottom,
            )

            red_overlay = np.zeros_like(im0, dtype=np.uint8)
            red_overlay[mask_binary_np == 1.0] = (0, 0, 255)
            im0 = cv2.addWeighted(im0, 1.0, red_overlay, 0.5, 0)

            mask_area = float(np.sum(mask_binary_np))
            status_text = "Wait"
            offset_text = "N/A"
            offset_color = (128, 128, 128)

            delta_pixel_val = 0.0
            delta_height_val = 0.0
            stop_flag_val = True

            if mask_area >= 2000:
                mask_tensor = torch.from_numpy(mask_binary_np).to(model.device)
                y_norm, _ = center_estimator.compute(mask_tensor, ref_pos=ref_pos, return_cpu=True)
                current_y_norm = float(y_norm[0])
                y_center_px = int(current_y_norm * (h - 1))
                ref_px = int(ref_pos * (h - 1))
                delta_pixel_val = float(ref_px - y_center_px)
                cv2.line(im0, (0, y_center_px), (w - 1, y_center_px), (0, 255, 0), 2)
                status_text = "Tracking"
                offset_text = f"{delta_pixel_val:.1f} px"
                offset_color = (0, 255, 255)
                print(
                    f"\r[INFO] FPS: {fps:.1f} | Area: {int(mask_area)} | Delta: {delta_pixel_val:+.1f} px",
                    end="",
                )
            else:
                status_text = "Lost (Area < 2000)"
                print(
                    f"\r[INFO] FPS: {fps:.1f} | Area: {int(mask_area)} | Target Lost",
                    end="",
                )

            ref_px_draw = int(ref_pos * (h - 1))
            cv2.line(im0, (0, ref_px_draw), (w - 1, ref_px_draw), (0, 0, 255), 1)

            if crop_top > 0:
                y_top = int(h * crop_top)
                cv2.line(im0, (0, y_top), (w - 1, y_top), (0, 0, 0), 1)
            if crop_bottom > 0:
                y_bot = int(h * (1.0 - crop_bottom))
                cv2.line(im0, (0, y_bot), (w - 1, y_bot), (0, 0, 0), 1)
            if crop_x > 0:
                x_left = int(w * crop_x)
                x_right = int(w * (1.0 - crop_x))
                cv2.line(im0, (x_left, 0), (x_left, h - 1), (0, 0, 0), 1)
                cv2.line(im0, (x_right, 0), (x_right, h - 1), (0, 0, 0), 1)

            cv2.rectangle(im0, (5, 5), (420, 75), (0, 0, 0), -1)
            cv2.putText(
                im0,
                f"Status: {status_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                im0,
                f"Delta Pixel: {offset_text}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                offset_color,
                2,
            )

            fps_text = f"FPS: {fps:.1f}"
            fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv2.rectangle(im0, (w - fps_size[0] - 20, 5), (w - 5, 45), (0, 0, 0), -1)
            cv2.putText(
                im0,
                fps_text,
                (w - fps_size[0] - 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            if abs(delta_pixel_val) < 5:
                delta_pixel_val = 0.0

            if show_det and det is not None and len(det):
                annotator = Annotator(im0, line_width=2, pil=False, example=str(names))
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf_det, cls_det in det:
                    c = int(cls_det)
                    name = names[c] if isinstance(names, (list, tuple)) and c < len(names) else str(c)
                    label = f"{name} {float(conf_det):.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))
                im0 = annotator.im

            if ros_pub is not None:
                try:
                    msg = moveview_return()
                    msg.header = Header()
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = "camera_link"
                    msg.stop_flag = stop_flag_val
                    msg.delta_camera_pixel_height = -delta_pixel_val
                    msg.delta_camera_height = delta_height_val
                    msg.strawberries = []
                    msg.cloud_out = PointCloud2()
                    if ros_publish_image:
                        try:
                            msg.image_out = numpy_to_ros_image(im0, encoding="bgr8")
                        except Exception as e_img:
                            LOGGER.warning(f"[WARN] image_out 发布失败（字段可能不存在）：{e_img}")
                    ros_pub.publish(msg)
                except Exception as e:
                    if i % 30 == 0:
                        LOGGER.warning(f"Publish failed: {e}")

            if save_video:
                if video_writer is None:
                    video_writer = create_video_writer(save_path, im0, fps=max(fps, 25.0))
                video_writer.write(im0)

            if show_window:
                try:
                    cv2.imshow("Rack Line Topic Bridge", im0)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print()
                        LOGGER.info("[INFO] 用户按下 'q'，退出。")
                        if video_writer is not None:
                            video_writer.release()
                        cv2.destroyAllWindows()
                        return
                except cv2.error as e:
                    LOGGER.warning(f"[WARN] cv2.imshow 不可用，已自动关闭窗口显示: {e}")
                    show_window = False

    print()
    if video_writer is not None:
        video_writer.release()
    if show_window:
        cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser(description="ROS 图像话题订阅 → moveview_return 发布")
    parser.add_argument(
        "--ros-topic",
        type=str,
        default="arm_0/perception_binocular_image_raw_pair",
        help="订阅的 ROS 图像话题",
    )
    parser.add_argument("--crop-top", type=float, default=0.1, help="掩码顶部裁剪比例 (0~1)")
    parser.add_argument("--crop-bottom", type=float, default=0.1, help="掩码底部裁剪比例 (0~1)")
    parser.add_argument(
        "--crop-x",
        type=float,
        default=0.1,
        help="掩码左右裁剪比例 (0~0.5)，左右对称",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=["./weights/yolov9-pan-strawberry7cls_rackcls1_v2.pt"],
        help="模型权重",
    )
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="推理尺寸 h,w")
    parser.add_argument("--conf", type=float, default=0.8, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    parser.add_argument("--ref-pos", type=float, default=0.3, help="参考线高度 (0~1)")
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        help="语义 stuff class id；多个时只取第一个",
    )
    parser.add_argument("--device", type=str, default="", help="cuda device 或 cpu")
    parser.add_argument("--save-video", action="store_true", help="保存结果视频")
    parser.add_argument("--save-path", type=str, default="rack_result.mp4", help="视频保存路径")
    parser.add_argument("--show-window", default=True, help="弹窗显示")
    parser.set_defaults(ros_publish=True, ros_publish_image=True)
    parser.add_argument("--no-ros-publish", action="store_false", dest="ros_publish", help="不发布 moveview_return")
    parser.add_argument(
        "--no-ros-publish-image",
        action="store_false",
        dest="ros_publish_image",
        help="发布消息中不包含 image_out",
    )
    parser.add_argument(
        "--ros-publish-topic",
        type=str,
        default="arm_0/picking/moveview_return",
        help="moveview_return 发布话题",
    )
    parser.add_argument("--show-det", type=int, default=1, help="1=画 det 框，0=不画")

    opt = parser.parse_args()
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    main(parse_opt())
