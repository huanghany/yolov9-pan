# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : rack_line_debug.py
# @Create      : 2025/12/8-19:51
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2025, ZenoAI Robot Inc. All Rights Reserved.
# @Description : 作物架实时跟踪调试（支持普通输入与 ROS）

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

try:
    import rospy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Header

    ROS_PY_AVAILABLE = True
except ImportError:
    rospy = None
    PointCloud2 = None
    Header = None
    ROS_PY_AVAILABLE = False

try:
    from dangkang_picking_msgs.msg import moveview_return

    ROS_MOVEVIEW_MSG_AVAILABLE = True
except ImportError:
    ROS_MOVEVIEW_MSG_AVAILABLE = False
    moveview_return = None

from utils.segment.general import masks2segments, process_mask
from tools.rack_line_utils import (
    Annotator,
    DetectMultiBackend,
    IMG_FORMATS,
    LoadImages,
    LoadScreenshots,
    LoadRosTopic,
    LoadStreams,
    LOGGER,
    MaskCenterEstimator,
    VID_FORMATS,
    apply_mask_roi,
    check_file,
    check_imshow,
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


def get_class_color_by_name(class_name: str, fallback_color):
    """按类别名称返回固定 BGR 颜色；未知类别回退到默认颜色。"""
    class_color_map = {
        "Unripe": (0, 200, 0),      # 绿
        "Ripe2": (0, 255, 180),     # 黄绿
        "Ripe4": (0, 255, 255),     # 黄
        "Ripe7": (0, 165, 255),     # 橙
        "Ripe": (0, 0, 255),        # 红
        "Disease": (180, 0, 180),   # 紫
    }
    return class_color_map.get(class_name, fallback_color)


@smart_inference_mode()
def run(
    input_mode: int,
    ros_topic: str,
    crop_top: float,
    crop_bottom: float,
    weights,
    source: str,
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
    ros_publish: bool = False,
    ros_publish_topic: str = "arm_0/picking/moveview_return",
    ros_publish_image: bool = False,
    show_det: bool = True,
    mask_area_thres: float = 2000.0,
    small_area_strategy: str = "lost",
):
    show_det = bool(show_det)
    need_ros = input_mode == 1 or ros_publish or ros_publish_image
    if need_ros:
        if not ROS_PY_AVAILABLE:
            LOGGER.error(
                "[ERROR] 当前环境缺少 rospy 依赖，请先安装 ROS Python 依赖，或改用 --input-mode 0 并关闭 ROS 发布。"
            )
            return
        if not check_roscore():
            LOGGER.error("[ERROR] roscore 未启动，请先运行 'roscore' 后再启动本脚本。")
            return
        if not rospy.core.is_initialized():
            rospy.init_node("rack_line_standalone", anonymous=True)

    ros_pub = None
    if ros_publish or ros_publish_image:
        if not ROS_MOVEVIEW_MSG_AVAILABLE:
            LOGGER.warning(
                "[WARN] 未找到 dangkang_picking_msgs.msg.moveview_return，跳过结果发布。"
            )
        else:
            try:
                ros_pub = rospy.Publisher(
                    ros_publish_topic, moveview_return, queue_size=1
                )
                LOGGER.info(f"[INFO] ROS Publisher initialized: '{ros_publish_topic}'")
            except Exception as e:
                LOGGER.error(f"[ERROR] 创建 ROS Publisher 失败: {e}")

    # --- 模型加载 ---
    device_torch = select_device(device)
    model = DetectMultiBackend(weights, device=device_torch, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # --- 数据加载 ---
    bs = 1
    webcam = False
    single_image_input = False

    if input_mode == 1:
        LOGGER.info(f"[MODE] ROS Topic 输入: {ros_topic}")
        dataset = LoadRosTopic(ros_topic, img_size=imgsz, stride=stride, auto=pt)
    else:
        source = str(source)
        LOGGER.info(f"[MODE] 普通输入源: {source}")
        src_path = Path(source)
        src_suffix = src_path.suffix[1:].lower()
        is_file = src_suffix in (IMG_FORMATS + VID_FORMATS)
        single_image_input = src_suffix in IMG_FORMATS
        is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
        screenshot = source.lower().startswith("screen")

        if is_url and is_file:
            source = check_file(source)

        if webcam:
            check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))

    center_estimator = MaskCenterEstimator(kernel_size=5, use_bbox=False, use_ema=True)
    prev_time = 0.0
    video_writer = None

    LOGGER.info(
        f"Start inference | ref-pos={ref_pos}, crop-top={crop_top}, crop-bottom={crop_bottom}, "
        f"save_video={save_video}, show_window={show_window}, ros_publish={ros_publish}"
    )

    stuff_id = 7  # 语义分割的目标类别 id

    for path, im, im0s, vid_cap, s in dataset:
        if input_mode == 1 and rospy.is_shutdown():
            break

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time != 0 else 0.0
        prev_time = curr_time

        im_tensor = torch.from_numpy(im).to(model.device).float()
        im_tensor /= 255.0
        if im_tensor.ndim == 3:
            im_tensor = im_tensor.unsqueeze(0)

        pred, panoptic_outs = model(im_tensor, augment=False, visualize=False)[:2]
        mask_proto, semantic_logits = panoptic_outs[2], panoptic_outs[3]
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
            im0 = im0s[i].copy() if webcam and not input_mode == 1 else im0s.copy()
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

            if mask_area >= mask_area_thres:
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
                if small_area_strategy == "fruit":
                    fruit_y_centers = []
                    if det is not None and len(det):
                        det_scaled = det.clone()
                        det_scaled[:, :4] = scale_boxes(
                            im_tensor.shape[2:], det_scaled[:, :4], im0.shape
                        ).round()
                        for *xyxy, _, _ in det_scaled:
                            y1 = float(xyxy[1])
                            y2 = float(xyxy[3])
                            fruit_y_centers.append(0.5 * (y1 + y2))

                    if len(fruit_y_centers) > 0:
                        # 选择使到所有果实 y 距离和最小的位置（L1 最优为中位数）。
                        best_y_px = int(np.median(np.array(fruit_y_centers, dtype=np.float32)))
                        best_y_px = int(np.clip(best_y_px, 0, h - 1))
                        ref_px = int(ref_pos * (h - 1))
                        delta_pixel_val = float(ref_px - best_y_px)
                        status_text = f"FruitFallback (Area < {int(mask_area_thres)})"
                        offset_text = f"{delta_pixel_val:.1f} px"
                        offset_color = (255, 200, 0)
                        cv2.line(im0, (0, best_y_px), (w - 1, best_y_px), (255, 200, 0), 2)
                        print(
                            f"\r[INFO] FPS: {fps:.1f} | Area: {int(mask_area)} | Fallback Delta: {delta_pixel_val:+.1f} px",
                            end="",
                        )
                    else:
                        status_text = f"Lost (Area < {int(mask_area_thres)})"
                        print(
                            f"\r[INFO] FPS: {fps:.1f} | Area: {int(mask_area)} | Target Lost",
                            end="",
                        )
                else:
                    status_text = f"Lost (Area < {int(mask_area_thres)})"
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

            # det head 检测结果可视化（bbox + cls/conf）
            if show_det and det is not None and len(det):
                annotator = Annotator(im0, line_width=2, pil=False, example=str(names))
                det = det.clone() # 避免修改原始预测
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
                # masks = process_mask(mask_proto, det[:, 6:], det[:, :4], im0.shape[:2], upsample=True)
                # masks_np = masks.permute(1, 2, 0).cpu().numpy()

                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    c = int(cls)
                    # mask_bool = masks_np[:, :, j] > 0.5
                    # color_layer = np.zeros_like(im0, dtype=np.uint8)
                    # color_layer[mask_bool] = colors(c, True)
                    # im0 = cv2.addWeighted(im0, 1.0, color_layer, 0.3, 0)
                    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(im0, p1, p2, colors(c, True), thickness=2)
                    label = f'{names[c]} {conf:.2f}'
                    w_text, h_text = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    cv2.rectangle(im0, p1, (p1[0] + w_text, p1[1] - h_text - 3), colors(c, True), -1)
                    cv2.putText(im0, label, (p1[0], p1[1] - 2), 0, 1, [255, 255, 255], thickness=2)


                for *xyxy, conf_det, cls_det in det:
                    c = int(cls_det)
                    if isinstance(names, dict):
                        name = str(names.get(c, c))
                    elif isinstance(names, (list, tuple)) and c < len(names):
                        name = str(names[c])
                    else:
                        name = str(c)
                    y_center = int((float(xyxy[1]) + float(xyxy[3])) * 0.5)
                    label = f"{name} {float(conf_det):.2f} y:{y_center}"
                    det_color = get_class_color_by_name(name, colors(c, True))
                    annotator.box_label(xyxy, label, color=det_color)
                # 写回带框图像（后续发布/保存使用）
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
                        # moveview_return 消息体内的图像字段（参考 public_rack_line_ros.py）
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
                    cv2.imshow("Rack Line Tracking", im0)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print()
                        LOGGER.info("[INFO] 用户按下 'q'，退出。")
                        if video_writer is not None:
                            video_writer.release()
                        cv2.destroyAllWindows()
                        return
                except cv2.error as e:
                    # OpenCV 无 GUI 支持时，imshow 会直接抛异常，避免脚本崩溃。
                    LOGGER.warning(f"[WARN] cv2.imshow 不可用，已自动关闭窗口显示: {e}")
                    show_window = False

    if show_window and single_image_input:
        try:
            LOGGER.info("[INFO] 单张图片模式：按任意键关闭窗口。")
            cv2.waitKey(0)
        except cv2.error:
            pass

    print()
    if video_writer is not None:
        video_writer.release()
    if show_window:
        cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-mode", type=int, default=1,
        help="输入模式: 0=普通(source)，1=ROS Topic",
    )
    parser.add_argument(
        "--ros-topic", type=str, default="arm_0/perception_binocular_image_raw_pair",
        help="ROS 图像话题名，仅在 --input-mode=1 时有效",
    )
    parser.add_argument(
        "--crop-top", type=float, default=0.1,
        help="顶部裁剪比例 (0.0-1.0)，该区域的掩码将被忽略",
    )
    parser.add_argument(
        "--crop-bottom", type=float, default=0.1,
        help="底部裁剪比例 (0.0-1.0)，该区域的掩码将被忽略",
    )
    parser.add_argument(
        "--crop-x", type=float, default=0.1,
        help="左右裁剪比例 (0.0-0.5)，该区域的掩码将被忽略（左右各裁剪同等比例）",
    )

    parser.add_argument(
        "--weights", nargs="+", type=str, default="./weights/yolov9-pan-strawberry7cls_rackcls1_v4_0.pt",
        help="模型权重路径",
    )
    parser.add_argument(
        "--source", type=str, 
        # default="/home/hhy/Datasets/strawberry/shanxing_pick/20260306_bag_select_3/2026-03-06_16-30-36_061.jpg",
        default="data/images/shanxing_test-20250403-150108_rack-7_left_layer-1_001360.jpg",
        help="输入源: file/dir/URL/glob/screen/0(webcam)",
    )
    parser.add_argument(
        "--imgsz", nargs="+", type=int, default=[640],
        help="推理尺寸 h,w",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="置信度阈值",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU 阈值",
    )
    parser.add_argument(
        "--ref-pos", type=float, default=0.5,
        help="参考水平线高度 (0.0-1.0)",
    )
    parser.add_argument(
        "--classes", nargs="+", type=int, default=None,
        help="语义目标 stuff class id（替代 data/yaml）；传多个时只取第一个用于掩码/轨迹",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="cuda device, 如 0 或 0,1,2,3 或 cpu",
    )
    parser.add_argument(
        "--save-video", default=False,
        help="是否保存结果视频",
    )
    parser.add_argument(
        "--save-path", type=str, default="rack_result.mp4",
        help="结果视频保存路径（当 --save-video 启用时有效）",
    )
    parser.add_argument(
        "--show-window", default=True,
        help="是否显示可视化窗口",
    )
    parser.add_argument(
        "--ros-publish", default=True,
        help="发布 moveview_return 到 ROS（需 dangkang_picking_msgs 与 roscore）",
    )
    parser.add_argument(
        "--ros-publish-topic", type=str, default="arm_0/picking/moveview_return",
        help="moveview_return 发布话题名（默认与 public_rack_line_ros 一致）",
    )
    parser.add_argument(
        "--ros-publish-image", default=True,
        help="在发布 moveview_return 时把结果图像填入 msg.image_out",
    )
    parser.add_argument(
        "--show-det", type=int, default=1,
        help="是否可视化 det head 的 bbox/cls/conf：1=显示，0=不显示",
    )
    parser.add_argument(
        "--mask-area-thres", type=float, default=20000.0,
        help="作物架掩码面积阈值；低于该值时启用果实位置估计 y",
    )
    parser.add_argument(
        "--small-area-strategy", type=str, default="fruit", choices=["fruit", "lost"],
        help="小面积策略: lost=保持旧策略(直接丢失), fruit=使用果实位置估计 y",
    )

    opt = parser.parse_args()
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    main(parse_opt())

