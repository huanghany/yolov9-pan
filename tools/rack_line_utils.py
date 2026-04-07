import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rosgraph
import rospy
import torch

# ---------------------------------------------------------------------------
# 项目内 YOLO / 工具依赖集中入口：rack_line_standalone 只 import 本文件即可与路径解耦。
# 拷贝到独立工程时，只需在本文件内替换为等价实现或保持对 yolov9-pan 的 PYTHONPATH。
# ---------------------------------------------------------------------------
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    check_file,
    check_img_size,
    check_imshow,
    non_max_suppression,
    print_args,
    scale_boxes,
)
from utils.mask_center_cal import MaskCenterEstimator
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

__all__ = [
    "DetectMultiBackend",
    "IMG_FORMATS",
    "VID_FORMATS",
    "LoadImages",
    "LoadScreenshots",
    "LoadStreams",
    "LOGGER",
    "check_file",
    "check_img_size",
    "check_imshow",
    "non_max_suppression",
    "print_args",
    "scale_boxes",
    "MaskCenterEstimator",
    "Annotator",
    "colors",
    "select_device",
    "smart_inference_mode",
    "numpy_to_ros_image",
    "check_roscore",
    "ros_image_to_numpy",
    "draw_semantic_mask",
    "apply_mask_roi",
    "create_video_writer",
    "LoadRosTopic",
]


def numpy_to_ros_image(img_np: np.ndarray, encoding: str = "bgr8"):
    """
    将 OpenCV (numpy) 图像转为 ROS sensor_msgs/Image。
    不依赖 cv_bridge。
    """
    from sensor_msgs.msg import Image

    if img_np is None:
        raise ValueError("img_np is None")
    if img_np.ndim != 3:
        raise ValueError(f"expected HWC image, got shape={img_np.shape}")

    msg = Image()
    msg.height = int(img_np.shape[0])
    msg.width = int(img_np.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(len(img_np[0]) * img_np.itemsize * img_np.shape[2])
    msg.data = img_np.tobytes()
    return msg


def check_roscore(timeout: float = 2.0) -> bool:
    """
    检查 roscore 是否在线。

    :param timeout: 超时时间（秒）
    :return: True 表示 roscore 正常可用，False 表示不可用
    """
    start = time.time()
    while time.time() - start < timeout:
        if rosgraph.is_master_online():
            return True
        time.sleep(0.1)
    return False


def ros_image_to_numpy(msg) -> np.ndarray:
    """
    将 ROS Image 消息转换为 OpenCV/Numpy BGR 图像。
    这里不依赖 cv_bridge，方便在 Conda 等环境中使用。
    """
    dtype_map = {
        "8UC1": np.uint8,
        "8UC3": np.uint8,
        "bgr8": np.uint8,
        "rgb8": np.uint8,
        "mono8": np.uint8,
    }
    np_dtype = dtype_map.get(msg.encoding, np.uint8)

    if "bgr" in msg.encoding or "rgb" in msg.encoding or "C3" in msg.encoding:
        channels = 3
    else:
        channels = 1

    img = np.frombuffer(msg.data, dtype=np_dtype)
    img = img.reshape((msg.height, msg.width, -1))

    if msg.encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if channels == 1 and len(img.shape) == 3:
        img = img.squeeze(-1)

    return img


def draw_semantic_mask(
    im0: np.ndarray,
    semantic_logits: Optional[torch.Tensor],
    stuff_class_id: int,
    alpha: float,
    im_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    在原图上叠加语义分割结果（底色）。
    """
    if semantic_logits is None:
        return im0

    h, w = im0.shape[:2]
    semantic_pred = torch.argmax(semantic_logits, dim=0).cpu().numpy().astype(np.uint8)

    if im_shape is None:
        semantic_resized = cv2.resize(semantic_pred, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        semantic_resized = cv2.resize(
            semantic_pred, (im_shape[1], im_shape[0]), interpolation=cv2.INTER_NEAREST
        )
        semantic_resized = cv2.resize(semantic_resized, (w, h), interpolation=cv2.INTER_NEAREST)

    stuff_mask = semantic_resized == stuff_class_id
    if np.any(stuff_mask):
        color_layer = np.zeros_like(im0, dtype=np.uint8)
        color_layer[stuff_mask] = colors(stuff_class_id, True)
        im0 = cv2.addWeighted(im0, 1.0, color_layer, alpha, 0)
    return im0


def apply_mask_roi(
    mask_binary: np.ndarray,
    crop_x: float = 0.3,
    crop_y_top: float = 0.0,
    crop_y_bot: float = 0.0,
) -> np.ndarray:
    """
    对二值掩码进行上下左右裁剪，去掉边缘区域噪声。
    """
    h, w = mask_binary.shape

    if crop_x > 0:
        crop_px = int(w * crop_x)
        mask_binary[:, :crop_px] = 0
        mask_binary[:, -crop_px:] = 0

    if crop_y_top > 0:
        crop_px_top = int(h * crop_y_top)
        mask_binary[:crop_px_top, :] = 0

    if crop_y_bot > 0:
        crop_px_bot = int(h * crop_y_bot)
        mask_binary[-crop_px_bot:, :] = 0

    return mask_binary


def create_video_writer(
    save_path: str,
    frame: np.ndarray,
    fps: float = 25.0,
) -> cv2.VideoWriter:
    """
    根据首帧创建视频保存对象。
    """
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    return writer


class LoadRosTopic:
    """
    简化版 ROS 图像话题数据加载器。
    返回格式与 YOLOv9 中的 LoadStreams/LoadImages 兼容：
    (path, im, im0, cap, s)
    """

    def __init__(
        self,
        topic: str,
        img_size=640,
        stride=32,
        auto=True,
        wait_warn_interval: float = 2.0,
        wait_sleep: float = 0.01,
    ):
        from utils.augmentations import letterbox
        from sensor_msgs.msg import Image
        import threading

        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.topic = topic
        self.letterbox = letterbox
        self.LOGGER = LOGGER

        self.latest_msg = None
        self.lock = threading.Lock()
        self.new_data = False
        self.wait_warn_interval = float(wait_warn_interval)
        self.wait_sleep = float(wait_sleep)
        self._waiting_printed = False
        self._last_msg_time = 0.0

        LOGGER.info(f"[INFO] Subscribing to ROS topic: {topic}")
        # ROS1: rospy.Subscriber(name, data_class, callback=..., queue_size=...)
        # 这里默认订阅 sensor_msgs/Image，避免不同 rospy 版本对关键字参数不兼容。
        rospy.Subscriber(topic, Image, callback=self._raw_callback, queue_size=1)

    def _raw_callback(self, msg):
        # 这里假设消息类型为 sensor_msgs/Image；如果外部已经指定类型可自行修改
        from sensor_msgs.msg import Image

        if not isinstance(msg, Image):
            return
        with self.lock:
            self.latest_msg = msg
            self.new_data = True
            self._last_msg_time = time.time()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        if rospy.is_shutdown():
            raise StopIteration

        import time as _time

        while not self.new_data:
            if rospy.is_shutdown():
                raise StopIteration

            # 只有“超过阈值时间仍未收到新消息”才提示一次，避免正常帧间间隔也刷“等待数据”
            now = _time.time()
            if (
                not self._waiting_printed
                and (self._last_msg_time <= 0.0 or (now - self._last_msg_time) >= self.wait_warn_interval)
            ):
                # 外部可能用 print(..., end="")，这里加换行避免粘连
                self.LOGGER.info(f"\n[ROS] 等待数据: {self.topic}")
                self._waiting_printed = True

            _time.sleep(self.wait_sleep)

        with self.lock:
            msg = self.latest_msg
            self.new_data = False
            self._waiting_printed = False

        im0 = ros_image_to_numpy(msg)
        h, w = im0.shape[:2]
        im0 = im0[:, : w // 2]

        im = self.letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        return self.topic, im, im0, None, ""

    def __len__(self):
        # 无限流
        return 0

