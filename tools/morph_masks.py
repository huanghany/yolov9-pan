# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : morph_masks.py
# @Create      : 2025/12/04
# @Description : 对mask文件夹中的所有mask进行膨胀+腐蚀，使边界更加贴合

import cv2
import numpy as np
from pathlib import Path
import argparse

def process_mask(mask_path, out_path, kernel_size=3, iterations=1, mode="close"):
    """
    对单个mask进行形态学操作
    mode: "close" (膨胀再腐蚀, 平滑边缘), "open" (腐蚀再膨胀, 去噪)
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"[WARN] 读取失败: {mask_path}")
        return False

    # 如果是彩色mask，取单通道（假设为灰度标签）
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    h, w = mask.shape[:2]

    # 判断分辨率
    if (w, h) == (4624, 3472) or (w, h) == (3472, 4624):
        print(f"[INFO] 检测到分辨率 {w}x{h} -> 使用大卷积核")
        kernel_size = max(7, kernel_size)  # 确保至少使用7
    else:
        print(f"[INFO] 分辨率 {w}x{h} -> 使用默认卷积核 {kernel_size}")

    # 形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if mode == "close":
        processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif mode == "open":
        processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    else:
        print(f"[ERROR] 不支持的模式: {mode}")
        return False

    cv2.imwrite(str(out_path), processed)
    return True


def main():
    parser = argparse.ArgumentParser(description="对mask进行膨胀+腐蚀以贴合边界")
    parser.add_argument("--mask_dir", default="/home/huanghanyang/Datasets/rack_datasets/origin_v1/masks", help="输入mask目录")
    parser.add_argument("--out_dir", default="/home/huanghanyang/Datasets/rack_datasets/origin_v1/masks_1", help="输出目录")
    parser.add_argument("--kernel_size", type=int, default=5, help="卷积核大小(默认)，建议奇数")
    parser.add_argument("--iterations", type=int, default=3, help="形态学操作迭代次数")
    parser.add_argument("--mode", choices=["close", "open"], default="close", help="形态学模式")
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for mask_path in mask_dir.rglob("*"):
        if not mask_path.is_file():
            continue
        out_path = out_dir / mask_path.name
        if process_mask(mask_path, out_path, kernel_size=args.kernel_size, iterations=args.iterations, mode=args.mode):
            processed_count += 1
            print(f"[OK] 处理完成: {mask_path.name}")

    print(f"[DONE] 共处理 {processed_count} 张mask")


if __name__ == "__main__":
    main()