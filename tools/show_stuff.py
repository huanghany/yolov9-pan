# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : show_stuff.py
# @Create      : 2025/12/4-16:21
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2025, Huayi Robot Inc. All Rights Reserved.
# @Description : 语义标签可视化

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- 配置参数 ---
# ⚠️ 请根据你的实际路径修改以下变量
# IMAGE_DIR = '/home/huanghanyang/Datasets/rack_datasets/project-145/images'  # 包含原始图像 (e.g., .jpg, .png) 的文件夹
# STUFF_LABEL_DIR = '/home/huanghanyang/Datasets/rack_datasets/project-145/masks-add-bg8'  # 包含 Stuff 标签 (.txt) 的文件夹
# OUTPUT_DIR = '/home/huanghanyang/Datasets/rack_datasets/project-145/vis'  # 结果保存文件夹

IMAGE_DIR = '/home/huanghanyang/Datasets/strawberry/strawberry_datasets_v3/project-120/images_shanxing'  # 包含原始图像 (e.g., .jpg, .png) 的文件夹
STUFF_LABEL_DIR = '/home/huanghanyang/Datasets/strawberry/strawberry_datasets_v3/project-120/hrnet-result/labels'  # 包含 Stuff 标签 (.txt) 的文件夹
OUTPUT_DIR = '/home/huanghanyang/Datasets/strawberry/strawberry_datasets_v3/project-120/hrnet-result/vis'  # 结果保存文件夹

# 类别 ID 到颜色的映射（可以根据需要扩展）
# 颜色为 BGR 格式
# 如果你的 Stuff 类别 ID 是 7，这里应该定义 ID 7 的颜色
COLOR_MAP = {
    # 示例颜色：ID 7 对应 Planting_Rack，使用绿色 (BGR)
    7: (0, 165, 0),  # 深绿色
    # 如果有多个 Stuff 类别，可以在这里添加：
    8: (255, 0, 0),  # 蓝色
}

# 绘制参数
ALPHA = 0.2  # 填充透明度
LINE_THICKNESS = 1  # 多边形边框粗细


# ----------------------------------------

def normalize_polygon(points, img_w, img_h):
    """将归一化的 [x1, y1, x2, y2, ...] 转换为像素坐标 [x1, y1, x2, y2, ...]"""
    pixels = np.array(points, dtype=np.float32)
    pixels[0::2] *= img_w  # x 坐标
    pixels[1::2] *= img_h  # y 坐标
    return pixels.astype(np.int32).reshape((-1, 1, 2))


def visualize_stuff_labels(image_dir, label_dir, output_dir):
    """
    读取图像和 YOLO Stuff 标签，并进行可视化。
    标签绘制顺序：除7外所有标签先绘制，标签7最后绘制，保证7在上层。
    """
    os.makedirs(output_dir, exist_ok=True)

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    if not label_files:
        print(f"找不到 {label_dir} 中的任何 .txt 标签文件。")
        return

    for label_file in tqdm(label_files, desc="Visualizing Stuff Labels"):
        label_path = Path(label_dir) / label_file
        base_name = label_file.replace('.txt', '')

        # 匹配图像文件
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            potential_path = Path(image_dir) / (base_name + ext)
            if potential_path.exists():
                img_path = str(potential_path)
                break

        if not img_path:
            print(f"警告：找不到 {base_name} 的对应图像，跳过。")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"错误：无法读取图像 {img_path}。")
            continue

        img_h, img_w, _ = img.shape
        overlay = img.copy()

        try:
            with open(label_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"读取标签文件 {label_path} 时出错: {e}")
            continue

        # 按类别分组
        lines_7 = [line for line in lines if line.split()[0] == '7']
        lines_other = [line for line in lines if line.split()[0] != '7']

        # 绘制函数
        def draw_lines(draw_lines_list):
            for line in draw_lines_list:
                parts = line.split()
                try:
                    cls_id = int(parts[0])
                    polygon_norm = [float(p) for p in parts[1:]]
                except ValueError:
                    print(f"警告：标签格式错误，跳过该行: {line}")
                    continue

                if cls_id not in COLOR_MAP:
                    print(f"警告：Stuff ID {cls_id} 没有颜色定义，跳过。")
                    continue

                polygon_pts = normalize_polygon(polygon_norm, img_w, img_h)
                color = COLOR_MAP[cls_id]

                # 填充
                cv2.fillPoly(overlay, [polygon_pts], color, lineType=cv2.LINE_AA)
                # 边框
                cv2.polylines(img, [polygon_pts], True, color, thickness=LINE_THICKNESS, lineType=cv2.LINE_AA)

        # 先绘制除7外的标签
        draw_lines(lines_other)
        # 再绘制7标签
        draw_lines(lines_7)

        img_visualized = cv2.addWeighted(img, 1.0 - ALPHA, overlay, ALPHA, 0)
        output_path = Path(output_dir) / (base_name + '.jpg')
        cv2.imwrite(str(output_path), img_visualized)


# ----------------------------------------

if __name__ == "__main__":
    # 确保 Path 对象是字符串，以便 os.path.join 工作
    visualize_stuff_labels(
        image_dir=IMAGE_DIR,
        label_dir=STUFF_LABEL_DIR,
        output_dir=OUTPUT_DIR
    )
    print(f"\n可视化结果已保存到 {OUTPUT_DIR}")