# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : gen_empty_labels.py
# @Create      : 2026/4/16-15:00
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2025, ZenoAI Inc. All Rights Reserved.
# @Description : 生成全部为背景的标签

import os
import argparse
from pathlib import Path


def generate_background_labels(image_dir, save_dir, class_id=8):
    """
    为目录下所有图像生成全图覆盖的背景标签
    YOLO 分割格式: class_id x1 y1 x2 y2 x3 y3 x4 y4 (归一化坐标)
    """
    # 建立目标目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 常见图像后缀
    img_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif')

    # 全图覆盖的多边形坐标 (左上, 右上, 右下, 左下)
    # 格式: x1 y1 x2 y2 x3 y3 x4 y4
    full_poly = "0.000000 0.000000 1.000000 0.000000 1.000000 1.000000 0.000000 1.000000"
    label_content = f"{class_id} {full_poly}\n"

    # 遍历图像
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(img_formats)]

    print(f"开始处理: {image_dir}")
    print(f"目标目录: {save_dir}")
    print(f"设置类别 ID: {class_id}")

    count = 0
    for fname in image_files:
        stem = Path(fname).stem
        txt_name = f"{stem}.txt"

        with open(save_path / txt_name, 'w') as f:
            f.write(label_content)

        count += 1
        if count % 100 == 0:
            print(f"已生成 {count} 个标签...")

    print(f"处理完成！共生成 {count} 个背景标签文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为误识别图像制作全背景标签")
    parser.add_argument('--image-dir', type=str, default="/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v4/project-132/images", help="输入图像文件夹路径")
    parser.add_argument('--save-dir', type=str, default="/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v4/project-132/hrnet-result/labels", help="标签保存文件夹路径")
    parser.add_argument('--id', type=int, default=8, help="背景类别的 ID (默认 8)")

    args = parser.parse_args()

    generate_background_labels(args.image_dir, args.save_dir, args.id)

