# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : split_datasets_new.py
# @Create      : 2026/4/16-15:20
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2025, ZenoAI Inc. All Rights Reserved.
# @Description : 更新后数据集分割脚本（同果梗数据集） 遍历母文件夹下的子文件夹，对 images/labels/hrnet-result(stuff) 进行划分并汇总 并支持生成标签可视化图像
# 划分数据集，并严格检查 label 和 stuff 的配对情况

import random
import shutil
import argparse
import cv2
import numpy as np
import os
from pathlib import Path

# 定义可视化颜色 (BGR)
COLORS = {
    0: (255, 0, 0),  # 蓝色
    1: (0, 255, 0),  # 绿色
    7: (0, 255, 255),  # 黄色 (Rack)
    8: (255, 0, 255),  # 紫色 (Background)
}


def draw_yolo_labels(image_path, label_path, stuff_path, save_path):
    """可视化函数：绘制多边形标签"""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]

    def draw_file(path, is_stuff=False):
        if not path or not Path(path).exists():
            return
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            data = line.strip().split()
            if len(data) < 3: continue
            try:
                cls_id = int(data[0])
                points = np.array([float(x) for x in data[1:]]).reshape(-1, 2)
                points[:, 0] *= w
                points[:, 1] *= h
                points = points.astype(np.int32)
                color = COLORS.get(cls_id, (0, 0, 255))
                # 填充半透明层
                overlay = img.copy()
                cv2.fillPoly(overlay, [points], color)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                # 绘制轮廓
                cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
                label_text = f"{'S' if is_stuff else 'L'}:{cls_id}"
                cv2.putText(img, label_text, (points[0][0], points[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except:
                pass

    draw_file(label_path)
    draw_file(stuff_path, is_stuff=True)
    cv2.imwrite(str(save_path), img)


def split_and_merge(root_dirs, output_root, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                    seed=42, overwrite=False, visualize=False):
    output_root = Path(output_root)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        print("[ERROR] 比例总和不为 1")
        return

    # 创建目录
    categories = ["images", "labels", "stuff"]
    if visualize: categories.append("visuals")
    splits = ["train", "val", "test"]
    for cat in categories:
        for sp in splits:
            (output_root / cat / sp).mkdir(parents=True, exist_ok=True)

    counts = {"total": 0, "train": 0, "val": 0, "test": 0, "missing": 0}

    for root_path in root_dirs:
        root_path = Path(root_path)
        subfolders = [d for d in root_path.iterdir() if d.is_dir()]

        for sub in subfolders:
            images_dir = sub / "images"
            labels_dir = sub / "labels"
            stuff_dir = sub / "hrnet-result" / "labels"

            if not images_dir.exists():
                continue

            image_files = sorted([p for p in images_dir.iterdir() if p.is_file()])
            random.seed(seed)
            random.shuffle(image_files)

            num_total = len(image_files)
            n_train = int(num_total * train_ratio)
            n_val = int(num_total * val_ratio)

            split_data = [
                (image_files[:n_train], "train"),
                (image_files[n_train: n_train + n_val], "val"),
                (image_files[n_train + n_val:], "test")
            ]

            for current_files, split_name in split_data:
                for img_p in current_files:
                    stem = img_p.stem

                    # --- 1. 检查并拷贝 Instance Labels ---
                    target_label_file = None
                    label_src = None
                    for ext in [".txt", ".json", ".xml"]:
                        tmp = labels_dir / f"{stem}{ext}"
                        if tmp.exists():
                            label_src = tmp
                            break

                    if label_src:
                        target_label_file = output_root / "labels" / split_name / label_src.name
                        shutil.copy2(label_src, target_label_file)
                    else:
                        print(f"\033[91m[MISSING LABEL]\033[0m {img_p.name} in {sub}")
                        counts["missing"] += 1

                    # --- 2. 检查并拷贝 Stuff Labels ---
                    target_stuff_file = None
                    stuff_src = stuff_dir / f"{stem}.txt"
                    if stuff_src.exists():
                        target_stuff_file = output_root / "stuff" / split_name / f"{stem}.txt"
                        shutil.copy2(stuff_src, target_stuff_file)
                    else:
                        print(f"\033[93m[MISSING STUFF]\033[0m {img_p.name} in {sub}")
                        counts["missing"] += 1

                    # --- 3. 拷贝 Image ---
                    shutil.copy2(img_p, output_root / "images" / split_name / img_p.name)

                    # --- 4. 可视化 ---
                    if visualize:
                        vis_path = output_root / "visuals" / split_name / f"{stem}.jpg"
                        # 只有是 .txt 的 label 才能参与绘图
                        draw_yolo_labels(img_p,
                                         target_label_file if target_label_file and target_label_file.suffix == '.txt' else None,
                                         target_stuff_file,
                                         vis_path)

            counts["total"] += num_total
            counts["train"] += len(split_data[0][0])
            counts["val"] += len(split_data[1][0])
            counts["test"] += len(split_data[2][0])
            print(f"[INFO] 子文件夹 {sub.name} 处理完毕")

    print(f"\n汇总报告: {counts}")
    if counts["missing"] > 0:
        print(f"\033[91m警告：共有 {counts['missing']} 处标签缺失，请查看上方详细日志！\033[0m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dirs", nargs="+",
                        default=["/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v4"])
    parser.add_argument("--output_root", default="/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v4_0_cls1")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--visualize", default=True)
    args = parser.parse_args()

    split_and_merge(args.root_dirs, args.output_root, args.train_ratio, args.val_ratio,
                    args.test_ratio, args.seed, args.overwrite, args.visualize)


if __name__ == "__main__":
    main()