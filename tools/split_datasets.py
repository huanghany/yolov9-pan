# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : split_datasets.py
# @Create      : 2025/12/4-17:56
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2025, Huayi Robot Inc. All Rights Reserved.
# @Description : 对多个数据("images", "labels", "stuff")版本进行划分并汇总，先分类型目录，再分train/val

import random
import shutil
from pathlib import Path


def copy_files_by_split(file_list, src_dir, dst_dir, split, overwrite=False):
    """按 split(train/val) 拷贝文件"""
    for file_path in file_list:
        dst_path = dst_dir / split / file_path.name
        if overwrite or not dst_path.exists():
            shutil.copy2(src_dir / file_path.name, dst_path)


def split_and_merge(root_dirs, output_root, train_ratio=0.8, seed=42, overwrite=False):
    """
    对多个数据版本进行划分并合并到三大类型目录(images/labels/stuff)，每个目录内部再分train/val
    Args:
        root_dirs (list): 母文件夹路径列表
        output_root (str or Path): 输出根目录
        train_ratio (float): 训练集比例
        seed (int): 随机种子
        overwrite (bool): 是否覆盖现有文件
    """
    output_root = Path(output_root)

    # 创建大类目录及train/val子目录
    for category in ["images", "labels", "stuff"]:
        for split in ["train", "val"]:
            (output_root / category / split).mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_train = 0
    total_val = 0

    for root_dir in root_dirs:
        root_dir = Path(root_dir)
        images_dir = root_dir / "images"
        labels_dir = root_dir / "labels"
        stuff_dir = root_dir / "stuff"  # 这里 stuff 是 txt 文件

        if not images_dir.exists() or not labels_dir.exists() or not stuff_dir.exists():
            print(f"[WARN] {root_dir} 缺少 images/labels/stuff，跳过")
            continue

        # 按 images 作为划分基准
        image_files = sorted([p for p in images_dir.iterdir() if p.is_file()])
        random.seed(seed)
        random.shuffle(image_files)

        num_train = int(len(image_files) * train_ratio)
        train_images = image_files[:num_train]
        val_images = image_files[num_train:]

        # 复制 images
        copy_files_by_split(train_images, images_dir, output_root / "images", "train", overwrite)
        copy_files_by_split(val_images, images_dir, output_root / "images", "val", overwrite)

        # 复制 labels (.txt/.json/.xml)
        for split_set, split_name in [(train_images, "train"), (val_images, "val")]:
            for img in split_set:
                stem = img.stem
                for ext in [".txt", ".json", ".xml"]:
                    label_file = labels_dir / f"{stem}{ext}"
                    if label_file.exists():
                        shutil.copy2(label_file, output_root / "labels" / split_name / label_file.name)
                        break

        # 复制 stuff（txt文件）
        for split_set, split_name in [(train_images, "train"), (val_images, "val")]:
            for img in split_set:
                stem = img.stem
                stuff_file = stuff_dir / f"{stem}.txt"
                if stuff_file.exists():
                    shutil.copy2(stuff_file, output_root / "stuff" / split_name / stuff_file.name)

        total_images += len(image_files)
        total_train += len(train_images)
        total_val += len(val_images)

        print(f"[INFO] {root_dir} 处理完成：train={len(train_images)}, val={len(val_images)}")

    print(f"[DONE] 所有数据版本汇总完成")
    print(f"       总计图片: {total_images} 张")
    print(f"       训练集: {total_train} 张, 验证集: {total_val} 张")
    print(f"       输出目录: {output_root}/images, {output_root}/labels, {output_root}/stuff")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="对多个数据版本先分类型目录，再划分train/val")
    parser.add_argument("--root_dirs", nargs="+", default=[

        # rack_datasets_v2
        # "/home/huanghanyang/Datasets/rack_datasets/project-145",
        # "/home/huanghanyang/Datasets/rack_datasets/project-140"

        # rack_datasets_v3
        # "/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v3"

        # rack_datasets_v4
        "/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v4"

    ], help="母文件夹路径列表")

    parser.add_argument("--output_root", default="/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v4_0_1cls", help="输出根目录")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--seed", type=int, default=15, help="随机种子")  # rack 15
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在文件")
    args = parser.parse_args()

    split_and_merge(args.root_dirs, args.output_root, args.train_ratio, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
