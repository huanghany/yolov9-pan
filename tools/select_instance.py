# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : select_instance.py
# @Create      : 2025/12/4-20:51
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2025, Huayi Robot Inc. All Rights Reserved.
# @Description : 根据 images 文件夹里的图片名字，从另一个 labels 文件夹中挑选对应 txt 标签文件

import shutil
from pathlib import Path


def select_labels(images_dir, labels_dir, output_labels_dir, overwrite=False):
    """
    根据 images_dir 中的图片文件名，从 labels_dir 中挑选同名 txt 文件
    Args:
        images_dir (str or Path): 图片所在文件夹
        labels_dir (str or Path): 标签所在文件夹（txt）
        output_labels_dir (str or Path): 输出标签文件夹
        overwrite (bool): 是否覆盖已存在文件
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_labels_dir = Path(output_labels_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir 不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir 不存在: {labels_dir}")

    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # 获取图片的文件名（不含后缀）
    stems = {p.stem for p in images_dir.iterdir() if p.is_file()}

    copied_count = 0
    missing_count = 0
    for stem in stems:
        label_file = labels_dir / f"{stem}.txt"
        if label_file.exists():
            dst_file = output_labels_dir / label_file.name
            if overwrite or not dst_file.exists():
                shutil.copy2(label_file, dst_file)
            copied_count += 1
        else:
            missing_count += 1

    print(f"[DONE] 已复制 {copied_count} 个标签文件到 {output_labels_dir}")
    if missing_count > 0:
        print(f"[WARN] 有 {missing_count} 个标签文件缺失")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="根据图片文件名选择对应标签文件")
    parser.add_argument("--images_dir", default="/home/huanghanyang/Datasets/rack_datasets/project-140/images", help="图片文件夹路径")
    parser.add_argument("--labels_dir", default="/home/huanghanyang/Datasets/rack_datasets/project-140/labels_origin_0to6", help="全部标签文件夹路径")
    parser.add_argument("--output_labels_dir", default="/home/huanghanyang/Datasets/rack_datasets/project-140/labels", help="输出标签文件夹路径")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在文件")
    args = parser.parse_args()

    select_labels(args.images_dir, args.labels_dir, args.output_labels_dir, args.overwrite)


if __name__ == "__main__":
    main()
