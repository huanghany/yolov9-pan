# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : transform_semantic_label.py
# @Create      : 2025/12/4-16:54
# @Contact     : huanghanyang345@163.com
# @Description : 将label-studio导出的语义分割标签提取出来，并同时提取对应标签文件

import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


def build_taskid_to_filename_map(json_path):
    """解析 Label Studio 导出 JSON，返回 {task_id: original_filename}"""
    def extract_filename_from_url_or_path(v):
        if not v:
            return None
        name = str(v).split("?")[0].split("#")[0]
        return Path(name).name

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for item in data:
        task_id = None
        for key in ("id", "task_id"):
            if key in item and isinstance(item[key], int):
                task_id = item[key]
                break
        if task_id is None and "task" in item and isinstance(item["task"], dict):
            if isinstance(item["task"].get("id"), int):
                task_id = item["task"]["id"]

        filename = None
        data_field = item.get("data") or {}
        for k in ("image", "img", "image_url", "imagePath"):
            if k in data_field:
                filename = extract_filename_from_url_or_path(data_field[k])
                if filename:
                    break
        if not filename and "meta" in item and isinstance(item["meta"], dict):
            filename = item["meta"].get("filename") or item["meta"].get("file_name")
            if filename:
                filename = Path(filename).name
        if not filename and "task" in item and isinstance(item["task"], dict):
            td = item["task"].get("data") or {}
            for k in ("image", "img", "file_name", "image_url", "imagePath"):
                if k in td:
                    filename = extract_filename_from_url_or_path(td[k])
                    if filename:
                        break
        if task_id is None or not filename:
            continue
        mapping[int(task_id)] = filename
    return mapping


def rename_masks_and_select_images(
    json_path,
    images_dir,
    masks_dir,
    out_images_dir,
    out_masks_dir=None,
    mask_suffix="_mask",
    overwrite=False,
    labels_dir=None
):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    out_images_dir = Path(out_images_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)

    if out_masks_dir:
        out_masks_dir = Path(out_masks_dir)
        out_masks_dir.mkdir(parents=True, exist_ok=True)

    out_labels_dir = out_images_dir.parent / "labels"
    if labels_dir:
        labels_dir = Path(labels_dir)
        if labels_dir.exists():
            out_labels_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] 使用标签目录: {labels_dir}")
        else:
            labels_dir = None
            print("[WARN] 标签目录不存在，跳过标签提取")
    else:
        labels_dir = None
        print("[WARN] 未传入标签目录，跳过标签提取")

    print(f"[INFO] 读取 JSON 映射: {json_path}")
    task2name = build_taskid_to_filename_map(json_path)
    if not task2name:
        print("[ERROR] 未提取到映射")
        return
    print(f"[DEBUG] 从 JSON 提取到 {len(task2name)} 条映射")

    pattern = re.compile(r"^task-(\d+)-.*\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)
    task_masks_map = {}
    for mask_path in masks_dir.rglob("*"):
        if not mask_path.is_file():
            continue
        m = pattern.match(mask_path.name)
        if not m:
            continue
        task_id = int(m.group(1))
        task_masks_map.setdefault(task_id, []).append(mask_path)

    matched_task_ids = set()

    # 合并 mask
    for task_id, mask_list in task_masks_map.items():
        original_name = task2name.get(task_id)
        if not original_name:
            continue
        stem = Path(original_name).stem
        ext = Path(mask_list[0]).suffix.lower().lstrip(".")
        new_mask_name = f"{stem}{mask_suffix}.{ext}"

        merged_mask = None
        for mp in mask_list:
            mask_img = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                continue
            if merged_mask is None:
                merged_mask = mask_img.copy()
            else:
                if merged_mask.ndim == 2:
                    merged_mask = np.maximum(merged_mask, mask_img)
                else:
                    merged_mask = np.where(mask_img > 0, mask_img, merged_mask)

        if merged_mask is not None and out_masks_dir:
            out_path = Path(out_masks_dir) / new_mask_name
            cv2.imwrite(str(out_path), merged_mask)

        matched_task_ids.add(task_id)

    # 拷贝原图和对应标签
    copied = 0
    copied_labels = 0
    for task_id in matched_task_ids:
        original_name = task2name.get(task_id)
        if not original_name:
            continue
        candidates = list(images_dir.rglob(original_name))
        if not candidates:
            candidates = [p for p in images_dir.rglob("*") if p.is_file() and p.name == original_name]
        if not candidates:
            continue
        src = candidates[0]
        dst = out_images_dir / src.name
        if not dst.exists() or overwrite:
            shutil.copy2(src, dst)
            copied += 1

        # 处理标签
        if labels_dir:
            stem = Path(original_name).stem
            for ext in ["txt", "json", "xml"]:
                candidate_label = labels_dir / f"{stem}.{ext}"
                if candidate_label.exists():
                    out_label_path = out_labels_dir / candidate_label.name
                    shutil.copy2(candidate_label, out_label_path)
                    copied_labels += 1
                    break

    print(f"[DONE] 合并 {len(matched_task_ids)} 个 mask，复制原图 {copied} 张，复制标签 {copied_labels} 个")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="重命名 Label Studio 导出的 mask，并挑选有 mask 的原图及标签")
    parser.add_argument("--json", default="/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v3/project-149-at-2026-01-08-08-07-faad4ecd.json")
    parser.add_argument("--images_dir", default="/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v3/project-149-yolo/images")
    parser.add_argument("--masks_dir", default="/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v3/project-149-png")
    parser.add_argument("--out_images_dir", default="/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v3/project-149-yolo/images-1")
    parser.add_argument("--out_masks_dir", default="/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v3/project-149-yolo/masks")
    parser.add_argument("--labels_dir", default="/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v3/project-149-yolo/labels_0to6")
    parser.add_argument("--mask_suffix", default="")
    parser.add_argument("--overwrite", default=True)
    args = parser.parse_args()

    rename_masks_and_select_images(
        json_path=args.json,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        out_images_dir=args.out_images_dir,
        out_masks_dir=args.out_masks_dir,
        mask_suffix=args.mask_suffix,
        overwrite=args.overwrite,
        labels_dir=args.labels_dir
    )


if __name__ == "__main__":
    main()