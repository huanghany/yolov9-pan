# -*- coding: utf-8 -*-
# @Author      : huanghany
# @File        : converse_pixel.py
# @Create      : 2025/12/18-16:50
# @Contact     : huanghanyang345@163.com
# @Copyright   : Copyright (c) 2025, Huayi Robot Inc. All Rights Reserved.
# @Description :
import numpy as np
from PIL import Image
import os

fg_ids = {255,}  # 你要当作前景的 COCO-Stuff 类别 ID


def convert_mask(src_path, dst_path):
    m = np.array(Image.open(src_path))
    bin_mask = np.zeros_like(m, dtype=np.uint8)

    # 把属于前景类的像素设为 1，其它保持 0
    for cid in fg_ids:
        bin_mask[m == cid] = 1

    Image.fromarray(bin_mask).save(dst_path)


# 批量转换
src_dir = "/home/huanghanyang/Datasets/rack_datasets/project-140/masks_morph"
dst_dir = "/home/huanghanyang/Datasets/rack_datasets/project-140/masks_morph_0_1"
os.makedirs(dst_dir, exist_ok=True)

for name in os.listdir(src_dir):
    if not name.endswith(".png"):
        continue
    convert_mask(os.path.join(src_dir, name),
                 os.path.join(dst_dir, name))
