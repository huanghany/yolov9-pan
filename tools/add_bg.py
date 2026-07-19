import os
import cv2
import numpy as np
from pathlib import Path

# ===== 配置 =====
SPLIT = ""
# inst_txt_dir = Path(f"/home/huanghanyang/Datasets/rack_datasets/project-145/labels_0to6")  # instance多边形txt目录
# stuff_mask_dir = Path(f"/home/huanghanyang/Datasets/rack_datasets/project-145/masks_1")  # stuff掩码图目录
# out_txt_dir = Path(f"/home/huanghanyang/Datasets/rack_datasets/project-145/masks-add-bg8")

inst_txt_dir = Path(f"/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v3/project-149-yolo/labels_0to6")  # instance多边形txt目录
stuff_mask_dir = Path(f"/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v3/project-149-yolo/masks")  # stuff掩码图目录
out_txt_dir = Path(f"/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v3/project-149-yolo/masks_1")

out_txt_dir.mkdir(parents=True, exist_ok=True)

# 像素值映射：掩码中的值 -> 类别ID
# 需求：将 255 像素值映射为 7
PIXEL_TO_CLASS = {255: 7}
# 需求：将背景类保持为 8
BACKGROUND_CLASS_ID = 8

# ==============================================================================
# 辅助函数
# ==============================================================================

def read_instance_mask(txt_path, shape_hw):
    """
    从实例标签txt文件（YOLO Polygon格式）生成二值实例mask。
    注意：此函数假定实例txt中的坐标是归一化的 (0-1)。

    Args:
        txt_path (Path): 实例标签文件路径。
        shape_hw (tuple): 图像的 (height, width)。

    Returns:
        np.array: 实例二值掩码 (0: 背景, 1: 实例)。
    """
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if not txt_path.exists():
        return mask

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            # 假设格式: <cls_id> <x1> <y1> <x2> <y2> ... (归一化坐标)
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # 提取归一化多边形坐标 (跳过 cls_id)
            normalized_coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)

            # 还原到像素坐标
            pts = np.round(normalized_coords * np.array([w, h])).astype(np.int32)

            # 填充实例掩码 (值为 1)
            cv2.fillPoly(mask, [pts], color=1)

    return mask


def mask_to_polygons(mask):
    """
    将包含不同类别ID的mask转换为归一化的多边形坐标。

    Args:
        mask (np.array): 包含类别ID的灰度掩码。

    Returns:
        dict: {cls_id: [poly1, poly2, ...]}，其中 poly 是像素坐标。
    """
    polygons_by_class = {}
    classes = np.unique(mask)

    for cls_id in classes:
        if cls_id == 0:  # 跳过 0，因为 0 通常用于填充，但在这里 8 是背景。
            continue

        bin_mask = (mask == cls_id).astype(np.uint8)
        # 寻找外部轮廓
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for cnt in contours:
            # 轮廓点数大于等于 3 才能形成多边形
            if len(cnt) >= 3:
                # 简化多边形 (可选，但推荐用于减少点数)
                epsilon = 0.001 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                polygons.append(approx.reshape(-1, 2))

        if polygons:
            polygons_by_class[cls_id] = polygons

    return polygons_by_class


def save_polygons(path, polygons_by_class, img_size):
    """
    将多边形保存为 TXT 文件 (格式: <cls_id> <x1> <y1> <x2> <y2> ... 归一化)

    Args:
        path (Path): 输出文件路径。
        polygons_by_class (dict): {cls_id: [poly1, poly2, ...]}。
        img_size (tuple): 图像的 (height, width)。
    """
    h, w = img_size
    with open(path, "w", encoding="utf-8") as f:
        for cls_id, polys in polygons_by_class.items():
            for poly in polys:
                # 展平并归一化坐标
                coords = []
                for x, y in poly:
                    coords.extend([x / w, y / h])  # 归一化

                # 写入文件，坐标保留 6 位小数
                f.write(f"{cls_id} " + " ".join(f"{c:.6f}" for c in coords) + "\n")


# ==============================================================================
# 主处理逻辑
# ==============================================================================
def main():
    """
    处理所有 Stuff 掩码，从中减去实例区域，并保存为 Stuff Polygon 标签。
    """
    print(f"Starting Stuff label conversion for split: {SPLIT}")

    for mask_path in stuff_mask_dir.glob("*.png"):
        img_name = mask_path.stem

        # 1. 读取 Stuff 掩码
        stuff_mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if stuff_mask_raw is None:
            print(f"Warning: Could not read mask at {mask_path}")
            continue

        h, w = stuff_mask_raw.shape

        # 2. 读取实例标签并生成实例掩码
        inst_txt_path = inst_txt_dir / f"{img_name}.txt"
        # inst_mask: 实例区域为 1，其他为 0
        inst_mask = read_instance_mask(inst_txt_path, (h, w))

        # 3. 初始化最终的类别掩码
        class_mask = np.zeros_like(stuff_mask_raw, dtype=np.uint8)

        # 4. 映射 Stuff 区域
        # 将 255 像素值映射为 PIXEL_TO_CLASS[255] (即 7)
        for pix_val, cls_id in PIXEL_TO_CLASS.items():
            class_mask[stuff_mask_raw == pix_val] = cls_id

        # 5. 确定背景区域并映射
        # 背景定义：(原掩码中像素值为 0) 且 (不属于任何实例区域)
        # 这样可以确保 Stuff 标签不会覆盖 Things 标签。
        bg_mask = (stuff_mask_raw == 0) & (inst_mask == 0)
        class_mask[bg_mask] = BACKGROUND_CLASS_ID  # 背景类 ID (即 8)

        # 6. 提取 Polygons
        # 仅处理 class_mask 中非 0 的区域 (即 7 和 8)
        polygons_by_class = mask_to_polygons(class_mask)

        # 7. 保存到目标 TXT 文件
        save_polygons(out_txt_dir / f"{img_name}.txt", polygons_by_class, (h, w))

        print(f"Processed and saved Stuff labels for {img_name}")


if __name__ == "__main__":
    main()