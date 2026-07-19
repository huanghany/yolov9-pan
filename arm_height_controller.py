# -*- coding: utf-8 -*-
# @Author      : huayi.cv
# @File        : arm_height_controller.py
# @Create      : 2025/11/18 15:20
# @Contact     : huayi.aieyes@protonmail.com
# @Copyright   : Copyright (c) 2024, Huayi Robot Inc. All Rights Reserved.
# @Description : 通过图像调整相机的高度以适应采摘
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Optional, Tuple

@contextmanager
def _nullcontext():
    """当不需要 CUDA stream 时的占位 context"""
    yield


class MaskCenterEstimator:
    """
    零拷贝 GPU Mask 中心线估计器（含 GPU 形态学、可选外接矩形补全、EMA）
    - 所有计算尽量留在 GPU，不做 CPU<->GPU 往返
    - 外接矩形补全已矢量化（无 python 循环），适合大 batch
    """

    def __init__(
        self,
        kernel_size: int = 5,
        dtype=torch.float32,
        eps: float = 1e-9,
        use_bbox: bool = False,
        use_ema: bool = False,
        ema_alpha: float = 0.7,
    ):
        assert kernel_size % 2 == 1
        self.k = kernel_size
        self.pad = kernel_size // 2
        self.dtype = dtype
        self.eps = eps
        self.use_bbox = use_bbox
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha

        # 缓存
        self._cached_H = None
        self._y_idx = None
        self._ema_y_center_norm = None  # EMA 缓存（GPU tensor）

    # ------------------------------------------------------------------
    def _ensure_y_idx(self, H: int, device, dtype):
        """确保 y 索引存在并位于目标 device"""
        if self._cached_H == H and self._y_idx is not None and self._y_idx.device == device:
            return
        self._y_idx = torch.arange(H, device=device, dtype=dtype)
        self._cached_H = H

    # ------------------------------------------------------------------
    def _apply_bbox_completion(self, clean: torch.Tensor):
        """
        矢量化外接矩形补全（全 GPU）：
        输入:
            clean: (B,H,W) 二值化/浮点 mask（0/1）
        输出:
            out: (B,H,W) 补全后的 mask（0/1，dtype 与 clean 相同）
        算法路线（矢量化）：
            1. 计算每行是否含前景 row_any = (clean.sum(dim=2) > 0) -> (B,H)
            2. y_min = min(idx where row_any True) ； y_max = max(...)
               用 torch.where(row_any, idxs, H) + .min/.max 来矢量化得到
            3. 同理计算列的 x_min/x_max
            4. 利用广播：y_mask = (y_idx >= y_min[:,None]) & (y_idx <= y_max[:,None]) -> (B,H)
                         x_mask = (x_idx >= x_min[:,None]) & (x_idx <= x_max[:,None]) -> (B,W)
               最终 out = y_mask.unsqueeze(2) & x_mask.unsqueeze(1) -> (B,H,W)
        """
        B, H, W = clean.shape
        device = clean.device
        dtype = clean.dtype

        # 计算行是否有前景： (B,H)
        row_any = (clean.sum(dim=2) > 0)  # bool (B,H)
        col_any = (clean.sum(dim=1) > 0)  # bool (B,W)

        # 若全无前景，直接返回原 clean
        if not row_any.any():
            return clean

        # 准备索引向量
        # y_idx: (H,), x_idx: (W,)
        # 注意使用与 clean 相同的 device
        y_idx = torch.arange(H, device=device)
        x_idx = torch.arange(W, device=device)

        # 计算 y_min: 若 row_any 为 True 则取 y_idx else 取 H，随后取 min
        # 这样若某行没有 True，其值为 H，min 会忽略
        y_min_candidates = torch.where(row_any, y_idx.view(1, H).expand(B, H), H)
        y_min = y_min_candidates.min(dim=1).values  # (B,), dtype same as y_idx (int64)

        # 计算 y_max: 若 row_any True 则取 y_idx else -1，随后取 max
        y_max_candidates = torch.where(row_any, y_idx.view(1, H).expand(B, H), -1)
        y_max = y_max_candidates.max(dim=1).values  # (B,)

        # 列方向同理
        x_min_candidates = torch.where(col_any, x_idx.view(1, W).expand(B, W), W)
        x_min = x_min_candidates.min(dim=1).values  # (B,)

        x_max_candidates = torch.where(col_any, x_idx.view(1, W).expand(B, W), -1)
        x_max = x_max_candidates.max(dim=1).values  # (B,)

        # 将 y_min/y_max/x_min/x_max 转为整型（int64），以便比较
        y_min = y_min.to(torch.int64)
        y_max = y_max.to(torch.int64)
        x_min = x_min.to(torch.int64)
        x_max = x_max.to(torch.int64)

        # 现构造矩形掩码（B,H,W）：通过广播:
        # y_mask: (B,H) True 表示该行在 bbox 范围内
        # x_mask: (B,W)
        # out = y_mask.unsqueeze(2) & x_mask.unsqueeze(1)
        # 为广播正确，先把 y_min,y_max扩展为 (B,1) 与 y_idx (1,H)
        y_min_exp = y_min.view(B, 1)
        y_max_exp = y_max.view(B, 1)
        x_min_exp = x_min.view(B, 1)
        x_max_exp = x_max.view(B, 1)

        # y_idx: (1,H), x_idx: (1,W)
        y_idx_row = y_idx.view(1, H)
        x_idx_row = x_idx.view(1, W)

        # 计算布尔掩码
        y_mask = (y_idx_row >= y_min_exp) & (y_idx_row <= y_max_exp)  # (B,H)
        x_mask = (x_idx_row >= x_min_exp) & (x_idx_row <= x_max_exp)  # (B,W)

        # 若某张图没有前景（y_min==H 或 x_min==W 或 y_max==-1 或 x_max==-1），上面的比较会生成全 False
        out_bool = y_mask.unsqueeze(2) & x_mask.unsqueeze(1)  # (B,H,W) boolean

        # 转为与 clean 相同 dtype（float32）
        out = out_bool.to(dtype)

        return out

    # ------------------------------------------------------------------
    def _apply_ema(self, y_center_norm: torch.Tensor):
        """在 GPU 上对 y_center_norm 做 EMA 平滑"""
        if self._ema_y_center_norm is None:
            self._ema_y_center_norm = y_center_norm.clone()
            return y_center_norm
        alpha = self.ema_alpha
        self._ema_y_center_norm = alpha * y_center_norm + (1 - alpha) * self._ema_y_center_norm
        return self._ema_y_center_norm

    # ------------------------------------------------------------------
    def compute(self, masks: torch.Tensor, *, ref_pos: float = 0.5,
                return_cpu: bool = False, stream: torch.cuda.Stream = None):
        """
        masks: (B,H,W) 或 (H,W) 的 GPU tensor（0/1 或 float）
        ref_pos: 参考位置 0~1（图顶=0, 底=1）
        返回 y_center_norm, dy_norm（默认 GPU tensor；return_cpu=True 则返回 numpy）
        """
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)

        B, H, W = masks.shape
        device = masks.device
        dtype = self.dtype

        masks_t = masks.to(dtype=dtype, non_blocking=True)
        masks_t = (masks_t > 0.5).to(dtype=dtype)  # 二值化为 0/1

        # 缓存 y 索引
        self._ensure_y_idx(H, device, dtype)

        # 选择 stream
        if device.type == "cuda" and stream is not None:
            ctx = torch.cuda.stream(stream)
        else:
            ctx = _nullcontext()

        with ctx:
            # 形态学开运算（erosion -> dilation）
            x = masks_t.unsqueeze(1)                        # (B,1,H,W)
            neg = -x
            eroded = -F.max_pool2d(neg, kernel_size=self.k, stride=1, padding=self.pad)
            clean = F.max_pool2d(eroded, kernel_size=self.k, stride=1, padding=self.pad)
            clean = clean.squeeze(1)                       # (B,H,W)

            # 可选外接矩形补全（已矢量化）
            if self.use_bbox:
                clean = self._apply_bbox_completion(clean)

            # 行投影 + 加权中心
            row_sum = clean.sum(dim=2)                     # (B,H)
            total = row_sum.sum(dim=1)                     # (B,)
            total_safe = torch.clamp(total, min=self.eps)

            weighted = (row_sum * self._y_idx.view(1, H)).sum(dim=1)
            y_center = weighted / total_safe               # (B,)

            y_center_norm = y_center / (H - 1)
            if self.use_ema:
                y_center_norm = self._apply_ema(y_center_norm)

            dy_norm = y_center_norm - ref_pos

        if return_cpu:
            return y_center_norm.detach().cpu().numpy(), dy_norm.detach().cpu().numpy()

        return y_center_norm, dy_norm


def gen_complex_mask_batch(B=4, H=480, W=640, seed=None):
    """
    生成复杂 mask batch，用于测试中心线估计
    """
    if seed is not None:
        np.random.seed(seed)

    masks = torch.zeros((B, H, W), dtype=torch.float32)

    for b in range(B):
        mask = np.zeros((H, W), dtype=np.float32)
        # 随机决定包含几种类型
        num_shapes = np.random.randint(1, 4)  # 1~3 种组合
        types = np.random.choice([0, 1, 2, 3, 4], size=num_shapes, replace=False)

        for t in types:
            if t == 0:
                # 不完整矩形：矩形边界随机裁切
                h1 = np.random.randint(50, 150)
                h2 = np.random.randint(300, 400)
                w1 = np.random.randint(50, 150)
                w2 = np.random.randint(400, 600)
                mask[h1:h2, w1:w2] = 1.0
                # 随机裁切部分边缘
                cut_top = np.random.randint(0, int((h2 - h1) * 0.3))
                cut_bottom = np.random.randint(0, int((h2 - h1) * 0.3))
                mask[h1:h1 + cut_top, w1:w2] = 0
                mask[h2 - cut_bottom:h2, w1:w2] = 0
            elif t == 1:
                # 倾斜矩形
                h, w = np.random.randint(100, 200), np.random.randint(150, 300)
                x0, y0 = np.random.randint(0, W - w), np.random.randint(0, H - h)
                rect = np.ones((h, w), dtype=np.float32)
                # 生成旋转矩形
                angle = np.random.randint(-20, 20)
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                rect_rot = cv2.warpAffine(rect, M, (w, h))
                mask[y0:y0 + h, x0:x0 + w] = np.clip(mask[y0:y0 + h, x0:x0 + w] + rect_rot, 0, 1)
            elif t == 2:
                # 边缘波浪矩形
                h1 = np.random.randint(60, 150)
                h2 = np.random.randint(300, 400)
                w1 = np.random.randint(50, 150)
                w2 = np.random.randint(400, 600)
                x = np.arange(w2 - w1)
                y_wave = (np.sin(2 * np.pi * 3 * x / (w2 - w1)) * 5).astype(int)  # 振幅5像素
                for xi, dy in enumerate(y_wave):
                    mask[h1 + dy:h2 + dy, w1 + xi] = 1.0
            elif t == 3:
                # 被遮挡的矩形
                h1 = np.random.randint(50, 150)
                h2 = np.random.randint(300, 400)
                w1 = np.random.randint(50, 150)
                w2 = np.random.randint(400, 600)
                mask[h1:h2, w1:w2] = 1.0
                # 遮挡小块
                for _ in range(np.random.randint(1, 4)):
                    oc_h = np.random.randint(20, 50)
                    oc_w = np.random.randint(20, 50)
                    oy = np.random.randint(h1, h2 - oc_h)
                    ox = np.random.randint(w1, w2 - oc_w)
                    mask[oy:oy + oc_h, ox:ox + oc_w] = 0
            elif t == 4:
                # 大量孔洞缺失
                h1 = np.random.randint(50, 150)
                h2 = np.random.randint(300, 400)
                w1 = np.random.randint(50, 150)
                w2 = np.random.randint(400, 600)
                mask[h1:h2, w1:w2] = 1.0
                # 挖洞
                for _ in range(np.random.randint(5, 15)):
                    dh = np.random.randint(10, 30)
                    dw = np.random.randint(10, 30)
                    oy = np.random.randint(h1, h2 - dh)
                    ox = np.random.randint(w1, w2 - dw)
                    mask[oy:oy + dh, ox:ox + dw] = 0

        # 随机噪声
        noise = (np.random.rand(H, W) > 0.985).astype(np.float32)
        mask = np.clip(mask + noise, 0, 1)
        masks[b] = torch.from_numpy(mask)

    return masks


def process_masks_from_directory(
    masks_dir: str = "../tests/assets/masks",
    output_dir: str = "../tests/masks_center_res",
    ref_pos: float = 0.5,
    kernel_size: int = 5,
    use_bbox: bool = False,
    use_ema: bool = True,
    ema_alpha: float = 0.7,
    batch_size: int = 8,
    device: Optional[str] = None,
    save_visualization: bool = True,
    save_results_csv: bool = True,
):
    """
    批量处理 masks 目录中的二值化图像，计算中心线并保存结果
    
    参数:
        masks_dir: 输入 masks 目录路径（相对于当前文件或绝对路径）
        output_dir: 输出结果目录路径
        ref_pos: 参考位置 0~1（图顶=0, 底=1）
        kernel_size: 形态学操作核大小（必须为奇数）
        use_bbox: 是否使用外接矩形补全
        use_ema: 是否使用 EMA 平滑
        ema_alpha: EMA 平滑系数
        batch_size: 批处理大小
        device: 设备 ('cuda' 或 'cpu')，None 则自动选择
        save_visualization: 是否保存可视化图像（带中心线标记）
        save_results_csv: 是否保存数值结果到 CSV 文件
    
    返回:
        results: 字典，包含所有处理结果
    """
    # 确定设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 解析路径（相对于当前文件所在目录）
    current_file_dir = Path(__file__).parent.absolute()
    masks_path = Path(masks_dir) if Path(masks_dir).is_absolute() else current_file_dir / masks_dir
    output_path = Path(output_dir) if Path(output_dir).is_absolute() else current_file_dir / output_dir
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    if save_visualization:
        (output_path / "visualizations").mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = sorted([f for f in masks_path.iterdir() 
                         if f.suffix.lower() in image_extensions])
    
    if len(image_files) == 0:
        print(f"警告: 在 {masks_path} 中未找到图像文件")
        return {}
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 初始化估计器
    estimator = MaskCenterEstimator(
        kernel_size=kernel_size,
        use_bbox=use_bbox,
        use_ema=use_ema,
        ema_alpha=ema_alpha,
    )
    
    # 存储所有结果
    all_results = []
    
    # 批量处理
    for batch_start in range(0, len(image_files), batch_size):
        batch_files = image_files[batch_start:batch_start + batch_size]
        batch_masks = []
        batch_names = []
        
        # 读取当前批次的图像
        for img_file in batch_files:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"警告: 无法读取 {img_file}")
                continue
            
            # 转换为二值化 mask (0/1)
            # 如图像不是纯二值化，使用阈值处理
            if len(np.unique(img)) > 2:
                _, img_binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
            else:
                img_binary = (img > 0).astype(np.float32)
            
            batch_masks.append(img_binary)
            batch_names.append(img_file.stem)
        
        if len(batch_masks) == 0:
            continue
        
        # 获取图像尺寸（假设所有图像尺寸相同）
        H, W = batch_masks[0].shape
        
        # 转换为 torch tensor 并移动到设备
        masks_tensor = torch.stack([torch.from_numpy(m).float() for m in batch_masks])
        masks_tensor = masks_tensor.to(device)
        
        # 计算中心线
        y_norm, dy_norm = estimator.compute(masks_tensor, ref_pos=ref_pos, return_cpu=True)
        
        # 逐个结果处理
        for i, name in enumerate(batch_names):
            y_center_norm = float(y_norm[i])
            dy_center_norm = float(dy_norm[i])
            y_center_pixel = int(y_center_norm * (H - 1))
            ref_pixel = int(ref_pos * (H - 1))
            
            result = {
                'filename': name,
                'y_center_norm': y_center_norm,
                'dy_norm': dy_center_norm,
                'y_center_pixel': y_center_pixel,
                'ref_pixel': ref_pixel,
                'image_height': H,
                'image_width': W,
            }
            all_results.append(result)
            
            # 保存可视化图像
            if save_visualization:
                mask_img = batch_masks[i]
                img_color = cv2.cvtColor((mask_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                
                # 绘制中心线（绿色）
                cv2.line(img_color, (0, y_center_pixel), (W - 1, y_center_pixel), (0, 255, 0), 2)
                # 绘制参考线（红色）
                cv2.line(img_color, (0, ref_pixel), (W - 1, ref_pixel), (0, 0, 255), 1)
                
                # 添加文本信息
                text = f"y_norm={y_center_norm:.3f}, dy={dy_center_norm:.3f}"
                cv2.putText(img_color, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                
                vis_path = output_path / "visualizations" / f"{name}_vis.png"
                cv2.imwrite(str(vis_path), img_color)
        
        print(f"已处理 {min(batch_start + batch_size, len(image_files))}/{len(image_files)} 个图像")
    
    # 保存 CSV 结果
    if save_results_csv and len(all_results) > 0:
        import csv
        csv_path = output_path / "results.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"结果已保存到 {csv_path}")
    
    print(f"处理完成！共处理 {len(all_results)} 个图像")
    print(f"结果保存在: {output_path}")
    
    return {
        'results': all_results,
        'total_images': len(all_results),
        'output_dir': str(output_path),
    }


if __name__ == "__main__":
    # B, H, W = 4, 480, 640
    # masks = gen_complex_mask_batch(B, H, W).cuda()

    # est = MaskCenterEstimator(kernel_size=5, use_bbox=True, use_ema=True, ema_alpha=0.6)
    # y_norm, dy_norm = est.compute(masks, ref_pos=0.5)

    # print("y_norm:", y_norm)
    # print("dy_norm:", dy_norm)

    # # 可视化
    # masks_cpu = masks.cpu().numpy()
    # y_cpu = y_norm.detach().cpu().numpy()

    # for i in range(B):
    #     img = (masks_cpu[i] * 255).astype(np.uint8)
    #     img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     cy = int(y_cpu[i] * (H - 1))
    #     ref = int(0.5 * (H - 1))
    #     cv2.line(img_color, (0, cy), (W - 1, cy), (0, 255, 0), 2)
    #     cv2.line(img_color, (0, ref), (W - 1, ref), (0, 0, 255), 1)
    #     cv2.imshow(f"mask_{i}", img_color)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()\
    process_masks_from_directory()