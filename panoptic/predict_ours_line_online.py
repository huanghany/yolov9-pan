import cv2
import torch
import numpy as np

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.segment.general import process_mask, masks2segments
from utils.plots import colors
from arm_height_controller import MaskCenterEstimator


def draw_semantic_mask(im0, semantic_logits, stuff_class_id, alpha):
    """绘制语义分割掩码（Stuff 类别）"""
    if semantic_logits is None:
        return im0

    H, W = im0.shape[:2]
    semantic_pred = torch.argmax(semantic_logits, dim=0).cpu().numpy().astype(np.uint8)
    semantic_resized = cv2.resize(semantic_pred, (W, H), interpolation=cv2.INTER_NEAREST)

    stuff_mask = (semantic_resized == stuff_class_id)
    if np.any(stuff_mask):
        color_layer = np.zeros_like(im0, dtype=np.uint8)
        color_layer[stuff_mask] = colors(stuff_class_id, True)
        im0 = cv2.addWeighted(im0, 1.0, color_layer, alpha, 0)
    return im0


def draw_instance_masks(im0, det, mask_proto, names, alpha=0.5):
    """绘制实例掩码（Thing 类别）+ 边界框"""
    if not len(det):
        return im0

    masks = process_mask(mask_proto, det[:, 6:], det[:, :4], im0.shape[:2], upsample=True)
    masks_np = masks.permute(1, 2, 0).cpu().numpy()

    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
        c = int(cls)
        mask_bool = masks_np[:, :, j] > 0.5
        color_layer = np.zeros_like(im0, dtype=np.uint8)
        color_layer[mask_bool] = colors(c, True)
        im0 = cv2.addWeighted(im0, 1.0, color_layer, alpha, 0)
        cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), colors(c, True), 2)
        label = f'{names[c]} {conf:.2f}'
        cv2.putText(im0, label, (int(xyxy[0]), int(xyxy[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return im0


def remove_edge_noise(mask_binary, crop_ratio=0.02):
    """去除左右边缘噪声"""
    H, W = mask_binary.shape
    crop_px = int(W * crop_ratio)
    mask_binary[:, :crop_px] = 0
    mask_binary[:, -(crop_px):] = 0
    return mask_binary


def run_camera(weights, stuff_class_id=7, ref_pos=0.5, alpha=0.5, device=''):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = (640, 640)

    center_estimator = MaskCenterEstimator(kernel_size=5, use_bbox=False, use_ema=True)

    cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        im = cv2.resize(frame, imgsz)
        im = torch.from_numpy(im).to(model.device).float()
        im /= 255
        im = im.permute(2, 0, 1).unsqueeze(0)  # [B,3,H,W]

        pred, panoptic_outs = model(im)
        mask_proto, semantic_logits = panoptic_outs[2], panoptic_outs[3]
        det = non_max_suppression(pred, 0.25, 0.45, nm=32)[0]

        # 绘制语义分割
        frame = draw_semantic_mask(frame, semantic_logits[0], stuff_class_id, alpha)
        # 绘制实例分割
        frame = draw_instance_masks(frame, det, mask_proto[0], names, alpha)

        # 绘制中心线
        H, W = frame.shape[:2]
        sem_pred = torch.argmax(semantic_logits[0], dim=0)
        sem_resized = cv2.resize(sem_pred.cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        mask_binary = (sem_resized == stuff_class_id).astype(np.float32)
        mask_binary = remove_edge_noise(mask_binary, crop_ratio=0.03)
        mask_tensor = torch.from_numpy(mask_binary).to(model.device)
        y_norm, dy_norm = center_estimator.compute(mask_tensor, ref_pos=ref_pos, return_cpu=True)

        y_center_px = int(y_norm[0] * (H - 1))
        ref_px = int(ref_pos * (H - 1))
        cv2.line(frame, (0, y_center_px), (W - 1, y_center_px), (0, 255, 0), 2)
        cv2.line(frame, (0, ref_px), (W - 1, ref_px), (0, 0, 255), 1)

        cv2.imshow("YOLOv9 Panoptic Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera(
        weights='/home/huanghanyang/Project/yolov9/runs/train-pan/strawberry_rack_v2/weights/yolov9-pan-strawberry7cls_rackcls1_v2.pt',
        stuff_class_id=7,
        ref_pos=0.5,
        alpha=0.5,
        device='cuda'
    )