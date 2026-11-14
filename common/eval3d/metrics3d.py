# -*- coding: utf-8 -*-
"""
common/eval3d/metrics3d.py - Các chỉ số đánh giá 3D cấp độ bệnh nhân (3D Patient-level Metrics).

Các chỉ số y tế chuẩn mực:
  - 3D Dice, 3D IoU, Precision, Recall.
  - Thể tích 3D khối u (mm³): pred_volume_mm3, gt_volume_mm3.
  - Khoảng cách bề mặt 95th percentile Hausdorff Distance (HD95, mm):
    Tính bằng khoảng cách Euclid thực trên lưới voxel có spacing dị hướng (spacing_z, 1.0, 1.0).
    Xử lý an toàn các ca biên (mask rỗng hoàn toàn).
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as ndi


def _compute_hd95_mm(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Tính 95th percentile Hausdorff Distance (HD95) theo đơn vị mm vật lý giữa 2 mặt nạ 3D.
    Xử lý an toàn:
      - Cả hai rỗng: 0.0 mm (hoàn hảo không có u và không phát hiện nhầm).
      - Một trong hai rỗng: float('nan').
    """
    pred_any = bool(pred_bin.any())
    gt_any = bool(gt_bin.any())
    if not pred_any and not gt_any:
        return 0.0
    if not pred_any or not gt_any:
        return float("nan")

    # Trích xuất biên bề mặt 3D bằng phép co (erosion)
    pred_border = pred_bin ^ ndi.binary_erosion(pred_bin)
    gt_border = gt_bin ^ ndi.binary_erosion(gt_bin)
    if not pred_border.any():
        pred_border = pred_bin
    if not gt_border.any():
        gt_border = gt_bin

    dt_gt = ndi.distance_transform_edt(~gt_border, sampling=spacing_zyx)
    dt_pred = ndi.distance_transform_edt(~pred_border, sampling=spacing_zyx)

    dists_pred_to_gt = dt_gt[pred_border]
    dists_gt_to_pred = dt_pred[gt_border]
    all_dists = np.concatenate([dists_pred_to_gt, dists_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def compute_binary_overlap_stats(
    tp: float,
    pred_sum: float,
    gt_sum: float,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Tính dice_3d, iou_3d, precision_3d, recall_3d trực tiếp từ các tổng TP/Pred/GT."""
    tp = float(tp)
    pred_sum = float(pred_sum)
    gt_sum = float(gt_sum)
    union = pred_sum + gt_sum - tp

    if pred_sum == 0 and gt_sum == 0:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
    elif tp == 0:
        dice = 0.0
        iou = 0.0
        precision = 0.0
        recall = 0.0
    else:
        dice = (2.0 * tp + eps) / (pred_sum + gt_sum + eps)
        iou = (tp + eps) / (union + eps)
        precision = (tp + eps) / (pred_sum + eps) if pred_sum > 0 else 0.0
        recall = (tp + eps) / (gt_sum + eps) if gt_sum > 0 else 0.0

    return {
        "dice_3d": float(dice),
        "iou_3d": float(iou),
        "precision_3d": float(precision),
        "recall_3d": float(recall),
    }


def compute_3d_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing_z: float = 1.0,
    compute_hd95: bool = True,
    eps: float = 1e-6,
) -> Dict[str, Union[float, int]]:
    """
    Tính bộ chỉ số phân vùng 3D trên toàn bộ thể tích phổi của bệnh nhân:
      - dice_3d, iou_3d, precision_3d, recall_3d
      - pred_volume_mm3, gt_volume_mm3
      - hd95_mm (Hausdorff Distance 95% tính bằng mm)
    """
    pred_b = np.asarray(pred_mask) > 0
    gt_b = np.asarray(gt_mask) > 0

    tp = float(np.logical_and(pred_b, gt_b).sum())
    pred_sum = float(pred_b.sum())
    gt_sum = float(gt_b.sum())
    overlap = compute_binary_overlap_stats(tp, pred_sum, gt_sum, eps=eps)

    hd95 = (
        _compute_hd95_mm(pred_b, gt_b, spacing_zyx=(float(spacing_z), 1.0, 1.0))
        if compute_hd95
        else float("nan")
    )

    return {
        **overlap,
        "pred_voxels": int(pred_sum),
        "gt_voxels": int(gt_sum),
        "pred_volume_mm3": round(float(pred_sum * spacing_z), 2),
        "gt_volume_mm3": round(float(gt_sum * spacing_z), 2),
        "hd95_mm": round(float(hd95), 3) if np.isfinite(hd95) else float("nan"),
    }


def _selftest():
    print("🧪 Bắt đầu selftest common.eval3d.metrics3d...")
    # 1. Trùng khớp hoàn hảo
    m1 = np.zeros((8, 32, 32), dtype=np.uint8)
    m1[2:6, 10:20, 10:20] = 1
    res1 = compute_3d_metrics(m1, m1, spacing_z=1.5)
    assert abs(res1["dice_3d"] - 1.0) < 1e-5
    assert abs(res1["iou_3d"] - 1.0) < 1e-5
    assert res1["hd95_mm"] == 0.0

    # 2. Cả hai cùng rỗng
    empty = np.zeros((8, 32, 32), dtype=np.uint8)
    res_empty = compute_3d_metrics(empty, empty)
    assert res_empty["dice_3d"] == 1.0
    assert res_empty["hd95_mm"] == 0.0

    # 3. Một bên rỗng
    res_half = compute_3d_metrics(m1, empty)
    assert res_half["dice_3d"] == 0.0
    assert np.isnan(res_half["hd95_mm"])

    # 4. Lệch vị trí có khoảng cách cụ thể
    m2 = np.zeros_like(m1)
    m2[2:6, 12:22, 10:20] = 1  # dịch 2 pixel XY
    res2 = compute_3d_metrics(m1, m2, spacing_z=1.0)
    assert 0.0 < res2["dice_3d"] < 1.0
    assert 1.0 <= res2["hd95_mm"] <= 3.0

    print("✅ Selftest common.eval3d.metrics3d hoàn tất thành công!")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
