# -*- coding: utf-8 -*-
"""
common/metrics.py - Chỉ số đánh giá phân đoạn ảnh y tế dùng chung (thước đo chuẩn).

Chứa hàm compute_segmentation_metrics được sao chép nguyên văn từ mã nguồn gốc,
đảm bảo cả hai kiến trúc unet_base và mamba_unet được so sánh khách quan trên cùng thang đo.
TUYỆT ĐỐI KHÔNG chứa lớp nn.Module nào (R11).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Dict

# Đoạn bootstrap bắt buộc ở đầu mọi file có __main__ (R5, mục 3.2)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch


@torch.no_grad()
def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Tính toán các chỉ số phân đoạn cơ bản từ raw logits và targets nhị phân.
    Trả về dict chứa intersection, pred_sum, target_sum, dice, và iou.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.float32)
    targets_bin = (targets > 0.5).to(torch.float32)

    preds_f = preds.reshape(-1)
    targets_f = targets_bin.reshape(-1)

    inter = float((preds_f * targets_f).sum().item())
    pred_sum = float(preds_f.sum().item())
    target_sum = float(targets_f.sum().item())
    union = pred_sum + target_sum - inter

    dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
    iou = (inter + eps) / (union + eps)

    return {
        "intersection": inter,
        "pred_sum": pred_sum,
        "target_sum": target_sum,
        "dice": dice,
        "iou": iou,
    }


if __name__ == "__main__":
    print("🔬 Đang chạy selftest common.metrics...")

    # Ca 1: Khớp hoàn toàn (Perfect match)
    logits_perfect = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
    targets_perfect = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    m1 = compute_segmentation_metrics(logits_perfect, targets_perfect, threshold=0.5)
    assert abs(m1["dice"] - 1.0) < 1e-4, f"Lỗi dice tại Ca 1: {m1}"
    assert abs(m1["iou"] - 1.0) < 1e-4, f"Lỗi iou tại Ca 1: {m1}"
    print(f"  + Ca 1 (Perfect match): Dice={m1['dice']:.4f}, IoU={m1['iou']:.4f} - OK")

    # Ca 2: Sai lệch hoàn toàn (Total mismatch)
    logits_mismatch = torch.tensor([[[[-10.0, -10.0], [10.0, 10.0]]]])
    targets_mismatch = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    m2 = compute_segmentation_metrics(logits_mismatch, targets_mismatch, threshold=0.5)
    assert m2["intersection"] == 0.0, f"Lỗi intersection tại Ca 2: {m2}"
    assert m2["dice"] < 1e-4, f"Lỗi dice tại Ca 2: {m2}"
    print(f"  + Ca 2 (Total mismatch): Dice={m2['dice']:.4f}, IoU={m2['iou']:.4f} - OK")

    # Ca 3: Trùng khớp một phần (Partial overlap: 1 pixel đúng / 2 pixels nhãn)
    logits_partial = torch.tensor([[[[10.0, -10.0], [-10.0, -10.0]]]])
    targets_partial = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    m3 = compute_segmentation_metrics(logits_partial, targets_partial, threshold=0.5)
    # inter=1, pred_sum=1, target_sum=2 -> dice = 2*1 / (1+2) = 2/3 ≈ 0.6667
    assert abs(m3["dice"] - (2.0 / 3.0)) < 1e-3, f"Lỗi dice tại Ca 3: {m3}"
    print(f"  + Ca 3 (Partial overlap): Dice={m3['dice']:.4f}, IoU={m3['iou']:.4f} - OK")

    # Ca 4: Kiểm tra đầy đủ các trường thông tin trả về
    expected_keys = {"intersection", "pred_sum", "target_sum", "dice", "iou"}
    assert set(m3.keys()) == expected_keys, f"Thiếu trường dữ liệu: {set(m3.keys())}"
    print(f"  + Ca 4 (Keys check): {list(m3.keys())} - OK")

    print("✅ Selftest common.metrics: HOÀN TẤT XUẤT SẮC!")
