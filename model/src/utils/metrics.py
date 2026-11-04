"""
metrics.py
----------
Định nghĩa các hàm đánh giá (metric) cho bài toán phân đoạn ảnh y tế nhị phân.

Hỗ trợ:
    - dice_score      : Dice Similarity Coefficient
    - iou_score       : Intersection over Union (Jaccard Index)
    - precision_score : Precision (Độ chính xác)
    - recall_score    : Recall (Độ nhạy / Sensitivity)
    - MetricTracker   : Class tổng hợp theo dõi metrics qua nhiều batch / epoch

Lưu ý:
    Tất cả hàm làm việc trực tiếp với PyTorch Tensor, không phụ thuộc sklearn.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch


# =============================================================================
# Hàm tiện ích nội bộ
# =============================================================================

def _threshold(pred: torch.Tensor, threshold: float) -> torch.Tensor:
    """Chuyển tensor xác suất thành nhãn nhị phân 0/1 theo ngưỡng."""
    return (pred >= threshold).float()


def _flatten(tensor: torch.Tensor) -> torch.Tensor:
    """Làm phẳng tensor về dạng (B, N) để tính toán theo batch."""
    return tensor.view(tensor.size(0), -1)


# =============================================================================
# dice_score
# =============================================================================

def dice_score(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    threshold: float = 0.5,
    smooth:    float = 1e-6,
) -> float:
    """
    Tính Dice Similarity Coefficient (DSC).

    Công thức:
        DSC = (2 * |pred ∩ target| + smooth) / (|pred| + |target| + smooth)

    Args:
        pred      (Tensor): Dự đoán sau Sigmoid, shape (B, 1, H, W), giá trị [0, 1].
        target    (Tensor): Nhãn nhị phân,        shape (B, 1, H, W), giá trị {0, 1}.
        threshold (float) : Ngưỡng chuyển pred thành nhị phân. Mặc định: 0.5.
        smooth    (float) : Hệ số làm mịn tránh chia cho 0.   Mặc định: 1e-6.

    Returns:
        float: Giá trị Dice trung bình theo batch.
    """
    # Áp dụng ngưỡng để chuyển sang nhị phân
    pred_bin = _threshold(pred, threshold)

    # Làm phẳng về (B, N)
    pred_flat   = _flatten(pred_bin)
    target_flat = _flatten(target)

    # Tính intersection và tổng
    intersection = (pred_flat * target_flat).sum(dim=1)   # (B,)
    pred_sum     = pred_flat.sum(dim=1)                   # (B,)
    target_sum   = target_flat.sum(dim=1)                 # (B,)

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    return dice.mean().item()


# =============================================================================
# iou_score
# =============================================================================

def iou_score(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    threshold: float = 0.5,
    smooth:    float = 1e-6,
) -> float:
    """
    Tính Intersection over Union (IoU / Jaccard Index).

    Công thức:
        IoU = (|pred ∩ target| + smooth) / (|pred ∪ target| + smooth)
            = (intersection + smooth) / (pred_sum + target_sum - intersection + smooth)

    Args:
        pred      (Tensor): Dự đoán sau Sigmoid, shape (B, 1, H, W), giá trị [0, 1].
        target    (Tensor): Nhãn nhị phân,        shape (B, 1, H, W), giá trị {0, 1}.
        threshold (float) : Ngưỡng chuyển pred thành nhị phân. Mặc định: 0.5.
        smooth    (float) : Hệ số làm mịn tránh chia cho 0.   Mặc định: 1e-6.

    Returns:
        float: Giá trị IoU trung bình theo batch.
    """
    # Áp dụng ngưỡng
    pred_bin = _threshold(pred, threshold)

    # Làm phẳng
    pred_flat   = _flatten(pred_bin)
    target_flat = _flatten(target)

    # Tính intersection và union
    intersection = (pred_flat * target_flat).sum(dim=1)                        # (B,)
    union        = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection  # (B,)

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()


# =============================================================================
# precision_score
# =============================================================================

def precision_score(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    threshold: float = 0.5,
    smooth:    float = 1e-6,
) -> float:
    """
    Tính Precision (Độ chính xác dương tính).

    Công thức:
        Precision = TP / (TP + FP + smooth)

    Args:
        pred      (Tensor): Dự đoán sau Sigmoid, shape (B, 1, H, W), giá trị [0, 1].
        target    (Tensor): Nhãn nhị phân,        shape (B, 1, H, W), giá trị {0, 1}.
        threshold (float) : Ngưỡng chuyển pred thành nhị phân. Mặc định: 0.5.
        smooth    (float) : Hệ số làm mịn tránh chia cho 0.   Mặc định: 1e-6.

    Returns:
        float: Giá trị Precision trung bình theo batch.
    """
    # Áp dụng ngưỡng
    pred_bin = _threshold(pred, threshold)

    # Làm phẳng
    pred_flat   = _flatten(pred_bin)
    target_flat = _flatten(target)

    # True Positive: dự đoán 1 và nhãn 1
    tp = (pred_flat * target_flat).sum(dim=1)           # (B,)

    # False Positive: dự đoán 1 nhưng nhãn 0
    fp = (pred_flat * (1.0 - target_flat)).sum(dim=1)   # (B,)

    precision = (tp + smooth) / (tp + fp + smooth)

    return precision.mean().item()


# =============================================================================
# recall_score
# =============================================================================

def recall_score(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    threshold: float = 0.5,
    smooth:    float = 1e-6,
) -> float:
    """
    Tính Recall (Độ nhạy / Sensitivity).

    Công thức:
        Recall = TP / (TP + FN + smooth)

    Args:
        pred      (Tensor): Dự đoán sau Sigmoid, shape (B, 1, H, W), giá trị [0, 1].
        target    (Tensor): Nhãn nhị phân,        shape (B, 1, H, W), giá trị {0, 1}.
        threshold (float) : Ngưỡng chuyển pred thành nhị phân. Mặc định: 0.5.
        smooth    (float) : Hệ số làm mịn tránh chia cho 0.   Mặc định: 1e-6.

    Returns:
        float: Giá trị Recall trung bình theo batch.
    """
    # Áp dụng ngưỡng
    pred_bin = _threshold(pred, threshold)

    # Làm phẳng
    pred_flat   = _flatten(pred_bin)
    target_flat = _flatten(target)

    # True Positive: dự đoán 1 và nhãn 1
    tp = (pred_flat * target_flat).sum(dim=1)           # (B,)

    # False Negative: dự đoán 0 nhưng nhãn 1
    fn = ((1.0 - pred_flat) * target_flat).sum(dim=1)   # (B,)

    recall = (tp + smooth) / (tp + fn + smooth)

    return recall.mean().item()

def dice_score_positive(pred, target, threshold=0.5, smooth=1e-6):
    """Dice chỉ tính trên các mẫu có nốt. Trả None nếu batch không có mẫu dương."""
    pred_flat   = _flatten(_threshold(pred, threshold))
    target_flat = _flatten(target)
    has_pos = target_flat.sum(dim=1) > 0
    if not has_pos.any():
        return None
    inter = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * inter + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return dice[has_pos].mean().item()



# =============================================================================
# MetricTracker
# =============================================================================

class MetricTracker:
    """
    Class tổng hợp để theo dõi và tích lũy nhiều metric qua nhiều batch / epoch.

    Cách dùng:
        tracker = MetricTracker()

        # Trong vòng lặp batch
        for pred, target in dataloader:
            tracker.update(pred, target)

        # Lấy kết quả trung bình của cả epoch
        results = tracker.mean()
        print(results)  # {"dice": 0.85, "iou": 0.76, ...}

        # Reset trước epoch tiếp theo
        tracker.reset()

    Tham số:
        threshold (float): Ngưỡng nhị phân hoá dự đoán. Mặc định: 0.5.
        smooth    (float): Hệ số làm mịn cho các metric.  Mặc định: 1e-6.
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth    = smooth

        # Lưu lịch sử giá trị từng metric theo batch
        self._history: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Tính toán tất cả metric cho một batch và thêm vào lịch sử.

        Args:
            pred   (Tensor): Dự đoán sau Sigmoid, shape (B, 1, H, W).
            target (Tensor): Nhãn nhị phân,        shape (B, 1, H, W).

        Returns:
            Dict[str, float]: Giá trị metric của batch hiện tại.
        """
        batch_metrics = {
            "dice":      dice_score(pred,      target, self.threshold, self.smooth),
            "iou":       iou_score(pred,       target, self.threshold, self.smooth),
            "precision": precision_score(pred, target, self.threshold, self.smooth),
            "recall":    recall_score(pred,    target, self.threshold, self.smooth),
        }
        
        dp = dice_score_positive(pred, target, self.threshold, self.smooth)
        if dp is not None:
            batch_metrics["dice_pos"] = dp

        # Lưu vào lịch sử
        for name, value in batch_metrics.items():
            self._history[name].append(value)

        return batch_metrics

    # ------------------------------------------------------------------
    def mean(self) -> Dict[str, float]:
        """
        Tính giá trị trung bình của tất cả metric trong lịch sử hiện tại.

        Returns:
            Dict[str, float]: Trung bình từng metric kể từ lần reset() gần nhất.
        """
        if not self._history:
            return {}

        return {
            name: sum(values) / len(values)
            for name, values in self._history.items()
        }

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        Xoá toàn bộ lịch sử metric, thường gọi vào đầu mỗi epoch.
        """
        self._history.clear()

    # ------------------------------------------------------------------
    def history(self) -> Dict[str, List[float]]:
        """
        Trả về toàn bộ lịch sử metric theo từng batch.

        Returns:
            Dict[str, List[float]]: Danh sách giá trị theo thứ tự batch.
        """
        return dict(self._history)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        means = self.mean()
        parts = ", ".join(f"{k}={v:.4f}" for k, v in means.items())
        return f"MetricTracker({parts})"
