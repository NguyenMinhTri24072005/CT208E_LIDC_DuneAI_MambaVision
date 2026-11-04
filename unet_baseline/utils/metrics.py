"""
metrics.py
----------
Các chỉ số đánh giá (Evaluation Metrics) cho phân đoạn ảnh:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU / Jaccard Index)
- 95th Percentile Hausdorff Distance (HD95)
- Class MetricTracker hỗ trợ tính toán và theo dõi theo từng ảnh riêng lẻ
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist


def calculate_dice(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Tính Dice Coefficient giữa 2 mảng nhị phân nhãn {0, 1}.
    Args:
        pred: Mảng nhị phân dự đoán (H, W)
        target: Mảng nhị phân ground truth (H, W)
    Returns:
        Điểm số Dice [0.0, 1.0]
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    pred_sum = np.sum(pred)
    target_sum = np.sum(target)

    # Trường hợp cả hai cùng rỗng (không có khối u và dự đoán đúng không có khối u)
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    # Một bên có, một bên không
    if pred_sum == 0 or target_sum == 0:
        return 0.0

    intersection = np.sum(pred & target)
    return float((2.0 * intersection) / (pred_sum + target_sum + smooth))


def calculate_iou(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Tính IoU (Jaccard Index) giữa 2 mảng nhị phân.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    pred_sum = np.sum(pred)
    target_sum = np.sum(target)

    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0

    intersection = np.sum(pred & target)
    union = np.sum(pred | target)
    return float((intersection) / (union + smooth))


def _get_surface_points(mask: np.ndarray) -> np.ndarray:
    """Trích xuất tọa độ các pixel đường viền (surface boundary) của mask."""
    if mask.sum() == 0:
        return np.empty((0, 2))
    # Đường viền = Mask trừ đi phần xói mòn (erosion)
    eroded = binary_erosion(mask)
    border = mask ^ eroded
    return np.argwhere(border)


def calculate_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Tính Hausdorff Distance percentile 95 (HD95).
    Nếu pred hoặc target rỗng hoàn toàn, trả về np.nan theo quy ước.
    """
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    # Cả hai cùng rỗng
    if pred_bool.sum() == 0 and target_bool.sum() == 0:
        return 0.0

    # Một trong hai rỗng -> Không thể đo khoảng cách biên
    if pred_bool.sum() == 0 or target_bool.sum() == 0:
        return float("nan")

    pred_pts = _get_surface_points(pred_bool)
    target_pts = _get_surface_points(target_bool)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float("nan")

    # Khoảng cách euclid 2 chiều
    dists_pred_to_target = cdist(pred_pts, target_pts).min(axis=1)
    dists_target_to_pred = cdist(target_pts, pred_pts).min(axis=1)

    hd95 = max(
        np.percentile(dists_pred_to_target, 95),
        np.percentile(dists_target_to_pred, 95),
    )
    return float(hd95)


class MetricTracker:
    """
    Lớp lưu trữ và theo dõi các chỉ số đánh giá cho từng ảnh trong batch/dataset.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.dices: List[float] = []
        self.ious: List[float] = []
        self.hd95s: List[float] = []

    def update(
        self,
        dice: float,
        iou: float,
        hd95: Optional[float] = None,
    ) -> None:
        self.dices.append(dice)
        self.ious.append(iou)
        if hd95 is not None and not np.isnan(hd95):
            self.hd95s.append(hd95)

    def summary(self) -> Dict[str, Tuple[float, float]]:
        """
        Trả về dict {metric: (mean, std)}.
        """
        result = {}
        result["dice"] = (
            float(np.mean(self.dices)) if self.dices else 0.0,
            float(np.std(self.dices)) if self.dices else 0.0,
        )
        result["iou"] = (
            float(np.mean(self.ious)) if self.ious else 0.0,
            float(np.std(self.ious)) if self.ious else 0.0,
        )
        result["hd95"] = (
            float(np.mean(self.hd95s)) if self.hd95s else float("nan"),
            float(np.std(self.hd95s)) if self.hd95s else float("nan"),
        )
        return result
