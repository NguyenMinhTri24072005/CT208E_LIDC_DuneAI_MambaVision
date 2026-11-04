"""
losses.py
---------
Định nghĩa các hàm loss cho bài toán phân đoạn ảnh y tế nhị phân.

Hỗ trợ:
    - DiceLoss
    - FocalLoss
    - DiceBCELoss
    - build_loss(name, config): factory function tạo instance loss từ config dict
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# DiceLoss
# =============================================================================
class DiceLoss(nn.Module):
    """
    Dice Loss cho bài toán phân đoạn nhị phân.

    Công thức:
        DiceLoss = 1 - (2 * intersection + smooth) / (pred_sum + target_sum + smooth)

    Tham số:
        smooth (float): Hệ số làm mịn tránh chia cho 0. Mặc định: 1e-6.
    """

    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Tính Dice Loss.

        Args:
            pred   (Tensor): Đầu ra sau Sigmoid, shape (B, 1, H, W), giá trị trong [0, 1].
            target (Tensor): Nhãn nhị phân,        shape (B, 1, H, W), giá trị {0, 1}.

        Returns:
            Tensor: Giá trị loss vô hướng.
        """
        # Làm phẳng theo chiều không gian để tính toán
        pred_flat   = pred.view(pred.size(0), -1)       # (B, H*W)
        target_flat = target.view(target.size(0), -1)   # (B, H*W)

        # Tính intersection và tổng
        intersection = (pred_flat * target_flat).sum(dim=1)          # (B,)
        pred_sum     = pred_flat.sum(dim=1)                          # (B,)
        target_sum   = target_flat.sum(dim=1)                        # (B,)

        # Tính Dice coefficient rồi chuyển sang loss
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        return (1.0 - dice).mean()


# =============================================================================
# FocalLoss
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss cho bài toán phân đoạn nhị phân.

    Focal Loss tập trung vào các pixel khó phân loại bằng cách giảm
    trọng số của các pixel dễ phân loại (xác suất cao).

    Công thức:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Tham số:
        alpha (float): Hệ số cân bằng lớp. Mặc định: 0.25.
        gamma (float): Hệ số điều chỉnh độ khó. Mặc định: 2.0.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Tính Focal Loss.

        Args:
            pred   (Tensor): Đầu ra sau Sigmoid, shape (B, 1, H, W), giá trị trong [0, 1].
            target (Tensor): Nhãn nhị phân,        shape (B, 1, H, W), giá trị {0, 1}.

        Returns:
            Tensor: Giá trị loss vô hướng.
        """
        # Tính Binary Cross Entropy theo từng phần tử (không reduction)
        bce = F.binary_cross_entropy(pred, target, reduction='none')

        # Tính p_t: xác suất dự đoán đúng nhãn thực
        # Với target=1: p_t = pred; với target=0: p_t = 1 - pred
        p_t = pred * target + (1.0 - pred) * (1.0 - target)

        # Tính alpha_t: hệ số cân bằng theo nhãn thực
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)

        # Tính Focal Loss: giảm trọng số pixel dễ phân loại
        focal_loss = alpha_t * (1.0 - p_t) ** self.gamma * bce

        return focal_loss.mean()


# =============================================================================
# DiceBCELoss
# =============================================================================
class DiceBCELoss(nn.Module):
    """
    Kết hợp Dice Loss và Binary Cross Entropy Loss với trọng số bằng nhau.

    Công thức:
        DiceBCELoss = 0.5 * DiceLoss + 0.5 * BCELoss

    Tham số:
        smooth (float): Hệ số làm mịn cho DiceLoss. Mặc định: 1e-6.
    """

    def __init__(self, smooth: float = 1e-6):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss  = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Tính DiceBCE Loss.

        Args:
            pred   (Tensor): Đầu ra sau Sigmoid, shape (B, 1, H, W), giá trị trong [0, 1].
            target (Tensor): Nhãn nhị phân,        shape (B, 1, H, W), giá trị {0, 1}.

        Returns:
            Tensor: Giá trị loss vô hướng.
        """
        dice = self.dice_loss(pred, target)
        bce  = self.bce_loss(pred, target)

        # Kết hợp hai loss với trọng số bằng nhau (0.5 + 0.5 = 1.0)
        return 0.5 * dice + 0.5 * bce


# =============================================================================
# Factory function
# =============================================================================

# Bảng ánh xạ tên loss -> class tương ứng
_LOSS_REGISTRY: dict = {
    "dice":     DiceLoss,
    "focal":    FocalLoss,
    "dice_bce": DiceBCELoss,
}


def build_loss(name: str, config: dict = None) -> nn.Module:
    """
    Factory function tạo instance loss từ tên và config dict.

    Args:
        name   (str):  Tên hàm loss. Các giá trị hợp lệ: 'dice', 'focal', 'dice_bce',
                       'DiceLoss', 'FocalLoss', 'DiceBCELoss' (không phân biệt hoa thường).
        config (dict): Dict cấu hình chung (ví dụ config['TRAIN']).
                       Hàm này tự trích xuất các tham số phù hợp cho từng loss.
                       Nếu None, dùng giá trị mặc định của class.

    Returns:
        nn.Module: Instance của class loss tương ứng.

    Raises:
        ValueError: Nếu tên loss không được hỗ trợ.

    Ví dụ:
        >>> loss_fn = build_loss("DiceBCELoss", config['TRAIN'])
        >>> loss_fn = build_loss("dice")
    """
    # Chuẩn hóa tên: hỗ trợ cả "DiceBCELoss" lẫn "dice_bce"
    _name_map = {
        "diceloss": "dice",
        "focalloss": "focal",
        "dicebceloss": "dice_bce",
    }
    normalized = name.lower().strip().replace("_", "")
    name_key = _name_map.get(normalized, name.lower().strip())

    if name_key not in _LOSS_REGISTRY:
        raise ValueError(
            f"Loss '{name}' không được hỗ trợ. "
            f"Các loss hợp lệ: {list(_LOSS_REGISTRY.keys())}"
        )

    # Trích xuất tham số phù hợp cho từng loss class từ config
    config = config or {}
    if name_key == "dice":
        kwargs = {"smooth": config.get("smooth", 1e-6)}
    elif name_key == "focal":
        kwargs = {
            "alpha": config.get("focal_alpha", 0.25),
            "gamma": config.get("focal_gamma", 2.0),
        }
    elif name_key == "dice_bce":
        kwargs = {"smooth": config.get("smooth", 1e-6)}
    else:
        kwargs = {}

    return _LOSS_REGISTRY[name_key](**kwargs)
