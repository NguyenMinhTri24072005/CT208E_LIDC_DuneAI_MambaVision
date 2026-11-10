# -*- coding: utf-8 -*-
"""
mamba_unet/losses.py - Registry và các hàm mất mát (Loss Functions) riêng của mamba_unet.

Quy tắc bất biến (R1, R11):
  - File tự đủ: CHỈ import torch và thư viện chuẩn, không import bất kỳ file nào khác của dự án.
  - Tuyệt đối KHÔNG chứa metrics (metrics nằm tại common/metrics.py theo R11).
  - Ép logits.float() trong forward để an toàn tuyệt đối khi bật AMP.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. CÁC LỚP LOSS RIÊNG CỦA MAMBA_UNET
# =============================================================================
class PureDiceLoss(nn.Module):
    """Pure Soft Dice Loss cho phân đoạn ảnh y tế."""
    def __init__(self, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.smooth = float(smooth)
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.to(dtype=pred.dtype)
        probs = torch.sigmoid(pred) if self.from_logits else pred

        probs_flat = probs.reshape(-1)
        target_flat = target.reshape(-1)

        intersection = (probs_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice


class PureTverskyLoss(nn.Module):
    """Tversky Loss phạt nặng bỏ sót (False Negatives)."""
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.to(dtype=pred.dtype)
        probs = torch.sigmoid(pred) if self.from_logits else pred

        p_flat = probs.reshape(-1)
        t_flat = target.reshape(-1)

        tp = (p_flat * t_flat).sum()
        fp = (p_flat * (1.0 - t_flat)).sum()
        fn = ((1.0 - p_flat) * t_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class PureTverskyFocalLoss(nn.Module):
    """Focal Tversky Loss: Lũy thừa hệ số gamma cho khối u nhỏ."""
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.to(dtype=pred.dtype)
        probs = torch.sigmoid(pred) if self.from_logits else pred

        p_flat = probs.reshape(-1)
        t_flat = target.reshape(-1)

        tp = (p_flat * t_flat).sum()
        fp = (p_flat * (1.0 - t_flat)).sum()
        fn = ((1.0 - p_flat) * t_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma)


class PureFocalLoss(nn.Module):
    """Binary Focal Loss."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, from_logits: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.to(dtype=pred.dtype)
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            probs = torch.sigmoid(pred)
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
            probs = pred

        p_t = probs * target + (1.0 - probs) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_loss = alpha_t * torch.pow(1.0 - p_t, self.gamma) * bce
        return focal_loss.mean()


class PureDiceBCELoss(nn.Module):
    """Hàm mất mát chuẩn kết hợp BCE và Soft Dice Loss."""
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
        smooth: float = 1.0,
        batch_dice: bool = True,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.batch_dice = bool(batch_dice)
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)

        probs = torch.sigmoid(logits)
        if self.batch_dice:
            probs_flat = probs.reshape(-1)
            targets_flat = targets.reshape(-1)
            intersection = (probs_flat * targets_flat).sum()
            dice_score = (2.0 * intersection + self.smooth) / (
                probs_flat.sum() + targets_flat.sum() + self.smooth
            )
        else:
            probs_flat = probs.reshape(probs.size(0), -1)
            targets_flat = targets.reshape(targets.size(0), -1)
            intersection = (probs_flat * targets_flat).sum(dim=1)
            dice_score = (2.0 * intersection + self.smooth) / (
                probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
            )
            dice_score = dice_score.mean()

        dice_loss = 1.0 - dice_score
        return self.bce_weight * bce + self.dice_weight * dice_loss


class PureBCELoss(nn.Module):
    """BCEWithLogitsLoss đơn thuần ép float32 an toàn AMP."""
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


# =============================================================================
# 2. BẢNG ĐĂNG KÝ LOSS (REGISTRY)
# =============================================================================
@dataclass
class LossSpec:
    factory: Callable[..., nn.Module]
    doc: str


LOSSES: Dict[str, LossSpec] = {
    "dice_bce": LossSpec(
        factory=lambda bce_weight=0.5, dice_weight=0.5, pos_weight=None, smooth=1.0, batch_dice=True, **kw: PureDiceBCELoss(
            bce_weight=bce_weight, dice_weight=dice_weight, pos_weight=pos_weight, smooth=smooth, batch_dice=batch_dice
        ),
        doc="Kết hợp BCEWithLogitsLoss và Soft Dice Loss (mặc định)",
    ),
    "dice": LossSpec(
        factory=lambda smooth=1e-6, from_logits=True, **kw: PureDiceLoss(smooth=smooth, from_logits=from_logits),
        doc="Soft Dice Loss thuần túy",
    ),
    "bce": LossSpec(
        factory=lambda pos_weight=None, **kw: PureBCELoss(pos_weight=pos_weight),
        doc="Binary Cross Entropy with Logits",
    ),
    "tversky": LossSpec(
        factory=lambda alpha=0.3, beta=0.7, smooth=1e-6, from_logits=True, **kw: PureTverskyLoss(
            alpha=alpha, beta=beta, smooth=smooth, from_logits=from_logits
        ),
        doc="Tversky Loss phạt nặng bỏ sót (beta > alpha)",
    ),
    "focal_tversky": LossSpec(
        factory=lambda alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0, from_logits=True, **kw: PureTverskyFocalLoss(
            alpha=alpha, beta=beta, gamma=gamma, smooth=smooth, from_logits=from_logits
        ),
        doc="Tversky Focal Loss cho khối u nhỏ và lát cắt khó",
    ),
    "focal": LossSpec(
        factory=lambda alpha=0.25, gamma=2.0, from_logits=True, **kw: PureFocalLoss(
            alpha=alpha, gamma=gamma, from_logits=from_logits
        ),
        doc="Binary Focal Loss",
    ),
}

DEFAULT_LOSS: str = "dice_bce"
DEFAULT_LOSS_PARAMS: Dict[str, float] = {
    "bce_weight": 0.5,
    "dice_weight": 0.5,
    "smooth": 1.0,
    "batch_dice": True,
}


def get_loss(name: Optional[str] = None, **params) -> nn.Module:
    """Khởi tạo instance hàm loss từ tên và tham số cho mamba_unet."""
    if name is None:
        name = DEFAULT_LOSS

    key = name.lower().strip().replace("-", "_")
    if key in ("diceloss",):
        key = "dice"
    elif key in ("dicebceloss",):
        key = "dice_bce"
    elif key in ("tverskyloss",):
        key = "tversky"
    elif key in ("tversky_focal", "tverskyfocalloss", "focaltversky"):
        key = "focal_tversky"
    elif key in ("focalloss",):
        key = "focal"
    elif key in ("bceloss", "bce_with_logits"):
        key = "bce"

    if key not in LOSSES:
        valid_keys = list(LOSSES.keys())
        raise ValueError(
            f"❌ Không hỗ trợ loss '{name}' trong mamba_unet!\n"
            f"  Các lựa chọn hợp lệ: {valid_keys}"
        )

    spec = LOSSES[key]
    return spec.factory(**params)


def list_losses() -> List[str]:
    """Liệt kê danh sách tên các hàm loss hợp lệ trong mamba_unet."""
    return list(LOSSES.keys())


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.losses...")
    logits = torch.randn(4, 1, 32, 32)
    targets = torch.randint(0, 2, (4, 1, 32, 32)).float()

    for loss_name in list_losses():
        loss_fn = get_loss(loss_name)
        val = loss_fn(logits, targets)
        assert isinstance(val, torch.Tensor), f"Loss {loss_name} không trả về Tensor!"
        assert val.ndim == 0, f"Loss {loss_name} phải là scalar, got shape {val.shape}"
        assert not torch.isnan(val), f"Loss {loss_name} bị NaN!"
        assert not torch.isinf(val), f"Loss {loss_name} bị Inf!"
        print(f"  ✓ {loss_name:15s}: {val.item():.4f}")

    print(f"✅ Đã kiểm thử thành công {len(LOSSES)} hàm loss trong mamba_unet.losses!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Quản lý và kiểm thử Loss Functions của mamba_unet")
    parser.add_argument("--list", action="store_true", help="In danh sách các loss hợp lệ")
    args = parser.parse_args()

    if args.list:
        print("\n".join(list_losses()))
    else:
        _selftest()
