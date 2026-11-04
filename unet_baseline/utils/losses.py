"""
losses.py
---------
Các hàm mất mát (Loss functions) cho bài toán Semantic Segmentation:
- DiceLoss (Hybrid Sample-wise và Batch-wise Dice)
- BCEWithLogitsLossWrapper (hỗ trợ pos_weight tùy chỉnh từ config)
- FocalLoss (chống mất cân bằng dữ liệu, alpha=0.5 mặc định)
- FocalTverskyLoss (tối ưu hóa Recall cho các tổn thương nốt u nhỏ)
- CombinedLoss (kết hợp tuyến tính các hàm loss theo trọng số)
- Factory function build_loss(cfg)
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Hybrid Dice Loss kết hợp Sample-wise Dice và Batch-wise Dice
    giúp ổn định gradient cho các nốt u phổi kích thước siêu nhỏ.
    """

    def __init__(self, smooth: float = 1.0, batch_dice_weight: float = 0.5) -> None:
        super().__init__()
        self.smooth = smooth
        self.batch_dice_weight = batch_dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()

        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)

            # 1. Sample-level Dice (tính trên từng ảnh H, W)
            inter_sample = (probs * targets).sum(dim=(2, 3))
            union_sample = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
            sample_dice = ((2.0 * inter_sample + self.smooth) / (union_sample + self.smooth)).mean()

            # 2. Batch-level Dice (tính gộp trên toàn batch B, C, H, W)
            inter_batch = (probs * targets).sum()
            union_batch = probs.sum() + targets.sum()
            batch_dice = (2.0 * inter_batch + self.smooth) / (union_batch + self.smooth)

            dice = (1.0 - self.batch_dice_weight) * sample_dice + self.batch_dice_weight * batch_dice
            return 1.0 - dice
        else:
            probs = F.softmax(logits, dim=1)
            num_classes = logits.shape[1]
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
            intersection = (probs * targets_one_hot).sum(dim=(2, 3))
            union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - dice.mean()


class BCEWithLogitsLossWrapper(nn.Module):
    """
    Wrapper cho BCEWithLogitsLoss hỗ trợ trọng số pos_weight linh hoạt từ cấu hình.
    """

    def __init__(self, pos_weight: Optional[float] = 1.0) -> None:
        super().__init__()
        self.pos_weight_val = pos_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if logits.shape[1] == 1:
            if self.pos_weight_val is not None and self.pos_weight_val != 1.0:
                pw = torch.tensor([self.pos_weight_val], device=logits.device, dtype=logits.dtype)
                return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pw)
            return F.binary_cross_entropy_with_logits(logits, targets.float())
        else:
            return self.ce(logits, targets.long())


class FocalLoss(nn.Module):
    """
    Focal Loss nhằm giảm trọng số của các mẫu dễ học và tập trung vào mẫu khó.
    Mặc định alpha=0.5 để cân bằng hoặc không phạt giảm lớp dương tính.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        if logits.shape[1] == 1:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            p = torch.sigmoid(logits)
            p_t = p * targets + (1.0 - p) * (1.0 - targets)
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

            focal_loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce_loss
            return focal_loss.mean()
        else:
            ce_loss = F.cross_entropy(logits, targets.long(), reduction="none")
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1.0 - p_t) ** self.gamma) * ce_loss
            return focal_loss.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss giúp tối ưu hóa Recall và giảm False Negatives cho các nốt u phổi nhỏ.
    alpha: trọng số phạt False Positives (mặc định 0.3)
    beta: trọng số phạt False Negatives (mặc định 0.7 để ưu tiên Recall)
    exponent: hệ số điều chế độ khó (mặc định 0.75)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        exponent: float = 0.75,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.exponent = exponent
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        targets = targets.float()

        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1.0 - targets)).sum(dim=(2, 3))
        fn = ((1.0 - probs) * targets).sum(dim=(2, 3))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1.0 - tversky) ** self.exponent
        return focal_tversky.mean()


class CombinedLoss(nn.Module):
    """
    Kết hợp có trọng số giữa BCE/CE, Dice Loss và Focal Loss.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.0,
        pos_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.bce = BCEWithLogitsLossWrapper(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=0.5) if focal_weight > 0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        loss = 0.0
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce(logits, targets)
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice(logits, targets)
        if self.focal_weight > 0 and self.focal is not None:
            loss += self.focal_weight * self.focal(logits, targets)
        return loss


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    """
    Factory function khởi tạo loss function từ cấu hình dict.
    """
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "CombinedLoss")
    pos_weight = float(loss_cfg.get("pos_weight", 1.0))
    smooth = float(loss_cfg.get("smooth", 1.0))

    if loss_type == "DiceLoss":
        return DiceLoss(smooth=smooth)
    elif loss_type in ["BCE", "BCEWithLogitsLoss"]:
        return BCEWithLogitsLossWrapper(pos_weight=pos_weight)
    elif loss_type == "FocalLoss":
        return FocalLoss(
            alpha=float(loss_cfg.get("alpha", 0.5)),
            gamma=float(loss_cfg.get("gamma", 2.0)),
        )
    elif loss_type in ["FocalTversky", "FocalTverskyLoss"]:
        return FocalTverskyLoss(
            alpha=float(loss_cfg.get("tversky_alpha", 0.3)),
            beta=float(loss_cfg.get("tversky_beta", 0.7)),
            exponent=float(loss_cfg.get("tversky_exponent", 0.75)),
            smooth=smooth,
        )
    elif loss_type == "CombinedLoss":
        return CombinedLoss(
            bce_weight=float(loss_cfg.get("bce_weight", 0.3)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.7)),
            focal_weight=float(loss_cfg.get("focal_weight", 0.0)),
            pos_weight=pos_weight,
            smooth=smooth,
        )
    else:
        raise ValueError(f"Không hỗ trợ loss type: {loss_type}")
