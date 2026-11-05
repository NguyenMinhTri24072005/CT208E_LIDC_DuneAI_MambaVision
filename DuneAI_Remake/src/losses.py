# -*- coding: utf-8 -*-
"""
losses.py - Hàm mất mát (Loss Functions) và chỉ số đánh giá (Metrics) cho phân vùng khối u phổi.

Bao gồm:
  - DiceBCELoss: Kết hợp BCEWithLogitsLoss và Batch-level Soft Dice Loss.
  - TverskyFocalLoss: Tùy chọn chuyên trị khối u siêu nhỏ (phạt nặng False Negative).
  - compute_segmentation_metrics: Tính Intersection, Union, Dice, IoU từ raw logits.

Dùng:
  python DuneAI_Remake/src/losses.py selftest
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """
    Hàm mất mát tiêu chuẩn cho phân vùng y khoa:
        Loss = bce_weight * BCEWithLogits + dice_weight * (1 - SoftDice)
    Mặc định bật batch_dice=True để ổn định đạo hàm khi trong batch có lát cắt âm tính (không có u).
    """
    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=None,
        smooth=1.0,
        batch_dice=True,
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

    def forward(self, logits, targets):
        targets = targets.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )

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


class TverskyFocalLoss(nn.Module):
    """
    Hàm mất mát Tversky-Focal:
      - alpha < beta (mặc định 0.3 / 0.7) giúp phạt lỗi bỏ sót khối u (False Negative) nặng hơn.
      - gamma < 1.0 (hoặc > 1.0 tùy quy ước lũy thừa) tập trung đạo hàm vào các ca khó.
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        targets = targets.to(dtype=logits.dtype)
        probs = torch.sigmoid(logits).reshape(-1)
        targets_flat = targets.reshape(-1)

        tp = (probs * targets_flat).sum()
        fp = (probs * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma)


@torch.no_grad()
def compute_segmentation_metrics(logits, targets, threshold=0.5, eps=1e-6):
    """
    Tính toán các đại lượng cơ sở từ raw logits (B, 1, H, W) và nhãn (B, 1, H, W).
    Trả về dict chứa intersection, pred_sum, target_sum để cộng dồn toàn Epoch.
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


def _cmd_selftest(_):
    print("Đang chạy kiểm thử tự động losses.py...")
    targets = torch.zeros((4, 1, 64, 64), dtype=torch.float32)
    targets[0:3, 0, 16:32, 16:32] = 1.0  # 3 lát có u, 1 lát âm tính

    # 1. Dự đoán hoàn hảo (logits = +10 ở vùng u, -10 ở nền)
    logits_good = torch.where(targets > 0.5, torch.tensor(10.0), torch.tensor(-10.0))
    logits_good.requires_grad_(True)

    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    loss_good = criterion(logits_good, targets)
    loss_good.backward()

    m_good = compute_segmentation_metrics(logits_good, targets)
    assert loss_good.item() < 0.01, f"Loss dự đoán đúng phải ~0, nhận được: {loss_good.item()}"
    assert m_good["dice"] > 0.999 and m_good["iou"] > 0.999
    assert logits_good.grad is not None and not torch.isnan(logits_good.grad).any()

    # 2. Dự đoán sai hoàn toàn (ngược với nhãn)
    logits_bad = torch.where(targets > 0.5, torch.tensor(-10.0), torch.tensor(10.0))
    loss_bad = criterion(logits_bad, targets)
    m_bad = compute_segmentation_metrics(logits_bad, targets)
    assert loss_bad.item() > 5.0 and m_bad["dice"] < 0.001

    # 3. Kiểm thử TverskyFocalLoss
    tf_loss = TverskyFocalLoss()(logits_good.detach(), targets)
    assert tf_loss.item() < 0.01

    print(
        f"✅ selftest losses.py: OK! "
        f"(Good Loss={loss_good.item():.5f}, Dice={m_good['dice']:.4f} | "
        f"Bad Loss={loss_bad.item():.2f}, Dice={m_bad['dice']:.4f})"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=False)
    s = sub.add_parser("selftest", help="Kiểm thử hàm Loss và Metrics")
    s.set_defaults(fn=_cmd_selftest)
    args = ap.parse_args()
    _cmd_selftest(args)