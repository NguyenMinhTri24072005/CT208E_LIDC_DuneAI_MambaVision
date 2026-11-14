# -*- coding: utf-8 -*-
"""
mamba_unet/trainer.py - Trainer chuyên biệt cho Mamba-UNet (kế thừa BaseTrainer).

Quy tắc bất biến (spec mục 5.3):
  - Lịch đóng băng backbone suy ra trực tiếp từ số epoch (resume ở đâu cũng đúng):
      * freeze_backbone_epochs = 0: không đóng băng.
      * freeze_backbone_epochs = N > 0: đóng băng trong N epoch đầu (epoch <= N).
      * freeze_backbone_epochs = -1: đóng băng suốt quá trình.
  - Khi đóng băng, BatchNorm của backbone giữ eval() để không lệch running mean/var.
  - Phân chia 2 nhóm tham số Optimizer:
      * Nhóm 'decoder' : lr = base_lr
      * Nhóm 'backbone': lr = base_lr * backbone_lr_mult (mặc định 0.1 khi pretrained, 1.0 khi scratch).
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.engine.base_trainer import BaseTrainer
from configs.config import IMAGE_SIZE


class MambaUNetTrainer(BaseTrainer):
    """Trainer chuyên biệt cho mô hình MambaUNet."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        backbone_name: str = "mamba_vision_T",
        activation_name: str = "relu",
        in_channels: int = 3,
        num_classes: int = 1,
        freeze_backbone_epochs: int = 0,
        backbone_lr_mult: float = 0.1,
        **kwargs,
    ):
        self.backbone_name = backbone_name
        self.activation_name = activation_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.freeze_backbone_epochs = int(freeze_backbone_epochs)
        self.backbone_lr_mult = float(backbone_lr_mult)
        super().__init__(model=model, criterion=criterion, **kwargs)

    def on_epoch_start(self, epoch: int):
        """Hook điều khiển đóng/mở băng backbone suy ra từ số epoch hiện tại."""
        unwrapped = self._unwrap_model()
        should_freeze = (self.freeze_backbone_epochs == -1) or (epoch <= self.freeze_backbone_epochs and self.freeze_backbone_epochs > 0)

        if should_freeze:
            # Đóng băng toàn bộ trọng số của backbone encoder
            for param in unwrapped.encoder.parameters():
                param.requires_grad = False
            # BatchNorm của backbone giữ eval()
            for m in unwrapped.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
        else:
            # Mở băng backbone cho huấn luyện fine-tuning
            for param in unwrapped.encoder.parameters():
                param.requires_grad = True

    def build_param_groups(self, lr: float) -> List[Dict[str, Any]]:
        """Phân chia 2 nhóm learning rate: decoder và backbone."""
        unwrapped = self._unwrap_model()

        decoder_params = [p for p in unwrapped.decoder.parameters() if p.requires_grad]
        backbone_params = [p for p in unwrapped.encoder.parameters() if p.requires_grad]

        groups = [
            {"params": decoder_params, "lr": lr, "name": "decoder"},
        ]
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": lr * self.backbone_lr_mult,
                "name": "backbone",
            })
        return groups

    def checkpoint_meta(self) -> Dict[str, Any]:
        """Metadata kiến trúc chuẩn của mamba_unet."""
        bb = getattr(self, "backbone_name", "mamba_vision_T")
        act = getattr(self, "activation_name", "relu")
        in_ch = getattr(self, "in_channels", 3)
        num_cls = getattr(self, "num_classes", 1)
        freeze_ep = getattr(self, "freeze_backbone_epochs", 0)
        lr_mult = getattr(self, "backbone_lr_mult", 0.1)
        img_sz = int(os.environ.get("DUNEAI_IMAGE_SIZE", IMAGE_SIZE))
        return {
            "arch": "mamba_unet",
            "backbone": bb,
            "activation": act,
            "in_channels": in_ch,
            "num_classes": num_cls,
            "image_size": img_sz,
            "freeze_backbone_epochs": freeze_ep,
            "backbone_lr_mult": lr_mult,
            "model_config": {
                "backbone": bb,
                "activation": act,
                "in_channels": in_ch,
                "num_classes": num_cls,
                "image_size": img_sz,
            },
        }


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.trainer...")
    from mamba_unet.backbone.mamba_vision import MambaVision
    from mamba_unet.losses import get_loss
    from mamba_unet.model import MambaUNet

    tiny_bb = MambaVision(
        dim=16,
        in_dim=8,
        depths=[1, 1, 2, 1],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 2, 4, 4],
        in_chans=3,
    )
    model = MambaUNet(in_channels=3, num_classes=1, backbone=tiny_bb)
    criterion = get_loss("dice_bce")

    trainer = MambaUNetTrainer(
        model=model,
        criterion=criterion,
        freeze_backbone_epochs=2,
        backbone_lr_mult=0.1,
        show_progress=False,
        use_amp=False,
    )

    # 1. Kiểm tra epoch 1: backbone phải bị đóng băng
    trainer.on_epoch_start(1)
    for p in model.encoder.parameters():
        assert not p.requires_grad, "Backbone phải bị đóng băng ở Epoch 1!"
    print("  ✓ Epoch 1: Backbone đóng băng chính xác.")

    # 2. Kiểm tra epoch 2: backbone vẫn bị đóng băng
    trainer.on_epoch_start(2)
    for p in model.encoder.parameters():
        assert not p.requires_grad, "Backbone phải bị đóng băng ở Epoch 2!"
    print("  ✓ Epoch 2: Backbone đóng băng chính xác.")

    # 3. Kiểm tra epoch 3: backbone phải được mở băng
    trainer.on_epoch_start(3)
    for p in model.encoder.parameters():
        assert p.requires_grad, "Backbone phải được mở băng ở Epoch 3!"
    print("  ✓ Epoch 3: Backbone mở băng chính xác.")

    # 4. Kiểm tra nhóm tham số Optimizer
    groups = trainer.build_param_groups(1e-4)
    assert len(groups) == 2, f"Phải có đúng 2 nhóm tham số, nhận được {len(groups)}"
    assert groups[0]["name"] == "decoder" and groups[0]["lr"] == 1e-4
    assert groups[1]["name"] == "backbone" and abs(groups[1]["lr"] - 1e-5) < 1e-9
    print("  ✓ Phân nhóm Learning Rate chính xác (decoder: 1e-4, backbone: 1e-5).")

    print("✅ Selftest mamba_unet.trainer hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
