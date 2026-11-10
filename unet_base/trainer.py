# -*- coding: utf-8 -*-
"""
unet_base/trainer.py - Trainer chuyên biệt cho kiến trúc U-Net Base (kế thừa BaseTrainer).

Quy tắc bất biến:
  - Kế thừa common.engine.base_trainer.BaseTrainer.
  - Ghi đè checkpoint_meta() trả về siêu dữ liệu chuẩn (arch="unet_base", features, activation).
  - Không import bất kỳ hàm loss nào trực tiếp (nhận criterion qua tiêm phụ thuộc).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.engine.base_trainer import BaseTrainer


class UNetBaseTrainer(BaseTrainer):
    """Trainer chuyên biệt cho mô hình UNetBase."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        features: int = 64,
        activation_name: str = "relu",
        in_channels: int = 3,
        classes: int = 1,
        **kwargs,
    ):
        super().__init__(model=model, criterion=criterion, **kwargs)
        self.features = features
        self.activation_name = activation_name
        self.in_channels = in_channels
        self.classes = classes

    def checkpoint_meta(self) -> Dict[str, Any]:
        """Trả về thông tin metadata đặc trưng của unet_base để lưu checkpoint."""
        return {
            "arch": "unet_base",
            "features": self.features,
            "activation": self.activation_name,
            "in_channels": self.in_channels,
            "classes": self.classes,
            "model_config": {
                "features": self.features,
                "activation": self.activation_name,
                "in_channels": self.in_channels,
                "classes": self.classes,
            },
        }

    def build_param_groups(self, lr: float) -> List[Dict[str, Any]]:
        """UNetBase huấn luyện đồng bộ toàn bộ trọng số với cùng learning rate."""
        return [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": lr}]


def _selftest():
    print("🧪 Bắt đầu selftest unet_base.trainer...")
    from unet_base.model import UNetBase
    from unet_base.losses import get_loss

    model = UNetBase(features=8)
    criterion = get_loss("dice_bce")
    trainer = UNetBaseTrainer(
        model=model,
        criterion=criterion,
        features=8,
        activation_name="relu",
        show_progress=False,
        use_amp=False,
    )
    meta = trainer.checkpoint_meta()
    assert meta["arch"] == "unet_base"
    assert meta["features"] == 8
    assert meta["activation"] == "relu"
    print("✅ Selftest unet_base.trainer hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
