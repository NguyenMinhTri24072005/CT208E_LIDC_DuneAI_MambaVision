# -*- coding: utf-8 -*-
"""
unet_base/encoder.py - Bộ mã hóa CNN cổ điển 5 mức (CNNBaselineEncoder) riêng của unet_base.

Quy tắc bất biến:
  - Bảo toàn tên các lớp con: inc, down1, down2, down3, down4.
  - Trích xuất 5 mức đặc trưng: [H, H/2, H/4, H/8, H/16] với số kênh [features, 2f, 4f, 8f, 16f].
  - Import khối DoubleConv từ unet_base.blocks, KHÔNG import từ common hay mamba_unet.
"""

import sys
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unet_base.blocks import DoubleConv


class Down(nn.Module):
    """Khối giảm độ phân giải: MaxPool2d(2) theo sau bởi DoubleConv."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: Optional[Union[str, Callable[[], nn.Module]]] = None,
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, act=act),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class CNNBaselineEncoder(nn.Module):
    """
    CNN Baseline Encoder trích xuất các cấp độ đặc trưng:
      - Khi use_stride4_stem=False (Chuẩn U-Net cổ điển 5 mức):
          x1: (B, features,     H,    W)
          x2: (B, features*2,   H/2,  W/2)
          x3: (B, features*4,   H/4,  W/4)
          x4: (B, features*8,   H/8,  W/8)
          x5: (B, features*16,  H/16, W/16) (Bottleneck)
    """
    def __init__(
        self,
        in_channels: int = 3,
        features: int = 64,
        use_stride4_stem: bool = False,
        act: Optional[Union[str, Callable[[], nn.Module]]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.features = features
        self.use_stride4_stem = use_stride4_stem
        self.act = act

        if use_stride4_stem:
            # Nhánh stem giảm xuống H/4 (giữ lại để tương thích tối đa)
            from unet_base.activations import get_activation
            act_fac = act if callable(act) else get_activation(act)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, features // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(features // 2),
                act_fac(),
                nn.Conv2d(features // 2, features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(features),
                act_fac(),
            )
            self.stage1 = DoubleConv(features, features, act=act)
            self.stage2 = Down(features, features * 2, act=act)
            self.stage3 = Down(features * 2, features * 4, act=act)
            self.stage4 = Down(features * 4, features * 8, act=act)
            self.out_channels = [features, features * 2, features * 4, features * 8]
        else:
            # Kiến trúc U-Net Encoder chuẩn 5 tầng
            self.inc = DoubleConv(in_channels, features, act=act)
            self.down1 = Down(features, features * 2, act=act)
            self.down2 = Down(features * 2, features * 4, act=act)
            self.down3 = Down(features * 4, features * 8, act=act)
            self.down4 = Down(features * 8, features * 16, act=act)
            self.out_channels = [features, features * 2, features * 4, features * 8, features * 16]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.use_stride4_stem:
            x0 = self.stem(x)          # H/4
            f1 = self.stage1(x0)       # H/4
            f2 = self.stage2(f1)       # H/8
            f3 = self.stage3(f2)       # H/16
            f4 = self.stage4(f3)       # H/32
            return [f1, f2, f3, f4]
        else:
            x1 = self.inc(x)           # H
            x2 = self.down1(x1)        # H/2
            x3 = self.down2(x2)        # H/4
            x4 = self.down3(x3)        # H/8
            x5 = self.down4(x4)        # H/16
            return [x1, x2, x3, x4, x5]


def _selftest():
    print("🧪 Bắt đầu selftest unet_base.encoder...")
    dummy = torch.randn(2, 3, 64, 64)
    enc = CNNBaselineEncoder(in_channels=3, features=16, use_stride4_stem=False)
    feats = enc(dummy)
    assert len(feats) == 5, f"Phải trả về 5 tầng đặc trưng, nhận được {len(feats)}"
    assert feats[0].shape == (2, 16, 64, 64)
    assert feats[1].shape == (2, 32, 32, 32)
    assert feats[2].shape == (2, 64, 16, 16)
    assert feats[3].shape == (2, 128, 8, 8)
    assert feats[4].shape == (2, 256, 4, 4)
    print("✅ Selftest unet_base.encoder hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
