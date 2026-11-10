# -*- coding: utf-8 -*-
"""
unet_base/decoder.py - Bộ giải mã U-Net cổ điển 4 tầng phóng to (ClassicUNetDecoder) riêng của unet_base.

Quy tắc bất biến:
  - Nhận 5 mức đặc trưng từ CNNBaselineEncoder: [x1 (H), x2 (H/2), x3 (H/4), x4 (H/8), x5 (H/16)].
  - Bảo toàn tên các lớp con: up1, up2, up3, up4, outc.
  - Sử dụng khối DoubleConv và OutConv từ unet_base.blocks.
"""

import sys
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unet_base.blocks import DoubleConv, OutConv


class UpBlock(nn.Module):
    """
    Khối giải mã: Phóng to Tensor và ghép nối (Concat) với Tensor từ Encoder (Skip Connection).
    """
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        use_conv_transpose: bool = True,
        act: Optional[Union[str, Callable[[], nn.Module]]] = None,
    ):
        super().__init__()
        self.use_conv_transpose = use_conv_transpose
        if use_conv_transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            conv_in = (in_ch // 2) + skip_ch
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            conv_in = in_ch + skip_ch

        self.conv = DoubleConv(conv_in, out_ch, act=act)

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[2:] != skip_x.shape[2:]:
            x = F.interpolate(x, size=skip_x.shape[2:], mode="bilinear", align_corners=True)

        # Gắn kết dọc theo trục kênh: skip connection từ encoder đi trước, upsampled đi sau
        x = torch.cat([skip_x, x], dim=1)
        return self.conv(x)


class ClassicUNetDecoder(nn.Module):
    """
    Bộ giải mã U-Net cổ điển 4 tầng phóng to đối xứng hoàn toàn với CNNBaselineEncoder.
    """
    def __init__(
        self,
        features: int = 64,
        num_classes: int = 1,
        act: Optional[Union[str, Callable[[], nn.Module]]] = None,
        use_conv_transpose: bool = True,
        init_bias: Optional[float] = -4.0,
    ):
        super().__init__()
        self.features = features
        self.act = act

        self.up1 = UpBlock(features * 16, features * 8, features * 8, use_conv_transpose=use_conv_transpose, act=act)
        self.up2 = UpBlock(features * 8, features * 4, features * 4, use_conv_transpose=use_conv_transpose, act=act)
        self.up3 = UpBlock(features * 4, features * 2, features * 2, use_conv_transpose=use_conv_transpose, act=act)
        self.up4 = UpBlock(features * 2, features, features, use_conv_transpose=use_conv_transpose, act=act)
        self.outc = OutConv(features, num_classes, init_bias=init_bias)

    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4, x5 = features_list
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def _selftest():
    print("🧪 Bắt đầu selftest unet_base.decoder...")
    features = 16
    dec = ClassicUNetDecoder(features=features, num_classes=1)

    x1 = torch.randn(2, features, 64, 64)
    x2 = torch.randn(2, features * 2, 32, 32)
    x3 = torch.randn(2, features * 4, 16, 16)
    x4 = torch.randn(2, features * 8, 8, 8)
    x5 = torch.randn(2, features * 16, 4, 4)

    out = dec([x1, x2, x3, x4, x5])
    assert out.shape == (2, 1, 64, 64), f"Decoder output shape sai: {out.shape}"
    print("✅ Selftest unet_base.decoder hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
