# -*- coding: utf-8 -*-
"""
mamba_unet/decoder.py - Bộ giải mã UNetDecoder 4 tầng liên kết với MambaVision Backbone.

Quy tắc bất biến:
  - Nhận 4 feature maps đa tỷ lệ [H/4, H/8, H/16, H/32] từ MambaVision.
  - Sử dụng các khối DoubleConv và OutConv từ mamba_unet.blocks (không import từ unet_base).
  - Bảo toàn 100% cấu trúc và tên lớp con: up1, up2, up3, final_up, outc.
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

from mamba_unet.blocks import DoubleConv, OutConv


class UpBlock(nn.Module):
    """
    Khối giải mã: Phóng to Tensor và ghép nối (Concat) với Tensor từ Encoder (Skip Connection).
    Mặc định dùng Bilinear Upsampling (use_conv_transpose=False) cho Mamba-UNet.
    """
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        use_conv_transpose: bool = False,
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

        x = torch.cat([skip_x, x], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    U-Net Decoder 4 tầng kết nối với MambaVision Backbone:
      f1 (H/4), f2 (H/8), f3 (H/16), f4 (H/32 - Bottleneck).
    """
    def __init__(
        self,
        encoder_channels: Optional[List[int]] = None,
        num_classes: int = 1,
        final_channels: int = 32,
        act: Optional[Union[str, Callable[[], nn.Module]]] = None,
        init_bias: Optional[float] = -4.0,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [80, 160, 320, 640]  # Mặc định MambaVision-T

        c1, c2, c3, c4 = encoder_channels
        self.encoder_channels = encoder_channels

        self.up1 = UpBlock(c4, c3, c3, use_conv_transpose=False, act=act)
        self.up2 = UpBlock(c3, c2, c2, use_conv_transpose=False, act=act)
        self.up3 = UpBlock(c2, c1, c1, use_conv_transpose=False, act=act)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            DoubleConv(c1, final_channels, act=act),
        )
        self.outc = OutConv(final_channels, num_classes, init_bias=init_bias)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        d1 = self.up1(f4, f3)
        d2 = self.up2(d1, f2)
        d3 = self.up3(d2, f1)
        x = self.final_up(d3)
        logits = self.outc(x)
        return logits


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.decoder...")
    dec = UNetDecoder(encoder_channels=[80, 160, 320, 640], num_classes=1)

    f1 = torch.randn(2, 80, 32, 32)
    f2 = torch.randn(2, 160, 16, 16)
    f3 = torch.randn(2, 320, 8, 8)
    f4 = torch.randn(2, 640, 4, 4)

    out = dec([f1, f2, f3, f4])
    assert out.shape == (2, 1, 128, 128), f"Lỗi shape đầu ra UNetDecoder: {out.shape}"
    print("✅ Selftest mamba_unet.decoder hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
