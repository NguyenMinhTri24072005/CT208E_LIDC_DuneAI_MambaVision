# -*- coding: utf-8 -*-
"""
mamba_unet/blocks.py - Các khối tích chập chuẩn DoubleConv và OutConv riêng của mamba_unet.

Quy tắc bất biến (R1, R11, R3):
  - Khối mạng thuộc sở hữu độc quyền của mamba_unet, không dùng chung và không import từ unet_base.
  - Bảo toàn cấu trúc chỉ số lớp con (0: Conv2d, 1: BatchNorm2d, 2: Act, 3: Conv2d, 4: BatchNorm2d, 5: Act)
    để tương thích 100% với tập khóa decoder của mamba_unet_keys.json.
"""

import sys
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mamba_unet.activations import DEFAULT_ACTIVATION, get_activation


class DoubleConv(nn.Module):
    """
    Khối tích chập kép chuẩn:
      Conv2d(3x3, bias=False) -> BatchNorm2d -> Act -> Conv2d(3x3, bias=False) -> BatchNorm2d -> Act
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        act: Optional[Union[str, Callable[[], nn.Module]]] = None,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if act is None:
            act_factory = get_activation(DEFAULT_ACTIVATION)
        elif isinstance(act, str):
            act_factory = get_activation(act)
        elif callable(act):
            act_factory = act
        else:
            raise TypeError(f"Tham số 'act' phải là str hoặc Callable, nhận được {type(act)}")

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # 0
            nn.BatchNorm2d(mid_channels),                                                # 1
            act_factory(),                                                               # 2
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), # 3
            nn.BatchNorm2d(out_channels),                                                # 4
            act_factory(),                                                               # 5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class OutConv(nn.Module):
    """Lớp tích chập 1x1 xuất mặt nạ phân đoạn với khởi tạo bias."""
    def __init__(self, in_channels: int, out_channels: int, init_bias: Optional[float] = -4.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if init_bias is not None and self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, float(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.blocks...")
    x = torch.randn(2, 16, 32, 32)
    dc = DoubleConv(16, 32)
    out = dc(x)
    assert out.shape == (2, 32, 32, 32), f"DoubleConv shape sai: {out.shape}"

    oc = OutConv(32, 1)
    out_oc = oc(out)
    assert out_oc.shape == (2, 1, 32, 32), f"OutConv shape sai: {out_oc.shape}"

    # Kiểm tra bảo toàn khóa khi đổi activation không tham số
    keys_relu = set(DoubleConv(16, 32, act="relu").state_dict().keys())
    keys_gelu = set(DoubleConv(16, 32, act="gelu").state_dict().keys())
    assert keys_relu == keys_gelu, "Đổi activation không tham số làm lệch tập khóa!"

    # Kiểm tra star_relu thêm khóa
    keys_star = set(DoubleConv(16, 32, act="star_relu").state_dict().keys())
    assert len(keys_star - keys_relu) > 0, "star_relu phải thêm khóa tham số học được!"

    print("✅ Selftest mamba_unet.blocks hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
