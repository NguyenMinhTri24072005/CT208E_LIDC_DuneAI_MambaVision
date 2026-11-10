# -*- coding: utf-8 -*-
"""
unet_base/blocks.py - Các khối tích chập chuẩn DoubleConv và OutConv riêng của unet_base.

Quy tắc bất biến (R1, R11, R3):
  - Khối mạng thuộc sở hữu độc quyền của unet_base, không dùng chung và không import từ mamba_unet.
  - Bảo toàn cấu trúc và chỉ số lớp con (0: Conv2d, 1: BatchNorm2d, 2: Act, 3: Conv2d, 4: BatchNorm2d, 5: Act)
    để tương thích 100% với tập khóa state_dict ban đầu.
"""

import sys
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unet_base.activations import DEFAULT_ACTIVATION, get_activation


class DoubleConv(nn.Module):
    """
    Khối tích chập kép chuẩn của U-Net:
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

        # Phân giải activation factory
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
    """
    Lớp tích chập xuất kết quả cuối cùng (Final 1x1 Convolution):
      Conv2d(1x1) có khởi tạo bias tùy chọn (mặc định -4.0 cho phân đoạn khối u thưa).
    """
    def __init__(self, in_channels: int, out_channels: int, init_bias: Optional[float] = -4.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if init_bias is not None and self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, float(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _selftest():
    print("🧪 Bắt đầu selftest unet_base.blocks...")
    x = torch.randn(2, 16, 32, 32)

    # 1. Kiểm tra shape forward
    dc = DoubleConv(16, 32)
    out = dc(x)
    assert out.shape == (2, 32, 32, 32), f"DoubleConv shape sai: {out.shape}"

    oc = OutConv(32, 1)
    out_oc = oc(out)
    assert out_oc.shape == (2, 1, 32, 32), f"OutConv shape sai: {out_oc.shape}"

    # 2. Kiểm tra bất biến khóa state_dict khi đổi activation không tham số
    keys_relu = set(DoubleConv(16, 32, act="relu").state_dict().keys())
    keys_gelu = set(DoubleConv(16, 32, act="gelu").state_dict().keys())
    keys_silu = set(DoubleConv(16, 32, act="silu").state_dict().keys())
    assert keys_relu == keys_gelu == keys_silu, "Đổi activation không tham số làm lệch tập khóa!"

    # 3. Kiểm tra activation có tham số (star_relu) thêm khóa học được
    keys_star = set(DoubleConv(16, 32, act="star_relu").state_dict().keys())
    diff = keys_star - keys_relu
    assert len(diff) > 0, "star_relu phải thêm khóa tham số học được!"
    print(f"  ✓ Khóa thêm vào bởi star_relu: {diff}")

    print("✅ Selftest unet_base.blocks hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
