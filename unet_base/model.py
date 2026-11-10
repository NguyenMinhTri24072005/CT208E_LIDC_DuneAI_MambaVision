# -*- coding: utf-8 -*-
"""
unet_base/model.py - Mô hình U-Net Base hoàn chỉnh với 2 thuộc tính self.encoder và self.decoder.

Quy tắc bất biến:
  - Duy nhất lớp UNetBase với 2 thuộc tính: self.encoder và self.decoder.
  - Xóa sạch mọi alias cũ (DuneUNet, PureUNet, ClassicUNet, BaselineUNet).
  - Tương thích 100% tập khóa state_dict ban đầu (118 keys).
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unet_base.encoder import CNNBaselineEncoder
from unet_base.decoder import ClassicUNetDecoder


class UNetBase(nn.Module):
    """
    Kiến trúc U-Net thuần (Pure Baseline U-Net):
      Input (B, 3, H, W) -> CNN Baseline Encoder -> 5 mức đặc trưng [x1, x2, x3, x4, x5]
                         -> Classic U-Net Decoder -> Logits (B, classes, H, W).

    Args:
        in_channels (int): Số kênh ảnh đầu vào (3 cho lát cắt 2.5D: z-1, z, z+1).
        classes (int): Số lớp phân đoạn (mặc định 1 cho mặt nạ khối u).
        features (int): Số kênh đặc trưng cơ sở (mặc định 64).
        act (str | Callable): Tên hoặc factory hàm kích hoạt.
        use_conv_transpose (bool): Sử dụng ConvTranspose2d (True) hay Bilinear Upsampling (False).
        init_bias (float): Giá trị khởi tạo bias cho lớp output (mặc định -4.0).
    """
    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 1,
        features: int = 64,
        act: Optional[Union[str, Callable[[], nn.Module]]] = None,
        use_conv_transpose: bool = True,
        init_bias: Optional[float] = -4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.features = features
        self.act = act

        self.encoder = CNNBaselineEncoder(
            in_channels=in_channels,
            features=features,
            use_stride4_stem=False,
            act=act,
        )
        self.decoder = ClassicUNetDecoder(
            features=features,
            num_classes=classes,
            act=act,
            use_conv_transpose=use_conv_transpose,
            init_bias=init_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits


def build_unet_base(config: Optional[Dict[str, Any]] = None, **kwargs) -> UNetBase:
    """
    Factory function khởi tạo UNetBase từ dictionary cấu hình hoặc kwargs.
    """
    cfg = config or {}
    model_cfg = cfg.get("model", cfg.get("MODEL", {}))

    in_channels = kwargs.get("in_channels", model_cfg.get("in_channels", 3))
    classes = kwargs.get("classes", model_cfg.get("classes", model_cfg.get("num_classes", 1)))
    features = kwargs.get("features", model_cfg.get("features", 64))
    act = kwargs.get("act", kwargs.get("activation", model_cfg.get("activation", "relu")))
    use_conv_transpose = kwargs.get("use_conv_transpose", model_cfg.get("use_conv_transpose", True))
    init_bias = kwargs.get("init_bias", model_cfg.get("init_bias", -4.0))

    return UNetBase(
        in_channels=in_channels,
        classes=classes,
        features=features,
        act=act,
        use_conv_transpose=use_conv_transpose,
        init_bias=init_bias,
    )


def _selftest():
    print("🧪 Bắt đầu selftest unet_base.model...")
    model = UNetBase(in_channels=3, classes=1, features=16)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64), f"UNetBase output shape sai: {out.shape}"

    # Kiểm tra số lượng keys trong state_dict
    sd_keys = list(model.state_dict().keys())
    assert len(sd_keys) == 118, f"Số keys phải là 118, nhận được {len(sd_keys)}"
    assert all(k.startswith("encoder.") or k.startswith("decoder.") for k in sd_keys), "Keys phải có tiền tố encoder. hoặc decoder."
    print("✅ Selftest unet_base.model hoàn tất thành công (118/118 keys chuẩn xác)!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
