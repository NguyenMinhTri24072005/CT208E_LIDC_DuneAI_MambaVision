# -*- coding: utf-8 -*-
"""
mamba_unet/model.py - Mô hình Mamba-UNet hoàn chỉnh cho phân đoạn ảnh y tế 2.5D.

Quy tắc bất biến:
  - 2 thuộc tính rõ ràng: self.encoder (MambaVision) và self.decoder (UNetDecoder).
  - Loại bỏ hoàn toàn alias ModularUNet và loại bỏ nhánh backbone in ('cnn', 'cnn_baseline').
  - Tương thích 100% tập khóa tests/golden/mamba_unet_keys.json (367 keys).
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mamba_unet.backbone.mamba_vision import MambaVision, build_mambavision_backbone
from mamba_unet.decoder import UNetDecoder


class MambaUNet(nn.Module):
    """
    Kiến trúc Mamba-UNet:
        Input (B, 3, H, W) -> MambaVision Encoder -> 4 feature maps [f1, f2, f3, f4]
                           -> UNet Decoder + Skip Connections -> Logits (B, num_classes, H, W).

    Args:
        in_channels (int): Số kênh ảnh đầu vào (3 cho lát cắt 2.5D: z-1, z, z+1).
        num_classes (int): Số lớp phân đoạn (mặc định 1 cho nhãn khối u).
        backbone (str | nn.Module): Tên biến thể ('mamba_vision_T', ...) hoặc module MambaVision.
        pretrained (bool): Nạp trọng số ImageNet-1K pretrained từ NVIDIA.
        final_channels (int): Số kênh tích chập trước lớp xuất (mặc định 32).
        act (str | Callable): Hàm kích hoạt cho Decoder.
        init_bias (float): Giá trị khởi tạo bias cho lớp output (mặc định -4.0).
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone: Union[str, nn.Module] = "mamba_vision_T",
        pretrained: bool = False,
        final_channels: int = 32,
        act: Optional[Union[str, Callable[[], nn.Module]]] = None,
        init_bias: Optional[float] = -4.0,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_name = backbone if isinstance(backbone, str) else backbone.__class__.__name__

        # 1. Khởi tạo MambaVision Encoder
        if isinstance(backbone, str):
            self.encoder = build_mambavision_backbone(
                name=backbone,
                pretrained=pretrained,
                in_chans=in_channels,
            )
        elif isinstance(backbone, nn.Module):
            self.encoder = backbone
        else:
            raise TypeError(f"Backbone phải là str hoặc nn.Module, nhận được {type(backbone)}")

        encoder_channels = getattr(self.encoder, "out_channels", [80, 160, 320, 640])
        self.encoder_channels = encoder_channels

        # 2. Khởi tạo UNet Decoder
        self.decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            final_channels=final_channels,
            act=act,
            init_bias=init_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits


def build_mamba_unet(config: Optional[Dict[str, Any]] = None, **kwargs) -> MambaUNet:
    """Factory function tạo MambaUNet từ config dict hoặc kwargs."""
    cfg = config or {}
    model_cfg = cfg.get("model", cfg.get("MODEL", {}))

    in_channels = kwargs.get("in_channels", model_cfg.get("in_channels", 3))
    num_classes = kwargs.get("num_classes", model_cfg.get("num_classes", model_cfg.get("classes", 1)))
    backbone = kwargs.get("backbone", model_cfg.get("backbone", "mamba_vision_T"))
    pretrained = kwargs.get("pretrained", model_cfg.get("pretrained", False))
    final_channels = kwargs.get("final_channels", model_cfg.get("final_channels", 32))
    act = kwargs.get("act", kwargs.get("activation", model_cfg.get("activation", "relu")))
    init_bias = kwargs.get("init_bias", model_cfg.get("init_bias", -4.0))

    return MambaUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        final_channels=final_channels,
        act=act,
        init_bias=init_bias,
    )


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.model...")
    from mamba_unet.backbone.mamba_vision import MambaVision

    # Dùng tiny backbone để chạy siêu nhanh
    tiny_bb = MambaVision(
        dim=16,
        in_dim=8,
        depths=[1, 1, 2, 1],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 2, 4, 4],
        in_chans=3,
    )
    model = MambaUNet(in_channels=3, num_classes=1, backbone=tiny_bb)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64), f"Lỗi shape đầu ra: {out.shape}"

    # Kiểm tra số lượng keys của mamba_vision_T đầy đủ
    full_model = MambaUNet(in_channels=3, num_classes=1, backbone="mamba_vision_T", pretrained=False)
    keys = list(full_model.state_dict().keys())
    assert len(keys) == 367, f"Số keys phải là 367, nhận được {len(keys)}"
    assert all(k.startswith("encoder.") or k.startswith("decoder.") for k in keys), "Keys phải có tiền tố encoder. hoặc decoder."
    print("✅ Selftest mamba_unet.model hoàn tất thành công (367/367 keys chuẩn xác)!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
