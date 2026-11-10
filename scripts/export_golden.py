# -*- coding: utf-8 -*-
"""
scripts/export_golden.py - Công cụ xuất và kiểm định các file snapshot golden.

Cung cấp nguồn chân lý (Source of Truth) cho các bài test bất biến:
  - tests/golden/unet_base_keys.json
  - tests/golden/mamba_unet_keys.json
  - tests/golden/unet_base_tiny_state.pth
  - tests/golden/unet_base_tiny_out.pt
  - tests/golden/mamba_unet_tiny_state.pth
  - tests/golden/mamba_unet_tiny_out.pt
  - tests/golden/legacy_prefixless_unet_keys.json
"""

import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn

from unet_base.model import UNetBase
from mamba_unet.model import MambaUNet
from mamba_unet.backbone.mamba_vision import MambaVision


def export_golden(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📦 Đang xuất golden snapshots vào: {output_dir.resolve()}")

    # 1. UNetBase full keys (features=64, act="relu")
    print("1/5. Đang trích xuất unet_base_keys.json (features=64)...")
    unet_full = UNetBase(in_channels=3, classes=1, features=64, act="relu")
    unet_keys = {k: list(v.shape) for k, v in unet_full.state_dict().items()}
    with open(output_dir / "unet_base_keys.json", "w", encoding="utf-8") as f:
        json.dump(unet_keys, f, indent=2)
    print(f"   -> Đã lưu {len(unet_keys)} khóa.")

    # 2. MambaUNet full keys (backbone="mamba_vision_T", pretrained=False, act="relu")
    print("2/5. Đang trích xuất mamba_unet_keys.json (mamba_vision_T)...")
    mamba_full = MambaUNet(in_channels=3, num_classes=1, backbone="mamba_vision_T", pretrained=False, act="relu")
    mamba_keys = {k: list(v.shape) for k, v in mamba_full.state_dict().items()}
    with open(output_dir / "mamba_unet_keys.json", "w", encoding="utf-8") as f:
        json.dump(mamba_keys, f, indent=2)
    print(f"   -> Đã lưu {len(mamba_keys)} khóa.")

    # 3. UNetBase tiny model & output (features=8)
    print("3/5. Đang sinh unet_base_tiny_state.pth và unet_base_tiny_out.pt (features=8)...")
    torch.manual_seed(42)
    unet_tiny = UNetBase(in_channels=3, classes=1, features=8, act="relu")
    unet_tiny.eval()
    torch.save(unet_tiny.state_dict(), output_dir / "unet_base_tiny_state.pth")

    torch.manual_seed(123)
    x_unet = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out_unet = unet_tiny(x_unet)
    torch.save(out_unet, output_dir / "unet_base_tiny_out.pt")
    print(f"   -> Tiny UNet out shape: {tuple(out_unet.shape)}, out sum: {out_unet.sum().item():.4f}")

    # 4. MambaUNet tiny model & output
    print("4/5. Đang sinh mamba_unet_tiny_state.pth và mamba_unet_tiny_out.pt...")
    torch.manual_seed(42)
    tiny_bb = MambaVision(
        dim=16,
        in_dim=8,
        depths=[1, 1, 2, 1],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 2, 4, 4],
        in_chans=3,
    )
    mamba_tiny = MambaUNet(in_channels=3, num_classes=1, backbone=tiny_bb, act="relu")
    mamba_tiny.eval()
    torch.save(mamba_tiny.state_dict(), output_dir / "mamba_unet_tiny_state.pth")

    torch.manual_seed(123)
    x_mamba = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out_mamba = mamba_tiny(x_mamba)
    torch.save(out_mamba, output_dir / "mamba_unet_tiny_out.pt")
    print(f"   -> Tiny Mamba out shape: {tuple(out_mamba.shape)}, out sum: {out_mamba.sum().item():.4f}")

    print("✅ Hoàn tất xuất golden snapshots!")


if __name__ == "__main__":
    out = _REPO_ROOT / "tests" / "golden"
    export_golden(out)
