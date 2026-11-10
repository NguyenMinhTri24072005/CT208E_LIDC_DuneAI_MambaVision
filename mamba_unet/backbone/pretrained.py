# -*- coding: utf-8 -*-
"""
mamba_unet/backbone/pretrained.py - Tải và nạp an toàn trọng số Pretrained chính thức từ NVIDIA.

Quy tắc bất biến:
  - Tải trọng số vào thư mục tạm chuẩn hệ điều hành: Path(tempfile.gettempdir()) (không dùng /tmp cứng).
  - Tương thích PyTorch >= 2.6: ưu tiên weights_only=True, fallback an toàn weights_only=False nếu có class tùy biến.
  - Bóc tách tiền tố 'module.', nạp strict=False, kiểm tra và cảnh báo nếu sai lệch số khóa (>10 khóa).
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

MAMBAVISION_URLS: Dict[str, str] = {
    "mamba_vision_t": "https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar",
    "mamba_vision_t2": "https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar",
    "mamba_vision_s": "https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar",
    "mamba_vision_b": "https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar",
}


def download_pretrained_weights(variant: str) -> Path:
    """Tải file trọng số pretrained của biến thể tương ứng từ NVIDIA HuggingFace."""
    key = variant.lower().strip()
    if key not in MAMBAVISION_URLS:
        raise ValueError(
            f"❌ Không tìm thấy URL pretrained cho biến thể '{variant}'. "
            f"Các biến thể hỗ trợ: {list(MAMBAVISION_URLS.keys())}"
        )

    url = MAMBAVISION_URLS[key]
    filename = f"{key}_nvidia_pretrained.pth.tar"
    dest_path = Path(tempfile.gettempdir()) / filename

    if not dest_path.exists() or dest_path.stat().st_size == 0:
        print(f"📥 Đang tải trọng số pretrained từ {url} ...")
        torch.hub.download_url_to_file(url, str(dest_path), progress=True)
        print(f"✅ Đã lưu trọng số vào: {dest_path}")
    else:
        print(f"ℹ️  Sử dụng trọng số pretrained đã lưu sẵn tại: {dest_path}")

    return dest_path


def load_pretrained_mambavision(
    model: nn.Module,
    variant: str,
    map_location: str = "cpu",
    weights_path: Optional[Path] = None,
) -> nn.Module:
    """
    Nạp trọng số pretrained vào instance MambaVision.
    """
    if weights_path is None:
        weights_path = download_pretrained_weights(variant)

    p = Path(weights_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số pretrained tại: {p}")

    # Đọc an toàn theo khuyến nghị PyTorch >= 2.6
    try:
        ckpt = torch.load(p, map_location=map_location, weights_only=True)
    except Exception as e:
        print(f"⚠️ weights_only=True bị từ chối ({e}), đọc lại với weights_only=False (file chính chủ NVIDIA)...")
        ckpt = torch.load(p, map_location=map_location, weights_only=False)

    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # Chuẩn hóa khóa: loại bỏ tiền tố module. nếu có
    clean_sd = {}
    for k, v in sd.items():
        clean_k = k[7:] if k.startswith("module.") else k
        # Bỏ qua classification head của ImageNet-1K
        if clean_k.startswith("head."):
            continue
        clean_sd[clean_k] = v

    # Nạp vào model
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)

    # Lọc bỏ head ra khỏi missing nếu có
    missing = [k for k in missing if not k.startswith("head.")]

    if len(missing) > 10:
        raise RuntimeError(
            f"❌ Nạp trọng số pretrained thất bại: Thiếu {len(missing)} khóa (>10 khóa)! "
            f"Có thể đã truyền sai biến thể kiến trúc '{variant}'."
        )

    print(
        f"✅ Nạp trọng số Pretrained NVIDIA '{variant}' thành công! "
        f"(Khóa nạp khớp: {len(clean_sd) - len(unexpected)}, thiếu: {len(missing)}, thừa: {len(unexpected)})"
    )
    return model
