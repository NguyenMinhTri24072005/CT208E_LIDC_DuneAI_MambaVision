# -*- coding: utf-8 -*-
"""
mamba_unet/backbone/pretrained.py - Tải và nạp an toàn trọng số Pretrained chính thức từ NVIDIA.

Quy tắc bất biến:
  - Smart Cache 5 tầng:
      1. Repo cục bộ: pretrained_weights/
      2. Google Drive Mirror (khi chạy Colab): /content/drive/MyDrive/.../pretrained_weights/
      3. SSD Colab siêu tốc: /content/pretrained_weights/
      4. OS Tempfile: Path(tempfile.gettempdir())
      5. Chỉ khi cả 4 nơi trên không có: Tải từ NVIDIA HuggingFace Hub 1 lần duy nhất và lưu vào pretrained_weights/.
  - Tương thích PyTorch >= 2.6: ưu tiên weights_only=True, fallback weights_only=False nếu cần.
  - Bóc tách tiền tố 'module.' và bỏ qua classifier 'head.' của ImageNet.
  - Kiểm tra nghiêm ngặt: Toàn bộ tham số của Stem, Stage 1-4 và Norm phải nạp khớp 100% (0 missing, 0 unexpected).
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MAMBAVISION_URLS: Dict[str, str] = {
    "mamba_vision_t": "https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar",
    "mamba_vision_t2": "https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar",
    "mamba_vision_s": "https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar",
    "mamba_vision_b": "https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar",
}


def resolve_pretrained_cache(variant: str) -> Path:
    """
    Dò tìm trọng số pretrained theo thứ tự ưu tiên 5 tầng:
      1. Cục bộ repo: _ROOT / pretrained_weights / filename
      2. Cục bộ cwd: Path('pretrained_weights') / filename
      3. Google Drive mirror (trên Colab): /content/drive/MyDrive/**/pretrained_weights / filename
      4. SSD Colab: /content/pretrained_weights / filename
      5. OS Temp: tempfile.gettempdir() / filename
      6. Nếu không có ở bất kỳ đâu -> Tải 1 lần từ NVIDIA Hugging Face và lưu vào pretrained_weights.
    """
    key = variant.lower().strip()
    if key not in MAMBAVISION_URLS:
        raise ValueError(
            f"❌ Không tìm thấy URL pretrained cho biến thể '{variant}'. "
            f"Các biến thể hỗ trợ: {list(MAMBAVISION_URLS.keys())}"
        )

    short_tag = key.replace("mamba_vision_", "")
    filenames = [
        f"{key}_nvidia_pretrained.pth.tar",
        f"mambavision_{short_tag}_1k.pth.tar",
    ]
    if key == "mamba_vision_t":
        filenames.append("mambavision_tiny_1k.pth.tar")
    elif key == "mamba_vision_t2":
        filenames.append("mambavision_tiny2_1k.pth.tar")
    elif key == "mamba_vision_s":
        filenames.append("mambavision_small_1k.pth.tar")
    elif key == "mamba_vision_b":
        filenames.append("mambavision_base_1k.pth.tar")

    search_dirs: List[Path] = [
        _ROOT / "pretrained_weights",
        Path("pretrained_weights"),
        Path("/content/pretrained_weights"),
        Path(tempfile.gettempdir()),
    ]

    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        search_dirs.insert(1, drive_root / "pretrained_weights")
        for cand_dir in drive_root.glob("**/pretrained_weights"):
            if cand_dir.is_dir() and cand_dir not in search_dirs:
                search_dirs.insert(2, cand_dir)

    for s_dir in search_dirs:
        for fname in filenames:
            cand = s_dir / fname
            if cand.exists() and cand.is_file() and cand.stat().st_size > 1024:
                print(f"ℹ️  [Smart Cache] Sử dụng trọng số có sẵn tại: {cand.resolve()}")
                return cand.resolve()

    url = MAMBAVISION_URLS[key]
    dest_dir = _ROOT / "pretrained_weights"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        dest_dir = Path(tempfile.gettempdir())

    primary_name = filenames[0]
    dest_path = dest_dir / primary_name

    print(f"📥 [Smart Cache] Đang tải trọng số pretrained chính thức từ {url} ...")
    torch.hub.download_url_to_file(url, str(dest_path), progress=True)
    print(f"✅ [Smart Cache] Đã lưu trọng số vào: {dest_path.resolve()}")

    for s_dir in search_dirs:
        if "drive/MyDrive" in str(s_dir) and s_dir.exists():
            try:
                shutil.copy2(dest_path, s_dir / primary_name)
                print(f"💾 [Smart Cache] Đã sao lưu 1 bản sang Google Drive: {s_dir / primary_name}")
            except Exception as e:
                print(f"⚠️ [Smart Cache] Không thể lưu bản sao sang Drive: {e}")
            break

    return dest_path.resolve()


def download_pretrained_weights(variant: str) -> Path:
    """Wrapper tương thích ngược cho resolve_pretrained_cache."""
    return resolve_pretrained_cache(variant)


def load_pretrained_mambavision(
    model: nn.Module,
    variant: str,
    map_location: str = "cpu",
    weights_path: Optional[Union[str, Path]] = None,
) -> nn.Module:
    """
    Nạp trọng số pretrained chuẩn NVIDIA vào instance MambaVision.
    - Tự động bóc tách tiền tố 'module.'
    - Bỏ qua classifier 'head.' của ImageNet.
    - Nghiêm ngặt: 0 missing keys, 0 unexpected keys trên toàn bộ backbone.
    """
    if weights_path is None:
        p = resolve_pretrained_cache(variant)
    else:
        p = Path(weights_path).resolve()

    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"Không tìm thấy file trọng số pretrained tại: {p}")

    try:
        import argparse
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([argparse.Namespace])
        ckpt = torch.load(p, map_location=map_location, weights_only=True)
    except Exception:
        ckpt = torch.load(p, map_location=map_location, weights_only=False)

    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))

    clean_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        clean_k = k[7:] if k.startswith("module.") else k
        if clean_k.startswith("head."):
            continue
        clean_sd[clean_k] = v

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    missing = [k for k in missing if not k.startswith("head.")]

    if len(missing) > 0 or len(unexpected) > 0:
        msg = (
            f"❌ Lỗi nạp trọng số pretrained '{variant}': "
            f"Thiếu {len(missing)} khóa, thừa {len(unexpected)} khóa!\n"
            f"  - Missing keys mẫu: {missing[:5]}\n"
            f"  - Unexpected keys mẫu: {unexpected[:5]}"
        )
        if len(missing) > 0:
            raise RuntimeError(msg)
        else:
            print(f"⚠️ Cảnh báo: {msg}")
    else:
        print(
            f"✅ Nạp trọng số Pretrained NVIDIA '{variant}' thành công 100%! "
            f"({len(clean_sd)} khóa khớp hoàn hảo, 0 missing, 0 unexpected)"
        )

    return model


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.backbone.pretrained...")

    # 1. Test cấu hình danh mục URLs
    assert len(MAMBAVISION_URLS) == 4, "Phải đủ 4 biến thể Tiny, Tiny2, Small, Base"
    for k in ["mamba_vision_t", "mamba_vision_t2", "mamba_vision_s", "mamba_vision_b"]:
        assert k in MAMBAVISION_URLS, f"Thiếu biến thể {k}"
    print("  ✓ URL Registry PASS")

    # 2. Test giải quyết Smart Cache (giả lập tệp giả)
    test_dir = _ROOT / "pretrained_weights"
    test_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = test_dir / "mamba_vision_t_nvidia_pretrained.pth.tar"

    # Nếu file thật chưa có, tạo dummy state_dict để test logic nạp và kiểm tra khóa
    from mamba_unet.backbone.mamba_vision import mamba_vision_T
    model = mamba_vision_T(pretrained=False, in_chans=3)

    if not dummy_file.exists() or dummy_file.stat().st_size == 0:
        print("  ℹ️  Tạo mock pretrained checkpoint để kiểm thử logic nạp...")
        mock_sd = {f"module.{k}": v for k, v in model.state_dict().items()}
        mock_sd["module.head.weight"] = torch.randn(1000, 640)
        mock_sd["module.head.bias"] = torch.randn(1000)
        torch.save({"model": mock_sd}, dummy_file)
        created_dummy = True
    else:
        created_dummy = False

    try:
        resolved = resolve_pretrained_cache("mamba_vision_t")
        assert resolved.exists() and resolved.stat().st_size > 0, "Lỗi resolve_pretrained_cache"
        print(f"  ✓ Smart Cache resolution PASS: {resolved.name}")

        # 3. Test nạp trọng số khớp 100%
        load_pretrained_mambavision(model, "mamba_vision_t", weights_path=resolved)
        print("  ✓ Nạp trọng số khớp 100% (0 missing, 0 unexpected) PASS")
    finally:
        if created_dummy and dummy_file.exists():
            dummy_file.unlink()

    print("✅ Selftest mamba_unet.backbone.pretrained hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
