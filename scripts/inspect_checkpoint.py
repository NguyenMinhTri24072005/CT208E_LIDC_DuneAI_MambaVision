# -*- coding: utf-8 -*-
"""
scripts/inspect_checkpoint.py - Kiểm tra thông tin, cấu trúc và metadata của checkpoint PyTorch.

In chi tiết:
  - Kiến trúc (arch), số epoch, best_val_dice, cấu hình mô hình (features, backbone, activation).
  - Số lượng keys, tổng tham số, kiểu dữ liệu trọng số (float32, float16).
  - Nhận diện định dạng (full training checkpoint, fp16 weights export, hay checkpoint legacy).
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.engine.checkpoint_io import load_checkpoint_file


def inspect_checkpoint(checkpoint_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """Kiểm tra và trả về metadata của file checkpoint."""
    p = Path(checkpoint_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {p}")

    size_mb = p.stat().st_size / (1024 * 1024)
    data = load_checkpoint_file(p, map_location="cpu")

    is_dict = isinstance(data, dict)
    sd = data.get("model_state_dict", data.get("state_dict", data)) if is_dict else data
    meta = data.get("meta", {}) if is_dict else {}

    num_keys = len(sd)
    total_params = sum(t.numel() for t in sd.values() if isinstance(t, torch.Tensor))
    dtypes = {str(t.dtype) for t in sd.values() if isinstance(t, torch.Tensor)}

    has_opt = is_dict and ("optimizer_state_dict" in data and data["optimizer_state_dict"] is not None)
    has_sch = is_dict and ("scheduler_state_dict" in data and data["scheduler_state_dict"] is not None)

    prefixes = set()
    for k in sd.keys():
        parts = k.split(".")
        prefixes.add(parts[0])

    info = {
        "path": str(p),
        "size_mb": round(size_mb, 2),
        "arch": meta.get("arch", "unknown"),
        "epoch": data.get("epoch", None) if is_dict else None,
        "best_val_dice": data.get("best_val_dice", None) if is_dict else None,
        "meta": meta,
        "num_keys": num_keys,
        "total_params": total_params,
        "dtypes": list(dtypes),
        "has_optimizer": has_opt,
        "has_scheduler": has_sch,
        "prefixes": list(sorted(prefixes)),
    }

    print("=" * 70)
    print(f"🔍 THÔNG TIN CHECKPOINT: {p.name} ({info['size_mb']} MB)")
    print("=" * 70)
    print(f"  - Kiến trúc (arch)       : {info['arch']}")
    print(f"  - Epoch đã hoàn tất      : {info['epoch']}")
    print(f"  - Best Val Dice          : {info['best_val_dice']}")
    print(f"  - Tổng số lượng keys     : {info['num_keys']}")
    print(f"  - Tổng số tham số (params): {info['total_params']:,}")
    print(f"  - Kiểu dữ liệu trọng số  : {', '.join(info['dtypes'])}")
    print(f"  - Chứa Optimizer state   : {'Có' if info['has_optimizer'] else 'Không (weights-only)'}")
    print(f"  - Tiền tố khóa (prefixes): {', '.join(info['prefixes'])}")
    if meta:
        print("  - Metadata chi tiết      :")
        for k, v in meta.items():
            print(f"      * {k}: {v}")

    if verbose:
        print("\n  - Danh sách các keys:")
        for k, v in sd.items():
            shape_str = list(v.shape) if hasattr(v, "shape") else "N/A"
            print(f"      * {k}: {shape_str} ({v.dtype})")
    print("=" * 70)
    return info


def _selftest():
    print("🧪 Bắt đầu selftest scripts.inspect_checkpoint...")
    tmp = Path(tempfile.mkdtemp())
    try:
        from unet_base.model import UNetBase
        m = UNetBase(features=8)
        ckpt = {
            "model_state_dict": m.state_dict(),
            "epoch": 5,
            "best_val_dice": 0.85,
            "meta": {"arch": "unet_base", "features": 8},
        }
        p = tmp / "test_model.pth"
        torch.save(ckpt, p)
        info = inspect_checkpoint(p, verbose=False)
        assert info["arch"] == "unet_base"
        assert info["epoch"] == 5
        assert info["best_val_dice"] == 0.85
        assert info["num_keys"] == 118
        print("✅ Selftest scripts.inspect_checkpoint hoàn tất thành công!")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Kiểm tra thông tin chi tiết file checkpoint")
    parser.add_argument("checkpoint", nargs="?", default=None, help="Đường dẫn tới file .pth")
    parser.add_argument("--verbose", action="store_true", help="In chi tiết danh sách tất cả các keys và shape")
    parser.add_argument("--selftest", action="store_true", help="Chạy kiểm thử tự động")
    args = parser.parse_args()

    if args.selftest or not args.checkpoint:
        _selftest()
    else:
        inspect_checkpoint(Path(args.checkpoint), verbose=args.verbose)


if __name__ == "__main__":
    main()
