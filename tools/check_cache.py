# -*- coding: utf-8 -*-
"""tools/check_cache.py - Kiểm tra P3 (mask phổi cắt u) và FOV khi đổi IMAGE_SIZE."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def check_cache(cache_dir: str, image_size: int = 448):
    root = Path(cache_dir)
    fr, lost_fov = [], []
    meta_files = sorted(root.glob("LIDC-IDRI-*/meta.json"))
    if not meta_files:
        # Hỗ trợ tìm kiếm theo thư mục con
        meta_files = sorted(root.glob("*/meta.json"))

    for mp in meta_files:
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        f = m.get("mask_in_lung_fraction")
        if f is not None:
            fr.append((float(f), mp.parent.name))
        shape = m.get("shape_1mm_zyx", None)
        if not shape or len(shape) < 3:
            continue
        z, h, w = shape
        if h > image_size or w > image_size:  # Sẽ bị crop giữa khi pad_or_crop_xy
            mask_file = mp.parent / "mask_1mm.npy"
            if not mask_file.exists():
                continue
            try:
                proj = np.load(mask_file, mmap_mode="r")
                proj = np.asarray(proj).any(axis=0)
                if proj.sum() == 0:
                    continue
                sy, sx = max(0, (h - image_size) // 2), max(0, (w - image_size) // 2)
                inside = proj[sy : sy + min(h, image_size), sx : sx + min(w, image_size)].sum()
                lost_fov.append((1.0 - float(inside) / float(proj.sum()), mp.parent.name))
            except Exception:
                pass

    print("=" * 70)
    print(f"📊 BÁO CÁO KIỂM TRA CACHE & FOV (Image Size: {image_size})")
    print("=" * 70)
    if fr:
        v = np.array([x for x, _ in fr])
        print(
            f"mask_in_lung_fraction: n={len(v)} | <0.9: {(v<0.9).sum()} | <0.7: {(v<0.7).sum()} | "
            f"median={np.median(v):.3f} | p5={np.percentile(v,5):.3f}"
        )
        print("5 ca có mask_in_lung_fraction thấp nhất:", sorted(fr)[:5])
    else:
        print("ℹ️ Không tìm thấy mask_in_lung_fraction trong meta.json (cache build thiếu --extract-lungs?)")

    print(f"Ca có FOV > {image_size}: {len(lost_fov)} | ca mất nhãn do crop: {sum(x>0 for x,_ in lost_fov)}")
    print("Top ca mất nhãn do crop:", sorted(lost_fov, reverse=True)[:5])
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng cache và FOV.")
    parser.add_argument("cache_dir", help="Đường dẫn thư mục cache")
    parser.add_argument("image_size", nargs="?", type=int, default=448, help="Kích thước ảnh cần kiểm tra (mặc định 448)")
    args = parser.parse_args()
    check_cache(args.cache_dir, args.image_size)


if __name__ == "__main__":
    main()
