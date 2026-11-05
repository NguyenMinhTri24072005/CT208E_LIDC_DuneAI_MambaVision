# -*- coding: utf-8 -*-
"""
offline_builder.py - Lọc lát cắt, cân bằng âm/dương và chia tập Train/Val từ cache_1mm.

Dùng:
  python offline_builder.py selftest
  python offline_builder.py build --cache-dir data/cache_1mm
"""
import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Nạp cấu hình và các hàm lõi từ medical_prep.py (cùng thư mục)
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
for p in (str(THIS_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.config import TUMOR_SIZE_LOW_LIM, ZERO_SL_RATIO, VAL_SPLIT, SEED
from medical_prep import load_cached, lung_cache_path


def build_dataset_index(
    cache_dir,
    out_csv_path=None,
    tumor_size_low_lim=TUMOR_SIZE_LOW_LIM,
    zero_sl_ratio=ZERO_SL_RATIO,
    val_split=VAL_SPLIT,
    seed=SEED,
    require_lung=True,
):
    """
    Quét thư mục cache_1mm, lọc lát cắt theo tiêu chuẩn y khoa của DuneAI và xuất slices.csv.
    Lưu ý: mask_1mm.npy đã ở tỷ lệ 1mm/pixel nên 1 pixel = 1.0 mm^2.
    """
    cache_dir = Path(cache_dir)
    if out_csv_path is None:
        out_csv_path = cache_dir / "slices.csv"
    out_csv_path = Path(out_csv_path)

    if not cache_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục cache: {cache_dir}")

    patient_ids = sorted([
        p.name for p in cache_dir.iterdir()
        if p.is_dir() and (p / "meta.json").exists() and (p / "image_1mm.npy").exists()
    ])
    if not patient_ids:
        raise RuntimeError(f"Không tìm thấy bệnh nhân hợp lệ nào trong {cache_dir}")

    # Chia Train/Val theo cấp độ Bệnh nhân (Patient-level split) để chống Data Leakage
    rng = random.Random(seed)
    shuffled_pids = list(patient_ids)
    rng.shuffle(shuffled_pids)

    if len(shuffled_pids) == 1:
        val_patients = set()
        train_patients = set(shuffled_pids)
    else:
        val_size = max(1, int(round(len(shuffled_pids) * val_split)))
        val_patients = set(shuffled_pids[:val_size])
        train_patients = set(shuffled_pids[val_size:])

    rows = []
    stats = {
        "patients_total": len(patient_ids),
        "patients_train": len(train_patients),
        "patients_val": len(val_patients),
        "dropped_no_lung": 0,
        "dropped_small_tumor": 0,
        "dropped_neg_sampled": 0,
    }

    for pid in patient_ids:
        p_dir = cache_dir / pid
        _, msk, meta = load_cached(p_dir, mmap=True)
        num_slices = int(msk.shape[0])

        lung_p = lung_cache_path(p_dir)
        has_lung = lung_p.exists() and (meta.get("lung_ok", True) is not False)
        lung = np.load(lung_p, mmap_mode="r") if has_lung else None

        # Tính diện tích khối u (mm^2) trên từng lát cắt Z (1 pixel ở lưới 1mm = 1 mm^2)
        msk_arr = np.asarray(msk)
        tumor_areas = (msk_arr > 0).reshape(num_slices, -1).sum(axis=1).astype(np.float32)

        if lung is not None and require_lung:
            lung_arr = np.asarray(lung)
            lung_pixels = (lung_arr > 0).reshape(num_slices, -1).sum(axis=1)
        else:
            lung_pixels = np.full(num_slices, 1, dtype=np.int64)

        pos_slices = []
        neg_candidates = []

        for z in range(num_slices):
            area = float(tumor_areas[z])
            has_l = int(lung_pixels[z]) > 0

            if area > 0.0:
                if area < tumor_size_low_lim:
                    stats["dropped_small_tumor"] += 1
                    continue
                pos_slices.append((z, area))
            else:
                if not has_l:
                    stats["dropped_no_lung"] += 1
                    continue
                neg_candidates.append((z, 0.0))

        # Cân bằng lát cắt âm tính (Negative Sampling) theo ZERO_SL_RATIO
        if len(pos_slices) > 0:
            num_neg_keep = max(1, int(round(len(pos_slices) * zero_sl_ratio))) if zero_sl_ratio > 0 else 0
        else:
            num_neg_keep = 1 if len(neg_candidates) > 0 else 0

        if len(neg_candidates) > num_neg_keep:
            kept_neg = sorted(rng.sample(neg_candidates, num_neg_keep), key=lambda x: x[0])
            stats["dropped_neg_sampled"] += len(neg_candidates) - num_neg_keep
        else:
            kept_neg = neg_candidates

        split = "val" if pid in val_patients else "train"
        selected = [(z, 1, area) for z, area in pos_slices] + [(z, 0, area) for z, area in kept_neg]
        selected.sort(key=lambda x: x[0])

        for z, has_tumor, area in selected:
            rows.append({
                "patient_id": pid,
                "z": int(z),
                "num_slices": num_slices,
                "has_tumor": int(has_tumor),
                "tumor_area_mm2": round(float(area), 2),
                "has_lung_mask": int(has_lung),
                "split": split,
            })

    df = pd.DataFrame(rows)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    pos_count = int((df["has_tumor"] == 1).sum()) if len(df) else 0
    neg_count = int((df["has_tumor"] == 0).sum()) if len(df) else 0
    train_count = int((df["split"] == "train").sum()) if len(df) else 0
    val_count = int((df["split"] == "val").sum()) if len(df) else 0

    print("=" * 50)
    print("📊 THỐNG KÊ CHỈ MỤC DỮ LIỆU (OFFLINE BUILDER)")
    print("=" * 50)
    print(f"Tổng số bệnh nhân : {stats['patients_total']} ({stats['patients_train']} Train | {stats['patients_val']} Val)")
    print(f"Tổng lát cắt giữ lại: {len(df)} ({train_count} Train | {val_count} Val)")
    print(f"  + Dương tính (Có u >= {tumor_size_low_lim} mm²): {pos_count}")
    print(f"  + Âm tính (Không u, có phổi)       : {neg_count}")
    print("Các lát cắt đã lọc bỏ:")
    print(f"  - Không chứa nhu mô phổi           : {stats['dropped_no_lung']}")
    print(f"  - Khối u quá nhỏ (< {tumor_size_low_lim} mm²)     : {stats['dropped_small_tumor']}")
    print(f"  - Âm tính dư thừa (Negative Sample): {stats['dropped_neg_sampled']}")
    print(f"✅ Đã xuất bảng chỉ mục tại: {out_csv_path}")
    return df, stats


def _cmd_selftest(_):
    print("Đang chạy kiểm thử tự động offline_builder.py...")
    tmp = Path(tempfile.mkdtemp())
    try:
        for i in range(1, 5):
            p_dir = tmp / f"LIDC-IDRI-000{i}"
            p_dir.mkdir(parents=True)
            img = np.full((10, 100, 100), -800, dtype=np.int16)
            msk = np.zeros((10, 100, 100), dtype=np.uint8)
            lung = np.zeros((10, 100, 100), dtype=np.uint8)

            lung[2:8, 10:90, 10:90] = 1        # Lát 0,1,8,9 không có phổi (4 lát/ca)
            msk[3, 20:22, 20:25] = 1           # Lát 3: u 10 mm² (< 20 mm²) -> phải bị loại
            msk[5, 30:35, 30:35] = 1           # Lát 5: u 25 mm² (>= 20 mm²) -> giữ lại
            msk[6, 30:36, 30:36] = 1           # Lát 6: u 36 mm² (>= 20 mm²) -> giữ lại

            np.save(p_dir / "image_1mm.npy", img)
            np.save(p_dir / "mask_1mm.npy", msk)
            np.save(p_dir / "lung_1mm.npy", lung)
            with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump({"orig_spacing_xyz": [0.7, 0.7, 1.5], "lung_ok": True}, f)

        df, stats = build_dataset_index(tmp)
        assert stats["dropped_no_lung"] == 16, f"Sai lọc phổi: {stats}"
        assert stats["dropped_small_tumor"] == 4, f"Sai lọc u nhỏ: {stats}"
        assert (df["has_tumor"] == 1).sum() == 8, "Phải giữ đúng 8 lát có u hợp lệ"
        assert set(df["split"].unique()) == {"train", "val"}, "Phải có cả tập train và val"
        print("✅ selftest offline_builder: OK!")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Tạo file slices.csv từ thư mục cache_1mm")
    b.add_argument("--cache-dir", default=str(ROOT / "data" / "cache_1mm"))
    b.add_argument("--out-csv", default=None)
    b.add_argument("--no-require-lung", action="store_true", help="Không lọc bỏ lát thiếu phổi")
    b.set_defaults(fn=lambda a: build_dataset_index(a.cache_dir, a.out_csv, require_lung=not a.no_require_lung))

    s = sub.add_parser("selftest", help="Kiểm thử logic bằng dữ liệu giả lập")
    s.set_defaults(fn=_cmd_selftest)

    args = ap.parse_args()
    args.fn(args)