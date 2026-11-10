# -*- coding: utf-8 -*-
"""
scripts/prepare_dataset.py - Công cụ CLI chuẩn bị dữ liệu và xây dựng chỉ mục lát cắt (slices.csv).

Các tác vụ:
  1. build-index: Quét thư mục cache_1mm, tính diện tích khối u từng lát cắt,
                  chia tập Train/Val cấp độ bệnh nhân (Patient-level split) và xuất slices.csv.
  2. runtime    : Thiết lập dữ liệu phù hợp với runtime (Colab, Kaggle, Local).
  3. selftest   : Kiểm thử tự động trên dữ liệu giả lập.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import SEED, TUMOR_SIZE_LOW_LIM, VAL_SPLIT, ZERO_SL_RATIO
from common.data_pipeline.offline_builder import build_dataset_index
from common.data_pipeline.runtime_data import run_auto, run_colab, run_kaggle, run_local


def cmd_build_index(args):
    cache_dir = Path(args.cache_dir).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else (cache_dir / "slices.csv")
    print(f"📦 Đang xây dựng chỉ mục lát cắt slices.csv từ: {cache_dir}")
    df = build_dataset_index(
        cache_dir=cache_dir,
        out_csv_path=out_csv,
        tumor_size_low_lim=args.tumor_size_low_lim,
        zero_sl_ratio=args.zero_sl_ratio,
        val_split=args.val_split,
        seed=args.seed,
        require_lung=not args.no_lung,
    )
    print(f"✅ Hoàn tất! Đã lưu {len(df)} lát cắt tại: {out_csv}")


def cmd_runtime(args):
    if args.platform == "colab":
        run_colab(
            zip_path=args.zip_path,
            dest_dir=args.dest_dir,
            n_patients=args.n_patients,
            select=args.select,
            seed=args.seed,
        )
    elif args.platform == "kaggle":
        run_kaggle(
            hint=args.hint,
            view_dir=args.view_dir,
            n_patients=args.n_patients,
            select=args.select,
            seed=args.seed,
        )
    elif args.platform == "local":
        run_local(cache_dir=args.cache_dir)
    elif args.platform == "auto":
        run_auto(
            hint_kaggle=args.hint,
            zip_colab=args.zip_path,
            dest_colab=args.dest_dir,
            local_cache=args.cache_dir,
            n_patients=args.n_patients,
            select=args.select,
            seed=args.seed,
        )


def cmd_selftest(_):
    print("🧪 Bắt đầu selftest scripts.prepare_dataset...")
    tmp = Path(tempfile.mkdtemp())
    try:
        p_dir = tmp / "LIDC-IDRI-0001"
        p_dir.mkdir(parents=True)
        img = np.full((10, 64, 64), -800, dtype=np.int16)
        msk = np.zeros((10, 64, 64), dtype=np.uint8)
        lung = np.zeros((10, 64, 64), dtype=np.uint8)
        lung[1:9, 10:54, 10:54] = 1
        msk[4:7, 20:30, 20:30] = 1  # 100 mm2 > 20 mm2

        np.save(p_dir / "image_1mm.npy", img)
        np.save(p_dir / "mask_1mm.npy", msk)
        np.save(p_dir / "lung_1mm.npy", lung)
        with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "orig_shape_zyx": [10, 64, 64],
                "orig_spacing_xyz": [0.8, 0.8, 1.5],
                "shape_1mm_zyx": [10, 64, 64],
                "lung_ok": True,
            }, f)

        out_csv = tmp / "slices.csv"
        df = build_dataset_index(cache_dir=tmp, out_csv_path=out_csv, val_split=0.0)
        assert out_csv.exists()
        assert len(df) > 0
        print("✅ Selftest scripts.prepare_dataset hoàn tất thành công!")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Công cụ chuẩn bị dữ liệu DuneAI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # 1. build-index
    build_p = subparsers.add_parser("build-index", help="Quét cache_1mm và tạo slices.csv")
    build_p.add_argument("--cache-dir", default="data/cache_1mm")
    build_p.add_argument("--out-csv", default=None)
    build_p.add_argument("--tumor-size-low-lim", type=float, default=TUMOR_SIZE_LOW_LIM)
    build_p.add_argument("--zero-sl_ratio", type=float, default=ZERO_SL_RATIO)
    build_p.add_argument("--val-split", type=float, default=VAL_SPLIT)
    build_p.add_argument("--seed", type=int, default=SEED)
    build_p.add_argument("--no-lung", action="store_true")
    build_p.set_defaults(func=cmd_build_index)

    # 2. runtime
    run_p = subparsers.add_parser("runtime", help="Đồng bộ dữ liệu runtime cho Colab/Kaggle/Local")
    run_p.add_argument("--platform", choices=["colab", "kaggle", "local", "auto"], default="auto")
    run_p.add_argument("--zip-path", default=None, help="Đường dẫn file zip cache (Colab)")
    run_p.add_argument("--dest-dir", default="/content/data/cache_1mm", help="Thư mục giải nén (Colab)")
    run_p.add_argument("--hint", default="/kaggle/input", help="Đường dẫn tìm kiếm dataset (Kaggle)")
    run_p.add_argument("--view-dir", default="/kaggle/working/data/cache_1mm", help="Thư mục symlink (Kaggle)")
    run_p.add_argument("--cache-dir", default="data/cache_1mm", help="Thư mục cục bộ (Local)")
    run_p.add_argument("--n-patients", type=int, default=0, help="Số bệnh nhân cần lấy (0 = toàn bộ)")
    run_p.add_argument("--select", choices=["first", "random"], default="first")
    run_p.add_argument("--seed", type=int, default=42)
    run_p.set_defaults(func=cmd_runtime)

    # 3. selftest
    self_p = subparsers.add_parser("selftest", help="Kiểm thử tự động")
    self_p.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
