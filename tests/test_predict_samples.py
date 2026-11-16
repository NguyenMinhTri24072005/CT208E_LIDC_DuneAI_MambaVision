# -*- coding: utf-8 -*-
"""
tests/test_predict_samples.py - Kiểm thử nhẹ & nhanh trên CPU 2 nhân cho luồng Predict mẫu thật.

Phạm vi test hợp lý cho máy local (CPU 2 nhân, KHÔNG tải weights nặng, KHÔNG chạy khối 448x448):
  1. Kiểm tra cơ chế đọc slices*.csv của đợt train, phân loại TRAIN / VAL / UNSEEN và chặn rò rỉ tập train.
  2. Kiểm tra bộ tiêu chí lọc ngưỡng: confidence threshold, đường kính < 5mm / < 10mm, diện tích, thể tích, per_slice vs 3d_component.
  3. Kiểm tra toán học % chuẩn từng lát cắt (TP, FP, FN, Slice Dice, Pooled Dice trên các lát Predict) & 3D metrics (Dice, ASSD, HD95).
  4. Kiểm tra CLI end-to-end (prepare từ thư mục/zip giả + predict với tiny MambaUNet 64x64 + vẽ ảnh PNG & xuất báo cáo).
  5. Kiểm tra hợp chuẩn nbformat v4 của notebook `mamba_unet/notebooks/colab_predict.ipynb` và quy tắc R1.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.predict_real_samples import (
    FilterCriteria,
    build_predict_out_subdir,
    compute_slice_and_patient_stats,
    discover_run_splits,
    filter_and_analyze_components,
    find_run_artifacts,
    render_patient_visualizations,
    resolve_model_metadata,
    run_predict_pipeline,
    scan_available_patients,
    select_non_train_patients,
    stage_model_to_workspace,
    stage_patient_samples,
)


def _create_fake_patient(cache_dir: Path, pid: str, shape=(10, 64, 64), tumor_slices=None):
    """Tạo nhanh dữ liệu bệnh nhân 1mm giả lập nhẹ cho CPU test."""
    p = cache_dir / pid
    p.mkdir(parents=True, exist_ok=True)
    z, h, w = shape
    img = np.full(shape, -800, dtype=np.int16)
    msk = np.zeros(shape, dtype=np.uint8)
    lung = np.zeros(shape, dtype=np.uint8)
    lung[1 : z - 1, 8 : h - 8, 8 : w - 8] = 1
    if tumor_slices:
        for zs, r0, r1, c0, c1 in tumor_slices:
            msk[zs, r0:r1, c0:c1] = 1
            img[zs, r0:r1, c0:c1] = 40
    np.save(p / "image_1mm.npy", img)
    np.save(p / "mask_1mm.npy", msk)
    np.save(p / "lung_1mm.npy", lung)
    (p / "meta.json").write_text(
        json.dumps({"orig_shape_zyx": list(shape), "orig_spacing_xyz": [0.8, 0.8, 1.5], "lung_ok": True}),
        encoding="utf-8",
    )


class TestPredictRealSamples(unittest.TestCase):
    def test_train_val_unseen_leak_prevention(self):
        """Phải tách đúng TRAIN / VAL / UNSEEN từ slices.csv đợt train và chặn rò rỉ tập train."""
        tmp = Path(tempfile.mkdtemp())
        try:
            slices_csv = tmp / "slices_r1c_static.csv"
            pd.DataFrame([
                {"patient_id": "LIDC-IDRI-0001", "z": 3, "num_slices": 10, "has_tumor": 1, "split": "train"},
                {"patient_id": "LIDC-IDRI-0002", "z": 4, "num_slices": 10, "has_tumor": 1, "split": "val"},
                {"patient_id": "LIDC-IDRI-0003", "z": 5, "num_slices": 10, "has_tumor": 0, "split": "val"},
            ]).to_csv(slices_csv, index=False)

            split_info = discover_run_splits(slices_csv)
            avail = ["LIDC-IDRI-0001", "LIDC-IDRI-0002", "LIDC-IDRI-0003", "LIDC-IDRI-0501", "LIDC-IDRI-0502"]
            hints = {"LIDC-IDRI-0002": True, "LIDC-IDRI-0003": False, "LIDC-IDRI-0501": True}

            sel_val, st_val = select_non_train_patients(avail, split_info, source_mode="val", selection_mode="first", num_samples=10)
            self.assertEqual(sel_val, ["LIDC-IDRI-0002", "LIDC-IDRI-0003"])
            self.assertEqual(st_val["LIDC-IDRI-0002"], "VAL")

            sel_unseen, st_unseen = select_non_train_patients(avail, split_info, source_mode="unseen", selection_mode="first", num_samples=10)
            self.assertEqual(sel_unseen, ["LIDC-IDRI-0501", "LIDC-IDRI-0502"])
            self.assertEqual(st_unseen["LIDC-IDRI-0501"], "UNSEEN")

            sel_mixed, _ = select_non_train_patients(
                avail, split_info, source_mode="non_train", selection_mode="mixed", num_samples=4, tumor_hints=hints
            )
            self.assertNotIn("LIDC-IDRI-0001", sel_mixed)
            self.assertIn("LIDC-IDRI-0003", sel_mixed)

            with self.assertRaises(ValueError):
                select_non_train_patients(avail, split_info, source_mode="custom", custom_patient_ids=["LIDC-IDRI-0001"])

            sel_leak, st_leak = select_non_train_patients(
                avail, split_info, source_mode="custom", custom_patient_ids=["LIDC-IDRI-0001"], allow_train_leak=True
            )
            self.assertEqual(sel_leak, ["LIDC-IDRI-0001"])
            self.assertEqual(st_leak["LIDC-IDRI-0001"], "TRAIN")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_diameter_under_5mm_and_10mm_and_threshold_filtering(self):
        """Kiểm tra tiêu chí lọc độ tin cậy, đường kính < 5mm / < 10mm và per_slice."""
        shape = (10, 64, 64)
        prob = np.zeros(shape, dtype=np.float32)
        gt = np.zeros(shape, dtype=np.uint8)
        lung = np.ones(shape, dtype=np.uint8)

        # Cụm A (< 5mm): 3x3 = 9 mm² -> d ≈ 3.39 mm, max_prob = 0.85
        prob[2, 10:13, 10:13] = 0.85
        gt[2, 10:13, 10:13] = 1
        # Cụm B (5-10mm): 6x6 = 36 mm² -> d ≈ 6.77 mm, max_prob = 0.90 (rìa lát z=3 có 2x2=4 mm² -> d≈2.26mm)
        prob[4, 24:30, 24:30] = 0.90
        prob[3, 25:27, 25:27] = 0.90
        gt[4, 24:30, 24:30] = 1
        # Cụm C (>= 10mm): 11x11 = 121 mm² -> d ≈ 12.41 mm, max_prob = 0.95
        prob[6, 40:51, 40:51] = 0.95
        gt[6, 40:51, 40:51] = 1
        # Nhiễu mờ dưới ngưỡng xác suất: 0.42
        prob[8, 15:22, 40:47] = 0.42

        # 1. Lấy tất cả >= 0.5 (kể cả dưới 5mm)
        r_all = filter_and_analyze_components(
            prob, gt, lung, spacing_z=1.5,
            criteria=FilterCriteria(threshold=0.5, min_diameter_mm=0.0, min_area_mm2=0.0),
        )
        self.assertEqual(len([c for c in r_all["pred_components"] if c["kept"]]), 3)

        # 2. Loại nốt dưới 5mm (giữ B và C, loại A)
        r_5 = filter_and_analyze_components(
            prob, gt, lung, spacing_z=1.5,
            criteria=FilterCriteria(threshold=0.5, min_diameter_mm=5.0, min_area_mm2=0.0),
        )
        kept_5 = [c for c in r_5["pred_components"] if c["kept"]]
        rej_5 = [c for c in r_5["pred_components"] if not c["kept"]]
        self.assertEqual(len(kept_5), 2)
        self.assertEqual(len(rej_5), 1)
        self.assertIn("Diam=", rej_5[0]["filter_reason"])
        self.assertEqual(r_5["clean_mask_1mm"][2].sum(), 0)

        # 3. Loại nốt dưới 10mm (chỉ giữ C)
        r_10 = filter_and_analyze_components(
            prob, gt, lung, spacing_z=1.5,
            criteria=FilterCriteria(threshold=0.5, min_diameter_mm=10.0, min_area_mm2=0.0),
        )
        kept_10 = [c for c in r_10["pred_components"] if c["kept"]]
        self.assertEqual(len(kept_10), 1)
        self.assertGreater(kept_10[0]["max_diam_mm"], 10.0)

        # 4. Lọc theo phạm vi từng lát (per_slice) cắt bỏ lát z=3 (d=2.26mm) của cụm B nhưng giữ z=4 (d=6.77mm)
        r_slice = filter_and_analyze_components(
            prob, gt, lung, spacing_z=1.5,
            criteria=FilterCriteria(threshold=0.5, min_diameter_mm=5.0, min_area_mm2=0.0, filter_scope="per_slice"),
        )
        self.assertEqual(r_slice["clean_mask_1mm"][3].sum(), 0)
        self.assertEqual(r_slice["clean_mask_1mm"][4].sum(), 36)

    def test_slice_comparison_math_and_metrics(self):
        """Kiểm tra chính xác % chuẩn từng lát và tổng hợp trên tập lát Predict."""
        shape = (8, 32, 32)
        gt = np.zeros(shape, dtype=np.uint8)
        pred = np.zeros(shape, dtype=np.uint8)
        prob = np.zeros(shape, dtype=np.float32)

        # z=3: Trùng hoàn toàn 10x10=100 px -> Dice=100%
        gt[3, 5:15, 5:15] = 1
        pred[3, 5:15, 5:15] = 1
        prob[3, 5:15, 5:15] = 0.9

        # z=4: GT 10x10=100, Pred 10x10=100 nhưng lệch 5 cột -> TP=50 -> Dice=50%
        gt[4, 5:15, 5:15] = 1
        pred[4, 5:15, 10:20] = 1
        prob[4, 5:15, 10:20] = 0.8

        # z=5: Chỉ Pred 4x10=40 -> FP
        pred[5, 5:9, 5:15] = 1
        prob[5, 5:9, 5:15] = 0.7

        # z=6: Chỉ GT 4x10=40 -> FN
        gt[6, 5:9, 5:15] = 1

        filt = filter_and_analyze_components(
            prob, gt, np.ones(shape, dtype=np.uint8), spacing_z=1.0,
            criteria=FilterCriteria(threshold=0.5, min_diameter_mm=0.0, min_area_mm2=0.0),
        )
        p_sum, s_rows = compute_slice_and_patient_stats(
            pid="LIDC-TEST-0001",
            split_status="VAL (Chưa train)",
            prob_1mm=prob,
            raw_mask_1mm=pred,
            clean_mask_1mm=filt["clean_mask_1mm"],
            gt_mask_1mm=gt,
            gt_nodules=filt["gt_nodules"],
            pred_components=filt["pred_components"],
            meta={"orig_spacing_xyz": [1.0, 1.0, 1.0]},
            elapsed_sec=0.1,
        )
        self.assertEqual(len(s_rows), 4)
        self.assertEqual(p_sum["n_gt_slices"], 3)
        self.assertEqual(p_sum["n_pred_slices"], 3)
        self.assertEqual(p_sum["n_pred_hit_slices"], 2)
        # Pooled trên 3 lát pred (z=3,4,5): TP=150, Pred=240, GT=200 -> 2*150/(240+200)*100 = 68.18%
        self.assertAlmostEqual(p_sum["pooled_dice_on_pred_slices_pct"], 68.18, places=2)
        self.assertEqual(s_rows[0]["slice_dice_pct"], 100.0)
        self.assertEqual(s_rows[1]["slice_dice_pct"], 50.0)
        self.assertTrue(np.isfinite(p_sum["assd_mm"]))

    def test_stage_from_zip_and_cli_predict_end_to_end(self):
        """Kiểm thử end-to-end nạp từ zip giả + predict CLI bằng tiny model trên CPU."""
        tmp = Path(tempfile.mkdtemp())
        try:
            drive_src = tmp / "drive"
            run_dir = drive_src / "mamba_unet" / "runs" / "tiny_e2e"
            raw_cache = tmp / "raw_cache"
            run_dir.mkdir(parents=True)
            _create_fake_patient(raw_cache, "LIDC-IDRI-0001", tumor_slices=[(slice(3, 6), 24, 34, 24, 34)])
            _create_fake_patient(raw_cache, "LIDC-IDRI-0002", tumor_slices=[(slice(3, 6), 24, 34, 24, 34)])
            _create_fake_patient(raw_cache, "LIDC-IDRI-0003", tumor_slices=None)

            zip_p = drive_src / "data" / "cache_1mm.zip"
            zip_p.parent.mkdir(parents=True)
            with zipfile.ZipFile(zip_p, "w") as zf:
                for f in raw_cache.rglob("*"):
                    if f.is_file():
                        zf.write(f, arcname=Path("cache_1mm") / f.relative_to(raw_cache))

            pd.DataFrame([
                {"patient_id": "LIDC-IDRI-0001", "z": 4, "num_slices": 10, "has_tumor": 1, "split": "train"},
                {"patient_id": "LIDC-IDRI-0002", "z": 4, "num_slices": 10, "has_tumor": 1, "split": "val"},
            ]).to_csv(run_dir / "slices_r1c_static.csv", index=False)

            from unet_base.model import UNetBase

            model = UNetBase(in_channels=3, classes=1, features=8, act="relu")
            ckpt_p = run_dir / "best_model.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": {"arch": "unet_base", "features": 8, "image_size": 64, "activation": "relu"},
                },
                ckpt_p,
            )
            (run_dir / "run_config.json").write_text(
                json.dumps({"run_id": "tiny_e2e", "arch": "unet_base", "model": {"image_size": 64, "features": 8}}),
                encoding="utf-8",
            )

            dest_cache = tmp / "pred_cache"
            model_ws = tmp / "ws_model"
            out_dir = tmp / "out"
            drive_out = tmp / "drive_out"

            cmd_prep = [
                sys.executable, str(_ROOT / "tools" / "predict_real_samples.py"), "prepare",
                "--run-dir", str(run_dir),
                "--model-dest", str(model_ws),
                "--data-source", str(zip_p),
                "--dest-cache", str(dest_cache),
                "--source", "non_train",
                "--select", "first",
                "--num-samples", "5",
                "--workers", "1",
            ]
            r1 = subprocess.run(cmd_prep, capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(r1.returncode, 0, msg=f"prepare failed:\n{r1.stdout}\n{r1.stderr}")

            manifest_p = dest_cache / "predict_manifest.json"
            self.assertTrue(manifest_p.exists())
            man = json.loads(manifest_p.read_text(encoding="utf-8"))
            self.assertEqual(man["selected_ids"], ["LIDC-IDRI-0002", "LIDC-IDRI-0003"])
            self.assertNotIn("LIDC-IDRI-0001", man["selected_ids"])

            cmd_pred = [
                sys.executable, str(_ROOT / "tools" / "predict_real_samples.py"), "predict",
                "--manifest", str(manifest_p),
                "--patient-ids", "LIDC-IDRI-0002",
                "--no-cuda", "--no-amp", "--no-progress",
                "--batch-size", "2",
                "--threshold", "0.5",
                "--min-diameter-mm", "5.0",
                "--out-dir", str(out_dir),
                "--auto-subdir",
                "--drive-out-dir", str(drive_out),
            ]
            r2 = subprocess.run(cmd_pred, capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(r2.returncode, 0, msg=f"predict failed:\n{r2.stdout}\n{r2.stderr}")

            # Chạy tiếp ca khác vào cùng base out_dir với --auto-subdir -> KHÔNG được ghi đè!
            cmd_pred_3 = [
                sys.executable, str(_ROOT / "tools" / "predict_real_samples.py"), "predict",
                "--manifest", str(manifest_p),
                "--patient-ids", "LIDC-IDRI-0003",
                "--no-cuda", "--no-amp", "--no-progress",
                "--batch-size", "2",
                "--threshold", "0.5",
                "--min-diameter-mm", "5.0",
                "--out-dir", str(out_dir),
                "--auto-subdir",
                "--drive-out-dir", str(drive_out),
            ]
            r3 = subprocess.run(cmd_pred_3, capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(r3.returncode, 0, msg=f"predict 2nd run failed:\n{r3.stdout}\n{r3.stderr}")

            local_subdirs = sorted(d for d in out_dir.iterdir() if d.is_dir())
            drive_subdirs = sorted(d for d in drive_out.iterdir() if d.is_dir())
            self.assertEqual(len(local_subdirs), 2, f"Phải có 2 folder riêng biệt, thấy: {local_subdirs}")
            self.assertEqual(len(drive_subdirs), 2, f"Drive phải có 2 folder riêng biệt, thấy: {drive_subdirs}")
            self.assertTrue(any(d.name.startswith("LIDC-IDRI-0002_") for d in local_subdirs))
            self.assertTrue(any(d.name.startswith("LIDC-IDRI-0003_") for d in local_subdirs))
            self.assertTrue((out_dir / "latest_predict_dir.txt").is_file())

            sub_0002 = next(d for d in local_subdirs if d.name.startswith("LIDC-IDRI-0002_"))
            for fname in ("predict_patients_summary.csv", "predict_nodules_detail.csv", "predict_slices_detail.csv", "predict_report.json", "PREDICT_REPORT.md"):
                self.assertTrue((sub_0002 / fname).is_file(), f"Thiếu {fname}")
            self.assertTrue(any((sub_0002 / "figures").glob("LIDC-IDRI-0002_*.png")))

            # Kiểm tra naming helper cho 1 ca, 2 ca và >3 ca
            crit = FilterCriteria(threshold=0.5, min_diameter_mm=5.0)
            self.assertEqual(build_predict_out_subdir(["LIDC-IDRI-0051"], crit, "20261116_120000"), "LIDC-IDRI-0051_th0.5_d5_20261116_120000")
            self.assertEqual(build_predict_out_subdir(["LIDC-IDRI-0005", "LIDC-IDRI-0051"], crit, "20261116_120000"), "LIDC-IDRI-0005_0051_th0.5_d5_20261116_120000")
            self.assertEqual(build_predict_out_subdir(["LIDC-IDRI-0005", "LIDC-IDRI-0051", "LIDC-IDRI-0060", "LIDC-IDRI-0089"], crit, "20261116_120000"), "4ca_LIDC-IDRI-0005_to_0089_th0.5_d5_20261116_120000")

            # Kiểm tra resolve_model_metadata cho Mamba-UNet thực tế (448x448, EMA)
            mamba_dir = tmp / "mamba_meta_test"
            mamba_dir.mkdir(parents=True)
            mamba_ckpt = mamba_dir / "best_model.pth"
            torch.save(
                {"model_state_dict": {}, "is_ema": True, "epoch": 28, "best_val_dice": 0.7242, "meta": {"arch": "mamba_unet", "backbone": "mamba_vision_T", "activation": "relu", "image_size": 448}},
                mamba_ckpt,
            )
            m_info = resolve_model_metadata(mamba_ckpt)
            self.assertEqual(m_info["arch"], "mamba_unet")
            self.assertEqual(m_info["backbone"], "mamba_vision_T")
            self.assertEqual(m_info["image_size"], 448)
            self.assertTrue(m_info["is_ema"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_colab_predict_notebook_valid_and_complete(self):
        """Kiểm tra colab_predict.ipynb hợp lệ nbformat 4 và có đủ các bước theo yêu cầu."""
        nb_path = _ROOT / "mamba_unet" / "notebooks" / "colab_predict.ipynb"
        self.assertTrue(nb_path.is_file(), f"Không tìm thấy {nb_path}")
        data = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(data.get("nbformat"), 4)
        self.assertGreaterEqual(len(data["cells"]), 10)

        src = "\n".join("".join(c.get("source", [])) for c in data["cells"])
        for token in (
            "torch==2.6.0",
            "git clone",
            "git pull origin",
            "tools/predict_real_samples.py prepare",
            "tools/predict_real_samples.py predict",
            "THRESHOLD =",
            "MIN_DIAMETER_MM =",
            "MIN_AREA_MM2 =",
            "NUM_SAMPLES =",
            "SAMPLE_SOURCE =",
            "PATIENT_IDS =",
            "--auto-subdir",
            "latest_predict_dir.txt",
            "display(Image(",
        ):
            self.assertIn(token, src, f"Notebook thiếu thành phần: {token}")
        self.assertNotIn("unet_base", src, "Notebook trong mamba_unet/notebooks/ phải tách biệt khỏi unet_base (R1)")

        chk = subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "make_predict_notebook.py"), "--check"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        self.assertEqual(chk.returncode, 0, msg=chk.stdout + "\n" + chk.stderr)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    unittest.main()
