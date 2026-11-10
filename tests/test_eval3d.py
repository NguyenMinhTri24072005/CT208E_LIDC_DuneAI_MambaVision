# -*- coding: utf-8 -*-
"""
tests/test_eval3d.py - Kiểm thử toàn diện pipeline đánh giá 3D và hậu xử lý hình thái học.

Quy tắc kiểm thử (spec mục 7, 10):
  1. unpad_or_uncrop_xy nghịch đảo chính xác tuyệt đối pad_or_crop_xy trên cả numpy và torch.
  2. resize_from_1mm_3d đưa về đúng shape gốc, bảo toàn tính nhị phân của mặt nạ.
  3. postprocess_3d_mask: lọc sạch nhiễu ngoài phổi, nhiễu diện tích < 20mm², lấp lỗ rỗng.
  4. compute_3d_metrics: tính toán chuẩn mực Dice, IoU, HD95 mm trên các ca chuẩn và ca biên (rỗng).
  5. nrrd_io: lưu và đọc lại đúng dữ liệu voxel và spacing.
  6. volume_inference: predict_patient_volume chạy chuẩn xác trên volume giả lập.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.data_pipeline.medical_prep import pad_or_crop_xy
from common.eval3d.eval_runner import evaluate_dataset_3d
from common.eval3d.metrics3d import _compute_hd95_mm, compute_3d_metrics
from common.eval3d.nrrd_io import load_nrrd_mask, save_prediction_nrrd
from common.eval3d.postprocess3d import (
    postprocess_3d_mask,
    resize_from_1mm_3d,
    unpad_or_uncrop_xy,
)
from common.eval3d.volume_inference import predict_patient_volume


class TestPostprocess3D(unittest.TestCase):
    def test_unpad_or_uncrop_inversion(self):
        """unpad_or_uncrop_xy phải là nghịch đảo chính xác của pad_or_crop_xy."""
        for hw in [(360, 380), (512, 512), (580, 600)]:
            arr = np.zeros((4, hw[0], hw[1]), dtype=np.uint8)
            ch, cw = hw[0] // 2, hw[1] // 2
            arr[1:3, ch - 20 : ch + 20, cw - 20 : cw + 20] = 1

            # Kiểm tra với numpy
            padded_np = pad_or_crop_xy(arr, size=512, pad_value=0)
            unpadded_np = unpad_or_uncrop_xy(padded_np, hw, pad_value=0)
            self.assertTrue(np.array_equal(arr, unpadded_np))

            # Kiểm tra với torch.Tensor
            t_arr = torch.from_numpy(arr)
            padded_t = pad_or_crop_xy(t_arr, size=512, pad_value=0)
            unpadded_t = unpad_or_uncrop_xy(padded_t, hw, pad_value=0)
            self.assertTrue(torch.equal(t_arr, unpadded_t))

    def test_resize_from_1mm_3d(self):
        """resize_from_1mm_3d đưa về đúng shape gốc và giữ nguyên tính nhị phân."""
        vol = np.zeros((5, 200, 200), dtype=np.uint8)
        vol[2, 50:100, 50:100] = 1
        resized = resize_from_1mm_3d(vol, (5, 512, 512), is_mask=True)
        self.assertEqual(resized.shape, (5, 512, 512))
        self.assertTrue(set(np.unique(resized)).issubset({0, 1}))

    def test_postprocess_3d_mask_filtering(self):
        """postprocess_3d_mask lọc đúng nhiễu diện tích < 20mm2 và ngoài phổi."""
        gt = np.zeros((10, 64, 64), dtype=np.uint8)
        lung = np.zeros((10, 64, 64), dtype=np.uint8)
        lung[1:9, 10:54, 10:54] = 1

        # Khối u thật: 8x8 = 64 mm2 (> 20mm2) ở các lát 3, 4, 5
        gt[3:6, 25:33, 25:33] = 1

        noisy = gt.astype(np.float32) * 0.95
        # Lỗ thủng ở giữa khối u
        noisy[4, 28:30, 28:30] = 0.1
        # Nhiễu nhỏ trong phổi: 3x3 = 9 mm2 (< 20mm2) -> phải bị lọc bỏ
        noisy[2, 15:18, 15:18] = 0.9
        # Nhiễu to nằm ngoài phổi -> phải bị cắt bỏ
        noisy[0, 10:40, 10:40] = 0.95

        cleaned = postprocess_3d_mask(
            noisy, lung_mask_zyx=lung, threshold=0.5, min_slice_area_mm2=20.0, fill_holes=True
        )
        self.assertTrue(np.array_equal(cleaned, gt), "Hậu xử lý 3D phải khôi phục 100% khối u gốc!")


class TestMetrics3D(unittest.TestCase):
    def test_perfect_overlap(self):
        """Trùng khớp 100% -> Dice=1, IoU=1, HD95=0."""
        mask = np.zeros((8, 32, 32), dtype=np.uint8)
        mask[2:6, 10:20, 10:20] = 1
        m = compute_3d_metrics(mask, mask, spacing_z=1.5)
        self.assertAlmostEqual(m["dice_3d"], 1.0, places=5)
        self.assertAlmostEqual(m["iou_3d"], 1.0, places=5)
        self.assertEqual(m["hd95_mm"], 0.0)

    def test_both_empty(self):
        """Cả hai cùng rỗng -> Dice=1, IoU=1, HD95=0."""
        empty = np.zeros((8, 32, 32), dtype=np.uint8)
        m = compute_3d_metrics(empty, empty)
        self.assertEqual(m["dice_3d"], 1.0)
        self.assertEqual(m["hd95_mm"], 0.0)

    def test_one_empty(self):
        """Một bên rỗng -> Dice=0, HD95=nan."""
        mask = np.zeros((8, 32, 32), dtype=np.uint8)
        mask[2:6, 10:20, 10:20] = 1
        empty = np.zeros_like(mask)
        m = compute_3d_metrics(mask, empty)
        self.assertEqual(m["dice_3d"], 0.0)
        self.assertTrue(np.isnan(m["hd95_mm"]))

    def test_hd95_calculation(self):
        """Kiểm tra khoảng cách HD95 khi dịch chuyển khối u 3 pixel."""
        m1 = np.zeros((8, 32, 32), dtype=np.uint8)
        m1[2:6, 10:20, 10:20] = 1
        m2 = np.zeros_like(m1)
        m2[2:6, 13:23, 10:20] = 1  # dịch 3 pixel dọc theo trục Y
        m = compute_3d_metrics(m1, m2, spacing_z=1.0)
        self.assertGreater(m["dice_3d"], 0.0)
        self.assertLess(m["dice_3d"], 1.0)
        self.assertAlmostEqual(m["hd95_mm"], 3.0, delta=0.5)


class TestNRRDIO(unittest.TestCase):
    def test_save_and_load_nrrd(self):
        """Kiểm tra lưu và nạp lại file mặt nạ NRRD."""
        tmp = Path(tempfile.mkdtemp())
        try:
            mask = np.zeros((6, 32, 32), dtype=np.uint8)
            mask[2:4, 10:20, 10:20] = 1
            meta = {
                "orig_shape_zyx": [6, 32, 32],
                "orig_spacing_xyz": [0.75, 0.75, 1.25],
            }
            out_file = tmp / "test_mask.nrrd"
            save_prediction_nrrd(mask, meta, out_file)
            self.assertTrue(out_file.exists())

            loaded_arr, spacing = load_nrrd_mask(out_file)
            self.assertTrue(np.array_equal(loaded_arr, mask))
            self.assertAlmostEqual(spacing[0], 0.75, places=3)
            self.assertAlmostEqual(spacing[2], 1.25, places=3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestVolumeInference(unittest.TestCase):
    def test_predict_patient_volume_and_eval_dataset(self):
        """Kiểm tra toàn trình hàm predict_patient_volume và evaluate_dataset_3d."""
        tmp = Path(tempfile.mkdtemp())
        try:
            cache_dir = tmp / "cache_1mm"
            p_dir = cache_dir / "LIDC-IDRI-0001"
            p_dir.mkdir(parents=True)

            img = np.full((8, 32, 32), -800, dtype=np.int16)
            msk = np.zeros((8, 32, 32), dtype=np.uint8)
            lung = np.zeros((8, 32, 32), dtype=np.uint8)
            lung[1:7, 5:27, 5:27] = 1
            msk[3:5, 12:20, 12:20] = 1
            img[3:5, 12:20, 12:20] = 50

            np.save(p_dir / "image_1mm.npy", img)
            np.save(p_dir / "mask_1mm.npy", msk)
            np.save(p_dir / "lung_1mm.npy", lung)
            with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump({
                    "orig_shape_zyx": [8, 32, 32],
                    "orig_spacing_xyz": [0.8, 0.8, 1.5],
                    "shape_1mm_zyx": [8, 32, 32],
                    "lung_ok": True,
                }, f)

            # Tạo model giả lập nn.Sequential
            class DummyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 1, kernel_size=1)
                    # Khởi tạo bias âm lớn để mô phỏng dự đoán nền
                    self.conv.bias.data.fill_(-4.0)

                def forward(self, x):
                    return self.conv(x)

            model = DummyModel()
            res = predict_patient_volume(
                model=model,
                patient_cache_dir=p_dir,
                batch_size=2,
                image_size=32,
                use_amp=False,
            )
            self.assertEqual(res["prob_1mm"].shape, (8, 32, 32))
            self.assertEqual(res["clean_mask_1mm"].shape, (8, 32, 32))
            self.assertIn("dice_3d", res["metrics_raw"])
            self.assertIn("dice_3d", res["metrics_post"])

            # Kiểm tra evaluate_dataset_3d
            csv_out = tmp / "results.csv"
            json_out = tmp / "summary.json"
            df = evaluate_dataset_3d(
                model=model,
                cache_dir=cache_dir,
                split="all",
                batch_size=2,
                out_csv=csv_out,
                out_json=json_out,
                use_amp=False,
            )
            self.assertEqual(len(df), 1)
            self.assertTrue(csv_out.exists())
            self.assertTrue(json_out.exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    unittest.main()
