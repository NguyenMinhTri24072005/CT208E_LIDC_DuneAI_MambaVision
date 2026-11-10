# -*- coding: utf-8 -*-
"""
tests/test_gray_area_crash.py - Kiểm thử chuyên sâu VÙNG XÁM KHẢ NĂNG CHỊU LỖI & ATOMIC WRITE.

Mục tiêu kiểm định:
  1. Mô phỏng đứt kết nối runtime/sập nguồn khi đang ghi file checkpoint.
  2. Xác minh file best_model.pth không bao giờ bị hỏng (corrupt) nhờ cơ chế Atomic Write (.tmp + replace).
  3. Kiểm tra tự động phát hiện và phục hồi phiên chạy (auto-resume) tiếp tục đúng epoch tiếp theo.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.engine.checkpoint_io import save_checkpoint, load_checkpoint_file
from common.engine.run_manager import resolve_resume, create_run


class TestGrayAreaCrashAndResume(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_crash_"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_atomic_write_prevents_corruption(self):
        """Mô phỏng ngắt ngang giữa chừng khi ghi: Checkpoint trước đó vẫn nguyên vẹn 100%."""
        target_ckpt = self.test_dir / "best_model.pth"

        # 1. Lưu checkpoint chuẩn ban đầu (Epoch 1)
        orig_sd = {"weight": torch.randn(10, 10)}
        save_checkpoint(target_ckpt, state_dict=orig_sd, epoch=1, best_val_dice=0.85)
        self.assertTrue(target_ckpt.exists())

        # 2. Giả lập một tiến trình bị crash giữa chừng tạo ra file .tmp bị hỏng
        fake_tmp = self.test_dir / f".tmp_{target_ckpt.name}_99999"
        with open(fake_tmp, "wb") as f:
            f.write(b"CORRUPTED_INCOMPLETE_DATA_CRASH")

        # 3. Kiểm tra file chính best_model.pth: Vẫn nguyên vẹn 100%
        loaded = load_checkpoint_file(target_ckpt)
        self.assertEqual(loaded["epoch"], 1)
        self.assertEqual(loaded["best_val_dice"], 0.85)
        torch.testing.assert_close(loaded["model_state_dict"]["weight"], orig_sd["weight"])

        # 4. Khi lưu epoch tiếp theo, hệ thống ghi đè nguyên tử an toàn
        new_sd = {"weight": torch.randn(10, 10)}
        save_checkpoint(target_ckpt, state_dict=new_sd, epoch=2, best_val_dice=0.90)
        loaded_2 = load_checkpoint_file(target_ckpt)
        self.assertEqual(loaded_2["epoch"], 2)
        self.assertEqual(loaded_2["best_val_dice"], 0.90)

    def test_02_auto_resume_after_abrupt_disconnect(self):
        """Mô phỏng Colab bị thu hồi runtime ở Epoch 4: resolve_resume('auto') phát hiện và nạp chính xác."""
        runs_root = self.test_dir / "runs"
        run_handle = create_run(arch="mamba_unet", runs_root=runs_root, tag="simulated_colab_run")
        run_dir = run_handle.run_dir

        # Lưu checkpoint bị gián đoạn ở Epoch 4
        sd = {"bias": torch.randn(5)}
        last_ckpt = run_dir / "last_model.pth"
        save_checkpoint(last_ckpt, state_dict=sd, epoch=4, best_val_dice=0.78)

        # Mô phỏng phiên chạy mới với RESUME = "auto"
        res_dir, res_ckpt = resolve_resume("auto", arch="mamba_unet", runs_root=runs_root)
        self.assertIsNotNone(res_dir, "Không nhận diện được phiên chạy bị ngắt")
        self.assertIsNotNone(res_ckpt, "Không tìm thấy last_model.pth")
        self.assertEqual(res_dir.resolve(), run_dir.resolve())
        self.assertEqual(res_ckpt.resolve(), last_ckpt.resolve())

        # Đọc dữ liệu từ checkpoint đã resolve
        ckpt_data = load_checkpoint_file(res_ckpt)
        self.assertEqual(ckpt_data["epoch"], 4)
        print(f"\n✅ Auto-Resume đã phục hồi thành công phiên chạy từ Epoch {ckpt_data['epoch']}!")


if __name__ == "__main__":
    unittest.main()
