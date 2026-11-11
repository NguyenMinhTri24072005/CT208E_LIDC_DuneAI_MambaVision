# -*- coding: utf-8 -*-
"""
tests/test_auto_sync_drive.py - Kiểm thử cơ chế Auto-Sync Google Drive thời gian thực.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class TestAutoSyncDrive(unittest.TestCase):
    def setUp(self):
        self.tmp_drive = Path(tempfile.mkdtemp(prefix="test_mock_drive_"))

    def tearDown(self):
        shutil.rmtree(self.tmp_drive, ignore_errors=True)

    def test_overfit_with_drive_sync(self):
        """Chạy 2 epoch overfit với cờ --drive-sync-dir và kiểm tra các tệp được đồng bộ."""
        cmd = [
            sys.executable,
            "-m",
            "mamba_unet.train_main",
            "overfit",
            "--epochs",
            "2",
            "--drive-sync-dir",
            str(self.tmp_drive),
        ]
        res = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Lệnh train thất bại: {res.stderr}\n{res.stdout}")

        # Kiểm tra thư mục run đã được tự động tạo trong mock drive
        synced_runs = [p for p in self.tmp_drive.iterdir() if p.is_dir()]
        self.assertGreater(len(synced_runs), 0, "Không tìm thấy thư mục run nào trên Drive!")

        run_dir = synced_runs[0]
        synced_files = [p.name for p in run_dir.iterdir()]
        print(f"📁 Tệp xuất hiện trong mock drive ({run_dir.name}): {synced_files}")

        # Kiểm tra các tệp bắt buộc phải có
        self.assertIn("DATA_SPEC_AND_CONFIG.md", synced_files)
        self.assertIn("REPORT.md", synced_files)
        self.assertIn("history.csv", synced_files)
        self.assertIn("last_model.pth", synced_files)


if __name__ == "__main__":
    unittest.main()
