# -*- coding: utf-8 -*-
"""
tests/test_run_folder.py - Kiểm tra cấu trúc thư mục Run và schema run_config.json (v1).

Quy tắc bắt buộc:
  - KHÔNG import từ unet_base hay mamba_unet.
  - Sử dụng thư mục tạm (tempfile.TemporaryDirectory) để độc lập hoàn toàn.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.engine.run_manager import (
    create_run,
    find_resumable,
    generate_run_id,
    resolve_resume,
    RunHandle,
)
from common.engine.run_config import SCHEMA_VERSION


class TestRunFolder(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_no_forbidden_architecture_imports(self):
        """Đảm bảo test_run_folder không import unet_base hay mamba_unet."""
        import ast
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in ["unet_base", "mamba_unet"]:
                        self.assertNotIn(forbidden, alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for forbidden in ["unet_base", "mamba_unet"]:
                        self.assertNotIn(forbidden, node.module)

    def test_create_run_and_schema_v1(self):
        """Kiểm tra khởi tạo Run và cấu trúc JSON chuẩn Schema v1."""
        arch = "dummy_arch"
        args = {"epochs": 10, "lr": 1e-3, "batch_size": 4}
        model_meta = {"name": "DummyNet", "features": 16}
        loss_meta = {"name": "dummy_bce"}

        run_handle = create_run(
            arch=arch,
            runs_root=self.runs_root,
            run_name="test_run_01",
            args=args,
            model_meta=model_meta,
            loss_meta=loss_meta,
            notes="Testing schema v1",
        )

        self.assertTrue(run_handle.run_dir.exists(), "Thư mục run không tồn tại!")
        self.assertTrue(run_handle.config_path.exists(), "run_config.json không tồn tại!")

        with open(run_handle.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Kiểm tra các khóa bắt buộc cấp cao nhất
        required_keys = [
            "run_id",
            "schema_version",
            "arch",
            "started_at",
            "status",
            "git",
            "environment",
            "args",
            "model",
            "data",
            "loss",
            "optimizer",
            "scheduler",
            "training",
            "result",
        ]
        for key in required_keys:
            self.assertIn(key, cfg, f"Thiếu trường '{key}' trong run_config.json!")

        self.assertEqual(cfg["schema_version"], SCHEMA_VERSION)
        self.assertEqual(cfg["arch"], arch)
        self.assertEqual(cfg["status"], "running")
        self.assertEqual(cfg["args"]["lr"], 1e-3)
        self.assertEqual(cfg["model"]["name"], "DummyNet")
        self.assertEqual(cfg["loss"]["name"], "dummy_bce")

    def test_duplicate_run_name_collision(self):
        """Đảm bảo nếu trùng run_name sẽ tự động thêm hậu tố -2, -3."""
        run1 = create_run(arch="dummy", runs_root=self.runs_root, run_name="same_name")
        run2 = create_run(arch="dummy", runs_root=self.runs_root, run_name="same_name")
        self.assertEqual(run1.run_id, "same_name")
        self.assertEqual(run2.run_id, "same_name-2")
        self.assertTrue(run1.run_dir.exists())
        self.assertTrue(run2.run_dir.exists())

    def test_logging(self):
        """Kiểm tra ghi log vào train.log."""
        run_handle = create_run(arch="dummy", runs_root=self.runs_root)
        run_handle.log("Hello world message", echo=False)

        self.assertTrue(run_handle.log_path.exists())
        with open(run_handle.log_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Hello world message", content)

    def test_atomic_result_update_and_finish(self):
        """Kiểm tra cập nhật tiến độ (update_result) và kết thúc run (finish)."""
        run_handle = create_run(arch="dummy", runs_root=self.runs_root)

        # Cập nhật epoch 1
        run_handle.update_result(
            epoch=1,
            best_val_dice=0.75,
            best_epoch=1,
            last_val_loss=0.35,
            last_val_dice=0.75,
            last_val_iou=0.60,
        )

        with open(run_handle.config_path, "r", encoding="utf-8") as f:
            cfg1 = json.load(f)
        self.assertEqual(cfg1["result"]["epochs_completed"], 1)
        self.assertAlmostEqual(cfg1["result"]["best_val_dice"], 0.75)
        self.assertEqual(cfg1["status"], "running")

        # Cập nhật epoch 2
        run_handle.update_result(
            epoch=2,
            best_val_dice=0.82,
            best_epoch=2,
            last_val_loss=0.25,
            last_val_dice=0.82,
            last_val_iou=0.70,
        )

        with open(run_handle.config_path, "r", encoding="utf-8") as f:
            cfg2 = json.load(f)
        self.assertEqual(cfg2["result"]["epochs_completed"], 2)
        self.assertAlmostEqual(cfg2["result"]["best_val_dice"], 0.82)

        # Kết thúc run
        run_handle.finish(status="completed", extra_metrics={"final_score": 0.99})

        with open(run_handle.config_path, "r", encoding="utf-8") as f:
            cfg_final = json.load(f)
        self.assertEqual(cfg_final["status"], "completed")
        self.assertIn("finished_at", cfg_final)
        self.assertEqual(cfg_final["result"]["final_score"], 0.99)

    def test_find_resumable_and_resolve_resume(self):
        """Kiểm tra cơ chế phát hiện run dở dang và giải quyết tham số --resume."""
        # Tạo run 1: đã hoàn thành
        r1 = create_run(arch="test_arch", runs_root=self.runs_root, run_name="run_done")
        (r1.run_dir / "last_model.pth").write_text("dummy checkpoint")
        r1.finish(status="completed")

        # Tạo run 2: còn dở dang (running)
        r2 = create_run(arch="test_arch", runs_root=self.runs_root, run_name="run_in_progress")
        (r2.run_dir / "last_model.pth").write_text("dummy checkpoint")

        # find_resumable phải tìm thấy r2 chứ không phải r1
        resumable = find_resumable("test_arch", runs_root=self.runs_root)
        self.assertIsNotNone(resumable)
        self.assertEqual(resumable.name, "run_in_progress")

        # find_resumable với kiến trúc khác phải trả về None
        none_res = find_resumable("other_arch", runs_root=self.runs_root)
        self.assertIsNone(none_res)

        # resolve_resume 'auto'
        res_dir, res_ckpt = resolve_resume("auto", arch="test_arch", runs_root=self.runs_root)
        self.assertEqual(res_dir, r2.run_dir)
        self.assertEqual(res_ckpt, r2.run_dir / "last_model.pth")

        # resolve_resume theo đường dẫn thư mục
        d_dir, d_ckpt = resolve_resume(str(r2.run_dir), arch="test_arch", runs_root=self.runs_root)
        self.assertEqual(d_dir, r2.run_dir)
        self.assertEqual(d_ckpt, r2.run_dir / "last_model.pth")

        # resolve_resume theo file cụ thể
        f_dir, f_ckpt = resolve_resume(str(r2.run_dir / "last_model.pth"), arch="test_arch", runs_root=self.runs_root)
        self.assertEqual(f_dir, r2.run_dir)
        self.assertEqual(f_ckpt, r2.run_dir / "last_model.pth")


if __name__ == "__main__":
    unittest.main()
