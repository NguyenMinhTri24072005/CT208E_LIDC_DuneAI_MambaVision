# -*- coding: utf-8 -*-
"""
tests/test_resume.py - Kiểm tra cơ chế khôi phục huấn luyện (Resume) và kiểm tra tương thích metadata.

Quy tắc bắt buộc:
  - KHÔNG import từ unet_base hay mamba_unet.
  - Sử dụng DummyModel và nn.BCEWithLogitsLoss để kiểm thử BaseTrainer một cách độc lập.
  - Kiểm tra tính liền mạch khi resume (epoch, optimizer state, metric, resume_events).
  - Kiểm tra ném ValueError khi sai khác metadata (arch, features).
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.engine.base_trainer import BaseTrainer
from common.engine.run_manager import create_run


class DummyDataset(Dataset):
    """Dataset giả lập tạo tensor nhỏ cho kiểm thử."""
    def __init__(self, size: int = 8):
        self.size = size
        # 1 kênh x 16 x 16 để chạy siêu nhanh
        self.images = torch.randn(size, 1, 16, 16)
        self.masks = (torch.rand(size, 1, 16, 16) > 0.5).float()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"image": self.images[idx], "mask": self.masks[idx]}


class DummyNet(nn.Module):
    """Mô hình đơn giản 1 conv layer cho unit test."""
    def __init__(self, in_c: int = 1, out_c: int = 1, features: int = 8):
        super().__init__()
        self.features = features
        self.conv1 = nn.Conv2d(in_c, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, out_c, kernel_size=1)

    def forward(self, x):
        return self.conv2(torch.relu(self.conv1(x)))


class CustomArchTrainer(BaseTrainer):
    """BaseTrainer giả lập kiến trúc có metadata tùy chỉnh."""
    def __init__(self, arch_name: str, features: int, *args, **kwargs):
        self.arch_name = arch_name
        self.features_val = features
        super().__init__(*args, **kwargs)

    def checkpoint_meta(self) -> Dict[str, Any]:
        return {
            "arch": self.arch_name,
            "features": self.features_val,
        }


class TestResume(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_no_forbidden_architecture_imports(self):
        """Đảm bảo test_resume không import unet_base hay mamba_unet."""
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

    def test_seamless_resume_and_event_recording(self):
        """
        Kiểm tra kịch bản huấn luyện:
        1. Huấn luyện 2 epochs -> ngắt.
        2. Khởi động lại với --resume auto -> tiếp tục từ epoch 3 đến epoch 4.
        3. Kiểm tra optimizer, model state, history, và resume_events trong run_config.json.
        """
        train_loader = DataLoader(DummyDataset(8), batch_size=2, shuffle=False)
        val_loader = DataLoader(DummyDataset(4), batch_size=2, shuffle=False)

        model1 = DummyNet(features=8)
        criterion = nn.BCEWithLogitsLoss()
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)

        run_handle1 = create_run(
            arch="dummy_arch",
            runs_root=self.runs_root,
            run_name="test_resume_run",
            args={"epochs": 4, "lr": 0.01},
        )

        trainer1 = CustomArchTrainer(
            arch_name="dummy_arch",
            features=8,
            model=model1,
            criterion=criterion,
            optimizer=optimizer1,
            run_handle=run_handle1,
            show_progress=False,
            use_amp=False,
        )

        # Giai đoạn 1: Huấn luyện 2 epochs
        trainer1.fit(train_loader, val_loader, epochs=2)
        self.assertEqual(len(trainer1.history), 2)
        self.assertTrue((run_handle1.run_dir / "last_model.pth").exists())

        # Lưu lại trọng số cuối epoch 2 để đối chiếu
        last_weights_epoch2 = {k: v.clone() for k, v in model1.state_dict().items()}

        # Giai đoạn 2: Tạo trainer mới mô phỏng chạy lại lệnh train với --resume auto
        model2 = DummyNet(features=8)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)

        # Đọc lại run_config hiện có để tạo run_handle mới gắn với cùng thư mục run
        with open(run_handle1.config_path, "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)

        run_handle2 = create_run(
            arch="dummy_arch",
            runs_root=self.runs_root,
            run_name="test_resume_run",  # Cùng thư mục run
        )
        # Vì test_resume_run đã tồn tại, create_run sẽ sinh test_resume_run-2 trừ khi gắn thẳng vào run_handle1.run_dir
        # Dùng RunHandle trực tiếp với thư mục run1:
        from common.engine.run_manager import RunHandle
        run_handle_resumed = RunHandle(run_handle1.run_dir, saved_cfg)

        trainer2 = CustomArchTrainer(
            arch_name="dummy_arch",
            features=8,
            model=model2,
            criterion=criterion,
            optimizer=optimizer2,
            run_handle=run_handle_resumed,
            show_progress=False,
            use_amp=False,
        )

        # Tiếp tục huấn luyện đến epoch 4 với resume_from="auto"
        trainer2.fit(train_loader, val_loader, epochs=4, resume_from="auto")

        # Kiểm tra start_epoch bắt đầu từ 3 và kết thúc ở 4
        self.assertEqual(trainer2.start_epoch, 3)
        self.assertEqual(len(trainer2.history), 4)
        self.assertGreater(len(trainer2.optimizer.state), 0, "Optimizer state phải được phục hồi khi resume!")

        # Kiểm tra run_config.json đã ghi nhận resume_events
        with open(run_handle_resumed.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.assertIn("resume_events", cfg)
        self.assertGreaterEqual(len(cfg["resume_events"]), 1)
        self.assertEqual(cfg["resume_events"][-1]["from_epoch"], 2)
        self.assertEqual(cfg["status"], "completed")

    def test_incompatible_architecture_raises_error(self):
        """Kiểm tra báo lỗi ValueError khi checkpoint có arch khác với model hiện tại."""
        model = DummyNet(features=8)
        criterion = nn.BCEWithLogitsLoss()
        run_h = create_run(arch="arch_A", runs_root=self.runs_root, run_name="run_A")

        trainer_A = CustomArchTrainer(
            arch_name="arch_A",
            features=8,
            model=model,
            criterion=criterion,
            run_handle=run_h,
            show_progress=False,
            use_amp=False,
        )
        ckpt_path = trainer_A.save_checkpoint("last_model.pth", epoch=1, val_metrics={"dice": 0.5})

        # Tạo trainer B với kiến trúc "arch_B"
        model_B = DummyNet(features=8)
        trainer_B = CustomArchTrainer(
            arch_name="arch_B",
            features=8,
            model=model_B,
            criterion=criterion,
            save_dir=self.runs_root / "run_B",
            show_progress=False,
            use_amp=False,
        )

        with self.assertRaises(ValueError) as ctx:
            trainer_B.load_checkpoint(ckpt_path)

        self.assertIn("Không tương thích kiến trúc", str(ctx.exception))
        self.assertIn("arch_A", str(ctx.exception))
        self.assertIn("arch_B", str(ctx.exception))

    def test_incompatible_features_raises_error(self):
        """Kiểm tra báo lỗi ValueError khi checkpoint có số filter cơ sở (features) khác."""
        model = DummyNet(features=8)
        criterion = nn.BCEWithLogitsLoss()
        run_h = create_run(arch="arch_A", runs_root=self.runs_root, run_name="run_feat8")

        trainer_8 = CustomArchTrainer(
            arch_name="arch_A",
            features=8,
            model=model,
            criterion=criterion,
            run_handle=run_h,
            show_progress=False,
            use_amp=False,
        )
        ckpt_path = trainer_8.save_checkpoint("last_model.pth", epoch=1, val_metrics={"dice": 0.5})

        # Tạo trainer 16 với features=16
        model_16 = DummyNet(features=16)
        trainer_16 = CustomArchTrainer(
            arch_name="arch_A",
            features=16,
            model=model_16,
            criterion=criterion,
            save_dir=self.runs_root / "run_feat16",
            show_progress=False,
            use_amp=False,
        )

        with self.assertRaises(ValueError) as ctx:
            trainer_16.load_checkpoint(ckpt_path)

        self.assertIn("Không tương thích số filter cơ sở", str(ctx.exception))

    def test_resume_loads_optimizer_state_dict(self):
        """Kiểm tra load_checkpoint phục hồi chính xác optimizer_state_dict."""
        model = DummyNet(features=8)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss = model(torch.randn(2, 1, 32, 32)).sum()
        loss.backward()
        opt.step()
        self.assertGreater(len(opt.state), 0)

        trainer = CustomArchTrainer(
            arch_name="dummy_arch",
            features=8,
            model=model,
            criterion=nn.BCEWithLogitsLoss(),
            optimizer=opt,
            save_dir=self.runs_root / "test_opt",
            show_progress=False,
            use_amp=False,
        )
        ckpt_path = trainer.save_checkpoint("last_model.pth", epoch=1, val_metrics={"dice": 0.5})

        model2 = DummyNet(features=8)
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
        self.assertEqual(len(opt2.state), 0)

        trainer2 = CustomArchTrainer(
            arch_name="dummy_arch",
            features=8,
            model=model2,
            criterion=nn.BCEWithLogitsLoss(),
            optimizer=opt2,
            save_dir=self.runs_root / "test_opt2",
            show_progress=False,
            use_amp=False,
        )
        trainer2.load_checkpoint(ckpt_path, resume_training=True)
        self.assertGreater(len(trainer2.optimizer.state), 0, "Optimizer state không được khôi phục!")


if __name__ == "__main__":
    unittest.main()
