# -*- coding: utf-8 -*-
"""
tests/test_run1_data_pipeline.py - Kiểm thử chuẩn tái lập R0 & tính năng Run 1 (R1_Data_Loss).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.data_pipeline.dataset import LIDC25DDataset, PosAllNegResample, build_dataloaders
from common.data_pipeline.offline_builder import build_dataset_index
from common.engine.base_trainer import BaseTrainer
from common.engine.cli_args import add_common_args, validate_experiment_flags
from configs.config import IMAGE_SIZE
from mamba_unet.activations import DEFAULT_ACTIVATION, list_activations
from mamba_unet.losses import DEFAULT_LOSS, list_losses
from mamba_unet.train_main import _add_mamba_specific_args, train_action


def _make_synthetic_cache(cache_dir: Path, image_size_px: int = 128) -> None:
    """Tạo 4 ca bệnh giả lập: ca 1-3 có u (3 mm² + 9 mm² + 25 mm²), ca 4 không có u."""
    for i in range(1, 5):
        p_dir = cache_dir / f"LIDC-IDRI-000{i}"
        p_dir.mkdir(parents=True, exist_ok=True)
        img = np.full((12, image_size_px, image_size_px), -800, dtype=np.int16)
        msk = np.zeros((12, image_size_px, image_size_px), dtype=np.uint8)
        lung = np.zeros((12, image_size_px, image_size_px), dtype=np.uint8)
        lung[1:11, 10 : image_size_px - 10, 10 : image_size_px - 10] = 1
        if i < 4:
            msk[2, 20:21, 20:23] = 1   # 3 mm² (< 5.0)
            msk[4, 25:28, 25:28] = 1   # 9 mm² (5.0 <= A < 20.0)
            msk[6, 40:45, 40:45] = 1   # 25 mm² (>= 20.0)
        np.save(p_dir / "image_1mm.npy", img)
        np.save(p_dir / "mask_1mm.npy", msk)
        np.save(p_dir / "lung_1mm.npy", lung)
        with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"orig_spacing_xyz": [0.7, 0.7, 1.5], "lung_ok": True}, f)


class TestRun1DataPipeline(unittest.TestCase):
    def test_01_r0_cli_defaults_and_fail_loud(self):
        """Không truyền cờ nào phải giữ đúng mặc định R0; cờ Run 2/3 fail-loud khi bật."""
        parser = argparse.ArgumentParser()
        add_common_args(
            parser,
            activation_choices=list_activations(),
            loss_choices=list_losses(),
            default_activation=DEFAULT_ACTIVATION,
            default_loss=DEFAULT_LOSS,
        )
        _add_mamba_specific_args(parser)
        args = parser.parse_args([])

        self.assertEqual(args.min_train_area, 20.0)
        self.assertEqual(args.neg_ratio, 0.25)
        self.assertFalse(args.dynamic_neg)
        self.assertFalse(args.keep_all_train_neg)
        self.assertEqual(args.min_neg_per_case, 1)
        self.assertEqual(args.weight_decay, 1e-4)
        self.assertEqual(args.ema, 0.0)
        self.assertEqual(args.aug_level, 1)
        self.assertFalse(args.hr_skip)
        self.assertEqual(args.tumor_union_p, 1.0)
        self.assertEqual(IMAGE_SIZE, 512)
        validate_experiment_flags(args)

        for flag_args in (["--ema", "0.999"], ["--aug-level", "2"], ["--hr-skip"]):
            bad = parser.parse_args(flag_args)
            with self.assertRaises(NotImplementedError):
                validate_experiment_flags(bad)

    def test_02_per_run_cache_dir_isolation_and_run1_indexing(self):
        """Mỗi Run nhận cache-dir riêng; Run 1 giữ nốt 9mm², loại 3mm² và cấp >= 4 lát âm ca rỗng."""
        tmp = Path(tempfile.mkdtemp())
        try:
            cache_r0 = tmp / "cache_r0"
            cache_r1 = tmp / "cache_r1"
            _make_synthetic_cache(cache_r0)
            _make_synthetic_cache(cache_r1)

            df_r0, st_r0 = build_dataset_index(cache_r0)
            df_r1, st_r1 = build_dataset_index(
                cache_r1,
                min_train_area=5.0,
                neg_ratio=0.5,
                min_neg_per_case=4,
                keep_all_train_neg=True,
            )

            self.assertNotIn("in_static", df_r0.columns)
            self.assertIn("in_static", df_r1.columns)
            # R0: loại 3mm² và 9mm² (2*3=6 lát); giữ 25mm² (3 lát)
            self.assertEqual((df_r0["has_tumor"] == 1).sum(), 3)
            self.assertEqual(st_r0["dropped_small_tumor"], 6)
            # R1: chỉ loại 3mm² (3 lát); giữ 9mm² + 25mm² (6 lát)
            self.assertEqual((df_r1["has_tumor"] == 1).sum(), 6)
            self.assertEqual(st_r1["dropped_small_tumor"], 3)
            # Ca không u (LIDC-IDRI-0004) được cấp >= 4 lát âm tĩnh
            empty_case = df_r1[df_r1["patient_id"] == "LIDC-IDRI-0004"]
            self.assertGreaterEqual(int((empty_case["in_static"] == 1).sum()), 4)
            # Kho ứng viên âm cho train có cả in_static == 0
            train_r1 = df_r1[df_r1["split"] == "train"]
            self.assertTrue((train_r1["in_static"] == 0).any())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_03_pos_all_neg_resample_and_base_trainer_set_epoch(self):
        """PosAllNegResample giữ 100% mẫu dương, đổi mẫu âm qua mỗi epoch khi BaseTrainer.fit chạy."""
        has_tumor = np.array([1, 1, 1, 1] + [0] * 40)
        sampler = PosAllNegResample(has_tumor, neg_ratio=1.0, min_neg_per_case=4, seed=42)
        sampler.set_epoch(1)
        ep1 = list(sampler)
        sampler.set_epoch(2)
        ep2 = list(sampler)
        pos = {0, 1, 2, 3}
        self.assertTrue(pos.issubset(set(ep1)))
        self.assertTrue(pos.issubset(set(ep2)))
        self.assertEqual(len(ep1), 8)
        self.assertNotEqual(set(ep1) - pos, set(ep2) - pos)

        # Kiểm tra BaseTrainer.fit() tự gọi sampler.set_epoch(epoch)
        class _DummyDS(Dataset):
            def __len__(self):
                return len(has_tumor)

            def __getitem__(self, idx):
                return {"image": torch.randn(3, 16, 16), "mask": torch.zeros(1, 16, 16)}

        seen_epochs = []
        orig_set_epoch = sampler.set_epoch

        def _spy_set_epoch(ep):
            seen_epochs.append(ep)
            orig_set_epoch(ep)

        sampler.set_epoch = _spy_set_epoch
        loader = DataLoader(_DummyDS(), batch_size=4, sampler=sampler)
        model = torch.nn.Conv2d(3, 1, kernel_size=1)
        tmp = Path(tempfile.mkdtemp())
        try:
            trainer = BaseTrainer(
                model=model,
                criterion=torch.nn.BCEWithLogitsLoss(),
                optimizer=torch.optim.SGD(model.parameters(), lr=1e-2),
                save_dir=tmp,
                use_amp=False,
                show_progress=False,
            )
            trainer.fit(loader, loader, epochs=3)
            self.assertEqual(seen_epochs, [1, 2, 3])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_04_p3_lung_mask_sync_outside_lung(self):
        """Khi không union (use_union=False), nhãn u nằm ngoài mask phổi bị xóa đồng bộ."""
        tmp = Path(tempfile.mkdtemp())
        try:
            cache = tmp / "cache"
            p_dir = cache / "LIDC-IDRI-0001"
            p_dir.mkdir(parents=True)
            img = np.full((6, 64, 64), -800, dtype=np.int16)
            msk = np.zeros((6, 64, 64), dtype=np.uint8)
            lung = np.zeros((6, 64, 64), dtype=np.uint8)
            lung[1:5, 20:44, 20:44] = 1
            msk[3, 5:10, 5:10] = 1    # Nằm ngoài phổi
            msk[3, 30:35, 30:35] = 1  # Nằm trong phổi
            np.save(p_dir / "image_1mm.npy", img)
            np.save(p_dir / "mask_1mm.npy", msk)
            np.save(p_dir / "lung_1mm.npy", lung)
            with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump({"orig_spacing_xyz": [0.7, 0.7, 1.5], "lung_ok": True}, f)

            build_dataset_index(cache, min_train_area=5.0, min_neg_per_case=1, val_split=0.0)
            ds_no_union = LIDC25DDataset(
                cache, split="train", image_size=64, tumor_union_p=0.0, augment=False
            )
            pos_row = int(ds_no_union.df[ds_no_union.df["has_tumor"] == 1].index[0])
            sample_mask = ds_no_union[pos_row]["mask"].numpy()[0]
            # Vùng ngoài phổi (5:10, 5:10) phải về 0
            self.assertEqual(float(sample_mask[5:10, 5:10].sum()), 0.0)
            self.assertGreater(float(sample_mask[30:35, 30:35].sum()), 0.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_05_env_image_size_448_and_checkpoint_meta(self):
        """DUNEAI_IMAGE_SIZE=448 được đọc đúng ở configs.config và MambaUNetTrainer.checkpoint_meta()."""
        env = os.environ.copy()
        env["DUNEAI_IMAGE_SIZE"] = "448"
        env["PYTHONPATH"] = str(_ROOT)
        code = (
            "import torch; from configs.config import IMAGE_SIZE; "
            "from mamba_unet.trainer import MambaUNetTrainer; "
            "m = torch.nn.Conv2d(3, 1, 1); "
            "t = MambaUNetTrainer(model=m, criterion=torch.nn.BCEWithLogitsLoss(), show_progress=False); "
            "meta = t.checkpoint_meta(); "
            "assert IMAGE_SIZE == 448; assert meta['image_size'] == 448; "
            "assert meta['model_config']['image_size'] == 448; print('OK_448')"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(_ROOT), env=env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OK_448", proc.stdout)

    def test_06_run_config_json_records_all_flags(self):
        """run_config.json phải ghi đủ mọi cờ thử nghiệm và DUNEAI_IMAGE_SIZE."""
        tmp = Path(tempfile.mkdtemp())
        try:
            cache = tmp / "cache_r0"
            runs_root = tmp / "runs"
            _make_synthetic_cache(cache)

            parser = argparse.ArgumentParser()
            add_common_args(
                parser,
                activation_choices=list_activations(),
                loss_choices=list_losses(),
                default_activation=DEFAULT_ACTIVATION,
                default_loss=DEFAULT_LOSS,
            )
            _add_mamba_specific_args(parser)
            args = parser.parse_args(
                [
                    "--cache-dir", str(cache),
                    "--runs-root", str(runs_root),
                    "--run-name", "test_r0_parity",
                    "--epochs", "1",
                    "--batch-size", "2",
                    "--workers", "0",
                    "--rebuild-index",
                    "--no-amp",
                    "--no-progress",
                ]
            )

            from mamba_unet.backbone.mamba_vision import MambaVision
            from mamba_unet.model import MambaUNet

            def _tiny_build(**_kwargs):
                bb = MambaVision(
                    dim=16, in_dim=8, depths=[1, 1, 2, 1],
                    window_size=[8, 8, 14, 7], num_heads=[2, 2, 4, 4], in_chans=3,
                )
                return MambaUNet(in_channels=3, num_classes=1, backbone=bb, act="relu")

            # Dùng ảnh 64 cho bước chạy nhanh trên CPU
            with patch("mamba_unet.train_main.build_mamba_unet", side_effect=_tiny_build):
                orig_dl = build_dataloaders

                def _small_dl(*dl_args, **dl_kwargs):
                    dl_kwargs["cache_dir"] = cache
                    tr, va = orig_dl(*dl_args, **dl_kwargs)
                    tr.dataset.image_size = 64
                    va.dataset.image_size = 64
                    return tr, va

                with patch("common.data_pipeline.dataset.build_dataloaders", side_effect=_small_dl):
                    train_action(args)

            cfg_file = runs_root / "test_r0_parity" / "run_config.json"
            self.assertTrue(cfg_file.is_file())
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            for key in (
                "min_train_area", "neg_ratio", "min_neg_per_case",
                "dynamic_neg", "keep_all_train_neg", "tumor_union_p",
                "weight_decay", "ema", "aug_level", "hr_skip", "duneai_image_size",
            ):
                self.assertIn(key, cfg["args"], f"Thiếu {key} trong run_config['args']")
            self.assertEqual(cfg["data"]["image_size"], 512)
            self.assertEqual(cfg["data"]["min_train_area"], 20.0)
            self.assertEqual(cfg["data"]["neg_ratio"], 0.25)
            self.assertFalse(cfg["data"]["dynamic_neg"])
            self.assertFalse(cfg["model"]["hr_skip"])
            self.assertEqual(cfg["optimizer"]["ema"], 0.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
