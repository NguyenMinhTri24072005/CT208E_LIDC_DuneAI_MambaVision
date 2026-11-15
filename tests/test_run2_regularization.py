# -*- coding: utf-8 -*-
"""
tests/test_run2_regularization.py - Kiểm thử toàn diện cho Giai đoạn Run 2:
  1. Kiểm tra CLI flags validation (--ema, --aug-level, --hr-skip) & build_run_data_meta.
  2. Kiểm tra 4-group Weight Decay phân tách decoder/backbone x decay/no_decay.
  3. Kiểm tra ModelEMA (plain class, warmup, eval trên EMA, save best EMA, resume last EMA).
  4. Kiểm tra cập nhật tham số Run 2 vào DATA_SPEC_AND_CONFIG.md & run_config.json.
  5. Kiểm tra Augmentation Level 2 (Flip/Rot90 + Affine + Intensity jitter & bảo toàn nốt u nhỏ).
  6. Kiểm tra đồng bộ notebook colab_train.ipynb & colab_step0_measurement.ipynb.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.data_pipeline.dataset import LIDC25DDataset, build_dataloaders
from common.data_pipeline.offline_builder import build_dataset_index
from common.engine.base_trainer import BaseTrainer, ModelEMA
from common.engine.checkpoint_io import load_checkpoint_file
from common.engine.cli_args import add_common_args, build_run_data_meta, validate_experiment_flags
from common.engine.run_manager import RunHandle, create_run
from mamba_unet.activations import list_activations
from mamba_unet.backbone.mamba_vision import MambaVision
from mamba_unet.losses import get_loss, list_losses
from mamba_unet.model import MambaUNet
from mamba_unet.train_main import Synthetic25DDataset
from mamba_unet.trainer import MambaUNetTrainer


def _make_tiny_mamba_unet():
    bb = MambaVision(
        dim=16,
        in_dim=8,
        depths=[1, 1, 2, 1],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 2, 4, 4],
        in_chans=3,
    )
    return MambaUNet(in_channels=3, num_classes=1, backbone=bb, act="relu")


class TestRun2Regularization(unittest.TestCase):
    def test_cli_flags_validation_and_meta(self):
        parser = argparse.ArgumentParser()
        add_common_args(
            parser,
            activation_choices=list_activations(),
            loss_choices=list_losses(),
            default_activation="relu",
            default_loss="dice_bce",
        )
        args = parser.parse_args([
            "--cache-dir", "data/cache_1mm",
            "--ema", "0.999",
            "--aug-level", "2",
            "--weight-decay", "0.02",
        ])
        validate_experiment_flags(args)
        meta = build_run_data_meta(args, Path("data/cache_1mm"), 448)
        self.assertEqual(meta["aug_level"], 2)
        self.assertAlmostEqual(args.ema, 0.999)
        self.assertAlmostEqual(args.weight_decay, 0.02)

        for bad_ema in ("-0.1", "1.0", "1.5"):
            bad_args = parser.parse_args(["--cache-dir", "data/cache_1mm", "--ema", bad_ema])
            with self.assertRaises(ValueError):
                validate_experiment_flags(bad_args)

        for bad_aug in ("0", "3"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--cache-dir", "data/cache_1mm", "--aug-level", bad_aug])
            with self.assertRaises(ValueError):
                validate_experiment_flags(argparse.Namespace(ema=0.0, aug_level=int(bad_aug), hr_skip=False))

        args.hr_skip = True
        with self.assertRaises(NotImplementedError):
            validate_experiment_flags(args)

    def test_four_group_weight_decay_split(self):
        model = _make_tiny_mamba_unet()
        criterion = get_loss("dice_bce")
        trainer = MambaUNetTrainer(
            model=model,
            criterion=criterion,
            freeze_backbone_epochs=3,
            backbone_lr_mult=0.1,
            weight_decay=0.02,
            show_progress=False,
            use_amp=False,
        )

        # Ngay cả khi gọi ở epoch 1 (backbone tạm đóng băng), 4 nhóm vẫn phải được đăng ký đủ
        trainer.on_epoch_start(1)
        groups = trainer.build_param_groups(1e-4, weight_decay=0.02)
        by_name = {g["name"]: g for g in groups}
        self.assertEqual(
            set(by_name.keys()),
            {"decoder_decay", "decoder_no_decay", "backbone_decay", "backbone_no_decay"},
        )
        self.assertAlmostEqual(by_name["decoder_decay"]["lr"], 1e-4)
        self.assertAlmostEqual(by_name["decoder_decay"]["weight_decay"], 0.02)
        self.assertAlmostEqual(by_name["decoder_no_decay"]["lr"], 1e-4)
        self.assertEqual(by_name["decoder_no_decay"]["weight_decay"], 0.0)
        self.assertAlmostEqual(by_name["backbone_decay"]["lr"], 1e-5)
        self.assertAlmostEqual(by_name["backbone_decay"]["weight_decay"], 0.02)
        self.assertAlmostEqual(by_name["backbone_no_decay"]["lr"], 1e-5)
        self.assertEqual(by_name["backbone_no_decay"]["weight_decay"], 0.0)

        grouped_ids = {id(p) for g in groups for p in g["params"]}
        all_ids = {id(p) for p in model.parameters()}
        self.assertEqual(grouped_ids, all_ids)

        # Kiểm tra A_log, bias, và 1D norm nằm trong *_no_decay
        no_decay_ids = {id(p) for p in by_name["decoder_no_decay"]["params"]} | {
            id(p) for p in by_name["backbone_no_decay"]["params"]
        }
        for name, p in model.named_parameters():
            if p.ndim < 2 or name.endswith(".bias") or name.endswith("A_log"):
                self.assertIn(id(p), no_decay_ids, f"Tham số '{name}' lẽ ra phải nằm trong no_decay!")

        # Nếu đóng băng vĩnh viễn (freeze=-1) và đã on_epoch_start(1), backbone không vào Optimizer
        trainer_frozen = MambaUNetTrainer(
            model=_make_tiny_mamba_unet(),
            criterion=criterion,
            freeze_backbone_epochs=-1,
            backbone_lr_mult=0.1,
            weight_decay=0.02,
            show_progress=False,
            use_amp=False,
        )
        trainer_frozen.on_epoch_start(1)
        frozen_groups = trainer_frozen.build_param_groups(1e-4)
        self.assertEqual({g["name"] for g in frozen_groups}, {"decoder_decay", "decoder_no_decay"})

    def test_ema_training_eval_checkpoint_resume_and_metadata(self):
        tmp_root = Path(tempfile.mkdtemp())
        drive_dir = tmp_root / "drive_sync"
        try:
            train_loader = DataLoader(Synthetic25DDataset(num_samples=4, image_size=64), batch_size=2)
            val_loader = DataLoader(Synthetic25DDataset(num_samples=2, image_size=64), batch_size=2)

            run_h = create_run(
                arch="mamba_unet",
                runs_root=tmp_root,
                run_name="r2_ema_test",
                args={"epochs": 3, "lr": 1e-3, "weight_decay": 0.02, "ema": 0.99, "aug_level": 2},
                data_meta={"aug_level": 2, "min_train_area": 5.0, "neg_ratio": 0.35, "dynamic_neg": False, "image_size": 448},
                loss_meta={"name": "dice_bce", "params": {"bce_weight": 1.0, "dice_weight": 1.0, "pos_weight": 2.0}},
                optimizer_meta={"name": "AdamW", "lr": 1e-3, "weight_decay": 0.02, "ema": 0.99, "freeze_backbone_epochs": 1, "backbone_lr_mult": 0.1},
            )

            model1 = _make_tiny_mamba_unet()
            criterion = get_loss("dice_bce")
            trainer1 = MambaUNetTrainer(
                model=model1,
                criterion=criterion,
                backbone_name="tiny",
                freeze_backbone_epochs=1,
                backbone_lr_mult=0.1,
                weight_decay=0.02,
                ema_decay=0.99,
                run_handle=run_h,
                show_progress=False,
                use_amp=False,
                drive_sync_dir=drive_dir,
            )
            self.assertIsNotNone(trainer1.ema)
            self.assertNotIsInstance(trainer1.ema, torch.nn.Module)
            trainer1.optimizer = torch.optim.AdamW(trainer1.build_param_groups(1e-3, weight_decay=0.02))

            trainer1.fit(train_loader, val_loader, epochs=2)
            self.assertEqual(trainer1.ema.num_updates, 4)  # 2 epochs x 2 batches

            # Kiểm tra best_model.pth lưu EMA weights, còn last_model.pth lưu cả ema_state_dict
            best_ckpt = load_checkpoint_file(run_h.run_dir / "best_model.pth", map_location="cpu")
            last_ckpt = load_checkpoint_file(run_h.run_dir / "last_model.pth", map_location="cpu")
            self.assertTrue(best_ckpt.get("is_ema", False))
            self.assertIn("ema_state_dict", last_ckpt)
            self.assertEqual(int(last_ckpt["ema_updates"]), 4)

            # Kiểm tra DATA_SPEC_AND_CONFIG.md ghi đủ cấu hình Run 2
            spec_text = (run_h.run_dir / "DATA_SPEC_AND_CONFIG.md").read_text(encoding="utf-8")
            self.assertIn("aug_level = 2", spec_text)
            self.assertIn("EMA Decay", spec_text)
            self.assertIn("0.99", spec_text)
            self.assertIn("0.02", spec_text)

            # Kiểm tra Resume khôi phục ema_updates & ema_state_dict chính xác
            with open(run_h.config_path, "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
            run_h_res = RunHandle(run_h.run_dir, saved_cfg)
            model2 = _make_tiny_mamba_unet()
            trainer2 = MambaUNetTrainer(
                model=model2,
                criterion=criterion,
                backbone_name="tiny",
                freeze_backbone_epochs=1,
                backbone_lr_mult=0.1,
                weight_decay=0.02,
                ema_decay=0.99,
                run_handle=run_h_res,
                show_progress=False,
                use_amp=False,
            )
            trainer2.optimizer = torch.optim.AdamW(trainer2.build_param_groups(1e-3, weight_decay=0.02))
            trainer2.fit(train_loader, val_loader, epochs=3, resume_from="auto")
            self.assertEqual(trainer2.ema.num_updates, 6)  # 4 cũ + 2 mới ở epoch 3

            # Kiểm tra ema_decay=0.0 vẫn tắt EMA như Run 0/Run 1
            trainer_no_ema = MambaUNetTrainer(
                model=_make_tiny_mamba_unet(),
                criterion=criterion,
                ema_decay=0.0,
                save_dir=tmp_root / "no_ema",
                show_progress=False,
                use_amp=False,
            )
            self.assertIsNone(trainer_no_ema.ema)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_aug_level2_shape_background_and_small_nodule_preservation(self):
        # 1. Kiểm tra bảo toàn nốt u nhỏ ngay cả khi ép dịch chuyển cực đại 24px
        img = np.zeros((3, 128, 128), dtype=np.float32)
        img[:, 20:108, 20:108] = 0.4  # vùng phổi
        msk = np.zeros((1, 128, 128), dtype=np.float32)
        msk[0, 62:65, 62:65] = 1.0  # nốt 3x3 = 9 px

        for _ in range(25):
            aug_img, aug_msk = LIDC25DDataset._augment_pair_level2(img.copy(), msk.copy())
            self.assertEqual(aug_img.shape, (3, 128, 128))
            self.assertEqual(aug_msk.shape, (1, 128, 128))
            self.assertEqual(aug_img.dtype, np.float32)
            self.assertGreaterEqual(float(aug_img.min()), 0.0)
            self.assertLessEqual(float(aug_img.max()), 1.0)
            self.assertTrue(set(np.unique(aug_msk).tolist()).issubset({0.0, 1.0}))
            self.assertGreaterEqual(float(aug_msk.sum()), 1.0, "Không được làm biến mất hoàn toàn nốt u nhỏ!")

        # 2. Kiểm tra rollback khi Affine đẩy nốt u 2x2 ở rìa ra ngoài ảnh
        edge_img = np.full((3, 64, 64), 0.5, dtype=np.float32)
        edge_msk = np.zeros((1, 64, 64), dtype=np.float32)
        edge_msk[0, 1:3, 1:3] = 1.0  # 4 px sát mép trên-trái
        with mock.patch("common.data_pipeline.dataset.random.random", side_effect=[0.9, 0.9, 0.1, 0.9]), \
             mock.patch("common.data_pipeline.dataset.random.randint", return_value=0), \
             mock.patch("common.data_pipeline.dataset.random.uniform", side_effect=[0.0, 1.0, -24.0, -24.0]):
            res_img, res_msk = LIDC25DDataset._augment_pair_level2(edge_img.copy(), edge_msk.copy())
        self.assertEqual(float(res_msk.sum()), 4.0, "Cơ chế rollback phải khôi phục lại nốt u nhỏ khi Affine làm văng khỏi khung hình!")

        # 3. Kiểm tra bảo toàn nền 0.0 ngoài phổi khi intensity jitter bật
        with mock.patch("common.data_pipeline.dataset.random.random", side_effect=[0.9, 0.9, 0.9, 0.1]), \
             mock.patch("common.data_pipeline.dataset.random.randint", return_value=0), \
             mock.patch("common.data_pipeline.dataset.random.uniform", side_effect=[1.10, 0.05]):
            jit_img, _ = LIDC25DDataset._augment_pair_level2(img.copy(), msk.copy())
        self.assertEqual(float(jit_img[:, 0, 0].max()), 0.0, "Vùng nền ngoài phổi phải giữ nguyên 0.0 sau intensity jitter!")

        # 4. Kiểm tra mode int16 HU (--gpu-window) vẫn hoạt động trơn tru với aug_level=2
        img_int16 = np.full((3, 64, 64), -1000, dtype=np.int16)
        img_int16[:, 16:48, 16:48] = -500
        res_i16, res_m16 = LIDC25DDataset._augment_pair_level2(img_int16, edge_msk.copy())
        self.assertEqual(res_i16.dtype, np.int16)
        self.assertEqual(res_m16.shape, (1, 64, 64))

        # 5. Kiểm tra build_dataloaders truyền aug_level=2 cho train_ds và giữ val_ds không augment
        tmp = Path(tempfile.mkdtemp())
        try:
            for i in range(1, 5):
                p_dir = tmp / f"LIDC-IDRI-000{i}"
                p_dir.mkdir(parents=True)
                arr = np.random.randint(-1000, 200, size=(8, 128, 128), dtype=np.int16)
                ms = np.zeros((8, 128, 128), dtype=np.uint8)
                lg = np.ones((8, 128, 128), dtype=np.uint8)
                ms[3:5, 50:60, 50:60] = 1
                np.save(p_dir / "image_1mm.npy", arr)
                np.save(p_dir / "mask_1mm.npy", ms)
                np.save(p_dir / "lung_1mm.npy", lg)
                with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump({"orig_spacing_xyz": [1.0, 1.0, 1.0], "lung_ok": True}, f)
            build_dataset_index(tmp)
            tr_loader, va_loader = build_dataloaders(tmp, batch_size=2, num_workers=0, aug_level=2)
            self.assertEqual(tr_loader.dataset.aug_level, 2)
            self.assertTrue(tr_loader.dataset.augment)
            self.assertFalse(va_loader.dataset.augment)
            self.assertEqual(va_loader.dataset.aug_level, 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_notebooks_contain_run2_params_and_presets(self):
        train_nb = _ROOT / "mamba_unet" / "notebooks" / "colab_train.ipynb"
        step0_nb = _ROOT / "mamba_unet" / "notebooks" / "colab_step0_measurement.ipynb"

        with open(train_nb, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        self.assertEqual(train_data.get("nbformat"), 4)
        train_src = "\n".join("".join(c.get("source", [])) for c in train_data["cells"])
        self.assertIn("--ema {EMA}", train_src)
        self.assertIn("--aug-level {AUG_LEVEL}", train_src)
        self.assertIn("mamba_colab_500item_50epoch_R2a_WD002_EMA0999", train_src)

        with open(step0_nb, "r", encoding="utf-8") as f:
            step0_data = json.load(f)
        self.assertEqual(step0_data.get("nbformat"), 4)
        step0_src = "\n".join("".join(c.get("source", [])) for c in step0_data["cells"])
        for preset in ('"r2a"', '"r2b"', '"r2c"'):
            self.assertIn(preset, step0_src)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    unittest.main()
