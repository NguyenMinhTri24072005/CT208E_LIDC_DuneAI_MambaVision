# -*- coding: utf-8 -*-
"""
unet_base/train_main.py - Điểm nhập thực thi (CLI entrypoint) cho kiến trúc UNetBase.

Các chế độ hoạt động:
  - selftest: Kiểm thử tự động toàn diện (huấn luyện dữ liệu giả lập, ngắt, resume auto, xuất fp16).
  - overfit : Huấn luyện thử trên tập siêu nhỏ để kiểm tra khả năng hội tụ nhanh của mô hình.
  - train   : Huấn luyện chính thức với đầy đủ tùy chọn cấu hình và ghi nhận phiên chạy.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.engine.cli_args import add_common_args
from common.engine.checkpoint_io import load_checkpoint_file
from common.engine.run_manager import create_run, resolve_resume, RunHandle
from common.engine.utils import seed_everything
from common.paths import default_runs_root

from unet_base.activations import DEFAULT_ACTIVATION, list_activations
from unet_base.losses import DEFAULT_LOSS, DEFAULT_LOSS_PARAMS, get_loss, list_losses
from unet_base.model import UNetBase, build_unet_base
from unet_base.trainer import UNetBaseTrainer


# =============================================================================
# DATASET GIẢ LẬP CHO SELFTEST VÀ OVERFIT
# =============================================================================
class Synthetic25DDataset(Dataset):
    """Dataset giả lập tạo nhanh các tensor 2.5D (3 kênh x 64 x 64)."""
    def __init__(self, num_samples: int = 16, image_size: int = 64):
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        # Giả lập mặt nạ nhị phân có một đốm nhỏ ở giữa
        self.masks = torch.zeros(num_samples, 1, image_size, image_size)
        mid = image_size // 2
        self.masks[:, :, mid - 4 : mid + 4, mid - 4 : mid + 4] = 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {"image": self.images[idx], "mask": self.masks[idx]}


# =============================================================================
# BỘ XỬ LÝ LỆNH TRAIN / OVERFIT / SELFTEST
# =============================================================================
def _load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ Không đọc được file cấu hình YAML ({e}), dùng tham số mặc định.")
        return {}


def train_action(args):
    """Thực thi quy trình huấn luyện chính."""
    seed_everything(args.seed)

    # Đọc cấu hình bổ sung từ YAML nếu có
    yaml_cfg = _load_yaml_config(args.config) if args.config else {}

    # 1. Chuẩn bị dữ liệu
    cache_dir = Path(args.cache_dir).resolve()
    from common.data_pipeline.dataset import build_dataloaders
    train_loader, val_loader = build_dataloaders(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_gpu_window=args.gpu_window,
        lung_mask=not args.no_lung_mask,
        augment=not args.no_augment,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        rebuild_index=args.rebuild_index,
        seed=args.seed,
    )

    # 2. Xây dựng mô hình
    model = build_unet_base(
        config=yaml_cfg,
        features=args.features,
        act=args.activation,
    )

    # 3. Xây dựng Loss
    loss_params = dict(DEFAULT_LOSS_PARAMS)
    if args.loss_params:
        try:
            custom_p = json.loads(args.loss_params)
            loss_params.update(custom_p)
        except Exception as e:
            print(f"⚠️ Không parse được --loss-params ({e}), dùng mặc định.")
    criterion = get_loss(args.loss, **loss_params)

    # 4. Tối ưu hóa (Optimizer & Scheduler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # 5. Quản lý Run & Resume
    runs_root = args.runs_root or default_runs_root("unet_base")
    resumed_dir, resumed_ckpt = resolve_resume(args.resume, arch="unet_base", runs_root=runs_root)

    run_handle = None
    if resumed_dir and (resumed_dir / "run_config.json").exists():
        try:
            with open(resumed_dir / "run_config.json", "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
            run_handle = RunHandle(resumed_dir, saved_cfg)
        except Exception:
            pass

    if run_handle is None:
        run_handle = create_run(
            arch="unet_base",
            runs_root=runs_root,
            run_name=args.run_name,
            tag=args.tag,
            args=vars(args),
            model_meta={"class": "UNetBase", "features": args.features, "activation": args.activation},
            data_meta={"cache_dir": str(cache_dir)},
            loss_meta={"name": args.loss, "params": loss_params},
            optimizer_meta={"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
            scheduler_meta={"name": args.scheduler},
            training_meta={"epochs_target": args.epochs, "batch_size": args.batch_size},
            notes=args.notes,
            init_weights=args.init_weights,
        )

    # 6. Khởi tạo Trainer
    trainer = UNetBaseTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        features=args.features,
        activation_name=args.activation,
        run_handle=run_handle,
        grad_clip=args.grad_clip,
        use_amp=not args.no_amp,
        show_progress=not args.no_progress,
    )

    if args.init_weights:
        trainer.load_checkpoint(args.init_weights, resume_training=False)

    # 7. Bắt đầu huấn luyện
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        resume_from=resumed_ckpt or args.resume,
    )


def overfit_action(args):
    """Huấn luyện overfit trên 1 batch nhỏ để kiểm tra khả năng học."""
    print("🔥 Bắt đầu kiểm tra overfit (UNetBase)...")
    seed_everything(args.seed)
    train_loader = DataLoader(Synthetic25DDataset(num_samples=4, image_size=64), batch_size=2)
    val_loader = DataLoader(Synthetic25DDataset(num_samples=2, image_size=64), batch_size=2)

    model = build_unet_base(features=16, act=args.activation)
    criterion = get_loss("dice_bce")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = UNetBaseTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        features=16,
        activation_name=args.activation,
        save_dir=tempfile.mkdtemp(),
        show_progress=False,
        use_amp=False,
    )

    history = trainer.fit(train_loader, val_loader, epochs=5)
    final_dice = history[-1]["val_dice"]
    print(f"✅ Hoàn thành 5 epochs overfit! Final Val Dice: {final_dice:.4f}")


def selftest_action(_):
    """
    Kiểm thử tự động toàn diện theo chuẩn Spec:
      1. Train ngắn trên dữ liệu giả lập (2 epochs).
      2. Ngắt giả lập, kiểm tra lưu checkpoint.
      3. Resume auto tiếp tục liền mạch (epochs 3-4).
      4. Kiểm tra run_config.json đủ khóa.
      5. Kiểm tra nạp lại best_weights_fp16.pth.
    """
    print("🧪 Bắt đầu selftest toàn diện unet_base.train_main...")
    tmp_root = Path(tempfile.mkdtemp())

    try:
        train_loader = DataLoader(Synthetic25DDataset(num_samples=8, image_size=64), batch_size=2)
        val_loader = DataLoader(Synthetic25DDataset(num_samples=4, image_size=64), batch_size=2)

        # Giai đoạn 1: Huấn luyện có ngắt giả lập (Interrupted) tại đầu epoch 3
        class InterruptedUNetBaseTrainer(UNetBaseTrainer):
            def on_epoch_start(self, epoch: int):
                if epoch == 3:
                    raise KeyboardInterrupt("⚡ Giả lập người dùng nhấn Ctrl+C (ngắt tại Epoch 3)!")

        model1 = build_unet_base(features=8, act="relu")
        criterion1 = get_loss("dice_bce")
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)

        run_h = create_run(
            arch="unet_base",
            runs_root=tmp_root,
            run_name="selftest_run",
            args={"epochs": 4, "lr": 0.01, "features": 8, "activation": "relu"},
        )

        trainer1 = InterruptedUNetBaseTrainer(
            model=model1,
            criterion=criterion1,
            optimizer=optimizer1,
            features=8,
            activation_name="relu",
            run_handle=run_h,
            show_progress=False,
            use_amp=False,
        )

        trainer1.fit(train_loader, val_loader, epochs=4)
        assert (run_h.run_dir / "last_model.pth").exists(), "Thiếu last_model.pth sau khi ngắt!"
        assert (run_h.run_dir / "best_model.pth").exists(), "Thiếu best_model.pth!"
        assert (run_h.run_dir / "best_weights_fp16.pth").exists(), "Thiếu best_weights_fp16.pth!"

        # Giai đoạn 2: Giả lập ngắt và tiếp tục với resume_from="auto"
        res_dir, res_ckpt = resolve_resume("auto", arch="unet_base", runs_root=tmp_root)
        assert res_dir == run_h.run_dir, "resolve_resume không tìm đúng thư mục run!"
        assert res_ckpt == run_h.run_dir / "last_model.pth", "resolve_resume không tìm đúng last_model.pth!"

        with open(run_h.config_path, "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        run_h_resumed = RunHandle(run_h.run_dir, saved_cfg)

        model2 = build_unet_base(features=8, act="relu")
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        trainer2 = UNetBaseTrainer(
            model=model2,
            criterion=criterion1,
            optimizer=optimizer2,
            features=8,
            activation_name="relu",
            run_handle=run_h_resumed,
            show_progress=False,
            use_amp=False,
        )

        trainer2.fit(train_loader, val_loader, epochs=4, resume_from="auto")
        assert trainer2.start_epoch == 3, f"Phải resume từ epoch 3, got {trainer2.start_epoch}"
        assert len(trainer2.history) == 4, f"Tổng số epochs phải là 4, got {len(trainer2.history)}"

        # Giai đoạn 3: Kiểm tra tính nguyên vẹn của best_weights_fp16.pth
        fp16_data = load_checkpoint_file(run_h.run_dir / "best_weights_fp16.pth")
        sd = fp16_data.get("model_state_dict", fp16_data.get("state_dict"))
        assert sd is not None, "best_weights_fp16.pth phải chứa model_state_dict!"
        assert "meta" in fp16_data
        assert fp16_data["meta"]["arch"] == "unet_base"
        first_weight = next(iter(sd.values()))
        assert first_weight.dtype == torch.float16, f"Trọng số phải là fp16, got {first_weight.dtype}"

        # Giai đoạn 4: Kiểm tra đầy đủ trường trong run_config.json
        with open(run_h.config_path, "r", encoding="utf-8") as f:
            final_cfg = json.load(f)
        assert final_cfg["status"] == "completed"
        assert len(final_cfg["resume_events"]) >= 1
        assert final_cfg["resume_events"][-1]["from_epoch"] == 2
        assert final_cfg["result"]["epochs_completed"] == 4

        print("✅ Selftest unet_base.train_main: HOÀN TẤT TOÀN DIỆN VÀ THÀNH CÔNG!")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# =============================================================================
# CLI PARSER
# =============================================================================
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Huấn luyện mô hình UNetBase phân đoạn ảnh y tế")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # 1. train subparser
    train_p = subparsers.add_parser("train", help="Huấn luyện mô hình chuẩn")
    add_common_args(
        train_p,
        activation_choices=list_activations(),
        loss_choices=list_losses(),
        default_activation=DEFAULT_ACTIVATION,
        default_loss=DEFAULT_LOSS,
    )
    train_p.add_argument("--features", type=int, default=64, help="Số kênh đặc trưng cơ sở của UNetBase")
    train_p.set_defaults(func=train_action)

    # 2. overfit subparser
    overfit_p = subparsers.add_parser("overfit", help="Kiểm tra overfit nhanh trên batch giả lập")
    add_common_args(
        overfit_p,
        activation_choices=list_activations(),
        loss_choices=list_losses(),
        default_activation=DEFAULT_ACTIVATION,
        default_loss=DEFAULT_LOSS,
    )
    overfit_p.add_argument("--features", type=int, default=16)
    overfit_p.set_defaults(func=overfit_action)

    # 3. selftest subparser
    selftest_p = subparsers.add_parser("selftest", help="Kiểm thử tự động chu trình huấn luyện")
    selftest_p.set_defaults(func=selftest_action)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
