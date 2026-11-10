# -*- coding: utf-8 -*-
"""
mamba_unet/train_main.py - Điểm nhập thực thi (CLI entrypoint) cho kiến trúc Mamba-UNet.

Các chế độ hoạt động:
  - selftest: Kiểm thử tự động toàn diện (huấn luyện dữ liệu giả lập, ngắt, resume auto,
              kiểm tra lịch đóng băng backbone, xuất fp16 và run_config).
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

from mamba_unet.activations import DEFAULT_ACTIVATION, list_activations
from mamba_unet.losses import DEFAULT_LOSS, DEFAULT_LOSS_PARAMS, get_loss, list_losses
from mamba_unet.model import MambaUNet, build_mamba_unet
from mamba_unet.trainer import MambaUNetTrainer


# =============================================================================
# DATASET GIẢ LẬP CHO SELFTEST VÀ OVERFIT
# =============================================================================
class Synthetic25DDataset(Dataset):
    """Dataset giả lập tạo nhanh các tensor 2.5D (3 kênh x image_size x image_size)."""
    def __init__(self, num_samples: int = 8, image_size: int = 64):
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 3, image_size, image_size)
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
    """Thực thi quy trình huấn luyện chính cho Mamba-UNet."""
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
    model = build_mamba_unet(
        config=yaml_cfg,
        backbone=args.backbone,
        pretrained=args.pretrained,
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

    # 4. Quản lý Run & Resume
    runs_root = args.runs_root or default_runs_root("mamba_unet")
    resumed_dir, resumed_ckpt = resolve_resume(args.resume, arch="mamba_unet", runs_root=runs_root)

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
            arch="mamba_unet",
            runs_root=runs_root,
            run_name=args.run_name,
            tag=args.tag,
            args=vars(args),
            model_meta={
                "class": "MambaUNet",
                "backbone": args.backbone,
                "pretrained": args.pretrained,
                "activation": args.activation,
            },
            data_meta={"cache_dir": str(cache_dir)},
            loss_meta={"name": args.loss, "params": loss_params},
            optimizer_meta={
                "name": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "backbone_lr_mult": args.backbone_lr_mult,
                "freeze_backbone_epochs": args.freeze_backbone_epochs,
            },
            scheduler_meta={"name": args.scheduler},
            training_meta={"epochs_target": args.epochs, "batch_size": args.batch_size},
            notes=args.notes,
            init_weights=args.init_weights,
        )

    # 5. Khởi tạo Trainer
    trainer = MambaUNetTrainer(
        model=model,
        criterion=criterion,
        backbone_name=args.backbone,
        activation_name=args.activation,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        backbone_lr_mult=args.backbone_lr_mult,
        run_handle=run_handle,
        grad_clip=args.grad_clip,
        use_amp=not args.no_amp,
        show_progress=not args.no_progress,
    )

    # 6. Tối ưu hóa (Optimizer & Scheduler với phân nhóm param groups)
    param_groups = trainer.build_param_groups(args.lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    trainer.optimizer = optimizer

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    trainer.scheduler = scheduler

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
    """Huấn luyện overfit trên 1 batch nhỏ để kiểm tra khả năng hội tụ nhanh."""
    print("🔥 Bắt đầu kiểm tra overfit (Mamba-UNet)...")
    seed_everything(args.seed)
    train_loader = DataLoader(Synthetic25DDataset(num_samples=4, image_size=64), batch_size=2)
    val_loader = DataLoader(Synthetic25DDataset(num_samples=2, image_size=64), batch_size=2)

    from mamba_unet.backbone.mamba_vision import MambaVision
    tiny_bb = MambaVision(
        dim=16,
        in_dim=8,
        depths=[1, 1, 2, 1],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 2, 4, 4],
        in_chans=3,
    )
    model = MambaUNet(in_channels=3, num_classes=1, backbone=tiny_bb, act=args.activation)
    criterion = get_loss("dice_bce")

    trainer = MambaUNetTrainer(
        model=model,
        criterion=criterion,
        backbone_name="tiny",
        activation_name=args.activation,
        freeze_backbone_epochs=0,
        save_dir=tempfile.mkdtemp(),
        show_progress=False,
        use_amp=False,
    )
    optimizer = torch.optim.AdamW(trainer.build_param_groups(1e-3), weight_decay=1e-4)
    trainer.optimizer = optimizer

    history = trainer.fit(train_loader, val_loader, epochs=5)
    final_dice = history[-1]["val_dice"]
    print(f"✅ Hoàn thành 5 epochs overfit! Final Val Dice: {final_dice:.4f}")


def selftest_action(_):
    """
    Kiểm thử tự động toàn diện theo chuẩn Spec:
      1. Train ngắn trên dữ liệu giả lập (2 epochs, freeze_backbone_epochs=1).
      2. Ngắt giả lập tại đầu epoch 3, kiểm tra lưu checkpoint.
      3. Resume auto tiếp tục liền mạch (epochs 3-4).
      4. Kiểm tra trạng thái đóng/mở băng backbone theo lịch.
      5. Kiểm tra run_config.json đủ khóa theo Schema v1.
      6. Kiểm tra tính nguyên vẹn của best_weights_fp16.pth.
    """
    print("🧪 Bắt đầu selftest toàn diện mamba_unet.train_main...")
    tmp_root = Path(tempfile.mkdtemp())

    try:
        train_loader = DataLoader(Synthetic25DDataset(num_samples=4, image_size=64), batch_size=2)
        val_loader = DataLoader(Synthetic25DDataset(num_samples=2, image_size=64), batch_size=2)

        from mamba_unet.backbone.mamba_vision import MambaVision

        def make_tiny_model():
            bb = MambaVision(
                dim=16,
                in_dim=8,
                depths=[1, 1, 2, 1],
                window_size=[8, 8, 14, 7],
                num_heads=[2, 2, 4, 4],
                in_chans=3,
            )
            return MambaUNet(in_channels=3, num_classes=1, backbone=bb, act="relu")

        # Giai đoạn 1: Huấn luyện có ngắt giả lập tại đầu epoch 3
        class InterruptedMambaTrainer(MambaUNetTrainer):
            def on_epoch_start(self, epoch: int):
                super().on_epoch_start(epoch)
                if epoch == 3:
                    raise KeyboardInterrupt("⚡ Giả lập người dùng ngắt (Ctrl+C tại Epoch 3)!")

        model1 = make_tiny_model()
        criterion1 = get_loss("dice_bce")

        run_h = create_run(
            arch="mamba_unet",
            runs_root=tmp_root,
            run_name="selftest_run",
            args={"epochs": 4, "lr": 0.01, "backbone": "tiny", "activation": "relu"},
        )

        trainer1 = InterruptedMambaTrainer(
            model=model1,
            criterion=criterion1,
            backbone_name="tiny",
            activation_name="relu",
            freeze_backbone_epochs=1,
            backbone_lr_mult=0.1,
            run_handle=run_h,
            show_progress=False,
            use_amp=False,
        )
        optimizer1 = torch.optim.SGD(trainer1.build_param_groups(0.01))
        trainer1.optimizer = optimizer1

        trainer1.fit(train_loader, val_loader, epochs=4)
        assert (run_h.run_dir / "last_model.pth").exists(), "Thiếu last_model.pth sau khi ngắt!"
        assert (run_h.run_dir / "best_model.pth").exists(), "Thiếu best_model.pth!"
        assert (run_h.run_dir / "best_weights_fp16.pth").exists(), "Thiếu best_weights_fp16.pth!"

        # Giai đoạn 2: Giả lập ngắt và tiếp tục với resume_from="auto"
        res_dir, res_ckpt = resolve_resume("auto", arch="mamba_unet", runs_root=tmp_root)
        assert res_dir == run_h.run_dir, "resolve_resume không tìm đúng thư mục run!"
        assert res_ckpt == run_h.run_dir / "last_model.pth", "resolve_resume không tìm đúng last_model.pth!"

        with open(run_h.config_path, "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        run_h_resumed = RunHandle(run_h.run_dir, saved_cfg)

        model2 = make_tiny_model()
        trainer2 = MambaUNetTrainer(
            model=model2,
            criterion=criterion1,
            backbone_name="tiny",
            activation_name="relu",
            freeze_backbone_epochs=1,
            backbone_lr_mult=0.1,
            run_handle=run_h_resumed,
            show_progress=False,
            use_amp=False,
        )
        optimizer2 = torch.optim.SGD(trainer2.build_param_groups(0.01))
        trainer2.optimizer = optimizer2

        trainer2.fit(train_loader, val_loader, epochs=4, resume_from="auto")
        assert trainer2.start_epoch == 3, f"Phải resume từ epoch 3, nhận được {trainer2.start_epoch}"
        assert len(trainer2.history) == 4, f"Tổng số epochs phải là 4, nhận được {len(trainer2.history)}"

        # Giai đoạn 3: Kiểm tra trạng thái đóng/mở băng backbone
        # Tại epoch 1 (freeze=1): backbone phải bị đóng băng
        trainer2.on_epoch_start(1)
        for p in model2.encoder.parameters():
            assert not p.requires_grad, "Backbone phải bị đóng băng ở Epoch 1!"
        # Tại epoch 3 (freeze=1): backbone phải được mở băng
        trainer2.on_epoch_start(3)
        for p in model2.encoder.parameters():
            assert p.requires_grad, "Backbone phải được mở băng ở Epoch 3!"

        # Giai đoạn 4: Kiểm tra tính nguyên vẹn của best_weights_fp16.pth
        fp16_data = load_checkpoint_file(run_h.run_dir / "best_weights_fp16.pth")
        sd = fp16_data.get("model_state_dict", fp16_data.get("state_dict"))
        assert sd is not None, "best_weights_fp16.pth phải chứa model_state_dict!"
        assert "meta" in fp16_data
        assert fp16_data["meta"]["arch"] == "mamba_unet"
        first_weight = next(iter(sd.values()))
        assert first_weight.dtype == torch.float16, f"Trọng số phải là fp16, nhận được {first_weight.dtype}"

        # Giai đoạn 5: Kiểm tra đầy đủ trường trong run_config.json
        with open(run_h.config_path, "r", encoding="utf-8") as f:
            final_cfg = json.load(f)
        assert final_cfg["status"] == "completed"
        assert len(final_cfg["resume_events"]) >= 1
        assert final_cfg["resume_events"][-1]["from_epoch"] == 2
        assert final_cfg["result"]["epochs_completed"] == 4

        print("✅ Selftest mamba_unet.train_main: HOÀN TẤT TOÀN DIỆN VÀ THÀNH CÔNG!")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# =============================================================================
# CLI PARSER
# =============================================================================
def _add_mamba_specific_args(parser: argparse.ArgumentParser):
    mamba_grp = parser.add_argument_group("Tham số Mamba-UNet")
    mamba_grp.add_argument(
        "--backbone",
        choices=["mamba_vision_T", "mamba_vision_T2", "mamba_vision_S", "mamba_vision_B"],
        default="mamba_vision_T",
        help="Biến thể MambaVision backbone",
    )
    mamba_grp.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Nạp trọng số ImageNet-1K pretrained từ NVIDIA",
    )
    mamba_grp.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Khởi tạo backbone ngẫu nhiên (không dùng pretrained)",
    )
    mamba_grp.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=5,
        help="Số epoch đóng băng backbone đầu tiên (-1: đóng băng suốt quá trình, 0: không đóng băng)",
    )
    mamba_grp.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.1,
        help="Hệ số nhân LR cho backbone so với decoder (mặc định 0.1)",
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Mamba-UNet phân đoạn ảnh y tế")
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
    _add_mamba_specific_args(train_p)
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
    _add_mamba_specific_args(overfit_p)
    overfit_p.set_defaults(func=overfit_action)

    # 3. selftest subparser
    selftest_p = subparsers.add_parser("selftest", help="Kiểm thử tự động chu trình huấn luyện")
    selftest_p.set_defaults(func=selftest_action)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
