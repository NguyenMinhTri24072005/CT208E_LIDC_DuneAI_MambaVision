# -*- coding: utf-8 -*-
"""
trainer.py - Bộ máy huấn luyện (Training Engine) cho dự án DuneAI 2.5D.

Tính năng:
  - Tự động phát hiện và chia tải Multi-GPU (nn.DataParallel) hoặc chạy 1 GPU / CPU.
  - Hỗ trợ Mixed Precision (AMP float16) tăng tốc huấn luyện và tiết kiệm VRAM.
  - Tích hợp trực tiếp window_hu_torch trên GPU nếu DataLoader trả về HU int16.
  - Hỗ trợ lịch đóng/mở khóa Backbone tự động theo epoch (`freeze_backbone_epochs`).
  - Hiển thị đầy đủ Train/Val Loss, Dice, IoU, LR cùng thanh tiến trình batch và ước lượng thời gian (ETA).
  - Cơ chế Resume toàn diện (khôi phục model, optimizer, scheduler, AMP scaler, best_val_dice,
    early-stopping counter, kiểm tra tương thích meta và sao lưu history_before_resume.csv).
  - Ghi file an toàn (Atomic Save qua .tmp -> os.replace) chống hỏng checkpoint khi sập nguồn/mất kết nối.

Dùng:
  python DuneAI_Remake/src/trainer.py selftest
  python DuneAI_Remake/src/trainer.py overfit --cache-dir data/cache_1mm --epochs 15 --batch-size 2
  python DuneAI_Remake/src/trainer.py train --cache-dir data/cache_1mm --epochs 50 --batch-size 8 --resume auto
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
for p in (str(THIS_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import configs.config as cfg
from dataset import LIDC25DDataset, create_dataloaders
from models.dune_unet import DuneUNet
from losses import DiceBCELoss, TverskyFocalLoss, compute_segmentation_metrics
from medical_prep import window_hu_torch
from offline_builder import build_dataset_index

DEFAULT_BATCH_SIZE = getattr(cfg, "BATCH_SIZE", 8)
DEFAULT_EPOCHS = getattr(cfg, "EPOCHS", 50)
DEFAULT_LR = getattr(cfg, "LR", getattr(cfg, "LEARNING_RATE", 1e-4))
DEFAULT_SEED = getattr(cfg, "SEED", 42)


def seed_everything(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_file(path, map_location="cpu"):
    """Đọc file checkpoint (.pth) từ đĩa một cách an toàn."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint để resume/load: {path}")
    return torch.load(path, map_location=map_location)


def format_duration(seconds):
    """Định dạng thời gian (giây) sang chuỗi dễ đọc: 45.2s, 12m 30s, hoặc 02h 15m 10s."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    if seconds < 60:
        return f"{seconds:.1f}s"
    sec = int(round(seconds))
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m:02d}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def _make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(enabled):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _atomic_torch_save(obj, path):
    """Lưu checkpoint qua file tạm rồi đổi tên để tránh hỏng file nếu Colab ngắt kết nối giữa chừng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _atomic_csv_save(df, path):
    """Ghi history.csv an toàn qua file tạm."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler=None,
        device=None,
        use_amp=True,
        save_dir="checkpoints",
        grad_clip=1.0,
        show_progress=True,
        freeze_backbone_epochs=0,
        extra_meta=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.type == "cuda"

        if self.is_cuda and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
            print(f"🔥 Phát hiện {torch.cuda.device_count()} GPUs! Kích hoạt chế độ song song nn.DataParallel.")
            model = nn.DataParallel(model)

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = bool(use_amp and self.is_cuda)
        self.scaler = _make_grad_scaler(self.use_amp)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = grad_clip
        self.show_progress = bool(show_progress)
        self.freeze_backbone_epochs = int(freeze_backbone_epochs)

        core = self._unwrap_model()
        default_meta = {
            "features": int(getattr(core, "features", 64)),
            "in_channels": int(getattr(core, "in_channels", 3)),
            "backbone": str(getattr(core, "backbone_name", "none")),
        }
        if extra_meta:
            default_meta.update(extra_meta)
        self.extra_meta = default_meta

        # Trạng thái phục vụ huấn luyện và Resume
        self.start_epoch = 1
        self.best_val_dice = -1.0
        self.no_improve = 0
        self.history = []
        self._cur_epoch = None
        self._total_epochs = None

    def _unwrap_model(self):
        """Bóc lớp bọc DataParallel để lưu state_dict chuẩn, nạp lại được trên mọi máy."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _apply_backbone_schedule(self, epoch: int):
        """
        Tự động đóng băng hoặc mở khóa Backbone dựa trên số thứ tự `epoch` hiện tại
        và cấu hình `self.freeze_backbone_epochs`:
          - freeze_backbone_epochs == 0 : Mở khóa hoàn toàn.
          - freeze_backbone_epochs <  0 : Đóng băng suốt quá trình huấn luyện.
          - freeze_backbone_epochs >  0 : Đóng băng khi epoch <= freeze_backbone_epochs, mở khóa từ epoch tiếp theo.
        """
        core = self._unwrap_model()
        if not getattr(core, "has_backbone", False):
            return
        fb = int(self.freeze_backbone_epochs)
        should_freeze = (fb < 0) or (fb > 0 and int(epoch) <= fb)
        prev_frozen = core.encoder_frozen
        core.set_backbone_frozen(should_freeze)
        if should_freeze != prev_frozen or int(epoch) == self.start_epoch:
            state_txt = "ĐÓNG BĂNG (Frozen)" if should_freeze else "MỞ KHÓA (Unfrozen)"
            print(f"🧬 [Epoch {epoch:02d}] Backbone '{core.backbone_name}': {state_txt}")

    def _prepare_batch(self, batch):
        images = batch["image"].to(self.device, non_blocking=self.is_cuda)
        masks = batch["mask"].to(self.device, non_blocking=self.is_cuda)
        if images.dtype in (torch.int16, torch.int32, torch.int64):
            images = window_hu_torch(images)
        return images, masks

    def run_epoch(self, loader, is_train=True, epoch=None, total_epochs=None):
        if epoch is None:
            epoch = self._cur_epoch
        if total_epochs is None:
            total_epochs = self._total_epochs

        self.model.train(is_train)
        total_loss = 0.0
        total_samples = 0
        inter_sum = 0.0
        pred_sum = 0.0
        target_sum = 0.0

        phase = "Train" if is_train else "Val  "
        ep_prefix = f"Epoch [{epoch:02d}/{total_epochs:02d}] {phase}" if (epoch and total_epochs) else phase

        use_bar = self.show_progress and (tqdm is not None) and sys.stderr.isatty()
        iterator = tqdm(loader, desc=ep_prefix, leave=False, dynamic_ncols=True) if use_bar else loader

        grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
        with grad_ctx:
            for batch in iterator:
                images, masks = self._prepare_batch(batch)
                bs = images.size(0)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)

                with _autocast_context(self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                if is_train:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip is not None and self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += float(loss.item()) * bs
                total_samples += bs

                m = compute_segmentation_metrics(logits.detach(), masks)
                inter_sum += m["intersection"]
                pred_sum += m["pred_sum"]
                target_sum += m["target_sum"]

                if use_bar:
                    running_loss = total_loss / max(1, total_samples)
                    running_dice = (2.0 * inter_sum + 1e-6) / (pred_sum + target_sum + 1e-6)
                    iterator.set_postfix(loss=f"{running_loss:.4f}", dice=f"{running_dice:.4f}")

        eps = 1e-6
        epoch_loss = total_loss / max(1, total_samples)
        epoch_dice = (2.0 * inter_sum + eps) / (pred_sum + target_sum + eps)
        union_sum = pred_sum + target_sum - inter_sum
        epoch_iou = (inter_sum + eps) / (union_sum + eps)

        return {"loss": epoch_loss, "dice": epoch_dice, "iou": epoch_iou}

    def save_checkpoint(self, filename, epoch, val_metrics):
        path = self.save_dir / filename
        ckpt = {
            "epoch": int(epoch),
            "model_state_dict": self._unwrap_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_type": type(self.scheduler).__name__ if self.scheduler is not None else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "best_val_dice": float(self.best_val_dice),
            "no_improve": int(self.no_improve),
            "val_metrics": val_metrics,
            "history": list(self.history),
            "meta": dict(self.extra_meta),
        }
        _atomic_torch_save(ckpt, path)
        return path

    def load_checkpoint(
        self,
        checkpoint_path,
        resume_training=True,
        load_scheduler=True,
        check_meta=None,
    ):
        """
        Khôi phục trạng thái từ file checkpoint (.pth) và trả về dict `ckpt`.
        """
        ckpt = load_checkpoint_file(checkpoint_path, map_location=self.device)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            try:
                self._unwrap_model().load_state_dict(ckpt)
            except RuntimeError as e:
                raise ValueError(f"Không nạp được trọng số từ {checkpoint_path}: {e}") from e
            print(f"📥 Đã nạp trọng số mô hình (raw state_dict) từ: {checkpoint_path}")
            return {"epoch": 0, "model_state_dict": ckpt}

        if check_meta:
            saved_meta = ckpt.get("meta") or {}
            for k, expected_val in check_meta.items():
                if k == "backbone":
                    saved_val = saved_meta.get("backbone", "none")
                else:
                    saved_val = saved_meta.get(k)
                if saved_val is not None and saved_val != expected_val:
                    raise ValueError(
                        f"Checkpoint {checkpoint_path} được huấn luyện với {k}={saved_val!r}, "
                        f"không khớp với cấu hình hiện tại {k}={expected_val!r}."
                    )

        state = ckpt["model_state_dict"]
        cleaned_state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        try:
            self._unwrap_model().load_state_dict(cleaned_state)
        except RuntimeError as e:
            raise ValueError(
                f"Không nạp được model_state_dict từ {checkpoint_path} (kiến trúc/features/backbone không khớp): "
                f"{str(e).splitlines()[0]}"
            ) from e

        if not resume_training:
            print(f"📥 Đã nạp trọng số khởi tạo (init_weights) từ: {checkpoint_path}")
            return ckpt

        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if load_scheduler and self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"⚠️ Không nạp được scheduler_state_dict ({e}), giữ cấu hình scheduler hiện tại.")

        if self.use_amp and ckpt.get("scaler_state_dict") is not None:
            try:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as e:
                print(f"⚠️ Không nạp được scaler_state_dict ({e}), khởi tạo scaler mới.")

        last_epoch = int(ckpt.get("epoch", 0))
        self.start_epoch = last_epoch + 1
        self.best_val_dice = float(ckpt.get("best_val_dice", -1.0))
        self.no_improve = int(ckpt.get("no_improve", 0))

        # Nếu history.csv trên đĩa có nhiều epoch hơn checkpoint đang resume -> sao lưu bản gốc trước khi cắt
        hist_csv_path = self.save_dir / "history.csv"
        if hist_csv_path.exists():
            try:
                df_disk = pd.read_csv(hist_csv_path)
                if len(df_disk) > 0 and int(df_disk["epoch"].max()) > last_epoch:
                    backup_path = self.save_dir / "history_before_resume.csv"
                    shutil.copy2(hist_csv_path, backup_path)
            except Exception:
                pass

        # Khôi phục lịch sử huấn luyện tới đúng last_epoch
        if isinstance(ckpt.get("history"), list) and len(ckpt["history"]) > 0:
            self.history = [r for r in ckpt["history"] if int(r.get("epoch", 0)) <= last_epoch]
        elif hist_csv_path.exists():
            try:
                df_hist = pd.read_csv(hist_csv_path)
                df_hist = df_hist[df_hist["epoch"] <= last_epoch]
                self.history = df_hist.to_dict(orient="records")
            except Exception:
                self.history = []

        print(
            f"🔄 RESUME THÀNH CÔNG từ '{checkpoint_path}' | "
            f"Đã xong epoch {last_epoch} -> Tiếp tục từ epoch {self.start_epoch} | "
            f"Best Val Dice hiện tại: {self.best_val_dice:.4f}"
        )
        return ckpt

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=DEFAULT_EPOCHS,
        early_stopping_patience=0,
        resume_from=None,
    ):
        """
        Vòng lặp huấn luyện chính.
        """
        if resume_from:
            if str(resume_from).lower() == "auto":
                candidate = self.save_dir / "last_model.pth"
                if candidate.exists():
                    self.load_checkpoint(candidate, resume_training=True, load_scheduler=True)
                else:
                    print(f"ℹ️  Chế độ --resume auto: Chưa có '{candidate}', bắt đầu huấn luyện mới từ Epoch 1.")
            else:
                self.load_checkpoint(resume_from, resume_training=True, load_scheduler=True)

            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                if getattr(self.scheduler, "T_max", epochs) != epochs and epochs > 0:
                    self.scheduler.T_max = max(1, epochs)
                    if hasattr(self.scheduler, "_get_closed_form_lr"):
                        for group, lr_val in zip(self.optimizer.param_groups, self.scheduler._get_closed_form_lr()):
                            group["lr"] = lr_val
                        self.scheduler._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        if self.start_epoch > epochs:
            print(
                f"✅ Mô hình đã hoàn thành {self.start_epoch - 1}/{epochs} epochs trong checkpoint trước đó. "
                f"(Hãy tăng --epochs lớn hơn {self.start_epoch - 1} nếu muốn huấn luyện thêm)."
            )
            return self.history

        n_batches_tr = len(train_loader)
        n_batches_va = len(val_loader) if val_loader is not None else 0
        print("=" * 95)
        print(
            f"🚀 BẮT ĐẦU HUẤN LUYỆN | Thiết bị: {self.device} | AMP: {self.use_amp} | "
            f"Epochs: {self.start_epoch} -> {epochs} | Batch: {n_batches_tr} train, {n_batches_va} val"
        )
        print("=" * 95)

        epoch_durations = []
        fit_start_time = time.time()

        try:
            for epoch in range(self.start_epoch, epochs + 1):
                ep_t0 = time.time()
                self._cur_epoch = epoch
                self._total_epochs = epochs

                # Cập nhật trạng thái Đóng/Mở khóa Backbone theo lịch
                self._apply_backbone_schedule(epoch)

                # 1. Pha Train
                tr_t0 = time.time()
                train_m = self.run_epoch(train_loader, is_train=True)
                train_time = time.time() - tr_t0

                # 2. Pha Validation
                va_t0 = time.time()
                if val_loader is not None:
                    val_m = self.run_epoch(val_loader, is_train=False)
                else:
                    val_m = train_m
                val_time = time.time() - va_t0

                # 3. Cập nhật Learning Rate Scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_m["dice"])
                    else:
                        self.scheduler.step()

                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - ep_t0
                epoch_durations.append(elapsed)

                recent_avg_sec = float(np.mean(epoch_durations[-5:]))
                remaining_epochs = epochs - epoch
                eta_sec = recent_avg_sec * remaining_epochs

                row = {
                    "epoch": int(epoch),
                    "lr": float(lr),
                    "train_loss": round(train_m["loss"], 5),
                    "train_dice": round(train_m["dice"], 5),
                    "train_iou": round(train_m["iou"], 5),
                    "val_loss": round(val_m["loss"], 5),
                    "val_dice": round(val_m["dice"], 5),
                    "val_iou": round(val_m["iou"], 5),
                    "train_time_sec": round(train_time, 2),
                    "val_time_sec": round(val_time, 2),
                    "time_sec": round(elapsed, 2),
                    "eta_sec": round(eta_sec, 2),
                }
                self.history = [r for r in self.history if int(r.get("epoch", -1)) != epoch] + [row]
                _atomic_csv_save(pd.DataFrame(self.history), self.save_dir / "history.csv")

                is_best = val_m["dice"] > self.best_val_dice
                if is_best:
                    self.best_val_dice = float(val_m["dice"])
                    self.no_improve = 0
                    self.save_checkpoint("best_model.pth", epoch, val_m)
                else:
                    self.no_improve += 1

                self.save_checkpoint("last_model.pth", epoch, val_m)

                badge = "⭐ [BEST]" if is_best else ""
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] | LR: {lr:.2e} | "
                    f"Train [Loss: {train_m['loss']:.4f}, Dice: {train_m['dice']:.4f}, IoU: {train_m['iou']:.4f}] | "
                    f"Val [Loss: {val_m['loss']:.4f}, Dice: {val_m['dice']:.4f}, IoU: {val_m['iou']:.4f}] | "
                    f"Time: {format_duration(elapsed)} (Tr:{train_time:.1f}s/Va:{val_time:.1f}s) | "
                    f"ETA: {format_duration(eta_sec)} {badge}"
                )

                if early_stopping_patience > 0 and self.no_improve >= early_stopping_patience:
                    print(
                        f"⏹️ Dừng sớm (Early Stopping) tại epoch {epoch} do Val Dice không cải thiện "
                        f"sau {early_stopping_patience} epochs liên tiếp."
                    )
                    break
        except KeyboardInterrupt:
            print(
                f"\n⚠️ Đã nhận tín hiệu dừng (KeyboardInterrupt). "
                f"Đã lưu an toàn tới hết Epoch {self.history[-1]['epoch'] if self.history else 0} tại {self.save_dir}."
            )
            return self.history

        total_fit_time = time.time() - fit_start_time
        print("=" * 95)
        print(
            f"🎉 Hoàn tất! Best Val Dice: {self.best_val_dice:.4f} | "
            f"Tổng thời gian chạy: {format_duration(total_fit_time)} | Lưu tại: {self.save_dir}"
        )
        return self.history


# ======================================================================
# CÁC CHẾ ĐỘ CHẠY CLI (TRAIN / OVERFIT / SELFTEST)
# ======================================================================
def _build_criterion(loss_name):
    if loss_name == "tversky":
        return TverskyFocalLoss()
    return DiceBCELoss(bce_weight=0.5, dice_weight=0.5)


def _cmd_train(a):
    seed_everything(a.seed)
    train_loader, val_loader = create_dataloaders(
        cache_dir=a.cache_dir,
        batch_size=a.batch_size,
        num_workers=a.workers,
        in_channels=3,
        apply_lung=not a.no_lung_mask,
        window_on_cpu=not a.gpu_window,
        augment_train=not a.no_augment,
    )
    model = DuneUNet(
        in_channels=3,
        classes=1,
        features=a.features,
        backbone=a.backbone,
        pretrained=not a.no_pretrained,
    )
    bb_mult = a.backbone_lr_mult if (model.has_backbone and not a.no_pretrained) else 1.0
    criterion = _build_criterion(a.loss)
    optimizer = torch.optim.AdamW(model.param_groups(a.lr, bb_mult), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=1e-6)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=not a.no_amp,
        save_dir=a.save_dir,
        show_progress=not a.no_progress,
        freeze_backbone_epochs=a.freeze_backbone_epochs,
    )
    trainer.fit(
        train_loader,
        val_loader,
        epochs=a.epochs,
        early_stopping_patience=a.patience,
        resume_from=a.resume,
    )


def _cmd_overfit(a):
    seed_everything(a.seed)
    ds = LIDC25DDataset(
        cache_dir=a.cache_dir,
        split="train",
        in_channels=3,
        apply_lung=True,
        window_on_cpu=not a.gpu_window,
        augment=False,
    )
    pos_indices = ds.df.index[ds.df["has_tumor"] == 1].tolist()
    chosen = pos_indices[: a.num_samples] if len(pos_indices) >= a.num_samples else list(range(min(len(ds), a.num_samples)))
    subset = Subset(ds, chosen)
    loader = DataLoader(subset, batch_size=min(a.batch_size, len(chosen)), shuffle=True, num_workers=0)

    print(f"🔬 Chạy Overfit Sanity Check trên {len(chosen)} lát cắt có u (backbone={a.backbone})...")
    model = DuneUNet(
        in_channels=3,
        classes=1,
        features=a.features,
        backbone=a.backbone,
        pretrained=not a.no_pretrained,
    )
    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.param_groups(a.lr, 0.1), lr=a.lr, weight_decay=1e-5)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        use_amp=not a.no_amp,
        save_dir=a.save_dir,
        show_progress=not a.no_progress,
    )
    trainer.fit(loader, val_loader=loader, epochs=a.epochs, resume_from=a.resume)


def _cmd_selftest(_):
    print("Đang chạy kiểm thử tích hợp toàn trình trainer.py (Synthetic End-to-End + Resume)...")
    seed_everything(42)
    tmp = Path(tempfile.mkdtemp())
    try:
        for i in range(1, 5):
            p_dir = tmp / f"LIDC-IDRI-000{i}"
            p_dir.mkdir(parents=True)
            img = np.full((8, 128, 128), -800, dtype=np.int16)
            msk = np.zeros((8, 128, 128), dtype=np.uint8)
            lung = np.ones((8, 128, 128), dtype=np.uint8)
            img[3:6, 48:80, 48:80] = 50
            msk[3:6, 48:80, 48:80] = 1

            np.save(p_dir / "image_1mm.npy", img)
            np.save(p_dir / "mask_1mm.npy", msk)
            np.save(p_dir / "lung_1mm.npy", lung)
            with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump({"orig_spacing_xyz": [1.0, 1.0, 1.0], "lung_ok": True}, f)

        build_dataset_index(tmp)
        train_ds = LIDC25DDataset(tmp, split="train", image_size=128, window_on_cpu=False, augment=False)
        val_ds = LIDC25DDataset(tmp, split="val", image_size=128, window_on_cpu=False, augment=False)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        ckpt_dir = tmp / "checkpoints"

        # Pha 1: Huấn luyện 3 epochs đầu tiên (giả lập bị ngắt kết nối sau epoch 3)
        model1 = DuneUNet(in_channels=3, classes=1, features=16)
        criterion1 = DiceBCELoss()
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=3e-3, weight_decay=1e-4)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=5, eta_min=1e-5)
        trainer1 = Trainer(model1, criterion1, optimizer1, scheduler1, save_dir=ckpt_dir, show_progress=False)
        hist1 = trainer1.fit(train_loader, val_loader, epochs=3)
        assert len(hist1) == 3, f"Pha 1 phải có 3 epochs, nhận được {len(hist1)}"

        # Pha 2: Khởi tạo mô hình mới hoàn toàn và Resume tự động ('auto') để chạy tiếp tới epoch 5
        model2 = DuneUNet(in_channels=3, classes=1, features=16)
        criterion2 = DiceBCELoss()
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=3e-3, weight_decay=1e-4)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=5, eta_min=1e-5)
        trainer2 = Trainer(model2, criterion2, optimizer2, scheduler2, save_dir=ckpt_dir, show_progress=False)
        hist2 = trainer2.fit(train_loader, val_loader, epochs=5, resume_from="auto")

        assert trainer2.start_epoch == 4, f"Phải bắt đầu tiếp từ epoch 4, nhưng start_epoch={trainer2.start_epoch}"
        assert len(hist2) == 5, f"Tổng lịch sử sau resume phải đủ 5 epochs, nhận được {len(hist2)}"
        df_csv = pd.read_csv(ckpt_dir / "history.csv")
        assert list(df_csv["epoch"]) == [1, 2, 3, 4, 5], f"Cột epoch trong history.csv bị sai: {list(df_csv['epoch'])}"
        assert hist2[-1]["train_loss"] < hist2[0]["train_loss"], "Train Loss phải giảm!"
        assert hist2[-1]["val_loss"] < hist2[0]["val_loss"], "Val Loss phải giảm!"
        assert hist2[-1]["train_dice"] > 0.80, f"Train Dice quá thấp: {hist2[-1]['train_dice']}"
        assert (ckpt_dir / "best_model.pth").exists(), "Thiếu file best_model.pth"
        assert (ckpt_dir / "last_model.pth").exists(), "Thiếu file last_model.pth"
        print("✅ selftest trainer.py (bao gồm Resume tự động): OK!")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="Kiểm thử tự động toàn bộ luồng huấn luyện & resume")
    s.set_defaults(fn=_cmd_selftest)

    o = sub.add_parser("overfit", help="Train thử trên vài lát cắt có u để kiểm tra hội tụ")
    o.add_argument("--cache-dir", default="data/cache_1mm")
    o.add_argument("--save-dir", default="checkpoints/overfit_test")
    o.add_argument("--num-samples", type=int, default=4, help="Số lát cắt dùng để overfit")
    o.add_argument("--batch-size", type=int, default=2)
    o.add_argument("--epochs", type=int, default=15)
    o.add_argument("--lr", type=float, default=1e-3)
    o.add_argument("--features", type=int, default=32, help="Số filter cơ sở của DuneUNet (32 nhẹ hơn, 64 chuẩn)")
    o.add_argument("--backbone", default="none", help="none | mambavision_t/t2/s/b/l/l2 | mambavision_mock")
    o.add_argument("--no-pretrained", action="store_true")
    o.add_argument("--resume", default=None, help="Đường dẫn file .pth hoặc 'auto' để tự resume từ last_model.pth")
    o.add_argument("--gpu-window", action="store_true", help="Chạy window_hu_torch trên GPU")
    o.add_argument("--no-amp", action="store_true")
    o.add_argument("--no-progress", action="store_true", help="Tắt thanh tiến trình tqdm từng batch")
    o.add_argument("--seed", type=int, default=DEFAULT_SEED)
    o.set_defaults(fn=_cmd_overfit)

    t = sub.add_parser("train", help="Huấn luyện toàn bộ dataset")
    t.add_argument("--cache-dir", default="data/cache_1mm")
    t.add_argument("--save-dir", default="checkpoints/dune_unet_25d")
    t.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    t.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    t.add_argument("--lr", type=float, default=DEFAULT_LR)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--features", type=int, default=64)
    t.add_argument("--backbone", default="none", help="none | mambavision_t/t2/s/b/l/l2 | mambavision_mock")
    t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--freeze-backbone-epochs", type=int, default=0)
    t.add_argument("--backbone-lr-mult", type=float, default=0.1)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--patience", type=int, default=10, help="Early stopping patience (0 = tắt)")
    t.add_argument("--loss", choices=["dice_bce", "tversky"], default="dice_bce")
    t.add_argument("--resume", default=None, help="Đường dẫn file .pth hoặc 'auto' để tự resume từ last_model.pth")
    t.add_argument("--gpu-window", action="store_true", help="Chuyển int16 lên GPU rồi gọi window_hu_torch")
    t.add_argument("--no-lung-mask", action="store_true")
    t.add_argument("--no-augment", action="store_true")
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--no-progress", action="store_true", help="Tắt thanh tiến trình tqdm từng batch")
    t.add_argument("--seed", type=int, default=DEFAULT_SEED)
    t.set_defaults(fn=_cmd_train)

    args = ap.parse_args()
    args.fn(args)