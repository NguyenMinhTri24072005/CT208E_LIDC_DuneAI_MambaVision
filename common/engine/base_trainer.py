# -*- coding: utf-8 -*-
"""
common/engine/base_trainer.py - Vòng lặp huấn luyện chuẩn BaseTrainer dùng chung cho toàn dự án.

Đặc điểm kiến trúc (spec mục 4.4):
  - Nhận `criterion` (nn.Module đã dựng) và `metric_fn` từ bên ngoài qua tiêm phụ thuộc (Dependency Injection).
  - Tuyệt đối KHÔNG import bất kỳ hàm loss hay activation nào, không hard-code tên loss/activation.
  - Tự động phát hiện multi-GPU (nn.DataParallel), bóc tách _unwrap_model() khi lưu checkpoint.
  - Quản lý AMP (Mixed Precision GradScaler), gradient clipping, Early Stopping, đồng bộ Cosine Scheduler.
  - Bắt KeyboardInterrupt an toàn, bảo tồn checkpoint và lịch sử huấn luyện.
  - Điểm mở rộng cho lớp con kiến trúc:
      * checkpoint_meta(): trả về dict chứa arch và model_config
      * on_epoch_start(epoch): hook xử lý đầu epoch (đóng/mở băng backbone)
      * build_param_groups(lr): phân nhóm learning rate cho backbone/decoder
"""

import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.metrics import compute_segmentation_metrics
from common.engine.checkpoint_io import export_weights_fp16, load_checkpoint_file, save_checkpoint
from common.engine.run_manager import RunHandle
from common.engine.utils import format_duration


def _make_grad_scaler(enabled: bool):
    """Tạo GradScaler thích ứng phiên bản PyTorch."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (TypeError, ValueError):
            pass
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    return None


def _autocast_context(enabled: bool):
    """Tạo autocast context an toàn."""
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=True)
        except (TypeError, ValueError):
            pass
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


class BaseTrainer:
    """Lớp quản lý huấn luyện lõi cho các kiến trúc mạng."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metric_fn: Callable = compute_segmentation_metrics,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        save_dir: Optional[Union[str, Path]] = None,
        run_handle: Optional[RunHandle] = None,
        grad_clip: float = 1.0,
        show_progress: bool = True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.type == "cuda"

        # Tự động kích hoạt DataParallel khi có nhiều GPU
        if self.is_cuda and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
            print(f"🔥 Phát hiện {torch.cuda.device_count()} GPUs! Kích hoạt chế độ song song nn.DataParallel.")
            model = nn.DataParallel(model)

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_fn = metric_fn or compute_segmentation_metrics

        self.use_amp = bool(use_amp and self.is_cuda)
        self.scaler = _make_grad_scaler(self.use_amp)

        self.run_handle = run_handle
        if run_handle:
            self.save_dir = run_handle.run_dir
        elif save_dir:
            self.save_dir = Path(save_dir).resolve()
        else:
            self.save_dir = Path("checkpoints").resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.grad_clip = float(grad_clip) if grad_clip is not None else 0.0
        self.show_progress = bool(show_progress)

        self.start_epoch = 1
        self.best_val_dice = -1.0
        self.best_epoch = None
        self.no_improve = 0
        self.history: List[Dict[str, Any]] = []

    # ==================================================================
    # CÁC ĐIỂM MỞ RỘNG (HOOKS) CHO LỚP CON KIẾN TRÚC
    # ==================================================================
    def checkpoint_meta(self) -> Dict[str, Any]:
        """Lớp con ghi đè để trả về thông tin kiến trúc (arch, model_config)."""
        return {}

    def on_epoch_start(self, epoch: int):
        """Hook chạy trước mỗi epoch (vd: đóng/mở băng backbone)."""
        pass

    def build_param_groups(self, lr: float) -> List[Dict[str, Any]]:
        """Lớp con ghi đè để phân chia tham số (vd: decoder lr, backbone lr)."""
        return [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": lr}]

    # ==================================================================
    # LOGIC NỘI BỘ
    # ==================================================================
    def _unwrap_model(self) -> nn.Module:
        """Bóc lớp DataParallel để lưu checkpoint thuần nhất."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = batch["image"].to(self.device, non_blocking=self.is_cuda)
        masks = batch["mask"].to(self.device, non_blocking=self.is_cuda)
        if images.dtype in (torch.int16, torch.int32, torch.int64):
            from common.data_pipeline.medical_prep import window_hu_torch
            images = window_hu_torch(images)
        return images, masks

    def run_epoch(self, loader, is_train: bool = True, epoch: Optional[int] = None, total_epochs: Optional[int] = None) -> Dict[str, Any]:
        self.model.train(is_train)
        total_loss = 0.0
        total_samples = 0
        inter_sum = 0.0
        pred_sum = 0.0
        target_sum = 0.0
        sub_loss_sums: Dict[str, float] = {}

        phase = "Train" if is_train else "Val  "
        ep_prefix = f"Epoch [{epoch:02d}/{total_epochs:02d}] {phase}" if (epoch and total_epochs) else phase

        use_bar = self.show_progress and (tqdm is not None) and sys.stderr.isatty()
        iterator = tqdm(loader, desc=ep_prefix, leave=False, dynamic_ncols=True) if use_bar else loader

        grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
        with grad_ctx:
            for batch in iterator:
                images, masks = self._prepare_batch(batch)
                bs = images.size(0)

                if is_train and self.optimizer:
                    self.optimizer.zero_grad(set_to_none=True)

                with _autocast_context(self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                if is_train and self.optimizer:
                    if self.scaler is not None and self.use_amp:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip > 0:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                total_loss += float(loss.item()) * bs
                total_samples += bs

                # Thu thập sub-losses nếu hàm criterion hỗ trợ
                if hasattr(self.criterion, "get_components"):
                    comps = self.criterion.get_components()
                    for k, v in comps.items():
                        sub_loss_sums[k] = sub_loss_sums.get(k, 0.0) + float(v) * bs
                elif hasattr(self.criterion, "last_components"):
                    comps = self.criterion.last_components
                    for k, v in comps.items():
                        sub_loss_sums[k] = sub_loss_sums.get(k, 0.0) + float(v) * bs

                m = self.metric_fn(logits.detach(), masks)
                inter_sum += m["intersection"]
                pred_sum += m["pred_sum"]
                target_sum += m["target_sum"]

                if use_bar:
                    running_loss = total_loss / max(1, total_samples)
                    running_dice = (2.0 * inter_sum + 1e-6) / (pred_sum + target_sum + 1e-6)
                    running_union = pred_sum + target_sum - inter_sum
                    running_iou = (inter_sum + 1e-6) / (running_union + 1e-6)
                    p_dict = {
                        "loss": f"{running_loss:.4f}",
                        "dice": f"{running_dice:.4f}",
                        "iou": f"{running_iou:.4f}",
                    }
                    if sub_loss_sums:
                        for k, v in sub_loss_sums.items():
                            p_dict[k] = f"{(v / max(1, total_samples)):.4f}"
                    iterator.set_postfix(**p_dict)

        eps = 1e-6
        epoch_loss = total_loss / max(1, total_samples)
        epoch_dice = (2.0 * inter_sum + eps) / (pred_sum + target_sum + eps)
        union_sum = pred_sum + target_sum - inter_sum
        epoch_iou = (inter_sum + eps) / (union_sum + eps)

        out_metrics = {
            "loss": epoch_loss,
            "dice": epoch_dice,
            "iou": epoch_iou,
        }
        if sub_loss_sums:
            out_metrics["sub_losses"] = {k: (v / max(1, total_samples)) for k, v in sub_loss_sums.items()}

        return out_metrics

    def save_checkpoint(self, filename: str, epoch: int, val_metrics: Dict[str, float]) -> Path:
        """Lưu checkpoint và xuất fp16 nếu là best_model."""
        target_path = self.save_dir / filename
        meta = self.checkpoint_meta()
        extra = {
            "no_improve": self.no_improve,
            "val_metrics": val_metrics,
            "history": self.history,
        }
        opt_sd = self.optimizer.state_dict() if self.optimizer else None
        sch_sd = self.scheduler.state_dict() if self.scheduler else None
        scl_sd = self.scaler.state_dict() if (self.scaler and self.use_amp) else None

        save_checkpoint(
            path=target_path,
            state_dict=self._unwrap_model().state_dict(),
            optimizer_state=opt_sd,
            scheduler_state=sch_sd,
            scaler_state=scl_sd,
            epoch=epoch,
            best_val_dice=self.best_val_dice,
            meta=meta,
            extra=extra,
        )

        if filename == "best_model.pth":
            fp16_path = self.save_dir / "best_weights_fp16.pth"
            try:
                export_weights_fp16(target_path, fp16_path, meta=meta)
            except Exception as e:
                print(f"⚠️ Không xuất được best_weights_fp16.pth: {e}")

        return target_path

    def load_checkpoint(self, checkpoint_path: Union[str, Path], resume_training: bool = True) -> int:
        """Nạp checkpoint an toàn và kiểm tra tính tương thích của metadata."""
        p = Path(checkpoint_path).resolve()
        ckpt = load_checkpoint_file(p, map_location=str(self.device))

        # Kiểm tra metadata nếu có
        saved_meta = ckpt.get("meta", {})
        current_meta = self.checkpoint_meta()
        if saved_meta and current_meta:
            if "arch" in saved_meta and "arch" in current_meta:
                if saved_meta["arch"] != current_meta["arch"]:
                    raise ValueError(
                        f"❌ Không tương thích kiến trúc khi nạp checkpoint!\n"
                        f"  - Checkpoint: arch = '{saved_meta['arch']}'\n"
                        f"  - Hiện tại  : arch = '{current_meta['arch']}'\n"
                        f"Gợi ý: Kiểm tra lại tham số mô hình hoặc khởi tạo phiên chạy mới."
                    )
            if "features" in saved_meta and "features" in current_meta:
                if saved_meta["features"] != current_meta["features"]:
                    raise ValueError(
                        f"❌ Không tương thích số filter cơ sở (features)!\n"
                        f"  - Checkpoint: features = {saved_meta['features']}\n"
                        f"  - Hiện tại  : features = {current_meta['features']}"
                    )

        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self._unwrap_model().load_state_dict(sd)

        if not resume_training:
            print(f"📥 Đã nạp trọng số khởi tạo (init-weights) từ: {p.name}")
            return 1

        if self.optimizer and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"⚠️ Không nạp được optimizer state ({e}). Tiếp tục với optimizer mới.")

        if self.scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"⚠️ Không nạp được scheduler state ({e}).")

        if self.scaler and self.use_amp and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"]:
            try:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                pass

        last_epoch = int(ckpt.get("epoch", 0))
        self.start_epoch = last_epoch + 1
        self.best_val_dice = float(ckpt.get("best_val_dice", -1.0))
        self.no_improve = int(ckpt.get("no_improve", 0))
        self.history = ckpt.get("history", [])

        if self.run_handle:
            self.run_handle.record_resume_event(from_epoch=last_epoch)

        print(
            f"🔄 RESUME THÀNH CÔNG từ '{p.name}' | Đã xong epoch {last_epoch} "
            f"-> Tiếp tục từ epoch {self.start_epoch} | Best Val Dice: {self.best_val_dice:.4f}"
        )
        return self.start_epoch

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 50,
        early_stopping_patience: int = 0,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """Vòng lặp huấn luyện chính có hỗ trợ resume và bắt ngắt."""
        if resume_from:
            if str(resume_from).lower() == "auto":
                candidate = self.save_dir / "last_model.pth"
                if candidate.exists():
                    self.load_checkpoint(candidate, resume_training=True)
                else:
                    print(f"ℹ️  --resume auto: Chưa có '{candidate.name}', bắt đầu huấn luyện mới.")
            else:
                self.load_checkpoint(resume_from, resume_training=True)

        # Đồng bộ T_max cho CosineAnnealingLR khi tăng số epochs
        if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            if getattr(self.scheduler, "T_max", epochs) != epochs and epochs > 0:
                self.scheduler.T_max = max(1, epochs)

        if self.start_epoch > epochs:
            print(f"✅ Đã hoàn thành {self.start_epoch - 1}/{epochs} epochs trong checkpoint trước đó.")
            if self.run_handle:
                self.run_handle.finish(status="completed")
            return self.history

        epoch_durations = []
        fit_start_time = time.time()

        try:
            for epoch in range(self.start_epoch, epochs + 1):
                ep_t0 = time.time()
                self.on_epoch_start(epoch)

                tr_t0 = time.time()
                train_m = self.run_epoch(train_loader, is_train=True, epoch=epoch, total_epochs=epochs)
                train_time = time.time() - tr_t0

                va_t0 = time.time()
                if val_loader is not None:
                    val_m = self.run_epoch(val_loader, is_train=False, epoch=epoch, total_epochs=epochs)
                else:
                    val_m = train_m
                val_time = time.time() - va_t0

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_m["dice"])
                    else:
                        self.scheduler.step()

                lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
                elapsed = time.time() - ep_t0
                epoch_durations.append(elapsed)

                recent_avg_sec = float(np.mean(epoch_durations[-5:]))
                eta_sec = recent_avg_sec * (epochs - epoch)

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
                # Bổ sung các thành phần sub-losses vào history nếu có
                for k, v in train_m.get("sub_losses", {}).items():
                    row[f"train_{k}"] = round(v, 5)
                for k, v in val_m.get("sub_losses", {}).items():
                    row[f"val_{k}"] = round(v, 5)

                self.history = [r for r in self.history if int(r.get("epoch", -1)) != epoch] + [row]
                from common.engine.utils import atomic_save_json
                df_history = pd.DataFrame(self.history)
                csv_tmp = self.save_dir / f".tmp_history_{os.getpid()}.csv"
                df_history.to_csv(csv_tmp, index=False)
                os.replace(str(csv_tmp), str(self.save_dir / "history.csv"))

                is_best = val_m["dice"] > self.best_val_dice
                if is_best:
                    self.best_val_dice = float(val_m["dice"])
                    self.best_epoch = epoch
                    self.no_improve = 0
                    self.save_checkpoint("best_model.pth", epoch, val_m)
                else:
                    self.no_improve += 1

                self.save_checkpoint("last_model.pth", epoch, val_m)

                if self.run_handle:
                    self.run_handle.update_result(
                        epoch=epoch,
                        best_val_dice=self.best_val_dice,
                        best_epoch=self.best_epoch,
                        last_val_loss=val_m["loss"],
                        last_val_dice=val_m["dice"],
                        last_val_iou=val_m["iou"],
                    )

                def _fmt_phase(m: Dict[str, Any]) -> str:
                    s = f"Loss: {m['loss']:.4f}"
                    if m.get("sub_losses"):
                        sub_str = ", ".join(f"{k}: {v:.4f}" for k, v in m["sub_losses"].items())
                        s += f" ({sub_str})"
                    s += f", Dice: {m['dice']:.4f}, IoU: {m['iou']:.4f}"
                    return s

                badge = "⭐ [BEST]" if is_best else ""
                log_msg = (
                    f"Epoch [{epoch:02d}/{epochs:02d}] | LR: {lr:.2e} | "
                    f"Train [{_fmt_phase(train_m)}] | "
                    f"Val [{_fmt_phase(val_m)}] | "
                    f"Time: {format_duration(elapsed)} | ETA: {format_duration(eta_sec)} {badge}"
                )
                if self.run_handle:
                    self.run_handle.log(log_msg)
                else:
                    print(log_msg)

                if early_stopping_patience > 0 and self.no_improve >= early_stopping_patience:
                    stop_msg = f"⏹️ Dừng sớm (Early Stopping) tại epoch {epoch} sau {early_stopping_patience} epochs không cải thiện."
                    if self.run_handle:
                        self.run_handle.log(stop_msg)
                    else:
                        print(stop_msg)
                    break

            if self.run_handle:
                self.run_handle.finish(status="completed")

        except KeyboardInterrupt:
            last_saved = self.history[-1]["epoch"] if self.history else (self.start_epoch - 1)
            msg = (
                f"\n⚠️ ĐÃ NHẬN TÍN HIỆU DỪNG (KeyboardInterrupt)!\n"
                f"   - Checkpoints đã được bảo toàn an toàn tại: {self.save_dir}\n"
                f"   - Hoàn tất an toàn tới hết Epoch {last_saved}."
            )
            if self.run_handle:
                self.run_handle.log(msg)
                self.run_handle.finish(status="interrupted")
            else:
                print(msg)
            return self.history

        except Exception as e:
            if self.run_handle:
                self.run_handle.log(f"❌ Ngoại lệ trong khi huấn luyện: {e}")
                self.run_handle.finish(status="failed")
            raise e

        total_time = time.time() - fit_start_time
        final_msg = f"🎉 Hoàn tất fit! Best Val Dice: {self.best_val_dice:.4f} | Tổng thời gian: {format_duration(total_time)}"
        if self.run_handle:
            self.run_handle.log(final_msg)
        else:
            print(final_msg)

        return self.history
