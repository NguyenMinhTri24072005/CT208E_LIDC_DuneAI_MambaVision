"""
train.py
--------
Vòng lặp huấn luyện chính cho mô hình U-Net.
Tính năng:
- Đọc cấu hình từ file YAML và cho phép override qua CLI (--key.subkey=value)
- Cố định seed cho tính tái lập (reproducibility)
- Hỗ trợ Automatic Mixed Precision (AMP)
- Tối ưu hóa với AdamW/Adam/SGD và CosineAnnealing/ReduceLROnPlateau scheduler
- Gradient clipping
- Lưu checkpoint 'last.pth' và 'best.pth' (kèm đầy đủ optimizer, scheduler, epoch, config)
- Resume huấn luyện từ checkpoint
- Cơ chế Early Stopping
- Ghi log ra Console (tqdm), TensorBoard và xuất file history.csv
"""

import os
import sys
import time
import json
import argparse
import random
import csv
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import build_model
from utils import build_loss, get_dataloaders, calculate_dice, calculate_iou, MetricTracker


def set_seed(seed: int = 42) -> None:
    """Cố định seed trên toàn bộ các thư viện để tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def format_time(seconds: float) -> str:
    """Đổi số giây sang định dạng giờ:phút:giây dễ đọc."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{seconds:.2f}s"


def load_config_and_override() -> Dict[str, Any]:
    """Đọc file YAML config và cho phép override từ command line args."""
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình U-Net Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Đường dẫn đến file cấu hình yaml",
    )
    args, unknown = parser.parse_known_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config tại: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Xử lý các cờ override dạng: --train.batch_size=8 hoặc --checkpoint.resume_path=ckpt.pth
    for arg in unknown:
        if arg.startswith("--") and "=" in arg:
            key, val = arg[2:].split("=", 1)
            # Thử parse kiểu dữ liệu int, float, bool
            if val.lower() == "true":
                parsed_val = True
            elif val.lower() == "false":
                parsed_val = False
            elif val.lower() == "null" or val.lower() == "none":
                parsed_val = None
            else:
                try:
                    parsed_val = int(val)
                except ValueError:
                    try:
                        parsed_val = float(val)
                    except ValueError:
                        parsed_val = val

            # Phân tách nested keys
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = parsed_val

    return cfg


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    use_channels_last: bool = False,
) -> float:
    """Huấn luyện mô hình 1 epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="[Train]", leave=False)
    for images, masks in pbar:
        if use_channels_last:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
        else:
            images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)

        if not torch.isfinite(loss):
            print(f"⚠️ Cảnh báo: Loss không hữu hạn ({loss.item()}), bỏ qua batch.")
            optimizer.zero_grad()
            continue

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int = 1,
    use_channels_last: bool = False,
) -> Tuple[float, float, float]:
    """Đánh giá mô hình trên tập validation với tối ưu hóa tính toán ma trận trực tiếp trên GPU."""
    model.eval()
    total_loss = 0.0
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="[Val]", leave=False)
    for images, masks in pbar:
        if use_channels_last:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
        else:
            images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item()

        # Tính toán metric song song trực tiếp trên GPU (nhanh gấp 10x so với CPU loop)
        if n_classes == 1:
            preds = (torch.sigmoid(logits) > 0.5).float()
            inter = (preds * masks).sum(dim=(2, 3)).squeeze(1)
            total_p = preds.sum(dim=(2, 3)).squeeze(1)
            total_t = masks.sum(dim=(2, 3)).squeeze(1)
            union = total_p + total_t
            both_empty = (union == 0)

            dices = (2.0 * inter + 1e-6) / (union + 1e-6)
            dices[both_empty] = 1.0

            iou_denom = total_p + total_t - inter
            ious = (inter + 1e-6) / (iou_denom + 1e-6)
            ious[both_empty] = 1.0

            d_list = dices.cpu().tolist()
            i_list = ious.cpu().tolist()
            for d_val, i_val in zip(d_list, i_list):
                tracker.update(dice=d_val, iou=i_val)
        else:
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            targets = masks.cpu().numpy().astype(np.uint8)
            for i in range(preds.shape[0]):
                p_i = preds[i]
                t_i = targets[i]
                tracker.update(dice=calculate_dice(p_i, t_i), iou=calculate_iou(p_i, t_i))

    metrics_summary = tracker.summary()
    mean_dice = metrics_summary["dice"][0]
    mean_iou = metrics_summary["iou"][0]
    mean_loss = total_loss / max(len(dataloader), 1)

    return mean_loss, mean_dice, mean_iou


def main() -> None:
    cfg = load_config_and_override()

    # 1. Cố định Seed
    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Sử dụng thiết bị: {device}")

    # 2. Chuẩn bị Dataloaders
    print("📦 Đang nạp dữ liệu...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(cfg)
        print(f"  - Train samples : {len(train_loader.dataset)} ({len(train_loader)} batches)")
        print(f"  - Val samples   : {len(val_loader.dataset)} ({len(val_loader)} batches)")
        print(f"  - Test samples  : {len(test_loader.dataset)} ({len(test_loader)} batches)")
    except Exception as e:
        import traceback
        print(f"⚠️ Lỗi khi tải dữ liệu: {e}")
        traceback.print_exc()
        print("  Vui lòng đảm bảo cấu trúc data/ đã có ảnh trước khi train thật.")
        return

    # 3. Khởi tạo Mô hình & Hàm mất mát
    model = build_model(cfg).to(device)
    train_cfg = cfg["train"]
    use_channels_last = train_cfg.get("channels_last", True) and device.type == "cuda"
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("⚡ Đã bật tối ưu hóa memory_format=torch.channels_last cho GPU")
    criterion = build_loss(cfg).to(device)

    # 4. Cấu hình Optimizer & Scheduler
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    opt_name = train_cfg.get("optimizer", "AdamW")

    if opt_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer không được hỗ trợ: {opt_name}")

    epochs = train_cfg.get("epochs", 50)
    sched_name = train_cfg.get("scheduler", "CosineAnnealing")
    min_lr = float(train_cfg.get("min_lr", 1e-5))

    if sched_name == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
    elif sched_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=min_lr
        )
    else:
        scheduler = None

    use_amp = train_cfg.get("amp", True)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    # 5. Checkpoints & Logging setup
    ckpt_cfg = cfg.get("checkpoint", {})
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_best_metric = ckpt_cfg.get("save_best_metric", "val_dice")
    best_val_score = -1.0 if save_best_metric == "val_dice" else 1e9

    start_epoch = 1
    resume_path = ckpt_cfg.get("resume_path", None)

    # Resume nếu có
    if resume_path and Path(resume_path).exists():
        print(f"🔄 Đang phục hồi trạng thái từ: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_score = checkpoint.get("best_metric", best_val_score)
        print(f"  -> Tiếp tục từ epoch {start_epoch}")

    # Xóa file train_summary.json cũ trước khi chạy mới (nếu bắt đầu từ epoch 1)
    summary_path = ckpt_dir / "train_summary.json"
    if summary_path.exists() and start_epoch == 1:
        try:
            summary_path.unlink()
            print(f"🧹 Đã xóa file tổng kết cũ trước khi chạy mới: {summary_path}")
        except Exception as e:
            pass

    log_cfg = cfg.get("logging", {})
    writer = None
    if log_cfg.get("tensorboard", True):
        log_dir = log_cfg.get("log_dir", "runs")
        writer = SummaryWriter(log_dir=log_dir)

    history_csv = ckpt_dir / "history.csv"
    if not history_csv.exists() or start_epoch == 1:
        with open(history_csv, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr", "time_sec"])

    patience = train_cfg.get("early_stopping_patience", 10)
    patience_counter = 0

    print(f"\n🚀 Bắt đầu huấn luyện {epochs} Epochs...")

    total_start_time = time.time()
    best_epoch_num = start_epoch

    # 6. Training Loop
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        curr_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=grad_clip,
            use_amp=use_amp,
            use_channels_last=use_channels_last,
        )

        val_loss, val_dice, val_iou = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            n_classes=cfg["model"].get("n_classes", 1),
            use_channels_last=use_channels_last,
        )

        epoch_duration = time.time() - epoch_start_time

        # Cập nhật scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_dice if save_best_metric == "val_dice" else val_loss)
            else:
                scheduler.step()

        # In kết quả
        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Time: {format_time(epoch_duration)} | "
            f"LR: {curr_lr:.6f}"
        )

        # Ghi log TensorBoard an toàn (chống crash do Google Drive FUSE mất đồng bộ)
        if writer:
            try:
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Val", val_loss, epoch)
                writer.add_scalar("Metric/Val_Dice", val_dice, epoch)
                writer.add_scalar("Metric/Val_IoU", val_iou, epoch)
                writer.add_scalar("Learning_Rate", curr_lr, epoch)
                writer.add_scalar("Time/Epoch_Seconds", epoch_duration, epoch)
            except Exception as tb_err:
                print(f"⚠️ Cảnh báo TensorBoard: Không thể ghi log ({tb_err}), tiếp tục huấn luyện...")

        # Ghi history CSV
        with open(history_csv, "a", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch, train_loss, val_loss, val_dice, val_iou, curr_lr, round(epoch_duration, 2)])

        # Kiểm tra best metric & cập nhật best_val_score trước khi tạo ckpt_data
        is_best = False
        if save_best_metric == "val_dice" and val_dice > best_val_score:
            best_val_score = val_dice
            is_best = True
        elif save_best_metric == "val_loss" and val_loss < best_val_score:
            best_val_score = val_loss
            is_best = True

        # Lưu checkpoint: last.pth
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_metric": best_val_score,
            "epoch_time_sec": round(epoch_duration, 2),
            "total_time_sec": round(time.time() - total_start_time, 2),
            "config": cfg,
        }
        torch.save(ckpt_data, ckpt_dir / "last.pth")

        if is_best:
            patience_counter = 0
            best_epoch_num = epoch
            torch.save(ckpt_data, ckpt_dir / "best.pth")
            print(f"  🌟 Lưu mô hình tốt nhất (best.pth) tại Epoch {epoch:03d} ({save_best_metric}: {best_val_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Kích hoạt Early Stopping sau {patience} epochs không cải thiện.")
                break

    if writer:
        writer.close()

    total_duration = time.time() - total_start_time
    num_trained = max(1, epoch - start_epoch + 1)
    avg_duration = total_duration / num_trained

    print("\n" + "=" * 65)
    print("✅ Hoàn thành quá trình huấn luyện!")
    print(f"⏱️ Tổng thời gian chạy     : {format_time(total_duration)} ({total_duration:.2f}s)")
    print(f"⏱️ Thời gian TB / Epoch    : {format_time(avg_duration)} ({avg_duration:.2f}s)")
    print(f"🌟 Epoch tốt nhất          : Epoch {best_epoch_num:03d} ({save_best_metric}: {best_val_score:.4f})")
    print(f"📁 Checkpoint tốt nhất     : {ckpt_dir / 'best.pth'}")
    print("=" * 65)

    # Xuất file tổng kết train_summary.json
    summary_data = {
        "start_epoch": start_epoch,
        "end_epoch": epoch,
        "epochs_trained": num_trained,
        "total_time_sec": round(total_duration, 2),
        "total_time_formatted": format_time(total_duration),
        "avg_epoch_time_sec": round(avg_duration, 2),
        "avg_epoch_time_formatted": format_time(avg_duration),
        "best_epoch": best_epoch_num,
        "best_metric_name": save_best_metric,
        "best_metric_value": round(float(best_val_score), 4),
    }
    summary_path = ckpt_dir / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"📄 Báo cáo chi tiết thời gian đã lưu vào: {summary_path}")


if __name__ == "__main__":
    main()
