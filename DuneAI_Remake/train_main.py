# -*- coding: utf-8 -*-
"""
train_main.py - Điểm vào huấn luyện chính thức cho DuneAI 2.5D (baseline U-Net).

Việc script này làm:
  1. Đọc siêu tham số từ configs/config.py (có thể ghi đè bằng tham số dòng lệnh).
  2. Kiểm tra cache_1mm và tự tạo slices.csv nếu chưa có (hoặc khi dùng --rebuild-index).
  3. Dựng DataLoader (train có augment, val không), mô hình, loss, optimizer, scheduler.
  4. Hỗ trợ khôi phục huấn luyện (--resume auto hoặc --resume <path>) khi bị ngắt kết nối trên Colab/Kaggle,
     hoặc khi muốn tăng --epochs để train tiếp từ checkpoint trước đó.
  5. Lưu config.json (để tái lập thí nghiệm) rồi gọi Trainer.fit.
  6. Kết quả: best_model.pth, last_model.pth, history.csv, config.json trong --save-dir.

Dùng:
  python DuneAI_Remake/train_main.py selftest
  python DuneAI_Remake/train_main.py train --cache-dir data/cache_1mm --epochs 50 --batch-size 8 --resume auto
  python DuneAI_Remake/train_main.py train --cache-dir data/cache_1mm --epochs 2 --limit-train 16 --limit-val 8
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import configs.config as cfg
from dataset import LIDC25DDataset
from models.dune_unet import DuneUNet
from offline_builder import build_dataset_index
from trainer import Trainer, _build_criterion, seed_everything

DEFAULT_BATCH_SIZE = getattr(cfg, "BATCH_SIZE", 8)
DEFAULT_EPOCHS = getattr(cfg, "EPOCHS", 50)
DEFAULT_LR = getattr(cfg, "LR", getattr(cfg, "LEARNING_RATE", 1e-4))
DEFAULT_SEED = getattr(cfg, "SEED", 42)
DEFAULT_IMAGE_SIZE = getattr(cfg, "IMAGE_SIZE", 512)


# ======================================================================
# CÁC HÀM DỰNG THÀNH PHẦN
# ======================================================================
def ensure_index(cache_dir, rebuild=False):
    """Đảm bảo có slices.csv. Trả về đường dẫn tới file."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục cache: {cache_dir.resolve()}\n"
            f"Hãy chạy `python DuneAI_Remake/src/medical_prep.py build --extract-lungs` trước, "
            f"hoặc kiểm tra lại --cache-dir (đường dẫn tương đối tính từ thư mục bạn đang đứng)."
        )
    csv_path = cache_dir / "slices.csv"
    if rebuild or not csv_path.exists():
        print(f"📋 {'Dựng lại' if rebuild else 'Chưa có'} slices.csv -> đang tạo chỉ mục...")
        build_dataset_index(cache_dir)
    return csv_path


def summarize_index(csv_path):
    """In thống kê chia tập để phát hiện sớm tình huống mất cân bằng hoặc thiếu val."""
    df = pd.read_csv(csv_path)
    print("-" * 65)
    for split in ("train", "val"):
        sub = df[df["split"] == split]
        pos = int((sub["has_tumor"] == 1).sum())
        print(
            f"  {split:<5}: {len(sub):>6} lát | {sub['patient_id'].nunique():>4} bệnh nhân "
            f"| có u: {pos} | âm tính: {len(sub) - pos}"
        )
    print("-" * 65)
    return df


def make_loaders(
    cache_dir,
    batch_size,
    workers,
    image_size,
    apply_lung,
    window_on_cpu,
    augment,
    limit_train=0,
    limit_val=0,
    seed=DEFAULT_SEED,
):
    """Dựng train/val loader. Nếu không có mẫu val thì trả val_loader=None (Trainer tự xử lý)."""
    train_ds = LIDC25DDataset(
        cache_dir, split="train", in_channels=3, image_size=image_size,
        apply_lung=apply_lung, window_on_cpu=window_on_cpu, augment=augment,
    )
    try:
        val_ds = LIDC25DDataset(
            cache_dir, split="val", in_channels=3, image_size=image_size,
            apply_lung=apply_lung, window_on_cpu=window_on_cpu, augment=False,
        )
    except ValueError:
        print("⚠️  Không có mẫu nào thuộc tập val -> dùng chỉ số train thay cho val (chỉ nên xảy ra khi thử nghiệm).")
        val_ds = None

    rng = np.random.RandomState(seed)

    def _limit(ds, n):
        if ds is None or n <= 0 or n >= len(ds):
            return ds
        return Subset(ds, sorted(rng.choice(len(ds), size=n, replace=False).tolist()))

    train_ds, val_ds = _limit(train_ds, limit_train), _limit(val_ds, limit_val)

    pin = torch.cuda.is_available()
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=pin, drop_last=len(train_ds) > batch_size, generator=gen,
        persistent_workers=workers > 0,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=pin, persistent_workers=workers > 0,
        )
    return train_loader, val_loader


def make_scheduler(name, optimizer, epochs):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    return None


def load_init_weights(model, path):
    """Nạp trọng số model (không nạp optimizer/epoch) để fine-tune sang kịch bản mới."""
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"📥 Đã nạp trọng số khởi tạo (init-weights) từ: {path}")


# ======================================================================
# LỆNH TRAIN
# ======================================================================
def run_train(a):
    seed_everything(a.seed)
    csv_path = ensure_index(a.cache_dir, rebuild=a.rebuild_index)
    summarize_index(csv_path)

    train_loader, val_loader = make_loaders(
        cache_dir=a.cache_dir,
        batch_size=a.batch_size,
        workers=a.workers,
        image_size=a.image_size,
        apply_lung=not a.no_lung_mask,
        window_on_cpu=not a.gpu_window,
        augment=not a.no_augment,
        limit_train=a.limit_train,
        limit_val=a.limit_val,
        seed=a.seed,
    )

    n_tr = len(train_loader.dataset)
    n_va = len(val_loader.dataset) if val_loader is not None else 0
    print(f"📦 Số lát dùng thực tế: train={n_tr} | val={n_va}" + (" (đã giới hạn bởi --limit-*)" if (a.limit_train or a.limit_val) else ""))

    # Xác định chế độ resume (--auto-resume là viết tắt của --resume auto)
    resume_target = "auto" if getattr(a, "auto_resume", False) and not a.resume else a.resume

    model = DuneUNet(in_channels=3, classes=1, features=a.features)
    if a.init_weights and not resume_target:
        load_init_weights(model, a.init_weights)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = _build_criterion(a.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = make_scheduler(a.scheduler, optimizer, a.epochs)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=not a.no_amp,
        save_dir=a.save_dir,
        grad_clip=a.grad_clip,
        show_progress=not getattr(a, "no_progress", False),
    )

    save_dir = Path(a.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = save_dir / "config.json"
    prev_started_at = None
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                prev_started_at = json.load(f).get("started_at")
        except Exception:
            prev_started_at = None

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    run_info = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "fn"},
        "device": str(trainer.device),
        "amp": trainer.use_amp,
        "num_params": n_params,
        "train_samples": n_tr,
        "val_samples": n_va,
        "resume_target": str(resume_target) if resume_target else None,
        "started_at": prev_started_at or now_str,
        "last_updated_at": now_str,
    }
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print(
        f"🧠 DuneUNet(features={a.features}) | {n_params:,} tham số | "
        f"loss={a.loss} | lr={a.lr} | scheduler={a.scheduler} | batch={a.batch_size}"
    )
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=a.epochs,
        early_stopping_patience=a.patience,
        resume_from=resume_target,
    )
    print(f"💾 Checkpoint tốt nhất: {save_dir / 'best_model.pth'}")
    print(f"💾 Checkpoint mới nhất (để resume): {save_dir / 'last_model.pth'}")
    return history


# ======================================================================
# SELFTEST: chạy đúng đường code của `train` trên dữ liệu giả lập
# ======================================================================
def _make_synthetic_cache(tmp, n_patients=4):
    for i in range(1, n_patients + 1):
        p_dir = tmp / f"LIDC-IDRI-{i:04d}"
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


def _default_args(**over):
    base = dict(
        cache_dir="", save_dir="", batch_size=4, epochs=3, lr=3e-3, weight_decay=1e-4,
        features=8, workers=0, patience=0, loss="dice_bce", scheduler="cosine",
        grad_clip=1.0, image_size=128, limit_train=0, limit_val=0, init_weights=None,
        resume=None, auto_resume=False, rebuild_index=False, gpu_window=False,
        no_lung_mask=False, no_augment=True, no_amp=True, no_progress=True, seed=DEFAULT_SEED,
    )
    base.update(over)
    return argparse.Namespace(**base)


def run_selftest(_):
    print("Đang chạy kiểm thử toàn trình train_main.py (dữ liệu giả lập + Resume)...")
    tmp = Path(tempfile.mkdtemp())
    try:
        cache = tmp / "cache"
        _make_synthetic_cache(cache)
        save = tmp / "ckpt"

        # 1) Chưa có slices.csv -> tự tạo; chạy 3 epochs đầu tiên (giả lập bị ngắt kết nối ở epoch 3)
        assert not (cache / "slices.csv").exists()
        a1 = _default_args(cache_dir=str(cache), save_dir=str(save), epochs=3, resume="auto")
        hist1 = run_train(a1)
        assert (cache / "slices.csv").exists(), "Không tự tạo slices.csv"
        for fn in ("best_model.pth", "last_model.pth", "history.csv", "config.json"):
            assert (save / fn).exists(), f"Thiếu {fn}"
        assert len(hist1) == 3

        # 2) Giả lập kết nối lại Colab: chạy tiếp lên 6 epochs với --resume auto
        a_resume = _default_args(cache_dir=str(cache), save_dir=str(save), epochs=6, resume="auto")
        hist = run_train(a_resume)
        assert len(hist) == 6, f"Sau resume phải có đủ 6 epochs, nhận được {len(hist)}"
        df_csv = pd.read_csv(save / "history.csv")
        assert list(df_csv["epoch"]) == [1, 2, 3, 4, 5, 6], f"Thứ tự epoch sai: {list(df_csv['epoch'])}"
        assert hist[-1]["train_loss"] < hist[0]["train_loss"], "Train loss phải giảm"
        assert all(
            np.isfinite(r["train_loss"]) and np.isfinite(r["val_loss"])
            and "train_time_sec" in r and "eta_sec" in r
            for r in hist
        ), "Thiếu thông tin loss hoặc thời gian"

        # 3) Gọi lại --resume auto khi đã đủ 6 epochs (phải thoát êm mà không lỗi)
        hist_done = run_train(a_resume)
        assert len(hist_done) == 6

        # 4) Nạp lại trọng số từ best_model.pth bằng --init-weights rồi chạy kịch bản mới
        save2 = tmp / "ckpt2"
        a2 = _default_args(
            cache_dir=str(cache), save_dir=str(save2), epochs=2, scheduler="plateau",
            loss="tversky", init_weights=str(save / "best_model.pth"), limit_train=6,
        )
        hist2 = run_train(a2)
        assert len(hist2) == 2 and all(np.isfinite(r["train_loss"]) for r in hist2)

        # 5) Chế độ không có scheduler + dừng sớm
        a3 = _default_args(
            cache_dir=str(cache), save_dir=str(tmp / "ckpt3"), epochs=3,
            scheduler="none", patience=1,
        )
        hist3 = run_train(a3)
        assert 1 <= len(hist3) <= 3

        print(
            f"✅ selftest train_main.py: OK! "
            f"(run 1+resume: train_loss {hist[0]['train_loss']:.4f} -> {hist[-1]['train_loss']:.4f}, "
            f"train_dice cuối = {hist[-1]['train_dice']:.4f})"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ======================================================================
# CLI
# ======================================================================
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="Kiểm thử toàn trình bằng dữ liệu giả lập (bao gồm cả Resume)")
    s.set_defaults(fn=run_selftest)

    t = sub.add_parser("train", help="Huấn luyện chính thức trên cache_1mm")
    t.add_argument("--cache-dir", default="data/cache_1mm")
    t.add_argument("--save-dir", default="checkpoints/dune_unet_25d")
    t.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    t.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    t.add_argument("--lr", type=float, default=DEFAULT_LR)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--grad-clip", type=float, default=1.0, help="0 = tắt")
    t.add_argument("--features", type=int, default=64, help="Số filter cơ sở (32 nhẹ, 64 chuẩn)")
    t.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE,
                   help="Kích thước sau pad/crop ở giữa (KHÔNG resize; giữ 512 khi train thật)")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--patience", type=int, default=10, help="Early stopping patience (0 = tắt)")
    t.add_argument("--loss", choices=["dice_bce", "tversky"], default="dice_bce")
    t.add_argument("--scheduler", choices=["cosine", "plateau", "none"], default="cosine")
    t.add_argument("--resume", default=None,
                   help="Đường dẫn tới last_model.pth hoặc truyền 'auto' để tự động khôi phục từ --save-dir")
    t.add_argument("--auto-resume", action="store_true",
                   help="Tự động resume nếu tìm thấy <save_dir>/last_model.pth (tương đương --resume auto)")
    t.add_argument("--init-weights", default=None,
                   help="Chỉ nạp trọng số model từ checkpoint (reset epoch về 1, không nạp optimizer)")
    t.add_argument("--limit-train", type=int, default=0, help="Chỉ dùng N lát train ngẫu nhiên (0 = tất cả)")
    t.add_argument("--limit-val", type=int, default=0, help="Chỉ dùng N lát val ngẫu nhiên (0 = tất cả)")
    t.add_argument("--rebuild-index", action="store_true", help="Dựng lại slices.csv trước khi train")
    t.add_argument("--gpu-window", action="store_true", help="Chuyển int16 lên GPU rồi gọi window_hu_torch")
    t.add_argument("--no-lung-mask", action="store_true")
    t.add_argument("--no-augment", action="store_true")
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--no-progress", action="store_true", help="Tắt thanh tiến trình tqdm từng batch")
    t.add_argument("--seed", type=int, default=DEFAULT_SEED)
    t.set_defaults(fn=run_train)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)