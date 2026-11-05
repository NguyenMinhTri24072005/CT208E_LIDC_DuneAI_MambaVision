# -*- coding: utf-8 -*-
"""
train_main.py - Điểm vào huấn luyện chính thức cho DuneAI 2.5D (baseline U-Net).

Việc script này làm:
  1. Đọc siêu tham số từ configs/config.py (có thể ghi đè bằng tham số dòng lệnh).
  2. Kiểm tra cache_1mm và tự tạo slices.csv nếu chưa có (hoặc khi dùng --rebuild-index).
  3. Dựng DataLoader (train có augment, val không), mô hình, loss, optimizer, scheduler.
  4. Lưu config.json (để tái lập thí nghiệm) rồi gọi Trainer.fit.
  5. Kết quả: best_model.pth, last_model.pth, history.csv, config.json trong --save-dir.

TIẾP TỤC HUẤN LUYỆN (--resume):
  --epochs luôn là TỔNG số epoch đích. Ví dụ đã chạy xong 30 epoch, muốn train thêm 20 -> --epochs 50.
  Resume khôi phục: trọng số, optimizer, scheduler, Val Dice tốt nhất, bộ đếm early stopping, history.csv.
  Hai tình huống:
    a) Bị ngắt giữa chừng (Ctrl+C, mất điện) và giữ nguyên --epochs  -> tiếp tục ĐÚNG đường learning rate cũ.
    b) Đổi --epochs (kéo dài)                                         -> với cosine, khởi động lại chu kỳ mới từ --lr
                                                                         (vì chu kỳ cũ đã về ~0, tiếp tục sẽ không học được nữa).
  Nên dùng --save-dir MỚI khi kéo dài, để không ghi đè best_model.pth của lượt trước.

Dùng:
  python DuneAI_Remake/train_main.py selftest
  python DuneAI_Remake/train_main.py train --cache-dir data/cache_1mm --epochs 50 --batch-size 8
  python DuneAI_Remake/train_main.py train --cache-dir data/cache_1mm --epochs 50 --resume auto          # nối tiếp last_model.pth trong --save-dir (nếu có)
  python DuneAI_Remake/train_main.py train --cache-dir data/cache_1mm --epochs 70 --resume checkpoints/dune_unet_25d/last_model.pth --save-dir checkpoints/dune_unet_25d_v2
  python DuneAI_Remake/train_main.py train --cache-dir data/cache_1mm --epochs 2 --limit-train 16 --limit-val 8   # chạy thử nhanh trên CPU

CHỌN BACKBONE (baseline hay MambaVision):
  --backbone none                      U-Net thuần CNN (mặc định, không cần cài thêm gì)
  --backbone mambavision_t             Encoder MambaVision-T pretrained (cần Linux + GPU NVIDIA + mamba-ssm)
  --backbone mambavision_mock          Bản giả lập thuần PyTorch để thử luồng trên CPU
  --no-pretrained                      Khởi tạo MambaVision từ đầu thay vì nạp pretrained
  --freeze-backbone-epochs N           Đóng băng backbone N epoch đầu rồi mở khóa (-1 = đóng băng suốt)
  --backbone-lr-mult 0.1               lr của backbone = lr * 0.1 (chỉ khi dùng pretrained; scratch dùng 1.0)
  Mỗi checkpoint ghi nhớ backbone/features; resume sai cấu hình sẽ báo lỗi rõ ràng. Nên đặt --save-dir riêng
  cho mỗi cấu hình (vd. checkpoints/baseline và checkpoints/mambavision_t) để so sánh công bằng.
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
from trainer import Trainer, _build_criterion, load_checkpoint_file, seed_everything

DEFAULT_BATCH_SIZE = getattr(cfg, "BATCH_SIZE", 8)
DEFAULT_EPOCHS = getattr(cfg, "EPOCHS", 50)
DEFAULT_LR = getattr(cfg, "LR", getattr(cfg, "LEARNING_RATE", 1e-4))
DEFAULT_SEED = getattr(cfg, "SEED", 42)
DEFAULT_IMAGE_SIZE = getattr(cfg, "IMAGE_SIZE", 512)
ETA_MIN = 1e-6


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
    shuffle_seed=None,
):
    """
    Dựng train/val loader. Nếu không có mẫu val thì trả val_loader=None (Trainer tự xử lý).
    `seed` quyết định tập con khi dùng --limit-* (phải giữ nguyên khi resume);
    `shuffle_seed` quyết định thứ tự xáo trộn (đổi khi resume để không lặp lại đúng thứ tự epoch 1).
    """
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
    gen = torch.Generator().manual_seed(seed if shuffle_seed is None else shuffle_seed)
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
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=ETA_MIN)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    return None


def load_init_weights(model, path):
    """Nạp trọng số model (không nạp optimizer/epoch) để fine-tune từ checkpoint cũ."""
    ckpt = load_checkpoint_file(path, "cpu")
    meta = (ckpt.get("meta") or {}) if isinstance(ckpt, dict) else {}
    expect = {"features": model.features, "backbone": model.backbone_name}
    for k, v in expect.items():
        if k in meta and meta[k] != v:
            raise ValueError(
                f"Checkpoint {path} được train với {k}={meta[k]} nhưng lệnh hiện tại dùng {k}={v}. "
                f"Hãy đặt lại --{k} {meta[k]} (hoặc bỏ --init-weights)."
            )
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise ValueError(
            f"Không nạp được trọng số từ {path}: kiến trúc khác (thường do sai --features hoặc --backbone). "
            f"{str(e).splitlines()[0]}"
        ) from e
    print(f"📥 Đã nạp trọng số khởi tạo từ: {path}")


# ======================================================================
# RESUME
# ======================================================================
def resolve_resume_path(resume, save_dir):
    """'auto' -> <save_dir>/last_model.pth nếu có (không có thì train mới); ngược lại là đường dẫn cụ thể."""
    if not resume:
        return None
    if str(resume).lower() == "auto":
        cand = Path(save_dir) / "last_model.pth"
        if cand.exists():
            return cand
        print(f"ℹ️  --resume auto: chưa có {cand} -> bắt đầu train mới.")
        return None
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint để resume: {path.resolve()}")
    return path


def restore_scheduler(trainer, ckpt, a):
    """
    Quyết định cách nối tiếp learning rate sau khi trainer.load_checkpoint đã nạp optimizer:
      - cùng loại scheduler, cosine, T_max khớp --epochs -> nạp lại trạng thái (tiếp tục đúng đường cũ);
      - cùng loại, plateau                                -> nạp lại trạng thái;
      - cosine nhưng --epochs đổi / khác loại scheduler   -> chu kỳ mới: lr đặt lại = --lr cho phần epoch còn lại.
    """
    cur = trainer.scheduler
    if cur is None:
        return
    done_epoch = int(ckpt["epoch"])
    saved_type = ckpt.get("scheduler_type")
    saved_state = ckpt.get("scheduler_state_dict")
    is_cosine = isinstance(cur, torch.optim.lr_scheduler.CosineAnnealingLR)

    if saved_type == type(cur).__name__ and saved_state is not None:
        if is_cosine and saved_state.get("T_max") != a.epochs:
            reason = f"--epochs đổi ({saved_state.get('T_max')} -> {a.epochs})"
        else:
            cur.load_state_dict(saved_state)
            print(f"📈 Tiếp tục scheduler {saved_type} đúng đường cũ | lr hiện tại = {trainer.optimizer.param_groups[0]['lr']:.2e}")
            return
    else:
        reason = f"scheduler đổi ({saved_type} -> {type(cur).__name__})"

    for g in trainer.optimizer.param_groups:
        base = a.lr * g.get("lr_mult", 1.0)
        g["lr"] = base
        g["initial_lr"] = base
    if is_cosine:
        remaining = max(1, a.epochs - done_epoch)
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=remaining, eta_min=ETA_MIN
        )
        print(f"🔄 {reason}: khởi động chu kỳ cosine mới cho {remaining} epoch còn lại, lr = {a.lr:.2e}")
    else:
        print(f"🔄 {reason}: dùng scheduler mới, lr đặt lại = {a.lr:.2e}")


# ======================================================================
# LỆNH TRAIN
# ======================================================================
def run_train(a):
    seed_everything(a.seed)
    if a.resume and a.init_weights:
        raise ValueError("Chỉ dùng MỘT trong hai: --resume (tiếp tục đầy đủ) hoặc --init-weights (chỉ nạp trọng số).")
    resume_path = resolve_resume_path(a.resume, a.save_dir)

    csv_path = ensure_index(a.cache_dir, rebuild=a.rebuild_index)
    summarize_index(csv_path)

    model = DuneUNet(
        in_channels=3, classes=1, features=a.features,
        backbone=a.backbone, pretrained=not a.no_pretrained,
    )
    if a.init_weights:
        load_init_weights(model, a.init_weights)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    use_backbone = model.has_backbone
    bb_mult = a.backbone_lr_mult if (use_backbone and not a.no_pretrained) else 1.0
    if use_backbone:
        n_bb = sum(p.numel() for p in model.encoder.backbone.parameters())
        print(f"🧬 Backbone {model.backbone_name}: {n_bb:,} tham số "
              f"({'pretrained' if not a.no_pretrained else 'khởi tạo từ đầu'}) | "
              f"lr backbone = lr x {bb_mult} | đóng băng: {a.freeze_backbone_epochs} epoch")

    criterion = _build_criterion(a.loss)
    optimizer = torch.optim.AdamW(model.param_groups(a.lr, bb_mult), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = make_scheduler(a.scheduler, optimizer, a.epochs)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=not a.no_amp,
        save_dir=a.save_dir,
        grad_clip=a.grad_clip,
    )
    trainer.extra_meta = {"features": int(a.features), "in_channels": 3, "loss": a.loss,
                          "backbone": model.backbone_name}
    trainer.freeze_backbone_epochs = int(a.freeze_backbone_epochs)

    if resume_path is not None:
        print(f"♻️  Resume từ: {resume_path}")
        ckpt = trainer.load_checkpoint(resume_path, load_scheduler=False, check_meta={"features": int(a.features), "backbone": model.backbone_name})
        restore_scheduler(trainer, ckpt, a)
        wd = trainer.optimizer.param_groups[0].get("weight_decay")
        if wd is not None and abs(wd - a.weight_decay) > 1e-12:
            print(f"ℹ️  weight_decay lấy từ checkpoint ({wd}), bỏ qua --weight-decay {a.weight_decay}.")
        print(f"   Đã xong {trainer.start_epoch - 1} epoch | Best Val Dice = {trainer.best_val_dice:.4f}")

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
        shuffle_seed=a.seed + trainer.start_epoch - 1,
    )
    n_tr = len(train_loader.dataset)
    n_va = len(val_loader.dataset) if val_loader is not None else 0
    print(f"📦 Số lát dùng thực tế: train={n_tr} | val={n_va}" + (" (đã giới hạn bởi --limit-*)" if (a.limit_train or a.limit_val) else ""))

    run_info = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "fn"},
        "device": str(trainer.device),
        "amp": trainer.use_amp,
        "num_params": n_params,
        "train_samples": n_tr,
        "val_samples": n_va,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "start_epoch": trainer.start_epoch,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(a.save_dir).mkdir(parents=True, exist_ok=True)
    cfg_name = "config.json" if resume_path is None else f"config_resume_ep{trainer.start_epoch - 1}.json"
    with open(Path(a.save_dir) / cfg_name, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print(
        f"🧠 DuneUNet(features={a.features}) | {n_params:,} tham số | "
        f"loss={a.loss} | lr={a.lr} | scheduler={a.scheduler} | batch={a.batch_size} | backbone={model.backbone_name}"
    )
    history = trainer.fit(
        train_loader, val_loader, epochs=a.epochs, early_stopping_patience=a.patience
    )
    print(f"💾 Checkpoint tốt nhất: {Path(a.save_dir) / 'best_model.pth'}")
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
        cache_dir="", save_dir="", batch_size=4, epochs=6, lr=3e-3, weight_decay=1e-4,
        features=8, workers=0, patience=0, loss="dice_bce", scheduler="cosine",
        grad_clip=1.0, image_size=128, limit_train=0, limit_val=0, init_weights=None,
        resume=None, backbone="none", no_pretrained=False, freeze_backbone_epochs=0, backbone_lr_mult=1.0,
        rebuild_index=False, gpu_window=False, no_lung_mask=False,
        no_augment=True, no_amp=True, seed=DEFAULT_SEED,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _expect_error(exc_type, fragment, fn):
    """Khẳng định fn() ném exc_type và thông báo chứa `fragment`."""
    try:
        fn()
    except exc_type as e:
        assert fragment in str(e), f"Thông báo lỗi thiếu '{fragment}': {e}"
        return
    raise AssertionError(f"Phải ném {exc_type.__name__} chứa '{fragment}'")


def _selftest_basic(tmp, cache):
    save = tmp / "ckpt"
    assert not (cache / "slices.csv").exists()
    a = _default_args(cache_dir=str(cache), save_dir=str(save))
    hist = run_train(a)
    assert (cache / "slices.csv").exists(), "Không tự tạo slices.csv"
    for fn in ("best_model.pth", "last_model.pth", "history.csv", "config.json"):
        assert (save / fn).exists(), f"Thiếu {fn}"
    assert len(hist) == a.epochs
    assert hist[-1]["train_loss"] < hist[0]["train_loss"], "Train loss phải giảm"
    assert all(np.isfinite(r["train_loss"]) and np.isfinite(r["val_loss"]) for r in hist), "Loss có NaN/Inf"
    with open(save / "config.json", encoding="utf-8") as f:
        info = json.load(f)
    assert info["train_samples"] > 0 and info["val_samples"] > 0 and info["num_params"] > 0

    a2 = _default_args(
        cache_dir=str(cache), save_dir=str(tmp / "ckpt2"), epochs=2, scheduler="plateau",
        loss="tversky", init_weights=str(save / "best_model.pth"), limit_train=6,
    )
    hist2 = run_train(a2)
    assert len(hist2) == 2 and all(np.isfinite(r["train_loss"]) for r in hist2)

    a3 = _default_args(
        cache_dir=str(cache), save_dir=str(tmp / "ckpt3"), epochs=3, scheduler="none", patience=1,
    )
    hist3 = run_train(a3)
    assert 1 <= len(hist3) <= 3
    return hist


def _selftest_state_roundtrip(tmp, ckpt_path):
    """Nạp checkpoint vào Trainer MỚI (cố tình đặt lr=1.0) rồi so khớp từng thành phần với nội dung file."""
    ck = load_checkpoint_file(ckpt_path)
    model = DuneUNet(in_channels=3, classes=1, features=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    tr = Trainer(model, _build_criterion("dice_bce"), opt, make_scheduler("cosine", opt, 6),
                 use_amp=False, save_dir=tmp / "roundtrip")
    tr.load_checkpoint(ckpt_path, load_scheduler=True, check_meta={"features": 8})

    cur_w = tr.model.state_dict()
    for k, v in ck["model_state_dict"].items():
        assert torch.equal(cur_w[k].cpu(), v.cpu()), f"Trọng số {k} không khớp"
    saved_st, cur_st = ck["optimizer_state_dict"]["state"], tr.optimizer.state_dict()["state"]
    assert len(saved_st) > 0 and saved_st.keys() == cur_st.keys(), "Thiếu moment của optimizer"
    for i in saved_st:
        assert torch.equal(saved_st[i]["exp_avg"], cur_st[i]["exp_avg"]), "exp_avg không khớp"
        assert torch.equal(saved_st[i]["exp_avg_sq"], cur_st[i]["exp_avg_sq"]), "exp_avg_sq không khớp"
        assert float(saved_st[i]["step"]) == float(cur_st[i]["step"]), "step không khớp"
    assert tr.optimizer.param_groups[0]["lr"] == ck["optimizer_state_dict"]["param_groups"][0]["lr"] != 1.0
    assert tr.scheduler.last_epoch == ck["scheduler_state_dict"]["last_epoch"] == ck["epoch"]
    assert tr.best_val_dice == ck["best_val_dice"] and tr.no_improve == ck["no_improve"]
    assert tr.start_epoch == ck["epoch"] + 1 and len(tr.history) == ck["epoch"]


def _selftest_resume(tmp, cache, reference_hist):
    """Kiểm thử resume: ngắt giữa chừng thật (KeyboardInterrupt) -> tiếp tục -> kéo dài -> các tình huống lỗi."""
    print("\n" + "#" * 65 + "\n# KIỂM THỬ RESUME\n" + "#" * 65)
    ref_lr = [r["lr"] for r in reference_hist]
    save = tmp / "resume_run"
    common = dict(cache_dir=str(cache), epochs=6)

    # --- 1) Ngắt ở epoch 4 (giả lập Ctrl+C), kiểm tra checkpoint dừng đúng ở epoch 3 ---
    orig = Trainer.run_epoch
    calls = {"n": 0}

    def flaky(self, loader, is_train=True):
        if is_train:
            calls["n"] += 1
            if calls["n"] == 4:
                raise KeyboardInterrupt
        return orig(self, loader, is_train)

    Trainer.run_epoch = flaky
    try:
        h_int = run_train(_default_args(save_dir=str(save), **common))
    finally:
        Trainer.run_epoch = orig
    assert [r["epoch"] for r in h_int] == [1, 2, 3], "Ngắt ở epoch 4 phải còn đúng 3 epoch"
    ck3 = load_checkpoint_file(save / "last_model.pth")
    assert ck3["epoch"] == 3 and ck3["scheduler_state_dict"] is not None and ck3["meta"]["features"] == 8
    assert not list(save.glob("*.tmp")), "Không được để lại file tạm"
    shutil.copy(save / "last_model.pth", save / "ep3.pth")
    loss_ep3 = h_int[-1]["train_loss"]

    # --- 2) Resume với đúng --epochs: phải chạy epoch 4..6, lr khớp lượt chạy liền mạch ---
    h_res = run_train(_default_args(save_dir=str(save), resume=str(save / "last_model.pth"), **common))
    assert [r["epoch"] for r in h_res] == [1, 2, 3, 4, 5, 6], f"History phải liền mạch: {[r['epoch'] for r in h_res]}"
    assert np.allclose([r["lr"] for r in h_res], ref_lr, rtol=1e-6, atol=1e-12), \
        f"Đường lr sau resume phải trùng lượt chạy liền mạch:\n{[r['lr'] for r in h_res]}\n{ref_lr}"
    assert h_res[3]["train_loss"] < h_res[0]["train_loss"], "Epoch 4 sau resume không được quay lại mức khởi tạo"
    assert abs(h_res[3]["train_loss"] - loss_ep3) < 0.15, "Loss sau resume phải liên tục với trước khi ngắt"
    csv_hist = pd.read_csv(save / "history.csv")
    assert list(csv_hist["epoch"]) == [1, 2, 3, 4, 5, 6]
    ck6 = load_checkpoint_file(save / "last_model.pth")
    assert ck6["epoch"] == 6
    assert abs(ck6["best_val_dice"] - csv_hist["val_dice"].max()) < 1e-4, "best_val_dice phải nhất quán với history"
    assert (save / "config_resume_ep3.json").exists() and (save / "config.json").exists()
    print("✔ ngắt giữa chừng + resume đúng đường lr: OK")

    _selftest_state_roundtrip(tmp, save / "ep3.pth")
    print("✔ khôi phục đủ trạng thái (trọng số, moment optimizer, scheduler, best, bộ đếm): OK")

    # --- 3) Đã đủ epoch -> không train gì, không ghi đè ---
    mtime = (save / "last_model.pth").stat().st_mtime_ns
    h_done = run_train(_default_args(save_dir=str(save), resume=str(save / "last_model.pth"), **common))
    assert len(h_done) == 6 and (save / "last_model.pth").stat().st_mtime_ns == mtime, "Không được ghi đè khi đã đủ epoch"
    print("✔ resume khi đã đủ epoch: OK")

    # --- 4) Kéo dài 6 -> 9 epoch sang thư mục MỚI: chu kỳ cosine mới, history được mang theo ---
    save_ext = tmp / "resume_ext"
    h_ext = run_train(_default_args(save_dir=str(save_ext), resume=str(save / "last_model.pth"),
                                    cache_dir=str(cache), epochs=9))
    assert [r["epoch"] for r in h_ext] == list(range(1, 10))
    assert h_ext[6]["lr"] > 1e-3, f"Kéo dài phải khởi động lại lr, nhận được {h_ext[6]['lr']:.2e}"
    assert h_ext[-1]["lr"] < h_ext[6]["lr"], "lr phải giảm dần trong chu kỳ mới"
    assert [r["train_loss"] for r in h_ext[:6]] == [r["train_loss"] for r in h_res], "6 epoch đầu phải được mang sang nguyên vẹn"
    assert (save / "best_model.pth").stat().st_mtime_ns < (save_ext / "last_model.pth").stat().st_mtime_ns
    print("✔ kéo dài epoch (6 -> 9) sang thư mục mới: OK")

    # --- 5) Resume từ checkpoint CŨ trong cùng thư mục: history bị cắt, bản gốc được sao lưu ---
    h_old = run_train(_default_args(save_dir=str(save), resume=str(save / "ep3.pth"), **common))
    assert [r["epoch"] for r in h_old] == [1, 2, 3, 4, 5, 6]
    backup = pd.read_csv(save / "history_before_resume.csv")
    assert len(backup) == 6, "Phải sao lưu history trước khi cắt"
    print("✔ resume từ checkpoint cũ (cắt history + sao lưu): OK")

    # --- 6) Đổi loại scheduler khi resume -> vẫn chạy, lr đặt lại ---
    h_sw = run_train(_default_args(save_dir=str(tmp / "resume_sw"), resume=str(save / "ep3.pth"),
                                   scheduler="plateau", **common))
    assert [r["epoch"] for r in h_sw] == [1, 2, 3, 4, 5, 6]
    print("✔ đổi scheduler khi resume: OK")

    # --- 7) --resume auto ---
    auto_dir = tmp / "resume_auto"
    h_a1 = run_train(_default_args(save_dir=str(auto_dir), resume="auto", cache_dir=str(cache), epochs=2))
    assert [r["epoch"] for r in h_a1] == [1, 2], "auto khi chưa có checkpoint phải train mới"
    h_a2 = run_train(_default_args(save_dir=str(auto_dir), resume="auto", cache_dir=str(cache), epochs=3))
    assert [r["epoch"] for r in h_a2] == [1, 2, 3], "auto khi đã có checkpoint phải nối tiếp"
    print("✔ --resume auto: OK")

    # --- 8) Các tình huống lỗi phải có thông báo rõ ràng ---
    _expect_error(ValueError, "features", lambda: run_train(
        _default_args(save_dir=str(tmp / "err1"), resume=str(save / "last_model.pth"), features=16, **common)))
    _expect_error(ValueError, "MỘT trong hai", lambda: run_train(
        _default_args(save_dir=str(tmp / "err2"), resume=str(save / "last_model.pth"),
                      init_weights=str(save / "best_model.pth"), **common)))
    _expect_error(FileNotFoundError, "resume", lambda: run_train(
        _default_args(save_dir=str(tmp / "err3"), resume=str(tmp / "khong_co.pth"), **common)))
    assert not (tmp / "err1").exists() or not list((tmp / "err1").glob("*.pth")), "Lỗi tương thích không được để lại checkpoint"
    print("✔ các tình huống lỗi: OK")


def _bb_hash(bb):
    """Chữ ký của toàn bộ trọng số + thống kê BatchNorm của backbone (đổi khi có bất kỳ cập nhật nào)."""
    return sum(v.double().abs().sum().item() for v in bb.state_dict().values() if v.is_floating_point())


def _run_recorded(args, interrupt_at=None):
    """Chạy run_train, ghi lại trạng thái backbone ở từng epoch train; có thể giả lập Ctrl+C ở epoch thứ N."""
    rec, calls, orig = [], {"n": 0}, Trainer.run_epoch

    def wrapped(self, loader, is_train=True):
        if not is_train:
            return orig(self, loader, is_train)
        calls["n"] += 1
        if interrupt_at and calls["n"] == interrupt_at:
            raise KeyboardInterrupt
        core = self._unwrap_model()
        bb = core.encoder.backbone if core.has_backbone else None
        h0 = _bb_hash(bb) if bb is not None else None
        out = orig(self, loader, is_train)
        if bb is not None:
            rec.append(dict(frozen=core.encoder_frozen, bb_train=bb.training,
                            rg=any(q.requires_grad for q in bb.parameters()), changed=_bb_hash(bb) != h0))
        return out

    Trainer.run_epoch = wrapped
    try:
        hist = run_train(args)
    finally:
        Trainer.run_epoch = orig
    return hist, rec


def _selftest_backbone(tmp, cache):
    """Chuyển đổi có/không backbone: lịch đóng băng, lr theo nhóm, resume qua mốc mở khóa, tương thích checkpoint."""
    print("\n" + "#" * 65 + "\n# KIỂM THỬ BACKBONE (mambavision_mock)\n" + "#" * 65)
    col = lambda rec, k: [r[k] for r in rec]
    common = dict(cache_dir=str(cache), backbone="mambavision_mock", features=8, backbone_lr_mult=0.1,
                  freeze_backbone_epochs=2, epochs=4)

    # --- 1) Lịch đóng băng 2 epoch đầu rồi mở: trạng thái, chế độ BN và cập nhật trọng số đúng từng epoch ---
    save = tmp / "bb_run"
    hist, rec = _run_recorded(_default_args(save_dir=str(save), **common))
    assert col(rec, "frozen") == [True, True, False, False], col(rec, "frozen")
    assert col(rec, "rg") == [False, False, True, True], "requires_grad của backbone sai lịch"
    assert col(rec, "bb_train") == [False, False, True, True], "BN của backbone phải ở eval khi đóng băng"
    assert col(rec, "changed") == [False, False, True, True], "Backbone đóng băng không được thay đổi dù chỉ một giá trị"
    assert all(np.isfinite(r["train_loss"]) for r in hist) and hist[-1]["train_loss"] < hist[0]["train_loss"]
    ck = load_checkpoint_file(save / "last_model.pth")
    assert ck["meta"]["backbone"] == "mambavision_mock"
    g = ck["optimizer_state_dict"]["param_groups"]
    # (cuối chu kỳ cosine cả hai nhóm đều về eta_min tuyệt đối nên so tỉ lệ ở lr gốc `initial_lr`)
    assert len(g) == 2 and g[0]["lr_mult"] == 1.0 and g[1]["lr_mult"] == 0.1
    assert abs(g[1]["initial_lr"] / g[0]["initial_lr"] - 0.1) < 1e-9, f"lr gốc theo nhóm sai: {[x['initial_lr'] for x in g]}"
    print("✔ lịch đóng băng 2 epoch đầu + lr backbone x0.1: OK")

    # --- 2) Resume ngay SAU mốc mở khóa: epoch 3,4 phải ở trạng thái mở ---
    s2 = tmp / "bb_resume_a"
    _run_recorded(_default_args(save_dir=str(s2), **common), interrupt_at=3)
    h2, rec2 = _run_recorded(_default_args(save_dir=str(s2), resume=str(s2 / "last_model.pth"), **common))
    assert [r["epoch"] for r in h2] == [1, 2, 3, 4]
    assert col(rec2, "frozen") == [False, False] and col(rec2, "changed") == [True, True], "Resume sau mốc: backbone phải mở"
    # --- 3) Resume TRƯỚC mốc (giữa giai đoạn đóng băng): epoch 2 phải đóng băng lại ---
    s3 = tmp / "bb_resume_b"
    _run_recorded(_default_args(save_dir=str(s3), **common), interrupt_at=2)
    h3, rec3 = _run_recorded(_default_args(save_dir=str(s3), resume=str(s3 / "last_model.pth"), **common))
    assert [r["epoch"] for r in h3] == [1, 2, 3, 4]
    assert col(rec3, "frozen") == [True, False, False] and col(rec3, "changed") == [False, True, True], \
        f"Resume giữa giai đoạn đóng băng sai: {rec3}"
    print("✔ resume trước/sau mốc mở khóa (trạng thái suy ra từ số epoch): OK")

    # --- 4) Kéo dài epoch khi có 2 nhóm lr: tỉ lệ lr giữa các nhóm được giữ ---
    s4 = tmp / "bb_extend"
    h4 = run_train(_default_args(save_dir=str(s4), resume=str(save / "last_model.pth"), **{**common, "epochs": 6}))
    g4 = load_checkpoint_file(s4 / "last_model.pth")["optimizer_state_dict"]["param_groups"]
    assert g4[0]["initial_lr"] == 3e-3 and abs(g4[1]["initial_lr"] / g4[0]["initial_lr"] - 0.1) < 1e-9, \
        [x["initial_lr"] for x in g4]
    assert h4[4]["lr"] > 1e-3, f"Chu kỳ mới phải khởi động lại lr, nhận {h4[4]['lr']:.2e}"
    print("✔ kéo dài epoch giữ tỉ lệ lr giữa decoder/backbone: OK")

    # --- 5) Đóng băng suốt quá trình (-1): backbone bất động, decoder vẫn học ---
    h5, rec5 = _run_recorded(_default_args(save_dir=str(tmp / "bb_frozen"), **{**common, "freeze_backbone_epochs": -1}))
    assert all(col(rec5, "frozen")) and not any(col(rec5, "changed")) and h5[-1]["train_loss"] < h5[0]["train_loss"]
    print("✔ đóng băng suốt quá trình (-1): OK")

    # --- 6) Cờ đóng băng không làm hỏng baseline ---
    h6, rec6 = _run_recorded(_default_args(save_dir=str(tmp / "base_freezeflag"), cache_dir=str(cache),
                                           freeze_backbone_epochs=2, epochs=2))
    assert len(h6) == 2 and rec6 == []
    print("✔ baseline bỏ qua cờ đóng băng: OK")

    # --- 7) Chuyển cấu hình sai phải báo lỗi rõ, không âm thầm nạp bậy ---
    base_ck = tmp / "ckpt" / "last_model.pth"       # checkpoint baseline từ phần kiểm thử cơ bản
    _expect_error(ValueError, "backbone", lambda: run_train(_default_args(
        save_dir=str(tmp / "e1"), cache_dir=str(cache), resume=str(save / "last_model.pth"), epochs=8)))   # mock -> baseline
    _expect_error(ValueError, "backbone", lambda: run_train(_default_args(
        save_dir=str(tmp / "e2"), cache_dir=str(cache), resume=str(base_ck), backbone="mambavision_mock", epochs=8)))  # baseline -> mock
    _expect_error(ValueError, "backbone", lambda: run_train(_default_args(
        save_dir=str(tmp / "e3"), cache_dir=str(cache), init_weights=str(base_ck), backbone="mambavision_mock")))
    # checkpoint định dạng cũ (meta chưa có 'backbone') vẫn resume được ở chế độ baseline
    old = load_checkpoint_file(base_ck); old["meta"].pop("backbone", None)
    old_dir = tmp / "old_fmt"; old_dir.mkdir(); torch.save(old, old_dir / "last_model.pth")
    h7 = run_train(_default_args(save_dir=str(old_dir), cache_dir=str(cache), resume=str(old_dir / "last_model.pth"), epochs=8))
    assert h7[-1]["epoch"] == 8
    print("✔ chuyển cấu hình sai báo lỗi rõ + checkpoint định dạng cũ vẫn resume: OK")


def run_selftest(_):
    print("Đang chạy kiểm thử toàn trình train_main.py (dữ liệu giả lập)...")
    tmp = Path(tempfile.mkdtemp())
    try:
        cache = tmp / "cache"
        _make_synthetic_cache(cache)
        hist = _selftest_basic(tmp, cache)
        _selftest_resume(tmp, cache, hist)
        _selftest_backbone(tmp, cache)
        print(
            f"\n✅ selftest train_main.py: OK! "
            f"(run 1: train_loss {hist[0]['train_loss']:.4f} -> {hist[-1]['train_loss']:.4f}, "
            f"val_dice cuối = {hist[-1]['val_dice']:.4f})"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ======================================================================
# CLI
# ======================================================================
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="Kiểm thử toàn trình bằng dữ liệu giả lập (vài chục giây trên CPU)")
    s.set_defaults(fn=run_selftest)

    t = sub.add_parser("train", help="Huấn luyện chính thức trên cache_1mm")
    t.add_argument("--cache-dir", default="data/cache_1mm")
    t.add_argument("--save-dir", default="checkpoints/dune_unet_25d")
    t.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    t.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="TỔNG số epoch đích (kể cả khi resume)")
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
    t.add_argument("--backbone", default="none",
                   help="none | mambavision_t/t2/s/b/l/l2 | mambavision_mock (xem phần CHỌN BACKBONE ở đầu file)")
    t.add_argument("--no-pretrained", action="store_true", help="MambaVision: khởi tạo từ đầu, không nạp pretrained")
    t.add_argument("--freeze-backbone-epochs", type=int, default=0,
                   help="Đóng băng backbone N epoch đầu rồi mở khóa (-1 = đóng băng suốt, 0 = không đóng băng)")
    t.add_argument("--backbone-lr-mult", type=float, default=0.1,
                   help="Hệ số lr cho backbone pretrained so với phần còn lại")
    t.add_argument("--resume", default=None,
                   help="Đường dẫn checkpoint (.pth) để tiếp tục đầy đủ, hoặc 'auto' = last_model.pth trong --save-dir")
    t.add_argument("--init-weights", default=None, help="Chỉ nạp trọng số model từ checkpoint (không dùng chung với --resume)")
    t.add_argument("--limit-train", type=int, default=0, help="Chỉ dùng N lát train ngẫu nhiên (0 = tất cả)")
    t.add_argument("--limit-val", type=int, default=0, help="Chỉ dùng N lát val ngẫu nhiên (0 = tất cả)")
    t.add_argument("--rebuild-index", action="store_true", help="Dựng lại slices.csv trước khi train")
    t.add_argument("--gpu-window", action="store_true", help="Chuyển int16 lên GPU rồi gọi window_hu_torch")
    t.add_argument("--no-lung-mask", action="store_true")
    t.add_argument("--no-augment", action="store_true")
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--seed", type=int, default=DEFAULT_SEED)
    t.set_defaults(fn=run_train)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)