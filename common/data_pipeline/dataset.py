# -*- coding: utf-8 -*-
"""
dataset.py - PyTorch Dataset & DataLoader 2.5D cho DuneAI.

Đặc điểm:
  - Đọc trực tiếp từ cache_1mm thông qua mmap (chỉ đọc lát cắt cần dùng).
  - Ghép 3 lát cắt liên tiếp (z-1, z, z+1) tạo đầu vào 2.5D (3, 512, 512).
  - Áp mặt nạ phổi an toàn: Train hợp thêm vùng u giãn nở; Val/Test chỉ dùng phổi (không rò rỉ nhãn).
  - Hỗ trợ chuẩn hóa cửa sổ HU ngay trên CPU hoặc trả về int16 để chạy window_hu_torch trên GPU.

Dùng:
  python dataset.py selftest
  python dataset.py inspect --cache-dir data/cache_1mm
"""
import argparse
import json
import random
import shutil
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
ROOT = _ROOT

from configs.config import IMAGE_SIZE
from common.data_pipeline.medical_prep import (
    HU_FLOOR,
    apply_lung_mask,
    clip_and_window_hu,
    load_cached_slices,
    lung_cache_path,
    lung_keep_mask,
    window_hu_torch,
)
from common.data_pipeline.offline_builder import build_dataset_index


class PosAllNegResample(Sampler):
    """
    Mỗi epoch: giữ toàn bộ lát dương tính + lấy mẫu ngẫu nhiên (neg_ratio * n_pos) lát âm tính.
    Giúp mô hình quan sát đa dạng kho lát phổi sạch khác nhau qua từng chu kỳ.
    """

    def __init__(self, has_tumor_array, neg_ratio=0.25, min_neg_per_case=1, seed=42):
        has_tumor = np.asarray(has_tumor_array)
        self.pos_indices = np.where(has_tumor == 1)[0]
        self.neg_indices = np.where(has_tumor == 0)[0]
        target_neg = int(round(float(neg_ratio) * len(self.pos_indices)))
        if len(self.neg_indices) > 0:
            self.k_neg = min(len(self.neg_indices), max(int(min_neg_per_case), target_neg))
        else:
            self.k_neg = 0
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Cập nhật epoch hiện tại ở đầu mỗi chu kỳ huấn luyện."""
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.k_neg > 0 and len(self.neg_indices) > 0:
            sampled_neg = rng.choice(self.neg_indices, size=self.k_neg, replace=False)
        else:
            sampled_neg = np.empty(0, dtype=np.int64)
        epoch_indices = np.concatenate([self.pos_indices, sampled_neg])
        rng.shuffle(epoch_indices)
        return iter(epoch_indices.tolist())

    def __len__(self) -> int:
        return len(self.pos_indices) + self.k_neg


class LIDC25DDataset(Dataset):
    def __init__(
        self,
        cache_dir,
        csv_path=None,
        split="train",
        in_channels=3,
        image_size=IMAGE_SIZE,
        apply_lung=True,
        window_on_cpu=True,
        quantize=True,
        augment=False,
        tumor_union_p=1.0,
        static_neg_only=True,
    ):
        """
        Args:
            cache_dir: Đường dẫn tới thư mục cache_1mm.
            csv_path: Đường dẫn tới slices.csv (mặc định: <cache_dir>/slices.csv).
            split: 'train', 'val', hoặc 'all'.
            in_channels: Số lát cắt ghép kênh 2.5D (mặc định 3: z-1, z, z+1).
            image_size: Kích thước đầu ra sau pad/crop (mặc định 512).
            apply_lung: Có che vùng ngoài phổi về -1000 HU hay không.
            window_on_cpu: True -> trả về float32 [0, 1]; False -> trả về int16 HU để chạy window_hu_torch trên GPU.
            quantize: Bật/tắt lượng tử hóa bước 25 HU.
            augment: Bật lật/xoay ngẫu nhiên (chỉ có tác dụng khi split='train').
        """
        super().__init__()
        assert in_channels in (1, 3, 5), "in_channels phải là số lẻ (1, 3, hoặc 5)"
        self.cache_dir = Path(cache_dir)
        self.csv_path = Path(csv_path) if csv_path is not None else (self.cache_dir / "slices.csv")
        self.split = split
        self.in_channels = in_channels
        self.half_k = in_channels // 2
        self.image_size = image_size
        self.apply_lung = apply_lung
        self.window_on_cpu = window_on_cpu
        self.quantize = quantize
        self.augment = bool(augment and (split == "train"))
        self.tumor_union_p = float(tumor_union_p)
        self.static_neg_only = bool(static_neg_only)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {self.csv_path}. Hãy chạy `python offline_builder.py build` trước!"
            )

        df = pd.read_csv(self.csv_path)
        if split is not None and split != "all":
            df = df[df["split"] == split].reset_index(drop=True)
        if self.static_neg_only and "in_static" in df.columns:
            df = df[df["in_static"] == 1].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"Không có mẫu nào thuộc split='{split}' trong {self.csv_path}")
        self.df = df

    def __len__(self):
        return len(self.df)

    def _neighbor_indices(self, z, num_slices):
        """Lấy chỉ số các lát lân cận, tự động kẹp biên ở lát 0 và lát cuối."""
        return [min(max(0, z + offset), num_slices - 1) for offset in range(-self.half_k, self.half_k + 1)]

    @staticmethod
    def _augment_pair(img_chw, mask_1hw):
        """Biến đổi hình học đồng bộ trên cả khối kênh 2.5D và mask trung tâm."""
        if random.random() < 0.5:
            img_chw = np.flip(img_chw, axis=2)
            mask_1hw = np.flip(mask_1hw, axis=2)
        if random.random() < 0.5:
            img_chw = np.flip(img_chw, axis=1)
            mask_1hw = np.flip(mask_1hw, axis=1)
        k = random.randint(0, 3)
        if k > 0:
            img_chw = np.rot90(img_chw, k=k, axes=(1, 2))
            mask_1hw = np.rot90(mask_1hw, k=k, axes=(1, 2))
        return np.ascontiguousarray(img_chw), np.ascontiguousarray(mask_1hw)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row["patient_id"])
        z = int(row["z"])
        num_slices = int(row["num_slices"])

        p_dir = self.cache_dir / pid
        z_indices = self._neighbor_indices(z, num_slices)

        use_lung = (
            self.apply_lung
            and bool(row.get("has_lung_mask", 0))
            and lung_cache_path(p_dir).exists()
        )
        loaded = load_cached_slices(p_dir, z_indices, size=self.image_size, with_lung=use_lung)

        if use_lung:
            img_stack, msk_stack, lung_stack = loaded
            # Hợp mask u vào mask phổi khi split == 'train' theo xác suất tumor_union_p
            use_union = (self.split == "train") and (
                self.tumor_union_p >= 1.0 or random.random() < self.tumor_union_p
            )
            tumor_for_keep = msk_stack if use_union else None
            keep = lung_keep_mask(lung_stack, tumor_mask=tumor_for_keep)
            img_stack = apply_lung_mask(img_stack, keep, fill=HU_FLOOR)
            if not use_union:
                # Đồng bộ xóa nhãn ngoài vùng phổi giữ lại (Sửa P3)
                msk_stack = np.where(keep > 0, msk_stack, 0).astype(msk_stack.dtype)
        else:
            img_stack, msk_stack = loaded

        # Nhãn mục tiêu chỉ là lát cắt chính giữa (1, H, W)
        center_mask = (msk_stack[self.half_k : self.half_k + 1] > 0).astype(np.float32)

        if self.window_on_cpu:
            img_out = clip_and_window_hu(img_stack, quantize=self.quantize)
        else:
            img_out = img_stack.astype(np.int16)

        if self.augment:
            img_out, center_mask = self._augment_pair(img_out, center_mask)

        return {
            "image": torch.from_numpy(img_out),
            "mask": torch.from_numpy(center_mask),
            "patient_id": pid,
            "z": z,
        }


def create_dataloaders(
    cache_dir,
    batch_size=8,
    num_workers=2,
    in_channels=3,
    apply_lung=True,
    window_on_cpu=True,
    augment_train=True,
):
    """Hàm tiện ích khởi tạo cặp (train_loader, val_loader) sẵn sàng cho vòng lặp huấn luyện."""
    train_ds = LIDC25DDataset(
        cache_dir=cache_dir,
        split="train",
        in_channels=in_channels,
        apply_lung=apply_lung,
        window_on_cpu=window_on_cpu,
        augment=augment_train,
    )
    val_ds = LIDC25DDataset(
        cache_dir=cache_dir,
        split="val",
        in_channels=in_channels,
        apply_lung=apply_lung,
        window_on_cpu=window_on_cpu,
        augment=False,
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    return train_loader, val_loader


def build_dataloaders(
    cache_dir,
    batch_size=8,
    num_workers=2,
    in_channels=3,
    use_gpu_window=False,
    lung_mask=True,
    augment=True,
    limit_train=0,
    limit_val=0,
    rebuild_index=False,
    seed=42,
    dynamic_neg_sampling=False,
    neg_ratio=0.25,
    min_neg_per_case=1,
    min_train_area=20.0,
    keep_all_train_neg=False,
    tumor_union_p=1.0,
    slices_csv=None,
    **kwargs,
):
    cache_path = Path(cache_dir).resolve()
    csv_path = Path(slices_csv).resolve() if slices_csv is not None else (cache_path / "slices.csv")
    if rebuild_index or not csv_path.exists():
        build_dataset_index(
            cache_path,
            out_csv_path=csv_path,
            min_train_area=min_train_area,
            neg_ratio=neg_ratio,
            min_neg_per_case=min_neg_per_case,
            keep_all_train_neg=bool(keep_all_train_neg or dynamic_neg_sampling),
            seed=seed,
        )
    train_ds = LIDC25DDataset(
        cache_dir=cache_path,
        csv_path=csv_path,
        split="train",
        in_channels=in_channels,
        apply_lung=lung_mask,
        window_on_cpu=not use_gpu_window,
        augment=augment,
        tumor_union_p=tumor_union_p,
        static_neg_only=not dynamic_neg_sampling,
    )
    val_ds = LIDC25DDataset(
        cache_dir=cache_path,
        csv_path=csv_path,
        split="val",
        in_channels=in_channels,
        apply_lung=lung_mask,
        window_on_cpu=not use_gpu_window,
        augment=False,
        static_neg_only=True,
    )
    if limit_train and limit_train > 0:
        train_ds.df = train_ds.df.iloc[:limit_train].reset_index(drop=True)
    if limit_val and limit_val > 0:
        val_ds.df = val_ds.df.iloc[:limit_val].reset_index(drop=True)
    pin = torch.cuda.is_available()
    if dynamic_neg_sampling and len(train_ds.df) > 0:
        has_t = train_ds.df["has_tumor"].values
        if "in_static" not in train_ds.df.columns or (train_ds.df["in_static"] == 0).sum() == 0:
            print(
                "⚠️ --dynamic-neg đang bật nhưng slices.csv chưa có kho mẫu âm mở rộng (in_static=0); "
                "nên chạy lại offline_builder build với --keep-all-train-neg để đa dạng hóa mẫu âm mỗi epoch."
            )
        if (has_t == 1).any() and (has_t == 0).any():
            train_sampler = PosAllNegResample(
                has_t,
                neg_ratio=neg_ratio,
                min_neg_per_case=min_neg_per_case,
                seed=seed,
            )
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
            )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
        )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    return train_loader, val_loader


def build_dataloaders_from_args(args, cache_dir=None):
    """Khởi tạo DataLoader trực tiếp từ đối tượng CLI args (giữ mặc định R0)."""
    resolved_cache = Path(cache_dir or args.cache_dir).resolve()
    return build_dataloaders(
        cache_dir=resolved_cache,
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_gpu_window=args.gpu_window,
        lung_mask=not args.no_lung_mask,
        augment=not args.no_augment,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        rebuild_index=args.rebuild_index,
        seed=args.seed,
        dynamic_neg_sampling=bool(getattr(args, "dynamic_neg", False)),
        neg_ratio=float(getattr(args, "neg_ratio", 0.25)),
        min_neg_per_case=int(getattr(args, "min_neg_per_case", 1)),
        min_train_area=float(getattr(args, "min_train_area", 20.0)),
        keep_all_train_neg=bool(getattr(args, "keep_all_train_neg", False)),
        tumor_union_p=float(getattr(args, "tumor_union_p", 1.0)),
        slices_csv=getattr(args, "slices_csv", None),
    )

def _cmd_selftest(_):
    print("Đang chạy kiểm thử tự động dataset.py...")
    tmp = Path(tempfile.mkdtemp())
    try:
        for i in range(1, 5):
            p_dir = tmp / f"LIDC-IDRI-000{i}"
            p_dir.mkdir(parents=True)
            img = np.random.randint(-1000, 300, size=(12, 360, 360), dtype=np.int16)
            msk = np.zeros((12, 360, 360), dtype=np.uint8)
            lung = np.ones((12, 360, 360), dtype=np.uint8)
            msk[5:7, 150:160, 150:160] = 1  # 100 mm²

            np.save(p_dir / "image_1mm.npy", img)
            np.save(p_dir / "mask_1mm.npy", msk)
            np.save(p_dir / "lung_1mm.npy", lung)
            with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump({"orig_spacing_xyz": [0.7, 0.7, 1.5], "lung_ok": True}, f)

        build_dataset_index(tmp)
        train_loader, val_loader = create_dataloaders(tmp, batch_size=2, num_workers=0)
        batch = next(iter(train_loader))

        img_b, msk_b = batch["image"], batch["mask"]
        assert img_b.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE), f"Sai shape ảnh: {img_b.shape}"
        assert msk_b.shape == (2, 1, IMAGE_SIZE, IMAGE_SIZE), f"Sai shape mask: {msk_b.shape}"
        assert img_b.dtype == torch.float32 and 0.0 <= img_b.min() and img_b.max() <= 1.0
        assert set(torch.unique(msk_b).tolist()) <= {0.0, 1.0}

        # Kiểm tra chế độ đẩy HU int16 lên GPU rồi gọi window_hu_torch
        ds_cpu = LIDC25DDataset(tmp, split="val", window_on_cpu=True, augment=False)
        ds_raw = LIDC25DDataset(tmp, split="val", window_on_cpu=False, augment=False)
        assert torch.allclose(ds_cpu[0]["image"], window_hu_torch(ds_raw[0]["image"]), atol=1e-6)

        # Kiểm tra PosAllNegResample đổi mẫu âm theo epoch và giữ nguyên 100% mẫu dương
        r1_loader, _ = build_dataloaders(
            tmp,
            batch_size=2,
            num_workers=0,
            rebuild_index=True,
            dynamic_neg_sampling=True,
            neg_ratio=0.5,
            min_neg_per_case=2,
            min_train_area=5.0,
            keep_all_train_neg=True,
            slices_csv=tmp / "slices_r1.csv",
        )
        assert isinstance(r1_loader.sampler, PosAllNegResample)
        r1_loader.sampler.set_epoch(1)
        ep1 = set(r1_loader.sampler)
        r1_loader.sampler.set_epoch(2)
        ep2 = set(r1_loader.sampler)
        pos_set = set(r1_loader.sampler.pos_indices.tolist())
        assert pos_set.issubset(ep1) and pos_set.issubset(ep2)
        print("✅ selftest dataset.py: OK!")
    finally:
        shutil.rmtree(tmp)


def _cmd_inspect(a):
    train_loader, val_loader = create_dataloaders(
        a.cache_dir, batch_size=a.batch_size, num_workers=0
    )
    batch = next(iter(train_loader))
    img, msk = batch["image"], batch["mask"]
    print("=" * 50)
    print("🔍 KIỂM TRA BATCH THỰC TẾ TỪ DATALOADER")
    print("=" * 50)
    print(f"Số batch Train : {len(train_loader)} | Số batch Val: {len(val_loader)}")
    print(f"Image Tensor   : shape={tuple(img.shape)}, dtype={img.dtype}, range=[{img.min():.4f}, {img.max():.4f}]")
    print(f"Mask Tensor    : shape={tuple(msk.shape)}, dtype={msk.dtype}, unique={torch.unique(msk).tolist()}")
    print(f"Bệnh nhân mẫu  : {batch['patient_id']} | Lát Z: {batch['z'].tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="Kiểm thử tự động với dữ liệu giả lập")
    s.set_defaults(fn=_cmd_selftest)

    i = sub.add_parser("inspect", help="Kiểm tra nhanh một batch từ thư mục cache_1mm thật")
    i.add_argument("--cache-dir", default=str(ROOT / "data" / "cache_1mm"))
    i.add_argument("--batch-size", type=int, default=4)
    i.set_defaults(fn=_cmd_inspect)

    args = ap.parse_args()
    args.fn(args)