"""
lidc_dataset.py
---------------
Dataset và DataLoader cho dữ liệu LIDC-IDRI đã tiền xử lý.

Cấu trúc dữ liệu đầu vào:
    processed/
    ├── slices.csv                          # Chỉ mục tất cả lát cắt
    └── volumes/
        ├── LIDC-IDRI-0001_img.npy          # Volume 3D (Z, H, W) float32 [0,1]
        ├── LIDC-IDRI-0001_votes.npy        # Vote mask 3D (Z, H, W) uint8 0-4
        └── ...
"""

import os
import random
import warnings
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LIDCDataset(Dataset):
    """
    Dataset cho LIDC-IDRI.

    Mỗi mẫu trả về một tuple (image_tensor, mask_tensor):
      - image_tensor : (3, H, W) float32  — 3 lát cắt liên tiếp (z-1, z, z+1)
      - mask_tensor  : (1, H, W) float32  — nhãn nhị phân từ vote mask

    Args:
        csv_path      : Đường dẫn đến file slices.csv.
        processed_dir : Thư mục gốc chứa volumes/.
        split         : "train" | "val" | "test" — chỉ lấy các hàng khớp split.
        transform     : Callable(image_np, mask_np) -> (image_np, mask_np), nhận/trả numpy.
        config        : Dict hoặc object có thuộc tính để đọc các hyper-param.
                        Khoá được sử dụng:
                          config["vote_threshold"] (int, mặc định 2)
    """

    def __init__(
        self,
        csv_path: str,
        processed_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        config: Optional[dict] = None,
    ):
        super().__init__()

        self.processed_dir = Path(processed_dir)
        self.split = split
        self.transform = transform
        self.config = config or {}

        # Ngưỡng vote để coi là khối u (số annotator tối thiểu đồng ý)
        self.vote_threshold: int = self._get_cfg("vote_threshold", default=2)

        # ------------------------------------------------------------------ #
        # Đọc CSV và lọc theo split
        # ------------------------------------------------------------------ #
        df = pd.read_csv(csv_path)

        # Chuẩn hoá tên cột (phòng trường hợp có khoảng trắng thừa)
        df.columns = df.columns.str.strip()

        # Lọc đúng split
        df = df[df["split"].str.strip() == split].reset_index(drop=True)

        if len(df) == 0:
            warnings.warn(
                f"[LIDCDataset] Không tìm thấy mẫu nào cho split='{split}' "
                f"trong file {csv_path}."
            )

        # Lưu toàn bộ records dưới dạng list of dict để truy cập nhanh
        self.records: List[dict] = df.to_dict(orient="records")

        # ------------------------------------------------------------------ #
        # Giới hạn số mẫu khi test nhanh (đọc từ config)
        # ------------------------------------------------------------------ #
        if split in ("val", "test"):
            limit = self._get_cfg(f"limit_{split}_samples", None)
            if limit and len(self.records) > int(limit):
                self.records = self._balanced_subset(int(limit))

        # ------------------------------------------------------------------ #
        # Balanced sampling (chỉ cần cho split == 'train')
        # ------------------------------------------------------------------ #
        self.positive_indices: List[int] = []
        self.negative_indices: List[int] = []

        if split == "train":
            for i, rec in enumerate(self.records):
                # Hỗ trợ linh hoạt cả 2 tên cột: 'has_nodule' hoặc 'has_tumor'
                raw_val = rec.get("has_nodule", rec.get("has_tumor", 0))
                has_tumor = str(raw_val).lower().strip() in ['true', '1']
                if has_tumor:
                    self.positive_indices.append(i)
                else:
                    self.negative_indices.append(i)

            logger.info(
                "[LIDCDataset] Train split — "
                f"positive={len(self.positive_indices)}, "
                f"negative={len(self.negative_indices)}"
            )

    # ---------------------------------------------------------------------- #
    # Tiện ích đọc config
    # ---------------------------------------------------------------------- #
    def _balanced_subset(self, limit: int):
        is_pos = lambda r: str(r.get("has_nodule", r.get("has_tumor", 0))).lower().strip() in ['true', '1']
        pos = [r for r in self.records if is_pos(r)]
        neg = [r for r in self.records if not is_pos(r)]
        rng = random.Random(0)   # seed cố định: val/test giống nhau giữa các lần chạy
        pos = rng.sample(pos, min(limit // 2, len(pos)))
        neg = rng.sample(neg, min(limit - len(pos), len(neg)))
        return pos + neg
    
    
    
    
    def _get_cfg(self, key: str, default):
        """Đọc giá trị từ config (dict hoặc object có attribute)."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    # ---------------------------------------------------------------------- #
    # Dataset API
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        patient_id: str = rec["patient_id"]
        slice_idx: int = int(rec.get("z", rec.get("slice_idx", 0)))

        # ------------------------------------------------------------------ #
        # Đường dẫn file .npy
        # ------------------------------------------------------------------ #
        vol_path   = self.processed_dir / "volumes" / f"{patient_id}_img.npy"
        votes_path = self.processed_dir / "volumes" / f"{patient_id}_votes.npy"

        # Kiểm tra sự tồn tại của file — nếu thiếu thì skip (trả tensor 0)
        if not vol_path.exists() or not votes_path.exists():
            raise FileNotFoundError(f"Thiếu file cho {patient_id}: {vol_path} / {votes_path}")

        # ------------------------------------------------------------------ #
        # Load dữ liệu (mmap để tiết kiệm RAM)
        # ------------------------------------------------------------------ #
        volume: np.ndarray = np.load(str(vol_path),   mmap_mode="r")  # (Z, H, W)
        votes:  np.ndarray = np.load(str(votes_path), mmap_mode="r")  # (Z, H, W)

        Z = volume.shape[0]

        # ------------------------------------------------------------------ #
        # Clamp biên: xử lý volume có ít hơn 3 lát (hoặc biên đầu/cuối)
        # ------------------------------------------------------------------ #
        z     = max(0, min(slice_idx, Z - 1))   # Đảm bảo z nằm trong [0, Z-1]
        z_m1  = max(0, z - 1)                   # z-1 không âm
        z_p1  = min(Z - 1, z + 1)               # z+1 không vượt quá Z_max

        # ------------------------------------------------------------------ #
        # Bốc 3 lát → stack thành (3, H, W)
        # ------------------------------------------------------------------ #
        # Sao chép ra mảng thường (thoát khỏi mmap) để tránh rủi ro khi transform
        slice_m1 = np.array(volume[z_m1], dtype=np.float32)  # (H, W)
        slice_z  = np.array(volume[z],    dtype=np.float32)  # (H, W)
        slice_p1 = np.array(volume[z_p1], dtype=np.float32)  # (H, W)

        image = np.stack([slice_m1, slice_z, slice_p1], axis=0)  # (3, H, W)

        # ------------------------------------------------------------------ #
        # Bốc mask lát z: votes[z] >= threshold → binary uint8
        # ------------------------------------------------------------------ #
        vote_slice = np.array(votes[z], dtype=np.uint8)              # (H, W)
        mask = (vote_slice >= self.vote_threshold).astype(np.uint8)  # (H, W) binary
        mask = mask[np.newaxis, ...]                                  # (1, H, W)

        # ------------------------------------------------------------------ #
        # Áp dụng transform (nếu có) — nhận và trả về numpy
        # ------------------------------------------------------------------ #
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # ------------------------------------------------------------------ #
        # Chuyển sang tensor float32
        # ------------------------------------------------------------------ #
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()  # (3, H, W)
        mask_tensor  = torch.from_numpy(np.ascontiguousarray(mask)).float()   # (1, H, W)

        return image_tensor, mask_tensor

    # ---------------------------------------------------------------------- #
    # Tiện ích nội bộ
    # ---------------------------------------------------------------------- #

    def _empty_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trả về tensor rỗng (zeros) khi file không tồn tại.
        Shape mặc định (3,1,1) và (1,1,1) — DataLoader sẽ padding hoặc
        caller cần tự xử lý collate_fn nếu kích thước H, W cố định.
        """
        return (
            torch.zeros(3, 1, 1, dtype=torch.float32),
            torch.zeros(1, 1, 1, dtype=torch.float32),
        )

    def get_sample_weights(self) -> torch.Tensor:
        """
        Trả về tensor trọng số cho từng mẫu, dùng với WeightedRandomSampler.
        Positive và negative được gán trọng số ngược nhau để cân bằng 1:1.
        Chỉ có ý nghĩa với split == 'train'.
        """
        n_pos = len(self.positive_indices)
        n_neg = len(self.negative_indices)

        # Tránh chia cho 0 khi một nhóm không có mẫu nào
        weight_pos = 1.0 / n_pos if n_pos > 0 else 0.0
        weight_neg = 1.0 / n_neg if n_neg > 0 else 0.0

        weights = torch.zeros(len(self.records), dtype=torch.float32)
        for i in self.positive_indices:
            weights[i] = weight_pos
        for i in self.negative_indices:
            weights[i] = weight_neg

        return weights


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    config,
    split: str,
    transform: Optional[Callable] = None,
) -> DataLoader:
    """
    Tạo DataLoader cho LIDC-IDRI.

    Args:
        config  : Dict hoặc object chứa cấu hình. Khoá cần có:
                    - csv_path      : str — đường dẫn đến slices.csv
                    - processed_dir : str — thư mục processed/
                    - batch_size    : int (mặc định 8)
                    - num_workers   : int (mặc định 4)
                    - pin_memory    : bool (mặc định True)
                    - vote_threshold: int (mặc định 2)
        split   : "train" | "val" | "test"
        transform: Callable(image_np, mask_np) -> (image_np, mask_np)

    Returns:
        torch.utils.data.DataLoader
    """

    def _get(key, default):
        """Đọc giá trị từ config (dict hoặc object)."""
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    csv_path      = _get("csv_path",      None)
    processed_dir = _get("processed_dir", None)
    batch_size    = _get("batch_size",    8)
    num_workers   = _get("num_workers",   4)
    pin_memory    = _get("pin_memory",    True)

    if csv_path is None or processed_dir is None:
        raise ValueError(
            "[get_dataloader] Config phải có 'csv_path' và 'processed_dir'."
        )

    # Tạo dataset
    dataset = LIDCDataset(
        csv_path=csv_path,
        processed_dir=processed_dir,
        split=split,
        transform=transform,
        config=config,
    )

    sampler = None
    shuffle = False  # Mặc định không shuffle (val/test cần kết quả ổn định)

    if split == "train":
        # Cân bằng positive/negative (tỉ lệ 1:1)
        sample_weights = dataset.get_sample_weights()
        
        # Nếu muốn train thử nghiệm với ít mẫu, đọc từ config
        num_samples = config.get("limit_train_samples")
        if not num_samples:
            num_samples = len(sample_weights)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,   # Cho phép lấy lại mẫu thiểu số
        )
        shuffle = False  # Không dùng shuffle khi đã có sampler

    # Chỉ drop_last nếu tổng số mẫu lớn hơn batch_size (tránh lỗi 0 batch khi train thử ít mẫu)
    if split == "train":
        total_samples = num_samples if sampler is not None else len(dataset)
        drop_last = total_samples > batch_size
    else:
        drop_last = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    logger.info(
        f"[get_dataloader] split='{split}' | "
        f"n_samples={len(dataset)} | batch_size={batch_size} | "
        f"sampler={'WeightedRandom' if sampler else 'None'}"
    )

    return loader
