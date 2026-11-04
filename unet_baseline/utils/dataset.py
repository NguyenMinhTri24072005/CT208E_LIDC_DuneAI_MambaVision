"""
dataset.py
----------
Dataset phân đoạn ảnh (ghép cặp ảnh/mask theo file stem) + DataLoader factory.

Chế độ độ phân giải gốc (data.native: true)  <-- KHUYÊN DÙNG cho LIDC-IDRI ở đây
  * KHÔNG resize về 256x256. Ảnh trên đĩa có kích thước 375-550 px tuỳ bệnh nhân; ép về 256 làm
    diện tích nốt co còn ~22-47% (nốt 19-21 px còn 6-7 px) và mask NEAREST có thể xoá hẳn nốt rất nhỏ.
  * TRAIN: crop size x size (mặc định 192) quanh một pixel dương ngẫu nhiên (p_pos) hoặc crop ngẫu nhiên
    có thông tin (hard negative).
  * VAL/TEST: ảnh đầy đủ ở độ phân giải gốc; batch được pad (nền đen) lên bội số của 32 bằng PadCollate.
    Giá trị trả về vẫn là (images, masks) nên train.py / evaluate.py / diagnose.py dùng được nguyên vẹn.

Chế độ cũ (data.native: false) giữ nguyên hành vi Resize(image_size).
An toàn: bật crop mà native=false sẽ báo lỗi (tránh train ở thang gốc nhưng đánh giá ở thang 256).
"""

import random
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict, Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import get_train_transforms, get_val_transforms
from .native_ops import sample_crop, round_up

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}


class PadCollate:
    """Gộp batch ảnh có kích thước khác nhau bằng cách pad (phải/dưới) lên bội số `multiple`."""

    def __init__(self, multiple: int = 32, pad_value: Optional[Sequence[float]] = None) -> None:
        self.multiple = multiple
        self.pad_value = list(pad_value) if pad_value is not None else None

    def __call__(self, batch):
        images, masks = zip(*batch)
        c = images[0].shape[0]
        h = round_up(max(i.shape[-2] for i in images), self.multiple)
        w = round_up(max(i.shape[-1] for i in images), self.multiple)
        pv = torch.tensor(self.pad_value if self.pad_value else [0.0] * c, dtype=torch.float32)
        imgs = pv.view(c, 1, 1).expand(len(images), c, h, w).clone()
        msks = torch.zeros(len(images), masks[0].shape[0], h, w, dtype=torch.float32)
        for k, (im, mk) in enumerate(zip(images, masks)):
            ih, iw = im.shape[-2:]
            imgs[k, :, :ih, :iw] = im
            msks[k, :, :ih, :iw] = mk
        return imgs, msks


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        n_channels: int = 3,
        n_classes: int = 1,
        limit_samples: Optional[int] = None,
        crop_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.limit_samples = limit_samples
        self.crop_cfg = crop_cfg

        images_dir = self.root / split / "images"
        masks_dir = self.root / split / "masks"
        if not images_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục mask: {masks_dir}")

        image_files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS)
        mask_map = {f.stem: f for f in masks_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS}

        self.samples: List[Tuple[Path, Path]] = []
        missing_masks: List[str] = []
        for img_path in image_files:
            if img_path.stem in mask_map:
                self.samples.append((img_path, mask_map[img_path.stem]))
            else:
                missing_masks.append(img_path.name)
        if missing_masks:
            raise ValueError(
                f"[{split}] Có {len(missing_masks)} ảnh không tìm thấy mask tương ứng! "
                f"Ví dụ: {missing_masks[:5]}"
            )
        if len(self.samples) == 0:
            import warnings
            warnings.warn(f"[{split}] Cảnh báo: Thư mục {images_dir} không chứa mẫu dữ liệu nào!")

        if self.limit_samples is not None and 0 < self.limit_samples < len(self.samples):
            rng = random.Random(42)
            # sort lại để các lát cùng bệnh nhân (cùng kích thước) nằm gần nhau -> ít pad khi batch
            self.samples = sorted(rng.sample(self.samples, self.limit_samples))
            print(f"  ⚡ [{split.upper()}] Đã giới hạn mẫu thử nghiệm: {len(self.samples)} mẫu.")

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ I/O
    def _load_image(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            arr = np.load(str(path))
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            elif arr.ndim == 3 and arr.shape[0] in [1, 3]:
                arr = np.transpose(arr, (1, 2, 0))
            if self.n_channels == 3 and arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
            return arr

        img = None
        for _ in range(3):
            flag = cv2.IMREAD_GRAYSCALE if self.n_channels == 1 else cv2.IMREAD_COLOR
            img = cv2.imread(str(path), flag)
            if img is not None:
                break
            import time
            time.sleep(0.1)
        if img is None:
            raise IOError(f"Không thể đọc file ảnh sau 3 lần thử: {path}")
        if self.n_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
            img = img[:, :, np.newaxis]
        return img.astype(np.uint8)

    def _load_mask(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            mask = np.load(str(path))
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            elif mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            return mask
        mask = None
        for _ in range(3):
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                break
            import time
            time.sleep(0.1)
        if mask is None:
            raise IOError(f"Không thể đọc file mask sau 3 lần thử: {path}")
        return mask

    # ------------------------------------------------------- numpy pipeline
    def _numpy_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về (image HxWxC float32 đã chuẩn hóa, mask HxW). Không dùng torch -> kiểm thử được."""
        img_path, mask_path = self.samples[idx]
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        if self.n_classes == 1:
            mask = (mask > 127).astype(np.float32) if mask.max() > 1.0 else (mask > 0.5).astype(np.float32)
        else:
            mask = mask.astype(np.int64)

        if self.split == "train" and self.crop_cfg and self.crop_cfg.get("enabled", False):
            image, mask, _ = sample_crop(
                image,
                mask,
                size=int(self.crop_cfg.get("size", 192)),
                p_pos=float(self.crop_cfg.get("p_pos", 0.7)),
            )

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        if self.n_classes == 1:
            mask = (mask > 0.5).astype(np.float32)
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._numpy_sample(idx)

        if image.ndim == 2:
            image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float()
        elif image.ndim == 3:
            image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        else:
            raise ValueError(f"Shape ảnh không hợp lệ: {image.shape}")

        if self.n_classes == 1:
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
        else:
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return image_tensor, mask_tensor


def get_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cv2.setNumThreads(0)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    aug_cfg = cfg.get("augment", {})
    model_cfg = cfg.get("model", {})

    root_dir = data_cfg.get("root_dir", "data")
    image_size = data_cfg.get("image_size", 256)
    native = bool(data_cfg.get("native", False))
    n_channels = model_cfg.get("n_channels", 3)
    n_classes = model_cfg.get("n_classes", 1)
    mean = data_cfg.get("mean", [0.5, 0.5, 0.5])
    std = data_cfg.get("std", [0.5, 0.5, 0.5])

    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 2)
    pin_memory = train_cfg.get("pin_memory", True)

    crop_cfg = data_cfg.get("crop", {}) or {}
    use_crop = bool(crop_cfg.get("enabled", False))
    if use_crop and not native:
        raise ValueError(
            "data.crop.enabled=true yêu cầu data.native=true: nếu không, train ở thang gốc còn "
            "val/test bị Resize về 256 -> lệch thang đo nốt (~0.5x) giữa train và đánh giá."
        )

    train_transform = get_train_transforms(
        image_size=image_size, mean=mean, std=std, max_pixel_value=255.0,
        augment_cfg=aug_cfg, resize=not native,
    )
    val_transform = get_val_transforms(
        image_size=image_size, mean=mean, std=std, max_pixel_value=255.0, resize=not native,
    )

    def make(split: str, transform, limit, crop):
        return SegmentationDataset(
            root=root_dir, split=split, transform=transform, n_channels=n_channels,
            n_classes=n_classes, limit_samples=limit, crop_cfg=crop,
        )

    train_dataset = make("train", train_transform, train_cfg.get("limit_train_samples"), crop_cfg)
    val_dataset = make("val", val_transform, train_cfg.get("limit_val_samples"), None)
    test_dataset = make("test", val_transform, train_cfg.get("limit_test_samples"), None)

    collate_fn = None
    if native and n_classes == 1:
        pad_value = [(0.0 - m) / s for m, s in zip(mean, std)]  # pixel đen sau chuẩn hóa
        collate_fn = PadCollate(multiple=32, pad_value=pad_value)

    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None

    def loader(ds, shuffle, drop_last):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            pin_memory=pin_memory, persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor, drop_last=drop_last, collate_fn=collate_fn,
        )

    return (
        loader(train_dataset, True, len(train_dataset) > batch_size),
        loader(val_dataset, False, False),
        loader(test_dataset, False, False),
    )
