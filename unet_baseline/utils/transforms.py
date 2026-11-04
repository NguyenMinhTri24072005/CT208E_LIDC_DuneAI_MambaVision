"""
transforms.py
-------------
Albumentations pipeline cho ảnh + mask (cùng phép biến đổi hình học cho cả hai).

Thay đổi so với bản trước:
- `resize=False` cho train VÀ val/test: huấn luyện/đánh giá ở độ phân giải gốc, không ép về 256x256.
- A.Affine được khởi tạo theo chữ ký thật của phiên bản cài đặt (2.x dùng `border_mode`/`fill`,
  1.x dùng `mode`/`cval`) -> hết cảnh báo "Argument(s) 'mode' are not valid".
- Mask luôn dùng nội suy NEAREST.
"""

import inspect
from typing import Any, Dict, List, Optional

import albumentations as A
import cv2


def _make_affine(p: float) -> A.BasicTransform:
    params = set(inspect.signature(A.Affine).parameters)
    kw: Dict[str, Any] = dict(
        scale=(0.9, 1.1),
        translate_percent=(-0.0625, 0.0625),
        rotate=(-15, 15),
        interpolation=cv2.INTER_LINEAR,
        p=p,
    )
    if "mask_interpolation" in params:
        kw["mask_interpolation"] = cv2.INTER_NEAREST
    if "border_mode" in params:
        kw["border_mode"] = cv2.BORDER_CONSTANT
    elif "mode" in params:
        kw["mode"] = cv2.BORDER_CONSTANT
    if "fill" in params:
        kw["fill"] = 0
    elif "cval" in params:
        kw["cval"] = 0
    if "fill_mask" in params:
        kw["fill_mask"] = 0
    elif "cval_mask" in params:
        kw["cval_mask"] = 0
    return A.Affine(**kw)


def get_train_transforms(
    image_size: int = 256,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    max_pixel_value: float = 255.0,
    augment_cfg: Optional[Dict[str, Any]] = None,
    resize: bool = True,
) -> A.Compose:
    """Pipeline TRAIN. resize=False khi huấn luyện ở độ phân giải gốc (đã crop trong Dataset)."""
    mean = mean or [0.5, 0.5, 0.5]
    std = std or [0.5, 0.5, 0.5]
    aug = augment_cfg or {}

    tl: List[Any] = []
    if resize:
        tl.append(A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR))
    if aug.get("hflip_p", 0.5) > 0:
        tl.append(A.HorizontalFlip(p=aug.get("hflip_p", 0.5)))
    if aug.get("vflip_p", 0.0) > 0:
        tl.append(A.VerticalFlip(p=aug["vflip_p"]))
    if aug.get("rotate90_p", 0.0) > 0:
        tl.append(A.RandomRotate90(p=aug["rotate90_p"]))
    if aug.get("shift_scale_rotate_p", 0.5) > 0:
        tl.append(_make_affine(aug.get("shift_scale_rotate_p", 0.5)))
    if aug.get("brightness_contrast_p", 0.2) > 0:
        tl.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=aug.get("brightness_contrast_p", 0.2)
            )
        )
    tl.append(A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value))
    return A.Compose(tl)


def get_val_transforms(
    image_size: int = 256,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    max_pixel_value: float = 255.0,
    resize: bool = True,
) -> A.Compose:
    """Pipeline VAL/TEST/predict. resize=False -> giữ nguyên độ phân giải gốc (chỉ Normalize)."""
    mean = mean or [0.5, 0.5, 0.5]
    std = std or [0.5, 0.5, 0.5]
    tl: List[Any] = []
    if resize:
        tl.append(A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR))
    tl.append(A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value))
    return A.Compose(tl)
