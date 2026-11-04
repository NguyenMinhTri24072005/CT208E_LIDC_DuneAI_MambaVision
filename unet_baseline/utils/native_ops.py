"""
native_ops.py
-------------
Hàm thuần numpy (không phụ thuộc torch) cho pipeline huấn luyện ở ĐỘ PHÂN GIẢI GỐC:
- chọn tâm crop từ MỘT pixel dương ngẫu nhiên (đúng cả khi lát cắt có nhiều nốt xa nhau,
  khác với trung bình toạ độ vốn rơi vào "khoảng trống" giữa hai nốt)
- crop cửa sổ cố định, tự pad nếu ảnh nhỏ hơn cửa sổ
- crop ngẫu nhiên "có thông tin" (tránh crop toàn viền đen) làm mẫu âm / hard negative
- tính kích thước canvas pad cho ảnh validation/test
"""

import math
import random
from typing import Optional, Tuple

import cv2
import numpy as np


def pad_to_min_size(
    image: np.ndarray, mask: np.ndarray, size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad (viền đen / nền 0) để cả H và W >= size."""
    h, w = mask.shape[:2]
    ph, pw = max(size - h, 0), max(size - w, 0)
    if ph == 0 and pw == 0:
        return image, mask
    image = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    return image, mask


def pick_positive_center(
    mask: np.ndarray, rng: random.Random = random
) -> Optional[Tuple[int, int]]:
    """Chọn ngẫu nhiên MỘT pixel dương làm tâm. Trả None nếu mask rỗng."""
    ys, xs = np.nonzero(mask > 0)
    if len(ys) == 0:
        return None
    i = rng.randrange(len(ys))
    return int(ys[i]), int(xs[i])


def window_around(
    h: int, w: int, size: int, center: Tuple[int, int], jitter: int, rng: random.Random = random
) -> Tuple[int, int]:
    """Góc trên-trái (y0, x0) của cửa sổ size×size quanh center (+ lệch ngẫu nhiên ±jitter)."""
    cy, cx = center
    if jitter > 0:
        cy += rng.randint(-jitter, jitter)
        cx += rng.randint(-jitter, jitter)
    y0 = min(max(cy - size // 2, 0), max(h - size, 0))
    x0 = min(max(cx - size // 2, 0), max(w - size, 0))
    return y0, x0


def is_informative(img_crop: np.ndarray, min_std: float = 8.0) -> bool:
    """Crop có đủ cấu trúc (không phải toàn nền đen / đồng nhất)."""
    gray = img_crop.mean(axis=-1) if img_crop.ndim == 3 else img_crop
    return float(gray.std()) > min_std


def sample_crop(
    image: np.ndarray,
    mask: np.ndarray,
    size: int = 192,
    p_pos: float = 0.7,
    jitter: Optional[int] = None,
    rng: random.Random = random,
    max_tries: int = 5,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Lấy 1 crop size×size ở độ phân giải gốc.
    - Với xác suất p_pos (và mask không rỗng): crop quanh một pixel dương ngẫu nhiên, lệch ±jitter
      (mặc định size//3, đảm bảo nốt vẫn nằm trong cửa sổ).
    - Ngược lại: crop ngẫu nhiên có thông tin (mẫu âm / hard negative: mạch máu, màng phổi...).
    Trả về (image_crop, mask_crop, kind) với kind in {"pos", "rand"}.
    """
    image, mask = pad_to_min_size(image, mask, size)
    h, w = mask.shape[:2]
    if jitter is None:
        jitter = size // 3

    if rng.random() < p_pos:
        center = pick_positive_center(mask, rng)
        if center is not None:
            y0, x0 = window_around(h, w, size, center, jitter, rng)
            return image[y0:y0 + size, x0:x0 + size], mask[y0:y0 + size, x0:x0 + size], "pos"

    y0 = x0 = 0
    for _ in range(max_tries):
        y0 = rng.randint(0, h - size)
        x0 = rng.randint(0, w - size)
        if is_informative(image[y0:y0 + size, x0:x0 + size]):
            break
    return image[y0:y0 + size, x0:x0 + size], mask[y0:y0 + size, x0:x0 + size], "rand"


def round_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / float(multiple)) * multiple)
