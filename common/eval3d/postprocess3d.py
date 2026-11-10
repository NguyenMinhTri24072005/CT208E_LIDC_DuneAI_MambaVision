# -*- coding: utf-8 -*-
"""
common/eval3d/postprocess3d.py - Khôi phục hình học không gian và hậu xử lý hình thái học 3D.

Tuân thủ chuẩn y tế của tác giả DuneAI (s.primakov) & medical_prep.py:
  1. unpad_or_uncrop_xy: Hàm nghịch đảo chính xác của pad_or_crop_xy từ (..., 512, 512)
     về lại kích thước (..., H_1mm, W_1mm).
  2. resize_from_1mm_3d: Đưa khối 3D từ lưới 1mm về độ phân giải gốc của file CT/NRRD ban đầu.
  3. postprocess_3d_mask:
     - Giới hạn vùng dự đoán trong mặt nạ phổi.
     - Lấp lỗ khuyết 2D/3D (binary_fill_holes).
     - Lọc thành phần liên thông 3D (26-connectivity): Loại bỏ các cụm có diện tích lát cắt
       cực đại < TUMOR_SIZE_LOW_LIM (20 mm²) hoặc thể tích 3D < min_volume_mm3.
"""

import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import TUMOR_SIZE_LOW_LIM
from common.data_pipeline.medical_prep import _axis_window, lung_keep_mask, pad_or_crop_xy


def unpad_or_uncrop_xy(
    x_padded: Union[np.ndarray, torch.Tensor],
    target_hw: Sequence[int],
    pad_value: float = 0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Hàm nghịch đảo chính xác của `pad_or_crop_xy` trong `medical_prep.py`.
    Chuyển mảng/tensor từ kích thước chuẩn mạng (..., size, size)
    về lại kích thước vật lý 1mm của bệnh nhân (..., H_1mm, W_1mm).
    """
    size_h, size_w = x_padded.shape[-2], x_padded.shape[-1]
    assert size_h == size_w, f"Đầu vào phải có mặt phẳng vuông, nhận được: {x_padded.shape}"
    size = size_h
    h, w = int(target_hw[0]), int(target_hw[1])

    sy, dy, ly = _axis_window(h, size)
    sx, dx, lx = _axis_window(w, size)
    shape = (*x_padded.shape[:-2], h, w)

    if isinstance(x_padded, torch.Tensor):
        out = torch.full(shape, pad_value, dtype=x_padded.dtype, device=x_padded.device)
    else:
        out = np.full(shape, pad_value, dtype=x_padded.dtype)

    out[..., sy : sy + ly, sx : sx + lx] = x_padded[..., dy : dy + ly, dx : dx + lx]
    return out


def resize_from_1mm_3d(
    volume_1mm: np.ndarray,
    orig_shape_zyx: Sequence[int],
    is_mask: bool = True,
) -> np.ndarray:
    """
    Đưa khối 3D từ lưới 1mm (Z, H_1mm, W_1mm) về lại độ phân giải gốc (Z, H_orig, W_orig)
    của file NRRD/DICOM ban đầu.
    """
    z_orig, h_orig, w_orig = (int(v) for v in orig_shape_zyx)
    z, h, w = volume_1mm.shape
    assert z == z_orig, f"Số lát cắt Z không khớp: {z} != {z_orig}"

    if h == h_orig and w == w_orig:
        return volume_1mm.copy()

    flag = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    src = (volume_1mm > 0).astype(np.uint8) if is_mask else np.ascontiguousarray(volume_1mm)
    out = np.empty((z_orig, h_orig, w_orig), dtype=src.dtype)
    for i in range(z_orig):
        out[i] = cv2.resize(src[i], (w_orig, h_orig), interpolation=flag)
    return out


def postprocess_3d_mask(
    prob_or_mask_zyx: np.ndarray,
    lung_mask_zyx: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    min_slice_area_mm2: float = TUMOR_SIZE_LOW_LIM,
    min_volume_mm3: float = 0.0,
    spacing_z: float = 1.0,
    fill_holes: bool = True,
    min_slices_span: int = 1,
    top_k_components: int = 0,
) -> np.ndarray:
    """
    Hậu xử lý khối dự đoán 3D trên lưới 1mm (1 pixel trên mặt phẳng XY = 1.0 mm²):
      1. Ngưỡng hóa xác suất -> nhị phân (uint8).
      2. Cắt bỏ mọi dự đoán nằm ngoài mặt nạ phổi (nếu có lung_mask_zyx).
      3. Lấp các lỗ khuyết nhỏ bên trong tâm khối u trên từng lát cắt 2D (binary_fill_holes).
      4. Phân tích thành phần liên thông 3D (26-connectivity):
         - Loại bỏ cụm nhiễu có diện tích mặt cắt ngang lớn nhất < min_slice_area_mm2 (20.0 mm²).
         - Loại bỏ cụm có thể tích vật lý < min_volume_mm3 hoặc chỉ xuất hiện ít hơn min_slices_span lát cắt.
         - Tùy chọn chỉ giữ lại top_k_components khối lớn nhất.
    """
    arr = np.asarray(prob_or_mask_zyx)
    if np.issubdtype(arr.dtype, np.floating):
        mask = (arr >= float(threshold)).astype(np.uint8)
    else:
        mask = (arr > 0).astype(np.uint8)

    if lung_mask_zyx is not None:
        keep_lung = lung_keep_mask(lung_mask_zyx, tumor_mask=None)
        mask &= keep_lung

    if mask.sum() == 0:
        return mask

    # Lấp lỗ rỗng 2D trên từng lát cắt có dự đoán dương tính
    if fill_holes:
        for z in np.where(mask.sum(axis=(1, 2)) > 0)[0]:
            mask[z] = ndi.binary_fill_holes(mask[z]).astype(np.uint8)

    # Đánh nhãn các khối liên thông 3D (26-connectivity: khối lập phương 3x3x3)
    structure_3d = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, num_features = ndi.label(mask, structure=structure_3d)
    if num_features == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    objects = ndi.find_objects(labeled)
    valid_components = []

    for comp_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        sub_mask = (labeled[slc] == comp_id)
        voxel_count = int(sub_mask.sum())
        if voxel_count == 0:
            continue

        z_span = int(sub_mask.shape[0])
        if z_span < min_slices_span:
            continue

        slice_areas = sub_mask.reshape(z_span, -1).sum(axis=1)
        max_area_mm2 = float(slice_areas.max())
        if max_area_mm2 < float(min_slice_area_mm2):
            continue

        vol_mm3 = float(voxel_count) * float(spacing_z)
        if vol_mm3 < float(min_volume_mm3):
            continue

        valid_components.append((comp_id, voxel_count))

    if not valid_components:
        return np.zeros_like(mask, dtype=np.uint8)

    if top_k_components and top_k_components > 0 and len(valid_components) > top_k_components:
        valid_components.sort(key=lambda x: x[1], reverse=True)
        valid_components = valid_components[:top_k_components]

    keep_ids = np.array([cid for cid, _ in valid_components], dtype=labeled.dtype)
    cleaned_mask = np.isin(labeled, keep_ids).astype(np.uint8)
    return cleaned_mask


def _selftest():
    print("🧪 Bắt đầu selftest common.eval3d.postprocess3d...")
    # 1. Kiểm thử tính nghịch đảo hoàn hảo của unpad_or_uncrop_xy
    for hw in [(360, 380), (512, 512), (580, 600)]:
        arr = np.zeros((4, hw[0], hw[1]), dtype=np.uint8)
        ch, cw = hw[0] // 2, hw[1] // 2
        arr[1:3, ch - 20 : ch + 20, cw - 20 : cw + 20] = 1
        padded = pad_or_crop_xy(arr, size=512, pad_value=0)
        unpadded = unpad_or_uncrop_xy(padded, hw, pad_value=0)
        assert np.array_equal(arr, unpadded), f"Lỗi nghịch đảo pad/crop với kích thước {hw}"

    # 2. Resize mask
    orig_back = resize_from_1mm_3d(np.ones((4, 360, 360), dtype=np.uint8), (4, 512, 512), is_mask=True)
    assert orig_back.shape == (4, 512, 512) and set(np.unique(orig_back)) <= {0, 1}

    # 3. Lọc nhiễu 3D
    gt = np.zeros((10, 128, 128), dtype=np.uint8)
    lung = np.zeros((10, 128, 128), dtype=np.uint8)
    lung[1:9, 10:118, 10:118] = 1
    gt[4:7, 50:60, 50:60] = 1  # 10x10 = 100 mm2 per slice

    noisy = gt.astype(np.float32) * 0.95
    noisy[5, 54:56, 54:56] = 0.1   # lỗ khuyết ở giữa
    noisy[2, 30:33, 30:33] = 0.90  # 9 mm2 < 20 mm2 -> phải bị lọc
    noisy[0, 40:60, 40:60] = 0.95  # ngoài phổi -> phải bị cắt

    cleaned = postprocess_3d_mask(noisy, lung_mask_zyx=lung, threshold=0.5, min_slice_area_mm2=20.0, fill_holes=True)
    assert np.array_equal(cleaned, gt), "Hậu xử lý 3D phải khôi phục 100% khối u thật và loại bỏ nhiễu!"
    print("✅ Selftest common.eval3d.postprocess3d hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
