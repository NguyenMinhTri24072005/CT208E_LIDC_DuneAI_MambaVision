# -*- coding: utf-8 -*-
"""
common/eval3d/nrrd_io.py - Đọc và ghi mặt nạ 3D chuẩn y tế định dạng NRRD.

Chuyển đổi kích thước từ không gian 1mm về lại độ phân giải gốc của máy chụp CT,
bảo toàn thông tin tọa độ thực (Spacing, Origin, Direction matrix).
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.eval3d.postprocess3d import resize_from_1mm_3d


def save_prediction_nrrd(
    clean_mask_1mm: np.ndarray,
    meta: Dict[str, Any],
    out_nrrd_path: Union[str, Path],
    reference_nrrd_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Đưa mặt nạ dự đoán từ lưới 1mm (Z, H_1mm, W_1mm) về lưới gốc (Z, H_orig, W_orig)
    và lưu thành file `.nrrd` giữ nguyên hệ tọa độ y khoa (Spacing, Origin, Direction).
    """
    if sitk is None or not hasattr(sitk, "GetImageFromArray"):
        raise ImportError("Cần cài đặt SimpleITK để xuất file NRRD: pip install SimpleITK")

    orig_shape_zyx = meta.get("orig_shape_zyx", list(clean_mask_1mm.shape))
    mask_orig = resize_from_1mm_3d(clean_mask_1mm, orig_shape_zyx, is_mask=True)

    sitk_mask = sitk.GetImageFromArray(mask_orig.astype(np.uint8))
    if reference_nrrd_path and Path(reference_nrrd_path).exists():
        ref_img = sitk.ReadImage(str(reference_nrrd_path))
        sitk_mask.CopyInformation(ref_img)
    elif "orig_spacing_xyz" in meta:
        sitk_mask.SetSpacing(tuple(float(s) for s in meta["orig_spacing_xyz"]))

    out_nrrd_path = Path(out_nrrd_path)
    out_nrrd_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk_mask, str(out_nrrd_path), useCompression=True)
    return mask_orig


def load_nrrd_mask(nrrd_path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Đọc file NRRD và trả về mảng 3D numpy (Z, Y, X) kèm voxel spacing (sx, sy, sz)."""
    if sitk is None:
        raise ImportError("Cần cài đặt SimpleITK để đọc file NRRD: pip install SimpleITK")
    img = sitk.ReadImage(str(nrrd_path))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()  # (sx, sy, sz)
    return arr, spacing
