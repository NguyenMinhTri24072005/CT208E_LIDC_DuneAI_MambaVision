# -*- coding: utf-8 -*-
"""
common/eval3d/volume_inference.py - Suy luận 3D toàn khối cho một ca bệnh nhân (2.5D Sliding-Window).

Quy tắc bất biến:
  - Hàm `predict_patient_volume` nhận instance `nn.Module` độc lập kiến trúc.
  - Không import unet_base hay mamba_unet vào module này (R2).
  - Không chứa lớp kế thừa nn.Module (R11).
  - Tiền xử lý và suy luận theo batch để chống tràn bộ nhớ GPU (OOM).
  - Hỗ trợ Test-Time Augmentation (TTA).
"""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import TUMOR_SIZE_LOW_LIM
from common.data_pipeline.medical_prep import (
    HU_FLOOR,
    IMAGE_SIZE,
    apply_lung_mask,
    load_cached,
    lung_cache_path,
    lung_keep_mask,
    pad_or_crop_xy,
    window_hu_torch,
)
from common.eval3d.metrics3d import compute_3d_metrics
from common.eval3d.postprocess3d import postprocess_3d_mask, unpad_or_uncrop_xy


@torch.no_grad()
def predict_patient_volume(
    model: nn.Module,
    patient_cache_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    batch_size: int = 8,
    in_channels: int = 3,
    image_size: int = IMAGE_SIZE,
    apply_lung: bool = True,
    threshold: float = 0.5,
    min_slice_area_mm2: float = TUMOR_SIZE_LOW_LIM,
    min_volume_mm3: float = 0.0,
    tta: bool = False,
    use_amp: bool = True,
    compute_post_and_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Suy luận 3D hoàn chỉnh cho 1 bệnh nhân từ thư mục cache_1mm/<patient_id>:
      1. Đọc image_1mm.npy, mask_1mm.npy, lung_1mm.npy, meta.json.
      2. Áp mặt nạ phổi (KHÔNG dùng tumor_mask để tránh rò rỉ nhãn).
      3. Chỉ chạy mạng trên các lát cắt nằm trong vùng có phổi để tiết kiệm GPU và chặn dương tính giả ngoài phổi.
      4. Khôi phục kích thước từ (Z, 512, 512) về (Z, H_1mm, W_1mm) bằng `unpad_or_uncrop_xy`.
      5. Chạy hậu xử lý 3D (`postprocess_3d_mask`) để lọc nhiễu và tính Dice 3D trước/sau hậu xử lý.
    """
    device = device or next(model.parameters()).device
    is_cuda = device.type == "cuda"
    model.eval()

    patient_cache_dir = Path(patient_cache_dir)
    img_1mm, msk_1mm, meta = load_cached(patient_cache_dir, mmap=True)
    num_slices, h_1mm, w_1mm = img_1mm.shape
    spacing_z = float(meta.get("orig_spacing_xyz", [1.0, 1.0, 1.0])[2])

    lung_p = lung_cache_path(patient_cache_dir)
    has_lung = apply_lung and lung_p.exists() and (meta.get("lung_ok", True) is not False)
    lung_1mm = np.asarray(np.load(lung_p, mmap_mode="r")) if has_lung else None

    vol_hu = np.asarray(img_1mm)
    if lung_1mm is not None:
        keep_lung = lung_keep_mask(lung_1mm, tumor_mask=None)
        vol_hu = apply_lung_mask(vol_hu, keep_lung, fill=HU_FLOOR)
        active_z = np.where(keep_lung.reshape(num_slices, -1).sum(axis=1) > 0)[0].tolist()
    else:
        keep_lung = None
        active_z = list(range(num_slices))

    vol_padded = pad_or_crop_xy(vol_hu, size=image_size, pad_value=HU_FLOOR)
    prob_padded = np.zeros((num_slices, image_size, image_size), dtype=np.float32)

    half_k = in_channels // 2
    amp_ctx = (
        torch.amp.autocast("cuda", enabled=True)
        if (use_amp and is_cuda and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"))
        else nullcontext()
    )

    for start in range(0, len(active_z), batch_size):
        batch_z = active_z[start : start + batch_size]
        stacks = []
        for z in batch_z:
            z_idx = [min(max(0, z + off), num_slices - 1) for off in range(-half_k, half_k + 1)]
            stacks.append(vol_padded[z_idx])

        batch_hu = torch.from_numpy(np.stack(stacks, axis=0)).to(device, non_blocking=is_cuda)
        batch_x = window_hu_torch(batch_hu)

        with amp_ctx:
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            if tta:
                # Test-Time Augmentation: lật ngang và lật dọc
                logits_h = torch.flip(model(torch.flip(batch_x, dims=[3])), dims=[3])
                logits_v = torch.flip(model(torch.flip(batch_x, dims=[2])), dims=[2])
                probs = (probs + torch.sigmoid(logits_h) + torch.sigmoid(logits_v)) / 3.0

        probs_np = probs.squeeze(1).to(torch.float32).cpu().numpy()
        for i, z in enumerate(batch_z):
            prob_padded[z] = probs_np[i]

    prob_1mm = unpad_or_uncrop_xy(prob_padded, (h_1mm, w_1mm), pad_value=0.0)
    gt_1mm = (np.asarray(msk_1mm) > 0).astype(np.uint8)

    if not compute_post_and_metrics:
        return {
            "prob_1mm": prob_1mm,
            "raw_mask_1mm": None,
            "clean_mask_1mm": None,
            "gt_mask_1mm": gt_1mm,
            "meta": meta,
            "metrics_raw": None,
            "metrics_post": None,
        }

    raw_mask_1mm = (prob_1mm >= float(threshold)).astype(np.uint8)
    if keep_lung is not None:
        raw_mask_1mm &= keep_lung

    clean_mask_1mm = postprocess_3d_mask(
        prob_1mm,
        lung_mask_zyx=keep_lung,
        threshold=threshold,
        min_slice_area_mm2=min_slice_area_mm2,
        min_volume_mm3=min_volume_mm3,
        spacing_z=spacing_z,
        fill_holes=True,
    )

    metrics_raw = compute_3d_metrics(raw_mask_1mm, gt_1mm, spacing_z=spacing_z, compute_hd95=False)
    metrics_post = compute_3d_metrics(clean_mask_1mm, gt_1mm, spacing_z=spacing_z, compute_hd95=True)

    return {
        "prob_1mm": prob_1mm,
        "raw_mask_1mm": raw_mask_1mm,
        "clean_mask_1mm": clean_mask_1mm,
        "gt_mask_1mm": gt_1mm,
        "meta": meta,
        "metrics_raw": metrics_raw,
        "metrics_post": metrics_post,
    }
