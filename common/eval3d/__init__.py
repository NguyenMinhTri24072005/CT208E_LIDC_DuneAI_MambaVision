# -*- coding: utf-8 -*-
"""common.eval3d package - Các hàm hậu xử lý và đánh giá thể tích 3D độc lập kiến trúc."""

from common.eval3d.postprocess3d import (
    postprocess_3d_mask,
    resize_from_1mm_3d,
    unpad_or_uncrop_xy,
)
from common.eval3d.metrics3d import (
    _compute_hd95_mm,
    compute_3d_metrics,
)
from common.eval3d.volume_inference import predict_patient_volume
from common.eval3d.nrrd_io import (
    load_nrrd_mask,
    save_prediction_nrrd,
)
from common.eval3d.eval_runner import evaluate_dataset_3d

__all__ = [
    "unpad_or_uncrop_xy",
    "resize_from_1mm_3d",
    "postprocess_3d_mask",
    "_compute_hd95_mm",
    "compute_3d_metrics",
    "predict_patient_volume",
    "save_prediction_nrrd",
    "load_nrrd_mask",
    "evaluate_dataset_3d",
]
