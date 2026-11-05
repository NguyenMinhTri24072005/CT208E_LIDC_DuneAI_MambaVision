# -*- coding: utf-8 -*-
"""
postprocess.py - Suy luận 3D toàn khối (2.5D Sliding-Window), Hậu xử lý hình thái học 3D,
đánh giá chỉ số y khoa cấp độ bệnh nhân (3D Patient-level Metrics) và xuất file NRRD chuẩn.

Tuân thủ chuẩn kỹ thuật của tác giả DuneAI (s.primakov) & medical_prep.py:
  1. Suy luận 2.5D không rò rỉ nhãn:
     - Chỉ áp mặt nạ phổi thuần túy (lung_keep_mask với tumor_mask=None).
     - Bỏ qua (gán xác suất = 0) các lát cắt nằm ngoài vùng phổi để tăng tốc GPU và chặn dương tính giả ở cổ/bụng.
     - Hỗ trợ Test-Time Augmentation (TTA: lật ngang/dọc) tùy chọn.
  2. Khôi phục không gian chính xác (Exact Inverse Geometry):
     - unpad_or_uncrop_xy: Đảo ngược chính xác hàm pad_or_crop_xy từ (Z, 512, 512) về (Z, H_1mm, W_1mm).
     - resize_from_1mm_3d: Đưa từ lưới 1mm (Z, H_1mm, W_1mm) về lưới DICOM/NRRD gốc (Z, H_orig, W_orig).
  3. Hậu xử lý 3D (3D Morphological & Connected-Components Filtering):
     - Giới hạn vùng dự đoán trong mặt nạ phổi (nếu có).
     - Lấp lỗ rỗng 2D/3D bên trong khối u (binary_fill_holes).
     - Lọc liên thông 3D (26-connectivity): Loại bỏ các cụm nhiễu có diện tích lát cắt cực đại < TUMOR_SIZE_LOW_LIM (20 mm²)
       hoặc thể tích 3D < min_volume_mm3.
  4. Đánh giá chuẩn y khoa 3D:
     - 3D Dice, 3D IoU, Precision, Recall, chênh lệch thể tích (mm³) và khoảng cách bề mặt HD95 (mm).

Dùng:
  python DuneAI_Remake/src/postprocess.py selftest
  python DuneAI_Remake/src/postprocess.py eval --cache-dir data/cache_1mm --checkpoint checkpoints/dune_unet_25d/best_model.pth --split val
  python DuneAI_Remake/src/postprocess.py predict --patient-cache data/cache_1mm/LIDC-IDRI-0002 --checkpoint checkpoints/dune_unet_25d/best_model.pth --out-nrrd outputs/LIDC-IDRI-0002_pred.nrrd
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
for p in (str(THIS_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import configs.config as cfg
from models.dune_unet import DuneUNet
from medical_prep import (
    HU_FLOOR,
    IMAGE_SIZE,
    _axis_window,
    apply_lung_mask,
    load_cached,
    lung_cache_path,
    lung_keep_mask,
    pad_or_crop_xy,
    window_hu_torch,
)

DEFAULT_TUMOR_LOW_LIM = getattr(cfg, "TUMOR_SIZE_LOW_LIM", 20.0)


# ======================================================================
# 1. KHÔI PHỤC HÌNH HỌC KHÔNG GIAN (INVERSE SPATIAL TRANSFORMATIONS)
# ======================================================================
def unpad_or_uncrop_xy(x_padded, target_hw, pad_value=0):
    """
    Hàm nghịch đảo chính xác tuyệt đối của `pad_or_crop_xy` trong `medical_prep.py`.
    Chuyển mảng/tensor từ kích thước chuẩn mạng (..., IMAGE_SIZE, IMAGE_SIZE)
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


def resize_from_1mm_3d(volume_1mm, orig_shape_zyx, is_mask=True):
    """
    Đưa khối 3D từ lưới 1mm (Z, H_1mm, W_1mm) về lại độ phân giải gốc (Z, H_orig, W_orig)
    của file NRRD/DICOM ban đầu để xuất file cho bác sĩ hoặc đối chiếu trên 3D Slicer / Viewer.
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


# ======================================================================
# 2. HẬU XỬ LÝ HÌNH THÁI HỌC & LỌC LIÊN THÔNG 3D (3D POST-PROCESSING)
# ======================================================================
def postprocess_3d_mask(
    prob_or_mask_zyx,
    lung_mask_zyx=None,
    threshold=0.5,
    min_slice_area_mm2=DEFAULT_TUMOR_LOW_LIM,
    min_volume_mm3=0.0,
    spacing_z=1.0,
    fill_holes=True,
    min_slices_span=1,
    top_k_components=0,
):
    """
    Hậu xử lý khối dự đoán 3D trên lưới 1mm (1 pixel trên mặt phẳng XY = 1.0 mm²):
      1. Ngưỡng hóa xác suất -> nhị phân (uint8).
      2. Cắt bỏ mọi dự đoán nằm ngoài mặt nạ phổi (nếu có lung_mask_zyx).
      3. Lấp các lỗ khuyết nhỏ bên trong tâm khối u trên từng lát cắt 2D (binary_fill_holes).
      4. Phân tích thành phần liên thông 3D (26-connectivity):
         - Loại bỏ cụm nhiễu có diện tích mặt cắt ngang lớn nhất < min_slice_area_mm2 (20.0 mm² theo chuẩn tác giả).
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

        # Số lát cắt Z mà cụm này trải qua
        z_span = int(sub_mask.shape[0])
        if z_span < min_slices_span:
            continue

        # Diện tích mặt cắt ngang cực đại (ở lưới 1mm, 1 pixel = 1.0 mm²)
        slice_areas = sub_mask.reshape(z_span, -1).sum(axis=1)
        max_area_mm2 = float(slice_areas.max())
        if max_area_mm2 < float(min_slice_area_mm2):
            continue

        # Thể tích vật lý 3D (mm³) = số voxel * (1mm * 1mm * spacing_z)
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


# ======================================================================
# 3. CHỈ SỐ ĐÁNH GIÁ 3D CẤP ĐỘ BỆNH NHÂN (3D PATIENT-LEVEL METRICS)
# ======================================================================
def _compute_hd95_mm(pred_bin, gt_bin, spacing_zyx=(1.0, 1.0, 1.0)):
    """
    Tính 95th percentile Hausdorff Distance (HD95) theo đơn vị mm vật lý giữa 2 mặt nạ 3D.
    Nếu một trong hai rỗng: trả về 0.0 khi cả hai cùng rỗng, hoặc NaN khi chỉ một bên rỗng.
    """
    pred_any = bool(pred_bin.any())
    gt_any = bool(gt_bin.any())
    if not pred_any and not gt_any:
        return 0.0
    if not pred_any or not gt_any:
        return float("nan")

    # Trích xuất biên bề mặt 3D bằng phép co (erosion)
    pred_border = pred_bin ^ ndi.binary_erosion(pred_bin)
    gt_border = gt_bin ^ ndi.binary_erosion(gt_bin)
    if not pred_border.any():
        pred_border = pred_bin
    if not gt_border.any():
        gt_border = gt_bin

    dt_gt = ndi.distance_transform_edt(~gt_border, sampling=spacing_zyx)
    dt_pred = ndi.distance_transform_edt(~pred_border, sampling=spacing_zyx)

    dists_pred_to_gt = dt_gt[pred_border]
    dists_gt_to_pred = dt_pred[gt_border]
    all_dists = np.concatenate([dists_pred_to_gt, dists_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def compute_3d_metrics(pred_mask, gt_mask, spacing_z=1.0, compute_hd95=True, eps=1e-6):
    """
    Tính bộ chỉ số phân vùng 3D trên toàn bộ thể tích phổi của bệnh nhân (lưới 1mm):
      - dice_3d, iou_3d, precision_3d, recall_3d
      - pred_volume_mm3, gt_volume_mm3
      - hd95_mm (Hausdorff Distance 95% tính bằng mm)
    """
    pred_b = np.asarray(pred_mask) > 0
    gt_b = np.asarray(gt_mask) > 0

    tp = float(np.logical_and(pred_b, gt_b).sum())
    fp = float(np.logical_and(pred_b, ~gt_b).sum())
    fn = float(np.logical_and(~pred_b, gt_b).sum())

    pred_sum = tp + fp
    gt_sum = tp + fn
    union = tp + fp + fn

    # Trường hợp ca bệnh hoàn toàn không có u và AI cũng dự đoán không có u -> đúng tuyệt đối (1.0)
    if pred_sum == 0 and gt_sum == 0:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
    else:
        dice = (2.0 * tp + eps) / (pred_sum + gt_sum + eps)
        iou = (tp + eps) / (union + eps)
        precision = (tp + eps) / (pred_sum + eps) if pred_sum > 0 else 0.0
        recall = (tp + eps) / (gt_sum + eps) if gt_sum > 0 else 0.0

    hd95 = _compute_hd95_mm(pred_b, gt_b, spacing_zyx=(float(spacing_z), 1.0, 1.0)) if compute_hd95 else float("nan")

    return {
        "dice_3d": float(dice),
        "iou_3d": float(iou),
        "precision_3d": float(precision),
        "recall_3d": float(recall),
        "pred_voxels": int(pred_sum),
        "gt_voxels": int(gt_sum),
        "pred_volume_mm3": round(float(pred_sum * spacing_z), 2),
        "gt_volume_mm3": round(float(gt_sum * spacing_z), 2),
        "hd95_mm": round(float(hd95), 3) if np.isfinite(hd95) else float("nan"),
    }


# ======================================================================
# 4. SUY LUẬN 3D TOÀN BỆNH NHÂN (2.5D SLIDING-WINDOW RECONSTRUCTION)
# ======================================================================
def load_trained_model(checkpoint_path, device=None, default_features=64, in_channels=3):
    """
    Nạp mô hình DuneUNet từ file checkpoint (.pth).
    Tự động đọc `meta` (features, backbone, in_channels) hoặc suy luận từ kích thước trọng số
    để tương thích với cả Baseline U-Net và MambaVision-UNet.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    meta = (ckpt.get("meta") or {}) if isinstance(ckpt, dict) else {}
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Bóc tiền tố 'module.' nếu checkpoint từng lưu trực tiếp từ DataParallel
    cleaned_state = {}
    for k, v in state.items():
        cleaned_state[k[7:] if k.startswith("module.") else k] = v

    first_conv_key = "inc.double_conv.0.weight"
    if first_conv_key in cleaned_state:
        features = int(cleaned_state[first_conv_key].shape[0])
        in_channels = int(cleaned_state[first_conv_key].shape[1])
    else:
        features = int(meta.get("features", default_features))
        in_channels = int(meta.get("in_channels", in_channels))

    backbone = str(meta.get("backbone", "none"))
    if backbone == "none" and any(k.startswith("encoder.backbone.") for k in cleaned_state):
        backbone = "mambavision_t"

    # Khi nạp từ checkpoint đã train, đặt pretrained=False vì toàn bộ trọng số sẽ được nạp từ cleaned_state
    model = DuneUNet(
        in_channels=in_channels,
        classes=1,
        features=features,
        backbone=backbone,
        pretrained=False,
    )
    model.load_state_dict(cleaned_state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_patient_volume(
    model,
    patient_cache_dir,
    device=None,
    batch_size=8,
    in_channels=3,
    image_size=IMAGE_SIZE,
    apply_lung=True,
    threshold=0.5,
    min_slice_area_mm2=DEFAULT_TUMOR_LOW_LIM,
    min_volume_mm3=0.0,
    tta=False,
    use_amp=True,
):
    """
    Suy luận 3D hoàn chỉnh cho 1 bệnh nhân từ thư mục cache_1mm/<patient_id>:
      1. Đọc image_1mm.npy, mask_1mm.npy, lung_1mm.npy, meta.json.
      2. Áp mặt nạ phổi (KHÔNG dùng tumor_mask để chống rò rỉ nhãn).
      3. Chỉ chạy mạng U-Net trên các lát cắt nằm trong vùng có phổi (tiết kiệm 35-50% thời gian GPU).
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

    # Pad/crop toàn bộ khối về (Z, image_size, image_size) để trích xuất nhanh các lát 2.5D
    vol_padded = pad_or_crop_xy(vol_hu, size=image_size, pad_value=HU_FLOOR)
    prob_padded = np.zeros((num_slices, image_size, image_size), dtype=np.float32)

    half_k = in_channels // 2
    amp_ctx = (
        torch.amp.autocast("cuda", enabled=True)
        if (use_amp and is_cuda and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"))
        else nullcontext()
    )

    # Suy luận theo batch trên các lát cắt chứa nhu mô phổi
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
                # Test-Time Augmentation: trung bình cộng với dự đoán lật ngang và lật dọc
                logits_h = torch.flip(model(torch.flip(batch_x, dims=[3])), dims=[3])
                logits_v = torch.flip(model(torch.flip(batch_x, dims=[2])), dims=[2])
                probs = (probs + torch.sigmoid(logits_h) + torch.sigmoid(logits_v)) / 3.0

        probs_np = probs.squeeze(1).to(torch.float32).cpu().numpy()
        for i, z in enumerate(batch_z):
            prob_padded[z] = probs_np[i]

    # Khôi phục từ (Z, image_size, image_size) về đúng lưới 1mm (Z, H_1mm, W_1mm)
    prob_1mm = unpad_or_uncrop_xy(prob_padded, (h_1mm, w_1mm), pad_value=0.0)
    raw_mask_1mm = (prob_1mm >= float(threshold)).astype(np.uint8)
    if keep_lung is not None:
        raw_mask_1mm &= keep_lung

    # Hậu xử lý hình thái học 3D
    clean_mask_1mm = postprocess_3d_mask(
        prob_1mm,
        lung_mask_zyx=keep_lung,
        threshold=threshold,
        min_slice_area_mm2=min_slice_area_mm2,
        min_volume_mm3=min_volume_mm3,
        spacing_z=spacing_z,
        fill_holes=True,
    )

    gt_1mm = (np.asarray(msk_1mm) > 0).astype(np.uint8)
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


def save_prediction_nrrd(clean_mask_1mm, meta, out_nrrd_path, reference_nrrd_path=None):
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


# ======================================================================
# 5. ĐÁNH GIÁ TOÀN TẬP DỮ LIỆU (PATIENT-LEVEL 3D EVALUATION)
# ======================================================================
def evaluate_dataset_3d(
    cache_dir,
    checkpoint_path,
    split="val",
    batch_size=8,
    threshold=0.5,
    min_slice_area_mm2=DEFAULT_TUMOR_LOW_LIM,
    tta=False,
    out_csv=None,
):
    """
    Chạy suy luận 3D và hậu xử lý cho toàn bộ bệnh nhân thuộc `split` ('val', 'train', hoặc 'all').
    So sánh trực tiếp Dice 3D trước và sau hậu xử lý.
    """
    cache_dir = Path(cache_dir)
    csv_path = cache_dir / "slices.csv"
    if csv_path.exists() and split != "all":
        df_slices = pd.read_csv(csv_path)
        pids = sorted(df_slices[df_slices["split"] == split]["patient_id"].unique().tolist())
    else:
        pids = sorted([
            p.name for p in cache_dir.iterdir()
            if p.is_dir() and (p / "meta.json").exists() and (p / "image_1mm.npy").exists()
        ])

    if not pids:
        raise RuntimeError(f"Không tìm thấy bệnh nhân nào cho split='{split}' trong {cache_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(checkpoint_path, device=device)

    print("=" * 90)
    print(
        f"🩻 ĐÁNH GIÁ 3D CẤP ĐỘ BỆNH NHÂN | Split: {split} ({len(pids)} ca) | "
        f"Thiết bị: {device} | Ngưỡng lọc u: >= {min_slice_area_mm2} mm²"
    )
    print("=" * 90)

    rows = []
    for idx, pid in enumerate(pids, start=1):
        t0 = time.time()
        res = predict_patient_volume(
            model=model,
            patient_cache_dir=cache_dir / pid,
            device=device,
            batch_size=batch_size,
            threshold=threshold,
            min_slice_area_mm2=min_slice_area_mm2,
            tta=tta,
        )
        dt = time.time() - t0
        m_raw = res["metrics_raw"]
        m_post = res["metrics_post"]

        row = {
            "patient_id": pid,
            "dice_3d_raw": round(m_raw["dice_3d"], 4),
            "dice_3d_post": round(m_post["dice_3d"], 4),
            "iou_3d_post": round(m_post["iou_3d"], 4),
            "precision_3d": round(m_post["precision_3d"], 4),
            "recall_3d": round(m_post["recall_3d"], 4),
            "hd95_mm": m_post["hd95_mm"],
            "gt_vol_mm3": m_post["gt_volume_mm3"],
            "pred_vol_mm3": m_post["pred_volume_mm3"],
            "time_sec": round(dt, 2),
        }
        rows.append(row)
        diff_dice = m_post["dice_3d"] - m_raw["dice_3d"]
        print(
            f"[{idx:02d}/{len(pids):02d}] {pid} ({dt:.1f}s) | "
            f"3D Dice: {m_raw['dice_3d']:.4f} -> {m_post['dice_3d']:.4f} ({diff_dice:+.4f}) | "
            f"IoU: {m_post['iou_3d']:.4f} | HD95: {m_post['hd95_mm']} mm | "
            f"Vol(GT/Pred): {m_post['gt_volume_mm3']}/{m_post['pred_volume_mm3']} mm³"
        )

    df_res = pd.DataFrame(rows)
    mean_raw = float(df_res["dice_3d_raw"].mean())
    mean_post = float(df_res["dice_3d_post"].mean())
    mean_iou = float(df_res["iou_3d_post"].mean())

    print("=" * 90)
    print(
        f"📊 TỔNG KẾT ({len(pids)} bệnh nhân) | "
        f"Mean 3D Dice (Raw): {mean_raw:.4f} -> Mean 3D Dice (Post): {mean_post:.4f} "
        f"({mean_post - mean_raw:+.4f}) | Mean 3D IoU: {mean_iou:.4f}"
    )
    print("=" * 90)

    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(out_csv, index=False)
        print(f"✅ Đã lưu bảng báo cáo đánh giá 3D tại: {out_csv}")

    return df_res


# ======================================================================
# 6. KIỂM THỬ TỰ ĐỘNG (SELFTEST) & CLI
# ======================================================================
def _cmd_selftest(_):
    print("Đang chạy kiểm thử tự động postprocess.py (Inverse Geometry + 3D Filtering + 3D Pipeline)...")

    # 1. Kiểm thử tính nghịch đảo hoàn hảo của unpad_or_uncrop_xy và resize_from_1mm_3d
    for hw in [(360, 380), (512, 512), (580, 600)]:
        arr = np.zeros((4, hw[0], hw[1]), dtype=np.uint8)
        ch, cw = hw[0] // 2, hw[1] // 2
        arr[1:3, ch - 20 : ch + 20, cw - 20 : cw + 20] = 1
        padded = pad_or_crop_xy(arr, size=512, pad_value=0)
        unpadded = unpad_or_uncrop_xy(padded, hw, pad_value=0)
        assert np.array_equal(arr, unpadded), f"Lỗi nghịch đảo pad/crop với kích thước {hw}"

    orig_back = resize_from_1mm_3d(np.ones((4, 360, 360), dtype=np.uint8), (4, 512, 512), is_mask=True)
    assert orig_back.shape == (4, 512, 512) and set(np.unique(orig_back)) <= {0, 1}

    # 2. Kiểm thử bộ lọc hình thái học 3D (loại bỏ nhiễu < 20mm², nhiễu ngoài phổi, và lấp lỗ rỗng)
    gt = np.zeros((10, 128, 128), dtype=np.uint8)
    lung = np.zeros((10, 128, 128), dtype=np.uint8)
    lung[1:9, 10:118, 10:118] = 1

    # Khối u thật: 10x10 = 100 mm² mỗi lát trên các lát 4, 5, 6 (tổng 300 voxels)
    gt[4:7, 50:60, 50:60] = 1

    noisy_prob = gt.astype(np.float32) * 0.95
    # Tạo lỗ thủng ở giữa khối u để kiểm tra binary_fill_holes
    noisy_prob[5, 54:56, 54:56] = 0.1
    # Thêm đốm dương tính giả nhỏ trong phổi: 3x3 = 9 mm² (< 20 mm²) ở lát 2 -> phải bị lọc bỏ!
    noisy_prob[2, 30:33, 30:33] = 0.90
    # Thêm đám nhiễu to nằm NGOÀI phổi (lát 0 và góc biên) -> phải bị mặt nạ phổi cắt bỏ!
    noisy_prob[0, 40:60, 40:60] = 0.95
    noisy_prob[5, 0:8, 0:8] = 0.95

    raw_m = compute_3d_metrics(noisy_prob >= 0.5, gt, spacing_z=1.5)
    cleaned = postprocess_3d_mask(
        noisy_prob, lung_mask_zyx=lung, threshold=0.5, min_slice_area_mm2=20.0, spacing_z=1.5, fill_holes=True
    )
    post_m = compute_3d_metrics(cleaned, gt, spacing_z=1.5)

    assert np.array_equal(cleaned, gt), "Hậu xử lý 3D phải khôi phục chính xác 100% khối u gốc!"
    assert post_m["dice_3d"] > 0.999 and post_m["hd95_mm"] == 0.0
    assert raw_m["dice_3d"] < 0.70, f"Dice trước hậu xử lý phải thấp do nhiễu, nhận được {raw_m['dice_3d']}"

    # 3. Kiểm thử tích hợp toàn trình: Load Checkpoint -> Suy luận 2.5D trên Cache -> Hậu xử lý 3D
    tmp = Path(tempfile.mkdtemp())
    try:
        p_dir = tmp / "LIDC-IDRI-0001"
        p_dir.mkdir(parents=True)
        img = np.full((8, 100, 100), -800, dtype=np.int16)
        msk = np.zeros((8, 100, 100), dtype=np.uint8)
        lung_c = np.zeros((8, 100, 100), dtype=np.uint8)
        lung_c[1:7, 10:90, 10:90] = 1
        msk[3:5, 40:50, 40:50] = 1
        img[3:5, 40:50, 40:50] = 50

        np.save(p_dir / "image_1mm.npy", img)
        np.save(p_dir / "mask_1mm.npy", msk)
        np.save(p_dir / "lung_1mm.npy", lung_c)
        with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "orig_shape_zyx": [8, 128, 128],
                "orig_spacing_xyz": [0.78, 0.78, 1.5],
                "shape_1mm_zyx": [8, 100, 100],
                "lung_ok": True,
            }, f)

        model = DuneUNet(in_channels=3, classes=1, features=8)
        ckpt_path = tmp / "dummy_model.pth"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        loaded_model = load_trained_model(ckpt_path, device=torch.device("cpu"))
        res = predict_patient_volume(
            loaded_model, p_dir, batch_size=2, image_size=128, tta=True, use_amp=False
        )
        assert res["prob_1mm"].shape == (8, 100, 100)
        assert res["clean_mask_1mm"].shape == (8, 100, 100)
        assert "dice_3d" in res["metrics_post"]

        print(
            f"✅ selftest postprocess.py: OK! "
            f"(3D Noise Filter: Raw Dice {raw_m['dice_3d']:.4f} -> Clean Dice {post_m['dice_3d']:.4f}, HD95={post_m['hd95_mm']} mm)"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _cmd_predict(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(a.checkpoint, device=device)
    res = predict_patient_volume(
        model=model,
        patient_cache_dir=a.patient_cache,
        device=device,
        batch_size=a.batch_size,
        threshold=a.threshold,
        min_slice_area_mm2=a.min_area,
        tta=a.tta,
    )
    m_raw, m_post = res["metrics_raw"], res["metrics_post"]
    print("=" * 75)
    print(f"🩻 KẾT QUẢ SUY LUẬN 3D: {Path(a.patient_cache).name}")
    print("=" * 75)
    print(f"  - 3D Dice (Trước hậu xử lý): {m_raw['dice_3d']:.4f} ({m_raw['pred_volume_mm3']} mm³)")
    print(f"  - 3D Dice (Sau hậu xử lý)  : {m_post['dice_3d']:.4f} ({m_post['pred_volume_mm3']} mm³)")
    print(f"  - Thể tích khối u thực (GT): {m_post['gt_volume_mm3']} mm³ | HD95: {m_post['hd95_mm']} mm")
    if a.out_nrrd:
        save_prediction_nrrd(res["clean_mask_1mm"], res["meta"], a.out_nrrd, a.ref_nrrd)
        print(f"💾 Đã xuất file mặt nạ 3D NRRD tại: {a.out_nrrd}")


def _cmd_eval(a):
    evaluate_dataset_3d(
        cache_dir=a.cache_dir,
        checkpoint_path=a.checkpoint,
        split=a.split,
        batch_size=a.batch_size,
        threshold=a.threshold,
        min_slice_area_mm2=a.min_area,
        tta=a.tta,
        out_csv=a.out_csv,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="Kiểm thử tự động toàn bộ thuật toán hậu xử lý & suy luận 3D")
    s.set_defaults(fn=_cmd_selftest)

    p = sub.add_parser("predict", help="Suy luận 3D cho 1 bệnh nhân và xuất file NRRD")
    p.add_argument("--patient-cache", required=True, help="Đường dẫn tới data/cache_1mm/<patient_id>")
    p.add_argument("--checkpoint", required=True, help="Đường dẫn tới best_model.pth")
    p.add_argument("--out-nrrd", default=None, help="Đường dẫn xuất file pred_mask.nrrd (về lại độ phân giải gốc)")
    p.add_argument("--ref-nrrd", default=None, help="Đường dẫn tới image.nrrd gốc để copy header tọa độ")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-area", type=float, default=DEFAULT_TUMOR_LOW_LIM, help="Ngưỡng lọc diện tích lát cắt (mm²)")
    p.add_argument("--tta", action="store_true", help="Bật Test-Time Augmentation (lật ngang/dọc)")
    p.set_defaults(fn=_cmd_predict)

    e = sub.add_parser("eval", help="Đánh giá 3D cấp độ bệnh nhân trên toàn tập val/train")
    e.add_argument("--cache-dir", default="data/cache_1mm")
    e.add_argument("--checkpoint", required=True, help="Đường dẫn tới best_model.pth")
    e.add_argument("--split", choices=["val", "train", "all"], default="val")
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--threshold", type=float, default=0.5)
    e.add_argument("--min-area", type=float, default=DEFAULT_TUMOR_LOW_LIM)
    e.add_argument("--tta", action="store_true")
    e.add_argument("--out-csv", default="checkpoints/eval_3d_results.csv")
    e.set_defaults(fn=_cmd_eval)

    args = ap.parse_args()
    args.fn(args)