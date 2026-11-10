# -*- coding: utf-8 -*-
"""
medical_prep.py - Tiền xử lý CT theo pipeline DuneAI (s.primakov), tối ưu cho GPU.

CÁC HÀM TIỀN XỬ LÝ CHUẨN CỦA DUNEAI:
  load_nrrd(path)                                  -> (volume (Z,H,W), spacing (x,y,z))
  clip_and_window_hu(volume, ww, wl, step)         -> float32 [0,1]   (numpy)
  window_hu_torch(x, ...)                          -> cùng công thức, chạy trên GPU
  resize_and_pad_3d(volume, spacing, target_size, is_mask)
  extract_lungs_mask(volume)                       -> mask phổi (chuyển thể get_seg_lungs + create_flipped_mask)

THỨ TỰ BƯỚC (đúng Patient_data_generator._load_patient của tác giả):
  1. Cắt đáy HU            img[img < -1000] = -1000
  2. (Tùy chọn) tách phổi trên HU thô, lưới gốc 512x512, rồi che ngoài phổi bằng -1000
  3. Resize mặt phẳng trục về 1 mm/pixel trên HU thô (trục z giữ nguyên)
  4. Crop/pad về IMAGE_SIZE x IMAGE_SIZE, pad bằng -1000 HU (không khí)
  5. Cửa sổ [WL - WW/2, WL + WW/2], CẬN DƯỚI BỊ KẸP Ở -1000   (_apply_window, dòng 455-457)
  6. (Tùy chọn) lượng tử hóa theo RESAMPLING_STEP
  7. Đưa về [0, 1]  (lựa chọn của dự án; tác giả dùng z-score tùy chọn)

LƯU Ý VỀ CỬA SỔ: với WW=1500, WL=-600 cận dưới danh nghĩa là -1350 nhưng tác giả kẹp ở -1000, nên cửa sổ
thực tế là [-1000, 150]. Mô tả "low = -1350" trong tài liệu cần đổi thành -1000, nếu không không khí sẽ là
0.233 thay vì 0 và phần pad (0) sẽ lộ ra thành hình chữ nhật.

KIẾN TRÚC TỐI ƯU
  - Bước 1-4 nặng CPU và KHÔNG phụ thuộc epoch -> chạy MỘT LẦN bằng `build`, lưu .npy int16 HU.
  - Mask phổi cũng được tính một lần (--extract-lungs) và lưu riêng; việc che ảnh làm lúc nạp (rẻ).
  - Bước 5-7 là phép toán theo phần tử -> chạy trên GPU (window_hu_torch).
  - .npy không nén + mmap: chỉ đọc đúng các lát cần dùng.

KHÁC TÁC GIẢ (có chủ ý)
  - Kích thước mới dùng round() thay vì int(); spacing áp đúng trục (x -> rộng, y -> cao).
  - Nội suy ảnh mặc định "nearest" như tác giả (IMAGE_INTERP).
  - Lượng tử hóa mặc định bật; tác giả để tùy chọn và tắt (quantize=False để giống tác giả).
  - create_flipped_mask chỉ port method=0 (nhanh, không hiệu chỉnh lệch cột sống) - đây là cách get_seg_lungs gọi.
  - Bỏ qua phần "kéo dãn mask phổi theo z khi khối u nằm ngoài" (dòng 101-130 của generator); phần hợp mask
    khối u vào mask phổi (dòng 132-134) có trong apply_lung_mask.
  - Gỡ phụ thuộc statsmodels và scipy.ndimage.measurements (đã deprecated).
  - Hai chỗ thêm cho an toàn: lát không có xương thì bỏ qua khi tìm tâm phổi; mask phổi rỗng trả về None.

SimpleITK: GetSpacing() trả về (x, y, z) còn mảng numpy có thứ tự (Z, Y, X).

Dùng:
  python medical_prep.py selftest
  python medical_prep.py build --nrrd-dir <duneai_raw_nrrd> --cache-dir <cache_1mm> --workers 4 [--limit 3] [--extract-lungs]
"""
import argparse
import json
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

try:
    import torch
except ImportError:          # viewer / máy không có torch vẫn dùng được các hàm numpy
    torch = None

# nạp các hằng số từ file config
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
ROOT = _ROOT
from configs.config import IMAGE_SIZE, WINDOW_WIDTH, WINDOW_LEVEL, RESAMPLING_STEP

HU_FLOOR = -1000                 # đáy HU (không khí)
IMAGE_INTERP = "nearest"         # nội suy ảnh khi resize; tác giả dùng nearest
LUNG_THRESHOLD_HU = -320         # ngưỡng nhu mô phổi của tác giả
TRACHEA_COEFF = 0.0069           # tác giả đặt 0.0053 (dòng 229) rồi ghi đè bằng 0.0069 (dòng 230)
_CV2_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
}


# ======================================================================
# 1. ĐỌC DỮ LIỆU
# ======================================================================
def load_nrrd(file_path):
    """
    Đọc file NRRD bằng SimpleITK.
    Trả về:
        - volume: mảng numpy 3D (Z, H, W)
        - spacing: tuple (spacing_x, spacing_y, spacing_z) theo mm  (thứ tự của SimpleITK)
    """
    sitk_img = sitk.ReadImage(str(file_path))
    volume = sitk.GetArrayFromImage(sitk_img)      # (Z, H, W)
    spacing = sitk_img.GetSpacing()                # (x, y, z)
    return volume, spacing


# ======================================================================
# 2. CỬA SỔ HU + LƯỢNG TỬ HÓA  (numpy cho viewer/test, torch cho GPU)
# ======================================================================
def window_bounds(ww=WINDOW_WIDTH, wl=WINDOW_LEVEL):
    """Giống _apply_window của tác giả: cận dưới bị kẹp ở -1000. Mặc định (1500, -600) -> [-1000, 150]."""
    assert 0 < ww < 2000 and -1000 < wl < 200, "WW/WL ngoài miền hợp lệ của tác giả"
    low = max(wl - ww / 2.0, HU_FLOOR)
    high = wl + ww / 2.0
    return low, high


def clip_and_window_hu(volume, ww=WINDOW_WIDTH, wl=WINDOW_LEVEL, step=RESAMPLING_STEP, quantize=True):
    """
    Phiên bản numpy (CPU). Đầu vào là HU THÔ (nên là ảnh đã resize và pad -1000).
    1) cắt đáy -1000  2) kẹp cửa sổ  3) lượng tử hóa theo `step`  4) chuẩn hóa về [0, 1].
    Không sửa mảng đầu vào. Đầu ra float32; không khí = 0.
    """
    low, high = window_bounds(ww, wl)
    vol = np.clip(volume.astype(np.float32), low, high)          # gộp bước cắt đáy và cửa sổ
    if quantize:                                                  # bin = floor((HU - low) / step)
        vol = np.floor((vol - low) / step) * step + low
    return (vol - low) / (high - low)


def window_hu_torch(x, ww=WINDOW_WIDTH, wl=WINDOW_LEVEL, step=RESAMPLING_STEP, quantize=True, out_dtype=None):
    """
    Phiên bản GPU: cùng công thức với clip_and_window_hu nhưng chạy trên tensor (cuda).
    x: tensor int16/float bất kỳ shape (HU). out_dtype: ví dụ torch.float16 để tiết kiệm VRAM.
    """
    assert torch is not None, "Cần cài PyTorch để dùng window_hu_torch"
    low, high = window_bounds(ww, wl)
    x = torch.clamp(x.to(torch.float32), low, high)
    if quantize:
        x = torch.floor((x - low) / step) * step + low
    x = (x - low) / (high - low)
    return x if out_dtype is None else x.to(out_dtype)


# ======================================================================
# 3. RESIZE VỀ 1 mm/PIXEL + CROP/PAD  (làm trên HU thô)
# ======================================================================
def resize_to_1mm_3d(volume, spacing, is_mask=False, interp=IMAGE_INTERP):
    """
    Đưa mặt phẳng trục về 1 mm/pixel (CHƯA pad). Trục z giữ nguyên như tác giả.
    Mask luôn dùng nearest để không tạo giá trị trung gian.
    """
    sx, sy = float(spacing[0]), float(spacing[1])
    z, h, w = volume.shape
    new_h, new_w = max(1, int(round(h * sy))), max(1, int(round(w * sx)))

    if is_mask:
        src = (volume > 0).astype(np.uint8)
        flag = cv2.INTER_NEAREST
    else:
        src = np.ascontiguousarray(volume)
        flag = _CV2_INTERP[interp]

    out = np.empty((z, new_h, new_w), dtype=src.dtype)
    for i in range(z):
        out[i] = cv2.resize(src[i], (new_w, new_h), interpolation=flag)
    return out


def _axis_window(n, size):
    """Trả về (bắt đầu nguồn, bắt đầu đích, độ dài) khi crop (n>=size) hoặc pad (n<size) vào giữa."""
    if n >= size:
        return (n - size) // 2, 0, size
    return 0, (size - n) // 2, n


def pad_or_crop_xy(x, size=IMAGE_SIZE, pad_value=HU_FLOOR):
    """
    Crop hoặc pad hai chiều cuối về (size, size), đặt giữa. Dùng được cho numpy (kể cả memmap) và torch.
    Ảnh HU: pad_value = -1000. Mask: pad_value = 0.
    """
    h, w = x.shape[-2:]
    sy, dy, ly = _axis_window(h, size)
    sx, dx, lx = _axis_window(w, size)
    shape = (*x.shape[:-2], size, size)
    if torch is not None and isinstance(x, torch.Tensor):
        out = torch.full(shape, pad_value, dtype=x.dtype, device=x.device)
    else:
        out = np.full(shape, pad_value, dtype=x.dtype)
    out[..., dy:dy + ly, dx:dx + lx] = x[..., sy:sy + ly, sx:sx + lx]
    return out


def resize_and_pad_3d(volume, spacing, target_size=IMAGE_SIZE, is_mask=False, pad_value=None, interp=IMAGE_INTERP):
    """
    Một pixel ảnh = 1 mm thực tế, rồi crop/pad về target_size x target_size.
    pad_value mặc định: -1000 (HU, không khí) cho ảnh; 0 cho mask.
    LƯU Ý: truyền HU THÔ (chưa cửa sổ). Nếu truyền ảnh đã chuẩn hóa thì phải tự đặt pad_value = 0.
    """
    if pad_value is None:
        pad_value = 0 if is_mask else HU_FLOOR
    small = resize_to_1mm_3d(volume, spacing, is_mask=is_mask, interp=interp)
    return pad_or_crop_xy(small, target_size, pad_value)


# ======================================================================
# 4. TÁCH PHỔI  (chuyển thể lung_extraction_funcs_13_09.py của tác giả)
#    Chạy trên HU THÔ ở lưới gốc (512x512 trong LIDC), TRƯỚC khi resize.
# ======================================================================
def _import_lung_deps():
    try:
        from skimage.measure import label, regionprops
        from skimage.segmentation import clear_border
        from skimage.filters import roberts
        from scipy import ndimage as ndi
    except ImportError as e:
        raise ImportError("extract_lungs_mask cần scikit-image và scipy: pip install scikit-image scipy") from e
    return label, regionprops, clear_border, roberts, ndi


def _get_kernel():
    """Kernel hình tròn 11x11 kèm 8 điểm nhô ra, y hệt get_kernel của tác giả."""
    kernel = np.zeros((11, 11), np.uint8)
    kernel = cv2.circle(kernel, (5, 5), 4, 1, -1)
    for i in [4, 6]:
        for j in [1, 9]:
            kernel[i, j] = 1
            kernel[j, i] = 1
    return kernel


def _find_bounds_z_direction(image):
    z_dist = np.sum(image, axis=(1, 2))
    zi = np.where(z_dist != 0)
    return int((np.max(zi) - np.min(zi)) / 2), int(np.min(zi)), int(np.max(zi))


def _largest_labeled_volumes(im, bg=-1, ratio=0.6):
    """
    Giữ nguyên hành vi của tác giả: trả về nhãn lớn nhất; chỉ khi có ĐÚNG MỘT nhãn có thể tích
    trong (ratio, 1) so với nhãn lớn nhất thì trả thêm nhãn đó (để gộp phổi trái/phải bị tách).
    """
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) == 0:
        return None
    counts = counts / np.max(counts)
    cand = np.where((counts > ratio) & (counts < 1))[0]
    if len(cand) == 1:
        smax = np.where(counts == counts[cand[0]])[0][0]
        return vals[[np.argmax(counts), smax]]
    return [vals[np.argmax(counts)]]


def _create_flipped_mask(image, i_orig, kernel):
    """
    method=0 của tác giả: tìm tâm phổi theo phân bố xương (> 100 HU) rồi lật đối xứng mask qua tâm
    để mask phổi bao cả khối đặc dính thành ngực/trung thất. Trả về (mask, min_bound, max_bound).
    """
    _, min_bound, max_bound = _find_bounds_z_direction(image)
    fl_mask = np.zeros(image.shape, np.uint8)

    lc_buff = []
    for i in range(min_bound, max_bound + 1):
        rib_view = 1 * (i_orig[i, ...] > 100)
        distr = np.sum(rib_view, axis=0)
        thresh_l1 = np.max(distr)
        if thresh_l1 == 0:                    # (thêm) lát không có xương: bỏ qua
            continue
        idx = np.where(distr > (thresh_l1 * 0.15))[0]
        lc_buff.append((np.min(idx) + np.max(idx)) / 2)
    lungs_center = int(np.mean(lc_buff)) if lc_buff else image.shape[2] // 2
    half = int(image.shape[2] / 2)
    W = image.shape[2]

    for i in range(min_bound, max_bound + 1):
        ax = image[i, ...]
        if lungs_center > half:
            part1 = ax[:, 2 * lungs_center - W:lungs_center]
            part2 = np.hstack((ax[:, lungs_center:], np.zeros((image.shape[1], 2 * lungs_center - W))))
            part2 = cv2.dilate(np.array(part2, np.uint8), kernel, iterations=1)
            part1 = cv2.dilate(np.array(part1, np.uint8), kernel, iterations=1)
            fl_mask[i, ...] = np.hstack((np.flip(part2, axis=1), np.flip(part1, axis=1)))
            fl_mask[i, ...] = np.array(fl_mask[i, ...], np.uint8) | np.array(ax, np.uint8)
        elif lungs_center < half:
            part1 = np.hstack((np.zeros((image.shape[1], W - 2 * lungs_center)), ax[:, :lungs_center]))
            part2 = ax[:, lungs_center:W - (W - 2 * lungs_center)]
            part2 = cv2.dilate(np.array(part2, np.uint8), kernel, iterations=1)
            part1 = cv2.dilate(np.array(part1, np.uint8), kernel, iterations=1)
            fl_mask[i, ...] = np.hstack((np.flip(part2, axis=1), np.flip(part1, axis=1)))
            fl_mask[i, ...] = np.array(fl_mask[i, ...], np.uint8) | np.array(ax, np.uint8)
        else:
            fl = np.flip(ax, axis=1)
            fl_mask[i, ...] = np.array(ax, np.uint8) | np.array(fl, np.uint8)
    return fl_mask, min_bound, max_bound


def extract_lungs_mask(volume, return_bounds=False):
    """
    Chuyển thể get_seg_lungs của tác giả. Đầu vào: HU thô (Z, H, W) ở lưới gốc.
    Các bước: ngưỡng < -320 HU -> clear_border (bỏ không khí ngoài, bàn chụp) -> giữ 3 vùng lớn nhất
    -> roberts + binary_fill_holes (lấp mạch máu) -> bỏ khí quản (vùng nhỏ hơn 0.0069*sq/512^2)
    -> bỏ bàn/túi khí (tâm khối nằm dưới 75% chiều cao) -> giữ khối liên thông 3D lớn nhất
    -> lật đối xứng qua tâm cột sống -> morphology -> kéo dãn +-2 lát theo z.
    Trả về mask uint8 (Z, H, W) gồm 0/1, hoặc None nếu không tìm thấy phổi.
    Chạy chậm (vòng lặp từng lát) nên chỉ gọi trong build, không gọi trong vòng huấn luyện.
    """
    label, regionprops, clear_border, roberts, ndi = _import_lung_deps()
    im = np.asarray(volume)
    kernel = _get_kernel()
    sq = im.shape[1] * im.shape[2]
    seg = np.zeros(im.shape, np.uint8)

    # --- ngưỡng, bỏ vùng chạm biên, giữ 3 vùng lớn nhất, lấp lỗ ---
    for nm, ax_slice in enumerate(im < LUNG_THRESHOLD_HU):
        lab = label(clear_border(ax_slice))
        areas = sorted(r.area for r in regionprops(lab))
        if len(areas) > 3:
            for r in regionprops(lab):
                if r.area < areas[-3]:
                    lab[lab == r.label] = 0
        seg[nm] = ndi.binary_fill_holes(roberts(lab > 0))

    # --- bỏ khí quản: vùng nhỏ hơn TRACHEA_COEFF * (sq / 512^2) diện tích ảnh ---
    trachea_coeff = TRACHEA_COEFF * (sq / (512 ** 2))
    for i, ax_slice in enumerate(seg):
        lab = label(ax_slice, connectivity=1, background=0)
        vals, counts = np.unique(lab, return_counts=True)
        counts, vals = counts[vals != 0], vals[vals != 0]
        small = counts / sq < trachea_coeff
        if small.any():
            for v in vals[small]:
                lab[lab == v] = 0
            seg[i] = lab != 0

    # --- bỏ bàn chụp và túi khí: tâm khối ở dưới 75% chiều cao ảnh ---
    lab3 = label(seg, background=0)
    ids = np.unique(lab3)
    ids = ids[ids != 0]
    if len(ids):
        centers = ndi.center_of_mass(np.ones_like(lab3), lab3, ids)
        for i, c in zip(ids, centers):
            if c[1] > seg.shape[1] * 0.75:
                seg[lab3 == i] = 0

    # --- giữ khối liên thông 3D lớn nhất (gộp phổi trái/phải nếu tách đôi) ---
    lab3 = label(seg, background=0)
    l_max = _largest_labeled_volumes(lab3, bg=0)
    if l_max is not None:
        if len(l_max) > 1:
            lab3[lab3 == l_max[1]] = l_max[0]
        seg[lab3 != l_max[0]] = 0
    if seg.sum() == 0:
        return None

    # --- lật đối xứng qua tâm cột sống ---
    seg, min_bound, max_bound = _create_flipped_mask(seg, im, kernel)

    # --- morphology: lấp khoảng trống nhỏ trong mask ---
    for i, ax_slice in enumerate(seg):
        bi = np.array(ax_slice, np.uint8)
        opening = cv2.morphologyEx(1 - bi, cv2.MORPH_OPEN, kernel, iterations=7)
        dilat = cv2.dilate(1 - opening, kernel, iterations=2)
        seg[i, ...] = cv2.erode(dilat, kernel, iterations=1)

    # --- kéo dãn mask +-2 lát theo z ---
    if (min_bound > 2) and (max_bound < seg.shape[0] - 2):
        for i in range(1, 3):
            seg[min_bound - i, ...] = seg[min_bound, ...]
            seg[max_bound + i, ...] = seg[max_bound, ...]
        min_bound -= 2
        max_bound += 2

    seg = (seg > 0).astype(np.uint8)
    return (seg, min_bound, max_bound) if return_bounds else seg


def lung_keep_mask(lung_mask, tumor_mask=None, tumor_dilate_iter=3):
    """
    Vùng được giữ lại = phổi, hợp với mask khối u/nốt đã dilate (kernel 11x11, 3 lần) - như generator dòng 132-134.
    CHỈ truyền tumor_mask lúc HUẤN LUYỆN. Lúc đánh giá/dự đoán mà truyền mask thật vào thì ảnh đầu vào đã
    chứa thông tin nhãn (rò rỉ).
    """
    keep = (lung_mask > 0).astype(np.uint8)
    if tumor_mask is not None:
        kernel = _get_kernel()
        for k in range(len(keep)):
            if tumor_mask[k].any():
                keep[k] |= cv2.dilate((tumor_mask[k] > 0).astype(np.uint8), kernel, iterations=tumor_dilate_iter)
    return keep


def apply_lung_mask(volume_hu, keep, fill=HU_FLOOR):
    """Đặt mọi voxel ngoài vùng giữ lại về -1000 HU (apply_mask của tác giả). Không sửa mảng đầu vào."""
    out = np.array(volume_hu, copy=True)
    out[np.asarray(keep) == 0] = fill
    return out


# ======================================================================
# 5. CACHE MỘT LẦN (CPU)  ->  .npy int16 HU ở 1 mm, chưa pad
# ======================================================================
CACHE_VERSION = 2


def cache_paths(out_dir):
    out_dir = Path(out_dir)
    return out_dir / "image_1mm.npy", out_dir / "mask_1mm.npy", out_dir / "meta.json"


def lung_cache_path(out_dir):
    return Path(out_dir) / "lung_1mm.npy"


def _save_npy(path, arr):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def build_patient_cache(patient_dir, out_dir, interp=IMAGE_INTERP, overwrite=False, extract_lungs=False):
    """
    Bước 1-3 cho một bệnh nhân: đọc image.nrrd (+ mask.nrrd), cắt đáy -1000, (tùy chọn) tách phổi, resize 1 mm.
    Ảnh lưu int16 HU, mask uint8. Mask phổi lưu riêng (lung_1mm.npy), KHÔNG áp vào ảnh trong cache.
    Pad về IMAGE_SIZE làm lúc nạp (rất rẻ) để cache nhỏ hơn.
    """
    patient_dir, out_dir = Path(patient_dir), Path(out_dir)
    img_p, msk_p, meta_p = cache_paths(out_dir)
    lung_p = lung_cache_path(out_dir)
    done = img_p.exists() and msk_p.exists() and meta_p.exists() and (lung_p.exists() or not extract_lungs)
    if not overwrite and done:
        return "skip"

    image_nrrd, mask_nrrd = patient_dir / "image.nrrd", patient_dir / "mask.nrrd"
    if not image_nrrd.exists():
        return "missing_image"

    vol, spacing = load_nrrd(image_nrrd)
    vol = np.maximum(vol, HU_FLOOR).astype(np.int16)             # bước 1: cắt đáy -1000 HU

    if mask_nrrd.exists():
        msk, _ = load_nrrd(mask_nrrd)
        if msk.shape != vol.shape:
            raise ValueError(f"mask {msk.shape} khác image {vol.shape}")
    else:
        msk = np.zeros_like(vol, dtype=np.uint8)

    lung1, lung_meta = None, {}
    if extract_lungs:                                            # bước 2: tách phổi trên lưới gốc
        lung = extract_lungs_mask(vol)
        if lung is None:
            lung_meta = {"lung_ok": False}
            lung1 = np.zeros((vol.shape[0], ) + resize_to_1mm_3d(vol[:1], spacing).shape[1:], np.uint8)
        else:
            n_nod = int((msk > 0).sum())
            lung_meta = {
                "lung_ok": True,
                "lung_voxels_orig": int(lung.sum()),
                # tỉ lệ voxel nốt nằm trong mask phổi: thấp nghĩa là che ảnh bằng mask phổi sẽ cắt mất nốt
                "mask_in_lung_fraction": (float(((msk > 0) & (lung > 0)).sum() / n_nod) if n_nod else None),
            }
            lung1 = resize_to_1mm_3d(lung, spacing, is_mask=True)

    img1 = resize_to_1mm_3d(vol, spacing, is_mask=False, interp=interp)      # bước 3
    msk1 = resize_to_1mm_3d(msk, spacing, is_mask=True)

    sx, sy, sz = (float(s) for s in spacing)
    meta = {
        "cache_version": CACHE_VERSION,
        "interp": interp,
        "orig_shape_zyx": list(vol.shape),
        "orig_spacing_xyz": [sx, sy, sz],
        "shape_1mm_zyx": list(img1.shape),
        "mask_voxels_orig": int((msk > 0).sum()),
        "mask_voxels_1mm": int(msk1.sum()),
        "mask_volume_mm3_orig": float((msk > 0).sum() * sx * sy * sz),
        "mask_volume_mm3_1mm": float(msk1.sum() * 1.0 * 1.0 * sz),
        **lung_meta,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_npy(img_p, img1)
    _save_npy(msk_p, msk1)
    if lung1 is not None:
        _save_npy(lung_p, lung1)
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return "ok"


def load_cached(cache_dir, mmap=True):
    """Trả về (image_1mm, mask_1mm, meta). Với mmap=True chỉ đọc phần dữ liệu thật sự được truy cập."""
    img_p, msk_p, meta_p = cache_paths(cache_dir)
    mode = "r" if mmap else None
    img = np.load(img_p, mmap_mode=mode)
    msk = np.load(msk_p, mmap_mode=mode)
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return img, msk, meta


def load_cached_slices(cache_dir, z_indices, size=IMAGE_SIZE, with_lung=False):
    """
    Nạp các lát z_indices đã pad về (K, size, size): ảnh int16 HU (pad -1000), mask uint8 (pad 0),
    và (nếu with_lung) mask phổi uint8. Đầu ra là HU thô -> áp lung_keep_mask/apply_lung_mask nếu cần,
    chuyển lên GPU rồi gọi window_hu_torch.
    """
    img, msk, _ = load_cached(cache_dir, mmap=True)
    z = np.asarray(z_indices)
    out = [pad_or_crop_xy(np.asarray(img[z]), size, HU_FLOOR),
           pad_or_crop_xy(np.asarray(msk[z]), size, 0)]
    if with_lung:
        lung = np.load(lung_cache_path(cache_dir), mmap_mode="r")
        out.append(pad_or_crop_xy(np.asarray(lung[z]), size, 0))
    return tuple(out)


# ======================================================================
# 6. CLI
# ======================================================================
def _worker(args):
    pid, nrrd_dir, cache_dir, interp, overwrite, extract_lungs = args
    cv2.setNumThreads(1)                       # tránh tranh chấp luồng khi chạy nhiều tiến trình
    try:
        return pid, build_patient_cache(Path(nrrd_dir) / pid, Path(cache_dir) / pid, interp, overwrite, extract_lungs)
    except Exception as e:                     # một ca lỗi không làm dừng cả lượt chạy
        return pid, f"error: {e}"


def _cmd_build(a):
    nrrd_dir, cache_dir = Path(a.nrrd_dir), Path(a.cache_dir)
    pids = sorted(p.name for p in nrrd_dir.iterdir() if p.is_dir())
    if a.limit:
        pids = pids[:a.limit]
    print(f"{len(pids)} bệnh nhân -> {cache_dir}" + (" (có tách phổi, chậm hơn nhiều)" if a.extract_lungs else ""))
    jobs = [(p, str(nrrd_dir), str(cache_dir), a.interp, a.overwrite, a.extract_lungs) for p in pids]

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x                # noqa: E731

    results = []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in tqdm(ex.map(_worker, jobs), total=len(jobs)):
                results.append(r)
    else:
        for j in tqdm(jobs):
            results.append(_worker(j))

    bad = [(p, s) for p, s in results if s not in ("ok", "skip")]
    print(f"Xong: {sum(s == 'ok' for _, s in results)} mới, {sum(s == 'skip' for _, s in results)} bỏ qua, {len(bad)} lỗi")
    for p, s in bad:
        print("  ", p, s)


def _phantom_ct(z=14):
    """CT tổng hợp: thân người, hai phổi, khí quản, cột sống (xương), bàn chụp, không khí ngoài khung."""
    yy, xx = np.mgrid[:512, :512]
    vol = np.full((z, 512, 512), -1000, np.int16)
    vol[:, ((yy - 256) / 170.0) ** 2 + ((xx - 256) / 225.0) ** 2 <= 1] = 40          # thân
    lung_l = ((yy - 250) / 110.0) ** 2 + ((xx - 140) / 60.0) ** 2 <= 1
    lung_r = ((yy - 250) / 110.0) ** 2 + ((xx - 372) / 60.0) ** 2 <= 1
    vol[:, lung_l | lung_r] = -800                                                    # phổi
    vol[:, (yy - 230) ** 2 + (xx - 256) ** 2 <= 100] = -950                           # khí quản
    vol[:, (yy - 340) ** 2 + (xx - 256) ** 2 <= 18 ** 2] = 300                        # cột sống
    vol[:, 432:438, 110:400] = 60                                                     # bàn chụp
    return vol, (lung_l | lung_r)


def _cmd_selftest(_):
    rng = np.random.default_rng(0)
    spacing = (0.7, 0.7, 2.5)
    vol = rng.integers(-1024, 1500, size=(6, 512, 512)).astype(np.int16)
    vol[:, :60, :] = -2000                                           # mô phỏng vùng ngoài khung quét

    r = resize_and_pad_3d(vol, spacing)
    assert r.shape == (6, 512, 512), r.shape
    assert (r[:, :5, :] == HU_FLOOR).all() and (r[:, :, :5] == HU_FLOOR).all(), "pad phải là -1000 HU"

    n = clip_and_window_hu(r)
    assert n.dtype == np.float32 and n.min() == 0.0 and n.max() <= 1.0, (n.min(), n.max())
    assert (n[:, :5, :] == 0).all(), "viền pad phải bằng không khí (0) sau cửa sổ"
    assert len(np.unique(n)) <= 47, len(np.unique(n))
    assert vol.min() == -2000, "không được sửa mảng đầu vào"

    m = np.zeros(vol.shape, np.uint8)
    m[3, 100:110, 100:110] = 1
    mr = resize_and_pad_3d(m, spacing, is_mask=True)
    assert set(np.unique(mr)) <= {0, 1} and mr.sum() > 0 and mr.shape == (6, 512, 512)

    big = resize_and_pad_3d(vol, (1.5, 1.5, 2.5))                    # 768 mm > 512 -> crop
    assert big.shape == (6, 512, 512)
    print("resize/pad/cửa sổ: OK")

    if torch is not None:
        t = window_hu_torch(torch.from_numpy(r)).numpy()
        assert np.allclose(t, n, atol=1e-6), "torch và numpy phải cho cùng kết quả"
        print("torch == numpy: OK")
    else:
        print("(bỏ qua so sánh torch: chưa cài PyTorch)")

    try:
        _import_lung_deps()
    except ImportError:
        print("(bỏ qua kiểm thử tách phổi: chưa cài scikit-image/scipy)")
    else:
        ph, truth = _phantom_ct()
        lung = extract_lungs_mask(ph)
        assert lung is not None and lung.shape == ph.shape
        mid = ph.shape[0] // 2
        inter = (lung[mid] > 0) & truth
        union = (lung[mid] > 0) | truth
        iou = inter.sum() / union.sum()
        assert iou > 0.8, f"IoU phổi quá thấp: {iou:.3f}"
        assert lung[mid, 230, 256] == 0, "khí quản phải bị loại"
        assert lung[mid, 5, 5] == 0 and lung[mid, 435, 250] == 0, "không khí ngoài và bàn chụp phải bị loại"
        print(f"tách phổi trên phantom: OK (IoU = {iou:.3f})")
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="tạo cache 1 mm (chạy một lần)")
    b.add_argument("--nrrd-dir", default=str(ROOT / "data" / "duneai_raw_nrrd"))
    b.add_argument("--cache-dir", default=str(ROOT / "data" / "cache_1mm"))
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--limit", type=int, default=0, help="chỉ xử lý N ca đầu (0 = tất cả)")
    b.add_argument("--interp", choices=list(_CV2_INTERP), default=IMAGE_INTERP)
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--extract-lungs", action="store_true", help="tính và lưu thêm mask phổi (lung_1mm.npy)")
    b.set_defaults(fn=_cmd_build)

    s = sub.add_parser("selftest", help="kiểm thử bằng dữ liệu tổng hợp")
    s.set_defaults(fn=_cmd_selftest)

    args = ap.parse_args()
    args.fn(args)
