import SimpleITK as sitk
import numpy as np
import os
import pydicom
from pathlib import Path

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG
# ==========================================
PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR     = PIPELINE_DIR.parent
RAW_DIR      = DATA_DIR / "raw"
OUT_DIR      = DATA_DIR / "duneai_raw_nrrd"


def find_ct_dicom_series(patient_raw_dir: Path):
    reader = sitk.ImageSeriesReader()
    best_series_files = []
    for root, _, files in os.walk(patient_raw_dir):
        if any(f.lower().endswith(".dcm") for f in files):
            for sid in reader.GetGDCMSeriesIDs(root):
                dicom_names = reader.GetGDCMSeriesFileNames(root, sid)
                if len(dicom_names) > len(best_series_files):
                    best_series_files = dicom_names
    return best_series_files


def find_seg_files(patient_raw_dir: Path):
    """Tìm mọi file DICOM-SEG (Modality == 'SEG') trong thư mục bệnh nhân.
    Mỗi thư mục chỉ cần kiểm tra file .dcm đầu tiên vì cả series cùng modality."""
    seg_files = []
    for root, _, files in os.walk(patient_raw_dir):
        dcms = sorted(f for f in files if f.lower().endswith(".dcm"))
        if not dcms:
            continue
        first = os.path.join(root, dcms[0])
        try:
            ds = pydicom.dcmread(first, stop_before_pixels=True, specific_tags=["Modality"])
        except Exception:
            continue
        if getattr(ds, "Modality", "") == "SEG":
            seg_files.extend(os.path.join(root, f) for f in dcms)
    return seg_files


def build_union_mask(image_3d: sitk.Image, dicom_files, seg_files):
    """Gộp mask của TẤT CẢ file SEG bằng phép OR (có ít nhất 1 bác sĩ vẽ là tính).
    Trả về (sitk.Image uint8 cùng geometry với CT, số frame bỏ qua)."""
    z_size, y_size, x_size = sitk.GetArrayViewFromImage(image_3d).shape

    # SOPInstanceUID -> chỉ số lát z (đúng thứ tự mà SimpleITK xếp khi đọc series)
    uid_to_z = {}
    for z, f in enumerate(dicom_files):
        ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["SOPInstanceUID"])
        uid_to_z[str(ds.SOPInstanceUID)] = z

    mask = np.zeros((z_size, y_size, x_size), dtype=np.uint8)
    skipped = 0

    for seg_path in seg_files:
        ds = pydicom.dcmread(seg_path)
        px = ds.pixel_array
        if px.ndim == 2:                      # SEG chỉ có 1 frame
            px = px[None]

        # Nhị phân hóa (FRACTIONAL: lấy >= 50% giá trị tối đa)
        if str(getattr(ds, "SegmentationType", "BINARY")) == "FRACTIONAL":
            max_val = float(getattr(ds, "MaximumFractionalValue", 255))
            frames_bin = px >= 0.5 * max_val
        else:
            frames_bin = px > 0

        if frames_bin.shape[1:] != (y_size, x_size):
            print(f"   ⚠️ Bỏ qua {Path(seg_path).name}: kích thước {frames_bin.shape[1:]} khác CT {(y_size, x_size)}")
            skipped += frames_bin.shape[0]
            continue

        per_frame = ds.PerFrameFunctionalGroupsSequence
        for i, fg in enumerate(per_frame):
            z = None
            # Cách 1 (chuẩn): tham chiếu SOPInstanceUID của lát CT nguồn
            try:
                src = fg.DerivationImageSequence[0].SourceImageSequence[0]
                z = uid_to_z.get(str(src.ReferencedSOPInstanceUID))
            except (AttributeError, IndexError):
                pass
            # Cách 2 (dự phòng): khớp theo ImagePositionPatient
            if z is None:
                try:
                    ipp = [float(v) for v in fg.PlanePositionSequence[0].ImagePositionPatient]
                    z = image_3d.TransformPhysicalPointToIndex(ipp)[2]
                    if not (0 <= z < z_size):
                        z = None
                except (AttributeError, IndexError, RuntimeError):
                    pass
            if z is None:
                skipped += 1
                continue
            mask[z] |= frames_bin[i].astype(np.uint8)

    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(image_3d)
    return mask_img, skipped


def convert_patient_dicom_to_nrrd(patient_id: str):
    patient_raw_dir = RAW_DIR / patient_id
    if not patient_raw_dir.exists():
        print(f"❌ Không tìm thấy thư mục: {patient_raw_dir}")
        return False

    out_dir = OUT_DIR / patient_id
    out_img_path  = out_dir / "image.nrrd"
    out_mask_path = out_dir / "mask.nrrd"

    # Resume: xong cả ảnh lẫn mask thì bỏ qua
    if out_img_path.exists() and out_mask_path.exists():
        print(f"⏭️ Bỏ qua {patient_id} (Đã có image + mask).")
        return True

    print(f"🔍 Quét chuỗi DICOM của {patient_id}...")
    dicom_files = find_ct_dicom_series(patient_raw_dir)
    if len(dicom_files) < 10:
        print(f"❌ Không tìm thấy CT hợp lệ cho {patient_id}.")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    image_3d = reader.Execute()

    if not out_img_path.exists():
        sitk.WriteImage(image_3d, str(out_img_path), useCompression=True)
    arr_hu = sitk.GetArrayFromImage(image_3d)
    print(f"✅ Image {patient_id} | Shape: {arr_hu.shape} | HU: [{arr_hu.min()}, {arr_hu.max()}]")

    # ---- MASK: hợp của tất cả SEG ----
    seg_files = find_seg_files(patient_raw_dir)
    mask_img, skipped = build_union_mask(image_3d, dicom_files, seg_files)
    sitk.WriteImage(mask_img, str(out_mask_path), useCompression=True)

    n_vox = int(sitk.GetArrayViewFromImage(mask_img).sum())
    sp = image_3d.GetSpacing()
    vol_mm3 = n_vox * sp[0] * sp[1] * sp[2]
    print(f"   🩻 Mask | {len(seg_files)} file SEG | {n_vox} voxel (~{vol_mm3:.0f} mm³)"
            + (f" | ⚠️ bỏ qua {skipped} frame" if skipped else ""))
    if not seg_files or n_vox == 0:
        print(f"   ℹ️ {patient_id} không có nốt nào được ghi nhận (mask rỗng).")
    return True


if __name__ == "__main__":
    NUM_SAMPLES_TO_TEST = None   # None = xử lý toàn bộ

    if not RAW_DIR.exists():
        print(f"Lỗi: Không tìm thấy thư mục {RAW_DIR}")
        exit()

    all_patients = sorted([p.name for p in RAW_DIR.iterdir() if p.is_dir()])
    print(f"Tổng số bệnh nhân tìm thấy: {len(all_patients)}")

    if NUM_SAMPLES_TO_TEST is not None:
        print(f"-> CHẾ ĐỘ TEST NHANH: Chỉ xử lý {NUM_SAMPLES_TO_TEST} mẫu đầu tiên.\n")
        patients_to_process = all_patients[:NUM_SAMPLES_TO_TEST]
    else:
        print(f"-> CHẾ ĐỘ TOÀN BỘ: Xử lý tất cả {len(all_patients)} mẫu.\n")
        patients_to_process = all_patients

    success_count = 0
    for pid in patients_to_process:
        print("-" * 40)
        if convert_patient_dicom_to_nrrd(pid):
            success_count += 1

    print("=" * 40)
    print(f"🎉 Hoàn tất! Chuyển đổi thành công {success_count}/{len(patients_to_process)} mẫu.")
    print(f"Thư mục lưu trữ: {OUT_DIR}")