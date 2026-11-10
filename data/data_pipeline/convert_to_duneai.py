import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import time

# Cấu hình đường dẫn (Điều chỉnh lại cho khớp với ổ đĩa thực tế của Trí)
src_dir = Path(r"D:\GG_1\COMPUTER SCIENCE\NAM4_HK1_(2026-2027)\CT208E\SOURCES\data\processed\volumes")
dst_dir = Path(r"D:\GG_1\COMPUTER SCIENCE\NAM4_HK1_(2026-2027)\CT208E\SOURCES\data\duneai_gzip_nrrd")
dst_dir.mkdir(parents=True, exist_ok=True)

# Thông số hình học từ báo cáo: (dx=0.8, dy=0.8, dz=1.5)
target_spacing = (0.8, 0.8, 1.5) 

def convert_npy_to_nrrd(patient_id):
    # --- THÊM CƠ CHẾ TIẾP TỤC (RESUME) ---
    patient_dir = dst_dir / patient_id
    out_img = patient_dir / "image.nrrd"
    out_mask = patient_dir / "mask.nrrd"
    
    # Nếu cả 2 file đích đã tồn tại thì bỏ qua, không làm lại
    if out_img.exists() and out_mask.exists():
        return False  
    # ------------------------------------

    img_path = src_dir / f"{patient_id}_img.npy"
    vote_path = src_dir / f"{patient_id}_votes.npy"
    
    if not img_path.exists() or not vote_path.exists():
        return False
        
    # Đọc dữ liệu và ép kiểu ảnh sang float32 cho SimpleITK
    img_arr = np.load(img_path).astype(np.float32)
    vote_arr = np.load(vote_path)
    
    # Binarize mask: Lấy ngưỡng >= 2 bác sĩ đồng thuận
    mask_arr = (vote_arr >= 2).astype(np.uint8)
    
    # Khởi tạo đối tượng ảnh y tế
    img_sitk = sitk.GetImageFromArray(img_arr)
    mask_sitk = sitk.GetImageFromArray(mask_arr)
    
    # Ép chuẩn không gian
    img_sitk.SetSpacing(target_spacing)
    mask_sitk.SetSpacing(target_spacing)
    
    # Lưu xuất file
    patient_dir.mkdir(exist_ok=True)
    sitk.WriteImage(img_sitk, str(out_img))
    sitk.WriteImage(mask_sitk, str(out_mask))
    
    return True

print("Bắt đầu tiến trình chuyển đổi sang chuẩn DuneAI...")
start_time = time.time()

# Lọc các file đúng định dạng
patient_files = list(src_dir.glob("*_img.npy"))
patient_ids = [f.stem.replace("_img", "") for f in patient_files]
total = len(patient_ids)

count = 0
for i, pid in enumerate(patient_ids, 1):
    if convert_npy_to_nrrd(pid):
        count += 1
    if i % 50 == 0:
        print(f"Đã xử lý: {i}/{total} ca...")

print(f"Hoàn tất chuyển đổi {count} ca bệnh nhân trong {time.time() - start_time:.2f} giây.")
print(f"Dữ liệu sẵn sàng tại: {dst_dir}")