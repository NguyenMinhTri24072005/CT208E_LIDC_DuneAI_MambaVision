import shutil
import random
import csv
from pathlib import Path
import time

# 1. Cấu hình đường dẫn
base_dir = Path(r"D:\GG_1\COMPUTER SCIENCE\NAM4_HK1_(2026-2027)\CT208E\SOURCES\data")
src_dir = base_dir / "duneai_nrrd"
dst_dir = base_dir / "duneai_nrrd_mini"
csv_path = base_dir / "processed" / "cases.csv"

# Tạo thư mục đích
dst_dir.mkdir(parents=True, exist_ok=True)

def create_mini_subset():
    print("Đang phân tích tệp cases.csv để lấy tỷ lệ Train/Val/Test...")
    splits = {'train': [], 'val': [], 'test': []}
    
    # 2. Đọc tệp chỉ mục để phân loại bệnh nhân
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['patient_id']
            split = row['split']
            
            # Chỉ lấy những ca đã có file nrrd trong đợt chuyển đổi vừa rồi
            if (src_dir / pid).exists():
                if split in splits:
                    splits[split].append(pid)
                    
    # 3. Trích xuất ngẫu nhiên 200 ca theo tỷ lệ 70-15-15
    random.seed(42) # Cố định seed để kết quả không đổi nếu chạy lại
    selected_train = random.sample(splits['train'], min(140, len(splits['train'])))
    selected_val = random.sample(splits['val'], min(30, len(splits['val'])))
    selected_test = random.sample(splits['test'], min(30, len(splits['test'])))
    
    selected_all = selected_train + selected_val + selected_test
    total_selected = len(selected_all)
    
    print(f"\nBắt đầu copy {total_selected} ca bệnh nhân sang tập mini...")
    print(f"- Train: {len(selected_train)} ca")
    print(f"- Validation: {len(selected_val)} ca")
    print(f"- Test: {len(selected_test)} ca")
    
    start_time = time.time()
    
    # 4. Thực hiện copy thư mục
    count = 0
    for i, pid in enumerate(selected_all, 1):
        src_patient = src_dir / pid
        dst_patient = dst_dir / pid
        
        # Copy toàn bộ thư mục con (chứa image.nrrd và mask.nrrd)
        if not dst_patient.exists():
            shutil.copytree(src_patient, dst_patient)
            count += 1
            
        if i % 10 == 0 or i == total_selected:
            print(f"Đã copy {i}/{total_selected} ca...")
            
    print(f"\nHoàn tất sao chép {count} ca mới trong {time.time() - start_time:.2f} giây!")
    print(f"Dung lượng dự kiến của tập mini: ~40GB.")
    print(f"Dữ liệu sẵn sàng tại: {dst_dir}")

if __name__ == "__main__":
    create_mini_subset()