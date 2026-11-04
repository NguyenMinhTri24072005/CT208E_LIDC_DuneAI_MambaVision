import pandas as pd
import os
import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    csv_path = "processed/slices.csv"
    if not os.path.exists(csv_path):
        print(f"LỖI: Không tìm thấy file {csv_path}. Bạn đã chạy tiền xử lý chưa?")
        return

    df = pd.read_csv(csv_path)

    print("="*60)
    print(" BÁO CÁO KHẢO SÁT DỮ LIỆU MẶT CẮT 2D (SLICES.CSV)")
    print("="*60)

    total_slices = len(df)
    total_patients = df['patient_id'].nunique()
    
    # Tính tổng số lát cắt dương tính và âm tính
    positive = df['has_nodule'].sum()
    negative = total_slices - positive

    print(f"Tổng số bệnh nhân hiện có : {total_patients}")
    print(f"Tổng số lát cắt 2D (Slices): {total_slices:,}")
    print(f"Trung bình mỗi bệnh nhân   : {total_slices // total_patients} lát cắt")
    print("-" * 40)
    print(f"Số lát CÓ khối u (Positive)   : {positive:,} ({positive/total_slices*100:.2f}%)")
    print(f"Số lát KHÔNG khối u (Negative): {negative:,} ({negative/total_slices*100:.2f}%)")
    
    if positive > 0:
        ratio = negative / positive
        print(f"Mức độ mất cân bằng: Lát KHÔNG u nhiều gấp {ratio:.1f} lần lát CÓ u!")
    
    print("\n" + "="*60)
    print(" PHÂN BỐ THEO TẬP DỮ LIỆU (TRAIN / VAL / TEST)")
    print("="*60)
    
    # Thống kê theo từng tập
    for split in ['train', 'val', 'test']:
        sub_df = df[df['split'] == split]
        if len(sub_df) == 0: 
            continue
        
        s_pos = sub_df['has_nodule'].sum()
        s_neg = len(sub_df) - s_pos
        print(f"Tập {split.upper():<5} | Tổng: {len(sub_df):<6,} lát")
        print(f"          | CÓ u: {s_pos:<6,} | KHÔNG u: {s_neg:<6,}")
    
    print("="*60)
    print("KẾT LUẬN TỪ DATA:")
    print("Tỷ lệ chênh lệch quá lớn. Khi viết PyTorch Dataloader, ta BẮT BUỘC phải dùng kỹ thuật")
    print("Weighted Sampling hoặc bốc tỷ lệ 50-50 để mạng không bị thiên vị (bias)!")
    print("="*60)

if __name__ == "__main__":
    main()
