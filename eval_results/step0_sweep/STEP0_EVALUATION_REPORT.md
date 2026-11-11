# 📋 BÁO CÁO KẾT QUẢ ĐO LƯỜNG BƯỚC 0 (STEP 0 MEASUREMENT REPORT)
**Mô hình đánh giá**: `best_model.pth`
**Thư mục checkpoint**: `/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/mamba_unet/runs/mamba_colab_500item_25epoch_extra_test/best_model.pth`
**Thời gian thực hiện**: 2026-11-11 10:34:32
**Tập dữ liệu**: Validation set (75 bệnh nhân)

---

### 1. Kết Quả Quét Ngưỡng 3D Mức Bệnh Nhân
- **Ngưỡng tối ưu Dice 3D**: `Threshold = 0.6`
- **Hiệu năng tại ngưỡng tối ưu**:
  - **Dice 3D**: `0.3771`
  - **Precision 3D**: `0.4143`
  - **Recall 3D**: `0.4021`
  - **Dương tính giả trung bình**: `2.91 FP/ca`
- **So sánh với ngưỡng mặc định 0.5**:
  - Dice 3D tại 0.5: `0.3767785158182296`

---

### 2. Độ Nhạy Theo Kích Thước Nốt U
| Dải kích thước          |   Số nốt GT |   Th=0.3 |   Th=0.4 |   Th=0.5 |   Th=0.6 |   Th=0.7 |
|:------------------------|------------:|---------:|---------:|---------:|---------:|---------:|
| 0 - 5 mm (Rất nhỏ)      |          59 |    0.034 |    0.034 |    0.034 |    0.034 |    0.034 |
| 5 - 10 mm (Nhỏ)         |          99 |    0.576 |    0.576 |    0.576 |    0.566 |    0.556 |
| 10 - 20 mm (Trung bình) |          37 |    0.838 |    0.838 |    0.838 |    0.838 |    0.838 |
| >= 20 mm (Lớn)          |           8 |    1     |    1     |    1     |    1     |    1     |

---

### 3. Khuyến Nghị Hành Động (Action Plan) Cho Các Bước Tiếp Theo
1. **Dương tính giả**: ⚠️ fp_comps > 1/ca, cần giảm pos_weight và tăng lát âm tính (neg_ratio >= 0.5)
2. **Nốt nhỏ (< 5mm)**: Khuyến nghị hạ ngưỡng diện tích train xuống 0 mm² lúc dựng chỉ mục để bảo tồn toàn vẹn nốt u nhỏ trong quá trình học.
3. **Kích thước ảnh**: Giữ `IMAGE_SIZE = 512` hoặc cân nhắc `448` nếu không có ca bệnh nào bị cắt mất nhãn trong kiểm tra FOV.

---
*Báo cáo được tự động sinh bởi Colab Step 0 Measurement Notebook.*
