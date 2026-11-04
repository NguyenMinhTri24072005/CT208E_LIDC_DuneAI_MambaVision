# Dự Án U-Net Baseline Cho Phân Đoạn Ảnh Y Tế

Triển khai mạng **U-Net chuẩn (Ronneberger et al., 2015)** bằng PyTorch, tối ưu cho bài toán phân đoạn ảnh (Semantic Segmentation). Thiết kế dạng module độc lập, có thể tái lập kết quả (reproducibility), chạy mượt mà trên Google Colab (GPU T4/L4) hoặc máy trạm cá nhân, và dễ dàng mở rộng sang các kiến trúc backbone khác.

---

## 📁 Cấu Trúc Dự Án

Toàn bộ dự án nằm gọn bên trong thư mục `SOURCES/unet_baseline/`:

```text
SOURCES/unet_baseline/
├── train_on_colab.ipynb        # Notebook điều khiển chạy trên Colab
├── docs/                       # Tài liệu plan & spec chi tiết
│   ├── implementation_plan.md
│   └── spec.md
├── data/                       # Dữ liệu phân tách theo train / val / test
│   ├── train/{images,masks}
│   ├── val/{images,masks}
│   └── test/{images,masks}
├── models/
│   ├── __init__.py
│   ├── unet_model.py           # Class UNet và factory build_model
│   └── unet_parts.py           # DoubleConv, Down, Up, OutConv
├── utils/
│   ├── __init__.py
│   ├── dataset.py              # SegmentationDataset & get_dataloaders
│   ├── losses.py               # DiceLoss, BCE, FocalLoss, CombinedLoss
│   ├── metrics.py              # IoU, Dice Coefficient, HD95, MetricTracker
│   └── transforms.py           # Data Augmentation với Albumentations
├── checkpoints/                # Lưu weights (last.pth, best.pth) và history.csv
├── configs/
│   └── config.yaml             # Quản lý toàn bộ Hyperparameters
├── train.py                    # Vòng lặp huấn luyện chính
├── evaluate.py                 # Đánh giá trên tập test và xuất báo cáo CSV/JSON
├── predict.py                  # Dự đoán trên ảnh mới (xuất mask & overlay)
├── requirements.txt            # Danh sách thư viện cố định phiên bản
└── README.md                   # Hướng dẫn sử dụng
```

---

## 🛠️ Cài Đặt Môi Trường

Mọi lệnh chạy đều phải được thực thi từ thư mục gốc của dự án: `SOURCES/unet_baseline/`.

```bash
cd "SOURCES/unet_baseline"
pip install -r requirements.txt
```

---

## 📂 Chuẩn Bị Dữ Liệu

Tổ chức các file ảnh và mask theo cấu trúc thư mục sau:

```text
data/
├── train/
│   ├── images/  (sample_01.png, sample_02.png, ...)
│   └── masks/   (sample_01.png, sample_02.png, ...)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

> **Lưu ý:**
> - Code tự động ghép cặp ảnh và mask theo **tên file (stem)** (ví dụ: `patient_01.png` sẽ ghép với `patient_01.png` trong thư mục masks).
> - Hỗ trợ các định dạng ảnh: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.npy`.
> - Đối với bài toán nhị phân, mask chứa giá trị `0` (nền) và `1` hoặc `255` (khối u).

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Huấn luyện mô hình (Training)

Chạy huấn luyện với file cấu hình mặc định:
```bash
python train.py --config configs/config.yaml
```

Ghi đè (override) siêu tham số trực tiếp qua dòng lệnh:
```bash
python train.py --config configs/config.yaml --train.batch_size=8 --train.lr=0.0005 --train.epochs=100
```

Tiếp tục huấn luyện từ checkpoint gần nhất (Resume):
```bash
python train.py --config configs/config.yaml --checkpoint.resume_path=checkpoints/last.pth
```

### 2. Đánh giá trên tập Test (Evaluation)

Chạy kiểm thử trên tập test, tính toán các chỉ số `Dice`, `IoU`, `HD95` cho từng ảnh và xuất file `test_metrics.csv`:
```bash
python evaluate.py --checkpoint checkpoints/best.pth --config configs/config.yaml --output_dir evaluation_results --save_overlay
```

### 3. Dự đoán trên ảnh mới (Inference / Prediction)

Chạy dự đoán trên một thư mục ảnh:
```bash
python predict.py --input data/test/images --output predictions --checkpoint checkpoints/best.pth --threshold 0.5
```

Hoặc chạy trên 1 ảnh đơn lẻ:
```bash
python predict.py --input data/test/images/sample_01.png --output predictions --checkpoint checkpoints/best.pth
```
Kết quả sẽ tạo ra file mask nhị phân (`sample_01_mask.png`) và file trực quan hóa chồng ảnh (`sample_01_overlay.png`).

---

## ☁️ Chạy Trên Google Colab

1. Tải toàn bộ thư mục `unet_baseline` lên Google Drive của bạn.
2. Mở file `train_on_colab.ipynb` bằng Google Colab.
3. Chỉnh sửa đường dẫn `WORKSPACE_DIR` ở Cell số 3 để trỏ đúng vào thư mục của bạn trên Drive.
4. Chọn Runtime -> Change runtime type -> Chọn **GPU (T4 hoặc L4)**.
5. Bấm **Run All** để thực hiện toàn bộ luồng huấn luyện, đánh giá và trực quan hóa.
