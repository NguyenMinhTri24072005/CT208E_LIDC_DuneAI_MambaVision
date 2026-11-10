# DuneAI: Hệ Thống Phân Đoạn Nốt Phổi 2.5D (LIDC-IDRI)

Dự án nghiên cứu và phát triển mô hình học sâu phân đoạn khối u/nốt phổi trên ảnh chụp cắt lớp vi tính lồng ngực (CT LIDC-IDRI). Hệ thống hỗ trợ xử lý dữ liệu chuẩn 2.5D (ghép 3 lát cắt liền kề) và cung cấp hai kiến trúc mô hình hoàn toàn độc lập, song song:

1. **`unet_base/`**: Kiến trúc U-Net 5 tầng kinh điển (Classic CNN) dựa trên các khối DoubleConv tiêu chuẩn.
2. **`mamba_unet/`**: Kiến trúc lai hiện đại kết hợp Backbone thị giác trạng thái không gian **MambaVision** (`mamba_vision_T`) và Decoder tích chập đa tầng.

Dự án đã được tái cấu trúc triệt để theo tài liệu chuẩn **`docs/REFACTOR_SPEC.md` (v2.1)**, loại bỏ hoàn toàn mã legacy, đảm bảo tính độc lập cấu trúc tuyệt đối (0 cross-import giữa hai kiến trúc, không lớp `nn.Module` nào trong `common/`, bảo toàn 100% `state_dict` so với golden snapshots).

---

## 1. Cấu Trúc Thư Mục Dự Án

```text
SOURCES/
├── common/                         # Logic chung tái sử dụng (0 nn.Module, 0 import kiến trúc)
│   ├── data_pipeline/              # medical_prep, offline_builder, dataset, runtime_data
│   ├── engine/                     # base_trainer, run_manager, checkpoint_io, cli_args, utils
│   ├── eval3d/                     # postprocess3d, metrics3d, volume_inference, nrrd_io, eval_runner
│   ├── metrics.py                  # compute_segmentation_metrics (Dice, IoU 2D/2.5D)
│   └── paths.py                    # Quản lý đường dẫn tập trung không hard-code
│
├── configs/                        # Cấu hình siêu tham số mặc định toàn dự án
│   └── config.py
│
├── unet_base/                      # Kiến trúc U-Net CNN 5-Level Độc Lập
│   ├── activations.py              # Registry hàm kích hoạt riêng (ReLU, LeakyReLU, GELU, ...)
│   ├── losses.py                   # Registry hàm mất mát riêng (BCE, Dice, Focal, Combo, ...)
│   ├── blocks.py                   # DoubleConv, OutConv, Down, Up
│   ├── encoder.py / decoder.py     # Encoder và Decoder chuyên biệt
│   ├── model.py                    # UNetBase model & build_unet_base factory
│   ├── trainer.py                  # UNetBaseTrainer
│   ├── train_main.py               # CLI huấn luyện (train | overfit | selftest)
│   ├── evaluate.py                 # CLI đánh giá 3D (eval | predict | selftest)
│   ├── configs/default.yaml        # Cấu hình huấn luyện mặc định
│   ├── notebooks/                  # Jupyter notebooks cho Colab, Kaggle, Local
│   └── runs/                       # Thư mục chứa logs và checkpoints theo từng lượt chạy
│
├── mamba_unet/                     # Kiến trúc Mamba-UNet Độc Lập
│   ├── fallbacks.py                # selective_scan (CUDA mamba-ssm kèm PyTorch CPU fallback)
│   ├── backbone/                   # MambaVision backbone tách lớp sạch sẽ (không timm/einops)
│   ├── activations.py              # Registry kích hoạt Decoder riêng
│   ├── losses.py                   # Registry loss riêng
│   ├── blocks.py                   # Decoder CNN blocks riêng
│   ├── decoder.py / model.py       # MambaUNet model & build_mamba_unet factory
│   ├── trainer.py                  # MambaUNetTrainer (Lịch đóng băng backbone, BN eval, 2 LR groups)
│   ├── train_main.py               # CLI huấn luyện (train | overfit | selftest)
│   ├── evaluate.py                 # CLI đánh giá 3D (eval | predict | selftest)
│   ├── configs/default.yaml        # Cấu hình huấn luyện mặc định
│   ├── notebooks/                  # Jupyter notebooks cho Colab, Kaggle, Local
│   └── runs/                       # Thư mục run của Mamba-UNet
│
├── checkpoints/                    # Lưu trữ các file trọng số mô hình đã huấn luyện
├── scripts/                        # Công cụ hỗ trợ môi trường, chuẩn bị dữ liệu, sinh notebook
│   ├── prepare_dataset.py          # Hỗ trợ chuẩn bị dataset
│   ├── inspect_checkpoint.py       # Kiểm tra nội dung checkpoint
│   ├── make_notebooks.py           # Bộ sinh tự động 6 notebook chuẩn nbformat v4
│   └── export_golden.py            # Trích xuất và kiểm tra golden fixtures
│
├── tests/                          # Toàn bộ kiểm thử nghiệm thu tự động
│   ├── run_all.py                  # Script chạy toàn bộ 26 bài test nghiệm thu
│   ├── golden/                     # Golden state_dict keys và tensor outputs
│   └── test_*.py                   # 9 test modules chuyên sâu
│
└── docs/                           # Tài liệu thiết kế và đặc tả kỹ thuật
    ├── REFACTOR_SPEC.md            # Đặc tả tái cấu trúc v2.1
    ├── INVENTORY.md                # Kiểm kê toàn bộ hàm/lớp mã nguồn
    └── DEVIATIONS.md               # Bảng ghi nhận sai lệch & điều chỉnh kỹ thuật
```

---

## 2. Hướng Dẫn Sử Dụng Nhanh

### 2.1. Huấn Luyện Mạng Cục Bộ (Local)

**Huấn luyện U-Net kinh điển:**
```bash
python -m unet_base.train_main train \
    --cache-dir data/cache_1mm \
    --features 64 \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --activation relu \
    --loss combo
```

**Huấn luyện Mamba-UNet:**
```bash
python -m mamba_unet.train_main train \
    --cache-dir data/cache_1mm \
    --backbone mamba_vision_T \
    --pretrained \
    --freeze-backbone-epochs 5 \
    --backbone-lr-mult 0.1 \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --activation relu \
    --loss combo
```

### 2.2. Đánh Giá 3D Cấp Độ Bệnh Nhân (Patient-Level 3D Evaluation)

```bash
# Đánh giá UNetBase
python -m unet_base.evaluate eval \
    --cache-dir data/cache_1mm \
    --checkpoint checkpoints/unet_base_25d/best_model.pth \
    --split val

# Đánh giá Mamba-UNet
python -m mamba_unet.evaluate eval \
    --cache-dir data/cache_1mm \
    --checkpoint checkpoints/mamba_unet_baseline/mamba_unet_baseline_final_weights.pth \
    --split val
```

---

## 3. Huấn Luyện Trên Google Colab & Kaggle

Các notebook mẫu chuẩn hoá đã được sinh tự động sẵn trong thư mục `<arch>/notebooks/`:
- **Google Colab (`colab_train.ipynb`)**: Tự động mount Google Drive, nhận diện file zip `cache_1mm.zip`, giải nén phẳng có chọn lọc số lượng bệnh nhân ra SSD `/content` cực nhanh qua `runtime_data.py`, huấn luyện và sao lưu tự động về Drive.
- **Kaggle (`kaggle_train.ipynb`)**: Tự động dò tìm dataset (kể cả lồng thư mục dạng `cache_1mm/cache_1mm/LIDC-IDRI-XXXX`), thiết lập symlink view-dir ghi được trong `/kaggle/working`, thực thi huấn luyện và đánh giá mượt mà.

Để tạo lại hoặc kiểm tra tính hợp lệ toàn bộ 6 notebook:
```bash
python scripts/make_notebooks.py --check
```

---

## 4. Chạy Toàn Bộ Kiểm Thử Nghiệm Thu (Acceptance Testing)

Dự án cung cấp bộ kiểm thử nghiệm thu toàn diện tự động gồm 26 bài test:
```bash
python tests/run_all.py
```
Bộ kiểm thử sẽ kiểm tra tự động:
- 100% tính nguyên vẹn trọng số `state_dict` và tensor output với golden fixtures.
- 0 cross-import giữa `unet_base` và `mamba_unet`.
- `common/` có đúng 0 lớp con `nn.Module` và không import hai kiến trúc.
- Quy trình tự động resume và quản lý thư mục thực nghiệm `run_config.json`.
- Tính hợp lệ của toàn bộ notebook và tính sạch sẽ của cấu trúc mã nguồn.
