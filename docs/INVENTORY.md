# KIỂM KÊ MÃ NGUỒN HIỆN TẠI (INVENTORY - PHASE 0)

> **Dự án:** Phân đoạn khối u phổi 2.5D (LIDC-IDRI) — Tái cấu trúc DuneAI (v2.1)  
> **Thời điểm lập:** 2026-11-09  
> **Nhánh Git:** `refactor/split-architectures`  
> **Tài liệu tham chiếu:** `docs/REFACTOR_SPEC.md`

---

## 1. BẢNG ÁNH XẠ: FILE CŨ -> FILE MỚI THEO SPEC

| File mã nguồn cũ (`DuneAI_Remake/`) | File đích mới theo spec | Vai trò & Hướng xử lý |
|---|---|---|
| `configs/config.py` | `configs/config.py` | Chứa hằng số y khoa dùng chung (cửa sổ HU, `IMAGE_SIZE`, ngưỡng lát...) |
| `src/medical_prep.py` | `common/data_pipeline/medical_prep.py` | Tiền xử lý DICOM/NRRD sang 1mm, tách phổi, `window_hu_torch` |
| `src/offline_builder.py` | `common/data_pipeline/offline_builder.py` | Lọc lát cắt y khoa, tính diện tích u, chia train/val và xuất `slices.csv` |
| `src/dataset.py` | `common/data_pipeline/dataset.py` | `LIDC25DDataset`, `create_dataloaders` đọc dữ liệu 2.5D |
| *(Mã mới - spec 8.6)* | `common/data_pipeline/runtime_data.py` | Chuẩn bị dữ liệu theo nền tảng: Colab (giải nén chọn số lượng từ Drive), Kaggle (symlink view-dir từ `/kaggle/input`), local |
| `src/trainer.py` (vòng lặp lõi) | `common/engine/base_trainer.py` | `BaseTrainer`: AMP, DataParallel, `KeyboardInterrupt`, checkpoint atomic |
| *(Tách từ trainer.py / cli cũ)* | `common/engine/run_manager.py` | Tạo/đóng thư mục run, ghi `run_config.json`, tìm run để resume |
| *(Tách từ trainer.py / cli cũ)* | `common/engine/run_config.py` | Thu thập môi trường/git/dữ liệu và quản lý schema cấu hình run |
| *(Tách từ trainer.py / io cũ)* | `common/engine/checkpoint_io.py` | `save_atomic`, `load_checkpoint_file`, xuất `best_weights_fp16.pth` |
| *(Tách từ cli của train_*.py)* | `common/engine/cli_args.py` | `add_common_args()` chuẩn hóa tham số dòng lệnh cho 2 kiến trúc |
| *(Tách từ trainer.py)* | `common/engine/utils.py` | `seed_everything`, `format_duration`, `atomic_csv_save` |
| `src/components/losses.py` (`compute_segmentation_metrics`) | `common/metrics.py` | Metric đánh giá phân đoạn dùng chung (Dice, IoU, Precision, Recall) |
| `src/postprocess.py` (toán 3D) | `common/eval3d/postprocess3d.py` | Unpad/uncrop, resize về không gian gốc 3D, postprocess mask |
| `src/postprocess.py` (metric 3D) | `common/eval3d/metrics3d.py` | `compute_3d_metrics`, `_compute_hd95_mm` |
| `src/postprocess.py` (suy luận volume) | `common/eval3d/volume_inference.py` | `predict_patient_volume` (nhận `nn.Module` đã nạp trọng số) |
| `src/postprocess.py` (IO NRRD) | `common/eval3d/nrrd_io.py` | `save_prediction_nrrd` |
| `src/postprocess.py` (vòng lặp eval) | `common/eval3d/eval_runner.py` | `evaluate_dataset_3d` |
| `src/backbones/cnn_baseline.py` (Encoder) | `unet_base/encoder.py` | `CNNBaselineEncoder` 5 mức đặc trưng |
| `src/decoders/unet_decoder.py` (`ClassicUNetDecoder`) | `unet_base/decoder.py` | `ClassicUNetDecoder`, `UpBlock` tích hợp ConvTranspose2d |
| `src/backbones/cnn_baseline.py` (`DoubleConv`, `OutConv`) | `unet_base/blocks.py` | Bản riêng `DoubleConv`, `OutConv` cho U-Net Base |
| `src/models/unet_base.py` | `unet_base/model.py` | `UNetBase`, `build_unet_base(cfg)` |
| `src/components/activations.py` | `unet_base/activations.py` | Registry hàm kích hoạt riêng của U-Net Base |
| `src/components/losses.py` | `unet_base/losses.py` | Registry hàm mất mát riêng của U-Net Base |
| `src/trainer.py` (phần UNetBase) | `unet_base/trainer.py` | `UNetBaseTrainer(BaseTrainer)` |
| `train_main.py` | `unet_base/train_main.py` | CLI: selftest, overfit, train cho U-Net Base |
| *(Tách từ postprocess.py)* | `unet_base/evaluate.py` | CLI: selftest, eval, predict cho U-Net Base |
| `src/backbones/mamba_vision.py` (fallbacks) | `mamba_unet/fallbacks.py` | `selective_scan_fn` (CUDA / pure PyTorch fallback), `DropPath`, `LayerNorm2d` |
| `src/backbones/mamba_vision.py` (backbone) | `mamba_unet/backbone/` | Gói module hóa: `layers.py`, `mixer.py`, `blocks.py`, `mambavision.py`, `variants.py`, `pretrained.py` |
| `src/decoders/unet_decoder.py` (`UNetDecoder`) | `mamba_unet/decoder.py` | `UNetDecoder`, `UpBlock` 4 tầng với Bilinear Upsample |
| `src/decoders/unet_decoder.py` (`DoubleConv`, `OutConv`) | `mamba_unet/blocks.py` | Bản riêng `DoubleConv`, `OutConv` cho decoder Mamba-UNet |
| `src/models/mamba_unet.py` | `mamba_unet/model.py` | `MambaUNet`, `build_mamba_unet(cfg)` |
| `src/components/activations.py` | `mamba_unet/activations.py` | Registry hàm kích hoạt riêng (chỉ áp dụng cho decoder) |
| `src/components/losses.py` | `mamba_unet/losses.py` | Registry hàm mất mát riêng của Mamba-UNet |
| `src/trainer.py` (phần Mamba) | `mamba_unet/trainer.py` | `MambaUNetTrainer(BaseTrainer)` với lịch đóng băng và 2 nhóm lr |
| `train_mamba.py` | `mamba_unet/train_main.py` | CLI: selftest, overfit, train cho Mamba-UNet |
| *(Tách từ postprocess.py)* | `mamba_unet/evaluate.py` | CLI: selftest, eval, predict cho Mamba-UNet |

---

## 2. CÁC HÀM VÀ LỚP CÔNG KHAI CỦA MỖI FILE

### 2.1 `DuneAI_Remake/configs/config.py`
- **Hằng số y tế:** `IMAGE_SIZE` (512), `HU_WINDOW` ([-1000, 400]), `TUMOR_SIZE_LOW_LIM` (4.0), `ZERO_SL_RATIO` (2.0), `VAL_SPLIT` (0.15), `SEED` (42), `HU_STEP` (25), `BATCH_SIZE` (8), `EPOCHS` (50), `LR` (1e-4), `DEVICE`, `NUM_WORKERS`.

### 2.2 `DuneAI_Remake/src/medical_prep.py`
- **Lớp / Hàm:**
  - `load_cached(patient_dir, item, mmap=False)`
  - `load_cached_slices(patient_dir, z_indices, apply_window=False)`
  - `lung_cache_path(patient_dir)`
  - `clip_and_window_hu(vol, window=..., quantize=True)`
  - `window_hu_torch(tensor_int16, window_min=-1000, window_max=400, step=25)`
  - `segment_lungs_3d(image_hu, ...)`
  - `apply_lung_mask(image, mask, fill_val=-1000)`
  - `pad_or_crop_xy(vol, target_size=512)`
  - `resample_to_1mm_3d(image, original_spacing)`
  - `build_cache_single_patient(...)`, `_cmd_build(args)`, `_cmd_selftest(args)`

### 2.3 `DuneAI_Remake/src/offline_builder.py`
- **Lớp / Hàm:**
  - `build_dataset_index(cache_dir, out_csv_path=None, tumor_size_low_lim=4.0, zero_sl_ratio=2.0, val_split=0.15, seed=42, require_lung=True)`
  - `_cmd_build(args)`, `_cmd_selftest(args)`

### 2.4 `DuneAI_Remake/src/dataset.py`
- **Lớp / Hàm:**
  - `class LIDC25DDataset(Dataset)`: `__init__`, `__len__`, `__getitem__`
  - `create_dataloaders(cache_dir, batch_size=8, workers=2, ...)`
  - `_cmd_selftest(args)`, `_cmd_inspect(args)`

### 2.5 `DuneAI_Remake/src/trainer.py`
- **Lớp / Hàm:**
  - `class Trainer`: `__init__`, `fit()`, `train_epoch()`, `validate_epoch()`, `save_checkpoint()`, `load_checkpoint()`, `_unwrap_model()`
  - `_build_criterion(name, ...)`
  - `seed_everything(seed=42)`
  - `format_time(seconds)`
  - `_cmd_selftest(args)`, `_cmd_overfit(args)`

### 2.6 `DuneAI_Remake/src/components/activations.py`
- **Lớp / Hàm:**
  - `class SoftClampReLU(nn.Module)`: `forward`, `extra_repr`
  - `class SoftClampSiLU(nn.Module)`: `forward`, `extra_repr`
  - `class StarReLU(nn.Module)`: `forward`
  - `get_activation(name: str, **kwargs) -> nn.Module`

### 2.7 `DuneAI_Remake/src/components/losses.py`
- **Lớp / Hàm:**
  - `class DiceLoss(nn.Module)`: `forward`
  - `class TverskyLoss(nn.Module)`: `forward`
  - `class TverskyFocalLoss(nn.Module)`: `forward`
  - `class FocalLoss(nn.Module)`: `forward`
  - `class DiceBCELoss(nn.Module)`: `forward`
  - `compute_segmentation_metrics(logits, targets, threshold=0.5, eps=1e-6) -> dict`
  - `build_loss(name, config=None)`

### 2.8 `DuneAI_Remake/src/backbones/cnn_baseline.py`
- **Lớp / Hàm:**
  - `class DoubleConv(nn.Module)`: `forward`
  - `class Down(nn.Module)`: `forward`
  - `class CNNBaselineEncoder(nn.Module)`: `forward`
  - `cnn_baseline_encoder(in_channels=3, features=64, use_stride4_stem=False, act_type="relu", **kwargs)`

### 2.9 `DuneAI_Remake/src/backbones/mamba_vision.py`
- **Lớp / Hàm:**
  - `selective_scan_fn(...)`: Nhân chọn lọc CUDA hoặc pure PyTorch fallback
  - `selective_scan_torch(...)`: Fallback thuần PyTorch
  - `class Downsample(nn.Module)`, `class PatchEmbed(nn.Module)`, `class ConvBlock(nn.Module)`
  - `class Attention(nn.Module)`, `class MambaVisionMixer(nn.Module)`, `class Block(nn.Module)`
  - `class MambaVision(nn.Module)`: `forward` (trả về 4 feature map)
  - `mamba_vision_T`, `mamba_vision_T2`, `mamba_vision_S`, `mamba_vision_B`, `mamba_vision_L`, `mamba_vision_L2`
  - `build_mambavision_backbone(name, pretrained=False, **kwargs)`

### 2.10 `DuneAI_Remake/src/decoders/unet_decoder.py`
- **Lớp / Hàm:**
  - `class DoubleConv(nn.Module)`: `forward`
  - `class UpBlock(nn.Module)`: `forward` (hỗ trợ `ConvTranspose2d` hoặc `Upsample` bilinear)
  - `class OutConv(nn.Module)`: `forward` (khởi tạo `init_bias=-4.0`)
  - `class ClassicUNetDecoder(nn.Module)`: 4 tầng UpBlock đối xứng 5 mức đặc trưng
  - `class UNetDecoder(nn.Module)`: 4 tầng UpBlock cho MambaVision

### 2.11 `DuneAI_Remake/src/models/unet_base.py`
- **Lớp / Hàm:**
  - `class UNetBase(nn.Module)`: `encoder` (CNNBaselineEncoder), `decoder` (ClassicUNetDecoder), `forward`
  - `build_unet_base(config=None, **kwargs)`
  - Bí danh: `PureUNet = UNetBase`, `ClassicUNet = UNetBase`, `BaselineUNet = UNetBase`

### 2.12 `DuneAI_Remake/src/models/mamba_unet.py`
- **Lớp / Hàm:**
  - `class MambaUNet(nn.Module)`: `encoder` (MambaVision), `decoder` (UNetDecoder), `forward`
  - `build_mamba_unet(config=None, **kwargs)`
  - Bí danh: `ModularUNet = MambaUNet`, `build_model = build_mamba_unet`

### 2.13 `DuneAI_Remake/src/postprocess.py`
- **Lớp / Hàm:**
  - `unpad_or_uncrop_xy(mask_3d, original_shape)`
  - `resize_from_1mm_3d(mask_1mm, original_spacing, original_shape, is_mask=True)`
  - `postprocess_3d_mask(prob_vol, threshold=0.5, min_area_mm2=4.0, keep_top_k=0)`
  - `compute_3d_metrics(pred_mask, gt_mask, voxel_spacing=None)`
  - `_compute_hd95_mm(pred_pts, gt_pts)`
  - `predict_patient_volume(model, ...)`
  - `save_prediction_nrrd(mask_3d, out_path, ref_nrrd_path=None)`
  - `evaluate_dataset_3d(model, ...)`
  - `load_model_checkpoint(ckpt_path, device, arch_hint=None)`

---

## 3. KIỂM KÊ ALIAS LỚP MÔ HÌNH VÀ VỊ TRÍ GỌI (GREP)

| Alias | Trỏ tới | File định nghĩa | Nơi đang gọi trong mã nguồn | Hướng xử lý theo spec |
|---|---|---|---|---|
| `DuneUNet` | `UNetBase` | `src/models/__init__.py:27` | `src/postprocess.py:291, 321` (chú thích hàm nạp legacy), `test_cache_viewer.py:3` | Xóa alias trong code mới; sửa nơi gọi thành `UNetBase` (Phase 3/7) |
| `PureUNet` | `UNetBase` | `src/models/unet_base.py:116`, `src/models/__init__.py:15` | Không nơi nào trong logic thực thi gọi trực tiếp (chỉ xuất trong `__all__`) | Xóa alias trong code mới |
| `ClassicUNet` | `UNetBase` | `src/models/unet_base.py:117`, `src/models/__init__.py:16` | Không nơi nào trong logic thực thi gọi trực tiếp (chỉ xuất trong `__all__`) | Xóa alias trong code mới |
| `BaselineUNet` | `UNetBase` | `src/models/unet_base.py:118`, `src/models/__init__.py:17` | Không nơi nào trong logic thực thi gọi trực tiếp (chỉ xuất trong `__all__`) | Xóa alias trong code mới |
| `ModularUNet` | `MambaUNet` | `src/models/mamba_unet.py:144`, `src/models/__init__.py:22` | `src/postprocess.py:56, 304, 312` | Xóa alias trong code mới; sửa nơi gọi thành `MambaUNet` (Phase 4/7) |

---

## 4. BẢNG THAM SỐ CLI CŨ CỦA TỪNG LỆNH

### 4.1 `train_main.py` (UNet Base)
- Lệnh: `selftest`
- Lệnh: `train`
  - `--cache-dir` (default: `"data/cache_1mm"`)
  - `--save-dir` (default: `"checkpoints/unet_base_25d"`)
  - `--batch-size` (default: 8)
  - `--epochs` (default: 50)
  - `--lr` (default: 1e-4)
  - `--weight-decay` (default: 1e-4)
  - `--grad-clip` (default: 1.0)
  - `--features` (default: 64)
  - `--activation` (choices: `["relu", "star_relu", "silu", "gelu", "soft_clamp_relu"]`, default: `"relu"`)
  - `--image-size` (default: 512)
  - `--workers` (default: 2)
  - `--patience` (default: 10)
  - `--loss` (choices: `["dice_bce", "tversky"]`, default: `"dice_bce"`)
  - `--scheduler` (choices: `["cosine", "plateau", "none"]`, default: `"cosine"`)
  - `--resume` (default: `None`, hoặc `"auto"`, hoặc path)
  - `--auto-resume` (flag)
  - `--init-weights` (default: `None`)
  - `--limit-train`, `--limit-val` (default: 0)
  - `--rebuild-index` (flag)
  - `--gpu-window` (flag)
  - `--no-lung-mask`, `--no-augment`, `--no-amp`, `--no-progress` (flags)
  - `--seed` (default: 42)

### 4.2 `train_mamba.py` (Mamba-UNet)
- Lệnh: `selftest`
- Lệnh: `train`
  - `--cache-dir` (default: `"data/cache_1mm"`)
  - `--save-dir` (default: `"checkpoints/mamba_unet_25d"`)
  - `--backbone` (choices: `["mamba_vision_T", "mamba_vision_S", "mamba_vision_B"]`, default: `"mamba_vision_T"`)
  - `--activation` (choices: `["relu", "star_relu", "silu", "gelu", "soft_clamp_relu"]`, default: `"relu"`)
  - `--pretrained` (flag)
  - `--batch-size` (default: 8)
  - `--epochs` (default: 50)
  - `--lr` (default: 1e-4)
  - `--weight-decay` (default: 1e-4)
  - `--workers` (default: 2)
  - `--patience` (default: 10)
  - `--loss` (choices: `["dice_bce", "tversky"]`, default: `"dice_bce"`)
  - `--scheduler` (choices: `["cosine", "plateau", "step", "none"]`, default: `"cosine"`)
  - `--grad-clip` (default: 1.0)
  - `--image-size` (default: 512)
  - `--limit-train`, `--limit-val` (default: 0)
  - `--resume`, `--auto-resume`
  - `--rebuild-index`, `--gpu-window`, `--no-lung-mask`, `--no-augment`, `--no-amp`, `--no-progress`
  - `--seed` (default: 42)

### 4.3 `src/postprocess.py`
- Lệnh: `selftest`
- Lệnh: `predict` (`--patient-cache`, `--checkpoint`, `--out-nrrd`, `--ref-nrrd`, `--batch-size`, `--threshold`, `--min-area`, `--tta`)
- Lệnh: `eval` (`--cache-dir`, `--checkpoint`, `--split`, `--batch-size`, `--threshold`, `--min-area`, `--tta`, `--out-csv`)

### 4.4 `src/offline_builder.py`
- Lệnh: `build` (`--cache-dir`, `--out-csv`, `--no-require-lung`)
- Lệnh: `selftest`

### 4.5 `src/medical_prep.py`
- Lệnh: `build` (`--nrrd-dir`, `--cache-dir`, `--workers`, `--limit`, `--interp`, `--overwrite`, `--extract-lungs`)
- Lệnh: `selftest`

---

## 5. HÀM KÍCH HOẠT, HÀM MẤT MÁT VÀ CÁC KHỐI CONV

### 5.1 Hàm kích hoạt (Activations)
- **Tập hàm hiện có trong `src/components/activations.py`:**
  - `relu` (`nn.ReLU`)
  - `gelu` (`nn.GELU`)
  - `silu` / `swish` (`nn.SiLU`)
  - `mish` (`nn.Mish`)
  - `leaky_relu` (`nn.LeakyReLU`)
  - `soft_clamp_relu` (`SoftClampReLU`)
  - `soft_clamp_silu` (`SoftClampSiLU`)
  - `star_relu` (`StarReLU`)
  - `identity` (`nn.Identity`)
- **Mặc định:**
  - `UNetBase`: `DEFAULT_ACTIVATION = "relu"` (áp dụng cho toàn bộ `DoubleConv` của encoder và decoder).
  - `MambaUNet`: `DEFAULT_ACTIVATION = "relu"` (chỉ áp dụng cho các `DoubleConv` của `UNetDecoder`; MambaVision backbone cố định GELU/SiLU).

### 5.2 Hàm mất mát (Losses)
- **Tập hàm hiện có trong `src/components/losses.py`:**
  - `dice` (`DiceLoss`)
  - `tversky` (`TverskyLoss`)
  - `tversky_focal` (`TverskyFocalLoss`)
  - `focal` (`FocalLoss`)
  - `dice_bce` (`DiceBCELoss`)
- **Mặc định cả 2 kiến trúc:**
  - `DEFAULT_LOSS = "dice_bce"`, tham số mặc định: `{"bce_weight": 0.5, "dice_weight": 0.5, "smooth": 1.0, "batch_dice": True}`.

### 5.3 Các khối tích chập (Conv Blocks)
- **Khối `DoubleConv`:**
  - Trong `src/backbones/cnn_baseline.py`: `nn.Sequential(Conv2d(0), BatchNorm2d(1), Act(2), Conv2d(3), BatchNorm2d(4), Act(5))`
  - Trong `src/decoders/unet_decoder.py`: Cùng cấu trúc thứ tự chỉ số con (0, 1, 2, 3, 4, 5).
  - **Phân bổ mới:** `unet_base/blocks.py` và `mamba_unet/blocks.py` sẽ sở hữu bản riêng, độc lập tuyệt đối.
- **Khối `Down`:**
  - `nn.Sequential(MaxPool2d(0), DoubleConv(1))`. Dùng trong encoder U-Net.
- **Khối `UpBlock`:**
  - Trong `unet_base/decoder.py`: Dùng `ConvTranspose2d` hoặc `Upsample`.
  - Trong `mamba_unet/decoder.py`: Dùng `Upsample(scale_factor=2, mode="bilinear")` ghép `DoubleConv`.
- **Khối `OutConv`:**
  - `nn.Conv2d(in, out, kernel_size=1)` với `init_bias=-4.0`.

---

## 6. MỌI VỊ TRÍ NHẮC TỚI `use_stride4_stem`

1. `src/backbones/cnn_baseline.py:74, 80, 90, 96, 99, 124, 143, 151, 160, 166`: Khai báo và rẽ nhánh `stem` giảm xuống H/4.
2. `src/models/unet_base.py:71`: Gọi `CNNBaselineEncoder(..., use_stride4_stem=False)`.
3. `src/models/mamba_unet.py:76`: Gọi `CNNBaselineEncoder(..., use_stride4_stem=True)` ở nhánh backbone CNN dự phòng.
4. `test_refactored_architecture.py:85`: Gọi kiểm thử `CNNBaselineEncoder(..., use_stride4_stem=True)`.

---

## 7. QUY ƯỚC DỮ LIỆU VÀ CƠ CHẾ CHIA TẬP

1. **Các tệp bắt buộc trong mỗi thư mục bệnh nhân (`<cache_dir>/<patient_id>/`):**
   - `image_1mm.npy`: Mảng 3D int16 (HU), kích thước `(Z, 512, 512)`.
   - `mask_1mm.npy`: Mảng 3D uint8 (nhãn u), kích thước `(Z, 512, 512)`.
   - `lung_1mm.npy`: Mảng 3D uint8 (mặt nạ phổi), kích thước `(Z, 512, 512)`.
   - `meta.json`: Thông tin gốc DICOM (shape, spacing, `lung_ok`, v.v.).
2. **Nơi code cũ đọc / ghi `slices.csv`:**
   - Được ghi trực tiếp vào `cache_dir / "slices.csv"` bởi `offline_builder.build_dataset_index()`.
   - Được đọc trực tiếp từ `cache_dir / "slices.csv"` bởi `LIDC25DDataset.__init__()`.
3. **Cách chia Train / Val:**
   - Chia theo **cấp độ bệnh nhân (Patient-level split)** để chống rò rỉ dữ liệu (Data Leakage).
   - Tỉ lệ: `VAL_SPLIT = 0.15` (85% train, 15% val), xáo trộn danh sách bệnh nhân bằng `random.Random(seed=42).shuffle()`.
4. **Cách code cũ tìm danh sách bệnh nhân:**
   - `offline_builder.py` quét `cache_dir.iterdir()`, lọc các thư mục con có đủ `meta.json` và `image_1mm.npy`, sắp xếp tăng dần theo tên (`sorted([p.name for p in ...])`).
   - `dataset.py` đọc trực tiếp danh sách lát cắt và bệnh nhân từ `slices.csv`.
