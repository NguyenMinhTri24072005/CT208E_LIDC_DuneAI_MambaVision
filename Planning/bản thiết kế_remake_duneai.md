# BẢN THIẾT KẾ LỘ TRÌNH REMAKE DUNEAI (PYTORCH & COLAB TESTFLOW)

Tài liệu này là bản thiết kế kỹ thuật chi tiết giúp bạn tự tay xây dựng lại toàn bộ hệ thống phân đoạn khối u phổi **DuneAI** từ mã nguồn Keras cũ sang kiến trúc **PyTorch** hiện đại, tốc độ cao và dễ dàng nâng cấp (như thay lõi bằng ResNet hoặc MambaVision).

---

## 1. Kiến trúc Thư mục Dự án (Modular Python)

Toàn bộ mã nguồn được chia thành các file `.py` độc lập theo đúng 5 giai đoạn của Pipeline. Bạn code trên máy tính (VS Code / Cursor), đẩy lên GitHub (hoặc lưu trên Google Drive) và dùng một file Notebook duy nhất trên **Google Colab** để nạp và kiểm thử từng module.

```text
DuneAI_Remake/
├── configs/
│   └── config.py                 # Quản lý toàn bộ siêu tham số (Hyperparameters) & Đường dẫn
├── src/
│   ├── __init__.py
│   ├── medical_prep.py           # [Giai đoạn 1 & 2] Chuẩn hóa HU, Spacing & Bóc tách phổi 3D
│   ├── offline_builder.py        # Script chạy 1 lần: Biến NRRD thô thành kho ảnh .npz/.npy sạch
│   ├── dataset.py                # [Giai đoạn 3] PyTorch Dataset & DataLoader đa luồng
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dune_unet.py          # [Giai đoạn 4A] Mạng U-Net chuẩn DuneAI (Padding same, Nearest)
│   │   └── custom_backbone.py    # [Giai đoạn 4B] Nơi mở rộng sang ResNet / MambaVision
│   ├── losses.py                 # Hàm mất mát BCE + Dice Loss và chỉ số Dice Score trên GPU
│   ├── trainer.py                # Engine huấn luyện tích hợp Mixed Precision (AMP)
│   └── postprocess.py            # [Giai đoạn 5] Bộ lọc thể tích 3D lớn nhất & Xuất file .nrrd
├── notebooks/
│   └── Colab_Test_Lab.ipynb      # Phòng thí nghiệm trên Colab để test từng file .py
└── train_main.py                 # File kích hoạt huấn luyện toàn trình
```

---

## 2. Thiết lập Luồng Làm việc: Code `.py` $\rightarrow$ Test trên Colab

Để sửa file `.py` mà Colab nhận diện ngay lập tức không cần khởi động lại (Restart Runtime), bạn mở một Notebook trên Google Colab và đặt 1 cell cố định ở đầu file:

### Cell 1: Kết nối mã nguồn (Từ GitHub hoặc Google Drive) và bật `autoreload`

```python
# Tự động tải lại module .py mỗi khi file có sự thay đổi
%load_ext autoreload
%autoreload 2

import os
import sys

# CÁCH 1: Nếu bạn lưu project trên Google Drive
from google.colab import drive
drive.mount('/content/drive')
PROJECT_PATH = "/content/drive/MyDrive/DuneAI_Remake"

# CÁCH 2: Nếu bạn pull từ GitHub về Colab
# !git clone https://github.com/<USERNAME>/DuneAI_Remake.git /content/DuneAI_Remake
# PROJECT_PATH = "/content/DuneAI_Remake"

if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

!pip install -q SimpleITK opencv-python scikit-image tqdm
print("✅ Đã kết nối thư mục dự án:", PROJECT_PATH)
```

---

## 3. Lộ trình Xây dựng và Kiểm thử Từng Module (5 Bước)

### BƯỚC 1: Xây dựng `configs/config.py` và `src/medical_prep.py` (Giai đoạn 1 & 2)

#### A. Mục tiêu cần code
1. **`configs/config.py`**: Lưu các hằng số chuẩn của DuneAI:
   - `IMAGE_SIZE = 512`
   - `WINDOW_PARAMS = [1500, -600]` ($WW = 1500$, $WL = -600$)
   - `RESAMPLING_STEP = 25`
   - `TUMOR_SIZE_LOW_LIM = 20.0` ($\text{mm}^2$)
   - `ZERO_SL_RATIO = 0.25`
   - `VAL_SPLIT = 0.15` và `SEED = 42`
2. **`src/medical_prep.py`**: Đóng gói các hàm xử lý ảnh y khoa:
   - `load_nrrd(path)`: Dùng `SimpleITK` đọc mảng 3D `(Z, H, W)` và thông số `spacing` $(x, y, z)$.
   - `clip_and_window_hu(volume, ww=1500, wl=-600, step=25)`:
     - Cắt ngưỡng dưới: mọi giá trị $< -1000\text{ HU}$ gán bằng $-1000$.
     - Kẹp cửa sổ phổi: $\text{low} = WL - \frac{WW}{2} = -1350$, $\text{high} = WL + \frac{WW}{2} = +150$.
     - Lượng tử hóa theo bước nhảy `step=25` để khử nhiễu hạt cảm biến, sau đó chuẩn hóa (normalize) về đoạn $[0.0, 1.0]$.
   - `resize_and_pad_3d(volume, spacing, target_size=512, is_mask=False)`: Co giãn từng lát cắt theo `spacing` thực tế rồi pad/crop về đúng $512 \times 512$.
   - `extract_lungs_mask(volume)`: Chuyển thể hàm `get_seg_lungs` (phân ngưỡng $< -320\text{ HU}$, `clear_border` xóa bàn chụp, xóa khí quản theo tỷ lệ diện tích $0.0053 - 0.0069$, `binary_fill_holes` lấp đầy mạch máu, và lật đối xứng `create_flipped_mask`).

#### B. Kịch bản Test trên Colab

```python
from src.medical_prep import load_nrrd, clip_and_window_hu, resize_and_pad_3d, extract_lungs_mask
import matplotlib.pyplot as plt
import numpy as np

# 1. Nạp thử 1 bệnh nhân
img_path = "/content/sample_data/LIDC-IDRI-0001/image.nrrd"
mask_path = "/content/sample_data/LIDC-IDRI-0001/mask.nrrd"

raw_vol, spacing = load_nrrd(img_path)
raw_mask, _ = load_nrrd(mask_path)

# 2. Chạy thử chuỗi tiền xử lý
lung_mask = extract_lungs_mask(raw_vol)
clean_vol = clip_and_window_hu(raw_vol) * lung_mask
final_vol = resize_and_pad_3d(clean_vol, spacing, target_size=512)
final_mask = resize_and_pad_3d(raw_mask, spacing, target_size=512, is_mask=True)

# 3. In thông số và vẽ lát cắt có khối u lớn nhất
tumor_slice_idx = np.argmax(np.sum(final_mask, axis=(1, 2)))
print(f"Shape gốc: {raw_vol.shape} -> Shape chuẩn hóa: {final_vol.shape}")
print(f"Min/Max cường độ sau chuẩn hóa: {final_vol.min():.3f} / {final_vol.max():.3f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(raw_vol[tumor_slice_idx], cmap='gray'); ax[0].set_title("1. Ảnh CT Thô")
ax[1].imshow(final_vol[tumor_slice_idx], cmap='gray'); ax[1].set_title("2. Đã Tách Phổi & Kẹp HU")
ax[2].imshow(final_mask[tumor_slice_idx], cmap='jet'); ax[2].set_title("3. Mặt nạ Khối u (Ground Truth)")
plt.show()
```

#### C. Kết quả Mong đợi (Pass Criteria)
- **Shape đầu ra:** `(Số_lát_cắt, 512, 512)` cho cả ảnh và mặt nạ.
- **Giá trị Pixel:** `final_vol` nằm gọn trong khoảng $[0.0, 1.0]$, kiểu `float32`. `final_mask` chỉ chứa 2 giá trị $\{0.0, 1.0\}$.
- **Hình ảnh trực quan:** Ở hình số 2, bàn chụp bên dưới lưng và phần xương sườn/mô mềm bên ngoài lồng ngực biến thành màu đen hoàn toàn ($0.0$), hai lá phổi hiện rõ mạch máu và khối u khớp đúng vị trí trên hình số 3.

---

### BƯỚC 2: Xây dựng `src/offline_builder.py` và `src/dataset.py` (Giai đoạn 3)

#### A. Mục tiêu cần code
1. **`src/offline_builder.py`**:
   - Quét toàn bộ thư mục bệnh nhân, chia danh sách bệnh nhân thành 2 tập `train` ($85\%$) và `val` ($15\%$) theo ID bệnh nhân (tránh rò rỉ dữ liệu giữa các lát cắt của cùng 1 người).
   - Với mỗi bệnh nhân, gọi hàm ở Bước 1 để làm sạch, sau đó lọc lấy:
     - Tất cả các lát cắt 2D có diện tích khối u $\ge 20.0\text{ mm}^2$.
     - Một số lượng lát cắt không có khối u theo tỷ lệ `zero_sl_ratio = 0.25`.
   - Lưu từng lát cắt 2D thành file `.npz` (chứa 2 mảng `image` và `mask` kích thước $512 \times 512$) vào thư mục `processed_2d/train/` và `processed_2d/val/`.
2. **`src/dataset.py`**:
   - Viết class `LungTumorDataset(torch.utils.data.Dataset)`:
     - Hàm `__len__`: Trả về tổng số lát cắt `.npz`.
     - Hàm `__getitem__(idx)`: Đọc file `.npz`, áp dụng Augmentation nhẹ (lật ngang/dọc ngẫu nhiên) và chuyển thành PyTorch Tensor kích thước `(1, 512, 512)` kiểu `torch.float32`.

#### B. Kịch bản Test trên Colab

```python
import time
from torch.utils.data import DataLoader
from src.dataset import LungTumorDataset

train_dataset = LungTumorDataset(data_dir="/content/processed_2d/train", augment=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

start_time = time.time()
images, masks = next(iter(train_loader))
load_time = time.time() - start_time

print(f"⏱️ Thời gian nạp 1 batch (8 ảnh): {load_time:.4f} giây")
print(f"📦 Batch Images Shape: {images.shape} | Dtype: {images.dtype}")
print(f"🎯 Batch Masks Shape:  {masks.shape}  | Unique values: {masks.unique().tolist()}")
```

#### C. Kết quả Mong đợi (Pass Criteria)
- **Tốc độ nạp:** `load_time` $< 0.05\text{ giây}$ / batch (so với $\sim 10\text{ giây}$ / batch khi xử lý on-the-fly bằng CPU).
- **Kích thước Tensor:** `torch.Size([8, 1, 512, 512])` cho cả `images` và `masks`.
- **Giá trị Unique của Masks:** Chỉ xuất hiện `[0.0, 1.0]`.

---

### BƯỚC 3: Xây dựng `src/models/dune_unet.py` (Giai đoạn 4A - Kiến trúc Mạng)

#### A. Mục tiêu cần code
Tái tạo chính xác cấu trúc U-Net của tác giả trong PyTorch:
- **Khối tích chập kép (`DoubleConv`)**: Hai lớp `nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)` đi kèm `nn.ReLU(inplace=True)`. Áp dụng khởi tạo trọng số He Normal (`nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')`).
- **Nhánh Encoder**:
  - `enc1`: $1 \rightarrow 64$
  - `enc2`: `MaxPool2d(2)` $\rightarrow 64 \rightarrow 128$
  - `enc3`: `MaxPool2d(2)` $\rightarrow 128 \rightarrow 256$
  - `enc4`: `MaxPool2d(2)` $\rightarrow 256 \rightarrow 512 \rightarrow$ `Dropout2d(0.5)`
  - `bottleneck`: `MaxPool2d(2)` $\rightarrow 512 \rightarrow 1024 \rightarrow$ `Dropout2d(0.5)`
- **Nhánh Decoder (`UpBlock`)**:
  - Dùng `nn.Upsample(scale_factor=2, mode='nearest')` kết hợp `Conv2d(in_ch, out_ch, kernel_size=2, padding='same')` + `ReLU`.
  - Ghép kênh (`torch.cat`) với nhánh Encoder tương ứng rồi đi qua `DoubleConv`.
- **Tầng Đầu ra (`OutHead`)**:
  - Lớp đệm hạ chiều: `Conv2d(64, 2, kernel_size=3, padding=1)` + `ReLU`.
  - Lớp chốt hạ: `Conv2d(2, 1, kernel_size=1)` (Xuất ra `logits` để dùng với `BCEWithLogitsLoss` cho ổn định số học, khi dự đoán mới bọc thêm `torch.sigmoid`).

#### B. Kịch bản Test trên Colab

```python
import torch
from src.models.dune_unet import DuneUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DuneUNet(in_channels=1, out_channels=1).to(device)

# Tạo dữ liệu giả lập 2 ảnh 512x512
dummy_input = torch.randn(2, 1, 512, 512, device=device)
with torch.no_grad():
    output = model(dummy_input)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Input Shape:  {dummy_input.shape}")
print(f"Output Shape: {output.shape}")
print(f"Tổng số tham số (Parameters): {total_params:,}")
```

#### C. Kết quả Mong đợi (Pass Criteria)
- **Output Shape:** Đúng `torch.Size([2, 1, 512, 512])` (không bị lệch $1 - 2\text{ pixel}$ ở viền nhờ `padding='same'`).
- **Tổng số tham số:** Xấp xỉ $31,030,597$ parameters (khớp với mạng `model_v7.json` gốc).
- Không báo lỗi tràn bộ nhớ hay lệch chiều khi `torch.cat` ở các nhánh Skip Connection.

---

### BƯỚC 4: Xây dựng `src/losses.py` và `src/trainer.py` (Giai đoạn 4B - Huấn luyện)

#### A. Mục tiêu cần code
1. **`src/losses.py`**:
   - Hàm `dice_score(logits, targets, smooth=1e-6)`: Áp dụng `probs = torch.sigmoid(logits)`, làm phẳng (flatten) và tính theo công thức:
     $$\text{Dice} = \frac{2 \sum (P \cdot T) + \text{smooth}}{\sum P + \sum T + \text{smooth}}$$
   - Class `BCEDiceLoss(nn.Module)`: Tổng của `F.binary_cross_entropy_with_logits(logits, targets)` và $(1.0 - \text{dice\_score}(\text{logits}, \text{targets}))$.
2. **`src/trainer.py`**:
   - Vòng lặp `train_one_epoch` và `validate` sử dụng `torch.amp.autocast('cuda')` và `GradScaler` (giúp tăng tốc gấp 2 lần trên GPU T4/P100).
   - Thiết lập `ReduceLROnPlateau` (`factor=0.5`, `patience=5`, `min_lr=1e-6`), `EarlyStopping` (`patience=12`), tự động lưu file trọng số `best_model.pth` khi `val_dice` đạt đỉnh mới và ghi nhật ký ra `training_log.csv`.

#### B. Kịch bản Test trên Colab: Bài test "Overfit 1 Batch" (Quan trọng nhất)
Trước khi cắm máy train hàng chục tiếng, bạn bắt buộc phải chạy bài test kinh điển này: **Cho mạng học thuộc lòng duy nhất 1 batch (8 ảnh) trong 50 bước lặp**.

```python
import torch
import torch.optim as optim
from src.losses import BCEDiceLoss, dice_score

model.train()
# Tắt tạm thời Dropout để kiểm tra khả năng hội tụ thuần túy của mạng
for m in model.modules():
    if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
        m.p = 0.0

criterion = BCEDiceLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Lấy cố định 1 batch duy nhất có chứa khối u
single_images, single_masks = next(iter(train_loader))
single_images, single_masks = single_images.to(device), single_masks.to(device)

for step in range(1, 51):
    optimizer.zero_grad()
    logits = model(single_images)
    loss = criterion(logits, single_masks)
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0 or step == 1:
        dice = dice_score(logits.detach(), single_masks).item()
        print(f"Step {step:02d}/50 | Loss: {loss.item():.4f} | Dice Score: {dice:.4f}")
```

#### C. Kết quả Mong đợi (Pass Criteria)
- **Ở Step 01:** `Loss` khoảng $1.20 - 1.50$, `Dice Score` rất thấp ($0.01 - 0.05$).
- **Ở Step 50:** `Loss` phải giảm sâu xuống $< 0.15$ và `Dice Score` phải vọt lên **$> 0.90$** (thậm chí $0.95+$).
- **Ý nghĩa:** Nếu mô hình đạt $\text{Dice} > 0.90$ trên 1 batch, điều đó chứng minh $100\%$ rằng: nhãn dữ liệu khớp với ảnh, hàm Loss tính đạo hàm đúng chiều, và kiến trúc mạng hoạt động hoàn hảo. Nếu Dice vẫn dậm chân ở $0.00$, chắc chắn code Loss hoặc chuẩn hóa Mask đang bị lỗi.

---

### BƯỚC 5: Xây dựng `src/postprocess.py` (Giai đoạn 5 - Hậu xử lý 3D Lâm sàng)

#### A. Mục tiêu cần code
- Hàm `predict_patient_3d(model, nrrd_path)`:
  - Đọc khối 3D của bệnh nhân mới, chạy tiền xử lý Bước 1, dự đoán từng lát cắt 2D qua mạng và xếp chồng lại thành khối xác suất 3D.
  - Ngưỡng hóa nhị phân ($\text{prob} > 0.5$).
- Hàm `filter_max_connected_volume(binary_volume_3d)`:
  - Sử dụng `scipy.ndimage.label` để đánh số toàn bộ các cụm điểm ảnh 3D dính liền nhau trong không gian 3 chiều.
  - Đếm thể tích (số lượng voxel) của từng cụm và **chỉ giữ lại cụm có thể tích lớn nhất**, xóa toàn bộ các đốm nhiễu dương tính giả (False Positives) xung quanh về $0$.
- Hàm `export_to_nrrd(mask_3d, reference_nrrd_path, output_path)`:
  - Đưa mặt nạ 3D về lại kích thước gốc ban đầu, gán lại hệ tọa độ vật lý (Origin, Spacing, Direction) từ file `image.nrrd` gốc và lưu thành `DL_mask.nrrd`.

#### B. Kịch bản Test trên Colab

```python
import numpy as np
from scipy.ndimage import label
from src.postprocess import filter_max_connected_volume

# Tạo khối 3D giả lập (60 lát cắt, 512x512)
synthetic_3d = np.zeros((60, 512, 512), dtype=np.uint8)

# 1. Tạo 1 khối u thật lớn ở giữa (kích thước 20x30x30 = 18,000 voxels)
synthetic_3d[20:40, 200:230, 200:230] = 1

# 2. Cố tình tạo 4 đốm nhiễu dương tính giả rải rác ở các lát cắt khác
synthetic_3d[5:8, 50:55, 50:55] = 1
synthetic_3d[10:12, 400:404, 100:104] = 1
synthetic_3d[50:53, 120:125, 350:355] = 1
synthetic_3d[55:57, 300:303, 300:303] = 1

_, num_before = label(synthetic_3d)
cleaned_3d = filter_max_connected_volume(synthetic_3d)
_, num_after = label(cleaned_3d)

print(f"Số cụm khối u trước khi lọc 3D: {num_before} cụm (gồm 1 u thật + 4 nhiễu)")
print(f"Số cụm khối u sau khi lọc 3D:   {num_after} cụm")
print(f"Tổng số voxel còn lại:          {cleaned_3d.sum()} (Kỳ vọng: 18000)")
```

#### C. Kết quả Mong đợi (Pass Criteria)
- `Số cụm trước khi lọc`: `5 cụm`.
- `Số cụm sau khi lọc`: Đúng `1 cụm`.
- `Tổng số voxel còn lại`: Đúng `18000` (4 đốm nhiễu giả đã bị dọn sạch $100\%$, giữ nguyên vẹn khối u chính).

---

## 4. Bảng Tóm tắt Kiểm tra Tiến độ (Checklist)

| Thứ tự | Tệp mã nguồn cần tạo | Mục đích cốt lõi | Tiêu chuẩn Pass trên Colab |
| :--- | :--- | :--- | :--- |
| **1** | `configs/config.py`<br>`src/medical_prep.py` | Chuẩn hóa Spacing, HU Window & Tách phổi | Ảnh ra `(Z, 512, 512)`, nền ngoài phổi đen tuyệt đối ($0.0$), dải sáng $[0, 1]$. |
| **2** | `src/offline_builder.py`<br>`src/dataset.py` | Lưu lát cắt sạch `.npz` & Nạp qua PyTorch DataLoader | Nạp 1 batch 8 ảnh mất $< 0.05\text{s}$, Tensor shape `(8, 1, 512, 512)`. |
| **3** | `src/models/dune_unet.py` | Dựng mạng U-Net (Padding same, Nearest Upsample) | Output shape khớp Input `(B, 1, 512, 512)`, $\sim 31.03\text{ triệu}$ tham số. |
| **4** | `src/losses.py`<br>`src/trainer.py` | BCE + Dice Loss & Vòng lặp huấn luyện AMP | Vượt qua bài test **Overfit 1 Batch**: sau 50 bước, $\text{Dice Score} > 0.90$. |
| **5** | `src/postprocess.py` | Lọc thể tích 3D liền mạch lớn nhất & Xuất NRRD | Xóa sạch các đốm nhiễu rời rạc, giữ lại duy nhất 1 khối liên thông lớn nhất. |
| **6** | `src/models/custom_backbone.py` | *(Mở rộng)* Thay Encoder bằng ResNet / MambaVision | Cắm vào đúng khung `trainer.py` ở Bước 4 mà không cần sửa lại Bước 1, 2, 5. |