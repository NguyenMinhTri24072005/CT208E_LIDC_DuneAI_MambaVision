# Đặc tả Kỹ thuật (Technical Specification) - U-Net Baseline

Tài liệu này xác định quy chuẩn mã nguồn cho toàn bộ các module trong dự án.
*Quy định chung:* Sử dụng Python >= 3.10, PyTorch >= 2.0. Tất cả các public classes và functions bắt buộc phải có Type Hints và Docstring.

---

## 1. Module Mô hình (models/)

### 1.1 `models/unet_parts.py`
Chứa các thành phần cấu tạo (building blocks) cơ bản của U-Net.
*   **`class DoubleConv(nn.Module)`**
    *   *Init:* `(in_channels: int, out_channels: int, mid_channels: int = None)`
    *   *Mô tả:* Lớp chập kép bao gồm 2 lần chuỗi `Conv2d` (kernel=3, padding=1, bias=False) $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU(inplace=True)`. Nếu `mid_channels` là None, gán bằng `out_channels`.
*   **`class Down(nn.Module)`**
    *   *Init:* `(in_channels: int, out_channels: int)`
    *   *Mô tả:* Kết hợp `nn.MaxPool2d(2)` và `DoubleConv`.
*   **`class Up(nn.Module)`**
    *   *Init:* `(in_channels: int, out_channels: int, bilinear: bool = True)`
    *   *Mô tả:* Lớp giải mã (Decoder). Nếu `bilinear=True`, sử dụng `nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)` kết hợp với `DoubleConv`. Nếu `False`, dùng `nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)`.
    *   *Forward:* Nhận đầu vào `(x1: Tensor, x2: Tensor)` trong đó `x2` là skip connection. Trước khi `torch.cat`, tự động tính toán bù trừ không gian chiều cao/rộng và áp dụng `F.pad` để căn chỉnh shape chuẩn xác 100%.
*   **`class OutConv(nn.Module)`**
    *   *Init:* `(in_channels: int, out_channels: int)`
    *   *Mô tả:* `nn.Conv2d(in_channels, out_channels, kernel_size=1)`. Dùng ở lớp cuối cùng.

### 1.2 `models/unet_model.py`
*   **`class UNet(nn.Module)`**
    *   *Init:* `(n_channels: int, n_classes: int, bilinear: bool = False, base_channels: int = 64)`
    *   *Cấu trúc:* 
        *   Tầng vào `inc`: `DoubleConv(n_channels, base_channels)`
        *   Tầng xuống (`down1` -> `down4`): Kênh tăng dần `64 -> 128 -> 256 -> 512 -> 1024`. (Nếu `bilinear=True` thì tầng 4 chỉ tới `512` vì bước concat sẽ nhân đôi số kênh).
        *   Tầng lên (`up1` -> `up4`): Kênh giảm dần.
        *   Tầng ra `outc`: `OutConv(base_channels, n_classes)`.
    *   *Output:* Trả về Raw Logits của mạng, **KHÔNG** áp dụng `Sigmoid/Softmax` ở lớp cuối. Xử lý hàm kích hoạt sẽ nằm ở lớp Loss.
    *   *Đặc tả tính năng bổ sung:* Có hàm `init_weights()` với kỹ thuật Kaiming Normal. Hỗ trợ kích hoạt `torch.utils.checkpoint.checkpoint` trên các block nếu RAM yếu (dựa vào cờ tùy chọn).

---

## 2. Module Tiện ích (utils/)

### 2.1 `utils/dataset.py`
*   **`class SegmentationDataset(Dataset)`**
    *   *Khởi tạo:* `(root: str, split: str, transform: Callable = None, image_size: int = 256, n_classes: int = 1, mean: tuple = (0.5,), std: tuple = (0.5,))`
    *   *Logic:* Đọc danh sách file từ thư mục con `images/` và `masks/`. Gom cặp theo thuộc tính `stem` (tên file không lấy đuôi). Nếu phát hiện file ảnh mà không có mask tương ứng, raise Error cảnh báo mạnh.
    *   *Tiền xử lý:* Chuyển numpy/ảnh thành Tensor. Nếu n_classes == 1, trả về mask dạng FloatTensor shape `(1, H, W)` để dùng `BCEWithLogitsLoss`. Nếu n_classes > 1, trả mask LongTensor shape `(H, W)` để dùng `CrossEntropyLoss`.
*   **`def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]`**
    *   Dựa vào file config để nạp Dataset, cấu hình cho train_loader (`shuffle=True, drop_last=True`) và test/val loader.

### 2.2 `utils/transforms.py`
*   Gói bằng thư viện `Albumentations`.
*   **`get_train_transforms`**: Gồm ShiftScaleRotate, RandomRotate90, Horizontal/Vertical Flip, RandomBrightnessContrast. Mask sử dụng thuật toán nội suy `cv2.INTER_NEAREST` để không bị nát giá trị nhị phân.
*   **`get_val_transforms`**: Chỉ thực hiện thao tác Resize về kích thước chỉ định.

### 2.3 `utils/losses.py`
*   **`class DiceLoss`**: Tính Dice dựa trên dự đoán đã chạy qua qua `Sigmoid` (nếu n_classes=1) hoặc `Softmax` (n_classes>1). Hỗ trợ smoothing `eps=1e-6`.
*   **`class BCEWithLogitsLossWrapper`**: Bọc lại `nn.BCEWithLogitsLoss`.
*   **`class FocalLoss`**: Giúp tập trung học ở những vùng biên khối u khó dự đoán.
*   **`class CombinedLoss`**: Hàm nhận vào dict trọng số để cộng tuyến tính nhiều loại loss (Ví dụ: `0.5*BCE + 0.5*Dice`).

### 2.4 `utils/metrics.py`
*   Định nghĩa các hàm rời rạc nhận đầu vào là mảng Numpy hoặc Tensor: `calculate_iou`, `calculate_dice`, `calculate_hd95`.
*   **Lưu ý HD95**: Sử dụng `scipy.spatial.distance.directed_hausdorff` hoặc `medpy.metric.hd95`. Hàm bắt buộc phải có lệnh `try-except` chặn lỗi trường hợp mask Ground Truth rỗng (không có khối u), trong trường hợp đó quy ước trả về `NaN` hoặc `0.0` tùy cấu hình.

---

## 3. Cấu hình (configs/)

### 3.1 `configs/config.yaml`
Là trung tâm điều khiển của hệ thống. Phân tầng rõ rệt:
```yaml
data:
  root_dir: "data" # Tương đối so với unet_baseline
  image_size: 256
  n_classes: 1
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5]
model:
  n_channels: 3
  bilinear: False
  base_channels: 64
train:
  epochs: 50
  batch_size: 16
  lr: 0.001
  weight_decay: 1e-4
  amp: True # Automatic Mixed Precision
  seed: 42
  num_workers: 4
  early_stopping_patience: 10
loss:
  type: "CombinedLoss"
  bce_weight: 0.5
  dice_weight: 0.5
checkpoint:
  save_dir: "checkpoints"
  resume_path: null
logging:
  use_tensorboard: True
```

---

## 4. Các kịch bản chạy (Scripts)

### 4.1 `train.py`
*   *Mục đích:* Kịch bản vòng lặp chính.
*   *Tính năng:* Parse file YAML và parse command-line argument theo cặp key-value để ghi đè (override) tạm thời config.
*   Thiết lập tính năng cố định Seed (Determinism).
*   Đóng gói kiến trúc Trainer theo chuẩn: 1 hàm `train_epoch`, 1 hàm `validate_epoch`.
*   Xuất log ra màn hình Console (dùng `tqdm`) và ghi biểu đồ Tensorboard.
*   Lưu 2 file: `last.pth` (lưu cả optimizer state để resume) và `best.pth` (chỉ lưu weights có metric cao nhất).

### 4.2 `evaluate.py`
*   *Mục đích:* Dùng model đã huấn luyện (`best.pth`) ném vào tập Test để chốt số liệu.
*   *Tính năng:* Chạy 1 vòng qua Test Dataset (không tích lũy Gradient). Lưu lại một file `.csv` chứa điểm số Dice/IoU/HD95 của **từng tấm ảnh**, và dòng cuối hiển thị trung bình ± độ lệch chuẩn (mean ± std).

### 4.3 `predict.py`
*   *Mục đích:* Inference trên ảnh thực tế đưa vào.
*   *Tham số truyền vào:* `--input` (file/thư mục), `--output` (thư mục lưu kết quả), `--checkpoint`, `--threshold` (ngưỡng cắt sigmoid, mặc định 0.5).
*   *Tính năng:* Tự động resize mask AI trả về ngược lại kích thước của bức ảnh gốc. Tạo ra một bức ảnh ghép (overlay) đắp màu xanh/đỏ mờ lên khối u để user dễ nghiệm thu bằng mắt thường.

---

## 5. Google Colab Notebook

### 5.1 `train_on_colab.ipynb`
Bao gồm các cell tuần tự:
1. Gõ lệnh `!nvidia-smi` check VRAM.
2. Mount Google Drive.
3. Chuyển thư mục về đúng Workspace (tạo Symlink nếu lưu dataset/checkpoints vào Drive để chống trôi dữ liệu khi Colab tự reset).
4. `!pip install -r requirements.txt`.
5. `!python train.py --config configs/config.yaml`
6. `%tensorboard --logdir runs/`
7. Cung cấp cell Resume khi huấn luyện bị ngắt giữa chừng (`!python train.py --config ... --resume_path ...`)

---

## 6. Tài liệu mô tả

### 6.1 `requirements.txt`
Ghim cứng phiên bản an toàn cho các core libraries: `torch`, `torchvision`, `albumentations`, `numpy`, `scipy`, `opencv-python-headless`, `PyYAML`, `tqdm`, `tensorboard`, `medpy`.

### 6.2 `README.md`
Chứa toàn bộ hướng dẫn sử dụng nhanh, sơ đồ cấu trúc của dự án, và các lệnh mẫu minh họa chi tiết (VD: `python predict.py --input test.jpg --output results/`). Mọi lệnh đều được chỉ dẫn phải chạy tại thư mục `SOURCES/unet_baseline/`.
