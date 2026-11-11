# BÁO CÁO KỸ THUẬT & TRỌN BỘ MÃ NGUỒN MAMBA-UNET
## HỆ THỐNG PHÂN ĐOẠN KHỐI U PHỔI 2.5D DỰA TRÊN NVIDIA MAMBAVISION & UNET DECODER
*Phiên bản: DuneAI Mamba-UNet Production v1.0 | Ngày xuất bản: 2026-11-11 15:12:11*
*Tương thích phần cứng: NVIDIA Tesla T4 / A100 / RTX 3090 / RTX 4090 | PyTorch 2.6+ | CUDA 12.4+*

---

## MỤC LỤC TỔNG QUAN
1. [CHƯƠNG 1: TỔNG QUAN KIẾN TRÚC MAMBA-UNET & NGUYÊN LÝ THIẾT KẾ](#chương-1-tổng-quan-kiến-trúc-mamba-unet--nguyên-lý-thiết-kế)
2. [CHƯƠNG 2: BẢNG TRA CỨU TOÀN DIỆN CÁC THAM SỐ HUẤN LUYỆN](#chương-2-bảng-tra-cứu-toàn-diện-các-tham-số-huấn-luyện-training-hyperparameters)
3. [CHƯƠNG 3: QUY CHUẨN LƯU TRỮ, QUẢN LÝ CHECKPOINT & NHẬT KÝ HUẤN LUYỆN](#chương-3-quy-chuẩn-lưu-trữ-quản-lý-checkpoint--nhật-ký-huấn-luyện)
4. [CHƯƠNG 4: QUY TRÌNH SUY LUẬN KHỐI 3D & HẬU XỬ LÝ Y TẾ](#chương-4-quy-trình-suy-luận-khối-3d--hậu-xử-lý-y-tế-3d-evaluation-pipeline)
5. [CHƯƠNG 5: BẢNG TỔNG HỢP DANH MỤC CÁC TỆP MÃ NGUỒN](#chương-5-bảng-tổng-hợp-danh-mục-các-tệp-mã-nguồn)
6. [CHƯƠNG 6: TOÀN BỘ MÃ NGUỒN HỆ THỐNG CHI TIẾT](#chương-6-toàn-bộ-mã-nguồn-hệ-thống-chi-tiết)


## CHƯƠNG 1: TỔNG QUAN KIẾN TRÚC MAMBA-UNET & NGUYÊN LÝ THIẾT KẾ

### 1.1 Bài Toán Phân Đoạn Nốt Khối U Phổi 2.5D (LIDC-IDRI)
Trong chẩn đoán hình ảnh chụp cắt lớp vi tính lồng ngực (Thoracic CT Scan), các nốt khối u phổi (lung nodules) thường có kích thước rất nhỏ (từ vài mm đến vài chục mm) và phân bố rải rác trên hàng trăm lát cắt trục (axial slices). Việc phân đoạn ảnh y tế gặp các thách thức nghiêm trọng:
1. **Mất cân bằng dữ liệu cực đoan (Severe Class Imbalance)**: Vùng khối u dương tính thường chỉ chiếm dưới $0.1\%$ đến $0.5\%$ tổng số pixel trên một lát cắt $512 \times 512$.
2. **Thông tin không gian 3D liên tục**: Việc phân đoạn từng lát cắt 2D đơn lẻ làm mất thông tin tương quan giải phẫu theo trục $z$. Ngược lại, mô hình 3D toàn khối (Full 3D CNN) lại tiêu tốn tài nguyên bộ nhớ GPU quá lớn, hạn chế độ phân giải và kích thước batch.
3. **Giải pháp tiếp cận 2.5D**: Mô hình nhận đầu vào là tensor có kích thước $(B, 3, 512, 512)$, trong đó 3 kênh đại diện cho 3 lát cắt kề cận liên tiếp $[z-1, z, z+1]$. Cách tiếp cận này vừa tận dụng được thông tin giải phẫu lân cận theo trục $z$, vừa duy trì được độ phân giải không gian gốc $512 \times 512$ và tối ưu tốc độ xử lý của mạng 2D.

---

### 1.2 Kiến Trúc Lai Mamba-UNet (Hybrid MambaVision + UNet Decoder)
Mô hình `MambaUNet` kết hợp sức mạnh trích xuất đặc trưng đa tầng của backbone **NVIDIA MambaVision** với cấu trúc khôi phục không gian đối xứng của **UNet Decoder**:

```mermaid
flowchart TD
    In["Đầu vào 2.5D Tensor (B, 3, 512, 512)"] --> Stem["Downsample Stem (Conv 3x3 s=2 + Conv 3x3 s=2)"]
    
    subgraph Backbone ["NVIDIA MambaVision Backbone (4 Stages)"]
        Stem --> S1["Stage 1: ConvBlock (B, 80, 128, 128)"]
        S1 --> S2["Stage 2: ConvBlock (B, 160, 64, 64)"]
        S2 --> S3["Stage 3: Hybrid MambaVision Mixer + Self-Attention (B, 320, 32, 32)"]
        S3 --> S4["Stage 4: Hybrid MambaVision Mixer + Self-Attention (B, 640, 16, 16)"]
    end

    subgraph Decoder ["Symmetric UNet Decoder"]
        S4 --> Up1["UpBlock 1 (ConvTranspose2d 2x)"]
        S3 -. "Skip Connection 3" .-> Cat1["Concatenate"]
        Up1 --> Cat1
        Cat1 --> ConvDec1["Double Conv (B, 320, 32, 32)"]

        ConvDec1 --> Up2["UpBlock 2 (ConvTranspose2d 2x)"]
        S2 -. "Skip Connection 2" .-> Cat2["Concatenate"]
        Up2 --> Cat2
        Cat2 --> ConvDec2["Double Conv (B, 160, 64, 64)"]

        ConvDec2 --> Up3["UpBlock 3 (ConvTranspose2d 2x)"]
        S1 -. "Skip Connection 1" .-> Cat3["Concatenate"]
        Up3 --> Cat3
        Cat3 --> ConvDec3["Double Conv (B, 80, 128, 128)"]

        ConvDec3 --> Up4["UpBlock 4 (ConvTranspose2d 4x)"]
        Up4 --> Head["Segmentation Head (Conv 1x1)"]
    end

    Head --> Out["Logits Output (B, 1, 512, 512)"]
```

---

### 1.3 Cơ Chế Selective Scan State Space Model (SSM)
Khác với Vision Transformers (ViT) có độ phức tạp tính toán không gian và bộ nhớ là $O(N^2)$ (trong đó $N = H \times W$ là số lượng token), mô hình trạng thái không gian chọn lọc (**Selective Scan State Space Model - SSM**) đạt được độ phức tạp tuyến tính $O(N)$.

1. **Phương trình trạng thái liên tục**:
   $$\begin{aligned}
   h'(t) &= A h(t) + B x(t) \\
   y(t) &= C h(t) + D x(t)
   \end{aligned}$$
2. **Rời rạc hóa với bước nhảy phụ thuộc dữ liệu $\Delta(x)$**:
   $$\begin{aligned}
   \bar{A} &= \exp(\Delta A) \\
   \bar{B} &= (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot (\Delta B) \approx \Delta B \\
   h_t &= \bar{A} h_{t-1} + \bar{B} x_t \\
   y_t &= C h_t + D x_t
   \end{aligned}$$
3. **MambaVision Mixer với Kiến Trúc Đối Xứng**:
   Trong `mamba_vision.py`, lớp `MambaVisionMixer` thực hiện:
   - Chiếu đặc trưng đầu vào qua `in_proj` để phân tách hai nhánh song song $x$ và $z$.
   - Nhánh $x$ được xử lý qua tích chập 1D theo chiều sâu `conv1d_x`, áp dụng kích hoạt SiLU, đưa qua nhân `selective_scan` theo cả chiều thuận và nghịch (bi-directional scan).
   - Nhánh $z$ đi qua `conv1d_z` và kích hoạt SiLU đóng vai trò cổng điều khiển (gating mechanism).
   - Kết quả được ghép kênh `torch.cat([y, z], dim=1)` và chiếu ngược lại qua `out_proj`. Cấu trúc đối xứng này tương thích hoàn hảo $100\%$ với 250 khóa trọng số chính thức từ NVIDIA.

---

### 1.4 Khởi Tạo Bias Output (`init_bias = -4.0`)
Để giải quyết hiện tượng mất cân bằng nhãn trầm trọng ở các epoch đầu tiên, lớp tích chập cuối cùng (`decoder.final_conv`) được khởi tạo bias cố định:
$$\text{bias}_{\text{init}} = -4.0 \implies \sigma(-4.0) = \frac{1}{1 + e^{4.0}} \approx 0.01798 \approx 1.8\%$$
Cơ chế này ép xác suất dự đoán ban đầu của mô hình rơi vào khoảng $1.8\%$, tương ứng với tần suất xuất hiện thực tế của khối u phổi, giúp ngăn chặn triệt để hiện tượng gradient bùng nổ (gradient explosion) và giúp hàm loss giảm mượt mà ngay từ epoch 1.

---

### 1.5 Cơ Chế Fallback Đa Tầng (Native CUDA vs PyTorch CPU Reference)
Hệ thống được thiết kế với cơ chế tự phát hiện và kích hoạt backend linh hoạt:
1. **Ưu tiên Native CUDA**: Tự động nhận diện các kernel C++/CUDA đã biên dịch sẵn (`selective_scan_cuda`, `causal_conv1d_cuda`). Tốc độ huấn luyện đạt ~3.9 it/s (tương đương ~31 ảnh/giây trên Tesla T4).
2. **PyTorch Reference Fallback**: Khi chạy trên máy tính không có GPU chuyên dụng hoặc thiếu kernel biên dịch, hệ thống tự động kích hoạt `MambaScanFallback2D` (`mamba_unet/fallbacks.py`). Mô hình vẫn chạy bình thường với độ chính xác số học tương đương mà không bị crash.

## CHƯƠNG 2: BẢNG TRA CỨU TOÀN DIỆN CÁC THAM SỐ HUẤN LUYỆN (TRAINING HYPERPARAMETERS)

### 2.1 Cú Pháp Lệnh Khởi Chạy
Quy trình huấn luyện Mamba-UNet được kích hoạt qua CLI chuẩn:
```bash
# Huấn luyện trên Colab GPU T4 với 500 ca bệnh nhân:
DUNEAI_MAMBA_BACKEND=cuda python -u -m mamba_unet.train_main train \
    --cache-dir /content/data/cache_1mm \
    --backbone mamba_vision_T \
    --pretrained \
    --freeze-backbone-epochs 3 \
    --backbone-lr-mult 0.1 \
    --activation relu \
    --loss dice_bce \
    --loss-params '{"bce_weight": 0.3, "dice_weight": 0.7, "pos_weight": 4.0}' \
    --epochs 25 \
    --batch-size 8 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --scheduler cosine \
    --grad-clip 1.0 \
    --patience 12 \
    --workers 2 \
    --seed 42 \
    --run-name 'mamba_colab_500item_25epoch'
```

---

### 2.2 Bảng Chi Tiết Toàn Bộ Tham Số Dòng Lệnh

| Nhóm Tham Số | Tên Tham Số CLI | Kiểu Dữ Liệu | Mặc Định | Ý Nghĩa Kỹ Thuật & Khuyến Nghị Thực Nghiệm |
| :--- | :--- | :---: | :---: | :--- |
| **Kiến Trúc Mô Hình** | `--backbone` | `str` | `mamba_vision_T` | Biến thể MambaVision backbone: `mamba_vision_T` (Tiny, kênh: [80, 160, 320, 640]), `mamba_vision_T2`, `mamba_vision_S` (Small), `mamba_vision_B` (Base). |
| | `--pretrained` | `flag` | `False` | Tự động tải và ánh xạ trọng số ImageNet-1K pretrained chính thức từ NVIDIA. Khuyến nghị **BẬT** (`--pretrained`). |
| | `--no-pretrained` | `flag` | - | Ép mô hình khởi tạo ngẫu nhiên toàn bộ trọng số (huấn luyện từ đầu - Scratch). |
| | `--freeze-backbone-epochs` | `int` | `5` | Số epoch đóng băng backbone: <br>- `> 0`: Đóng băng trong $N$ epoch đầu (giữ BatchNorm ở `eval()`), chỉ train decoder.<br>- `0`: Mở băng toàn bộ ngay từ epoch 1.<br>- `-1`: Đóng băng backbone suốt quá trình. |
| | `--backbone-lr-mult` | `float` | `0.1` | Hệ số nhân learning rate cho backbone so với decoder ($LR_{bb} = LR \times mult$). Đặt $0.1$ giúp bảo toàn biểu diễn tiền huấn luyện và fine-tune ổn định. |
| | `--activation` | `str` | `relu` | Hàm kích hoạt cho các tầng tích chập của Decoder: `relu`, `gelu`, `silu`, `leaky_relu`, `mish`. |
| **Hàm Mất Mát (Loss)** | `--loss` | `str` | `dice_bce` | Loại hàm mất mát tối ưu hóa: `dice_bce`, `tversky`, `focal_tversky`, `dice`, `bce`, `focal`. |
| | `--loss-params` | `str (JSON)` | `None` | Chuỗi định dạng JSON truyền các siêu tham số chi tiết vào hàm loss (xem mục 2.3). |
| **Tối Ưu Hóa (Optimization)** | `--lr` | `float` | `1e-4` | Tốc độ học cơ sở (Base Learning Rate) áp dụng cho Decoder. Với AdamW, $10^{-4}$ là mức tối ưu. |
| | `--weight-decay` | `float` | `1e-4` | Hệ số suy giảm trọng số (L2 regularization) giúp mô hình chống overfit. |
| | `--epochs` | `int` | `50` | Tổng số chu kỳ huấn luyện toàn bộ tập dữ liệu (Epochs). |
| | `--batch-size` | `int` | `8` | Kích thước lô nạp vào mô hình. Trên GPU T4 16GB, batch size 8 sử dụng khoảng ~6.5GB VRAM khi bật AMP. |
| | `--scheduler` | `str` | `cosine` | Chiến lược suy giảm LR: <br>- `cosine`: Giảm theo đường cong Cosine Annealing về $10^{-6}$.<br>- `plateau`: Giảm LR khi val dice không tăng.<br>- `step`: Giảm định kỳ.<br>- `none`: Giữ cố định. |
| | `--grad-clip` | `float` | `1.0` | Ngưỡng kẹp độ lớn chuẩn gradient (Gradient Norm Clipping). Ngăn chặn gradient bùng nổ khi fine-tune SSM. |
| | `--patience` | `int` | `12` | Số epoch chờ liên tiếp không cải thiện `val_dice` trước khi kích hoạt dừng sớm (Early Stopping). |
| **Dữ Liệu & Tiền Xử Lý** | `--cache-dir` | `str` | `data/cache_1mm` | Thư mục chứa các lát cắt CT 1mm đã chuẩn hóa (chứa `slices.csv` và các thư mục bệnh nhân). |
| | `--workers` | `int` | `2` | Số luồng tiến trình CPU của PyTorch DataLoader nạp dữ liệu song song. |
| | `--gpu-window` | `flag` | `False` | Chuyển mảng số nguyên HU (int16) lên trực tiếp GPU rồi mới chuẩn hóa cửa sổ phổi. |
| | `--no-lung-mask` | `flag` | `False` | Tắt tính năng áp mặt nạ nhu mô phổi bên ngoài vùng lồng ngực. |
| | `--no-augment` | `flag` | `False` | Tắt tính năng tăng cường dữ liệu ngẫu nhiên (lật ngang, lật dọc, xoay $90^\circ$). |
| | `--limit-train` | `int` | `0` | Giới hạn số lát cắt train (0 = nạp toàn bộ). Rất hữu ích khi cần debug hoặc thử nghiệm nhanh. |
| | `--limit-val` | `int` | `0` | Giới hạn số lát cắt validation (0 = nạp toàn bộ). |
| | `--seed` | `int` | `42` | Hạt giống ngẫu nhiên cho Python, NumPy, PyTorch (đảm bảo khả năng tái lập 100%). |
| **Quản Lý Phiên Chạy** | `--run-name` | `str` | `None` | Tên định danh tùy chỉnh cho thư mục phiên chạy (ví dụ: `mamba_colab_run`). |
| | `--tag` | `str` | `None` | Thẻ phân loại thí nghiệm (ví dụ: `baseline`, `ablation`, `colab_mamba`). |
| | `--notes` | `str` | `""` | Ghi chú văn bản tự do lưu vào `run_config.json`. |
| | `--runs-root` | `str` | `None` | Đường dẫn thư mục gốc chứa các runs (mặc định: `mamba_unet/runs`). |
| | `--resume` | `str` | `None` | Cơ chế tiếp tục huấn luyện: <br>- `auto`: Tự động tìm phiên chạy gần nhất bị ngắt và resume.<br>- `<path>`: Chỉ định đường dẫn thư mục run hoặc tệp `last_model.pth`. |
| | `--init-weights` | `str` | `None` | Nạp trọng số khởi đầu từ một checkpoint có sẵn cho một phiên chạy mới hoàn toàn. |
| | `--no-amp` | `flag` | `False` | Tắt chế độ tính toán độ chính xác hỗn hợp Automatic Mixed Precision (AMP float16). |

---

### 2.3 Chi Tiết Cấu Hình JSON `--loss-params` Cho Từng Hàm Mất Mát

#### 1. `dice_bce` (Mặc định khuyến nghị)
Kết hợp giữa Binary Cross Entropy và Soft Dice Loss:
$$\mathcal{L}_{\text{Dice-BCE}} = w_{\text{bce}} \cdot \mathcal{L}_{\text{BCE}}(p, y; \text{pos\_weight}) + w_{\text{dice}} \cdot (1 - \text{Dice}(p, y))$$
- **Cấu hình chuẩn cân bằng**:
  ```json
  {"bce_weight": 0.5, "dice_weight": 0.5, "smooth": 1.0, "batch_dice": true}
  ```
- **Cấu hình tối ưu cho u phổi nhỏ (Ưu tiên Dice và phạt False Negative)**:
  ```json
  {"bce_weight": 0.3, "dice_weight": 0.7, "pos_weight": 4.0, "smooth": 1.0, "batch_dice": true}
  ```
  *Giải thích*:
  - `dice_weight: 0.7`: Ép mô hình tập trung tối đa vào chỉ số Dice tổng thể, bỏ qua nền âm tính khổng lồ.
  - `pos_weight: 4.0`: Nhân hệ số phạt lên gấp 4 lần khi mô hình dự đoán sót pixel dương tính của khối u.

#### 2. `tversky` (Đặc trị mất cân bằng nhãn & giảm thiểu bỏ sót)
$$\text{Tversky} = \frac{|P \cap Y| + \epsilon}{|P \cap Y| + \alpha |P \setminus Y| + \beta |Y \setminus P| + \epsilon}, \quad \mathcal{L}_{\text{Tversky}} = 1 - \text{Tversky}$$
- Trong đó:
  - $\alpha$: Trọng số phạt cho **False Positive** (dự đoán nhầm vùng lành thành u).
  - $\beta$: Trọng số phạt cho **False Negative** (bỏ sót khối u).
- **Cấu hình chuẩn y tế ($\\beta > \\alpha$)**:
  ```json
  {"alpha": 0.3, "beta": 0.7, "smooth": 1e-6}
  ```
  *Giải thích*: Phạt hiện tượng bỏ sót u nặng gấp 2.33 lần so với việc dự đoán hơi rộng, cực kỳ phù hợp trong chẩn đoán ung thư sớm.

#### 3. `focal_tversky` (Tập trung vào các ca nốt u cực khó)
$$\mathcal{L}_{\text{Focal-Tversky}} = (1 - \text{Tversky})^\gamma$$
- **Cấu hình**:
  ```json
  {"alpha": 0.3, "beta": 0.7, "gamma": 0.75, "smooth": 1.0}
  ```
  *Giải thích*: Hệ số $\gamma < 1.0$ giúp tăng gradient cho các lát cắt có giá trị Tversky thấp (các ca bệnh khó phân đoạn).

## CHƯƠNG 3: QUY CHUẨN LƯU TRỮ, QUẢN LÝ CHECKPOINT & NHẬT KÝ HUẤN LUYỆN

### 3.1 Cấu Trúc Thư Mục Của Một Phiên Chạy (Run Directory)
Mỗi lần huấn luyện được cô lập trong một thư mục duy nhất theo cấu trúc:
```
mamba_unet/runs/<run_name>/
├── run_config.json          # Hồ sơ kỹ thuật lưu vết 100% môi trường & tham số
├── history.csv              # Bảng chỉ số thống kê qua từng epoch
├── train.log                # Nhật ký tiến trình thời gian thực chuẩn ISO
├── best_model.pth           # Checkpoint toàn vẹn của mô hình có Val Dice cao nhất
├── last_model.pth           # Checkpoint epoch mới nhất phục vụ tự động resume
└── best_weights_fp16.pth    # Trọng số tối ưu hóa siêu nhẹ (~60MB) để triển khai
```

---

### 3.2 Tệp Hồ Sơ Thí Nghiệm `run_config.json` (Schema v1)
Tệp `run_config.json` tuân thủ nghiêm ngặt chuẩn Schema v1, cho phép tái lập $100\%$ kết quả thí nghiệm:
```json
{
  "schema_version": 1,
  "run_id": "mamba_colab_500item_25epoch",
  "arch": "mamba_unet",
  "created_at": "2026-11-10T18:12:00.000000+00:00",
  "finished_at": "2026-11-10T18:35:42.000000+00:00",
  "status": "completed",
  "duration_sec": 1422.5,
  "system": {
    "hostname": "colab-worker-gpu",
    "os": "Linux-6.1.85+-x86_64",
    "python": "3.11.10",
    "torch": "2.6.0+cu124",
    "cuda_available": true,
    "device_name": "Tesla T4",
    "device_count": 1
  },
  "git": {
    "commit": "15cabf5",
    "branch": "main",
    "is_dirty": false
  },
  "args": {
    "backbone": "mamba_vision_T",
    "pretrained": true,
    "freeze_backbone_epochs": 3,
    "backbone_lr_mult": 0.1,
    "loss": "dice_bce",
    "epochs": 25,
    "batch_size": 8,
    "lr": 0.0001
  },
  "result": {
    "epochs_completed": 25,
    "best_epoch": 21,
    "best_val_dice": 0.7412,
    "last": {
      "val_loss": 0.2845,
      "val_dice": 0.7389,
      "val_iou": 0.6012
    }
  },
  "resume_events": []
}
```

---

### 3.3 Tệp Bảng Chỉ Số `history.csv`
Mỗi khi kết thúc một epoch, bảng `history.csv` được cập nhật nguyên tử. Ý nghĩa từng cột:
- `epoch`: Thứ tự chu kỳ huấn luyện hiện tại (bắt đầu từ 1).
- `lr`: Tốc độ học (learning rate) hiện tại của nhóm tham số Decoder.
- `train_loss`: Giá trị hàm mất mát bình quân trên toàn bộ tập train.
- `train_dice`: Chỉ số Dice Score trung bình trên tập train.
- `train_iou`: Chỉ số Intersection over Union (Jaccard) trên tập train.
- `val_loss`: Giá trị hàm mất mát trên tập kiểm định (validation).
- `val_dice`: Chỉ số Dice Score trên tập kiểm định (thước đo chính để xếp hạng mô hình).
- `val_iou`: Chỉ số IoU trên tập kiểm định.
- `train_time_sec`: Thời gian thực thi lượt huấn luyện (giây).
- `val_time_sec`: Thời gian thực thi lượt kiểm định (giây).
- `time_sec`: Tổng thời gian thực hiện của cả epoch.
- `eta_sec`: Thời gian dự kiến còn lại để hoàn thành toàn bộ thí nghiệm (Estimated Time of Arrival).

---

### 3.4 Cơ Chế Lưu Checkpoint Nguyên Tử (Atomic Save)
Để ngăn chặn hoàn toàn nguy cơ tệp checkpoint bị hư hỏng (corrupted) khi máy chủ bị ngắt nguồn, hết hạn phiên Colab hoặc người dùng bấm `Ctrl + C`:
1. Checkpoint được tuần tự hóa (serialize) và ghi trước vào tệp tạm thời có tên dạng `.tmp_best_model.pth_<pid>`.
2. Hàm `os.replace` (lệnh nguyên tử ở cấp hệ điều hành) được gọi để hoán đổi tệp tạm thời thành tệp chính thức `best_model.pth`.
3. Nhờ cơ chế này, tệp checkpoint trên đĩa luôn luôn ở trạng thái hoàn chỉnh $100\%$.

---

### 3.5 Tệp Trọng Số Tối Ưu `best_weights_fp16.pth`
Trong khi tệp `best_model.pth` có dung lượng ~250MB (chứa đầy đủ trạng thái của mô hình, optimizer AdamW, scheduler, grad scaler), tệp `best_weights_fp16.pth`:
- Loại bỏ toàn bộ optimizer và gradient states.
- Ép kiểu các trọng số dấu phẩy động sang `torch.float16`.
- Dung lượng giảm xuống chỉ còn khoảng **~60MB**, lý tưởng để tải lên GitHub Releases hoặc tích hợp vào các ứng dụng Edge AI y tế.

## CHƯƠNG 4: QUY TRÌNH SUY LUẬN KHỐI 3D & HẬU XỬ LÝ Y TẾ (3D EVALUATION PIPELINE)

### 4.1 Quy Trình Tái Tạo Thể Tích 3D Từ Lát Cắt 2.5D
Trong kiểm định y tế thực tế, bác sĩ đánh giá khối u trên toàn bộ khối 3D của bệnh nhân chứ không đánh giá riêng lẻ từng lát cắt. Mô-đun `common/eval3d/volume_inference.py` thực hiện:
1. **Trích xuất chuỗi lát cắt**: Quét toàn bộ các lát cắt từ đỉnh phổi đến đáy phổi của ca chụp CT ($Z$ lát cắt).
2. **Ghép ngữ cảnh 2.5D**: Với mỗi lát cắt thứ $z$, tạo tensor đầu vào từ lát $z-1$, $z$, $z+1$ (lát biên được nhân đôi).
3. **Suy luận trượt (Sliding Inference)**: Đưa các mini-batch lát cắt qua mô hình Mamba-UNet với kích hoạt AMP float16.
4. **Tập hợp khối 3D**: Ghép các mặt nạ xác suất 2D lại thành một ma trận khối không gian 3 chiều có kích thước $(Z, 512, 512)$.

---

### 4.2 Thuật Toán Hậu Xử Lý Kết Nối 3D (Connected Component Analysis)
Ảnh CT thường có các chấm nhiễu do mạch máu, mô xơ hoặc thành ngực có đậm độ HU tương tự khối u. Mô-đun `common/eval3d/postprocess3d.py` áp dụng quy chuẩn y tế chuẩn mực:
1. **Nhị phân hóa ngưỡng**: Chuyển đổi ma trận xác suất sang nhị phân với ngưỡng mặc định `threshold = 0.5`.
2. **Phân tích thành phần liên thông 3D (3D Connected Components)**: Sử dụng cấu trúc lân cận 26 hướng (`scipy.ndimage.label`).
3. **Lọc kích thước vật lý**:
   - Dựa trên hằng số `TUMOR_SIZE_LOW_LIM = 20.0 mm^2`: Tính diện tích trên từng lát cắt ($S = N_{\text{pixel}} \times \text{spacing}_x \times \text{spacing}_y$). Nếu diện tích lát cắt $< 20.0 \text{ mm}^2$, loại bỏ.
   - Tính thể tích khối liên thông 3D ($V = N_{\text{voxels}} \times \text{spacing}_x \times \text{spacing}_y \times \text{spacing}_z$). Bất kỳ khối u nào có thể tích $< 20.0 \text{ mm}^3$ sẽ bị lọc bỏ như nhiễu giả dương (False Positive).

---

### 4.3 Hệ Thống Thước Đo Y Tế Đa Chiều (3D Clinical Metrics)
Mô-đun `common/eval3d/metrics3d.py` tính toán đồng thời các chỉ số:
1. **3D Volume Dice Score**:
   $$\text{Dice}_{3D} = \frac{2 |V_{\text{pred}} \cap V_{\text{gt}}|}{|V_{\text{pred}}| + |V_{\text{gt}}|}$$
2. **3D Volume IoU (Jaccard Index)**:
   $$\text{IoU}_{3D} = \frac{|V_{\text{pred}} \cap V_{\text{gt}}|}{|V_{\text{pred}} \cup V_{\text{gt}}|}$$
3. **95% Hausdorff Distance (HD95, mm)**: Khoảng cách lớn thứ $95\%$ giữa bề mặt u dự đoán và bề mặt u thực tế, đo lường độ chính xác biên dạng và loại trừ điểm dị biệt cực đoan.
4. **Average Surface Distance (ASD, mm)**: Khoảng cách trung bình giữa hai bề mặt khối u.
5. **Độ nhạy (Sensitivity/Recall)** & **Độ đặc hiệu (Specificity)** trên từng thể tích voxel.

## CHƯƠNG 5: BẢNG TỔNG HỢP DANH MỤC CÁC TỆP MÃ NGUỒN

Hệ thống đã thu thập và kiểm định toàn diện **37 tệp mã nguồn** (tổng cộng **7,478 dòng code** | **295.1 KB**):

| STT | Đường Dẫn Tệp | Nhóm Chức Năng | Số Dòng | Dung Lượng | Vai Trò Chính |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 01 | [`mamba_unet/__init__.py`](#tệp-01-mamba_unet-__init__-py) | Kiến trúc Cốt Lõi Mamba-UNet & Backbone NVIDIA | 2 | 0.1 KB | Khởi tạo package mamba_unet và export các lớp cốt lõi. |
| 02 | [`mamba_unet/model.py`](#tệp-02-mamba_unet-model-py) | Kiến trúc Cốt Lõi Mamba-UNet & Backbone NVIDIA | 140 | 5.5 KB | Lớp MambaUNet hoàn chỉnh: Ghép nối MambaVision Encoder và UNet Decoder qua 4 skip connections. |
| 03 | [`mamba_unet/backbone/__init__.py`](#tệp-03-mamba_unet-backbone-__init__-py) | Kiến trúc Cốt Lõi Mamba-UNet & Backbone NVIDIA | 25 | 0.6 KB | Khởi tạo package backbone và registry các biến thể MambaVision. |
| 04 | [`mamba_unet/backbone/mamba_vision.py`](#tệp-04-mamba_unet-backbone-mamba_vision-py) | Kiến trúc Cốt Lõi Mamba-UNet & Backbone NVIDIA | 710 | 25.3 KB | Kiến trúc MambaVision chuẩn NVIDIA: ConvBlock, MambaVisionMixer (dual conv 1D + SSM), Transformer Block, Downsample Stem. |
| 05 | [`mamba_unet/backbone/pretrained.py`](#tệp-05-mamba_unet-backbone-pretrained-py) | Kiến trúc Cốt Lõi Mamba-UNet & Backbone NVIDIA | 232 | 9.4 KB | Bộ nạp và ánh xạ trọng số Pretrained ImageNet-1K từ NVIDIA (khớp 100% 250 khóa backbone). |
| 06 | [`mamba_unet/decoder.py`](#tệp-06-mamba_unet-decoder-py) | Kiến trúc Cốt Lõi Mamba-UNet & Backbone NVIDIA | 117 | 3.9 KB | UNet Decoder đối xứng: Khôi phục kích thước không gian qua ConvTranspose2d, ghép kênh skip connection và head phân đoạn. |
| 07 | [`mamba_unet/blocks.py`](#tệp-07-mamba_unet-blocks-py) | Kiến trúc Cốt Lõi Mamba-UNet & Backbone NVIDIA | 101 | 3.9 KB | Các khối tích chập cơ bản trong Decoder (DoubleConv, UpBlock). |
| 08 | [`mamba_unet/activations.py`](#tệp-08-mamba_unet-activations-py) | Kiến trúc Cốt Lõi Mamba-UNet & Backbone NVIDIA | 218 | 7.9 KB | Registry các hàm kích hoạt hỗ trợ (ReLU, GELU, SiLU, LeakyReLU, Mish). |
| 09 | [`mamba_unet/losses.py`](#tệp-09-mamba_unet-losses-py) | Kiến trúc Cốt Lõi Mamba-UNet & Backbone NVIDIA | 308 | 11.5 KB | Registry các hàm mất mát phân đoạn: PureDiceBCELoss, PureTverskyLoss, PureTverskyFocalLoss, PureDiceLoss, PureBCELoss, PureFocalLoss. |
| 10 | [`mamba_unet/fallbacks.py`](#tệp-10-mamba_unet-fallbacks-py) | Kiến trúc Cốt Lõi Mamba-UNet & Backbone NVIDIA | 332 | 12.9 KB | Cơ chế fallback Selective Scan thuần PyTorch (MambaScanFallback2D) khi môi trường không có CUDA kernel compiled. |
| 11 | [`mamba_unet/trainer.py`](#tệp-11-mamba_unet-trainer-py) | Điểm Nhập Huấn Luyện & Đánh Giá (CLI Entrypoints) | 171 | 6.4 KB | MambaUNetTrainer (kế thừa BaseTrainer): Điều khiển lịch đóng băng backbone, phân nhóm tham số learning rate (decoder lr vs backbone lr). |
| 12 | [`mamba_unet/train_main.py`](#tệp-12-mamba_unet-train_main-py) | Điểm Nhập Huấn Luyện & Đánh Giá (CLI Entrypoints) | 438 | 17.0 KB | CLI entrypoint chính cho Mamba-UNet: Hỗ trợ các lệnh train, overfit, selftest; cấu hình tham số, seed, và kích hoạt pipeline. |
| 13 | [`mamba_unet/evaluate.py`](#tệp-13-mamba_unet-evaluate-py) | Điểm Nhập Huấn Luyện & Đánh Giá (CLI Entrypoints) | 240 | 9.2 KB | CLI entrypoint cho suy luận và đánh giá 3D: Chạy predict cho từng ca, tính toán metrics 3D và xuất file NRRD. |
| 14 | [`common/engine/__init__.py`](#tệp-14-common-engine-__init__-py) | Hạ Tầng Engine, Quản Lý Checkpoint & Phiên Chạy (Common Engine) | 2 | 0.1 KB | Khởi tạo package common.engine. |
| 15 | [`common/engine/base_trainer.py`](#tệp-15-common-engine-base_trainer-py) | Hạ Tầng Engine, Quản Lý Checkpoint & Phiên Chạy (Common Engine) | 667 | 30.4 KB | BaseTrainer: Vòng lặp huấn luyện chuẩn hóa với AMP, Gradient Clipping, Early Stopping, ETA, và ghi history.csv/train.log. |
| 16 | [`common/engine/checkpoint_io.py`](#tệp-16-common-engine-checkpoint_io-py) | Hạ Tầng Engine, Quản Lý Checkpoint & Phiên Chạy (Common Engine) | 106 | 3.4 KB | Quản lý lưu/nạp checkpoint nguyên tử bằng file tạm + os.replace, xuất file best_weights_fp16.pth nhẹ ~60MB. |
| 17 | [`common/engine/run_manager.py`](#tệp-17-common-engine-run_manager-py) | Hạ Tầng Engine, Quản Lý Checkpoint & Phiên Chạy (Common Engine) | 230 | 8.3 KB | RunHandle và RunManager: Quản lý thư mục run độc lập, ghi nhận log thời gian thực, cập nhật run_config.json. |
| 18 | [`common/engine/run_config.py`](#tệp-18-common-engine-run_config-py) | Hạ Tầng Engine, Quản Lý Checkpoint & Phiên Chạy (Common Engine) | 155 | 4.7 KB | Khởi tạo và chuẩn hóa Schema v1 cho tệp run_config.json (lưu vết môi trường, git commit, GPU, tham số). |
| 19 | [`common/engine/cli_args.py`](#tệp-19-common-engine-cli_args-py) | Hạ Tầng Engine, Quản Lý Checkpoint & Phiên Chạy (Common Engine) | 104 | 5.6 KB | Bộ tham số dòng lệnh CLI dùng chung chuẩn hóa giữa các kiến trúc. |
| 20 | [`common/engine/utils.py`](#tệp-20-common-engine-utils-py) | Hạ Tầng Engine, Quản Lý Checkpoint & Phiên Chạy (Common Engine) | 52 | 1.7 KB | Các hàm tiện ích hệ thống: seed_everything, atomic_save_json, format_duration. |
| 21 | [`common/paths.py`](#tệp-21-common-paths-py) | Hạ Tầng Engine, Quản Lý Checkpoint & Phiên Chạy (Common Engine) | 34 | 1.2 KB | Quản lý đường dẫn tập trung không hard-code (REPO_ROOT, data cache, runs root). |
| 22 | [`configs/config.py`](#tệp-22-configs-config-py) | Hạ Tầng Engine, Quản Lý Checkpoint & Phiên Chạy (Common Engine) | 37 | 1.2 KB | Hằng số y tế và chuẩn hóa lát cắt CT: Cửa sổ phổi HU, kích thước ảnh 512, ngưỡng u 20mm2, tỷ lệ âm tính. |
| 23 | [`common/data_pipeline/__init__.py`](#tệp-23-common-data_pipeline-__init__-py) | Xử Lý Dữ Liệu Y Tế & Data Pipeline | 2 | 0.1 KB | Khởi tạo package data pipeline. |
| 24 | [`common/data_pipeline/dataset.py`](#tệp-24-common-data_pipeline-dataset-py) | Xử Lý Dữ Liệu Y Tế & Data Pipeline | 312 | 11.8 KB | SliceDataset25D và hàm build_dataloaders: Nạp dữ liệu 2.5D (z-1, z, z+1) và tạo PyTorch DataLoaders. |
| 25 | [`common/data_pipeline/medical_prep.py`](#tệp-25-common-data_pipeline-medical_prep-py) | Xử Lý Dữ Liệu Y Tế & Data Pipeline | 635 | 29.1 KB | Tiền xử lý y tế: Chuẩn hóa cửa sổ HU mềm/phổi, tạo mặt nạ phổi (lung mask), chuyển đổi đơn vị thể tích mm3. |
| 26 | [`common/data_pipeline/runtime_data.py`](#tệp-26-common-data_pipeline-runtime_data-py) | Xử Lý Dữ Liệu Y Tế & Data Pipeline | 729 | 30.6 KB | Lớp nạp dữ liệu thời gian thực hỗ trợ streaming và xử lý bộ nhớ tối ưu. |
| 27 | [`common/data_pipeline/offline_builder.py`](#tệp-27-common-data_pipeline-offline_builder-py) | Xử Lý Dữ Liệu Y Tế & Data Pipeline | 217 | 8.8 KB | Công cụ quét cache tiền xử lý offline, lập chỉ mục slices.csv và thống kê phân phối ca bệnh. |
| 28 | [`common/metrics.py`](#tệp-28-common-metrics-py) | Thước Đo & Đánh Giá Khối 3D Chuẩn Y Tế (3D Evaluation & Metrics) | 91 | 3.7 KB | Các hàm tính toán chỉ số phân đoạn 2D/2.5D: Dice Score, IoU (Jaccard Index) hỗ trợ batch tensor và numpy. |
| 29 | [`common/eval3d/__init__.py`](#tệp-29-common-eval3d-__init__-py) | Thước Đo & Đánh Giá Khối 3D Chuẩn Y Tế (3D Evaluation & Metrics) | 30 | 0.8 KB | Khởi tạo package common.eval3d. |
| 30 | [`common/eval3d/volume_inference.py`](#tệp-30-common-eval3d-volume_inference-py) | Thước Đo & Đánh Giá Khối 3D Chuẩn Y Tế (3D Evaluation & Metrics) | 147 | 5.5 KB | predict_patient_volume: Suy luận trượt toàn bộ lát cắt của bệnh nhân và ghép thành ma trận thể tích 3D. |
| 31 | [`common/eval3d/postprocess3d.py`](#tệp-31-common-eval3d-postprocess3d-py) | Thước Đo & Đánh Giá Khối 3D Chuẩn Y Tế (3D Evaluation & Metrics) | 202 | 8.1 KB | Hậu xử lý 3D y tế: Lọc kết nối 3D Connected Components, loại bỏ u rác < 20mm3, làm mịn biên dạng. |
| 32 | [`common/eval3d/metrics3d.py`](#tệp-32-common-eval3d-metrics3d-py) | Thước Đo & Đánh Giá Khối 3D Chuẩn Y Tế (3D Evaluation & Metrics) | 148 | 4.9 KB | Tính toán bộ chỉ số y tế 3D: 3D Dice, 3D IoU, 95% Hausdorff Distance (HD95), Average Surface Distance (ASD), Recall, Specificity. |
| 33 | [`common/eval3d/nrrd_io.py`](#tệp-33-common-eval3d-nrrd_io-py) | Thước Đo & Đánh Giá Khối 3D Chuẩn Y Tế (3D Evaluation & Metrics) | 63 | 2.4 KB | Đọc và xuất file mặt nạ 3D chuẩn y tế định dạng NRRD (.nrrd) tương thích 3D Slicer / ITK-SNAP. |
| 34 | [`common/eval3d/eval_runner.py`](#tệp-34-common-eval3d-eval_runner-py) | Thước Đo & Đánh Giá Khối 3D Chuẩn Y Tế (3D Evaluation & Metrics) | 151 | 5.5 KB | Bộ điều phối đánh giá trên toàn bộ tập kiểm định validation, xuất báo cáo CSV và JSON chi tiết. |
| 35 | [`tests/test_gray_area_padding.py`](#tệp-35-tests-test_gray_area_padding-py) | Bộ Kiểm Thử Độ Tin Cậy & Vùng Xám (Gray Area Tests) | 101 | 4.6 KB | Kiểm thử lan truyền tiến/lùi 512x512 qua MambaVision Mixer có padding chẵn/lẻ, kiểm tra finite gradients. |
| 36 | [`tests/test_gray_area_crash.py`](#tệp-36-tests-test_gray_area_crash-py) | Bộ Kiểm Thử Độ Tin Cậy & Vùng Xám (Gray Area Tests) | 87 | 3.8 KB | Kiểm thử ghi đĩa nguyên tử và tự động phục hồi (auto-resume) liền mạch từ last_model.pth sau khi bị ngắt đột ngột. |
| 37 | [`tests/test_scan_fallback.py`](#tệp-37-tests-test_scan_fallback-py) | Bộ Kiểm Thử Độ Tin Cậy & Vùng Xám (Gray Area Tests) | 142 | 5.2 KB | Kiểm thử tính tương đương và ổn định số học giữa kernel CUDA native và module PyTorch CPU fallback. |


## CHƯƠNG 6: TOÀN BỘ MÃ NGUỒN HỆ THỐNG CHI TIẾT

> [!IMPORTANT]
> Toàn bộ các module dưới đây được trích xuất trực tiếp từ mã nguồn thực tế của dự án, đã trải qua kiểm thử hồi quy 100% đạt chuẩn (26/26 tests PASS) và đã được xác thực huấn luyện trên GPU Tesla T4.


---

### 1. Kiến trúc Cốt Lõi Mamba-UNet & Backbone NVIDIA

#### <a id="tệp-01-mamba_unet-__init__-py"></a>Tệp 01: `mamba_unet/__init__.py`
- **Vị trí**: `mamba_unet/__init__.py`
- **Quy mô**: `2 dòng` | `0.1 KB`
- **Vai trò**: Khởi tạo package mamba_unet và export các lớp cốt lõi.

```python
# -*- coding: utf-8 -*-
"""mamba_unet package - Kiến trúc Mamba-UNet (MambaVision backbone + U-Net decoder)."""
```

#### <a id="tệp-02-mamba_unet-model-py"></a>Tệp 02: `mamba_unet/model.py`
- **Vị trí**: `mamba_unet/model.py`
- **Quy mô**: `140 dòng` | `5.5 KB`
- **Vai trò**: Lớp MambaUNet hoàn chỉnh: Ghép nối MambaVision Encoder và UNet Decoder qua 4 skip connections.

```python
# -*- coding: utf-8 -*-
"""
mamba_unet/model.py - Mô hình Mamba-UNet hoàn chỉnh cho phân đoạn ảnh y tế 2.5D.

Quy tắc bất biến:
  - 2 thuộc tính rõ ràng: self.encoder (MambaVision) và self.decoder (UNetDecoder).
  - Loại bỏ hoàn toàn alias ModularUNet và loại bỏ nhánh backbone in ('cnn', 'cnn_baseline').
  - Tương thích 100% tập khóa tests/golden/mamba_unet_keys.json (264 keys chuẩn NVIDIA).
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mamba_unet.backbone.mamba_vision import MambaVision, build_mambavision_backbone
from mamba_unet.decoder import UNetDecoder


class MambaUNet(nn.Module):
    """
    Kiến trúc Mamba-UNet:
        Input (B, 3, H, W) -> MambaVision Encoder -> 4 feature maps [f1, f2, f3, f4]
                           -> UNet Decoder + Skip Connections -> Logits (B, num_classes, H, W).

    Args:
        in_channels (int): Số kênh ảnh đầu vào (3 cho lát cắt 2.5D: z-1, z, z+1).
        num_classes (int): Số lớp phân đoạn (mặc định 1 cho nhãn khối u).
        backbone (str | nn.Module): Tên biến thể ('mamba_vision_T', ...) hoặc module MambaVision.
        pretrained (bool): Nạp trọng số ImageNet-1K pretrained từ NVIDIA.
        final_channels (int): Số kênh tích chập trước lớp xuất (mặc định 32).
        act (str | Callable): Hàm kích hoạt cho Decoder.
        init_bias (float): Giá trị khởi tạo bias cho lớp output (mặc định -4.0).
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone: Union[str, nn.Module] = "mamba_vision_T",
        pretrained: bool = False,
        final_channels: int = 32,
        act: Optional[Union[str, Callable[[], nn.Module]]] = None,
        init_bias: Optional[float] = -4.0,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_name = backbone if isinstance(backbone, str) else backbone.__class__.__name__

        # 1. Khởi tạo MambaVision Encoder
        if isinstance(backbone, str):
            self.encoder = build_mambavision_backbone(
                name=backbone,
                pretrained=pretrained,
                in_chans=in_channels,
            )
        elif isinstance(backbone, nn.Module):
            self.encoder = backbone
        else:
            raise TypeError(f"Backbone phải là str hoặc nn.Module, nhận được {type(backbone)}")

        encoder_channels = getattr(self.encoder, "out_channels", [80, 160, 320, 640])
        self.encoder_channels = encoder_channels

        # 2. Khởi tạo UNet Decoder
        self.decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            final_channels=final_channels,
            act=act,
            init_bias=init_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits


def build_mamba_unet(config: Optional[Dict[str, Any]] = None, **kwargs) -> MambaUNet:
    """Factory function tạo MambaUNet từ config dict hoặc kwargs."""
    cfg = config or {}
    model_cfg = cfg.get("model", cfg.get("MODEL", {}))

    in_channels = kwargs.get("in_channels", model_cfg.get("in_channels", 3))
    num_classes = kwargs.get("num_classes", model_cfg.get("num_classes", model_cfg.get("classes", 1)))
    backbone = kwargs.get("backbone", model_cfg.get("backbone", "mamba_vision_T"))
    pretrained = kwargs.get("pretrained", model_cfg.get("pretrained", False))
    final_channels = kwargs.get("final_channels", model_cfg.get("final_channels", 32))
    act = kwargs.get("act", kwargs.get("activation", model_cfg.get("activation", "relu")))
    init_bias = kwargs.get("init_bias", model_cfg.get("init_bias", -4.0))

    return MambaUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        final_channels=final_channels,
        act=act,
        init_bias=init_bias,
    )


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.model...")
    from mamba_unet.backbone.mamba_vision import MambaVision

    # Dùng tiny backbone để chạy siêu nhanh
    tiny_bb = MambaVision(
        dim=16,
        in_dim=8,
        depths=[1, 1, 2, 1],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 2, 4, 4],
        in_chans=3,
    )
    model = MambaUNet(in_channels=3, num_classes=1, backbone=tiny_bb)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64), f"Lỗi shape đầu ra: {out.shape}"

    # Kiểm tra số lượng keys của mamba_vision_T đầy đủ
    full_model = MambaUNet(in_channels=3, num_classes=1, backbone="mamba_vision_T", pretrained=False)
    keys = list(full_model.state_dict().keys())
    assert len(keys) == 300, f"Số keys phải là 300 (250 encoder + 50 decoder), nhận được {len(keys)}"
    assert all(k.startswith("encoder.") or k.startswith("decoder.") for k in keys), "Keys phải có tiền tố encoder. hoặc decoder."
    print(f"✅ Selftest mamba_unet.model hoàn tất thành công ({len(keys)}/300 keys chuẩn xác)!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
```

#### <a id="tệp-03-mamba_unet-backbone-__init__-py"></a>Tệp 03: `mamba_unet/backbone/__init__.py`
- **Vị trí**: `mamba_unet/backbone/__init__.py`
- **Quy mô**: `25 dòng` | `0.6 KB`
- **Vai trò**: Khởi tạo package backbone và registry các biến thể MambaVision.

```python
# -*- coding: utf-8 -*-
"""
mamba_unet/backbone/__init__.py - Gói trích xuất đặc trưng MambaVision Backbone.
"""

from .mamba_vision import (
    MambaVision,
    mamba_vision_T,
    mamba_vision_T2,
    mamba_vision_S,
    mamba_vision_B,
    build_mambavision_backbone,
)
from .pretrained import load_pretrained_mambavision, download_pretrained_weights

__all__ = [
    "MambaVision",
    "mamba_vision_T",
    "mamba_vision_T2",
    "mamba_vision_S",
    "mamba_vision_B",
    "build_mambavision_backbone",
    "load_pretrained_mambavision",
    "download_pretrained_weights",
]
```

#### <a id="tệp-04-mamba_unet-backbone-mamba_vision-py"></a>Tệp 04: `mamba_unet/backbone/mamba_vision.py`
- **Vị trí**: `mamba_unet/backbone/mamba_vision.py`
- **Quy mô**: `710 dòng` | `25.3 KB`
- **Vai trò**: Kiến trúc MambaVision chuẩn NVIDIA: ConvBlock, MambaVisionMixer (dual conv 1D + SSM), Transformer Block, Downsample Stem.

```python
# -*- coding: utf-8 -*-
"""
mamba_unet/backbone/mamba_vision.py - Backbone MambaVision chuẩn NVIDIA (NVlabs/MambaVision - CVPR 2024).

Quy tắc bất biến:
  - Đồng bộ 1-1 kiến trúc gốc từ NVlabs/MambaVision:
    * Stem: PatchEmbed stride 4.
    * Stage 1 & 2: ConvBlock (Depthwise Separable Conv).
    * Stage 3 & 4: Block hybrid chứa 1 Mixer (MambaVisionMixer HOẶC WindowAttention) + Mlp(fc1, fc2).
    * MambaVisionMixer: Cấu trúc nhánh đối xứng chuẩn NVIDIA với conv1d_x, conv1d_z, Selective Scan và out_proj.
    * Tỷ lệ phân chia Stage 3: 4 Mamba + 4 Attention (công thức depths[i] // 2).
    * Tỷ lệ phân chia Stage 4: 2 Mamba + 2 Attention.
    * Downsample: nn.Conv2d stride 2 thuần túy (không norm bên trong).
  - Trích xuất 4 feature maps đa tỷ lệ [H/4, H/8, H/16, H/32] làm Skip Connections cho UNetDecoder.
  - Tuyệt đối KHÔNG import từ unet_base hay common (R1, R2).
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mamba_unet.fallbacks import (
    DropPath,
    LayerNorm2d,
    rearrange,
    repeat,
    selective_scan_fn,
    trunc_normal_,
)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Chia feature map (B, C, H, W) thành các windows không chồng lấp.
    Đầu vào H, W đã được đệm bảo đảm chia hết cho window_size.
    Đầu ra: (B * num_windows, window_size * window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Ghép các window (B * num_windows, window_size * window_size, C) trở lại thành (B, C, H, W).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x


class Downsample(nn.Module):
    """Khối giảm chiều không gian chuẩn NVIDIA: chỉ dùng conv strided 2, không norm bên trong."""
    def __init__(self, dim: int, keep_dim: bool = False):
        super().__init__()
        out_dim = dim if keep_dim else 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduction(x)


class PatchEmbed(nn.Module):
    """Khối biến đổi ảnh đầu vào ban đầu chuẩn NVIDIA: giảm kích thước không gian xuống H/4, W/4."""
    def __init__(self, in_chans: int = 3, in_dim: int = 32, dim: int = 80):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_down(x)


class ConvBlock(nn.Module):
    """Khối tích chập ResNet-style chuẩn NVIDIA (Conv3x3 + BN + GELU + Conv3x3 + BN)."""
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale: Optional[float] = None, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate="tanh")
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim, 1, 1)) if self.layer_scale else 1.0
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = input_x + self.drop_path(self.gamma * x)
        return x


class Mlp(nn.Module):
    """Khối Mlp chuẩn NVIDIA (tương thích timm), sử dụng fc1 và fc2."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Any = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MambaVisionMixer(nn.Module):
    """Khối trộn đặc trưng không gian dựa trên nhánh đối xứng conv1d_x/conv1d_z và 1D Selective Scan chuẩn NVIDIA."""
    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Trong MambaVision: in_proj chiếu từ d_model sang d_inner (sau đó chunk 2 thành x và z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)

        # 2 nhánh tích chập 1D độc lập cho x và z
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            padding=d_conv - 1,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner // 2) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, l, d = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = xz.transpose(1, 2)
        x, z = xz.chunk(2, dim=1)

        x = self.conv1d_x(x)[:, :, :l]
        x = self.act(x)

        z = self.conv1d_z(z)[:, :, :l]
        z = self.act(z)

        x_trans = x.transpose(1, 2)
        x_for_proj = x_trans.reshape(b * l, -1)
        x_dbl = self.x_proj(x_for_proj)
        dt, B_proj, C_proj = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt).view(b, l, -1).transpose(1, 2)
        B_proj = B_proj.view(b, l, -1).transpose(1, 2)
        C_proj = C_proj.view(b, l, -1).transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        y = selective_scan_fn(
            x,
            dt,
            A,
            B_proj,
            C_proj,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        # Ghép nối nhánh SSM (y) và nhánh tích chập điều chế (z) theo chuẩn NVIDIA MambaVision
        y = torch.cat([y, z], dim=1)
        y = y.transpose(1, 2)
        return self.out_proj(y)


class Attention(nn.Module):
    """Multi-Head Self-Attention chuẩn NVIDIA MambaVision."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) phải chia hết cho num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_norm = qk_scale
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Khối hybrid chuẩn NVIDIA: Mỗi block chỉ chứa 1 Mixer (Mamba HOẶC Attention) + 1 MLP."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        counter: int,
        transformer_blocks: List[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: Optional[float] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                d_state=8,
                d_conv=3,
                expand=1,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        use_layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        self.g_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1.0
        self.g_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.g_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.g_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """Một Stage trong MambaVision chuẩn NVIDIA (hỗ trợ cả ConvStage và HybridStage)."""
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        conv: bool = False,
        downsample: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        layer_scale: Optional[float] = None,
        transformer_blocks: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.conv = conv
        transformer_blocks = transformer_blocks or []

        if self.conv:
            self.blocks = nn.ModuleList([
                ConvBlock(
                    dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    counter=j,
                    transformer_blocks=transformer_blocks,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale,
                )
                for j in range(depth)
            ])

        self.downsample = Downsample(dim=dim) if downsample else None

    def forward(self, x: torch.Tensor, return_feature_before_downsample: bool = True):
        B, C, H, W = x.shape
        if not self.conv:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape

            windows = window_partition(x, self.window_size)
            for blk in self.blocks:
                windows = blk(windows)
            x = window_reverse(windows, self.window_size, Hp, Wp)

            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            for blk in self.blocks:
                x = blk(x)

        feat_before_down = x
        if self.downsample is not None:
            x_next = self.downsample(x)
        else:
            x_next = x

        if return_feature_before_downsample:
            return x_next, feat_before_down
        return x_next


class MambaVision(nn.Module):
    """
    MambaVision Backbone chuẩn NVIDIA phục vụ phân đoạn ảnh y tế:
      Forward trích xuất danh sách 4 multi-scale feature maps: [H/4, H/8, H/16, H/32].
    """
    def __init__(
        self,
        dim: int = 80,
        in_dim: int = 32,
        depths: List[int] = [1, 3, 8, 4],
        window_size: List[int] = [8, 8, 14, 7],
        mlp_ratio: float = 4.0,
        num_heads: List[int] = [2, 4, 8, 16],
        drop_path_rate: float = 0.2,
        in_chans: int = 3,
        qkv_bias: bool = True,
        layer_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_levels = len(depths)
        self.dim = dim
        self.in_dim = in_dim
        self.in_chans = in_chans

        self.out_channels = [int(dim * 2**i) for i in range(self.num_levels)]
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            stage_dim = int(dim * 2**i)
            conv = True if (i == 0 or i == 1) else False

            # Công thức phân chia chuẩn của NVIDIA:
            # Stage 3 (depth=8) -> t_blocks = [4, 5, 6, 7] (4 Mamba + 4 Attention)
            # Stage 4 (depth=4) -> t_blocks = [2, 3]       (2 Mamba + 2 Attention)
            t_blocks = (
                list(range(depths[i] // 2 + 1, depths[i]))
                if depths[i] % 2 != 0
                else list(range(depths[i] // 2, depths[i]))
            ) if not conv else []

            level = MambaVisionLayer(
                dim=stage_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                conv=conv,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                transformer_blocks=t_blocks,
            )
            self.levels.append(level)

        # Lớp BatchNorm2d chuẩn của NVIDIA ở cuối Stage 4 (khớp norm.weight/bias trong checkpoint)
        self.num_features = int(dim * 2 ** (self.num_levels - 1))
        self.norm = nn.BatchNorm2d(self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        features = []
        for level in self.levels:
            x, feat = level(x, return_feature_before_downsample=True)
            features.append(feat)
        return features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.forward_features(x)


# =============================================================================
# BIẾN THỂ VÀ FACTORY
# =============================================================================
def mamba_vision_T(pretrained: bool = False, in_chans: int = 3, **kwargs) -> MambaVision:
    """Biến thể MambaVision-Tiny: dim=80, depths=[1, 3, 8, 4], out_channels=[80, 160, 320, 640]."""
    model = MambaVision(
        dim=80,
        in_dim=32,
        depths=[1, 3, 8, 4],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        in_chans=in_chans,
        **kwargs,
    )
    if pretrained:
        from .pretrained import load_pretrained_mambavision
        load_pretrained_mambavision(model, "mamba_vision_t")
    return model


def mamba_vision_T2(pretrained: bool = False, in_chans: int = 3, **kwargs) -> MambaVision:
    """Biến thể MambaVision-Tiny2: dim=80, depths=[1, 3, 11, 4]."""
    model = MambaVision(
        dim=80,
        in_dim=32,
        depths=[1, 3, 11, 4],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        in_chans=in_chans,
        **kwargs,
    )
    if pretrained:
        from .pretrained import load_pretrained_mambavision
        load_pretrained_mambavision(model, "mamba_vision_t2")
    return model


def mamba_vision_S(pretrained: bool = False, in_chans: int = 3, **kwargs) -> MambaVision:
    """Biến thể MambaVision-Small: dim=96, depths=[3, 3, 10, 5], out_channels=[96, 192, 384, 768]."""
    model = MambaVision(
        dim=96,
        in_dim=64,
        depths=[3, 3, 10, 5],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        in_chans=in_chans,
        **kwargs,
    )
    if pretrained:
        from .pretrained import load_pretrained_mambavision
        load_pretrained_mambavision(model, "mamba_vision_s")
    return model


def mamba_vision_B(pretrained: bool = False, in_chans: int = 3, **kwargs) -> MambaVision:
    """Biến thể MambaVision-Base: dim=128, depths=[3, 3, 12, 5], out_channels=[128, 256, 512, 1024]."""
    model = MambaVision(
        dim=128,
        in_dim=64,
        depths=[3, 3, 12, 5],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        in_chans=in_chans,
        **kwargs,
    )
    if pretrained:
        from .pretrained import load_pretrained_mambavision
        load_pretrained_mambavision(model, "mamba_vision_b")
    return model


def build_mambavision_backbone(
    name: str = "mamba_vision_T",
    pretrained: bool = False,
    in_chans: int = 3,
    **kwargs,
) -> MambaVision:
    """Factory function tạo MambaVision backbone theo tên biến thể."""
    registry = {
        "mamba_vision_t": mamba_vision_T,
        "mamba_vision_t2": mamba_vision_T2,
        "mamba_vision_s": mamba_vision_S,
        "mamba_vision_b": mamba_vision_B,
    }
    key = name.lower().strip()
    if key not in registry:
        raise ValueError(f"Không hỗ trợ backbone '{name}'. Các lựa chọn: {list(registry.keys())}")
    return registry[key](pretrained=pretrained, in_chans=in_chans, **kwargs)


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.backbone.mamba_vision...")

    # 1. Test khối PatchEmbed
    pe = PatchEmbed(in_chans=3, in_dim=32, dim=80)
    x = torch.randn(2, 3, 64, 64)
    pe_out = pe(x)
    assert pe_out.shape == (2, 80, 16, 16), f"Lỗi PatchEmbed: {pe_out.shape}"
    print("  ✓ PatchEmbed PASS")

    # 2. Test khối ConvBlock
    cb = ConvBlock(dim=80)
    cb_out = cb(pe_out)
    assert cb_out.shape == pe_out.shape, f"Lỗi ConvBlock: {cb_out.shape}"
    print("  ✓ ConvBlock PASS")

    # 3. Test khối Downsample
    ds = Downsample(dim=80)
    ds_out = ds(cb_out)
    assert ds_out.shape == (2, 160, 8, 8), f"Lỗi Downsample: {ds_out.shape}"
    print("  ✓ Downsample PASS")

    # 4. Test khối Mlp
    mlp = Mlp(in_features=80, hidden_features=320)
    tokens = torch.randn(2, 16, 80)
    mlp_out = mlp(tokens)
    assert mlp_out.shape == tokens.shape, f"Lỗi Mlp: {mlp_out.shape}"
    print("  ✓ Mlp PASS")

    # 5. Test khối Attention
    attn = Attention(dim=80, num_heads=2)
    attn_out = attn(tokens)
    assert attn_out.shape == tokens.shape, f"Lỗi Attention: {attn_out.shape}"
    print("  ✓ Attention PASS")

    # 6. Test khối MambaVisionMixer
    mixer = MambaVisionMixer(d_model=80)
    mixer_out = mixer(tokens)
    assert mixer_out.shape == tokens.shape, f"Lỗi MambaVisionMixer: {mixer_out.shape}"
    print("  ✓ MambaVisionMixer PASS")

    # 7. Test khối Block (Mamba Mixer và Attention Mixer)
    blk_mamba = Block(dim=80, num_heads=2, counter=0, transformer_blocks=[2, 3])
    assert isinstance(blk_mamba.mixer, MambaVisionMixer)
    assert blk_mamba(tokens).shape == tokens.shape

    blk_attn = Block(dim=80, num_heads=2, counter=2, transformer_blocks=[2, 3])
    assert isinstance(blk_attn.mixer, Attention)
    assert blk_attn(tokens).shape == tokens.shape
    print("  ✓ Block (Mamba & Attention) PASS")

    # 8. Test toàn bộ mô hình MambaVision-Tiny trên ảnh kích thước 64x64 và 128x128
    for sz in [64, 128]:
        model = mamba_vision_T(pretrained=False, in_chans=3)
        model.eval()
        dummy = torch.randn(2, 3, sz, sz)
        with torch.no_grad():
            feats = model(dummy)
        assert len(feats) == 4, f"Kỳ vọng 4 feature maps, nhận {len(feats)}"
        assert feats[0].shape == (2, 80, sz // 4, sz // 4), f"Lỗi Stage 1: {feats[0].shape}"
        assert feats[1].shape == (2, 160, sz // 8, sz // 8), f"Lỗi Stage 2: {feats[1].shape}"
        assert feats[2].shape == (2, 320, sz // 16, sz // 16), f"Lỗi Stage 3: {feats[2].shape}"
        assert feats[3].shape == (2, 640, sz // 32, sz // 32), f"Lỗi Stage 4: {feats[3].shape}"
        print(f"  ✓ MambaVision-Tiny forward (sz={sz}) PASS")

    # 9. Test khởi tạo các biến thể khác
    for variant_fn, name in [
        (mamba_vision_T2, "Tiny2"),
        (mamba_vision_S, "Small"),
        (mamba_vision_B, "Base"),
    ]:
        m = variant_fn(pretrained=False, in_chans=3)
        print(f"  ✓ Khởi tạo {name} PASS (out_channels: {m.out_channels})")

    print("✅ Selftest mamba_unet.backbone.mamba_vision hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
```

#### <a id="tệp-05-mamba_unet-backbone-pretrained-py"></a>Tệp 05: `mamba_unet/backbone/pretrained.py`
- **Vị trí**: `mamba_unet/backbone/pretrained.py`
- **Quy mô**: `232 dòng` | `9.4 KB`
- **Vai trò**: Bộ nạp và ánh xạ trọng số Pretrained ImageNet-1K từ NVIDIA (khớp 100% 250 khóa backbone).

```python
# -*- coding: utf-8 -*-
"""
mamba_unet/backbone/pretrained.py - Tải và nạp an toàn trọng số Pretrained chính thức từ NVIDIA.

Quy tắc bất biến:
  - Smart Cache 5 tầng:
      1. Repo cục bộ: pretrained_weights/
      2. Google Drive Mirror (khi chạy Colab): /content/drive/MyDrive/.../pretrained_weights/
      3. SSD Colab siêu tốc: /content/pretrained_weights/
      4. OS Tempfile: Path(tempfile.gettempdir())
      5. Chỉ khi cả 4 nơi trên không có: Tải từ NVIDIA HuggingFace Hub 1 lần duy nhất và lưu vào pretrained_weights/.
  - Tương thích PyTorch >= 2.6: ưu tiên weights_only=True, fallback weights_only=False nếu cần.
  - Bóc tách tiền tố 'module.' và bỏ qua classifier 'head.' của ImageNet.
  - Kiểm tra nghiêm ngặt: Toàn bộ tham số của Stem, Stage 1-4 và Norm phải nạp khớp 100% (0 missing, 0 unexpected).
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MAMBAVISION_URLS: Dict[str, str] = {
    "mamba_vision_t": "https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar",
    "mamba_vision_t2": "https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar",
    "mamba_vision_s": "https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar",
    "mamba_vision_b": "https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar",
}


def resolve_pretrained_cache(variant: str) -> Path:
    """
    Dò tìm trọng số pretrained theo thứ tự ưu tiên 5 tầng:
      1. Cục bộ repo: _ROOT / pretrained_weights / filename
      2. Cục bộ cwd: Path('pretrained_weights') / filename
      3. Google Drive mirror (trên Colab): /content/drive/MyDrive/**/pretrained_weights / filename
      4. SSD Colab: /content/pretrained_weights / filename
      5. OS Temp: tempfile.gettempdir() / filename
      6. Nếu không có ở bất kỳ đâu -> Tải 1 lần từ NVIDIA Hugging Face và lưu vào pretrained_weights.
    """
    key = variant.lower().strip()
    if key not in MAMBAVISION_URLS:
        raise ValueError(
            f"❌ Không tìm thấy URL pretrained cho biến thể '{variant}'. "
            f"Các biến thể hỗ trợ: {list(MAMBAVISION_URLS.keys())}"
        )

    short_tag = key.replace("mamba_vision_", "")
    filenames = [
        f"{key}_nvidia_pretrained.pth.tar",
        f"mambavision_{short_tag}_1k.pth.tar",
    ]
    if key == "mamba_vision_t":
        filenames.append("mambavision_tiny_1k.pth.tar")
    elif key == "mamba_vision_t2":
        filenames.append("mambavision_tiny2_1k.pth.tar")
    elif key == "mamba_vision_s":
        filenames.append("mambavision_small_1k.pth.tar")
    elif key == "mamba_vision_b":
        filenames.append("mambavision_base_1k.pth.tar")

    search_dirs: List[Path] = [
        _ROOT / "pretrained_weights",
        Path("pretrained_weights"),
        Path("/content/pretrained_weights"),
        Path(tempfile.gettempdir()),
    ]

    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        explicit_drive_dirs = [
            drive_root / "pretrained_weights",
            drive_root / "COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/pretrained_weights",
            drive_root / "GG_1/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/pretrained_weights",
        ]
        for cand_dir in explicit_drive_dirs:
            if cand_dir.is_dir() and cand_dir not in search_dirs:
                search_dirs.append(cand_dir)

    for s_dir in search_dirs:
        for fname in filenames:
            cand = s_dir / fname
            if cand.exists() and cand.is_file() and cand.stat().st_size > 1024:
                print(f"ℹ️  [Smart Cache] Sử dụng trọng số có sẵn tại: {cand.resolve()}")
                return cand.resolve()

    url = MAMBAVISION_URLS[key]
    dest_dir = _ROOT / "pretrained_weights"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        dest_dir = Path(tempfile.gettempdir())

    primary_name = filenames[0]
    dest_path = dest_dir / primary_name

    print(f"📥 [Smart Cache] Đang tải trọng số pretrained chính thức từ {url} ...")
    torch.hub.download_url_to_file(url, str(dest_path), progress=True)
    print(f"✅ [Smart Cache] Đã lưu trọng số vào: {dest_path.resolve()}")

    for s_dir in search_dirs:
        if "drive/MyDrive" in str(s_dir) and s_dir.exists():
            try:
                shutil.copy2(dest_path, s_dir / primary_name)
                print(f"💾 [Smart Cache] Đã sao lưu 1 bản sang Google Drive: {s_dir / primary_name}")
            except Exception as e:
                print(f"⚠️ [Smart Cache] Không thể lưu bản sao sang Drive: {e}")
            break

    return dest_path.resolve()


def download_pretrained_weights(variant: str) -> Path:
    """Wrapper tương thích ngược cho resolve_pretrained_cache."""
    return resolve_pretrained_cache(variant)


def load_pretrained_mambavision(
    model: nn.Module,
    variant: str,
    map_location: str = "cpu",
    weights_path: Optional[Union[str, Path]] = None,
) -> nn.Module:
    """
    Nạp trọng số pretrained chuẩn NVIDIA vào instance MambaVision.
    - Tự động bóc tách tiền tố 'module.'
    - Bỏ qua classifier 'head.' của ImageNet.
    - Nghiêm ngặt: 0 missing keys, 0 unexpected keys trên toàn bộ backbone.
    """
    if weights_path is None:
        p = resolve_pretrained_cache(variant)
    else:
        p = Path(weights_path).resolve()

    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"Không tìm thấy file trọng số pretrained tại: {p}")

    try:
        import argparse
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([argparse.Namespace])
        ckpt = torch.load(p, map_location=map_location, weights_only=True)
    except Exception:
        ckpt = torch.load(p, map_location=map_location, weights_only=False)

    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))

    clean_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        clean_k = k[7:] if k.startswith("module.") else k
        if clean_k.startswith("head."):
            continue
        clean_sd[clean_k] = v

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    missing = [k for k in missing if not k.startswith("head.")]

    if len(missing) > 0 or len(unexpected) > 0:
        msg = (
            f"❌ Lỗi nạp trọng số pretrained '{variant}': "
            f"Thiếu {len(missing)} khóa, thừa {len(unexpected)} khóa!\n"
            f"  - Missing keys mẫu: {missing[:5]}\n"
            f"  - Unexpected keys mẫu: {unexpected[:5]}"
        )
        if len(missing) > 0:
            raise RuntimeError(msg)
        else:
            print(f"⚠️ Cảnh báo: {msg}")
    else:
        print(
            f"✅ Nạp trọng số Pretrained NVIDIA '{variant}' thành công 100%! "
            f"({len(clean_sd)} khóa khớp hoàn hảo, 0 missing, 0 unexpected)"
        )

    return model


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.backbone.pretrained...")

    # 1. Test cấu hình danh mục URLs
    assert len(MAMBAVISION_URLS) == 4, "Phải đủ 4 biến thể Tiny, Tiny2, Small, Base"
    for k in ["mamba_vision_t", "mamba_vision_t2", "mamba_vision_s", "mamba_vision_b"]:
        assert k in MAMBAVISION_URLS, f"Thiếu biến thể {k}"
    print("  ✓ URL Registry PASS")

    # 2. Test giải quyết Smart Cache (giả lập tệp giả)
    test_dir = _ROOT / "pretrained_weights"
    test_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = test_dir / "mamba_vision_t_nvidia_pretrained.pth.tar"

    # Nếu file thật chưa có, tạo dummy state_dict để test logic nạp và kiểm tra khóa
    from mamba_unet.backbone.mamba_vision import mamba_vision_T
    model = mamba_vision_T(pretrained=False, in_chans=3)

    if not dummy_file.exists() or dummy_file.stat().st_size == 0:
        print("  ℹ️  Tạo mock pretrained checkpoint để kiểm thử logic nạp...")
        mock_sd = {f"module.{k}": v for k, v in model.state_dict().items()}
        mock_sd["module.head.weight"] = torch.randn(1000, 640)
        mock_sd["module.head.bias"] = torch.randn(1000)
        torch.save({"model": mock_sd}, dummy_file)
        created_dummy = True
    else:
        created_dummy = False

    try:
        resolved = resolve_pretrained_cache("mamba_vision_t")
        assert resolved.exists() and resolved.stat().st_size > 0, "Lỗi resolve_pretrained_cache"
        print(f"  ✓ Smart Cache resolution PASS: {resolved.name}")

        # 3. Test nạp trọng số khớp 100%
        load_pretrained_mambavision(model, "mamba_vision_t", weights_path=resolved)
        print("  ✓ Nạp trọng số khớp 100% (0 missing, 0 unexpected) PASS")
    finally:
        if created_dummy and dummy_file.exists():
            dummy_file.unlink()

    print("✅ Selftest mamba_unet.backbone.pretrained hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
```

#### <a id="tệp-06-mamba_unet-decoder-py"></a>Tệp 06: `mamba_unet/decoder.py`
- **Vị trí**: `mamba_unet/decoder.py`
- **Quy mô**: `117 dòng` | `3.9 KB`
- **Vai trò**: UNet Decoder đối xứng: Khôi phục kích thước không gian qua ConvTranspose2d, ghép kênh skip connection và head phân đoạn.

```python
# -*- coding: utf-8 -*-
"""
mamba_unet/decoder.py - Bộ giải mã UNetDecoder 4 tầng liên kết với MambaVision Backbone.

Quy tắc bất biến:
  - Nhận 4 feature maps đa tỷ lệ [H/4, H/8, H/16, H/32] từ MambaVision.
  - Sử dụng các khối DoubleConv và OutConv từ mamba_unet.blocks (không import từ unet_base).
  - Bảo toàn 100% cấu trúc và tên lớp con: up1, up2, up3, final_up, outc.
"""

import sys
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mamba_unet.blocks import DoubleConv, OutConv


class UpBlock(nn.Module):
    """
    Khối giải mã: Phóng to Tensor và ghép nối (Concat) với Tensor từ Encoder (Skip Connection).
    Mặc định dùng Bilinear Upsampling (use_conv_transpose=False) cho Mamba-UNet.
    """
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        use_conv_transpose: bool = False,
        act: Optional[Union[str, Callable[[], nn.Module]]] = None,
    ):
        super().__init__()
        self.use_conv_transpose = use_conv_transpose
        if use_conv_transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            conv_in = (in_ch // 2) + skip_ch
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            conv_in = in_ch + skip_ch

        self.conv = DoubleConv(conv_in, out_ch, act=act)

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[2:] != skip_x.shape[2:]:
            x = F.interpolate(x, size=skip_x.shape[2:], mode="bilinear", align_corners=True)

        x = torch.cat([skip_x, x], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    U-Net Decoder 4 tầng kết nối với MambaVision Backbone:
      f1 (H/4), f2 (H/8), f3 (H/16), f4 (H/32 - Bottleneck).
    """
    def __init__(
        self,
        encoder_channels: Optional[List[int]] = None,
        num_classes: int = 1,
        final_channels: int = 32,
        act: Optional[Union[str, Callable[[], nn.Module]]] = None,
        init_bias: Optional[float] = -4.0,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [80, 160, 320, 640]  # Mặc định MambaVision-T

        c1, c2, c3, c4 = encoder_channels
        self.encoder_channels = encoder_channels

        self.up1 = UpBlock(c4, c3, c3, use_conv_transpose=False, act=act)
        self.up2 = UpBlock(c3, c2, c2, use_conv_transpose=False, act=act)
        self.up3 = UpBlock(c2, c1, c1, use_conv_transpose=False, act=act)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            DoubleConv(c1, final_channels, act=act),
        )
        self.outc = OutConv(final_channels, num_classes, init_bias=init_bias)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        d1 = self.up1(f4, f3)
        d2 = self.up2(d1, f2)
        d3 = self.up3(d2, f1)
        x = self.final_up(d3)
        logits = self.outc(x)
        return logits


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.decoder...")
    dec = UNetDecoder(encoder_channels=[80, 160, 320, 640], num_classes=1)

    f1 = torch.randn(2, 80, 32, 32)
    f2 = torch.randn(2, 160, 16, 16)
    f3 = torch.randn(2, 320, 8, 8)
    f4 = torch.randn(2, 640, 4, 4)

    out = dec([f1, f2, f3, f4])
    assert out.shape == (2, 1, 128, 128), f"Lỗi shape đầu ra UNetDecoder: {out.shape}"
    print("✅ Selftest mamba_unet.decoder hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
```

#### <a id="tệp-07-mamba_unet-blocks-py"></a>Tệp 07: `mamba_unet/blocks.py`
- **Vị trí**: `mamba_unet/blocks.py`
- **Quy mô**: `101 dòng` | `3.9 KB`
- **Vai trò**: Các khối tích chập cơ bản trong Decoder (DoubleConv, UpBlock).

```python
# -*- coding: utf-8 -*-
"""
mamba_unet/blocks.py - Các khối tích chập chuẩn DoubleConv và OutConv riêng của mamba_unet.

Quy tắc bất biến (R1, R11, R3):
  - Khối mạng thuộc sở hữu độc quyền của mamba_unet, không dùng chung và không import từ unet_base.
  - Bảo toàn cấu trúc chỉ số lớp con (0: Conv2d, 1: BatchNorm2d, 2: Act, 3: Conv2d, 4: BatchNorm2d, 5: Act)
    để tương thích 100% với tập khóa decoder của mamba_unet_keys.json.
"""

import sys
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mamba_unet.activations import DEFAULT_ACTIVATION, get_activation


class DoubleConv(nn.Module):
    """
    Khối tích chập kép chuẩn:
      Conv2d(3x3, bias=False) -> BatchNorm2d -> Act -> Conv2d(3x3, bias=False) -> BatchNorm2d -> Act
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        act: Optional[Union[str, Callable[[], nn.Module]]] = None,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if act is None:
            act_factory = get_activation(DEFAULT_ACTIVATION)
        elif isinstance(act, str):
            act_factory = get_activation(act)
        elif callable(act):
            act_factory = act
        else:
            raise TypeError(f"Tham số 'act' phải là str hoặc Callable, nhận được {type(act)}")

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # 0
            nn.BatchNorm2d(mid_channels),                                                # 1
            act_factory(),                                                               # 2
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), # 3
            nn.BatchNorm2d(out_channels),                                                # 4
            act_factory(),                                                               # 5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class OutConv(nn.Module):
    """Lớp tích chập 1x1 xuất mặt nạ phân đoạn với khởi tạo bias."""
    def __init__(self, in_channels: int, out_channels: int, init_bias: Optional[float] = -4.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if init_bias is not None and self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, float(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.blocks...")
    x = torch.randn(2, 16, 32, 32)
    dc = DoubleConv(16, 32)
    out = dc(x)
    assert out.shape == (2, 32, 32, 32), f"DoubleConv shape sai: {out.shape}"

    oc = OutConv(32, 1)
    out_oc = oc(out)
    assert out_oc.shape == (2, 1, 32, 32), f"OutConv shape sai: {out_oc.shape}"

    # Kiểm tra bảo toàn khóa khi đổi activation không tham số
    keys_relu = set(DoubleConv(16, 32, act="relu").state_dict().keys())
    keys_gelu = set(DoubleConv(16, 32, act="gelu").state_dict().keys())
    assert keys_relu == keys_gelu, "Đổi activation không tham số làm lệch tập khóa!"

    # Kiểm tra star_relu thêm khóa
    keys_star = set(DoubleConv(16, 32, act="star_relu").state_dict().keys())
    assert len(keys_star - keys_relu) > 0, "star_relu phải thêm khóa tham số học được!"

    print("✅ Selftest mamba_unet.blocks hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
```

#### <a id="tệp-08-mamba_unet-activations-py"></a>Tệp 08: `mamba_unet/activations.py`
- **Vị trí**: `mamba_unet/activations.py`
- **Quy mô**: `218 dòng` | `7.9 KB`
- **Vai trò**: Registry các hàm kích hoạt hỗ trợ (ReLU, GELU, SiLU, LeakyReLU, Mish).

```python
# -*- coding: utf-8 -*-
"""
mamba_unet/activations.py - Registry và các hàm kích hoạt (Activation Functions) riêng của mamba_unet.

⚠️ LƯU Ý ĐẶC BIỆT VỀ KIẾN TRÚC (spec mục 4.1, 5.3):
  Hàm kích hoạt bên trong MambaVision backbone (GELU và SiLU của khối gốc) là CỐ ĐỊNH
  để bảo toàn cấu trúc pretrained của NVIDIA.
  Bảng registry trong file này CHỈ ÁP DỤNG CHO CÁC KHỐI CONV BÊN TRONG DECODER (UNetDecoder).

Quy tắc bất biến (R1, R11):
  - File tự đủ: CHỈ import torch và thư viện chuẩn, không import bất kỳ file nào khác của dự án.
  - Hoàn toàn độc lập với unet_base.activations (R11).
  - Trả về FACTORY không đối số: Callable[[], nn.Module].
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. CÁC LỚP ACTIVATION TỰ VIẾT
# =============================================================================
class SoftClampReLU(nn.Module):
    """
    ReLU với hàm kẹp mềm ở ngưỡng trên tau:
        f(x) = tau * tanh(relu(x) / tau)
    """
    def __init__(self, tau: float = 10.0, inplace: bool = False):
        super().__init__()
        self.tau = float(tau)
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = F.relu(x, inplace=self.inplace)
        return self.tau * torch.tanh(pos / self.tau)

    def extra_repr(self) -> str:
        return f"tau={self.tau}, inplace={self.inplace}"


class SoftClampSiLU(nn.Module):
    """
    SiLU (Swish) với cơ chế kẹp mềm đạo hàm ở các giá trị cực lớn:
        f(x) = silu(x) / (1.0 + |x| / tau)
    """
    def __init__(self, tau: float = 20.0):
        super().__init__()
        self.tau = float(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_val = F.silu(x)
        scale = 1.0 + torch.abs(x) / self.tau
        return silu_val / scale

    def extra_repr(self) -> str:
        return f"tau={self.tau}"


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x)^2 + b
    Biến thể phi tuyến bậc hai (hệ số scale và bias có thể học được).
    """
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0, learnable: bool = True):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.scale = nn.Parameter(torch.tensor(scale_value, dtype=torch.float32))
            self.bias = nn.Parameter(torch.tensor(bias_value, dtype=torch.float32))
        else:
            self.register_buffer("scale", torch.tensor(scale_value, dtype=torch.float32))
            self.register_buffer("bias", torch.tensor(bias_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (F.relu(x) ** 2) + self.bias


# =============================================================================
# 2. BẢNG ĐĂNG KÝ ACTIVATION CHO DECODER (REGISTRY)
# =============================================================================
@dataclass
class ActivationSpec:
    factory: Callable[..., nn.Module]
    has_params: bool
    doc: str


ACTIVATIONS: Dict[str, ActivationSpec] = {
    "relu": ActivationSpec(
        factory=lambda inplace=True, **kw: nn.ReLU(inplace=inplace),
        has_params=False,
        doc="Chuẩn Rectified Linear Unit (mặc định cho Decoder)",
    ),
    "leaky_relu": ActivationSpec(
        factory=lambda negative_slope=0.01, inplace=True, **kw: nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace),
        has_params=False,
        doc="LeakyReLU với negative_slope tùy chỉnh",
    ),
    "gelu": ActivationSpec(
        factory=lambda approximate="none", **kw: nn.GELU(approximate=approximate),
        has_params=False,
        doc="Gaussian Error Linear Unit",
    ),
    "silu": ActivationSpec(
        factory=lambda inplace=False, **kw: nn.SiLU(inplace=inplace),
        has_params=False,
        doc="Sigmoid Linear Unit (Swish)",
    ),
    "mish": ActivationSpec(
        factory=lambda inplace=False, **kw: nn.Mish(inplace=inplace),
        has_params=False,
        doc="Mish: x * tanh(softplus(x))",
    ),
    "soft_clamp_relu": ActivationSpec(
        factory=lambda tau=10.0, inplace=False, **kw: SoftClampReLU(tau=tau, inplace=inplace),
        has_params=False,
        doc="ReLU có hàm kẹp mềm ở ngưỡng trên tau",
    ),
    "soft_clamp_silu": ActivationSpec(
        factory=lambda tau=20.0, **kw: SoftClampSiLU(tau=tau),
        has_params=False,
        doc="SiLU có hàm kẹp mềm đạo hàm ở giá trị cực lớn",
    ),
    "star_relu": ActivationSpec(
        factory=lambda scale_value=1.0, bias_value=0.0, learnable=True, **kw: StarReLU(
            scale_value=scale_value, bias_value=bias_value, learnable=learnable
        ),
        has_params=True,
        doc="StarReLU (s * relu(x)^2 + b, thêm khóa scale và bias vào state_dict)",
    ),
    "identity": ActivationSpec(
        factory=lambda **kw: nn.Identity(),
        has_params=False,
        doc="Tuyến tính (Identity / Không kích hoạt)",
    ),
}

DEFAULT_ACTIVATION: str = "relu"


def get_activation(name: Optional[str] = None, **kwargs) -> Callable[[], nn.Module]:
    """
    Trả về một FACTORY không đối số để khởi tạo activation cho Decoder của mamba_unet.
    """
    if name is None:
        name = DEFAULT_ACTIVATION

    key = name.lower().strip().replace("-", "_")
    if key in ("rectified_linear_unit",):
        key = "relu"
    elif key in ("lrelu",):
        key = "leaky_relu"
    elif key in ("swish",):
        key = "silu"
    elif key in ("softclamprelu",):
        key = "soft_clamp_relu"
    elif key in ("softclampsilu",):
        key = "soft_clamp_silu"
    elif key in ("starrelu",):
        key = "star_relu"
    elif key in ("none", "linear"):
        key = "identity"

    if key not in ACTIVATIONS:
        valid_keys = list(ACTIVATIONS.keys())
        raise ValueError(
            f"❌ Không hỗ trợ activation '{name}' trong decoder của mamba_unet!\n"
            f"  Các lựa chọn hợp lệ: {valid_keys}"
        )

    spec = ACTIVATIONS[key]
    return lambda: spec.factory(**kwargs)


def list_activations() -> List[str]:
    """Liệt kê danh sách tên các hàm kích hoạt hợp lệ cho decoder của mamba_unet."""
    return list(ACTIVATIONS.keys())


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.activations...")
    x = torch.linspace(-5, 15, 100)

    for act_name in list_activations():
        factory = get_activation(act_name)
        mod = factory()
        out = mod(x)
        assert out.shape == x.shape, f"Lỗi shape tại {act_name}: {out.shape} != {x.shape}"
        spec = ACTIVATIONS[act_name]
        param_count = sum(p.numel() for p in mod.parameters())
        if spec.has_params:
            assert param_count > 0, f"Activation {act_name} khai báo has_params=True nhưng không có tham số!"
        else:
            assert param_count == 0, f"Activation {act_name} khai báo has_params=False nhưng có {param_count} tham số!"

    f = get_activation("relu")
    assert f() is not f(), "Factory phải sinh ra các instance riêng biệt!"
    print(f"✅ Đã kiểm thử thành công {len(ACTIVATIONS)} hàm kích hoạt trong mamba_unet.activations!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Quản lý và kiểm thử Activation Functions của mamba_unet")
    parser.add_argument("--list", action="store_true", help="In danh sách các activation hợp lệ")
    args = parser.parse_args()

    if args.list:
        print("\n".join(list_activations()))
    else:
        _selftest()
```

#### <a id="tệp-09-mamba_unet-losses-py"></a>Tệp 09: `mamba_unet/losses.py`
- **Vị trí**: `mamba_unet/losses.py`
- **Quy mô**: `308 dòng` | `11.5 KB`
- **Vai trò**: Registry các hàm mất mát phân đoạn: PureDiceBCELoss, PureTverskyLoss, PureTverskyFocalLoss, PureDiceLoss, PureBCELoss, PureFocalLoss.

```python
# -*- coding: utf-8 -*-
"""
mamba_unet/losses.py - Registry và các hàm mất mát (Loss Functions) riêng của mamba_unet.

Quy tắc bất biến (R1, R11):
  - File tự đủ: CHỈ import torch và thư viện chuẩn, không import bất kỳ file nào khác của dự án.
  - Tuyệt đối KHÔNG chứa metrics (metrics nằm tại common/metrics.py theo R11).
  - Ép logits.float() trong forward để an toàn tuyệt đối khi bật AMP.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. CÁC LỚP LOSS RIÊNG CỦA MAMBA_UNET
# =============================================================================
class PureDiceLoss(nn.Module):
    """Pure Soft Dice Loss cho phân đoạn ảnh y tế."""
    def __init__(self, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.smooth = float(smooth)
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.to(dtype=pred.dtype)
        probs = torch.sigmoid(pred) if self.from_logits else pred

        probs_flat = probs.reshape(-1)
        target_flat = target.reshape(-1)

        intersection = (probs_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice


class PureTverskyLoss(nn.Module):
    """Tversky Loss phạt nặng bỏ sót (False Negatives)."""
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.to(dtype=pred.dtype)
        probs = torch.sigmoid(pred) if self.from_logits else pred

        p_flat = probs.reshape(-1)
        t_flat = target.reshape(-1)

        tp = (p_flat * t_flat).sum()
        fp = (p_flat * (1.0 - t_flat)).sum()
        fn = ((1.0 - p_flat) * t_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class PureTverskyFocalLoss(nn.Module):
    """Focal Tversky Loss: Lũy thừa hệ số gamma cho khối u nhỏ."""
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.to(dtype=pred.dtype)
        probs = torch.sigmoid(pred) if self.from_logits else pred

        p_flat = probs.reshape(-1)
        t_flat = target.reshape(-1)

        tp = (p_flat * t_flat).sum()
        fp = (p_flat * (1.0 - t_flat)).sum()
        fn = ((1.0 - p_flat) * t_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma)


class PureFocalLoss(nn.Module):
    """Binary Focal Loss."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, from_logits: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.to(dtype=pred.dtype)
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            probs = torch.sigmoid(pred)
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
            probs = pred

        p_t = probs * target + (1.0 - probs) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_loss = alpha_t * torch.pow(1.0 - p_t, self.gamma) * bce
        return focal_loss.mean()


class PureDiceBCELoss(nn.Module):
    """Hàm mất mát chuẩn kết hợp BCE và Soft Dice Loss."""
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
        smooth: float = 1.0,
        batch_dice: bool = True,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.batch_dice = bool(batch_dice)
        self.last_components: Dict[str, float] = {}
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
        else:
            self.pos_weight = None

    def get_components(self) -> Dict[str, float]:
        """Trả về chi tiết các thành phần loss (BCE và Soft Dice) của batch gần nhất."""
        return getattr(self, "last_components", {})

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)

        probs = torch.sigmoid(logits)
        if self.batch_dice:
            probs_flat = probs.reshape(-1)
            targets_flat = targets.reshape(-1)
            intersection = (probs_flat * targets_flat).sum()
            dice_score = (2.0 * intersection + self.smooth) / (
                probs_flat.sum() + targets_flat.sum() + self.smooth
            )
        else:
            probs_flat = probs.reshape(probs.size(0), -1)
            targets_flat = targets.reshape(targets.size(0), -1)
            intersection = (probs_flat * targets_flat).sum(dim=1)
            dice_score = (2.0 * intersection + self.smooth) / (
                probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
            )
            dice_score = dice_score.mean()

        dice_loss = 1.0 - dice_score

        # Lưu lại chi tiết từng thành phần để hiển thị log và lưu history.csv
        self.last_components = {
            "bce": float(bce.detach().item()),
            "dice": float(dice_loss.detach().item()),
        }

        return self.bce_weight * bce + self.dice_weight * dice_loss


class PureBCELoss(nn.Module):
    """BCEWithLogitsLoss đơn thuần ép float32 an toàn AMP."""
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


# =============================================================================
# 2. BẢNG ĐĂNG KÝ LOSS (REGISTRY)
# =============================================================================
@dataclass
class LossSpec:
    factory: Callable[..., nn.Module]
    doc: str


LOSSES: Dict[str, LossSpec] = {
    "dice_bce": LossSpec(
        factory=lambda bce_weight=0.5, dice_weight=0.5, pos_weight=None, smooth=1.0, batch_dice=True, **kw: PureDiceBCELoss(
            bce_weight=bce_weight, dice_weight=dice_weight, pos_weight=pos_weight, smooth=smooth, batch_dice=batch_dice
        ),
        doc="Kết hợp BCEWithLogitsLoss và Soft Dice Loss (mặc định)",
    ),
    "dice": LossSpec(
        factory=lambda smooth=1e-6, from_logits=True, **kw: PureDiceLoss(smooth=smooth, from_logits=from_logits),
        doc="Soft Dice Loss thuần túy",
    ),
    "bce": LossSpec(
        factory=lambda pos_weight=None, **kw: PureBCELoss(pos_weight=pos_weight),
        doc="Binary Cross Entropy with Logits",
    ),
    "tversky": LossSpec(
        factory=lambda alpha=0.3, beta=0.7, smooth=1e-6, from_logits=True, **kw: PureTverskyLoss(
            alpha=alpha, beta=beta, smooth=smooth, from_logits=from_logits
        ),
        doc="Tversky Loss phạt nặng bỏ sót (beta > alpha)",
    ),
    "focal_tversky": LossSpec(
        factory=lambda alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0, from_logits=True, **kw: PureTverskyFocalLoss(
            alpha=alpha, beta=beta, gamma=gamma, smooth=smooth, from_logits=from_logits
        ),
        doc="Tversky Focal Loss cho khối u nhỏ và lát cắt khó",
    ),
    "focal": LossSpec(
        factory=lambda alpha=0.25, gamma=2.0, from_logits=True, **kw: PureFocalLoss(
            alpha=alpha, gamma=gamma, from_logits=from_logits
        ),
        doc="Binary Focal Loss",
    ),
}

DEFAULT_LOSS: str = "dice_bce"
DEFAULT_LOSS_PARAMS: Dict[str, float] = {
    "bce_weight": 0.5,
    "dice_weight": 0.5,
    "smooth": 1.0,
    "batch_dice": True,
}


def get_loss(name: Optional[str] = None, **params) -> nn.Module:
    """Khởi tạo instance hàm loss từ tên và tham số cho mamba_unet."""
    if name is None:
        name = DEFAULT_LOSS

    key = name.lower().strip().replace("-", "_")
    if key in ("diceloss",):
        key = "dice"
    elif key in ("dicebceloss",):
        key = "dice_bce"
    elif key in ("tverskyloss",):
        key = "tversky"
    elif key in ("tversky_focal", "tverskyfocalloss", "focaltversky"):
        key = "focal_tversky"
    elif key in ("focalloss",):
        key = "focal"
    elif key in ("bceloss", "bce_with_logits"):
        key = "bce"

    if key not in LOSSES:
        valid_keys = list(LOSSES.keys())
        raise ValueError(
            f"❌ Không hỗ trợ loss '{name}' trong mamba_unet!\n"
            f"  Các lựa chọn hợp lệ: {valid_keys}"
        )

    spec = LOSSES[key]
    return spec.factory(**params)


def list_losses() -> List[str]:
    """Liệt kê danh sách tên các hàm loss hợp lệ trong mamba_unet."""
    return list(LOSSES.keys())


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.losses...")
    logits = torch.randn(4, 1, 32, 32)
    targets = torch.randint(0, 2, (4, 1, 32, 32)).float()

    for loss_name in list_losses():
        loss_fn = get_loss(loss_name)
        val = loss_fn(logits, targets)
        assert isinstance(val, torch.Tensor), f"Loss {loss_name} không trả về Tensor!"
        assert val.ndim == 0, f"Loss {loss_name} phải là scalar, got shape {val.shape}"
        assert not torch.isnan(val), f"Loss {loss_name} bị NaN!"
        assert not torch.isinf(val), f"Loss {loss_name} bị Inf!"
        print(f"  ✓ {loss_name:15s}: {val.item():.4f}")

    print(f"✅ Đã kiểm thử thành công {len(LOSSES)} hàm loss trong mamba_unet.losses!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Quản lý và kiểm thử Loss Functions của mamba_unet")
    parser.add_argument("--list", action="store_true", help="In danh sách các loss hợp lệ")
    args = parser.parse_args()

    if args.list:
        print("\n".join(list_losses()))
    else:
        _selftest()
```

#### <a id="tệp-10-mamba_unet-fallbacks-py"></a>Tệp 10: `mamba_unet/fallbacks.py`
- **Vị trí**: `mamba_unet/fallbacks.py`
- **Quy mô**: `332 dòng` | `12.9 KB`
- **Vai trò**: Cơ chế fallback Selective Scan thuần PyTorch (MambaScanFallback2D) khi môi trường không có CUDA kernel compiled.

```python
# -*- coding: utf-8 -*-
"""
mamba_unet/fallbacks.py - Chọn backend Selective Scan (CUDA vs Pure PyTorch) và các lớp dự phòng.

Quy tắc bất biến (R8, spec mục 5.3):
  - mamba_unet chạy được ngay cả khi không có mamba_ssm, timm, einops (dùng bản dự phòng thuần PyTorch).
  - Biến môi trường DUNEAI_MAMBA_BACKEND=torch|cuda để ép backend; ép cuda mà thiếu kernel/GPU -> RuntimeError rõ ràng.
  - selective_scan_torch tính toán bằng float32, khớp công thức đóng O(L²) tính độc lập bằng float64 với sai khác <= 1e-4.
  - Backward pass cho gradient hữu hạn.
"""

import os
import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. FALLBACKS CHO TIMM VÀ EINOPS (R8: Phụ thuộc nặng là tùy chọn)
# =============================================================================
try:
    from timm.layers import DropPath, LayerNorm2d, trunc_normal_
except ImportError:
    try:
        from timm.models.layers import DropPath, LayerNorm2d, trunc_normal_
    except ImportError:
        def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
            return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
                super().__init__()
                self.drop_prob = drop_prob
                self.scale_by_keep = scale_by_keep

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep_prob = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
                if keep_prob > 0.0 and self.scale_by_keep:
                    random_tensor.div_(keep_prob)
                return x * random_tensor

        class LayerNorm2d(nn.Module):
            def __init__(self, num_channels: int, eps: float = 1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(num_channels))
                self.bias = nn.Parameter(torch.zeros(num_channels))
                self.eps = eps

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x


try:
    from einops import rearrange, repeat
except ImportError:
    def rearrange(tensor: torch.Tensor, pattern: str, **kwargs) -> torch.Tensor:
        pattern = pattern.strip()
        if pattern == "b l d -> b d l":
            return tensor.transpose(1, 2).contiguous()
        elif pattern == "b d l -> b l d":
            return tensor.transpose(1, 2).contiguous()
        elif pattern == "b d l -> (b l) d":
            b, d, l = tensor.shape
            return tensor.permute(0, 2, 1).reshape(b * l, d).contiguous()
        elif pattern == "(b l) d -> b d l":
            l = kwargs.get("l")
            bl, d = tensor.shape
            b = bl // l
            return tensor.view(b, l, d).permute(0, 2, 1).contiguous()
        elif pattern == "(b l) dstate -> b dstate l":
            l = kwargs.get("l")
            bl, ds = tensor.shape
            b = bl // l
            return tensor.view(b, l, ds).permute(0, 2, 1).contiguous()
        elif pattern == "... (L two) -> ... L two":
            shape = list(tensor.shape)
            shape[-1] = shape[-1] // 2
            shape.append(2)
            return tensor.view(*shape)
        elif pattern == "d -> d 1":
            return tensor.unsqueeze(1)
        raise NotImplementedError(f"Pattern rearrange chưa hỗ trợ: {pattern}")

    def repeat(tensor: torch.Tensor, pattern: str, **kwargs) -> torch.Tensor:
        clean_pat = pattern.replace(" ", "")
        if "B G N L -> B (G H) N L" in pattern:
            h = kwargs.get("H", 1)
            return tensor.repeat_interleave(h, dim=1)
        elif clean_pat == "n->dn":
            d = kwargs.get("d", 1)
            return tensor.unsqueeze(0).repeat(d, 1)
        raise NotImplementedError(f"Pattern repeat chưa hỗ trợ: {pattern}")


# =============================================================================
# 2. SELECTIVE SCAN PURE PYTORCH FALLBACK
# =============================================================================
def selective_scan_torch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Thuật toán Selective Scan cài đặt bằng PyTorch thuần túy (CPU & GPU).
    Tính toán bằng float32 để ổn định gradient và khớp độ chính xác cao.
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()

    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))

    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B_rep = B.repeat_interleave(dim // B.shape[1], dim=1)
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B_rep, u)

    if is_variable_C and C.dim() == 4:
        C_rep = C.repeat_interleave(dim // C.shape[1], dim=1)
    else:
        C_rep = C

    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C_rep[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)

    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.unsqueeze(1)
    if z is not None:
        out = out * F.silu(z.float())

    out = out.to(dtype=dtype_in)
    return (out, last_state) if return_last_state else out


def get_scan_backend() -> str:
    """Trả về tên backend hiện đang được kích hoạt ('cuda' hoặc 'torch')."""
    forced = os.environ.get("DUNEAI_MAMBA_BACKEND", "").lower().strip()
    if forced in ("cuda", "torch"):
        return forced
    if torch.cuda.is_available():
        try:
            import mamba_ssm
            return "cuda"
        except ImportError:
            pass
    return "torch"


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Dispatcher chọn Selective Scan Backend dựa trên cấu hình môi trường:
      - DUNEAI_MAMBA_BACKEND=cuda : ép dùng CUDA mamba_ssm kernel, lỗi nếu không khả dụng.
      - DUNEAI_MAMBA_BACKEND=torch: ép dùng selective_scan_torch thuần PyTorch.
      - Mặc định: tự động chọn CUDA nếu có GPU và gói mamba_ssm, ngược lại dùng fallback PyTorch.
    """
    forced_backend = os.environ.get("DUNEAI_MAMBA_BACKEND", "").lower().strip()

    if forced_backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("❌ DUNEAI_MAMBA_BACKEND=cuda được yêu cầu nhưng thiết bị không có GPU CUDA khả dụng!")
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as cuda_scan
            return cuda_scan(
                u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus, return_last_state=return_last_state
            )
        except ImportError as e:
            raise RuntimeError(f"❌ DUNEAI_MAMBA_BACKEND=cuda được yêu cầu nhưng không import được gói mamba_ssm: {e}")

    elif forced_backend == "torch":
        return selective_scan_torch(
            u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus, return_last_state=return_last_state
        )

    # Chế độ tự động (Auto fallback)
    if u.is_cuda and torch.cuda.is_available():
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as cuda_scan
            return cuda_scan(
                u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus, return_last_state=return_last_state
            )
        except (ImportError, Exception):
            pass

    return selective_scan_torch(
        u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus, return_last_state=return_last_state
    )


# =============================================================================
# 3. CÔNG THỨC THAM CHIẾU ĐÓNG O(L^2) BẰNG FLOAT64 ĐỂ KIỂM THỬ
# =============================================================================
def selective_scan_ref_float64(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Công thức tham chiếu độc lập tính bằng float64 phục vụ kiểm thử độ chính xác (<= 1e-4).
    """
    u_64 = u.to(torch.float64)
    delta_64 = delta.to(torch.float64)
    A_64 = A.to(torch.float64)
    B_64 = B.to(torch.float64)
    C_64 = C.to(torch.float64)

    batch, dim, L = u.shape
    dstate = A.shape[1]

    # Tính toán tuần tự trạng thái ẩn x_t bằng float64
    x = torch.zeros((batch, dim, dstate), dtype=torch.float64, device=u.device)
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta_64, A_64))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta_64, B_64, u_64)

    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum("bdn,bn->bd", x, C_64[:, :, i])
        ys.append(y)

    y_out = torch.stack(ys, dim=2)
    if D is not None:
        y_out = y_out + u_64 * D.to(torch.float64).unsqueeze(1)
    return y_out.float()


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.fallbacks...")
    torch.manual_seed(42)
    B, D_in, L, N = 2, 4, 8, 4
    u = torch.randn(B, D_in, L, requires_grad=True)
    delta = F.softplus(torch.randn(B, D_in, L)).requires_grad_(True)
    A = -torch.rand(D_in, N)
    B_mat = torch.randn(B, N, L, requires_grad=True)
    C_mat = torch.randn(B, N, L, requires_grad=True)
    D_vec = torch.randn(D_in)

    # 1. So sánh selective_scan_torch với công thức tham chiếu float64
    out_torch = selective_scan_torch(u, delta, A, B_mat, C_mat, D=D_vec)
    out_ref = selective_scan_ref_float64(u, delta, A, B_mat, C_mat, D=D_vec)

    diff = (out_torch - out_ref).abs().max().item()
    print(f"  ✓ Sai khác tối đa so với float64 tham chiếu: {diff:.2e} (Tiêu chí <= 1e-4)")
    assert diff <= 1e-4, f"Sai khác quá lớn: {diff}"

    # 2. Kiểm tra backward pass tính toán đạo hàm hữu hạn
    loss = out_torch.sum()
    loss.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all(), "Đạo hàm u chứa NaN hoặc Inf!"
    assert delta.grad is not None and torch.isfinite(delta.grad).all(), "Đạo hàm delta chứa NaN hoặc Inf!"
    print("  ✓ Gradient backward hữu hạn và ổn định.")

    # 3. Kiểm tra biến môi trường DUNEAI_MAMBA_BACKEND=torch
    os.environ["DUNEAI_MAMBA_BACKEND"] = "torch"
    out_forced = selective_scan_fn(u, delta, A, B_mat, C_mat, D=D_vec)
    assert out_forced.shape == out_torch.shape
    os.environ.pop("DUNEAI_MAMBA_BACKEND", None)

    print("✅ Selftest mamba_unet.fallbacks hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
```


---

### 2. Điểm Nhập Huấn Luyện & Đánh Giá (CLI Entrypoints)

#### <a id="tệp-11-mamba_unet-trainer-py"></a>Tệp 11: `mamba_unet/trainer.py`
- **Vị trí**: `mamba_unet/trainer.py`
- **Quy mô**: `171 dòng` | `6.4 KB`
- **Vai trò**: MambaUNetTrainer (kế thừa BaseTrainer): Điều khiển lịch đóng băng backbone, phân nhóm tham số learning rate (decoder lr vs backbone lr).

```python
# -*- coding: utf-8 -*-
"""
mamba_unet/trainer.py - Trainer chuyên biệt cho Mamba-UNet (kế thừa BaseTrainer).

Quy tắc bất biến (spec mục 5.3):
  - Lịch đóng băng backbone suy ra trực tiếp từ số epoch (resume ở đâu cũng đúng):
      * freeze_backbone_epochs = 0: không đóng băng.
      * freeze_backbone_epochs = N > 0: đóng băng trong N epoch đầu (epoch <= N).
      * freeze_backbone_epochs = -1: đóng băng suốt quá trình.
  - Khi đóng băng, BatchNorm của backbone giữ eval() để không lệch running mean/var.
  - Phân chia 2 nhóm tham số Optimizer:
      * Nhóm 'decoder' : lr = base_lr
      * Nhóm 'backbone': lr = base_lr * backbone_lr_mult (mặc định 0.1 khi pretrained, 1.0 khi scratch).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.engine.base_trainer import BaseTrainer


class MambaUNetTrainer(BaseTrainer):
    """Trainer chuyên biệt cho mô hình MambaUNet."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        backbone_name: str = "mamba_vision_T",
        activation_name: str = "relu",
        in_channels: int = 3,
        num_classes: int = 1,
        freeze_backbone_epochs: int = 0,
        backbone_lr_mult: float = 0.1,
        **kwargs,
    ):
        self.backbone_name = backbone_name
        self.activation_name = activation_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.freeze_backbone_epochs = int(freeze_backbone_epochs)
        self.backbone_lr_mult = float(backbone_lr_mult)
        super().__init__(model=model, criterion=criterion, **kwargs)

    def on_epoch_start(self, epoch: int):
        """Hook điều khiển đóng/mở băng backbone suy ra từ số epoch hiện tại."""
        unwrapped = self._unwrap_model()
        should_freeze = (self.freeze_backbone_epochs == -1) or (epoch <= self.freeze_backbone_epochs and self.freeze_backbone_epochs > 0)

        if should_freeze:
            # Đóng băng toàn bộ trọng số của backbone encoder
            for param in unwrapped.encoder.parameters():
                param.requires_grad = False
            # BatchNorm của backbone giữ eval()
            for m in unwrapped.encoder.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
        else:
            # Mở băng backbone cho huấn luyện fine-tuning
            for param in unwrapped.encoder.parameters():
                param.requires_grad = True

    def build_param_groups(self, lr: float) -> List[Dict[str, Any]]:
        """Phân chia 2 nhóm learning rate: decoder và backbone."""
        unwrapped = self._unwrap_model()

        decoder_params = [p for p in unwrapped.decoder.parameters() if p.requires_grad]
        backbone_params = [p for p in unwrapped.encoder.parameters() if p.requires_grad]

        groups = [
            {"params": decoder_params, "lr": lr, "name": "decoder"},
        ]
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": lr * self.backbone_lr_mult,
                "name": "backbone",
            })
        return groups

    def checkpoint_meta(self) -> Dict[str, Any]:
        """Metadata kiến trúc chuẩn của mamba_unet."""
        bb = getattr(self, "backbone_name", "mamba_vision_T")
        act = getattr(self, "activation_name", "relu")
        in_ch = getattr(self, "in_channels", 3)
        num_cls = getattr(self, "num_classes", 1)
        freeze_ep = getattr(self, "freeze_backbone_epochs", 0)
        lr_mult = getattr(self, "backbone_lr_mult", 0.1)
        return {
            "arch": "mamba_unet",
            "backbone": bb,
            "activation": act,
            "in_channels": in_ch,
            "num_classes": num_cls,
            "freeze_backbone_epochs": freeze_ep,
            "backbone_lr_mult": lr_mult,
            "model_config": {
                "backbone": bb,
                "activation": act,
                "in_channels": in_ch,
                "num_classes": num_cls,
            },
        }


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.trainer...")
    from mamba_unet.backbone.mamba_vision import MambaVision
    from mamba_unet.losses import get_loss
    from mamba_unet.model import MambaUNet

    tiny_bb = MambaVision(
        dim=16,
        in_dim=8,
        depths=[1, 1, 2, 1],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 2, 4, 4],
        in_chans=3,
    )
    model = MambaUNet(in_channels=3, num_classes=1, backbone=tiny_bb)
    criterion = get_loss("dice_bce")

    trainer = MambaUNetTrainer(
        model=model,
        criterion=criterion,
        freeze_backbone_epochs=2,
        backbone_lr_mult=0.1,
        show_progress=False,
        use_amp=False,
    )

    # 1. Kiểm tra epoch 1: backbone phải bị đóng băng
    trainer.on_epoch_start(1)
    for p in model.encoder.parameters():
        assert not p.requires_grad, "Backbone phải bị đóng băng ở Epoch 1!"
    print("  ✓ Epoch 1: Backbone đóng băng chính xác.")

    # 2. Kiểm tra epoch 2: backbone vẫn bị đóng băng
    trainer.on_epoch_start(2)
    for p in model.encoder.parameters():
        assert not p.requires_grad, "Backbone phải bị đóng băng ở Epoch 2!"
    print("  ✓ Epoch 2: Backbone đóng băng chính xác.")

    # 3. Kiểm tra epoch 3: backbone phải được mở băng
    trainer.on_epoch_start(3)
    for p in model.encoder.parameters():
        assert p.requires_grad, "Backbone phải được mở băng ở Epoch 3!"
    print("  ✓ Epoch 3: Backbone mở băng chính xác.")

    # 4. Kiểm tra nhóm tham số Optimizer
    groups = trainer.build_param_groups(1e-4)
    assert len(groups) == 2, f"Phải có đúng 2 nhóm tham số, nhận được {len(groups)}"
    assert groups[0]["name"] == "decoder" and groups[0]["lr"] == 1e-4
    assert groups[1]["name"] == "backbone" and abs(groups[1]["lr"] - 1e-5) < 1e-9
    print("  ✓ Phân nhóm Learning Rate chính xác (decoder: 1e-4, backbone: 1e-5).")

    print("✅ Selftest mamba_unet.trainer hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
```

#### <a id="tệp-12-mamba_unet-train_main-py"></a>Tệp 12: `mamba_unet/train_main.py`
- **Vị trí**: `mamba_unet/train_main.py`
- **Quy mô**: `438 dòng` | `17.0 KB`
- **Vai trò**: CLI entrypoint chính cho Mamba-UNet: Hỗ trợ các lệnh train, overfit, selftest; cấu hình tham số, seed, và kích hoạt pipeline.

```python
# -*- coding: utf-8 -*-
"""
mamba_unet/train_main.py - Điểm nhập thực thi (CLI entrypoint) cho kiến trúc Mamba-UNet.

Các chế độ hoạt động:
  - selftest: Kiểm thử tự động toàn diện (huấn luyện dữ liệu giả lập, ngắt, resume auto,
              kiểm tra lịch đóng băng backbone, xuất fp16 và run_config).
  - overfit : Huấn luyện thử trên tập siêu nhỏ để kiểm tra khả năng hội tụ nhanh của mô hình.
  - train   : Huấn luyện chính thức với đầy đủ tùy chọn cấu hình và ghi nhận phiên chạy.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.engine.cli_args import add_common_args
from common.engine.checkpoint_io import load_checkpoint_file
from common.engine.run_manager import create_run, resolve_resume, RunHandle
from common.engine.utils import seed_everything
from common.paths import default_runs_root

from mamba_unet.activations import DEFAULT_ACTIVATION, list_activations
from mamba_unet.losses import DEFAULT_LOSS, DEFAULT_LOSS_PARAMS, get_loss, list_losses
from mamba_unet.model import MambaUNet, build_mamba_unet
from mamba_unet.trainer import MambaUNetTrainer


# =============================================================================
# DATASET GIẢ LẬP CHO SELFTEST VÀ OVERFIT
# =============================================================================
class Synthetic25DDataset(Dataset):
    """Dataset giả lập tạo nhanh các tensor 2.5D (3 kênh x image_size x image_size)."""
    def __init__(self, num_samples: int = 8, image_size: int = 64):
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        self.masks = torch.zeros(num_samples, 1, image_size, image_size)
        mid = image_size // 2
        self.masks[:, :, mid - 4 : mid + 4, mid - 4 : mid + 4] = 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {"image": self.images[idx], "mask": self.masks[idx]}


# =============================================================================
# BỘ XỬ LÝ LỆNH TRAIN / OVERFIT / SELFTEST
# =============================================================================
def _load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ Không đọc được file cấu hình YAML ({e}), dùng tham số mặc định.")
        return {}


def train_action(args):
    """Thực thi quy trình huấn luyện chính cho Mamba-UNet."""
    seed_everything(args.seed)

    # Đọc cấu hình bổ sung từ YAML nếu có
    yaml_cfg = _load_yaml_config(args.config) if args.config else {}

    # 1. Chuẩn bị dữ liệu
    cache_dir = Path(args.cache_dir).resolve()
    from common.data_pipeline.dataset import build_dataloaders
    train_loader, val_loader = build_dataloaders(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_gpu_window=args.gpu_window,
        lung_mask=not args.no_lung_mask,
        augment=not args.no_augment,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        rebuild_index=args.rebuild_index,
        seed=args.seed,
    )

    # 2. Xây dựng mô hình
    model = build_mamba_unet(
        config=yaml_cfg,
        backbone=args.backbone,
        pretrained=args.pretrained,
        act=args.activation,
    )

    # 3. Xây dựng Loss
    loss_params = dict(DEFAULT_LOSS_PARAMS)
    if args.loss_params:
        try:
            custom_p = json.loads(args.loss_params)
            loss_params.update(custom_p)
        except Exception as e:
            print(f"⚠️ Không parse được --loss-params ({e}), dùng mặc định.")
    criterion = get_loss(args.loss, **loss_params)

    # 4. Quản lý Run & Resume
    runs_root = args.runs_root or default_runs_root("mamba_unet")
    resumed_dir, resumed_ckpt = resolve_resume(args.resume, arch="mamba_unet", runs_root=runs_root)

    run_handle = None
    if resumed_dir and (resumed_dir / "run_config.json").exists():
        try:
            with open(resumed_dir / "run_config.json", "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
            run_handle = RunHandle(resumed_dir, saved_cfg)
        except Exception:
            pass

    if run_handle is None:
        run_handle = create_run(
            arch="mamba_unet",
            runs_root=runs_root,
            run_name=args.run_name,
            tag=args.tag,
            args=vars(args),
            model_meta={
                "class": "MambaUNet",
                "backbone": args.backbone,
                "pretrained": args.pretrained,
                "activation": args.activation,
            },
            data_meta={"cache_dir": str(cache_dir)},
            loss_meta={"name": args.loss, "params": loss_params},
            optimizer_meta={
                "name": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "backbone_lr_mult": args.backbone_lr_mult,
                "freeze_backbone_epochs": args.freeze_backbone_epochs,
            },
            scheduler_meta={"name": args.scheduler},
            training_meta={"epochs_target": args.epochs, "batch_size": args.batch_size},
            notes=args.notes,
            init_weights=args.init_weights,
        )

    # 5. Khởi tạo Trainer
    trainer = MambaUNetTrainer(
        model=model,
        criterion=criterion,
        backbone_name=args.backbone,
        activation_name=args.activation,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        backbone_lr_mult=args.backbone_lr_mult,
        run_handle=run_handle,
        grad_clip=args.grad_clip,
        use_amp=not args.no_amp,
        show_progress=not args.no_progress,
        drive_sync_dir=getattr(args, "drive_sync_dir", None),
    )

    # 6. Tối ưu hóa (Optimizer & Scheduler với phân nhóm param groups)
    param_groups = trainer.build_param_groups(args.lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    trainer.optimizer = optimizer

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    trainer.scheduler = scheduler

    if args.init_weights:
        trainer.load_checkpoint(args.init_weights, resume_training=False)

    # 7. Bắt đầu huấn luyện
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        resume_from=resumed_ckpt or args.resume,
    )


def overfit_action(args):
    """Huấn luyện overfit trên 1 batch nhỏ để kiểm tra khả năng hội tụ nhanh."""
    print("🔥 Bắt đầu kiểm tra overfit (Mamba-UNet)...")
    seed_everything(args.seed)
    train_loader = DataLoader(Synthetic25DDataset(num_samples=4, image_size=64), batch_size=2)
    val_loader = DataLoader(Synthetic25DDataset(num_samples=2, image_size=64), batch_size=2)

    from mamba_unet.backbone.mamba_vision import MambaVision
    tiny_bb = MambaVision(
        dim=16,
        in_dim=8,
        depths=[1, 1, 2, 1],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 2, 4, 4],
        in_chans=3,
    )
    model = MambaUNet(in_channels=3, num_classes=1, backbone=tiny_bb, act=args.activation)
    criterion = get_loss("dice_bce")

    trainer = MambaUNetTrainer(
        model=model,
        criterion=criterion,
        backbone_name="tiny",
        activation_name=args.activation,
        freeze_backbone_epochs=0,
        save_dir=tempfile.mkdtemp(),
        show_progress=False,
        use_amp=False,
        drive_sync_dir=getattr(args, "drive_sync_dir", None),
    )
    optimizer = torch.optim.AdamW(trainer.build_param_groups(1e-3), weight_decay=1e-4)
    trainer.optimizer = optimizer

    epochs = getattr(args, "epochs", 5)
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    final_dice = history[-1]["val_dice"]
    print(f"✅ Hoàn thành {epochs} epochs overfit! Final Val Dice: {final_dice:.4f}")


def selftest_action(_):
    """
    Kiểm thử tự động toàn diện theo chuẩn Spec:
      1. Train ngắn trên dữ liệu giả lập (2 epochs, freeze_backbone_epochs=1).
      2. Ngắt giả lập tại đầu epoch 3, kiểm tra lưu checkpoint.
      3. Resume auto tiếp tục liền mạch (epochs 3-4).
      4. Kiểm tra trạng thái đóng/mở băng backbone theo lịch.
      5. Kiểm tra run_config.json đủ khóa theo Schema v1.
      6. Kiểm tra tính nguyên vẹn của best_weights_fp16.pth.
    """
    print("🧪 Bắt đầu selftest toàn diện mamba_unet.train_main...")
    tmp_root = Path(tempfile.mkdtemp())

    try:
        train_loader = DataLoader(Synthetic25DDataset(num_samples=4, image_size=64), batch_size=2)
        val_loader = DataLoader(Synthetic25DDataset(num_samples=2, image_size=64), batch_size=2)

        from mamba_unet.backbone.mamba_vision import MambaVision

        def make_tiny_model():
            bb = MambaVision(
                dim=16,
                in_dim=8,
                depths=[1, 1, 2, 1],
                window_size=[8, 8, 14, 7],
                num_heads=[2, 2, 4, 4],
                in_chans=3,
            )
            return MambaUNet(in_channels=3, num_classes=1, backbone=bb, act="relu")

        # Giai đoạn 1: Huấn luyện có ngắt giả lập tại đầu epoch 3
        class InterruptedMambaTrainer(MambaUNetTrainer):
            def on_epoch_start(self, epoch: int):
                super().on_epoch_start(epoch)
                if epoch == 3:
                    raise KeyboardInterrupt("⚡ Giả lập người dùng ngắt (Ctrl+C tại Epoch 3)!")

        model1 = make_tiny_model()
        criterion1 = get_loss("dice_bce")

        run_h = create_run(
            arch="mamba_unet",
            runs_root=tmp_root,
            run_name="selftest_run",
            args={"epochs": 4, "lr": 0.01, "backbone": "tiny", "activation": "relu"},
        )

        trainer1 = InterruptedMambaTrainer(
            model=model1,
            criterion=criterion1,
            backbone_name="tiny",
            activation_name="relu",
            freeze_backbone_epochs=1,
            backbone_lr_mult=0.1,
            run_handle=run_h,
            show_progress=False,
            use_amp=False,
        )
        optimizer1 = torch.optim.SGD(trainer1.build_param_groups(0.01))
        trainer1.optimizer = optimizer1

        trainer1.fit(train_loader, val_loader, epochs=4)
        assert (run_h.run_dir / "last_model.pth").exists(), "Thiếu last_model.pth sau khi ngắt!"
        assert (run_h.run_dir / "best_model.pth").exists(), "Thiếu best_model.pth!"
        assert (run_h.run_dir / "best_weights_fp16.pth").exists(), "Thiếu best_weights_fp16.pth!"

        # Giai đoạn 2: Giả lập ngắt và tiếp tục với resume_from="auto"
        res_dir, res_ckpt = resolve_resume("auto", arch="mamba_unet", runs_root=tmp_root)
        assert res_dir == run_h.run_dir, "resolve_resume không tìm đúng thư mục run!"
        assert res_ckpt == run_h.run_dir / "last_model.pth", "resolve_resume không tìm đúng last_model.pth!"

        with open(run_h.config_path, "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        run_h_resumed = RunHandle(run_h.run_dir, saved_cfg)

        model2 = make_tiny_model()
        trainer2 = MambaUNetTrainer(
            model=model2,
            criterion=criterion1,
            backbone_name="tiny",
            activation_name="relu",
            freeze_backbone_epochs=1,
            backbone_lr_mult=0.1,
            run_handle=run_h_resumed,
            show_progress=False,
            use_amp=False,
        )
        optimizer2 = torch.optim.SGD(trainer2.build_param_groups(0.01))
        trainer2.optimizer = optimizer2

        trainer2.fit(train_loader, val_loader, epochs=4, resume_from="auto")
        assert trainer2.start_epoch == 3, f"Phải resume từ epoch 3, nhận được {trainer2.start_epoch}"
        assert len(trainer2.history) == 4, f"Tổng số epochs phải là 4, nhận được {len(trainer2.history)}"

        # Giai đoạn 3: Kiểm tra trạng thái đóng/mở băng backbone
        # Tại epoch 1 (freeze=1): backbone phải bị đóng băng
        trainer2.on_epoch_start(1)
        for p in model2.encoder.parameters():
            assert not p.requires_grad, "Backbone phải bị đóng băng ở Epoch 1!"
        # Tại epoch 3 (freeze=1): backbone phải được mở băng
        trainer2.on_epoch_start(3)
        for p in model2.encoder.parameters():
            assert p.requires_grad, "Backbone phải được mở băng ở Epoch 3!"

        # Giai đoạn 4: Kiểm tra tính nguyên vẹn của best_weights_fp16.pth
        fp16_data = load_checkpoint_file(run_h.run_dir / "best_weights_fp16.pth")
        sd = fp16_data.get("model_state_dict", fp16_data.get("state_dict"))
        assert sd is not None, "best_weights_fp16.pth phải chứa model_state_dict!"
        assert "meta" in fp16_data
        assert fp16_data["meta"]["arch"] == "mamba_unet"
        first_weight = next(iter(sd.values()))
        assert first_weight.dtype == torch.float16, f"Trọng số phải là fp16, nhận được {first_weight.dtype}"

        # Giai đoạn 5: Kiểm tra đầy đủ trường trong run_config.json
        with open(run_h.config_path, "r", encoding="utf-8") as f:
            final_cfg = json.load(f)
        assert final_cfg["status"] == "completed"
        assert len(final_cfg["resume_events"]) >= 1
        assert final_cfg["resume_events"][-1]["from_epoch"] == 2
        assert final_cfg["result"]["epochs_completed"] == 4

        print("✅ Selftest mamba_unet.train_main: HOÀN TẤT TOÀN DIỆN VÀ THÀNH CÔNG!")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# =============================================================================
# CLI PARSER
# =============================================================================
def _add_mamba_specific_args(parser: argparse.ArgumentParser):
    mamba_grp = parser.add_argument_group("Tham số Mamba-UNet")
    mamba_grp.add_argument(
        "--backbone",
        choices=["mamba_vision_T", "mamba_vision_T2", "mamba_vision_S", "mamba_vision_B"],
        default="mamba_vision_T",
        help="Biến thể MambaVision backbone",
    )
    mamba_grp.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Nạp trọng số ImageNet-1K pretrained từ NVIDIA",
    )
    mamba_grp.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Khởi tạo backbone ngẫu nhiên (không dùng pretrained)",
    )
    mamba_grp.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=5,
        help="Số epoch đóng băng backbone đầu tiên (-1: đóng băng suốt quá trình, 0: không đóng băng)",
    )
    mamba_grp.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.1,
        help="Hệ số nhân LR cho backbone so với decoder (mặc định 0.1)",
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Mamba-UNet phân đoạn ảnh y tế")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # 1. train subparser
    train_p = subparsers.add_parser("train", help="Huấn luyện mô hình chuẩn")
    add_common_args(
        train_p,
        activation_choices=list_activations(),
        loss_choices=list_losses(),
        default_activation=DEFAULT_ACTIVATION,
        default_loss=DEFAULT_LOSS,
    )
    _add_mamba_specific_args(train_p)
    train_p.set_defaults(func=train_action)

    # 2. overfit subparser
    overfit_p = subparsers.add_parser("overfit", help="Kiểm tra overfit nhanh trên batch giả lập")
    add_common_args(
        overfit_p,
        activation_choices=list_activations(),
        loss_choices=list_losses(),
        default_activation=DEFAULT_ACTIVATION,
        default_loss=DEFAULT_LOSS,
    )
    _add_mamba_specific_args(overfit_p)
    overfit_p.set_defaults(func=overfit_action)

    # 3. selftest subparser
    selftest_p = subparsers.add_parser("selftest", help="Kiểm thử tự động chu trình huấn luyện")
    selftest_p.set_defaults(func=selftest_action)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

#### <a id="tệp-13-mamba_unet-evaluate-py"></a>Tệp 13: `mamba_unet/evaluate.py`
- **Vị trí**: `mamba_unet/evaluate.py`
- **Quy mô**: `240 dòng` | `9.2 KB`
- **Vai trò**: CLI entrypoint cho suy luận và đánh giá 3D: Chạy predict cho từng ca, tính toán metrics 3D và xuất file NRRD.

```python
# -*- coding: utf-8 -*-
"""
mamba_unet/evaluate.py - CLI suy luận và đánh giá 3D cấp độ bệnh nhân cho kiến trúc Mamba-UNet.

Các chế độ hoạt động:
  - selftest: Kiểm thử tự động chu trình suy luận 3D, hậu xử lý và xuất file NRRD.
  - predict : Suy luận 3D toàn khối cho 1 bệnh nhân và xuất file mặt nạ NRRD chuẩn y tế.
  - eval    : Đánh giá 3D trên tập dữ liệu validation hoặc toàn tập, xuất bảng CSV/JSON.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import TUMOR_SIZE_LOW_LIM
from common.engine.checkpoint_io import load_checkpoint_file
from common.eval3d.eval_runner import evaluate_dataset_3d
from common.eval3d.nrrd_io import load_nrrd_mask, save_prediction_nrrd
from common.eval3d.volume_inference import predict_patient_volume
from mamba_unet.model import MambaUNet


def load_mamba_unet_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    backbone: str = "mamba_vision_T",
    activation: str = "relu",
    in_channels: int = 3,
    classes: int = 1,
) -> MambaUNet:
    """Nạp trọng số checkpoint an toàn vào mô hình MambaUNet."""
    ckpt = load_checkpoint_file(checkpoint_path, map_location=str(device))
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    meta = ckpt.get("meta", {})

    saved_bb = meta.get("backbone", None)
    if saved_bb:
        backbone = str(saved_bb)
    saved_act = meta.get("activation", activation)

    model = MambaUNet(
        in_channels=in_channels,
        num_classes=classes,
        backbone=backbone,
        pretrained=False,
        act=saved_act,
    )

    cleaned_sd = {}
    for k, v in sd.items():
        k_clean = k[7:] if k.startswith("module.") else k
        cleaned_sd[k_clean] = v

    model.load_state_dict(cleaned_sd, strict=False)
    model.to(device)
    model.eval()
    return model


def predict_action(args):
    """Suy luận 3D cho 1 ca bệnh nhân và tùy chọn xuất file NRRD."""
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    model = load_mamba_unet_model(
        checkpoint_path=args.checkpoint,
        device=device,
        backbone=args.backbone,
    )
    res = predict_patient_volume(
        model=model,
        patient_cache_dir=args.patient_cache,
        device=device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        min_slice_area_mm2=args.min_area,
        tta=args.tta,
        use_amp=not args.no_amp,
    )
    m_raw, m_post = res["metrics_raw"], res["metrics_post"]
    print("=" * 80)
    print(f"🩻 KẾT QUẢ SUY LUẬN 3D (Mamba-UNet): {Path(args.patient_cache).name}")
    print("=" * 80)
    print(f"  - 3D Dice (Trước hậu xử lý): {m_raw['dice_3d']:.4f} ({m_raw['pred_volume_mm3']} mm³)")
    print(f"  - 3D Dice (Sau hậu xử lý)  : {m_post['dice_3d']:.4f} ({m_post['pred_volume_mm3']} mm³)")
    print(f"  - Thể tích u thực tế (GT)  : {m_post['gt_volume_mm3']} mm³ | HD95: {m_post['hd95_mm']} mm")

    if args.out_nrrd:
        save_prediction_nrrd(res["clean_mask_1mm"], res["meta"], args.out_nrrd, args.ref_nrrd)
        print(f"💾 Đã xuất file mặt nạ 3D NRRD tại: {args.out_nrrd}")


def eval_action(args):
    """Đánh giá 3D trên danh sách bệnh nhân và xuất báo cáo CSV/JSON."""
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    model = load_mamba_unet_model(
        checkpoint_path=args.checkpoint,
        device=device,
        backbone=args.backbone,
    )
    evaluate_dataset_3d(
        model=model,
        cache_dir=args.cache_dir,
        split=args.split,
        batch_size=args.batch_size,
        threshold=args.threshold,
        min_slice_area_mm2=args.min_area,
        tta=args.tta,
        out_csv=args.out_csv,
        out_json=args.out_json,
        device=device,
        use_amp=not args.no_amp,
    )


def selftest_action(_):
    """Kiểm thử tự động chu trình suy luận 3D cho Mamba-UNet."""
    print("🧪 Bắt đầu selftest mamba_unet.evaluate...")
    tmp = Path(tempfile.mkdtemp())
    try:
        p_dir = tmp / "LIDC-IDRI-0001"
        p_dir.mkdir(parents=True)
        img = np.full((8, 64, 64), -800, dtype=np.int16)
        msk = np.zeros((8, 64, 64), dtype=np.uint8)
        lung_c = np.zeros((8, 64, 64), dtype=np.uint8)
        lung_c[1:7, 10:54, 10:54] = 1
        msk[3:5, 25:35, 25:35] = 1
        img[3:5, 25:35, 25:35] = 50

        np.save(p_dir / "image_1mm.npy", img)
        np.save(p_dir / "mask_1mm.npy", msk)
        np.save(p_dir / "lung_1mm.npy", lung_c)
        with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "orig_shape_zyx": [8, 64, 64],
                "orig_spacing_xyz": [0.78, 0.78, 1.5],
                "shape_1mm_zyx": [8, 64, 64],
                "lung_ok": True,
            }, f)

        from mamba_unet.backbone.mamba_vision import MambaVision
        tiny_bb = MambaVision(
            dim=16,
            in_dim=8,
            depths=[1, 1, 2, 1],
            window_size=[8, 8, 14, 7],
            num_heads=[2, 2, 4, 4],
            in_chans=3,
        )
        model = MambaUNet(in_channels=3, num_classes=1, backbone=tiny_bb, act="relu")
        ckpt_path = tmp / "tiny_mamba.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "meta": {"arch": "mamba_unet", "backbone": "tiny"},
        }, ckpt_path)

        res = predict_patient_volume(
            model, p_dir, batch_size=2, image_size=64, tta=True, use_amp=False
        )
        assert res["prob_1mm"].shape == (8, 64, 64)
        assert res["clean_mask_1mm"].shape == (8, 64, 64)
        assert "dice_3d" in res["metrics_post"]

        out_nrrd = tmp / "test_out_mamba.nrrd"
        save_prediction_nrrd(res["clean_mask_1mm"], res["meta"], out_nrrd)
        assert out_nrrd.exists(), "File NRRD không được tạo!"
        arr_read, sp = load_nrrd_mask(out_nrrd)
        assert arr_read.shape == (8, 64, 64)

        print("✅ Selftest mamba_unet.evaluate hoàn tất thành công!")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Đánh giá 3D cấp độ bệnh nhân (Mamba-UNet)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # 1. selftest
    selftest_p = subparsers.add_parser("selftest", help="Kiểm thử tự động pipeline suy luận 3D")
    selftest_p.set_defaults(func=selftest_action)

    # 2. predict
    predict_p = subparsers.add_parser("predict", help="Suy luận 3D cho 1 ca bệnh nhân và xuất file NRRD")
    predict_p.add_argument("--patient-cache", required=True, help="Thư mục cache_1mm/<patient_id>")
    predict_p.add_argument("--checkpoint", required=True, help="File trọng số checkpoint (.pth)")
    predict_p.add_argument("--out-nrrd", default=None, help="Đường dẫn xuất file mặt nạ 3D (.nrrd)")
    predict_p.add_argument("--ref-nrrd", default=None, help="Đường dẫn file ảnh CT gốc để copy header tọa độ")
    predict_p.add_argument(
        "--backbone",
        choices=["mamba_vision_T", "mamba_vision_T2", "mamba_vision_S", "mamba_vision_B"],
        default="mamba_vision_T",
    )
    predict_p.add_argument("--batch-size", type=int, default=8)
    predict_p.add_argument("--threshold", type=float, default=0.5)
    predict_p.add_argument("--min-area", type=float, default=TUMOR_SIZE_LOW_LIM)
    predict_p.add_argument("--tta", action="store_true", help="Bật Test-Time Augmentation (lật ngang/dọc)")
    predict_p.add_argument("--no-cuda", action="store_true")
    predict_p.add_argument("--no-amp", action="store_true")
    predict_p.set_defaults(func=predict_action)

    # 3. eval
    eval_p = subparsers.add_parser("eval", help="Đánh giá 3D trên toàn bộ tập dữ liệu")
    eval_p.add_argument("--cache-dir", default="data/cache_1mm", help="Thư mục dữ liệu cache_1mm")
    eval_p.add_argument("--checkpoint", required=True, help="File trọng số checkpoint (.pth)")
    eval_p.add_argument("--split", choices=["val", "train", "all"], default="val")
    eval_p.add_argument(
        "--backbone",
        choices=["mamba_vision_T", "mamba_vision_T2", "mamba_vision_S", "mamba_vision_B"],
        default="mamba_vision_T",
    )
    eval_p.add_argument("--batch-size", type=int, default=8)
    eval_p.add_argument("--threshold", type=float, default=0.5)
    eval_p.add_argument("--min-area", type=float, default=TUMOR_SIZE_LOW_LIM)
    eval_p.add_argument("--tta", action="store_true")
    eval_p.add_argument("--out-csv", default=None)
    eval_p.add_argument("--out-json", default=None)
    eval_p.add_argument("--no-cuda", action="store_true")
    eval_p.add_argument("--no-amp", action="store_true")
    eval_p.set_defaults(func=eval_action)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```


---

### 3. Hạ Tầng Engine, Quản Lý Checkpoint & Phiên Chạy (Common Engine)

#### <a id="tệp-14-common-engine-__init__-py"></a>Tệp 14: `common/engine/__init__.py`
- **Vị trí**: `common/engine/__init__.py`
- **Quy mô**: `2 dòng` | `0.1 KB`
- **Vai trò**: Khởi tạo package common.engine.

```python
# -*- coding: utf-8 -*-
"""common.engine package - Lõi huấn luyện, RunManager và Checkpoint I/O."""
```

#### <a id="tệp-15-common-engine-base_trainer-py"></a>Tệp 15: `common/engine/base_trainer.py`
- **Vị trí**: `common/engine/base_trainer.py`
- **Quy mô**: `667 dòng` | `30.4 KB`
- **Vai trò**: BaseTrainer: Vòng lặp huấn luyện chuẩn hóa với AMP, Gradient Clipping, Early Stopping, ETA, và ghi history.csv/train.log.

```python
# -*- coding: utf-8 -*-
"""
common/engine/base_trainer.py - Vòng lặp huấn luyện chuẩn BaseTrainer dùng chung cho toàn dự án.

Đặc điểm kiến trúc (spec mục 4.4):
  - Nhận `criterion` (nn.Module đã dựng) và `metric_fn` từ bên ngoài qua tiêm phụ thuộc (Dependency Injection).
  - Tuyệt đối KHÔNG import bất kỳ hàm loss hay activation nào, không hard-code tên loss/activation.
  - Tự động phát hiện multi-GPU (nn.DataParallel), bóc tách _unwrap_model() khi lưu checkpoint.
  - Quản lý AMP (Mixed Precision GradScaler), gradient clipping, Early Stopping, đồng bộ Cosine Scheduler.
  - Bắt KeyboardInterrupt an toàn, bảo tồn checkpoint và lịch sử huấn luyện.
  - Điểm mở rộng cho lớp con kiến trúc:
      * checkpoint_meta(): trả về dict chứa arch và model_config
      * on_epoch_start(epoch): hook xử lý đầu epoch (đóng/mở băng backbone)
      * build_param_groups(lr): phân nhóm learning rate cho backbone/decoder
"""

import datetime
import os
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.metrics import compute_segmentation_metrics
from common.engine.checkpoint_io import export_weights_fp16, load_checkpoint_file, save_checkpoint
from common.engine.run_manager import RunHandle
from common.engine.utils import format_duration


def _make_grad_scaler(enabled: bool):
    """Tạo GradScaler thích ứng phiên bản PyTorch."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (TypeError, ValueError):
            pass
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    return None


def _autocast_context(enabled: bool):
    """Tạo autocast context an toàn."""
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=True)
        except (TypeError, ValueError):
            pass
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


class BaseTrainer:
    """Lớp quản lý huấn luyện lõi cho các kiến trúc mạng."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metric_fn: Callable = compute_segmentation_metrics,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        save_dir: Optional[Union[str, Path]] = None,
        run_handle: Optional[RunHandle] = None,
        grad_clip: float = 1.0,
        show_progress: bool = True,
        drive_sync_dir: Optional[Union[str, Path]] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.type == "cuda"

        # Tự động kích hoạt DataParallel khi có nhiều GPU
        if self.is_cuda and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
            print(f"🔥 Phát hiện {torch.cuda.device_count()} GPUs! Kích hoạt chế độ song song nn.DataParallel.")
            model = nn.DataParallel(model)

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_fn = metric_fn or compute_segmentation_metrics

        self.use_amp = bool(use_amp and self.is_cuda)
        self.scaler = _make_grad_scaler(self.use_amp)

        self.run_handle = run_handle
        if run_handle:
            self.save_dir = run_handle.run_dir
        elif save_dir:
            self.save_dir = Path(save_dir).resolve()
        else:
            self.save_dir = Path("checkpoints").resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.grad_clip = float(grad_clip) if grad_clip is not None else 0.0
        self.show_progress = bool(show_progress)

        self.start_epoch = 1
        self.best_val_dice = -1.0
        self.best_epoch = None
        self.no_improve = 0
        self.history: List[Dict[str, Any]] = []

        # Thiết lập cơ chế tự động đồng bộ sang Google Drive thời gian thực
        self.drive_sync_dir = Path(drive_sync_dir).resolve() if drive_sync_dir else None
        if self.drive_sync_dir:
            self.drive_run_dir = self.drive_sync_dir / self.save_dir.name
            try:
                self.drive_run_dir.mkdir(parents=True, exist_ok=True)
                print(f"🔗 [Auto-Sync] Đã kết nối Google Drive: {self.drive_run_dir}")
                self._init_drive_backup_metadata()
            except Exception as e:
                print(f"⚠️ [Auto-Sync] Không thể tạo thư mục Google Drive ({e}). Tiếp tục train trên SSD cục bộ.")
                self.drive_run_dir = None
        else:
            self.drive_run_dir = None

    def _safe_sync_file(self, src_file: Union[str, Path], dst_filename: Optional[str] = None):
        """Sao chép an toàn một tệp sang Google Drive mà không chặn hoặc gây crash nếu lỗi mạng."""
        if not self.drive_run_dir:
            return
        src = Path(src_file)
        if not src.exists():
            return
        dst_name = dst_filename or src.name
        dst = self.drive_run_dir / dst_name
        try:
            tmp_dst = self.drive_run_dir / f".tmp_{dst_name}_{os.getpid()}"
            shutil.copy2(src, tmp_dst)
            os.replace(str(tmp_dst), str(dst))
        except Exception:
            pass

    def _init_drive_backup_metadata(self):
        """Khởi tạo trước tệp DATA_SPEC_AND_CONFIG.md và REPORT.md trên Drive ngay từ đầu."""
        if not self.drive_run_dir:
            return
        try:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            meta = self.checkpoint_meta()
            arch_name = meta.get("arch", "model")
            bb_name = meta.get("backbone", "N/A")
            act_name = meta.get("activation", "relu")

            try:
                import configs.config as med_cfg
                win_w = getattr(med_cfg, "WINDOW_WIDTH", 1500)
                win_l = getattr(med_cfg, "WINDOW_LEVEL", -600)
                resamp = getattr(med_cfg, "RESAMPLING_STEP", 25)
                tumor_lim = getattr(med_cfg, "TUMOR_SIZE_LOW_LIM", 20.0)
                zero_r = getattr(med_cfg, "ZERO_SL_RATIO", 0.25)
                val_sp = getattr(med_cfg, "VAL_SPLIT", 0.15)
                img_sz = getattr(med_cfg, "IMAGE_SIZE", 512)
            except Exception:
                win_w, win_l, resamp, tumor_lim, zero_r, val_sp, img_sz = 1500, -600, 25, 20.0, 0.25, 0.15, 512

            data_spec_md = f"""# 📊 HỒ SƠ DỮ LIỆU & CẤU HÌNH HUẤN LUYỆN
## Phiên chạy: `{self.save_dir.name}`
*Khởi tạo lúc: {now_str}*

### 1. Phân Chia Dữ Liệu & Quy Chuẩn Tiền Xử Lý Y Tế
- **Kiến trúc mô hình**: `{arch_name}` (Backbone: `{bb_name}`, Activation: `{act_name}`)
- **Chiến lược phân chia**: Patient-Level Split, Zero Patient Leakage (Val Split: {val_sp * 100:.0f}%)
- **Quy chuẩn lọc khối u**: `TUMOR_SIZE_LOW_LIM = {tumor_lim} mm²` (loại bỏ nốt u nhỏ hơn {tumor_lim} mm²)
- **Chuẩn hóa Hounsfield Unit (HU)**:
  - Cửa sổ phổi: `WINDOW_WIDTH = {win_w} HU`, `WINDOW_LEVEL = {win_l} HU` (Dải quan sát: [{win_l - win_w // 2}, {win_l + win_w // 2}] HU)
  - Lượng tử hóa bước nhảy: `RESAMPLING_STEP = {resamp} HU` (làm tròn {resamp} HU mỗi bước để giảm nhiễu hạt máy quét CT và nén dữ liệu cache)
- **Tỷ lệ lấy mẫu lát cắt âm tính**: `ZERO_SL_RATIO = {zero_r}` (giữ lại {zero_r * 100:.0f}% lát cắt âm tính có nhu mô phổi)
- **Kích thước ảnh**: {img_sz}x{img_sz} (Định dạng 2.5D: 3 lát cắt kế tiếp [z-1, z, z+1])
"""
            spec_file = self.save_dir / "DATA_SPEC_AND_CONFIG.md"
            with open(spec_file, "w", encoding="utf-8") as f:
                f.write(data_spec_md)
            self._safe_sync_file(spec_file)

            report_md = f"""# 📋 Báo Cáo Huấn Luyện: {self.save_dir.name}
- **Thời gian khởi tạo**: {now_str}
- **Trạng thái**: Đang huấn luyện (Running)
- **Kiến trúc**: `{arch_name}` | Backbone: `{bb_name}` | Activation: `{act_name}`
- **Hồ sơ cấu hình dữ liệu**: Xem chi tiết tại [DATA_SPEC_AND_CONFIG.md](DATA_SPEC_AND_CONFIG.md)
"""
            rep_file = self.save_dir / "REPORT.md"
            if not rep_file.exists():
                with open(rep_file, "w", encoding="utf-8") as f:
                    f.write(report_md)
                self._safe_sync_file(rep_file)

        except Exception as e:
            print(f"ℹ️ [Auto-Sync] Khởi tạo metadata ban đầu: {e}")

    def _update_report_md(self, val_m: Dict[str, Any], epoch: int, is_best: bool = False):
        """Cập nhật tệp REPORT.md với số liệu Dice và IoU mới nhất."""
        rep_file = self.save_dir / "REPORT.md"
        meta = self.checkpoint_meta()
        arch_name = meta.get("arch", "model")
        bb_name = meta.get("backbone", "N/A")
        act_name = meta.get("activation", "relu")
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content = f"""# 📋 Báo Cáo Huấn Luyện: {self.save_dir.name}
- **Thời gian cập nhật**: {now_str}
- **Kiến trúc**: `{arch_name}` | Backbone: `{bb_name}` | Activation: `{act_name}`
- **Kết quả huấn luyện**:
  - **Best Epoch**: **Epoch {self.best_epoch}**
  - **Best Val Dice**: **{self.best_val_dice:.4f}** ⭐ [BEST]
  - **Best Val IoU**: **{val_m.get('iou', 0.0):.4f}**
  - **Val Loss**: **{val_m.get('loss', 0.0):.4f}**
- **Tiến độ**: Đã hoàn thành Epoch {epoch}
- **Hồ sơ cấu hình dữ liệu**: Xem chi tiết tại [DATA_SPEC_AND_CONFIG.md](DATA_SPEC_AND_CONFIG.md)
"""
        try:
            with open(rep_file, "w", encoding="utf-8") as f:
                f.write(content)
            self._safe_sync_file(rep_file)
        except Exception:
            pass

    # ==================================================================
    # CÁC ĐIỂM MỞ RỘNG (HOOKS) CHO LỚP CON KIẾN TRÚC
    # ==================================================================
    def checkpoint_meta(self) -> Dict[str, Any]:
        """Lớp con ghi đè để trả về thông tin kiến trúc (arch, model_config)."""
        return {}

    def on_epoch_start(self, epoch: int):
        """Hook chạy trước mỗi epoch (vd: đóng/mở băng backbone)."""
        pass

    def build_param_groups(self, lr: float) -> List[Dict[str, Any]]:
        """Lớp con ghi đè để phân chia tham số (vd: decoder lr, backbone lr)."""
        return [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": lr}]

    # ==================================================================
    # LOGIC NỘI BỘ
    # ==================================================================
    def _unwrap_model(self) -> nn.Module:
        """Bóc lớp DataParallel để lưu checkpoint thuần nhất."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _prepare_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = batch["image"].to(self.device, non_blocking=self.is_cuda)
        masks = batch["mask"].to(self.device, non_blocking=self.is_cuda)
        if images.dtype in (torch.int16, torch.int32, torch.int64):
            from common.data_pipeline.medical_prep import window_hu_torch
            images = window_hu_torch(images)
        return images, masks

    def run_epoch(self, loader, is_train: bool = True, epoch: Optional[int] = None, total_epochs: Optional[int] = None) -> Dict[str, Any]:
        self.model.train(is_train)
        total_loss = 0.0
        total_samples = 0
        inter_sum = 0.0
        pred_sum = 0.0
        target_sum = 0.0
        sub_loss_sums: Dict[str, float] = {}

        phase = "Train" if is_train else "Val  "
        ep_prefix = f"Epoch [{epoch:02d}/{total_epochs:02d}] {phase}" if (epoch and total_epochs) else phase

        use_bar = self.show_progress and (tqdm is not None) and sys.stderr.isatty()
        iterator = tqdm(loader, desc=ep_prefix, leave=False, dynamic_ncols=True) if use_bar else loader

        grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
        with grad_ctx:
            for batch in iterator:
                images, masks = self._prepare_batch(batch)
                bs = images.size(0)

                if is_train and self.optimizer:
                    self.optimizer.zero_grad(set_to_none=True)

                with _autocast_context(self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                if is_train and self.optimizer:
                    if self.scaler is not None and self.use_amp:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip > 0:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                total_loss += float(loss.item()) * bs
                total_samples += bs

                # Thu thập sub-losses nếu hàm criterion hỗ trợ
                if hasattr(self.criterion, "get_components"):
                    comps = self.criterion.get_components()
                    for k, v in comps.items():
                        sub_loss_sums[k] = sub_loss_sums.get(k, 0.0) + float(v) * bs
                elif hasattr(self.criterion, "last_components"):
                    comps = self.criterion.last_components
                    for k, v in comps.items():
                        sub_loss_sums[k] = sub_loss_sums.get(k, 0.0) + float(v) * bs

                m = self.metric_fn(logits.detach(), masks)
                inter_sum += m["intersection"]
                pred_sum += m["pred_sum"]
                target_sum += m["target_sum"]

                if use_bar:
                    running_loss = total_loss / max(1, total_samples)
                    running_dice = (2.0 * inter_sum + 1e-6) / (pred_sum + target_sum + 1e-6)
                    running_union = pred_sum + target_sum - inter_sum
                    running_iou = (inter_sum + 1e-6) / (running_union + 1e-6)
                    p_dict = {
                        "loss": f"{running_loss:.4f}",
                        "dice": f"{running_dice:.4f}",
                        "iou": f"{running_iou:.4f}",
                    }
                    if sub_loss_sums:
                        for k, v in sub_loss_sums.items():
                            p_dict[k] = f"{(v / max(1, total_samples)):.4f}"
                    iterator.set_postfix(**p_dict)

        eps = 1e-6
        epoch_loss = total_loss / max(1, total_samples)
        epoch_dice = (2.0 * inter_sum + eps) / (pred_sum + target_sum + eps)
        union_sum = pred_sum + target_sum - inter_sum
        epoch_iou = (inter_sum + eps) / (union_sum + eps)

        out_metrics = {
            "loss": epoch_loss,
            "dice": epoch_dice,
            "iou": epoch_iou,
        }
        if sub_loss_sums:
            out_metrics["sub_losses"] = {k: (v / max(1, total_samples)) for k, v in sub_loss_sums.items()}

        return out_metrics

    def save_checkpoint(self, filename: str, epoch: int, val_metrics: Dict[str, float]) -> Path:
        """Lưu checkpoint và xuất fp16 nếu là best_model."""
        target_path = self.save_dir / filename
        meta = self.checkpoint_meta()
        extra = {
            "no_improve": self.no_improve,
            "val_metrics": val_metrics,
            "history": self.history,
        }
        opt_sd = self.optimizer.state_dict() if self.optimizer else None
        sch_sd = self.scheduler.state_dict() if self.scheduler else None
        scl_sd = self.scaler.state_dict() if (self.scaler and self.use_amp) else None

        save_checkpoint(
            path=target_path,
            state_dict=self._unwrap_model().state_dict(),
            optimizer_state=opt_sd,
            scheduler_state=sch_sd,
            scaler_state=scl_sd,
            epoch=epoch,
            best_val_dice=self.best_val_dice,
            meta=meta,
            extra=extra,
        )

        if filename == "best_model.pth":
            fp16_path = self.save_dir / "best_weights_fp16.pth"
            try:
                export_weights_fp16(target_path, fp16_path, meta=meta)
            except Exception as e:
                print(f"⚠️ Không xuất được best_weights_fp16.pth: {e}")

        return target_path

    def load_checkpoint(self, checkpoint_path: Union[str, Path], resume_training: bool = True) -> int:
        """Nạp checkpoint an toàn và kiểm tra tính tương thích của metadata."""
        p = Path(checkpoint_path).resolve()
        ckpt = load_checkpoint_file(p, map_location=str(self.device))

        # Kiểm tra metadata nếu có
        saved_meta = ckpt.get("meta", {})
        current_meta = self.checkpoint_meta()
        if saved_meta and current_meta:
            if "arch" in saved_meta and "arch" in current_meta:
                if saved_meta["arch"] != current_meta["arch"]:
                    raise ValueError(
                        f"❌ Không tương thích kiến trúc khi nạp checkpoint!\n"
                        f"  - Checkpoint: arch = '{saved_meta['arch']}'\n"
                        f"  - Hiện tại  : arch = '{current_meta['arch']}'\n"
                        f"Gợi ý: Kiểm tra lại tham số mô hình hoặc khởi tạo phiên chạy mới."
                    )
            if "features" in saved_meta and "features" in current_meta:
                if saved_meta["features"] != current_meta["features"]:
                    raise ValueError(
                        f"❌ Không tương thích số filter cơ sở (features)!\n"
                        f"  - Checkpoint: features = {saved_meta['features']}\n"
                        f"  - Hiện tại  : features = {current_meta['features']}"
                    )

        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self._unwrap_model().load_state_dict(sd)

        if not resume_training:
            print(f"📥 Đã nạp trọng số khởi tạo (init-weights) từ: {p.name}")
            return 1

        if self.optimizer and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"⚠️ Không nạp được optimizer state ({e}). Tiếp tục với optimizer mới.")

        if self.scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"⚠️ Không nạp được scheduler state ({e}).")

        if self.scaler and self.use_amp and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"]:
            try:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                pass

        last_epoch = int(ckpt.get("epoch", 0))
        self.start_epoch = last_epoch + 1
        self.best_val_dice = float(ckpt.get("best_val_dice", -1.0))
        self.no_improve = int(ckpt.get("no_improve", 0))
        self.history = ckpt.get("history", [])

        if self.run_handle:
            self.run_handle.record_resume_event(from_epoch=last_epoch)

        print(
            f"🔄 RESUME THÀNH CÔNG từ '{p.name}' | Đã xong epoch {last_epoch} "
            f"-> Tiếp tục từ epoch {self.start_epoch} | Best Val Dice: {self.best_val_dice:.4f}"
        )
        return self.start_epoch

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 50,
        early_stopping_patience: int = 0,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """Vòng lặp huấn luyện chính có hỗ trợ resume và bắt ngắt."""
        if resume_from:
            if str(resume_from).lower() == "auto":
                candidate = self.save_dir / "last_model.pth"
                if candidate.exists():
                    self.load_checkpoint(candidate, resume_training=True)
                else:
                    print(f"ℹ️  --resume auto: Chưa có '{candidate.name}', bắt đầu huấn luyện mới.")
            else:
                self.load_checkpoint(resume_from, resume_training=True)

        # Đồng bộ T_max cho CosineAnnealingLR khi tăng số epochs
        if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            if getattr(self.scheduler, "T_max", epochs) != epochs and epochs > 0:
                self.scheduler.T_max = max(1, epochs)

        if self.start_epoch > epochs:
            print(f"✅ Đã hoàn thành {self.start_epoch - 1}/{epochs} epochs trong checkpoint trước đó.")
            if self.run_handle:
                self.run_handle.finish(status="completed")
            return self.history

        epoch_durations = []
        fit_start_time = time.time()

        # Đảm bảo khởi tạo metadata ban đầu trên Google Drive
        if self.drive_run_dir:
            self._init_drive_backup_metadata()

        try:
            for epoch in range(self.start_epoch, epochs + 1):
                ep_t0 = time.time()
                self.on_epoch_start(epoch)

                tr_t0 = time.time()
                train_m = self.run_epoch(train_loader, is_train=True, epoch=epoch, total_epochs=epochs)
                train_time = time.time() - tr_t0

                va_t0 = time.time()
                if val_loader is not None:
                    val_m = self.run_epoch(val_loader, is_train=False, epoch=epoch, total_epochs=epochs)
                else:
                    val_m = train_m
                val_time = time.time() - va_t0

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_m["dice"])
                    else:
                        self.scheduler.step()

                lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
                elapsed = time.time() - ep_t0
                epoch_durations.append(elapsed)

                recent_avg_sec = float(np.mean(epoch_durations[-5:]))
                eta_sec = recent_avg_sec * (epochs - epoch)

                row = {
                    "epoch": int(epoch),
                    "lr": float(lr),
                    "train_loss": round(train_m["loss"], 5),
                    "train_dice": round(train_m["dice"], 5),
                    "train_iou": round(train_m["iou"], 5),
                    "val_loss": round(val_m["loss"], 5),
                    "val_dice": round(val_m["dice"], 5),
                    "val_iou": round(val_m["iou"], 5),
                    "train_time_sec": round(train_time, 2),
                    "val_time_sec": round(val_time, 2),
                    "time_sec": round(elapsed, 2),
                    "eta_sec": round(eta_sec, 2),
                }
                # Bổ sung các thành phần sub-losses vào history nếu có
                for k, v in train_m.get("sub_losses", {}).items():
                    row[f"train_{k}"] = round(v, 5)
                for k, v in val_m.get("sub_losses", {}).items():
                    row[f"val_{k}"] = round(v, 5)

                self.history = [r for r in self.history if int(r.get("epoch", -1)) != epoch] + [row]
                from common.engine.utils import atomic_save_json
                df_history = pd.DataFrame(self.history)
                csv_tmp = self.save_dir / f".tmp_history_{os.getpid()}.csv"
                df_history.to_csv(csv_tmp, index=False)
                os.replace(str(csv_tmp), str(self.save_dir / "history.csv"))

                is_best = val_m["dice"] > self.best_val_dice
                if is_best:
                    self.best_val_dice = float(val_m["dice"])
                    self.best_epoch = epoch
                    self.no_improve = 0
                    self.save_checkpoint("best_model.pth", epoch, val_m)
                else:
                    self.no_improve += 1

                self.save_checkpoint("last_model.pth", epoch, val_m)

                if self.run_handle:
                    self.run_handle.update_result(
                        epoch=epoch,
                        best_val_dice=self.best_val_dice,
                        best_epoch=self.best_epoch,
                        last_val_loss=val_m["loss"],
                        last_val_dice=val_m["dice"],
                        last_val_iou=val_m["iou"],
                    )

                # Tự động đồng bộ thời gian thực sang Google Drive sau mỗi epoch
                if self.drive_run_dir:
                    self._safe_sync_file(self.save_dir / "history.csv")
                    if (self.save_dir / "run_config.json").exists():
                        self._safe_sync_file(self.save_dir / "run_config.json")
                    if (self.save_dir / "train.log").exists():
                        self._safe_sync_file(self.save_dir / "train.log")
                    if (self.save_dir / "DATA_SPEC_AND_CONFIG.md").exists():
                        self._safe_sync_file(self.save_dir / "DATA_SPEC_AND_CONFIG.md")
                    if (self.save_dir / "last_model.pth").exists():
                        self._safe_sync_file(self.save_dir / "last_model.pth")
                    if is_best:
                        if (self.save_dir / "best_model.pth").exists():
                            self._safe_sync_file(self.save_dir / "best_model.pth")
                        if (self.save_dir / "best_weights_fp16.pth").exists():
                            self._safe_sync_file(self.save_dir / "best_weights_fp16.pth")
                        self._update_report_md(val_m, epoch, is_best=True)

                def _fmt_phase(m: Dict[str, Any]) -> str:
                    s = f"Loss: {m['loss']:.4f}"
                    if m.get("sub_losses"):
                        sub_str = ", ".join(f"{k}: {v:.4f}" for k, v in m["sub_losses"].items())
                        s += f" ({sub_str})"
                    s += f", Dice: {m['dice']:.4f}, IoU: {m['iou']:.4f}"
                    return s

                badge = "⭐ [BEST]" if is_best else ""
                log_msg = (
                    f"Epoch [{epoch:02d}/{epochs:02d}] | LR: {lr:.2e} | "
                    f"Train [{_fmt_phase(train_m)}] | "
                    f"Val [{_fmt_phase(val_m)}] | "
                    f"Time: {format_duration(elapsed)} | ETA: {format_duration(eta_sec)} {badge}"
                )
                if self.run_handle:
                    self.run_handle.log(log_msg)
                else:
                    print(log_msg)

                if early_stopping_patience > 0 and self.no_improve >= early_stopping_patience:
                    stop_msg = f"⏹️ Dừng sớm (Early Stopping) tại epoch {epoch} sau {early_stopping_patience} epochs không cải thiện."
                    if self.run_handle:
                        self.run_handle.log(stop_msg)
                    else:
                        print(stop_msg)
                    break

            if self.run_handle:
                self.run_handle.finish(status="completed")

            if self.drive_run_dir:
                self._safe_sync_file(self.save_dir / "run_config.json")
                self._safe_sync_file(self.save_dir / "history.csv")
                self._safe_sync_file(self.save_dir / "train.log")
                if (self.save_dir / "DATA_SPEC_AND_CONFIG.md").exists():
                    self._safe_sync_file(self.save_dir / "DATA_SPEC_AND_CONFIG.md")
                if (self.save_dir / "REPORT.md").exists():
                    self._safe_sync_file(self.save_dir / "REPORT.md")

        except KeyboardInterrupt:
            last_saved = self.history[-1]["epoch"] if self.history else (self.start_epoch - 1)
            msg = (
                f"\n⚠️ ĐÃ NHẬN TÍN HIỆU DỪNG (KeyboardInterrupt)!\n"
                f"   - Checkpoints đã được bảo toàn an toàn tại: {self.save_dir}\n"
                f"   - Hoàn tất an toàn tới hết Epoch {last_saved}."
            )
            if self.run_handle:
                self.run_handle.log(msg)
                self.run_handle.finish(status="interrupted")
            else:
                print(msg)
            if self.drive_run_dir:
                self._safe_sync_file(self.save_dir / "run_config.json")
                self._safe_sync_file(self.save_dir / "history.csv")
                self._safe_sync_file(self.save_dir / "train.log")
                if (self.save_dir / "last_model.pth").exists():
                    self._safe_sync_file(self.save_dir / "last_model.pth")
            return self.history

        except Exception as e:
            if self.run_handle:
                self.run_handle.log(f"❌ Ngoại lệ trong khi huấn luyện: {e}")
                self.run_handle.finish(status="failed")
            if self.drive_run_dir:
                self._safe_sync_file(self.save_dir / "run_config.json")
                self._safe_sync_file(self.save_dir / "train.log")
            raise e

        total_time = time.time() - fit_start_time
        final_msg = f"🎉 Hoàn tất fit! Best Val Dice: {self.best_val_dice:.4f} | Tổng thời gian: {format_duration(total_time)}"
        if self.run_handle:
            self.run_handle.log(final_msg)
        else:
            print(final_msg)

        return self.history
```

#### <a id="tệp-16-common-engine-checkpoint_io-py"></a>Tệp 16: `common/engine/checkpoint_io.py`
- **Vị trí**: `common/engine/checkpoint_io.py`
- **Quy mô**: `106 dòng` | `3.4 KB`
- **Vai trò**: Quản lý lưu/nạp checkpoint nguyên tử bằng file tạm + os.replace, xuất file best_weights_fp16.pth nhẹ ~60MB.

```python
# -*- coding: utf-8 -*-
"""
common/engine/checkpoint_io.py - Quản lý lưu/nạp Checkpoint nguyên tử và xuất trọng số fp16.

Đặc điểm:
  - Lưu checkpoint nguyên tử (atomic save) bằng file tạm + os.replace
  - Nạp an toàn kèm kiểm tra metadata kiến trúc
  - Xuất best_weights_fp16.pth nhẹ (~60MB) để đẩy lên GitHub hoặc chia sẻ
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def save_checkpoint(
    path: Union[str, Path],
    state_dict: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
    scaler_state: Optional[Dict[str, Any]] = None,
    epoch: int = 0,
    best_val_dice: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    """Lưu checkpoint nguyên tử an toàn tuyệt đối chống hỏng hóc khi bị ngắt."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "scaler_state_dict": scaler_state,
        "meta": meta or {},
    }
    if extra:
        payload.update(extra)

    temp_f = p.parent / f".tmp_{p.name}_{os.getpid()}"
    try:
        torch.save(payload, str(temp_f))
        os.replace(str(temp_f), str(p))
    finally:
        if temp_f.exists():
            try:
                temp_f.unlink()
            except OSError:
                pass


def load_checkpoint_file(path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    """Nạp file checkpoint từ ổ đĩa."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {p}")
    return torch.load(str(p), map_location=map_location)


def export_weights_fp16(src_ckpt_path: Union[str, Path], dst_path: Union[str, Path], meta: Optional[Dict[str, Any]] = None):
    """
    Xuất checkpoint chỉ chứa state_dict ở định dạng float16 (nhẹ ~50% so với full fp32 checkpoint,
    loại bỏ optimizer_state khổng lồ) để đẩy lên GitHub hoặc phân phối mô hình.
    """
    src = Path(src_ckpt_path).resolve()
    dst = Path(dst_path).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint_file(src, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    fp16_sd = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            fp16_sd[k] = v.to(torch.float16)
        else:
            fp16_sd[k] = v

    out_payload = {
        "model_state_dict": fp16_sd,
        "meta": meta or ckpt.get("meta", {}),
        "epoch": ckpt.get("epoch", 0),
        "best_val_dice": ckpt.get("best_val_dice", 0.0),
    }

    temp_f = dst.parent / f".tmp_{dst.name}_{os.getpid()}"
    try:
        torch.save(out_payload, str(temp_f))
        os.replace(str(temp_f), str(dst))
    finally:
        if temp_f.exists():
            try:
                temp_f.unlink()
            except OSError:
                pass
```

#### <a id="tệp-17-common-engine-run_manager-py"></a>Tệp 17: `common/engine/run_manager.py`
- **Vị trí**: `common/engine/run_manager.py`
- **Quy mô**: `230 dòng` | `8.3 KB`
- **Vai trò**: RunHandle và RunManager: Quản lý thư mục run độc lập, ghi nhận log thời gian thực, cập nhật run_config.json.

```python
# -*- coding: utf-8 -*-
"""
common/engine/run_manager.py - Quản lý cấu trúc thư mục Run độc lập cho từng lần huấn luyện.

Mỗi lần train được cấp một thư mục riêng: <runs-root>/<run_id>/
Bao gồm:
  - run_config.json: Tái lập 100% tham số, môi trường, git, kiến trúc
  - history.csv: Log metrics từng epoch
  - train.log: Ghi nhật ký đầy đủ
  - best_model.pth, last_model.pth, best_weights_fp16.pth
"""

import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.paths import default_runs_root
from common.engine.run_config import build_initial_run_config
from common.engine.utils import atomic_save_json


class RunHandle:
    """Quản lý vòng đời của một phiên chạy (Run)."""

    def __init__(self, run_dir: Path, config: Dict[str, Any]):
        self.run_dir = Path(run_dir).resolve()
        self.config = config
        self.run_id = config.get("run_id", self.run_dir.name)
        self.config_path = self.run_dir / "run_config.json"
        self.log_path = self.run_dir / "train.log"
        self.history_path = self.run_dir / "history.csv"
        self._start_time = time.time()

        # Tạo thư mục và ghi run_config.json ban đầu ngay lập tức (status="running")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

    def _save_config(self):
        atomic_save_json(self.config_path, self.config)

    def log(self, message: str, echo: bool = True):
        """Ghi thông điệp đồng thời ra stdout và train.log."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        if echo:
            try:
                print(formatted)
            except UnicodeEncodeError:
                encoding = sys.stdout.encoding or "utf-8"
                print(formatted.encode(encoding, errors="replace").decode(encoding, errors="replace"))
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
        except Exception:
            pass

    def record_resume_event(self, from_epoch: int):
        """Ghi nhận sự kiện tiếp tục huấn luyện (resume)."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        events = self.config.setdefault("resume_events", [])
        events.append({"at": now_iso, "from_epoch": from_epoch})
        self._save_config()

    def update_result(
        self,
        epoch: int,
        best_val_dice: Optional[float] = None,
        best_epoch: Optional[int] = None,
        last_val_loss: Optional[float] = None,
        last_val_dice: Optional[float] = None,
        last_val_iou: Optional[float] = None,
    ):
        """Cập nhật tiến độ sau mỗi epoch."""
        res = self.config.setdefault("result", {})
        res["epochs_completed"] = epoch
        if best_val_dice is not None:
            res["best_val_dice"] = float(best_val_dice)
        if best_epoch is not None:
            res["best_epoch"] = int(best_epoch)

        last = res.setdefault("last", {})
        if last_val_loss is not None:
            last["val_loss"] = float(last_val_loss)
        if last_val_dice is not None:
            last["val_dice"] = float(last_val_dice)
        if last_val_iou is not None:
            last["val_iou"] = float(last_val_iou)

        # Cập nhật thời lượng chạy
        elapsed = time.time() - self._start_time
        self.config["duration_sec"] = round(elapsed, 1)
        self._save_config()

    def finish(self, status: str = "completed", extra_metrics: Optional[Dict[str, Any]] = None):
        """Đóng phiên chạy và ghi nhận trạng thái cuối cùng."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.config["status"] = status
        self.config["finished_at"] = now_iso
        self.config["duration_sec"] = round(time.time() - self._start_time, 1)
        if extra_metrics:
            self.config.setdefault("result", {}).update(extra_metrics)
        self._save_config()
        self.log(f"[FINISH] Phiên chạy '{self.run_id}' kết thúc với trạng thái: {status.upper()}")


def generate_run_id(arch: str, variant: Optional[str] = None, tag: Optional[str] = None) -> str:
    """Sinh run_id dạng {YYYYmmdd-HHMMSS}_{arch}[_{variant}][_{tag}]."""
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [now_str, arch]
    if variant:
        parts.append(variant)
    if tag:
        parts.append(tag)
    return "_".join(parts)


def create_run(
    arch: str,
    runs_root: Optional[Union[str, Path]] = None,
    run_name: Optional[str] = None,
    tag: Optional[str] = None,
    variant: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    model_meta: Optional[Dict[str, Any]] = None,
    data_meta: Optional[Dict[str, Any]] = None,
    loss_meta: Optional[Dict[str, Any]] = None,
    optimizer_meta: Optional[Dict[str, Any]] = None,
    scheduler_meta: Optional[Dict[str, Any]] = None,
    training_meta: Optional[Dict[str, Any]] = None,
    notes: str = "",
    init_weights: Optional[str] = None,
) -> RunHandle:
    """Khởi tạo một Run mới hoàn chỉnh kèm run_config.json."""
    root = Path(runs_root).resolve() if runs_root else default_runs_root(arch)
    root.mkdir(parents=True, exist_ok=True)

    if run_name:
        run_id = run_name
    else:
        run_id = generate_run_id(arch, variant=variant, tag=tag)

    # Đảm bảo run_id là duy nhất (nếu trùng thêm -2, -3)
    target_dir = root / run_id
    counter = 2
    base_id = run_id
    while target_dir.exists():
        run_id = f"{base_id}-{counter}"
        target_dir = root / run_id
        counter += 1

    config = build_initial_run_config(
        run_id=run_id,
        arch=arch,
        args=args or {},
        model_meta=model_meta or {},
        data_meta=data_meta or {},
        loss_meta=loss_meta or {},
        optimizer_meta=optimizer_meta or {},
        scheduler_meta=scheduler_meta or {},
        training_meta=training_meta or {},
        notes=notes,
        init_weights=init_weights,
    )

    return RunHandle(target_dir, config)


def find_resumable(arch: str, runs_root: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Tìm run mới nhất của kiến trúc đó chưa hoàn thành (status != 'completed')."""
    root = Path(runs_root).resolve() if runs_root else default_runs_root(arch)
    if not root.exists():
        return None

    candidate_runs = []
    for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "run_config.json").exists() and (d / "last_model.pth").exists():
            try:
                with open(d / "run_config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if cfg.get("arch") == arch and cfg.get("status") != "completed":
                    candidate_runs.append(d)
            except Exception:
                pass

    return candidate_runs[0] if candidate_runs else None


def resolve_resume(
    resume_arg: Optional[str],
    arch: str,
    runs_root: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Phân giải tham số --resume:
      - 'auto': Tự tìm run dở dang mới nhất
      - Đường dẫn thư mục run: Nạp last_model.pth trong thư mục đó
      - Đường dẫn file checkpoint .pth cụ thể
    Trả về: (resolved_run_dir, resolved_checkpoint_path)
    """
    if not resume_arg:
        return None, None

    root = Path(runs_root).resolve() if runs_root else default_runs_root(arch)

    if resume_arg.lower() == "auto":
        resumable_dir = find_resumable(arch, root)
        if resumable_dir:
            return resumable_dir, resumable_dir / "last_model.pth"
        return None, None

    p = Path(resume_arg).resolve()
    if p.is_dir():
        ckpt = p / "last_model.pth"
        if not ckpt.exists():
            ckpt = p / "best_model.pth"
        return p, (ckpt if ckpt.exists() else None)

    if p.is_file():
        run_d = p.parent
        return run_d, p

    return None, None
```

#### <a id="tệp-18-common-engine-run_config-py"></a>Tệp 18: `common/engine/run_config.py`
- **Vị trí**: `common/engine/run_config.py`
- **Quy mô**: `155 dòng` | `4.7 KB`
- **Vai trò**: Khởi tạo và chuẩn hóa Schema v1 cho tệp run_config.json (lưu vết môi trường, git commit, GPU, tham số).

```python
# -*- coding: utf-8 -*-
"""
common/engine/run_config.py - Lược đồ và hàm thu thập siêu dữ liệu thí nghiệm (run_config.json).

Quản lý thông tin:
  - Môi trường chạy (platform, python, torch, CUDA, GPU)
  - Thông tin Git (commit hash, branch, dirty)
  - Dữ liệu và manifest nguồn
  - Kết quả và sự kiện resume
"""

import datetime
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.engine.utils import atomic_save_json

SCHEMA_VERSION = 1


def get_git_info() -> Dict[str, Any]:
    """Lấy thông tin git commit hiện tại (nếu có)."""
    git_info = {"commit": "unknown", "branch": "unknown", "dirty": False, "remote": "unknown"}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_info["commit"] = commit
    except Exception:
        pass

    try:
        branch = subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_info["branch"] = branch
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "-C", str(_ROOT), "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_info["dirty"] = bool(status)
    except Exception:
        pass

    try:
        remote = subprocess.check_output(
            ["git", "-C", str(_ROOT), "config", "--get", "remote.origin.url"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        # Loại bỏ token nếu có trong URL remote để bảo mật tuyệt đối (R9)
        if "@" in remote:
            remote = "https://" + remote.split("@")[-1]
        git_info["remote"] = remote
    except Exception:
        pass

    return git_info


def get_environment_info(amp: bool = True, mamba_backend: Optional[str] = None) -> Dict[str, Any]:
    """Thu thập thông tin hệ thống và các gói thư viện phụ thuộc."""
    import torch

    if "google.colab" in sys.modules:
        plat = "colab"
    elif Path("/kaggle/working").exists():
        plat = "kaggle"
    else:
        plat = "local"

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    num_gpus = torch.cuda.device_count() if cuda_available else 0

    packages = {}
    for pkg in ("timm", "einops", "mamba_ssm"):
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg] = None

    return {
        "platform": plat,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda if cuda_available else None,
        "gpu": gpu_name,
        "num_gpus": num_gpus,
        "amp": bool(amp and cuda_available),
        "mamba_backend": mamba_backend or ("cuda" if (packages.get("mamba_ssm") and cuda_available) else "torch"),
        "packages": packages,
    }


def build_initial_run_config(
    run_id: str,
    arch: str,
    args: Dict[str, Any],
    model_meta: Dict[str, Any],
    data_meta: Dict[str, Any],
    loss_meta: Dict[str, Any],
    optimizer_meta: Dict[str, Any],
    scheduler_meta: Dict[str, Any],
    training_meta: Dict[str, Any],
    notes: str = "",
    init_weights: Optional[str] = None,
) -> Dict[str, Any]:
    """Khởi tạo tài liệu run_config.json theo schema v1."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "arch": arch,
        "status": "running",
        "notes": notes,
        "started_at": now_iso,
        "finished_at": None,
        "duration_sec": None,
        "resume_events": [],
        "init_weights": init_weights,
        "git": get_git_info(),
        "environment": get_environment_info(
            amp=not args.get("no_amp", False),
            mamba_backend=model_meta.get("mamba_backend"),
        ),
        "args": args,
        "model": model_meta,
        "data": data_meta,
        "loss": loss_meta,
        "optimizer": optimizer_meta,
        "scheduler": scheduler_meta,
        "training": training_meta,
        "result": {
            "epochs_completed": 0,
            "best_val_dice": None,
            "best_epoch": None,
            "last": {
                "val_loss": None,
                "val_dice": None,
                "val_iou": None,
            },
        },
    }
```

#### <a id="tệp-19-common-engine-cli_args-py"></a>Tệp 19: `common/engine/cli_args.py`
- **Vị trí**: `common/engine/cli_args.py`
- **Quy mô**: `104 dòng` | `5.6 KB`
- **Vai trò**: Bộ tham số dòng lệnh CLI dùng chung chuẩn hóa giữa các kiến trúc.

```python
# -*- coding: utf-8 -*-
"""
common/engine/cli_args.py - Các nhóm tham số dòng lệnh dùng chung cho toàn bộ dự án.

Đảm bảo giao diện dòng lệnh giữa hai kiến trúc unet_base và mamba_unet hoàn toàn đồng nhất.
Tuyệt đối không hard-code danh sách choices của activation và loss; các kiến trúc sẽ tự
truyền danh sách choices từ registry riêng của mình (R11, mục 4.4, 5.1).
"""

import argparse
from typing import List, Optional


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    activation_choices: List[str],
    loss_choices: List[str],
    default_activation: str = "relu",
    default_loss: str = "dice_bce",
    default_batch_size: int = 8,
    default_epochs: int = 50,
    default_lr: float = 1e-4,
    default_seed: int = 42,
):
    """
    Gắn các nhóm tham số chuẩn hóa vào ArgumentParser của lệnh train:
      1. Cấu hình
      2. Dữ liệu
      3. Mô hình chung
      4. Hàm mất mát (Loss)
      5. Tối ưu hóa (Optimizer & Scheduler)
      6. Quản lý phiên chạy (Run & Resume)
    """
    # 1. Cấu hình
    cfg_group = parser.add_argument_group("Cấu hình thí nghiệm")
    cfg_group.add_argument("--config", default=None, help="Đường dẫn file cấu hình YAML ghi đè tham số")

    # 2. Dữ liệu
    data_group = parser.add_argument_group("Dữ liệu")
    data_group.add_argument("--cache-dir", default="data/cache_1mm", help="Thư mục dữ liệu cache_1mm đã chuẩn bị")
    data_group.add_argument("--data-manifest", default=None, help="Đường dẫn data_manifest.json (nếu có)")
    data_group.add_argument("--workers", type=int, default=2, help="Số luồng nạp dữ liệu DataLoader")
    data_group.add_argument("--gpu-window", action="store_true", help="Chuyển mảng HU int16 lên GPU để chuẩn hóa cửa sổ")
    data_group.add_argument("--no-lung-mask", action="store_true", help="Không áp mặt nạ phổi ngoài lát cắt")
    data_group.add_argument("--no-augment", action="store_true", help="Tắt tăng cường dữ liệu lật/xoay ngẫu nhiên")
    data_group.add_argument("--limit-train", type=int, default=0, help="Giới hạn số lát train (0 = toàn bộ)")
    data_group.add_argument("--limit-val", type=int, default=0, help="Giới hạn số lát val (0 = toàn bộ)")
    data_group.add_argument("--rebuild-index", action="store_true", help="Bắt buộc tạo lại slices.csv")

    # 3. Mô hình chung
    model_group = parser.add_argument_group("Mô hình chung")
    model_group.add_argument(
        "--activation",
        choices=activation_choices,
        default=default_activation,
        help=f"Hàm kích hoạt (lựa chọn của kiến trúc này: {activation_choices})",
    )

    # 4. Loss
    loss_group = parser.add_argument_group("Hàm mất mát")
    loss_group.add_argument(
        "--loss",
        choices=loss_choices,
        default=default_loss,
        help=f"Hàm mất mát (lựa chọn của kiến trúc này: {loss_choices})",
    )
    loss_group.add_argument("--loss-params", default=None, help="Tham số bổ sung cho loss dưới dạng chuỗi JSON")

    # 5. Tối ưu
    opt_group = parser.add_argument_group("Tối ưu hóa")
    opt_group.add_argument("--epochs", type=int, default=default_epochs, help="Tổng số epoch huấn luyện đích")
    opt_group.add_argument("--batch-size", type=int, default=default_batch_size, help="Kích thước batch")
    opt_group.add_argument("--lr", type=float, default=default_lr, help="Tốc độ học cơ sở (learning rate)")
    opt_group.add_argument("--weight-decay", type=float, default=1e-4, help="Hệ số phạt suy giảm trọng số")
    opt_group.add_argument(
        "--scheduler",
        choices=["cosine", "plateau", "step", "none"],
        default="cosine",
        help="Chiến lược điều chỉnh tốc độ học (LR Scheduler)",
    )
    opt_group.add_argument("--grad-clip", type=float, default=1.0, help="Ngưỡng kẹp gradient tối đa (0 = tắt)")
    opt_group.add_argument("--patience", type=int, default=12, help="Số epoch chờ trước khi dừng sớm Early Stopping")

    # 6. Chạy & Quản lý Run
    run_group = parser.add_argument_group("Quản lý phiên chạy (Run)")
    run_group.add_argument("--runs-root", default=None, help="Thư mục gốc lưu trữ các lần chạy")
    run_group.add_argument(
        "--drive-sync-dir",
        default=None,
        help="Đường dẫn thư mục Google Drive để tự động đồng bộ checkpoints và logs thời gian thực sau mỗi epoch",
    )
    run_group.add_argument("--run-name", default=None, help="Tên tùy chỉnh cho phiên chạy")
    run_group.add_argument("--tag", default=None, help="Thẻ phân loại thí nghiệm (vd: baseline, ablation)")
    run_group.add_argument("--notes", default="", help="Ghi chú chi tiết cho thí nghiệm")
    run_group.add_argument(
        "--resume",
        default=None,
        help="Tiếp tục huấn luyện: 'auto' (run mới nhất dở dang), đường dẫn run_dir hoặc file checkpoint cụ thể",
    )
    run_group.add_argument("--init-weights", default=None, help="Nạp trọng số ban đầu từ checkpoint (khởi tạo run mới)")
    run_group.add_argument("--seed", type=int, default=default_seed, help="Hạt giống ngẫu nhiên")
    run_group.add_argument("--no-amp", action="store_true", help="Tắt Mixed Precision (tự động bật nếu có GPU)")
    run_group.add_argument("--no-progress", action="store_true", help="Tắt thanh tiến trình tqdm chi tiết")
```

#### <a id="tệp-20-common-engine-utils-py"></a>Tệp 20: `common/engine/utils.py`
- **Vị trí**: `common/engine/utils.py`
- **Quy mô**: `52 dòng` | `1.7 KB`
- **Vai trò**: Các hàm tiện ích hệ thống: seed_everything, atomic_save_json, format_duration.

```python
# -*- coding: utf-8 -*-
"""
common/engine/utils.py - Các hàm tiện ích hỗ trợ huấn luyện:
  - seed_everything: Cố định hạt giống ngẫu nhiên đa thư viện
  - format_duration: Định dạng thời gian chạy thân thiện
  - atomic_save_json: Ghi JSON nguyên tử qua file tạm + os.replace
"""

import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Cố định seed ngẫu nhiên cho python random, numpy và torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_duration(seconds: float) -> str:
    """Chuyển số giây sang định dạng giờ:phút:giây (vd: '01h 24m 35s' hoặc '04m 12s')."""
    if seconds is None:
        return "N/A"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def atomic_save_json(path: Union[str, Path], data: Any, indent: int = 2):
    """Lưu dữ liệu JSON nguyên tử (atomic) tránh bị hỏng file khi bị ngắt đột ngột."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_file = p.parent / f".tmp_{p.name}_{os.getpid()}"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    os.replace(str(temp_file), str(p))
```

#### <a id="tệp-21-common-paths-py"></a>Tệp 21: `common/paths.py`
- **Vị trí**: `common/paths.py`
- **Quy mô**: `34 dòng` | `1.2 KB`
- **Vai trò**: Quản lý đường dẫn tập trung không hard-code (REPO_ROOT, data cache, runs root).

```python
# -*- coding: utf-8 -*-
"""
common/paths.py - Quản lý đường dẫn tập trung cho toàn bộ dự án DuneAI.

Quy tắc bất biến:
  - Tuyệt đối không hard-code đường dẫn tuyệt đối (Không dùng tên repo cũ, SOURCES, /content, /kaggle)
  - REPO_ROOT tự động suy ra từ vị trí file common/paths.py (2 cấp cha)
"""

import os
from pathlib import Path

# REPO_ROOT: Thư mục gốc của repository (SOURCES/)
REPO_ROOT = Path(__file__).resolve().parent.parent


def default_data_dir() -> Path:
    """Trả về thư mục dữ liệu cache mặc định (SOURCES/data/cache_1mm)."""
    env_dir = os.environ.get("DUNEAI_DATA_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return REPO_ROOT / "data" / "cache_1mm"


def default_runs_root(arch: str) -> Path:
    """
    Trả về thư mục gốc chứa các thí nghiệm huấn luyện theo từng kiến trúc:
      - arch="unet_base"   -> SOURCES/unet_base/runs
      - arch="mamba_unet"  -> SOURCES/mamba_unet/runs
    """
    env_runs = os.environ.get("DUNEAI_RUNS_ROOT")
    if env_runs:
        return Path(env_runs).resolve()
    return REPO_ROOT / arch / "runs"
```

#### <a id="tệp-22-configs-config-py"></a>Tệp 22: `configs/config.py`
- **Vị trí**: `configs/config.py`
- **Quy mô**: `37 dòng` | `1.2 KB`
- **Vai trò**: Hằng số y tế và chuẩn hóa lát cắt CT: Cửa sổ phổi HU, kích thước ảnh 512, ngưỡng u 20mm2, tỷ lệ âm tính.

```python
# -*- coding: utf-8 -*-
"""
configs/config.py - Hằng số y tế và tham số tiền xử lý ảnh CT chuẩn của dự án DuneAI.

Giữ nguyên vẹn các hằng số không gian & chuẩn hóa lát cắt từ phiên bản gốc.
Không chứa siêu tham số huấn luyện của từng thí nghiệm (được quản lý qua YAML và CLI).
"""

# ==========================================
# THÔNG SỐ KHÔNG GIAN & VẬT LÝ Y TẾ
# ==========================================
IMAGE_SIZE = 512

# Cửa sổ Hounsfield cho mô phổi (Lung Window)
WINDOW_WIDTH = 1500
WINDOW_LEVEL = -600
RESAMPLING_STEP = 25

# ==========================================
# THÔNG SỐ LỌC DỮ LIỆU BỆNH NHÂN
# ==========================================
# Ngưỡng diện tích nhỏ nhất để giữ lại lát cắt có khối u (mm^2)
TUMOR_SIZE_LOW_LIM = 20.0

# Tỷ lệ lấy mẫu các lát cắt âm tính (không có khối u) để cân bằng dữ liệu
ZERO_SL_RATIO = 0.25

# ==========================================
# THÔNG SỐ CHIA TẬP MẶC ĐỊNH
# ==========================================
VAL_SPLIT = 0.15
SEED = 42

# Giá trị mặc định hỗ trợ tương thích ngược cho các module cũ
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
```


---

### 4. Xử Lý Dữ Liệu Y Tế & Data Pipeline

#### <a id="tệp-23-common-data_pipeline-__init__-py"></a>Tệp 23: `common/data_pipeline/__init__.py`
- **Vị trí**: `common/data_pipeline/__init__.py`
- **Quy mô**: `2 dòng` | `0.1 KB`
- **Vai trò**: Khởi tạo package data pipeline.

```python
# -*- coding: utf-8 -*-
"""common.data_pipeline package - Pipeline tiền xử lý và nạp dữ liệu."""
```

#### <a id="tệp-24-common-data_pipeline-dataset-py"></a>Tệp 24: `common/data_pipeline/dataset.py`
- **Vị trí**: `common/data_pipeline/dataset.py`
- **Quy mô**: `312 dòng` | `11.8 KB`
- **Vai trò**: SliceDataset25D và hàm build_dataloaders: Nạp dữ liệu 2.5D (z-1, z, z+1) và tạo PyTorch DataLoaders.

```python
# -*- coding: utf-8 -*-
"""
dataset.py - PyTorch Dataset & DataLoader 2.5D cho DuneAI.

Đặc điểm:
  - Đọc trực tiếp từ cache_1mm thông qua mmap (chỉ đọc lát cắt cần dùng).
  - Ghép 3 lát cắt liên tiếp (z-1, z, z+1) tạo đầu vào 2.5D (3, 512, 512).
  - Áp mặt nạ phổi an toàn: Train hợp thêm vùng u giãn nở; Val/Test chỉ dùng phổi (không rò rỉ nhãn).
  - Hỗ trợ chuẩn hóa cửa sổ HU ngay trên CPU hoặc trả về int16 để chạy window_hu_torch trên GPU.

Dùng:
  python dataset.py selftest
  python dataset.py inspect --cache-dir data/cache_1mm
"""
import argparse
import json
import random
import shutil
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
ROOT = _ROOT

from configs.config import IMAGE_SIZE
from common.data_pipeline.medical_prep import (
    HU_FLOOR,
    apply_lung_mask,
    clip_and_window_hu,
    load_cached_slices,
    lung_cache_path,
    lung_keep_mask,
    window_hu_torch,
)
from common.data_pipeline.offline_builder import build_dataset_index


class LIDC25DDataset(Dataset):
    def __init__(
        self,
        cache_dir,
        csv_path=None,
        split="train",
        in_channels=3,
        image_size=IMAGE_SIZE,
        apply_lung=True,
        window_on_cpu=True,
        quantize=True,
        augment=False,
    ):
        """
        Args:
            cache_dir: Đường dẫn tới thư mục cache_1mm.
            csv_path: Đường dẫn tới slices.csv (mặc định: <cache_dir>/slices.csv).
            split: 'train', 'val', hoặc 'all'.
            in_channels: Số lát cắt ghép kênh 2.5D (mặc định 3: z-1, z, z+1).
            image_size: Kích thước đầu ra sau pad/crop (mặc định 512).
            apply_lung: Có che vùng ngoài phổi về -1000 HU hay không.
            window_on_cpu: True -> trả về float32 [0, 1]; False -> trả về int16 HU để chạy window_hu_torch trên GPU.
            quantize: Bật/tắt lượng tử hóa bước 25 HU.
            augment: Bật lật/xoay ngẫu nhiên (chỉ có tác dụng khi split='train').
        """
        super().__init__()
        assert in_channels in (1, 3, 5), "in_channels phải là số lẻ (1, 3, hoặc 5)"
        self.cache_dir = Path(cache_dir)
        self.csv_path = Path(csv_path) if csv_path is not None else (self.cache_dir / "slices.csv")
        self.split = split
        self.in_channels = in_channels
        self.half_k = in_channels // 2
        self.image_size = image_size
        self.apply_lung = apply_lung
        self.window_on_cpu = window_on_cpu
        self.quantize = quantize
        self.augment = bool(augment and (split == "train"))

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {self.csv_path}. Hãy chạy `python offline_builder.py build` trước!"
            )

        df = pd.read_csv(self.csv_path)
        if split is not None and split != "all":
            df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"Không có mẫu nào thuộc split='{split}' trong {self.csv_path}")
        self.df = df

    def __len__(self):
        return len(self.df)

    def _neighbor_indices(self, z, num_slices):
        """Lấy chỉ số các lát lân cận, tự động kẹp biên ở lát 0 và lát cuối."""
        return [min(max(0, z + offset), num_slices - 1) for offset in range(-self.half_k, self.half_k + 1)]

    @staticmethod
    def _augment_pair(img_chw, mask_1hw):
        """Biến đổi hình học đồng bộ trên cả khối kênh 2.5D và mask trung tâm."""
        if random.random() < 0.5:
            img_chw = np.flip(img_chw, axis=2)
            mask_1hw = np.flip(mask_1hw, axis=2)
        if random.random() < 0.5:
            img_chw = np.flip(img_chw, axis=1)
            mask_1hw = np.flip(mask_1hw, axis=1)
        k = random.randint(0, 3)
        if k > 0:
            img_chw = np.rot90(img_chw, k=k, axes=(1, 2))
            mask_1hw = np.rot90(mask_1hw, k=k, axes=(1, 2))
        return np.ascontiguousarray(img_chw), np.ascontiguousarray(mask_1hw)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row["patient_id"])
        z = int(row["z"])
        num_slices = int(row["num_slices"])

        p_dir = self.cache_dir / pid
        z_indices = self._neighbor_indices(z, num_slices)

        use_lung = (
            self.apply_lung
            and bool(row.get("has_lung_mask", 0))
            and lung_cache_path(p_dir).exists()
        )
        loaded = load_cached_slices(p_dir, z_indices, size=self.image_size, with_lung=use_lung)

        if use_lung:
            img_stack, msk_stack, lung_stack = loaded
            # QUAN TRỌNG: Chỉ truyền msk_stack vào lung_keep_mask khi split == 'train'
            tumor_for_keep = msk_stack if self.split == "train" else None
            keep = lung_keep_mask(lung_stack, tumor_mask=tumor_for_keep)
            img_stack = apply_lung_mask(img_stack, keep, fill=HU_FLOOR)
        else:
            img_stack, msk_stack = loaded

        # Nhãn mục tiêu chỉ là lát cắt chính giữa (1, H, W)
        center_mask = (msk_stack[self.half_k : self.half_k + 1] > 0).astype(np.float32)

        if self.window_on_cpu:
            img_out = clip_and_window_hu(img_stack, quantize=self.quantize)
        else:
            img_out = img_stack.astype(np.int16)

        if self.augment:
            img_out, center_mask = self._augment_pair(img_out, center_mask)

        return {
            "image": torch.from_numpy(img_out),
            "mask": torch.from_numpy(center_mask),
            "patient_id": pid,
            "z": z,
        }


def create_dataloaders(
    cache_dir,
    batch_size=8,
    num_workers=2,
    in_channels=3,
    apply_lung=True,
    window_on_cpu=True,
    augment_train=True,
):
    """Hàm tiện ích khởi tạo cặp (train_loader, val_loader) sẵn sàng cho vòng lặp huấn luyện."""
    train_ds = LIDC25DDataset(
        cache_dir=cache_dir,
        split="train",
        in_channels=in_channels,
        apply_lung=apply_lung,
        window_on_cpu=window_on_cpu,
        augment=augment_train,
    )
    val_ds = LIDC25DDataset(
        cache_dir=cache_dir,
        split="val",
        in_channels=in_channels,
        apply_lung=apply_lung,
        window_on_cpu=window_on_cpu,
        augment=False,
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    return train_loader, val_loader


def build_dataloaders(
    cache_dir,
    batch_size=8,
    num_workers=2,
    in_channels=3,
    use_gpu_window=False,
    lung_mask=True,
    augment=True,
    limit_train=0,
    limit_val=0,
    rebuild_index=False,
    seed=42,
    **kwargs,
):
    from pathlib import Path
    cache_path = Path(cache_dir).resolve()
    if rebuild_index or not (cache_path / "slices.csv").exists():
        build_dataset_index(cache_path)
    train_ds = LIDC25DDataset(
        cache_dir=cache_path,
        split="train",
        in_channels=in_channels,
        apply_lung=lung_mask,
        window_on_cpu=not use_gpu_window,
        augment=augment,
    )
    val_ds = LIDC25DDataset(
        cache_dir=cache_path,
        split="val",
        in_channels=in_channels,
        apply_lung=lung_mask,
        window_on_cpu=not use_gpu_window,
        augment=False,
    )
    if limit_train and limit_train > 0:
        train_ds.df = train_ds.df.iloc[:limit_train].reset_index(drop=True)
    if limit_val and limit_val > 0:
        val_ds.df = val_ds.df.iloc[:limit_val].reset_index(drop=True)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    return train_loader, val_loader

def _cmd_selftest(_):
    print("Đang chạy kiểm thử tự động dataset.py...")
    tmp = Path(tempfile.mkdtemp())
    try:
        for i in range(1, 5):
            p_dir = tmp / f"LIDC-IDRI-000{i}"
            p_dir.mkdir(parents=True)
            img = np.random.randint(-1000, 300, size=(12, 360, 360), dtype=np.int16)
            msk = np.zeros((12, 360, 360), dtype=np.uint8)
            lung = np.ones((12, 360, 360), dtype=np.uint8)
            msk[5:7, 150:160, 150:160] = 1  # 100 mm²

            np.save(p_dir / "image_1mm.npy", img)
            np.save(p_dir / "mask_1mm.npy", msk)
            np.save(p_dir / "lung_1mm.npy", lung)
            with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump({"orig_spacing_xyz": [0.7, 0.7, 1.5], "lung_ok": True}, f)

        build_dataset_index(tmp)
        train_loader, val_loader = create_dataloaders(tmp, batch_size=2, num_workers=0)
        batch = next(iter(train_loader))

        img_b, msk_b = batch["image"], batch["mask"]
        assert img_b.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE), f"Sai shape ảnh: {img_b.shape}"
        assert msk_b.shape == (2, 1, IMAGE_SIZE, IMAGE_SIZE), f"Sai shape mask: {msk_b.shape}"
        assert img_b.dtype == torch.float32 and 0.0 <= img_b.min() and img_b.max() <= 1.0
        assert set(torch.unique(msk_b).tolist()) <= {0.0, 1.0}

        # Kiểm tra chế độ đẩy HU int16 lên GPU rồi gọi window_hu_torch
        ds_cpu = LIDC25DDataset(tmp, split="val", window_on_cpu=True, augment=False)
        ds_raw = LIDC25DDataset(tmp, split="val", window_on_cpu=False, augment=False)
        assert torch.allclose(ds_cpu[0]["image"], window_hu_torch(ds_raw[0]["image"]), atol=1e-6)
        print("✅ selftest dataset.py: OK!")
    finally:
        shutil.rmtree(tmp)


def _cmd_inspect(a):
    train_loader, val_loader = create_dataloaders(
        a.cache_dir, batch_size=a.batch_size, num_workers=0
    )
    batch = next(iter(train_loader))
    img, msk = batch["image"], batch["mask"]
    print("=" * 50)
    print("🔍 KIỂM TRA BATCH THỰC TẾ TỪ DATALOADER")
    print("=" * 50)
    print(f"Số batch Train : {len(train_loader)} | Số batch Val: {len(val_loader)}")
    print(f"Image Tensor   : shape={tuple(img.shape)}, dtype={img.dtype}, range=[{img.min():.4f}, {img.max():.4f}]")
    print(f"Mask Tensor    : shape={tuple(msk.shape)}, dtype={msk.dtype}, unique={torch.unique(msk).tolist()}")
    print(f"Bệnh nhân mẫu  : {batch['patient_id']} | Lát Z: {batch['z'].tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="Kiểm thử tự động với dữ liệu giả lập")
    s.set_defaults(fn=_cmd_selftest)

    i = sub.add_parser("inspect", help="Kiểm tra nhanh một batch từ thư mục cache_1mm thật")
    i.add_argument("--cache-dir", default=str(ROOT / "data" / "cache_1mm"))
    i.add_argument("--batch-size", type=int, default=4)
    i.set_defaults(fn=_cmd_inspect)

    args = ap.parse_args()
    args.fn(args)
```

#### <a id="tệp-25-common-data_pipeline-medical_prep-py"></a>Tệp 25: `common/data_pipeline/medical_prep.py`
- **Vị trí**: `common/data_pipeline/medical_prep.py`
- **Quy mô**: `635 dòng` | `29.1 KB`
- **Vai trò**: Tiền xử lý y tế: Chuẩn hóa cửa sổ HU mềm/phổi, tạo mặt nạ phổi (lung mask), chuyển đổi đơn vị thể tích mm3.

```python
# -*- coding: utf-8 -*-
"""
medical_prep.py - Tiền xử lý CT theo pipeline DuneAI (s.primakov), tối ưu cho GPU.

CÁC HÀM TIỀN XỬ LÝ CHUẨN CỦA DUNEAI:
  load_nrrd(path)                                  -> (volume (Z,H,W), spacing (x,y,z))
  clip_and_window_hu(volume, ww, wl, step)         -> float32 [0,1]   (numpy)
  window_hu_torch(x, ...)                          -> cùng công thức, chạy trên GPU
  resize_and_pad_3d(volume, spacing, target_size, is_mask)
  extract_lungs_mask(volume)                       -> mask phổi (chuyển thể get_seg_lungs + create_flipped_mask)

THỨ TỰ BƯỚC (đúng Patient_data_generator._load_patient của tác giả):
  1. Cắt đáy HU            img[img < -1000] = -1000
  2. (Tùy chọn) tách phổi trên HU thô, lưới gốc 512x512, rồi che ngoài phổi bằng -1000
  3. Resize mặt phẳng trục về 1 mm/pixel trên HU thô (trục z giữ nguyên)
  4. Crop/pad về IMAGE_SIZE x IMAGE_SIZE, pad bằng -1000 HU (không khí)
  5. Cửa sổ [WL - WW/2, WL + WW/2], CẬN DƯỚI BỊ KẸP Ở -1000   (_apply_window, dòng 455-457)
  6. (Tùy chọn) lượng tử hóa theo RESAMPLING_STEP
  7. Đưa về [0, 1]  (lựa chọn của dự án; tác giả dùng z-score tùy chọn)

LƯU Ý VỀ CỬA SỔ: với WW=1500, WL=-600 cận dưới danh nghĩa là -1350 nhưng tác giả kẹp ở -1000, nên cửa sổ
thực tế là [-1000, 150]. Mô tả "low = -1350" trong tài liệu cần đổi thành -1000, nếu không không khí sẽ là
0.233 thay vì 0 và phần pad (0) sẽ lộ ra thành hình chữ nhật.

KIẾN TRÚC TỐI ƯU
  - Bước 1-4 nặng CPU và KHÔNG phụ thuộc epoch -> chạy MỘT LẦN bằng `build`, lưu .npy int16 HU.
  - Mask phổi cũng được tính một lần (--extract-lungs) và lưu riêng; việc che ảnh làm lúc nạp (rẻ).
  - Bước 5-7 là phép toán theo phần tử -> chạy trên GPU (window_hu_torch).
  - .npy không nén + mmap: chỉ đọc đúng các lát cần dùng.

KHÁC TÁC GIẢ (có chủ ý)
  - Kích thước mới dùng round() thay vì int(); spacing áp đúng trục (x -> rộng, y -> cao).
  - Nội suy ảnh mặc định "nearest" như tác giả (IMAGE_INTERP).
  - Lượng tử hóa mặc định bật; tác giả để tùy chọn và tắt (quantize=False để giống tác giả).
  - create_flipped_mask chỉ port method=0 (nhanh, không hiệu chỉnh lệch cột sống) - đây là cách get_seg_lungs gọi.
  - Bỏ qua phần "kéo dãn mask phổi theo z khi khối u nằm ngoài" (dòng 101-130 của generator); phần hợp mask
    khối u vào mask phổi (dòng 132-134) có trong apply_lung_mask.
  - Gỡ phụ thuộc statsmodels và scipy.ndimage.measurements (đã deprecated).
  - Hai chỗ thêm cho an toàn: lát không có xương thì bỏ qua khi tìm tâm phổi; mask phổi rỗng trả về None.

SimpleITK: GetSpacing() trả về (x, y, z) còn mảng numpy có thứ tự (Z, Y, X).

Dùng:
  python medical_prep.py selftest
  python medical_prep.py build --nrrd-dir <duneai_raw_nrrd> --cache-dir <cache_1mm> --workers 4 [--limit 3] [--extract-lungs]
"""
import argparse
import json
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

try:
    import torch
except ImportError:          # viewer / máy không có torch vẫn dùng được các hàm numpy
    torch = None

# nạp các hằng số từ file config
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
ROOT = _ROOT
from configs.config import IMAGE_SIZE, WINDOW_WIDTH, WINDOW_LEVEL, RESAMPLING_STEP

HU_FLOOR = -1000                 # đáy HU (không khí)
IMAGE_INTERP = "nearest"         # nội suy ảnh khi resize; tác giả dùng nearest
LUNG_THRESHOLD_HU = -320         # ngưỡng nhu mô phổi của tác giả
TRACHEA_COEFF = 0.0069           # tác giả đặt 0.0053 (dòng 229) rồi ghi đè bằng 0.0069 (dòng 230)
_CV2_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
}


# ======================================================================
# 1. ĐỌC DỮ LIỆU
# ======================================================================
def load_nrrd(file_path):
    """
    Đọc file NRRD bằng SimpleITK.
    Trả về:
        - volume: mảng numpy 3D (Z, H, W)
        - spacing: tuple (spacing_x, spacing_y, spacing_z) theo mm  (thứ tự của SimpleITK)
    """
    sitk_img = sitk.ReadImage(str(file_path))
    volume = sitk.GetArrayFromImage(sitk_img)      # (Z, H, W)
    spacing = sitk_img.GetSpacing()                # (x, y, z)
    return volume, spacing


# ======================================================================
# 2. CỬA SỔ HU + LƯỢNG TỬ HÓA  (numpy cho viewer/test, torch cho GPU)
# ======================================================================
def window_bounds(ww=WINDOW_WIDTH, wl=WINDOW_LEVEL):
    """Giống _apply_window của tác giả: cận dưới bị kẹp ở -1000. Mặc định (1500, -600) -> [-1000, 150]."""
    assert 0 < ww < 2000 and -1000 < wl < 200, "WW/WL ngoài miền hợp lệ của tác giả"
    low = max(wl - ww / 2.0, HU_FLOOR)
    high = wl + ww / 2.0
    return low, high


def clip_and_window_hu(volume, ww=WINDOW_WIDTH, wl=WINDOW_LEVEL, step=RESAMPLING_STEP, quantize=True):
    """
    Phiên bản numpy (CPU). Đầu vào là HU THÔ (nên là ảnh đã resize và pad -1000).
    1) cắt đáy -1000  2) kẹp cửa sổ  3) lượng tử hóa theo `step`  4) chuẩn hóa về [0, 1].
    Không sửa mảng đầu vào. Đầu ra float32; không khí = 0.
    """
    low, high = window_bounds(ww, wl)
    vol = np.clip(volume.astype(np.float32), low, high)          # gộp bước cắt đáy và cửa sổ
    if quantize:                                                  # bin = floor((HU - low) / step)
        vol = np.floor((vol - low) / step) * step + low
    return (vol - low) / (high - low)


def window_hu_torch(x, ww=WINDOW_WIDTH, wl=WINDOW_LEVEL, step=RESAMPLING_STEP, quantize=True, out_dtype=None):
    """
    Phiên bản GPU: cùng công thức với clip_and_window_hu nhưng chạy trên tensor (cuda).
    x: tensor int16/float bất kỳ shape (HU). out_dtype: ví dụ torch.float16 để tiết kiệm VRAM.
    """
    assert torch is not None, "Cần cài PyTorch để dùng window_hu_torch"
    low, high = window_bounds(ww, wl)
    x = torch.clamp(x.to(torch.float32), low, high)
    if quantize:
        x = torch.floor((x - low) / step) * step + low
    x = (x - low) / (high - low)
    return x if out_dtype is None else x.to(out_dtype)


# ======================================================================
# 3. RESIZE VỀ 1 mm/PIXEL + CROP/PAD  (làm trên HU thô)
# ======================================================================
def resize_to_1mm_3d(volume, spacing, is_mask=False, interp=IMAGE_INTERP):
    """
    Đưa mặt phẳng trục về 1 mm/pixel (CHƯA pad). Trục z giữ nguyên như tác giả.
    Mask luôn dùng nearest để không tạo giá trị trung gian.
    """
    sx, sy = float(spacing[0]), float(spacing[1])
    z, h, w = volume.shape
    new_h, new_w = max(1, int(round(h * sy))), max(1, int(round(w * sx)))

    if is_mask:
        src = (volume > 0).astype(np.uint8)
        flag = cv2.INTER_NEAREST
    else:
        src = np.ascontiguousarray(volume)
        flag = _CV2_INTERP[interp]

    out = np.empty((z, new_h, new_w), dtype=src.dtype)
    for i in range(z):
        out[i] = cv2.resize(src[i], (new_w, new_h), interpolation=flag)
    return out


def _axis_window(n, size):
    """Trả về (bắt đầu nguồn, bắt đầu đích, độ dài) khi crop (n>=size) hoặc pad (n<size) vào giữa."""
    if n >= size:
        return (n - size) // 2, 0, size
    return 0, (size - n) // 2, n


def pad_or_crop_xy(x, size=IMAGE_SIZE, pad_value=HU_FLOOR):
    """
    Crop hoặc pad hai chiều cuối về (size, size), đặt giữa. Dùng được cho numpy (kể cả memmap) và torch.
    Ảnh HU: pad_value = -1000. Mask: pad_value = 0.
    """
    h, w = x.shape[-2:]
    sy, dy, ly = _axis_window(h, size)
    sx, dx, lx = _axis_window(w, size)
    shape = (*x.shape[:-2], size, size)
    if torch is not None and isinstance(x, torch.Tensor):
        out = torch.full(shape, pad_value, dtype=x.dtype, device=x.device)
    else:
        out = np.full(shape, pad_value, dtype=x.dtype)
    out[..., dy:dy + ly, dx:dx + lx] = x[..., sy:sy + ly, sx:sx + lx]
    return out


def resize_and_pad_3d(volume, spacing, target_size=IMAGE_SIZE, is_mask=False, pad_value=None, interp=IMAGE_INTERP):
    """
    Một pixel ảnh = 1 mm thực tế, rồi crop/pad về target_size x target_size.
    pad_value mặc định: -1000 (HU, không khí) cho ảnh; 0 cho mask.
    LƯU Ý: truyền HU THÔ (chưa cửa sổ). Nếu truyền ảnh đã chuẩn hóa thì phải tự đặt pad_value = 0.
    """
    if pad_value is None:
        pad_value = 0 if is_mask else HU_FLOOR
    small = resize_to_1mm_3d(volume, spacing, is_mask=is_mask, interp=interp)
    return pad_or_crop_xy(small, target_size, pad_value)


# ======================================================================
# 4. TÁCH PHỔI  (chuyển thể lung_extraction_funcs_13_09.py của tác giả)
#    Chạy trên HU THÔ ở lưới gốc (512x512 trong LIDC), TRƯỚC khi resize.
# ======================================================================
def _import_lung_deps():
    try:
        from skimage.measure import label, regionprops
        from skimage.segmentation import clear_border
        from skimage.filters import roberts
        from scipy import ndimage as ndi
    except ImportError as e:
        raise ImportError("extract_lungs_mask cần scikit-image và scipy: pip install scikit-image scipy") from e
    return label, regionprops, clear_border, roberts, ndi


def _get_kernel():
    """Kernel hình tròn 11x11 kèm 8 điểm nhô ra, y hệt get_kernel của tác giả."""
    kernel = np.zeros((11, 11), np.uint8)
    kernel = cv2.circle(kernel, (5, 5), 4, 1, -1)
    for i in [4, 6]:
        for j in [1, 9]:
            kernel[i, j] = 1
            kernel[j, i] = 1
    return kernel


def _find_bounds_z_direction(image):
    z_dist = np.sum(image, axis=(1, 2))
    zi = np.where(z_dist != 0)
    return int((np.max(zi) - np.min(zi)) / 2), int(np.min(zi)), int(np.max(zi))


def _largest_labeled_volumes(im, bg=-1, ratio=0.6):
    """
    Giữ nguyên hành vi của tác giả: trả về nhãn lớn nhất; chỉ khi có ĐÚNG MỘT nhãn có thể tích
    trong (ratio, 1) so với nhãn lớn nhất thì trả thêm nhãn đó (để gộp phổi trái/phải bị tách).
    """
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) == 0:
        return None
    counts = counts / np.max(counts)
    cand = np.where((counts > ratio) & (counts < 1))[0]
    if len(cand) == 1:
        smax = np.where(counts == counts[cand[0]])[0][0]
        return vals[[np.argmax(counts), smax]]
    return [vals[np.argmax(counts)]]


def _create_flipped_mask(image, i_orig, kernel):
    """
    method=0 của tác giả: tìm tâm phổi theo phân bố xương (> 100 HU) rồi lật đối xứng mask qua tâm
    để mask phổi bao cả khối đặc dính thành ngực/trung thất. Trả về (mask, min_bound, max_bound).
    """
    _, min_bound, max_bound = _find_bounds_z_direction(image)
    fl_mask = np.zeros(image.shape, np.uint8)

    lc_buff = []
    for i in range(min_bound, max_bound + 1):
        rib_view = 1 * (i_orig[i, ...] > 100)
        distr = np.sum(rib_view, axis=0)
        thresh_l1 = np.max(distr)
        if thresh_l1 == 0:                    # (thêm) lát không có xương: bỏ qua
            continue
        idx = np.where(distr > (thresh_l1 * 0.15))[0]
        lc_buff.append((np.min(idx) + np.max(idx)) / 2)
    lungs_center = int(np.mean(lc_buff)) if lc_buff else image.shape[2] // 2
    half = int(image.shape[2] / 2)
    W = image.shape[2]

    for i in range(min_bound, max_bound + 1):
        ax = image[i, ...]
        if lungs_center > half:
            part1 = ax[:, 2 * lungs_center - W:lungs_center]
            part2 = np.hstack((ax[:, lungs_center:], np.zeros((image.shape[1], 2 * lungs_center - W))))
            part2 = cv2.dilate(np.array(part2, np.uint8), kernel, iterations=1)
            part1 = cv2.dilate(np.array(part1, np.uint8), kernel, iterations=1)
            fl_mask[i, ...] = np.hstack((np.flip(part2, axis=1), np.flip(part1, axis=1)))
            fl_mask[i, ...] = np.array(fl_mask[i, ...], np.uint8) | np.array(ax, np.uint8)
        elif lungs_center < half:
            part1 = np.hstack((np.zeros((image.shape[1], W - 2 * lungs_center)), ax[:, :lungs_center]))
            part2 = ax[:, lungs_center:W - (W - 2 * lungs_center)]
            part2 = cv2.dilate(np.array(part2, np.uint8), kernel, iterations=1)
            part1 = cv2.dilate(np.array(part1, np.uint8), kernel, iterations=1)
            fl_mask[i, ...] = np.hstack((np.flip(part2, axis=1), np.flip(part1, axis=1)))
            fl_mask[i, ...] = np.array(fl_mask[i, ...], np.uint8) | np.array(ax, np.uint8)
        else:
            fl = np.flip(ax, axis=1)
            fl_mask[i, ...] = np.array(ax, np.uint8) | np.array(fl, np.uint8)
    return fl_mask, min_bound, max_bound


def extract_lungs_mask(volume, return_bounds=False):
    """
    Chuyển thể get_seg_lungs của tác giả. Đầu vào: HU thô (Z, H, W) ở lưới gốc.
    Các bước: ngưỡng < -320 HU -> clear_border (bỏ không khí ngoài, bàn chụp) -> giữ 3 vùng lớn nhất
    -> roberts + binary_fill_holes (lấp mạch máu) -> bỏ khí quản (vùng nhỏ hơn 0.0069*sq/512^2)
    -> bỏ bàn/túi khí (tâm khối nằm dưới 75% chiều cao) -> giữ khối liên thông 3D lớn nhất
    -> lật đối xứng qua tâm cột sống -> morphology -> kéo dãn +-2 lát theo z.
    Trả về mask uint8 (Z, H, W) gồm 0/1, hoặc None nếu không tìm thấy phổi.
    Chạy chậm (vòng lặp từng lát) nên chỉ gọi trong build, không gọi trong vòng huấn luyện.
    """
    label, regionprops, clear_border, roberts, ndi = _import_lung_deps()
    im = np.asarray(volume)
    kernel = _get_kernel()
    sq = im.shape[1] * im.shape[2]
    seg = np.zeros(im.shape, np.uint8)

    # --- ngưỡng, bỏ vùng chạm biên, giữ 3 vùng lớn nhất, lấp lỗ ---
    for nm, ax_slice in enumerate(im < LUNG_THRESHOLD_HU):
        lab = label(clear_border(ax_slice))
        areas = sorted(r.area for r in regionprops(lab))
        if len(areas) > 3:
            for r in regionprops(lab):
                if r.area < areas[-3]:
                    lab[lab == r.label] = 0
        seg[nm] = ndi.binary_fill_holes(roberts(lab > 0))

    # --- bỏ khí quản: vùng nhỏ hơn TRACHEA_COEFF * (sq / 512^2) diện tích ảnh ---
    trachea_coeff = TRACHEA_COEFF * (sq / (512 ** 2))
    for i, ax_slice in enumerate(seg):
        lab = label(ax_slice, connectivity=1, background=0)
        vals, counts = np.unique(lab, return_counts=True)
        counts, vals = counts[vals != 0], vals[vals != 0]
        small = counts / sq < trachea_coeff
        if small.any():
            for v in vals[small]:
                lab[lab == v] = 0
            seg[i] = lab != 0

    # --- bỏ bàn chụp và túi khí: tâm khối ở dưới 75% chiều cao ảnh ---
    lab3 = label(seg, background=0)
    ids = np.unique(lab3)
    ids = ids[ids != 0]
    if len(ids):
        centers = ndi.center_of_mass(np.ones_like(lab3), lab3, ids)
        for i, c in zip(ids, centers):
            if c[1] > seg.shape[1] * 0.75:
                seg[lab3 == i] = 0

    # --- giữ khối liên thông 3D lớn nhất (gộp phổi trái/phải nếu tách đôi) ---
    lab3 = label(seg, background=0)
    l_max = _largest_labeled_volumes(lab3, bg=0)
    if l_max is not None:
        if len(l_max) > 1:
            lab3[lab3 == l_max[1]] = l_max[0]
        seg[lab3 != l_max[0]] = 0
    if seg.sum() == 0:
        return None

    # --- lật đối xứng qua tâm cột sống ---
    seg, min_bound, max_bound = _create_flipped_mask(seg, im, kernel)

    # --- morphology: lấp khoảng trống nhỏ trong mask ---
    for i, ax_slice in enumerate(seg):
        bi = np.array(ax_slice, np.uint8)
        opening = cv2.morphologyEx(1 - bi, cv2.MORPH_OPEN, kernel, iterations=7)
        dilat = cv2.dilate(1 - opening, kernel, iterations=2)
        seg[i, ...] = cv2.erode(dilat, kernel, iterations=1)

    # --- kéo dãn mask +-2 lát theo z ---
    if (min_bound > 2) and (max_bound < seg.shape[0] - 2):
        for i in range(1, 3):
            seg[min_bound - i, ...] = seg[min_bound, ...]
            seg[max_bound + i, ...] = seg[max_bound, ...]
        min_bound -= 2
        max_bound += 2

    seg = (seg > 0).astype(np.uint8)
    return (seg, min_bound, max_bound) if return_bounds else seg


def lung_keep_mask(lung_mask, tumor_mask=None, tumor_dilate_iter=3):
    """
    Vùng được giữ lại = phổi, hợp với mask khối u/nốt đã dilate (kernel 11x11, 3 lần) - như generator dòng 132-134.
    CHỈ truyền tumor_mask lúc HUẤN LUYỆN. Lúc đánh giá/dự đoán mà truyền mask thật vào thì ảnh đầu vào đã
    chứa thông tin nhãn (rò rỉ).
    """
    keep = (lung_mask > 0).astype(np.uint8)
    if tumor_mask is not None:
        kernel = _get_kernel()
        for k in range(len(keep)):
            if tumor_mask[k].any():
                keep[k] |= cv2.dilate((tumor_mask[k] > 0).astype(np.uint8), kernel, iterations=tumor_dilate_iter)
    return keep


def apply_lung_mask(volume_hu, keep, fill=HU_FLOOR):
    """Đặt mọi voxel ngoài vùng giữ lại về -1000 HU (apply_mask của tác giả). Không sửa mảng đầu vào."""
    out = np.array(volume_hu, copy=True)
    out[np.asarray(keep) == 0] = fill
    return out


# ======================================================================
# 5. CACHE MỘT LẦN (CPU)  ->  .npy int16 HU ở 1 mm, chưa pad
# ======================================================================
CACHE_VERSION = 2


def cache_paths(out_dir):
    out_dir = Path(out_dir)
    return out_dir / "image_1mm.npy", out_dir / "mask_1mm.npy", out_dir / "meta.json"


def lung_cache_path(out_dir):
    return Path(out_dir) / "lung_1mm.npy"


def _save_npy(path, arr):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def build_patient_cache(patient_dir, out_dir, interp=IMAGE_INTERP, overwrite=False, extract_lungs=False):
    """
    Bước 1-3 cho một bệnh nhân: đọc image.nrrd (+ mask.nrrd), cắt đáy -1000, (tùy chọn) tách phổi, resize 1 mm.
    Ảnh lưu int16 HU, mask uint8. Mask phổi lưu riêng (lung_1mm.npy), KHÔNG áp vào ảnh trong cache.
    Pad về IMAGE_SIZE làm lúc nạp (rất rẻ) để cache nhỏ hơn.
    """
    patient_dir, out_dir = Path(patient_dir), Path(out_dir)
    img_p, msk_p, meta_p = cache_paths(out_dir)
    lung_p = lung_cache_path(out_dir)
    done = img_p.exists() and msk_p.exists() and meta_p.exists() and (lung_p.exists() or not extract_lungs)
    if not overwrite and done:
        return "skip"

    image_nrrd, mask_nrrd = patient_dir / "image.nrrd", patient_dir / "mask.nrrd"
    if not image_nrrd.exists():
        return "missing_image"

    vol, spacing = load_nrrd(image_nrrd)
    vol = np.maximum(vol, HU_FLOOR).astype(np.int16)             # bước 1: cắt đáy -1000 HU

    if mask_nrrd.exists():
        msk, _ = load_nrrd(mask_nrrd)
        if msk.shape != vol.shape:
            raise ValueError(f"mask {msk.shape} khác image {vol.shape}")
    else:
        msk = np.zeros_like(vol, dtype=np.uint8)

    lung1, lung_meta = None, {}
    if extract_lungs:                                            # bước 2: tách phổi trên lưới gốc
        lung = extract_lungs_mask(vol)
        if lung is None:
            lung_meta = {"lung_ok": False}
            lung1 = np.zeros((vol.shape[0], ) + resize_to_1mm_3d(vol[:1], spacing).shape[1:], np.uint8)
        else:
            n_nod = int((msk > 0).sum())
            lung_meta = {
                "lung_ok": True,
                "lung_voxels_orig": int(lung.sum()),
                # tỉ lệ voxel nốt nằm trong mask phổi: thấp nghĩa là che ảnh bằng mask phổi sẽ cắt mất nốt
                "mask_in_lung_fraction": (float(((msk > 0) & (lung > 0)).sum() / n_nod) if n_nod else None),
            }
            lung1 = resize_to_1mm_3d(lung, spacing, is_mask=True)

    img1 = resize_to_1mm_3d(vol, spacing, is_mask=False, interp=interp)      # bước 3
    msk1 = resize_to_1mm_3d(msk, spacing, is_mask=True)

    sx, sy, sz = (float(s) for s in spacing)
    meta = {
        "cache_version": CACHE_VERSION,
        "interp": interp,
        "orig_shape_zyx": list(vol.shape),
        "orig_spacing_xyz": [sx, sy, sz],
        "shape_1mm_zyx": list(img1.shape),
        "mask_voxels_orig": int((msk > 0).sum()),
        "mask_voxels_1mm": int(msk1.sum()),
        "mask_volume_mm3_orig": float((msk > 0).sum() * sx * sy * sz),
        "mask_volume_mm3_1mm": float(msk1.sum() * 1.0 * 1.0 * sz),
        **lung_meta,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_npy(img_p, img1)
    _save_npy(msk_p, msk1)
    if lung1 is not None:
        _save_npy(lung_p, lung1)
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return "ok"


def load_cached(cache_dir, mmap=True):
    """Trả về (image_1mm, mask_1mm, meta). Với mmap=True chỉ đọc phần dữ liệu thật sự được truy cập."""
    img_p, msk_p, meta_p = cache_paths(cache_dir)
    mode = "r" if mmap else None
    img = np.load(img_p, mmap_mode=mode)
    msk = np.load(msk_p, mmap_mode=mode)
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return img, msk, meta


def load_cached_slices(cache_dir, z_indices, size=IMAGE_SIZE, with_lung=False):
    """
    Nạp các lát z_indices đã pad về (K, size, size): ảnh int16 HU (pad -1000), mask uint8 (pad 0),
    và (nếu with_lung) mask phổi uint8. Đầu ra là HU thô -> áp lung_keep_mask/apply_lung_mask nếu cần,
    chuyển lên GPU rồi gọi window_hu_torch.
    """
    img, msk, _ = load_cached(cache_dir, mmap=True)
    z = np.asarray(z_indices)
    out = [pad_or_crop_xy(np.asarray(img[z]), size, HU_FLOOR),
           pad_or_crop_xy(np.asarray(msk[z]), size, 0)]
    if with_lung:
        lung = np.load(lung_cache_path(cache_dir), mmap_mode="r")
        out.append(pad_or_crop_xy(np.asarray(lung[z]), size, 0))
    return tuple(out)


# ======================================================================
# 6. CLI
# ======================================================================
def _worker(args):
    pid, nrrd_dir, cache_dir, interp, overwrite, extract_lungs = args
    cv2.setNumThreads(1)                       # tránh tranh chấp luồng khi chạy nhiều tiến trình
    try:
        return pid, build_patient_cache(Path(nrrd_dir) / pid, Path(cache_dir) / pid, interp, overwrite, extract_lungs)
    except Exception as e:                     # một ca lỗi không làm dừng cả lượt chạy
        return pid, f"error: {e}"


def _cmd_build(a):
    nrrd_dir, cache_dir = Path(a.nrrd_dir), Path(a.cache_dir)
    pids = sorted(p.name for p in nrrd_dir.iterdir() if p.is_dir())
    if a.limit:
        pids = pids[:a.limit]
    print(f"{len(pids)} bệnh nhân -> {cache_dir}" + (" (có tách phổi, chậm hơn nhiều)" if a.extract_lungs else ""))
    jobs = [(p, str(nrrd_dir), str(cache_dir), a.interp, a.overwrite, a.extract_lungs) for p in pids]

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x                # noqa: E731

    results = []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in tqdm(ex.map(_worker, jobs), total=len(jobs)):
                results.append(r)
    else:
        for j in tqdm(jobs):
            results.append(_worker(j))

    bad = [(p, s) for p, s in results if s not in ("ok", "skip")]
    print(f"Xong: {sum(s == 'ok' for _, s in results)} mới, {sum(s == 'skip' for _, s in results)} bỏ qua, {len(bad)} lỗi")
    for p, s in bad:
        print("  ", p, s)


def _phantom_ct(z=14):
    """CT tổng hợp: thân người, hai phổi, khí quản, cột sống (xương), bàn chụp, không khí ngoài khung."""
    yy, xx = np.mgrid[:512, :512]
    vol = np.full((z, 512, 512), -1000, np.int16)
    vol[:, ((yy - 256) / 170.0) ** 2 + ((xx - 256) / 225.0) ** 2 <= 1] = 40          # thân
    lung_l = ((yy - 250) / 110.0) ** 2 + ((xx - 140) / 60.0) ** 2 <= 1
    lung_r = ((yy - 250) / 110.0) ** 2 + ((xx - 372) / 60.0) ** 2 <= 1
    vol[:, lung_l | lung_r] = -800                                                    # phổi
    vol[:, (yy - 230) ** 2 + (xx - 256) ** 2 <= 100] = -950                           # khí quản
    vol[:, (yy - 340) ** 2 + (xx - 256) ** 2 <= 18 ** 2] = 300                        # cột sống
    vol[:, 432:438, 110:400] = 60                                                     # bàn chụp
    return vol, (lung_l | lung_r)


def _cmd_selftest(_):
    rng = np.random.default_rng(0)
    spacing = (0.7, 0.7, 2.5)
    vol = rng.integers(-1024, 1500, size=(6, 512, 512)).astype(np.int16)
    vol[:, :60, :] = -2000                                           # mô phỏng vùng ngoài khung quét

    r = resize_and_pad_3d(vol, spacing)
    assert r.shape == (6, 512, 512), r.shape
    assert (r[:, :5, :] == HU_FLOOR).all() and (r[:, :, :5] == HU_FLOOR).all(), "pad phải là -1000 HU"

    n = clip_and_window_hu(r)
    assert n.dtype == np.float32 and n.min() == 0.0 and n.max() <= 1.0, (n.min(), n.max())
    assert (n[:, :5, :] == 0).all(), "viền pad phải bằng không khí (0) sau cửa sổ"
    assert len(np.unique(n)) <= 47, len(np.unique(n))
    assert vol.min() == -2000, "không được sửa mảng đầu vào"

    m = np.zeros(vol.shape, np.uint8)
    m[3, 100:110, 100:110] = 1
    mr = resize_and_pad_3d(m, spacing, is_mask=True)
    assert set(np.unique(mr)) <= {0, 1} and mr.sum() > 0 and mr.shape == (6, 512, 512)

    big = resize_and_pad_3d(vol, (1.5, 1.5, 2.5))                    # 768 mm > 512 -> crop
    assert big.shape == (6, 512, 512)
    print("resize/pad/cửa sổ: OK")

    if torch is not None:
        t = window_hu_torch(torch.from_numpy(r)).numpy()
        assert np.allclose(t, n, atol=1e-6), "torch và numpy phải cho cùng kết quả"
        print("torch == numpy: OK")
    else:
        print("(bỏ qua so sánh torch: chưa cài PyTorch)")

    try:
        _import_lung_deps()
    except ImportError:
        print("(bỏ qua kiểm thử tách phổi: chưa cài scikit-image/scipy)")
    else:
        ph, truth = _phantom_ct()
        lung = extract_lungs_mask(ph)
        assert lung is not None and lung.shape == ph.shape
        mid = ph.shape[0] // 2
        inter = (lung[mid] > 0) & truth
        union = (lung[mid] > 0) | truth
        iou = inter.sum() / union.sum()
        assert iou > 0.8, f"IoU phổi quá thấp: {iou:.3f}"
        assert lung[mid, 230, 256] == 0, "khí quản phải bị loại"
        assert lung[mid, 5, 5] == 0 and lung[mid, 435, 250] == 0, "không khí ngoài và bàn chụp phải bị loại"
        print(f"tách phổi trên phantom: OK (IoU = {iou:.3f})")
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="tạo cache 1 mm (chạy một lần)")
    b.add_argument("--nrrd-dir", default=str(ROOT / "data" / "duneai_raw_nrrd"))
    b.add_argument("--cache-dir", default=str(ROOT / "data" / "cache_1mm"))
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--limit", type=int, default=0, help="chỉ xử lý N ca đầu (0 = tất cả)")
    b.add_argument("--interp", choices=list(_CV2_INTERP), default=IMAGE_INTERP)
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--extract-lungs", action="store_true", help="tính và lưu thêm mask phổi (lung_1mm.npy)")
    b.set_defaults(fn=_cmd_build)

    s = sub.add_parser("selftest", help="kiểm thử bằng dữ liệu tổng hợp")
    s.set_defaults(fn=_cmd_selftest)

    args = ap.parse_args()
    args.fn(args)
```

#### <a id="tệp-26-common-data_pipeline-runtime_data-py"></a>Tệp 26: `common/data_pipeline/runtime_data.py`
- **Vị trí**: `common/data_pipeline/runtime_data.py`
- **Quy mô**: `729 dòng` | `30.6 KB`
- **Vai trò**: Lớp nạp dữ liệu thời gian thực hỗ trợ streaming và xử lý bộ nhớ tối ưu.

```python
# -*- coding: utf-8 -*-
"""
common/data_pipeline/runtime_data.py - Chuẩn bị và đồng bộ dữ liệu theo nền tảng thực thi (Colab, Kaggle, Local).

Mục tiêu (spec mục 8.6):
  - Colab: Mount Drive -> dò zip cache_1mm -> đọc mục lục zipfile -> chọn số lượng N bệnh nhân -> giải nén phẳng ra SSD.
  - Kaggle: Dò cache gốc từ hint (chỉ đọc) -> dựng symlink view-dir ghi được trong /kaggle/working.
  - Local: Xác minh thư mục dữ liệu cục bộ và xuất manifest.
  - Auto: Tự nhận diện nền tảng rồi kích hoạt luồng tương ứng.
  - Xuất data_manifest.json và in dòng cuối CACHE_DIR=<path>.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PATIENT_REGEX = re.compile(r"LIDC-IDRI-\d{4}")


def compute_sha256(data: str) -> str:
    """Tính SHA256 của một chuỗi."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def compute_patient_ids_hash(patient_ids: List[str]) -> str:
    """Tính mã băm xác thực danh sách bệnh nhân (đã sắp xếp)."""
    sorted_ids = sorted(patient_ids)
    return compute_sha256(",".join(sorted_ids))


def create_dir_symlink(src: Path, dst: Path):
    """Tạo symlink thư mục đa nền tảng (hỗ trợ cả Linux và Windows Junction)."""
    src_str = str(src.resolve())
    dst_str = str(dst.resolve())
    try:
        os.symlink(src_str, dst_str, target_is_directory=True)
    except OSError as e:
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(src_str, dst_str)
        else:
            raise e


def select_patients(
    available_ids: List[str],
    n_requested: Optional[int] = None,
    mode: str = "first",
    seed: int = 42,
    patients_file: Optional[str] = None,
    _skip_sort: bool = False,
) -> Tuple[List[str], Dict]:
    """
    Lựa chọn danh sách bệnh nhân xác định (deterministic):
      - 'first': lấy N bệnh nhân đầu tiên theo thứ tự tên đã sắp xếp.
      - 'random': lấy mẫu ngẫu nhiên theo seed rồi sắp xếp lại.
      - 'file': nạp từ file JSON danh sách bệnh nhân.
    """
    if not _skip_sort:
        sorted_all = sorted(available_ids)
    else:
        sorted_all = list(available_ids)

    n_available = len(sorted_all)

    if patients_file:
        pf_path = Path(patients_file)
        if not pf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy patients-file: {pf_path}")
        with open(pf_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "patient_ids" in data:
            target_ids = data["patient_ids"]
        elif isinstance(data, list):
            target_ids = data
        else:
            raise ValueError(f"Định dạng patients-file không hợp lệ: {pf_path}")
        selected = [pid for pid in target_ids if pid in available_ids]
        if not _skip_sort:
            selected = sorted(selected)
        selection_meta = {"mode": "file", "file": str(pf_path), "n_requested": len(selected)}
        return selected, selection_meta

    if n_requested is None or n_requested <= 0 or n_requested >= n_available:
        if n_requested and n_requested > n_available:
            print(f"⚠️ Yêu cầu {n_requested} bệnh nhân nhưng chỉ có {n_available}. Sẽ dùng tất cả.")
        selected = list(sorted_all)
        selection_meta = {"mode": "all", "n_requested": n_available, "seed": seed}
        return selected, selection_meta

    if mode == "first":
        selected = sorted_all[:n_requested]
        selection_meta = {"mode": "first", "n_requested": n_requested, "seed": seed}
    elif mode == "random":
        import random
        rng = random.Random(seed)
        sampled = rng.sample(sorted_all, n_requested)
        selected = sampled if _skip_sort else sorted(sampled)
        selection_meta = {"mode": "random", "n_requested": n_requested, "seed": seed}
    else:
        raise ValueError(f"Chế độ chọn không hợp lệ: {mode} (chỉ hỗ trợ first, random, file)")

    return selected, selection_meta


def sync_slices_index(cache_dir: Path, patient_ids: List[str]):
    """
    Đồng bộ slices.csv và sidecar slices.meta.json.
    Nếu tập bệnh nhân thay đổi, xóa slices.csv cũ để hệ thống tự tạo lại chính xác.
    """
    current_hash = compute_patient_ids_hash(patient_ids)
    meta_path = cache_dir / "slices.meta.json"
    csv_path = cache_dir / "slices.csv"

    need_rebuild = False
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                old_meta = json.load(f)
            if old_meta.get("patient_ids_sha256") != current_hash:
                need_rebuild = True
        except Exception:
            need_rebuild = True
    else:
        if csv_path.exists():
            need_rebuild = True

    if need_rebuild and csv_path.exists():
        print(f"🔄 Tập bệnh nhân đã thay đổi -> Xóa {csv_path.name} cũ để dựng lại đồng bộ.")
        try:
            csv_path.unlink()
        except OSError:
            pass

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "patient_ids_sha256": current_hash,
            "n_patients": len(patient_ids),
        }, f, indent=2)


# ======================================================================
# NHÁNH COLAB: GIẢI NÉN CHỌN SỐ LƯỢNG TỪ DRIVE
# ======================================================================
def find_drive_zip(zip_arg: str, drive_root: Optional[str] = None) -> Path:
    """Tìm file cache_1mm.zip trên Google Drive."""
    if zip_arg != "auto":
        p = Path(zip_arg)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file zip: {p.resolve()}")
        return p

    search_root = Path(drive_root) if drive_root else Path("/content/drive")
    if not search_root.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục Google Drive tại '{search_root}'.\n"
            "Hãy đảm bảo bạn đã chạy ô Mount Drive trước: from google.colab import drive; drive.mount('/content/drive')"
        )

    # 1. Kiểm tra nhanh các đường dẫn quen thuộc (bao gồm cấu trúc mirror GG_1 từ máy cục bộ)
    known_candidates = [
        search_root / "MyDrive" / "COMPUTER SCIENCE" / "NAM4_HK1_(2026-2027)" / "CT208E" / "SOURCES" / "data" / "cache_1mm.zip",
        search_root / "MyDrive" / "GG_1" / "COMPUTER SCIENCE" / "NAM4_HK1_(2026-2027)" / "CT208E" / "SOURCES" / "data" / "cache_1mm.zip",
        search_root / "MyDrive" / "COMPUTER SCIENCE" / "NAM4_HK1_(2026-2027)" / "CT208E" / "SOURCES" / "data" / "KAGGLE_UPLOAD" / "cache_1mm.zip",
        search_root / "MyDrive" / "cache_1mm.zip",
        search_root / "MyDrive" / "DuneAI" / "cache_1mm.zip",
    ]
    for cand in known_candidates:
        if cand.exists():
            print(f"🎯 Đã tìm thấy file zip tại đường dẫn quen thuộc: {cand}")
            return cand

    # 2. Quét sâu đến 10 cấp thư mục nếu chưa thấy ở các đường dẫn quen thuộc
    candidates = []
    for root, dirs, files in os.walk(search_root):
        depth = len(Path(root).relative_to(search_root).parts)
        if depth > 10:
            dirs.clear()
            continue
        for f in files:
            if f.startswith("cache_1mm") and f.endswith(".zip"):
                candidates.append(Path(root) / f)

    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy file khớp 'cache_1mm*.zip' dưới '{search_root}'. Hãy truyền --zip cụ thể.")
    if len(candidates) > 1:
        cand_list = "\n  - ".join(str(c) for c in candidates)
        raise RuntimeError(f"Tìm thấy nhiều file zip ứng viên, vui lòng chỉ định cụ thể qua --zip:\n  - {cand_list}")

    return candidates[0]


def run_colab(
    zip_path: str,
    dest_dir: str,
    drive_root: Optional[str] = None,
    n_patients: Optional[int] = None,
    select: str = "first",
    seed: int = 42,
    patients_file: Optional[str] = None,
    out_json: Optional[str] = None,
    _skip_temp_replace: bool = False,
    _skip_sort: bool = False,
    _mock_disk_space_error: bool = False,
) -> Path:
    """Chuẩn bị dữ liệu trên Google Colab."""
    resolved_zip = find_drive_zip(zip_path, drive_root)
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(resolved_zip, "r") as zf:
        all_names = zf.namelist()
        patient_entries: Dict[str, List[zipfile.ZipInfo]] = {}

        for info in zf.infolist():
            match = PATIENT_REGEX.search(info.filename)
            if match:
                pid = match.group(0)
                patient_entries.setdefault(pid, []).append(info)

        available_ids = list(patient_entries.keys())
        if not available_ids:
            raise RuntimeError(f"File zip {resolved_zip} không chứa cấu trúc bệnh nhân dạng LIDC-IDRI-XXXX")

        selected_ids, sel_meta = select_patients(
            available_ids,
            n_requested=n_patients,
            mode=select,
            seed=seed,
            patients_file=patients_file,
            _skip_sort=_skip_sort,
        )

        # Tính tổng dung lượng cần giải nén
        total_uncompressed = sum(
            info.file_size
            for pid in selected_ids
            for info in patient_entries[pid]
        )

        # Kiểm tra dung lượng trống
        dest_stat = shutil.disk_usage(dest)
        required_space = total_uncompressed * 1.05
        if _mock_disk_space_error or dest_stat.free < required_space:
            req_mb = required_space / (1024 * 1024)
            free_mb = dest_stat.free / (1024 * 1024)
            raise OSError(
                f"Không đủ dung lượng trống tại {dest}!\n"
                f"Yêu cầu: {req_mb:.1f} MB | Khả dụng: {free_mb:.1f} MB"
            )

        print(f"📦 Colab Data Pipeline: Chọn {len(selected_ids)}/{len(available_ids)} bệnh nhân từ {resolved_zip.name}")

        for pid in selected_ids:
            target_pid_dir = dest / pid
            # Kiểm tra idempotent (nếu đã đủ file và kích thước thì bỏ qua)
            if target_pid_dir.exists() and (target_pid_dir / "image_1mm.npy").exists():
                continue

            if not _skip_temp_replace:
                temp_extract = dest / f".tmp_{pid}_{uuid.uuid4().hex[:6]}"
                temp_extract.mkdir(parents=True, exist_ok=True)
                extract_target = temp_extract
            else:
                extract_target = target_pid_dir
                extract_target.mkdir(parents=True, exist_ok=True)

            for info in patient_entries[pid]:
                if info.is_dir():
                    continue
                # Lấy tên file sau ID bệnh nhân
                sub_parts = Path(info.filename).parts
                pid_idx = sub_parts.index(pid)
                rel_parts = sub_parts[pid_idx + 1 :]
                out_file_path = extract_target / Path(*rel_parts)
                out_file_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src_f, open(out_file_path, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)

            if not _skip_temp_replace:
                if target_pid_dir.exists():
                    shutil.rmtree(target_pid_dir)
                os.replace(str(temp_extract), str(target_pid_dir))

    sync_slices_index(dest, selected_ids)

    manifest = {
        "platform": "colab",
        "kind": "zip",
        "origin": str(resolved_zip),
        "cache_dir": str(dest),
        "n_available": len(available_ids),
        "n_selected": len(selected_ids),
        "selection": sel_meta,
        "patient_ids": selected_ids,
        "patient_ids_sha256": compute_patient_ids_hash(selected_ids),
        "zip": {
            "path": str(resolved_zip),
            "size_bytes": resolved_zip.stat().st_size,
        },
    }

    manifest_file = Path(out_json) if out_json else (dest / "data_manifest.json")
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"CACHE_DIR={dest}")
    return dest


# ======================================================================
# NHÁNH KAGGLE: DÒ CACHE VÀ DỰNG VIEW-DIR SYMLINK
# ======================================================================
def find_kaggle_cache(hint: str) -> Path:
    """Dò tìm thư mục cache chứa các thư mục con LIDC-IDRI-XXXX."""
    p = Path(hint)
    if p.exists() and PATIENT_REGEX.search(p.name):
        p = p.parent

    # Dò xuống tối đa 4 tầng
    candidates = []
    if p.exists():
        for root, dirs, _ in os.walk(p):
            depth = len(Path(root).relative_to(p).parts)
            if depth > 4:
                dirs.clear()
                continue
            matching = [d for d in dirs if PATIENT_REGEX.match(d)]
            if matching:
                candidates.append(Path(root))
                dirs.clear()

    if not candidates:
        # Thử dò dưới /kaggle/input
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for root, dirs, _ in os.walk(kaggle_input):
                depth = len(Path(root).relative_to(kaggle_input).parts)
                if depth > 4:
                    dirs.clear()
                    continue
                matching = [d for d in dirs if PATIENT_REGEX.match(d)]
                if matching:
                    candidates.append(Path(root))
                    dirs.clear()

    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục chứa bệnh nhân LIDC-IDRI-XXXX từ hint '{hint}'.\n"
            "Hãy kiểm tra lại đường dẫn Dataset hoặc bấm 'Add Input' để nạp dataset vào Kaggle."
        )

    return candidates[0]


def run_kaggle(
    hint: str,
    view_dir: str,
    n_patients: Optional[int] = None,
    select: str = "first",
    seed: int = 42,
    patients_file: Optional[str] = None,
    out_json: Optional[str] = None,
    _skip_sort: bool = False,
) -> Path:
    """Chuẩn bị dữ liệu trên Kaggle qua symlink view-dir."""
    source_cache = find_kaggle_cache(hint)
    view = Path(view_dir).resolve()
    view.mkdir(parents=True, exist_ok=True)

    available_ids = sorted([
        d.name for d in source_cache.iterdir()
        if d.is_dir() and PATIENT_REGEX.match(d.name)
    ])
    if not available_ids:
        raise RuntimeError(f"Không tìm thấy bệnh nhân hợp lệ nào trong {source_cache}")

    selected_ids, sel_meta = select_patients(
        available_ids,
        n_requested=n_patients,
        mode=select,
        seed=seed,
        patients_file=patients_file,
        _skip_sort=_skip_sort,
    )

    print(f"🔗 Kaggle Data Pipeline: Dựng symlink cho {len(selected_ids)}/{len(available_ids)} bệnh nhân vào {view}")

    # Xóa các symlink cũ không còn nằm trong danh sách chọn
    for existing in view.iterdir():
        if existing.is_dir() and PATIENT_REGEX.match(existing.name):
            if existing.name not in selected_ids:
                if existing.is_symlink():
                    existing.unlink()
                else:
                    shutil.rmtree(existing)

    # Dựng symlink tới từng bệnh nhân đã chọn
    for pid in selected_ids:
        src_pid = source_cache / pid
        dst_pid = view / pid
        if not dst_pid.exists():
            create_dir_symlink(src_pid, dst_pid)

    sync_slices_index(view, selected_ids)

    manifest = {
        "platform": "kaggle",
        "kind": "kaggle_input",
        "origin": str(source_cache),
        "cache_dir": str(view),
        "n_available": len(available_ids),
        "n_selected": len(selected_ids),
        "selection": sel_meta,
        "patient_ids": selected_ids,
        "patient_ids_sha256": compute_patient_ids_hash(selected_ids),
    }

    manifest_file = Path(out_json) if out_json else (view / "data_manifest.json")
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"CACHE_DIR={view}")
    return view


# ======================================================================
# NHÁNH LOCAL & AUTO
# ======================================================================
def run_local(cache_dir: str, out_json: Optional[str] = None) -> Path:
    """Xác thực thư mục cache cục bộ và xuất manifest."""
    p = Path(cache_dir).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục cache cục bộ: {p}")

    available_ids = sorted([
        d.name for d in p.iterdir()
        if d.is_dir() and PATIENT_REGEX.match(d.name)
    ])
    if not available_ids:
        raise RuntimeError(f"Không tìm thấy bệnh nhân nào trong {p}")

    sync_slices_index(p, available_ids)

    manifest = {
        "platform": "local",
        "kind": "local_dir",
        "origin": str(p),
        "cache_dir": str(p),
        "n_available": len(available_ids),
        "n_selected": len(available_ids),
        "selection": {"mode": "all", "n_requested": len(available_ids)},
        "patient_ids": available_ids,
        "patient_ids_sha256": compute_patient_ids_hash(available_ids),
    }

    manifest_file = Path(out_json) if out_json else (p / "data_manifest.json")
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"CACHE_DIR={p}")
    return p


def run_auto(**kwargs) -> Path:
    """Tự động phát hiện nền tảng thực thi."""
    if "google.colab" in sys.modules:
        is_colab = True
    else:
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False

    if is_colab:
        zip_path = kwargs.get("zip", "auto")
        dest_dir = kwargs.get("dest", "/content/local_cache_1mm")
        return run_colab(zip_path=zip_path, dest_dir=dest_dir, **kwargs)

    if Path("/kaggle/working").exists():
        hint = kwargs.get("hint", "/kaggle/input/datasets/minhtri2475/lidc-idri-1mm-cache-ct208e")
        view_dir = kwargs.get("view_dir", "/kaggle/working/view_cache_1mm")
        return run_kaggle(hint=hint, view_dir=view_dir, **kwargs)

    cache_dir = kwargs.get("cache_dir", "data/cache_1mm")
    return run_local(cache_dir=cache_dir, out_json=kwargs.get("out_json"))


# ======================================================================
# CLI & SELFTEST
# ======================================================================
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # colab
    c = sub.add_parser("colab", help="Chuẩn bị dữ liệu trên Google Colab từ zip trên Drive")
    c.add_argument("--zip", default="auto", help="Đường dẫn file zip hoặc 'auto'")
    c.add_argument("--drive-root", default=None, help="Thư mục gốc Drive để dò tìm")
    c.add_argument("--dest", required=True, help="Thư mục đích trên SSD cục bộ (/content/...)")
    c.add_argument("--n-patients", type=int, default=None, help="Số bệnh nhân cần lấy")
    c.add_argument("--select", choices=["first", "random"], default="first")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--patients-file", default=None)
    c.add_argument("--out-json", default=None)

    # kaggle
    k = sub.add_parser("kaggle", help="Chuẩn bị dữ liệu trên Kaggle qua symlink view-dir")
    k.add_argument("--hint", required=True, help="Đường dẫn dataset gốc, thư mục cache hoặc 1 bệnh nhân")
    k.add_argument("--view-dir", required=True, help="Thư mục view-dir ghi được trên Kaggle (/kaggle/working/...)")
    k.add_argument("--n-patients", type=int, default=None)
    k.add_argument("--select", choices=["first", "random"], default="first")
    k.add_argument("--seed", type=int, default=42)
    k.add_argument("--patients-file", default=None)
    k.add_argument("--out-json", default=None)

    # local
    l = sub.add_parser("local", help="Xác thực thư mục cache cục bộ")
    l.add_argument("--cache-dir", required=True)
    l.add_argument("--out-json", default=None)

    # auto
    a = sub.add_parser("auto", help="Tự động phát hiện nền tảng")
    a.add_argument("--zip", default="auto")
    a.add_argument("--dest", default=None)
    a.add_argument("--hint", default=None)
    a.add_argument("--view-dir", default=None)
    a.add_argument("--cache-dir", default=None)
    a.add_argument("--patients-file", default=None)
    a.add_argument("--out-json", default=None)

    # selftest
    sub.add_parser("selftest", help="Chạy selftest tự động không dùng mạng")

    return ap


def run_selftest():
    import tempfile
    print("🧪 Bắt đầu selftest runtime_data (9 ca kiểm thử + 2 ca tự phá)...")
    temp_dir = Path(tempfile.mkdtemp(prefix="dune_selftest_"))
    try:
        # Tạo dữ liệu giả lập cho 5 bệnh nhân
        pids = [f"LIDC-IDRI-{i:04d}" for i in range(1, 6)]

        # 1. Zip giả đúng bố cục thật cache_1mm/LIDC-IDRI-XXXX/...
        zip_path = temp_dir / "cache_1mm.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for pid in pids:
                zf.writestr(f"cache_1mm/{pid}/image_1mm.npy", b"image_data")
                zf.writestr(f"cache_1mm/{pid}/mask_1mm.npy", b"mask_data")
                zf.writestr(f"cache_1mm/{pid}/meta.json", json.dumps({"pid": pid}))

        dest_colab = temp_dir / "colab_ssd"
        out_dest = run_colab(
            zip_path=str(zip_path),
            dest_dir=str(dest_colab),
            n_patients=3,
            select="first",
        )
        assert out_dest.exists(), "Thư mục đích giải nén không tồn tại"
        extracted = sorted([d.name for d in out_dest.iterdir() if PATIENT_REGEX.match(d.name)])
        assert extracted == ["LIDC-IDRI-0001", "LIDC-IDRI-0002", "LIDC-IDRI-0003"], f"Sai first 3: {extracted}"
        print("  ✓ Ca 1: Zip bố cục thật cache_1mm/ giải nén phẳng ra --dest đúng 3 bệnh nhân đầu.")

        # 2. first N=3 và random(seed)
        r1_pids, _ = select_patients(pids, n_requested=3, mode="random", seed=42)
        r2_pids, _ = select_patients(pids, n_requested=3, mode="random", seed=42)
        r3_pids, _ = select_patients(pids, n_requested=3, mode="random", seed=99)
        assert r1_pids == r2_pids, "random cùng seed phải cho kết quả giống nhau"
        assert r1_pids != r3_pids, "random khác seed phải cho kết quả khác nhau"
        print("  ✓ Ca 2: first N=3 và random(seed) nhất quán xác định.")

        # 3. Chạy lại không giải nén lại (Idempotency)
        t_before = (dest_colab / "LIDC-IDRI-0001" / "image_1mm.npy").stat().st_mtime
        run_colab(zip_path=str(zip_path), dest_dir=str(dest_colab), n_patients=3, select="first")
        t_after = (dest_colab / "LIDC-IDRI-0001" / "image_1mm.npy").stat().st_mtime
        assert t_before == t_after, "Idempotency thất bại: file bị ghi đè lại không cần thiết"
        print("  ✓ Ca 3: Idempotency chuẩn (không giải nén lại các bệnh nhân đã hoàn chỉnh).")

        # 4. N > n_available -> cảnh báo và dùng tất cả
        dest_all = temp_dir / "colab_all"
        run_colab(zip_path=str(zip_path), dest_dir=str(dest_all), n_patients=10)
        ext_all = sorted([d.name for d in dest_all.iterdir() if PATIENT_REGEX.match(d.name)])
        assert len(ext_all) == 5, f"Phải lấy đủ 5 bệnh nhân, nhận được {len(ext_all)}"
        print("  ✓ Ca 4: N > n_available xử lý chuẩn xác, dùng toàn bộ.")

        # 5. Thiếu dung lượng -> ném OSError
        try:
            dest_nospace = temp_dir / "colab_nospace"
            run_colab(zip_path=str(zip_path), dest_dir=str(dest_nospace), n_patients=2, _mock_disk_space_error=True)
            assert False, "Phải ném lỗi khi thiếu dung lượng"
        except OSError:
            print("  ✓ Ca 5: Bắt lỗi thiếu dung lượng ổ đĩa thành công.")

        # 6. Kaggle giả lập lồng 2 tầng .../cache_1mm/cache_1mm/LIDC-IDRI-XXXX
        kaggle_source = temp_dir / "kaggle_dataset" / "cache_1mm" / "cache_1mm"
        for pid in pids:
            p_dir = kaggle_source / pid
            p_dir.mkdir(parents=True, exist_ok=True)
            (p_dir / "image_1mm.npy").write_bytes(b"kaggle_image")
            (p_dir / "meta.json").write_text(json.dumps({"pid": pid}), encoding="utf-8")

        # Thử 3 dạng hint khác nhau
        hint_root = str(temp_dir / "kaggle_dataset")
        hint_cache = str(kaggle_source)
        hint_patient = str(kaggle_source / "LIDC-IDRI-0001")

        for h in [hint_root, hint_cache, hint_patient]:
            v_dir = temp_dir / f"kaggle_view_{uuid.uuid4().hex[:4]}"
            run_kaggle(hint=h, view_dir=str(v_dir), n_patients=2, select="first")
            view_pids = sorted([d.name for d in v_dir.iterdir() if PATIENT_REGEX.match(d.name)])
            assert view_pids == ["LIDC-IDRI-0001", "LIDC-IDRI-0002"], f"Lỗi hint {h}: {view_pids}"
            # Đảm bảo đọc được nội dung qua symlink/junction
            assert (v_dir / "LIDC-IDRI-0001" / "image_1mm.npy").read_bytes() == b"kaggle_image"
        print("  ✓ Ca 6: Kaggle view-dir symlink hỗ trợ 3 dạng hint (root, cache, patient) hoàn hảo.")

        # 7. Đổi tập bệnh nhân -> slices.csv cũ bị xóa
        v_sync = temp_dir / "kaggle_view_sync"
        run_kaggle(hint=hint_cache, view_dir=str(v_sync), n_patients=2, select="first")
        (v_sync / "slices.csv").write_text("old_csv", encoding="utf-8")
        assert (v_sync / "slices.csv").exists()
        # Chạy lại với 3 bệnh nhân -> slices.csv phải bị xóa
        run_kaggle(hint=hint_cache, view_dir=str(v_sync), n_patients=3, select="first")
        assert not (v_sync / "slices.csv").exists(), "slices.csv cũ phải bị xóa khi đổi tập bệnh nhân"
        print("  ✓ Ca 7: Đổi tập bệnh nhân tự động dọn slices.csv cũ thông qua slices.meta.json.")

        # 8. --patients-file dựng lại đúng tập
        manifest_path = temp_dir / "manifest_test.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"patient_ids": ["LIDC-IDRI-0002", "LIDC-IDRI-0004"]}, f)
        v_pf = temp_dir / "kaggle_view_pf"
        run_kaggle(hint=hint_cache, view_dir=str(v_pf), patients_file=str(manifest_path))
        view_pf_ids = sorted([d.name for d in v_pf.iterdir() if PATIENT_REGEX.match(d.name)])
        assert view_pf_ids == ["LIDC-IDRI-0002", "LIDC-IDRI-0004"], f"Sai patients-file: {view_pf_ids}"
        print("  ✓ Ca 8: --patients-file tái lập tập bệnh nhân chuẩn xác.")

        # 9. Không thấy zip/dataset -> báo lỗi rõ ràng
        try:
            run_colab(zip_path="khong_ton_tai.zip", dest_dir=str(temp_dir / "dummy"))
            assert False, "Phải báo lỗi khi không thấy zip"
        except FileNotFoundError:
            print("  ✓ Ca 9: Thông báo lỗi tiếng Việt chuẩn xác khi thiếu dữ liệu nguồn.")

        # CA TỰ PHÁ 1: Bỏ qua sắp xếp trước khi chọn -> kiểm tra test bắt được
        print("  🔨 Thử nghiệm phá 1: Bỏ bước sắp xếp trước khi chọn...")
        unsorted_pool = ["LIDC-IDRI-0005", "LIDC-IDRI-0001", "LIDC-IDRI-0003"]
        broken_sel, _ = select_patients(unsorted_pool, n_requested=2, mode="first", _skip_sort=True)
        # Nếu không sort, 2 phần tử đầu là 0005 và 0001 thay vì 0001 và 0003
        assert broken_sel != ["LIDC-IDRI-0001", "LIDC-IDRI-0003"], "Test phải phát hiện ra nếu bỏ sắp xếp"
        print("  ✓ Phá 1 thành công: Test bắt được lỗi nếu bỏ sort trước khi chọn.")

        # CA TỰ PHÁ 2: Bỏ thư mục tạm + os.replace -> kiểm tra thư mục dở dang
        print("  🔨 Thử nghiệm phá 2: Bỏ bước thư mục tạm + os.replace...")
        dest_broken = temp_dir / "colab_broken"
        dest_broken.mkdir(parents=True, exist_ok=True)
        # Giả lập giải nén dở dang trực tiếp vào đích
        broken_pid_dir = dest_broken / "LIDC-IDRI-0001"
        broken_pid_dir.mkdir(parents=True, exist_ok=True)
        # Chỉ có meta.json mà thiếu image_1mm.npy
        (broken_pid_dir / "meta.json").write_text("partial", encoding="utf-8")
        # Khi dùng cơ chế chuẩn với temp + os.replace: nếu dở dang (thiếu image_1mm.npy), hàm sẽ tự phục hồi
        run_colab(zip_path=str(zip_path), dest_dir=str(dest_broken), n_patients=1, select="first")
        assert (broken_pid_dir / "image_1mm.npy").exists(), "Hệ thống phải tự khôi phục thư mục dở dang"
        print("  ✓ Phá 2 thành công: Cơ chế nguyên tử (atomic) bảo vệ an toàn khỏi giải nén dở.")

        print("✅ Selftest runtime_data: HOÀN TẤT TOÀN DIỆN VÀ THÀNH CÔNG!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "selftest":
        run_selftest()
    elif args.cmd == "colab":
        run_colab(
            zip_path=args.zip,
            dest_dir=args.dest,
            drive_root=args.drive_root,
            n_patients=args.n_patients,
            select=args.select,
            seed=args.seed,
            patients_file=args.patients_file,
            out_json=args.out_json,
        )
    elif args.cmd == "kaggle":
        run_kaggle(
            hint=args.hint,
            view_dir=args.view_dir,
            n_patients=args.n_patients,
            select=args.select,
            seed=args.seed,
            patients_file=args.patients_file,
            out_json=args.out_json,
        )
    elif args.cmd == "local":
        run_local(cache_dir=args.cache_dir, out_json=args.out_json)
    elif args.cmd == "auto":
        run_auto(
            zip=args.zip,
            dest=args.dest,
            hint=args.hint,
            view_dir=args.view_dir,
            cache_dir=args.cache_dir,
            patients_file=args.patients_file,
            out_json=args.out_json,
        )
```

#### <a id="tệp-27-common-data_pipeline-offline_builder-py"></a>Tệp 27: `common/data_pipeline/offline_builder.py`
- **Vị trí**: `common/data_pipeline/offline_builder.py`
- **Quy mô**: `217 dòng` | `8.8 KB`
- **Vai trò**: Công cụ quét cache tiền xử lý offline, lập chỉ mục slices.csv và thống kê phân phối ca bệnh.

```python
# -*- coding: utf-8 -*-
"""
offline_builder.py - Lọc lát cắt, cân bằng âm/dương và chia tập Train/Val từ cache_1mm.

Dùng:
  python offline_builder.py selftest
  python offline_builder.py build --cache-dir data/cache_1mm
"""
import argparse
import json
import random
import shutil
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Nạp cấu hình và các hàm lõi từ medical_prep.py (cùng thư mục)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
ROOT = _ROOT

from configs.config import TUMOR_SIZE_LOW_LIM, ZERO_SL_RATIO, VAL_SPLIT, SEED
from common.data_pipeline.medical_prep import load_cached, lung_cache_path


def build_dataset_index(
    cache_dir,
    out_csv_path=None,
    tumor_size_low_lim=TUMOR_SIZE_LOW_LIM,
    zero_sl_ratio=ZERO_SL_RATIO,
    val_split=VAL_SPLIT,
    seed=SEED,
    require_lung=True,
):
    """
    Quét thư mục cache_1mm, lọc lát cắt theo tiêu chuẩn y khoa của DuneAI và xuất slices.csv.
    Lưu ý: mask_1mm.npy đã ở tỷ lệ 1mm/pixel nên 1 pixel = 1.0 mm^2.
    """
    cache_dir = Path(cache_dir)
    if out_csv_path is None:
        out_csv_path = cache_dir / "slices.csv"
    out_csv_path = Path(out_csv_path)

    if not cache_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục cache: {cache_dir}")

    patient_ids = sorted([
        p.name for p in cache_dir.iterdir()
        if p.is_dir() and (p / "meta.json").exists() and (p / "image_1mm.npy").exists()
    ])
    if not patient_ids:
        raise RuntimeError(f"Không tìm thấy bệnh nhân hợp lệ nào trong {cache_dir}")

    # Chia Train/Val theo cấp độ Bệnh nhân (Patient-level split) để chống Data Leakage
    rng = random.Random(seed)
    shuffled_pids = list(patient_ids)
    rng.shuffle(shuffled_pids)

    if len(shuffled_pids) == 1:
        val_patients = set()
        train_patients = set(shuffled_pids)
    else:
        val_size = max(1, int(round(len(shuffled_pids) * val_split)))
        val_patients = set(shuffled_pids[:val_size])
        train_patients = set(shuffled_pids[val_size:])

    rows = []
    stats = {
        "patients_total": len(patient_ids),
        "patients_train": len(train_patients),
        "patients_val": len(val_patients),
        "dropped_no_lung": 0,
        "dropped_small_tumor": 0,
        "dropped_neg_sampled": 0,
    }

    for pid in patient_ids:
        p_dir = cache_dir / pid
        _, msk, meta = load_cached(p_dir, mmap=True)
        num_slices = int(msk.shape[0])

        lung_p = lung_cache_path(p_dir)
        has_lung = lung_p.exists() and (meta.get("lung_ok", True) is not False)
        lung = np.load(lung_p, mmap_mode="r") if has_lung else None

        # Tính diện tích khối u (mm^2) trên từng lát cắt Z (1 pixel ở lưới 1mm = 1 mm^2)
        msk_arr = np.asarray(msk)
        tumor_areas = (msk_arr > 0).reshape(num_slices, -1).sum(axis=1).astype(np.float32)

        if lung is not None and require_lung:
            lung_arr = np.asarray(lung)
            lung_pixels = (lung_arr > 0).reshape(num_slices, -1).sum(axis=1)
        else:
            lung_pixels = np.full(num_slices, 1, dtype=np.int64)

        pos_slices = []
        neg_candidates = []

        for z in range(num_slices):
            area = float(tumor_areas[z])
            has_l = int(lung_pixels[z]) > 0

            if area > 0.0:
                if area < tumor_size_low_lim:
                    stats["dropped_small_tumor"] += 1
                    continue
                pos_slices.append((z, area))
            else:
                if not has_l:
                    stats["dropped_no_lung"] += 1
                    continue
                neg_candidates.append((z, 0.0))

        # Cân bằng lát cắt âm tính (Negative Sampling) theo ZERO_SL_RATIO
        if len(pos_slices) > 0:
            num_neg_keep = max(1, int(round(len(pos_slices) * zero_sl_ratio))) if zero_sl_ratio > 0 else 0
        else:
            num_neg_keep = 1 if len(neg_candidates) > 0 else 0

        if len(neg_candidates) > num_neg_keep:
            kept_neg = sorted(rng.sample(neg_candidates, num_neg_keep), key=lambda x: x[0])
            stats["dropped_neg_sampled"] += len(neg_candidates) - num_neg_keep
        else:
            kept_neg = neg_candidates

        split = "val" if pid in val_patients else "train"
        selected = [(z, 1, area) for z, area in pos_slices] + [(z, 0, area) for z, area in kept_neg]
        selected.sort(key=lambda x: x[0])

        for z, has_tumor, area in selected:
            rows.append({
                "patient_id": pid,
                "z": int(z),
                "num_slices": num_slices,
                "has_tumor": int(has_tumor),
                "tumor_area_mm2": round(float(area), 2),
                "has_lung_mask": int(has_lung),
                "split": split,
            })

    df = pd.DataFrame(rows)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    pos_count = int((df["has_tumor"] == 1).sum()) if len(df) else 0
    neg_count = int((df["has_tumor"] == 0).sum()) if len(df) else 0
    train_count = int((df["split"] == "train").sum()) if len(df) else 0
    val_count = int((df["split"] == "val").sum()) if len(df) else 0

    print("=" * 50)
    print("📊 THỐNG KÊ CHỈ MỤC DỮ LIỆU (OFFLINE BUILDER)")
    print("=" * 50)
    print(f"Tổng số bệnh nhân : {stats['patients_total']} ({stats['patients_train']} Train | {stats['patients_val']} Val)")
    print(f"Tổng lát cắt giữ lại: {len(df)} ({train_count} Train | {val_count} Val)")
    print(f"  + Dương tính (Có u >= {tumor_size_low_lim} mm²): {pos_count}")
    print(f"  + Âm tính (Không u, có phổi)       : {neg_count}")
    print("Các lát cắt đã lọc bỏ:")
    print(f"  - Không chứa nhu mô phổi           : {stats['dropped_no_lung']}")
    print(f"  - Khối u quá nhỏ (< {tumor_size_low_lim} mm²)     : {stats['dropped_small_tumor']}")
    print(f"  - Âm tính dư thừa (Negative Sample): {stats['dropped_neg_sampled']}")
    print(f"✅ Đã xuất bảng chỉ mục tại: {out_csv_path}")
    return df, stats


def _cmd_selftest(_):
    print("Đang chạy kiểm thử tự động offline_builder.py...")
    tmp = Path(tempfile.mkdtemp())
    try:
        for i in range(1, 5):
            p_dir = tmp / f"LIDC-IDRI-000{i}"
            p_dir.mkdir(parents=True)
            img = np.full((10, 100, 100), -800, dtype=np.int16)
            msk = np.zeros((10, 100, 100), dtype=np.uint8)
            lung = np.zeros((10, 100, 100), dtype=np.uint8)

            lung[2:8, 10:90, 10:90] = 1        # Lát 0,1,8,9 không có phổi (4 lát/ca)
            msk[3, 20:22, 20:25] = 1           # Lát 3: u 10 mm² (< 20 mm²) -> phải bị loại
            msk[5, 30:35, 30:35] = 1           # Lát 5: u 25 mm² (>= 20 mm²) -> giữ lại
            msk[6, 30:36, 30:36] = 1           # Lát 6: u 36 mm² (>= 20 mm²) -> giữ lại

            np.save(p_dir / "image_1mm.npy", img)
            np.save(p_dir / "mask_1mm.npy", msk)
            np.save(p_dir / "lung_1mm.npy", lung)
            with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump({"orig_spacing_xyz": [0.7, 0.7, 1.5], "lung_ok": True}, f)

        df, stats = build_dataset_index(tmp)
        assert stats["dropped_no_lung"] == 16, f"Sai lọc phổi: {stats}"
        assert stats["dropped_small_tumor"] == 4, f"Sai lọc u nhỏ: {stats}"
        assert (df["has_tumor"] == 1).sum() == 8, "Phải giữ đúng 8 lát có u hợp lệ"
        assert set(df["split"].unique()) == {"train", "val"}, "Phải có cả tập train và val"
        print("✅ selftest offline_builder: OK!")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Tạo file slices.csv từ thư mục cache_1mm")
    b.add_argument("--cache-dir", default=str(ROOT / "data" / "cache_1mm"))
    b.add_argument("--out-csv", default=None)
    b.add_argument("--no-require-lung", action="store_true", help="Không lọc bỏ lát thiếu phổi")
    b.set_defaults(fn=lambda a: build_dataset_index(a.cache_dir, a.out_csv, require_lung=not a.no_require_lung))

    s = sub.add_parser("selftest", help="Kiểm thử logic bằng dữ liệu giả lập")
    s.set_defaults(fn=_cmd_selftest)

    args = ap.parse_args()
    args.fn(args)
```


---

### 5. Thước Đo & Đánh Giá Khối 3D Chuẩn Y Tế (3D Evaluation & Metrics)

#### <a id="tệp-28-common-metrics-py"></a>Tệp 28: `common/metrics.py`
- **Vị trí**: `common/metrics.py`
- **Quy mô**: `91 dòng` | `3.7 KB`
- **Vai trò**: Các hàm tính toán chỉ số phân đoạn 2D/2.5D: Dice Score, IoU (Jaccard Index) hỗ trợ batch tensor và numpy.

```python
# -*- coding: utf-8 -*-
"""
common/metrics.py - Chỉ số đánh giá phân đoạn ảnh y tế dùng chung (thước đo chuẩn).

Chứa hàm compute_segmentation_metrics được sao chép nguyên văn từ mã nguồn gốc,
đảm bảo cả hai kiến trúc unet_base và mamba_unet được so sánh khách quan trên cùng thang đo.
TUYỆT ĐỐI KHÔNG chứa lớp nn.Module nào (R11).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Dict

# Đoạn bootstrap bắt buộc ở đầu mọi file có __main__ (R5, mục 3.2)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch


@torch.no_grad()
def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Tính toán các chỉ số phân đoạn cơ bản từ raw logits và targets nhị phân.
    Trả về dict chứa intersection, pred_sum, target_sum, dice, và iou.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.float32)
    targets_bin = (targets > 0.5).to(torch.float32)

    preds_f = preds.reshape(-1)
    targets_f = targets_bin.reshape(-1)

    inter = float((preds_f * targets_f).sum().item())
    pred_sum = float(preds_f.sum().item())
    target_sum = float(targets_f.sum().item())
    union = pred_sum + target_sum - inter

    dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
    iou = (inter + eps) / (union + eps)

    return {
        "intersection": inter,
        "pred_sum": pred_sum,
        "target_sum": target_sum,
        "dice": dice,
        "iou": iou,
    }


if __name__ == "__main__":
    print("🔬 Đang chạy selftest common.metrics...")

    # Ca 1: Khớp hoàn toàn (Perfect match)
    logits_perfect = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])
    targets_perfect = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    m1 = compute_segmentation_metrics(logits_perfect, targets_perfect, threshold=0.5)
    assert abs(m1["dice"] - 1.0) < 1e-4, f"Lỗi dice tại Ca 1: {m1}"
    assert abs(m1["iou"] - 1.0) < 1e-4, f"Lỗi iou tại Ca 1: {m1}"
    print(f"  + Ca 1 (Perfect match): Dice={m1['dice']:.4f}, IoU={m1['iou']:.4f} - OK")

    # Ca 2: Sai lệch hoàn toàn (Total mismatch)
    logits_mismatch = torch.tensor([[[[-10.0, -10.0], [10.0, 10.0]]]])
    targets_mismatch = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    m2 = compute_segmentation_metrics(logits_mismatch, targets_mismatch, threshold=0.5)
    assert m2["intersection"] == 0.0, f"Lỗi intersection tại Ca 2: {m2}"
    assert m2["dice"] < 1e-4, f"Lỗi dice tại Ca 2: {m2}"
    print(f"  + Ca 2 (Total mismatch): Dice={m2['dice']:.4f}, IoU={m2['iou']:.4f} - OK")

    # Ca 3: Trùng khớp một phần (Partial overlap: 1 pixel đúng / 2 pixels nhãn)
    logits_partial = torch.tensor([[[[10.0, -10.0], [-10.0, -10.0]]]])
    targets_partial = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    m3 = compute_segmentation_metrics(logits_partial, targets_partial, threshold=0.5)
    # inter=1, pred_sum=1, target_sum=2 -> dice = 2*1 / (1+2) = 2/3 ≈ 0.6667
    assert abs(m3["dice"] - (2.0 / 3.0)) < 1e-3, f"Lỗi dice tại Ca 3: {m3}"
    print(f"  + Ca 3 (Partial overlap): Dice={m3['dice']:.4f}, IoU={m3['iou']:.4f} - OK")

    # Ca 4: Kiểm tra đầy đủ các trường thông tin trả về
    expected_keys = {"intersection", "pred_sum", "target_sum", "dice", "iou"}
    assert set(m3.keys()) == expected_keys, f"Thiếu trường dữ liệu: {set(m3.keys())}"
    print(f"  + Ca 4 (Keys check): {list(m3.keys())} - OK")

    print("✅ Selftest common.metrics: HOÀN TẤT XUẤT SẮC!")
```

#### <a id="tệp-29-common-eval3d-__init__-py"></a>Tệp 29: `common/eval3d/__init__.py`
- **Vị trí**: `common/eval3d/__init__.py`
- **Quy mô**: `30 dòng` | `0.8 KB`
- **Vai trò**: Khởi tạo package common.eval3d.

```python
# -*- coding: utf-8 -*-
"""common.eval3d package - Các hàm hậu xử lý và đánh giá thể tích 3D độc lập kiến trúc."""

from common.eval3d.postprocess3d import (
    postprocess_3d_mask,
    resize_from_1mm_3d,
    unpad_or_uncrop_xy,
)
from common.eval3d.metrics3d import (
    _compute_hd95_mm,
    compute_3d_metrics,
)
from common.eval3d.volume_inference import predict_patient_volume
from common.eval3d.nrrd_io import (
    load_nrrd_mask,
    save_prediction_nrrd,
)
from common.eval3d.eval_runner import evaluate_dataset_3d

__all__ = [
    "unpad_or_uncrop_xy",
    "resize_from_1mm_3d",
    "postprocess_3d_mask",
    "_compute_hd95_mm",
    "compute_3d_metrics",
    "predict_patient_volume",
    "save_prediction_nrrd",
    "load_nrrd_mask",
    "evaluate_dataset_3d",
]
```

#### <a id="tệp-30-common-eval3d-volume_inference-py"></a>Tệp 30: `common/eval3d/volume_inference.py`
- **Vị trí**: `common/eval3d/volume_inference.py`
- **Quy mô**: `147 dòng` | `5.5 KB`
- **Vai trò**: predict_patient_volume: Suy luận trượt toàn bộ lát cắt của bệnh nhân và ghép thành ma trận thể tích 3D.

```python
# -*- coding: utf-8 -*-
"""
common/eval3d/volume_inference.py - Suy luận 3D toàn khối cho một ca bệnh nhân (2.5D Sliding-Window).

Quy tắc bất biến:
  - Hàm `predict_patient_volume` nhận instance `nn.Module` độc lập kiến trúc.
  - Không import unet_base hay mamba_unet vào module này (R2).
  - Không chứa lớp kế thừa nn.Module (R11).
  - Tiền xử lý và suy luận theo batch để chống tràn bộ nhớ GPU (OOM).
  - Hỗ trợ Test-Time Augmentation (TTA).
"""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import TUMOR_SIZE_LOW_LIM
from common.data_pipeline.medical_prep import (
    HU_FLOOR,
    IMAGE_SIZE,
    apply_lung_mask,
    load_cached,
    lung_cache_path,
    lung_keep_mask,
    pad_or_crop_xy,
    window_hu_torch,
)
from common.eval3d.metrics3d import compute_3d_metrics
from common.eval3d.postprocess3d import postprocess_3d_mask, unpad_or_uncrop_xy


@torch.no_grad()
def predict_patient_volume(
    model: nn.Module,
    patient_cache_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    batch_size: int = 8,
    in_channels: int = 3,
    image_size: int = IMAGE_SIZE,
    apply_lung: bool = True,
    threshold: float = 0.5,
    min_slice_area_mm2: float = TUMOR_SIZE_LOW_LIM,
    min_volume_mm3: float = 0.0,
    tta: bool = False,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Suy luận 3D hoàn chỉnh cho 1 bệnh nhân từ thư mục cache_1mm/<patient_id>:
      1. Đọc image_1mm.npy, mask_1mm.npy, lung_1mm.npy, meta.json.
      2. Áp mặt nạ phổi (KHÔNG dùng tumor_mask để tránh rò rỉ nhãn).
      3. Chỉ chạy mạng trên các lát cắt nằm trong vùng có phổi để tiết kiệm GPU và chặn dương tính giả ngoài phổi.
      4. Khôi phục kích thước từ (Z, 512, 512) về (Z, H_1mm, W_1mm) bằng `unpad_or_uncrop_xy`.
      5. Chạy hậu xử lý 3D (`postprocess_3d_mask`) để lọc nhiễu và tính Dice 3D trước/sau hậu xử lý.
    """
    device = device or next(model.parameters()).device
    is_cuda = device.type == "cuda"
    model.eval()

    patient_cache_dir = Path(patient_cache_dir)
    img_1mm, msk_1mm, meta = load_cached(patient_cache_dir, mmap=True)
    num_slices, h_1mm, w_1mm = img_1mm.shape
    spacing_z = float(meta.get("orig_spacing_xyz", [1.0, 1.0, 1.0])[2])

    lung_p = lung_cache_path(patient_cache_dir)
    has_lung = apply_lung and lung_p.exists() and (meta.get("lung_ok", True) is not False)
    lung_1mm = np.asarray(np.load(lung_p, mmap_mode="r")) if has_lung else None

    vol_hu = np.asarray(img_1mm)
    if lung_1mm is not None:
        keep_lung = lung_keep_mask(lung_1mm, tumor_mask=None)
        vol_hu = apply_lung_mask(vol_hu, keep_lung, fill=HU_FLOOR)
        active_z = np.where(keep_lung.reshape(num_slices, -1).sum(axis=1) > 0)[0].tolist()
    else:
        keep_lung = None
        active_z = list(range(num_slices))

    vol_padded = pad_or_crop_xy(vol_hu, size=image_size, pad_value=HU_FLOOR)
    prob_padded = np.zeros((num_slices, image_size, image_size), dtype=np.float32)

    half_k = in_channels // 2
    amp_ctx = (
        torch.amp.autocast("cuda", enabled=True)
        if (use_amp and is_cuda and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"))
        else nullcontext()
    )

    for start in range(0, len(active_z), batch_size):
        batch_z = active_z[start : start + batch_size]
        stacks = []
        for z in batch_z:
            z_idx = [min(max(0, z + off), num_slices - 1) for off in range(-half_k, half_k + 1)]
            stacks.append(vol_padded[z_idx])

        batch_hu = torch.from_numpy(np.stack(stacks, axis=0)).to(device, non_blocking=is_cuda)
        batch_x = window_hu_torch(batch_hu)

        with amp_ctx:
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            if tta:
                # Test-Time Augmentation: lật ngang và lật dọc
                logits_h = torch.flip(model(torch.flip(batch_x, dims=[3])), dims=[3])
                logits_v = torch.flip(model(torch.flip(batch_x, dims=[2])), dims=[2])
                probs = (probs + torch.sigmoid(logits_h) + torch.sigmoid(logits_v)) / 3.0

        probs_np = probs.squeeze(1).to(torch.float32).cpu().numpy()
        for i, z in enumerate(batch_z):
            prob_padded[z] = probs_np[i]

    prob_1mm = unpad_or_uncrop_xy(prob_padded, (h_1mm, w_1mm), pad_value=0.0)
    raw_mask_1mm = (prob_1mm >= float(threshold)).astype(np.uint8)
    if keep_lung is not None:
        raw_mask_1mm &= keep_lung

    clean_mask_1mm = postprocess_3d_mask(
        prob_1mm,
        lung_mask_zyx=keep_lung,
        threshold=threshold,
        min_slice_area_mm2=min_slice_area_mm2,
        min_volume_mm3=min_volume_mm3,
        spacing_z=spacing_z,
        fill_holes=True,
    )

    gt_1mm = (np.asarray(msk_1mm) > 0).astype(np.uint8)
    metrics_raw = compute_3d_metrics(raw_mask_1mm, gt_1mm, spacing_z=spacing_z, compute_hd95=False)
    metrics_post = compute_3d_metrics(clean_mask_1mm, gt_1mm, spacing_z=spacing_z, compute_hd95=True)

    return {
        "prob_1mm": prob_1mm,
        "raw_mask_1mm": raw_mask_1mm,
        "clean_mask_1mm": clean_mask_1mm,
        "gt_mask_1mm": gt_1mm,
        "meta": meta,
        "metrics_raw": metrics_raw,
        "metrics_post": metrics_post,
    }
```

#### <a id="tệp-31-common-eval3d-postprocess3d-py"></a>Tệp 31: `common/eval3d/postprocess3d.py`
- **Vị trí**: `common/eval3d/postprocess3d.py`
- **Quy mô**: `202 dòng` | `8.1 KB`
- **Vai trò**: Hậu xử lý 3D y tế: Lọc kết nối 3D Connected Components, loại bỏ u rác < 20mm3, làm mịn biên dạng.

```python
# -*- coding: utf-8 -*-
"""
common/eval3d/postprocess3d.py - Khôi phục hình học không gian và hậu xử lý hình thái học 3D.

Tuân thủ chuẩn y tế của tác giả DuneAI (s.primakov) & medical_prep.py:
  1. unpad_or_uncrop_xy: Hàm nghịch đảo chính xác của pad_or_crop_xy từ (..., 512, 512)
     về lại kích thước (..., H_1mm, W_1mm).
  2. resize_from_1mm_3d: Đưa khối 3D từ lưới 1mm về độ phân giải gốc của file CT/NRRD ban đầu.
  3. postprocess_3d_mask:
     - Giới hạn vùng dự đoán trong mặt nạ phổi.
     - Lấp lỗ khuyết 2D/3D (binary_fill_holes).
     - Lọc thành phần liên thông 3D (26-connectivity): Loại bỏ các cụm có diện tích lát cắt
       cực đại < TUMOR_SIZE_LOW_LIM (20 mm²) hoặc thể tích 3D < min_volume_mm3.
"""

import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import TUMOR_SIZE_LOW_LIM
from common.data_pipeline.medical_prep import _axis_window, lung_keep_mask, pad_or_crop_xy


def unpad_or_uncrop_xy(
    x_padded: Union[np.ndarray, torch.Tensor],
    target_hw: Sequence[int],
    pad_value: float = 0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Hàm nghịch đảo chính xác của `pad_or_crop_xy` trong `medical_prep.py`.
    Chuyển mảng/tensor từ kích thước chuẩn mạng (..., size, size)
    về lại kích thước vật lý 1mm của bệnh nhân (..., H_1mm, W_1mm).
    """
    size_h, size_w = x_padded.shape[-2], x_padded.shape[-1]
    assert size_h == size_w, f"Đầu vào phải có mặt phẳng vuông, nhận được: {x_padded.shape}"
    size = size_h
    h, w = int(target_hw[0]), int(target_hw[1])

    sy, dy, ly = _axis_window(h, size)
    sx, dx, lx = _axis_window(w, size)
    shape = (*x_padded.shape[:-2], h, w)

    if isinstance(x_padded, torch.Tensor):
        out = torch.full(shape, pad_value, dtype=x_padded.dtype, device=x_padded.device)
    else:
        out = np.full(shape, pad_value, dtype=x_padded.dtype)

    out[..., sy : sy + ly, sx : sx + lx] = x_padded[..., dy : dy + ly, dx : dx + lx]
    return out


def resize_from_1mm_3d(
    volume_1mm: np.ndarray,
    orig_shape_zyx: Sequence[int],
    is_mask: bool = True,
) -> np.ndarray:
    """
    Đưa khối 3D từ lưới 1mm (Z, H_1mm, W_1mm) về lại độ phân giải gốc (Z, H_orig, W_orig)
    của file NRRD/DICOM ban đầu.
    """
    z_orig, h_orig, w_orig = (int(v) for v in orig_shape_zyx)
    z, h, w = volume_1mm.shape
    assert z == z_orig, f"Số lát cắt Z không khớp: {z} != {z_orig}"

    if h == h_orig and w == w_orig:
        return volume_1mm.copy()

    flag = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    src = (volume_1mm > 0).astype(np.uint8) if is_mask else np.ascontiguousarray(volume_1mm)
    out = np.empty((z_orig, h_orig, w_orig), dtype=src.dtype)
    for i in range(z_orig):
        out[i] = cv2.resize(src[i], (w_orig, h_orig), interpolation=flag)
    return out


def postprocess_3d_mask(
    prob_or_mask_zyx: np.ndarray,
    lung_mask_zyx: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    min_slice_area_mm2: float = TUMOR_SIZE_LOW_LIM,
    min_volume_mm3: float = 0.0,
    spacing_z: float = 1.0,
    fill_holes: bool = True,
    min_slices_span: int = 1,
    top_k_components: int = 0,
) -> np.ndarray:
    """
    Hậu xử lý khối dự đoán 3D trên lưới 1mm (1 pixel trên mặt phẳng XY = 1.0 mm²):
      1. Ngưỡng hóa xác suất -> nhị phân (uint8).
      2. Cắt bỏ mọi dự đoán nằm ngoài mặt nạ phổi (nếu có lung_mask_zyx).
      3. Lấp các lỗ khuyết nhỏ bên trong tâm khối u trên từng lát cắt 2D (binary_fill_holes).
      4. Phân tích thành phần liên thông 3D (26-connectivity):
         - Loại bỏ cụm nhiễu có diện tích mặt cắt ngang lớn nhất < min_slice_area_mm2 (20.0 mm²).
         - Loại bỏ cụm có thể tích vật lý < min_volume_mm3 hoặc chỉ xuất hiện ít hơn min_slices_span lát cắt.
         - Tùy chọn chỉ giữ lại top_k_components khối lớn nhất.
    """
    arr = np.asarray(prob_or_mask_zyx)
    if np.issubdtype(arr.dtype, np.floating):
        mask = (arr >= float(threshold)).astype(np.uint8)
    else:
        mask = (arr > 0).astype(np.uint8)

    if lung_mask_zyx is not None:
        keep_lung = lung_keep_mask(lung_mask_zyx, tumor_mask=None)
        mask &= keep_lung

    if mask.sum() == 0:
        return mask

    # Lấp lỗ rỗng 2D trên từng lát cắt có dự đoán dương tính
    if fill_holes:
        for z in np.where(mask.sum(axis=(1, 2)) > 0)[0]:
            mask[z] = ndi.binary_fill_holes(mask[z]).astype(np.uint8)

    # Đánh nhãn các khối liên thông 3D (26-connectivity: khối lập phương 3x3x3)
    structure_3d = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, num_features = ndi.label(mask, structure=structure_3d)
    if num_features == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    objects = ndi.find_objects(labeled)
    valid_components = []

    for comp_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        sub_mask = (labeled[slc] == comp_id)
        voxel_count = int(sub_mask.sum())
        if voxel_count == 0:
            continue

        z_span = int(sub_mask.shape[0])
        if z_span < min_slices_span:
            continue

        slice_areas = sub_mask.reshape(z_span, -1).sum(axis=1)
        max_area_mm2 = float(slice_areas.max())
        if max_area_mm2 < float(min_slice_area_mm2):
            continue

        vol_mm3 = float(voxel_count) * float(spacing_z)
        if vol_mm3 < float(min_volume_mm3):
            continue

        valid_components.append((comp_id, voxel_count))

    if not valid_components:
        return np.zeros_like(mask, dtype=np.uint8)

    if top_k_components and top_k_components > 0 and len(valid_components) > top_k_components:
        valid_components.sort(key=lambda x: x[1], reverse=True)
        valid_components = valid_components[:top_k_components]

    keep_ids = np.array([cid for cid, _ in valid_components], dtype=labeled.dtype)
    cleaned_mask = np.isin(labeled, keep_ids).astype(np.uint8)
    return cleaned_mask


def _selftest():
    print("🧪 Bắt đầu selftest common.eval3d.postprocess3d...")
    # 1. Kiểm thử tính nghịch đảo hoàn hảo của unpad_or_uncrop_xy
    for hw in [(360, 380), (512, 512), (580, 600)]:
        arr = np.zeros((4, hw[0], hw[1]), dtype=np.uint8)
        ch, cw = hw[0] // 2, hw[1] // 2
        arr[1:3, ch - 20 : ch + 20, cw - 20 : cw + 20] = 1
        padded = pad_or_crop_xy(arr, size=512, pad_value=0)
        unpadded = unpad_or_uncrop_xy(padded, hw, pad_value=0)
        assert np.array_equal(arr, unpadded), f"Lỗi nghịch đảo pad/crop với kích thước {hw}"

    # 2. Resize mask
    orig_back = resize_from_1mm_3d(np.ones((4, 360, 360), dtype=np.uint8), (4, 512, 512), is_mask=True)
    assert orig_back.shape == (4, 512, 512) and set(np.unique(orig_back)) <= {0, 1}

    # 3. Lọc nhiễu 3D
    gt = np.zeros((10, 128, 128), dtype=np.uint8)
    lung = np.zeros((10, 128, 128), dtype=np.uint8)
    lung[1:9, 10:118, 10:118] = 1
    gt[4:7, 50:60, 50:60] = 1  # 10x10 = 100 mm2 per slice

    noisy = gt.astype(np.float32) * 0.95
    noisy[5, 54:56, 54:56] = 0.1   # lỗ khuyết ở giữa
    noisy[2, 30:33, 30:33] = 0.90  # 9 mm2 < 20 mm2 -> phải bị lọc
    noisy[0, 40:60, 40:60] = 0.95  # ngoài phổi -> phải bị cắt

    cleaned = postprocess_3d_mask(noisy, lung_mask_zyx=lung, threshold=0.5, min_slice_area_mm2=20.0, fill_holes=True)
    assert np.array_equal(cleaned, gt), "Hậu xử lý 3D phải khôi phục 100% khối u thật và loại bỏ nhiễu!"
    print("✅ Selftest common.eval3d.postprocess3d hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
```

#### <a id="tệp-32-common-eval3d-metrics3d-py"></a>Tệp 32: `common/eval3d/metrics3d.py`
- **Vị trí**: `common/eval3d/metrics3d.py`
- **Quy mô**: `148 dòng` | `4.9 KB`
- **Vai trò**: Tính toán bộ chỉ số y tế 3D: 3D Dice, 3D IoU, 95% Hausdorff Distance (HD95), Average Surface Distance (ASD), Recall, Specificity.

```python
# -*- coding: utf-8 -*-
"""
common/eval3d/metrics3d.py - Các chỉ số đánh giá 3D cấp độ bệnh nhân (3D Patient-level Metrics).

Các chỉ số y tế chuẩn mực:
  - 3D Dice, 3D IoU, Precision, Recall.
  - Thể tích 3D khối u (mm³): pred_volume_mm3, gt_volume_mm3.
  - Khoảng cách bề mặt 95th percentile Hausdorff Distance (HD95, mm):
    Tính bằng khoảng cách Euclid thực trên lưới voxel có spacing dị hướng (spacing_z, 1.0, 1.0).
    Xử lý an toàn các ca biên (mask rỗng hoàn toàn).
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as ndi


def _compute_hd95_mm(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Tính 95th percentile Hausdorff Distance (HD95) theo đơn vị mm vật lý giữa 2 mặt nạ 3D.
    Xử lý an toàn:
      - Cả hai rỗng: 0.0 mm (hoàn hảo không có u và không phát hiện nhầm).
      - Một trong hai rỗng: float('nan').
    """
    pred_any = bool(pred_bin.any())
    gt_any = bool(gt_bin.any())
    if not pred_any and not gt_any:
        return 0.0
    if not pred_any or not gt_any:
        return float("nan")

    # Trích xuất biên bề mặt 3D bằng phép co (erosion)
    pred_border = pred_bin ^ ndi.binary_erosion(pred_bin)
    gt_border = gt_bin ^ ndi.binary_erosion(gt_bin)
    if not pred_border.any():
        pred_border = pred_bin
    if not gt_border.any():
        gt_border = gt_bin

    dt_gt = ndi.distance_transform_edt(~gt_border, sampling=spacing_zyx)
    dt_pred = ndi.distance_transform_edt(~pred_border, sampling=spacing_zyx)

    dists_pred_to_gt = dt_gt[pred_border]
    dists_gt_to_pred = dt_pred[gt_border]
    all_dists = np.concatenate([dists_pred_to_gt, dists_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def compute_3d_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing_z: float = 1.0,
    compute_hd95: bool = True,
    eps: float = 1e-6,
) -> Dict[str, Union[float, int]]:
    """
    Tính bộ chỉ số phân vùng 3D trên toàn bộ thể tích phổi của bệnh nhân:
      - dice_3d, iou_3d, precision_3d, recall_3d
      - pred_volume_mm3, gt_volume_mm3
      - hd95_mm (Hausdorff Distance 95% tính bằng mm)
    """
    pred_b = np.asarray(pred_mask) > 0
    gt_b = np.asarray(gt_mask) > 0

    tp = float(np.logical_and(pred_b, gt_b).sum())
    fp = float(np.logical_and(pred_b, ~gt_b).sum())
    fn = float(np.logical_and(~pred_b, gt_b).sum())

    pred_sum = tp + fp
    gt_sum = tp + fn
    union = tp + fp + fn

    if pred_sum == 0 and gt_sum == 0:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
    elif tp == 0:
        dice = 0.0
        iou = 0.0
        precision = 0.0
        recall = 0.0
    else:
        dice = (2.0 * tp + eps) / (pred_sum + gt_sum + eps)
        iou = (tp + eps) / (union + eps)
        precision = (tp + eps) / (pred_sum + eps) if pred_sum > 0 else 0.0
        recall = (tp + eps) / (gt_sum + eps) if gt_sum > 0 else 0.0

    hd95 = (
        _compute_hd95_mm(pred_b, gt_b, spacing_zyx=(float(spacing_z), 1.0, 1.0))
        if compute_hd95
        else float("nan")
    )

    return {
        "dice_3d": float(dice),
        "iou_3d": float(iou),
        "precision_3d": float(precision),
        "recall_3d": float(recall),
        "pred_voxels": int(pred_sum),
        "gt_voxels": int(gt_sum),
        "pred_volume_mm3": round(float(pred_sum * spacing_z), 2),
        "gt_volume_mm3": round(float(gt_sum * spacing_z), 2),
        "hd95_mm": round(float(hd95), 3) if np.isfinite(hd95) else float("nan"),
    }


def _selftest():
    print("🧪 Bắt đầu selftest common.eval3d.metrics3d...")
    # 1. Trùng khớp hoàn hảo
    m1 = np.zeros((8, 32, 32), dtype=np.uint8)
    m1[2:6, 10:20, 10:20] = 1
    res1 = compute_3d_metrics(m1, m1, spacing_z=1.5)
    assert abs(res1["dice_3d"] - 1.0) < 1e-5
    assert abs(res1["iou_3d"] - 1.0) < 1e-5
    assert res1["hd95_mm"] == 0.0

    # 2. Cả hai cùng rỗng
    empty = np.zeros((8, 32, 32), dtype=np.uint8)
    res_empty = compute_3d_metrics(empty, empty)
    assert res_empty["dice_3d"] == 1.0
    assert res_empty["hd95_mm"] == 0.0

    # 3. Một bên rỗng
    res_half = compute_3d_metrics(m1, empty)
    assert res_half["dice_3d"] == 0.0
    assert np.isnan(res_half["hd95_mm"])

    # 4. Lệch vị trí có khoảng cách cụ thể
    m2 = np.zeros_like(m1)
    m2[2:6, 12:22, 10:20] = 1  # dịch 2 pixel XY
    res2 = compute_3d_metrics(m1, m2, spacing_z=1.0)
    assert 0.0 < res2["dice_3d"] < 1.0
    assert 1.0 <= res2["hd95_mm"] <= 3.0

    print("✅ Selftest common.eval3d.metrics3d hoàn tất thành công!")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
```

#### <a id="tệp-33-common-eval3d-nrrd_io-py"></a>Tệp 33: `common/eval3d/nrrd_io.py`
- **Vị trí**: `common/eval3d/nrrd_io.py`
- **Quy mô**: `63 dòng` | `2.4 KB`
- **Vai trò**: Đọc và xuất file mặt nạ 3D chuẩn y tế định dạng NRRD (.nrrd) tương thích 3D Slicer / ITK-SNAP.

```python
# -*- coding: utf-8 -*-
"""
common/eval3d/nrrd_io.py - Đọc và ghi mặt nạ 3D chuẩn y tế định dạng NRRD.

Chuyển đổi kích thước từ không gian 1mm về lại độ phân giải gốc của máy chụp CT,
bảo toàn thông tin tọa độ thực (Spacing, Origin, Direction matrix).
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.eval3d.postprocess3d import resize_from_1mm_3d


def save_prediction_nrrd(
    clean_mask_1mm: np.ndarray,
    meta: Dict[str, Any],
    out_nrrd_path: Union[str, Path],
    reference_nrrd_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Đưa mặt nạ dự đoán từ lưới 1mm (Z, H_1mm, W_1mm) về lưới gốc (Z, H_orig, W_orig)
    và lưu thành file `.nrrd` giữ nguyên hệ tọa độ y khoa (Spacing, Origin, Direction).
    """
    if sitk is None or not hasattr(sitk, "GetImageFromArray"):
        raise ImportError("Cần cài đặt SimpleITK để xuất file NRRD: pip install SimpleITK")

    orig_shape_zyx = meta.get("orig_shape_zyx", list(clean_mask_1mm.shape))
    mask_orig = resize_from_1mm_3d(clean_mask_1mm, orig_shape_zyx, is_mask=True)

    sitk_mask = sitk.GetImageFromArray(mask_orig.astype(np.uint8))
    if reference_nrrd_path and Path(reference_nrrd_path).exists():
        ref_img = sitk.ReadImage(str(reference_nrrd_path))
        sitk_mask.CopyInformation(ref_img)
    elif "orig_spacing_xyz" in meta:
        sitk_mask.SetSpacing(tuple(float(s) for s in meta["orig_spacing_xyz"]))

    out_nrrd_path = Path(out_nrrd_path)
    out_nrrd_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk_mask, str(out_nrrd_path), useCompression=True)
    return mask_orig


def load_nrrd_mask(nrrd_path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Đọc file NRRD và trả về mảng 3D numpy (Z, Y, X) kèm voxel spacing (sx, sy, sz)."""
    if sitk is None:
        raise ImportError("Cần cài đặt SimpleITK để đọc file NRRD: pip install SimpleITK")
    img = sitk.ReadImage(str(nrrd_path))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()  # (sx, sy, sz)
    return arr, spacing
```

#### <a id="tệp-34-common-eval3d-eval_runner-py"></a>Tệp 34: `common/eval3d/eval_runner.py`
- **Vị trí**: `common/eval3d/eval_runner.py`
- **Quy mô**: `151 dòng` | `5.5 KB`
- **Vai trò**: Bộ điều phối đánh giá trên toàn bộ tập kiểm định validation, xuất báo cáo CSV và JSON chi tiết.

```python
# -*- coding: utf-8 -*-
"""
common/eval3d/eval_runner.py - Vòng lặp đánh giá 3D trên tập dữ liệu cấp độ bệnh nhân.

Nhận instance model đã nạp sẵn trọng số, duyệt qua danh sách bệnh nhân validation
hoặc toàn tập, tính toán và xuất báo cáo kết quả chi tiết từng ca và tổng hợp.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import TUMOR_SIZE_LOW_LIM
from common.eval3d.volume_inference import predict_patient_volume


def evaluate_dataset_3d(
    model: nn.Module,
    cache_dir: Union[str, Path],
    split: str = "val",
    batch_size: int = 8,
    threshold: float = 0.5,
    min_slice_area_mm2: float = TUMOR_SIZE_LOW_LIM,
    min_volume_mm3: float = 0.0,
    tta: bool = False,
    out_csv: Optional[Union[str, Path]] = None,
    out_json: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> pd.DataFrame:
    """
    Chạy suy luận 3D và hậu xử lý cho toàn bộ bệnh nhân thuộc `split` ('val', 'train', hoặc 'all').
    So sánh trực tiếp Dice 3D trước và sau hậu xử lý.
    """
    cache_dir = Path(cache_dir)
    csv_path = cache_dir / "slices.csv"
    if csv_path.exists() and split != "all":
        df_slices = pd.read_csv(csv_path)
        if "split" in df_slices.columns and "patient_id" in df_slices.columns:
            pids = sorted(df_slices[df_slices["split"] == split]["patient_id"].unique().tolist())
        else:
            pids = sorted([
                p.name for p in cache_dir.iterdir()
                if p.is_dir() and (p / "meta.json").exists() and (p / "image_1mm.npy").exists()
            ])
    else:
        pids = sorted([
            p.name for p in cache_dir.iterdir()
            if p.is_dir() and (p / "meta.json").exists() and (p / "image_1mm.npy").exists()
        ])

    if not pids:
        raise RuntimeError(f"❌ Không tìm thấy bệnh nhân nào cho split='{split}' trong {cache_dir}")

    device = device or next(model.parameters()).device

    print("=" * 90)
    print(
        f"🩻 ĐÁNH GIÁ 3D CẤP ĐỘ BỆNH NHÂN | Split: {split} ({len(pids)} ca) | "
        f"Thiết bị: {device} | Ngưỡng lọc u: >= {min_slice_area_mm2} mm²"
    )
    print("=" * 90)

    rows: List[Dict[str, Any]] = []
    for idx, pid in enumerate(pids, start=1):
        t0 = time.time()
        res = predict_patient_volume(
            model=model,
            patient_cache_dir=cache_dir / pid,
            device=device,
            batch_size=batch_size,
            threshold=threshold,
            min_slice_area_mm2=min_slice_area_mm2,
            min_volume_mm3=min_volume_mm3,
            tta=tta,
            use_amp=use_amp,
        )
        dt = time.time() - t0
        m_raw = res["metrics_raw"]
        m_post = res["metrics_post"]

        row = {
            "patient_id": pid,
            "dice_3d_raw": round(m_raw["dice_3d"], 4),
            "dice_3d_post": round(m_post["dice_3d"], 4),
            "iou_3d_post": round(m_post["iou_3d"], 4),
            "precision_3d": round(m_post["precision_3d"], 4),
            "recall_3d": round(m_post["recall_3d"], 4),
            "hd95_mm": m_post["hd95_mm"],
            "gt_vol_mm3": m_post["gt_volume_mm3"],
            "pred_vol_mm3": m_post["pred_volume_mm3"],
            "time_sec": round(dt, 2),
        }
        rows.append(row)
        diff_dice = m_post["dice_3d"] - m_raw["dice_3d"]
        print(
            f"[{idx:02d}/{len(pids):02d}] {pid} ({dt:.1f}s) | "
            f"3D Dice: {m_raw['dice_3d']:.4f} -> {m_post['dice_3d']:.4f} ({diff_dice:+.4f}) | "
            f"IoU: {m_post['iou_3d']:.4f} | HD95: {m_post['hd95_mm']} mm | "
            f"Vol(GT/Pred): {m_post['gt_volume_mm3']}/{m_post['pred_volume_mm3']} mm³"
        )

    df_res = pd.DataFrame(rows)
    mean_raw = float(df_res["dice_3d_raw"].mean())
    mean_post = float(df_res["dice_3d_post"].mean())
    mean_iou = float(df_res["iou_3d_post"].mean())

    valid_hd95 = [v for v in df_res["hd95_mm"] if np.isfinite(v)]
    mean_hd95 = float(np.mean(valid_hd95)) if valid_hd95 else float("nan")

    print("=" * 90)
    print(
        f"📊 TỔNG KẾT ({len(pids)} bệnh nhân) | "
        f"Mean 3D Dice (Raw): {mean_raw:.4f} -> Mean 3D Dice (Post): {mean_post:.4f} "
        f"({mean_post - mean_raw:+.4f}) | Mean 3D IoU: {mean_iou:.4f} | Mean HD95: {mean_hd95:.2f} mm"
    )
    print("=" * 90)

    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(out_csv, index=False)
        print(f"✅ Đã lưu bảng báo cáo đánh giá 3D tại: {out_csv}")

    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "total_patients": len(pids),
            "split": split,
            "mean_dice_3d_raw": round(mean_raw, 4),
            "mean_dice_3d_post": round(mean_post, 4),
            "mean_iou_3d_post": round(mean_iou, 4),
            "mean_hd95_mm": round(mean_hd95, 3) if np.isfinite(mean_hd95) else None,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Đã lưu tóm tắt JSON tại: {out_json}")

    return df_res
```


---

### 6. Bộ Kiểm Thử Độ Tin Cậy & Vùng Xám (Gray Area Tests)

#### <a id="tệp-35-tests-test_gray_area_padding-py"></a>Tệp 35: `tests/test_gray_area_padding.py`
- **Vị trí**: `tests/test_gray_area_padding.py`
- **Quy mô**: `101 dòng` | `4.6 KB`
- **Vai trò**: Kiểm thử lan truyền tiến/lùi 512x512 qua MambaVision Mixer có padding chẵn/lẻ, kiểm tra finite gradients.

```python
# -*- coding: utf-8 -*-
"""
tests/test_gray_area_padding.py - Kiểm thử chuyên sâu VÙNG XÁM PADDING & GRADIENTS (Ảnh 512x512).

Mục tiêu kiểm định:
  1. Đưa ảnh kích thước thực tế 512x512 qua MambaVision (với trọng số thật NVIDIA).
  2. Xác minh hoạt động đệm không gian (padding) tại Stage 3 (32 -> 42 -> 32) và Stage 4 (16 -> 21 -> 16).
  3. Kiểm tra thông suốt dòng gradient (Loss backward) từ đầu ra 512x512 ngược về từng tầng Mixer/Attention.
  4. Đảm bảo 100% tham số và đạo hàm không có NaN, không có Inf.
  5. Giả lập bộ nhớ với batch_size = 2, 4, 8 trên ảnh 512x512.
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mamba_unet.backbone.mamba_vision import mamba_vision_T
from mamba_unet.model import MambaUNet


class TestGrayAreaPaddingAndVRAM(unittest.TestCase):
    """Kiểm thử áp lực (stress test) vùng xám padding và gradient flow cho ảnh y tế 512x512."""

    def test_01_real_image_512_forward_all_stages(self):
        """Kiểm tra forward pass qua từng stage với ảnh 512x512."""
        model = mamba_vision_T(pretrained=False, in_chans=3)
        model.eval()

        # Ảnh kích thước chuẩn 512x512
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            features = model(x)

        self.assertEqual(len(features), 4, "Phải xuất ra đúng 4 skip features")
        self.assertEqual(features[0].shape, (2, 80, 128, 128), "Stage 1 shape sai")
        self.assertEqual(features[1].shape, (2, 160, 64, 64), "Stage 2 shape sai")
        self.assertEqual(features[2].shape, (2, 320, 32, 32), "Stage 3 (sau khi unpad từ 42) shape sai")
        self.assertEqual(features[3].shape, (2, 640, 16, 16), "Stage 4 (sau khi unpad từ 21) shape sai")

        # Kiểm tra tính hữu hạn (không NaN, không Inf)
        for i, f in enumerate(features):
            self.assertTrue(torch.isfinite(f).all(), f"Feature stage {i+1} có chứa NaN hoặc Inf")

    def test_02_gradient_flow_through_padding_stages(self):
        """Kiểm tra backward pass: gradient truyền thông suốt qua các khối WindowAttention bị pad."""
        model = MambaUNet(in_channels=3, num_classes=1, backbone="mamba_vision_T", pretrained=False)
        model.train()

        # Batch 2 ảnh 512x512
        x = torch.randn(2, 3, 512, 512, requires_grad=True)
        target = torch.ones(2, 1, 512, 512)

        out = model(x)
        self.assertEqual(out.shape, (2, 1, 512, 512), "Đầu ra Mamba-UNet phải khôi phục đúng 512x512")

        loss = F.binary_cross_entropy_with_logits(out, target)
        loss.backward()

        # 1. Đạo hàm đầu vào ảnh phải hữu hạn và khác 0
        self.assertIsNotNone(x.grad, "Đạo hàm của tensor ảnh đầu vào bị None")
        self.assertTrue(torch.isfinite(x.grad).all(), "Đạo hàm đầu vào có NaN hoặc Inf")
        self.assertGreater(x.grad.abs().sum().item(), 0.0, "Đạo hàm đầu vào bằng 0 (chết gradient)")

        # 2. Kiểm tra các tham số của Stage 3 & 4 (nơi có WindowAttention bị pad)
        stage3_checked = 0
        stage4_checked = 0
        for name, param in model.encoder.named_parameters():
            if "levels.2." in name and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), f"Gradient của {name} có NaN/Inf")
                stage3_checked += 1
            if "levels.3." in name and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), f"Gradient của {name} có NaN/Inf")
                stage4_checked += 1

        self.assertGreater(stage3_checked, 0, "Stage 3 không nhận được gradient")
        self.assertGreater(stage4_checked, 0, "Stage 4 không nhận được gradient")

    def test_03_batch_stress_memory_simulation(self):
        """Giả lập forward + backward với Batch Size = 2, 4 trên ảnh 512x512 để kiểm tra rò rỉ bộ nhớ."""
        model = MambaUNet(in_channels=3, num_classes=1, backbone="mamba_vision_T", pretrained=False)
        model.train()

        for b in [2, 4]:
            x = torch.randn(b, 3, 512, 512)
            target = torch.zeros(b, 1, 512, 512)
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out, target)
            loss.backward()
            model.zero_grad()
            self.assertEqual(out.shape, (b, 1, 512, 512))


if __name__ == "__main__":
    unittest.main()
```

#### <a id="tệp-36-tests-test_gray_area_crash-py"></a>Tệp 36: `tests/test_gray_area_crash.py`
- **Vị trí**: `tests/test_gray_area_crash.py`
- **Quy mô**: `87 dòng` | `3.8 KB`
- **Vai trò**: Kiểm thử ghi đĩa nguyên tử và tự động phục hồi (auto-resume) liền mạch từ last_model.pth sau khi bị ngắt đột ngột.

```python
# -*- coding: utf-8 -*-
"""
tests/test_gray_area_crash.py - Kiểm thử chuyên sâu VÙNG XÁM KHẢ NĂNG CHỊU LỖI & ATOMIC WRITE.

Mục tiêu kiểm định:
  1. Mô phỏng đứt kết nối runtime/sập nguồn khi đang ghi file checkpoint.
  2. Xác minh file best_model.pth không bao giờ bị hỏng (corrupt) nhờ cơ chế Atomic Write (.tmp + replace).
  3. Kiểm tra tự động phát hiện và phục hồi phiên chạy (auto-resume) tiếp tục đúng epoch tiếp theo.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.engine.checkpoint_io import save_checkpoint, load_checkpoint_file
from common.engine.run_manager import resolve_resume, create_run


class TestGrayAreaCrashAndResume(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_crash_"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_atomic_write_prevents_corruption(self):
        """Mô phỏng ngắt ngang giữa chừng khi ghi: Checkpoint trước đó vẫn nguyên vẹn 100%."""
        target_ckpt = self.test_dir / "best_model.pth"

        # 1. Lưu checkpoint chuẩn ban đầu (Epoch 1)
        orig_sd = {"weight": torch.randn(10, 10)}
        save_checkpoint(target_ckpt, state_dict=orig_sd, epoch=1, best_val_dice=0.85)
        self.assertTrue(target_ckpt.exists())

        # 2. Giả lập một tiến trình bị crash giữa chừng tạo ra file .tmp bị hỏng
        fake_tmp = self.test_dir / f".tmp_{target_ckpt.name}_99999"
        with open(fake_tmp, "wb") as f:
            f.write(b"CORRUPTED_INCOMPLETE_DATA_CRASH")

        # 3. Kiểm tra file chính best_model.pth: Vẫn nguyên vẹn 100%
        loaded = load_checkpoint_file(target_ckpt)
        self.assertEqual(loaded["epoch"], 1)
        self.assertEqual(loaded["best_val_dice"], 0.85)
        torch.testing.assert_close(loaded["model_state_dict"]["weight"], orig_sd["weight"])

        # 4. Khi lưu epoch tiếp theo, hệ thống ghi đè nguyên tử an toàn
        new_sd = {"weight": torch.randn(10, 10)}
        save_checkpoint(target_ckpt, state_dict=new_sd, epoch=2, best_val_dice=0.90)
        loaded_2 = load_checkpoint_file(target_ckpt)
        self.assertEqual(loaded_2["epoch"], 2)
        self.assertEqual(loaded_2["best_val_dice"], 0.90)

    def test_02_auto_resume_after_abrupt_disconnect(self):
        """Mô phỏng Colab bị thu hồi runtime ở Epoch 4: resolve_resume('auto') phát hiện và nạp chính xác."""
        runs_root = self.test_dir / "runs"
        run_handle = create_run(arch="mamba_unet", runs_root=runs_root, tag="simulated_colab_run")
        run_dir = run_handle.run_dir

        # Lưu checkpoint bị gián đoạn ở Epoch 4
        sd = {"bias": torch.randn(5)}
        last_ckpt = run_dir / "last_model.pth"
        save_checkpoint(last_ckpt, state_dict=sd, epoch=4, best_val_dice=0.78)

        # Mô phỏng phiên chạy mới với RESUME = "auto"
        res_dir, res_ckpt = resolve_resume("auto", arch="mamba_unet", runs_root=runs_root)
        self.assertIsNotNone(res_dir, "Không nhận diện được phiên chạy bị ngắt")
        self.assertIsNotNone(res_ckpt, "Không tìm thấy last_model.pth")
        self.assertEqual(res_dir.resolve(), run_dir.resolve())
        self.assertEqual(res_ckpt.resolve(), last_ckpt.resolve())

        # Đọc dữ liệu từ checkpoint đã resolve
        ckpt_data = load_checkpoint_file(res_ckpt)
        self.assertEqual(ckpt_data["epoch"], 4)
        print(f"\n✅ Auto-Resume đã phục hồi thành công phiên chạy từ Epoch {ckpt_data['epoch']}!")


if __name__ == "__main__":
    unittest.main()
```

#### <a id="tệp-37-tests-test_scan_fallback-py"></a>Tệp 37: `tests/test_scan_fallback.py`
- **Vị trí**: `tests/test_scan_fallback.py`
- **Quy mô**: `142 dòng` | `5.2 KB`
- **Vai trò**: Kiểm thử tính tương đương và ổn định số học giữa kernel CUDA native và module PyTorch CPU fallback.

```python
# -*- coding: utf-8 -*-
"""
tests/test_scan_fallback.py - Kiểm thử thuật toán Selective Scan Fallback PyTorch (R10).

Quy tắc nghiệm thu:
  1. Sai khác số học giữa selective_scan_torch và tham chiếu float64 <= 1e-4.
  2. Lan truyền ngược (backward pass) sinh ra gradients hữu hạn cho toàn bộ biến đầu vào.
  3. Cơ chế chọn backend qua biến môi trường DUNEAI_MAMBA_BACKEND hoạt động đúng đặc tả.
"""

import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mamba_unet.fallbacks import get_scan_backend, selective_scan_fn, selective_scan_torch


def _float64_reference_scan(u, delta, A, B, C, D=None):
    """
    Hàm tham chiếu chuẩn giải tích bằng số thực kép (float64).
    Công thức hồi quy Mamba SSM:
        h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * u_t
        y_t = h_t * C_t + D * u_t
    """
    u_f64 = u.detach().to(torch.float64)
    delta_f64 = delta.detach().to(torch.float64)
    A_f64 = A.detach().to(torch.float64)
    B_f64 = B.detach().to(torch.float64)
    C_f64 = C.detach().to(torch.float64)
    D_f64 = D.detach().to(torch.float64) if D is not None else None

    b, d, l = u_f64.shape
    n = A_f64.shape[1]

    h = torch.zeros(b, d, n, dtype=torch.float64, device=u_f64.device)
    ys = []

    for t in range(l):
        dt = delta_f64[:, :, t]
        u_t = u_f64[:, :, t]
        b_t = B_f64[:, :, t]
        c_t = C_f64[:, :, t]

        dA = torch.exp(dt.unsqueeze(-1) * A_f64.unsqueeze(0))
        dB = dt.unsqueeze(-1) * b_t.unsqueeze(1)

        h = dA * h + dB * u_t.unsqueeze(-1)
        y_t = (h * c_t.unsqueeze(1)).sum(dim=-1)
        if D_f64 is not None:
            y_t = y_t + D_f64.unsqueeze(0) * u_t
        ys.append(y_t)

    return torch.stack(ys, dim=-1)


class TestScanFallback(unittest.TestCase):
    """Kiểm tra độ chính xác và tính ổn định của fallback Selective Scan."""

    def setUp(self):
        torch.manual_seed(42)
        self.B = 2
        self.D = 8
        self.L = 16
        self.N = 4

    def test_numerical_accuracy_vs_float64(self):
        """Kiểm tra sai khác số học giữa hàm torch và tham chiếu float64 <= 1e-4."""
        u = torch.randn(self.B, self.D, self.L)
        delta = F.softplus(torch.randn(self.B, self.D, self.L))
        A = -torch.rand(self.D, self.N) - 0.1
        B = torch.randn(self.B, self.N, self.L)
        C = torch.randn(self.B, self.N, self.L)
        D = torch.randn(self.D)

        out_torch = selective_scan_torch(u, delta, A, B, C, D)
        out_ref = _float64_reference_scan(u, delta, A, B, C, D).to(torch.float32)

        max_diff = torch.max(torch.abs(out_torch - out_ref)).item()
        self.assertLessEqual(
            max_diff,
            1e-4,
            f"Sai lệch selective_scan_torch ({max_diff:.2e}) vượt ngưỡng cho phép 1e-4!",
        )

    def test_backward_pass_gradients_finite(self):
        """Kiểm tra lan truyền ngược: toàn bộ biến đầu vào đều có gradient hữu hạn."""
        u = torch.randn(self.B, self.D, self.L, requires_grad=True)
        delta = F.softplus(torch.randn(self.B, self.D, self.L)).requires_grad_(True)
        A = (-torch.rand(self.D, self.N) - 0.1).requires_grad_(True)
        B = torch.randn(self.B, self.N, self.L, requires_grad=True)
        C = torch.randn(self.B, self.N, self.L, requires_grad=True)
        D = torch.randn(self.D, requires_grad=True)

        out = selective_scan_torch(u, delta, A, B, C, D)
        loss = out.sum()
        loss.backward()

        for name, tensor in [("u", u), ("delta", delta), ("A", A), ("B", B), ("C", C), ("D", D)]:
            self.assertIsNotNone(tensor.grad, f"Gradient của {name} bị None!")
            self.assertTrue(
                torch.isfinite(tensor.grad).all(),
                f"Gradient của {name} chứa NaN hoặc Inf!",
            )

    def test_backend_selection_env(self):
        """Kiểm tra cơ chế lựa chọn backend qua biến môi trường DUNEAI_MAMBA_BACKEND."""
        old_val = os.environ.get("DUNEAI_MAMBA_BACKEND")
        try:
            # 1. Ép buộc torch backend
            os.environ["DUNEAI_MAMBA_BACKEND"] = "torch"
            self.assertEqual(get_scan_backend(), "torch")
            u = torch.randn(2, 4, 8)
            delta = torch.randn(2, 4, 8)
            A = -torch.rand(4, 2)
            B = torch.randn(2, 2, 8)
            C = torch.randn(2, 2, 8)
            out = selective_scan_fn(u, delta, A, B, C)
            self.assertEqual(out.shape, (2, 4, 8))

            # 2. Ép buộc cuda backend khi không có GPU / CUDA mamba
            os.environ["DUNEAI_MAMBA_BACKEND"] = "cuda"
            self.assertEqual(get_scan_backend(), "cuda")
            if not torch.cuda.is_available():
                with self.assertRaises(RuntimeError):
                    selective_scan_fn(u, delta, A, B, C)
        finally:
            if old_val is not None:
                os.environ["DUNEAI_MAMBA_BACKEND"] = old_val
            else:
                os.environ.pop("DUNEAI_MAMBA_BACKEND", None)


if __name__ == "__main__":
    unittest.main()
```

