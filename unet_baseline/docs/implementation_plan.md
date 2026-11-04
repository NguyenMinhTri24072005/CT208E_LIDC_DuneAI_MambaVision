# Kế hoạch Triển khai (Implementation Plan) - U-Net Baseline

## 1. Các mốc thời gian (Milestones) & Trình tự thực hiện
Quá trình xây dựng được chia thành 4 giai đoạn, ưu tiên tính độc lập của các module và có thể test ngay sau khi viết xong.

### Giai đoạn 1: Khởi tạo và Xây dựng Mô hình Cốt lõi
- **Mục tiêu:** Xây dựng xong kiến trúc mạng U-Net chuẩn và kiểm thử forward pass.
- **Checklist:**
  - [ ] Khởi tạo cấu trúc thư mục dự án `unet_baseline/`.
  - [ ] Viết `requirements.txt` và `README.md`.
  - [ ] Tạo `models/__init__.py`.
  - [ ] Tạo `models/unet_parts.py` (DoubleConv, Down, Up, OutConv).
  - [ ] Tạo `models/unet_model.py` (UNet).
  - [ ] Tạo `configs/config.yaml` (các tham số cơ bản về model).
- **Kiểm thử:** Viết đoạn code khởi tạo `UNet` và chạy một dummy tensor qua mô hình để đảm bảo output shape chính xác kể cả khi Input Height/Width không chia hết cho 16.

### Giai đoạn 2: Chuẩn bị Dữ liệu và Hàm đánh giá
- **Mục tiêu:** Viết xong luồng load dữ liệu, augmentation và các hàm tính loss/metric.
- **Checklist:**
  - [ ] Cập nhật `configs/config.yaml` (thêm phần data, loss, augment).
  - [ ] Tạo `utils/__init__.py`.
  - [ ] Tạo `utils/transforms.py` (sử dụng thư viện Albumentations).
  - [ ] Tạo `utils/dataset.py` (Custom `SegmentationDataset` & `get_dataloaders`).
  - [ ] Tạo `utils/losses.py` (DiceLoss, BCE, FocalLoss, CombinedLoss).
  - [ ] Tạo `utils/metrics.py` (Tính IoU, Dice, HD95).
- **Kiểm thử:** Viết hàm test load 1 batch từ Dataset, trực quan hoá (matplotlib) ảnh, mask gốc và mask sau khi augmentation để kiểm tra độ đồng nhất.

### Giai đoạn 3: Vòng lặp Huấn luyện (Training Loop) & Đánh giá
- **Mục tiêu:** Xây dựng pipeline huấn luyện hoàn chỉnh, tích hợp Early Stopping, AMP, TensorBoard và khả năng lưu/resume.
- **Checklist:**
  - [ ] Khởi tạo thư mục `checkpoints/`.
  - [ ] Tạo `train.py` (Xử lý args CLI override config, train loop, validation, logging, saving/resuming).
  - [ ] Tạo `evaluate.py` (Tính toán chỉ số chi tiết trên tập test, tính std, xuất CSV/JSON).
  - [ ] Tạo `predict.py` (Inference trên ảnh lẻ/thư mục, lưu overlay).
- **Kiểm thử:** Chạy tính năng Overfit trên 1 batch nhỏ để đảm bảo loss hội tụ về gần 0 và Dice tiến tới xấp xỉ 1 sau vài chục iterations.

### Giai đoạn 4: Tích hợp Google Colab
- **Mục tiêu:** Đảm bảo dự án có một luồng chạy "End-to-End" trơn tru trên môi trường Colab.
- **Checklist:**
  - [ ] Tạo `train_on_colab.ipynb` với các cell được phân tách rõ ràng (Mount Drive, Cài đặt, Huấn luyện, Resume, Đánh giá, Inference).
- **Kiểm thử:** Reset runtime Colab và bấm "Run All" để xác nhận code tự động trích xuất cấu hình, train và trả về kết quả mượt mà.

---

## 2. Rủi ro và Cách giảm thiểu (Risks & Mitigations)
- **Rủi ro 1 (Lệch shape kích thước ảnh trong nhánh Up):** Khi kích thước đầu vào (Height/Width) lẻ hoặc không chia hết cho 16, tại nhánh Decoder kích thước Upsample có thể bị lệch 1 pixel so với Skip Connection từ Encoder, làm hàm `torch.cat` báo lỗi kích thước.
  - *Cách giảm thiểu:* Tại module `Up` trong `unet_parts.py`, tính chênh lệch chiều không gian `(diffY, diffX)` và sử dụng `F.pad` vào tensor được upsample để cân bằng kích thước hoàn hảo trước khi nối.
- **Rủi ro 2 (Overfitting nhanh / Data Imbalance):** Mask của ảnh y tế thường chứa rất ít pixel dương (mất cân bằng lớp).
  - *Cách giảm thiểu:* Bổ sung trọng số cấu hình loss (`CombinedLoss = BCE + Dice`), và kỹ thuật Data Augmentation mạnh mẽ như ElasticTransform.
- **Rủi ro 3 (Tràn VRAM trên GPU T4/L4):** Training ảnh to hoặc Batch size lớn.
  - *Cách giảm thiểu:* Tích hợp AMP (`torch.cuda.amp`), tuỳ chọn Gradient Checkpointing (`torch.utils.checkpoint`) bên trong `unet_model.py`.
- **Rủi ro 4 (Colab ngắt kết nối):**
  - *Cách giảm thiểu:* Tự động lưu `last.pth` sau mỗi epoch. Thiết lập hàm đọc resume state (optimizer, scheduler) ở đầu file `train.py`.

---

## 3. Ước lượng độ phức tạp
- **Quy mô code:** Khoảng 12-15 file (nhỏ gọn nhưng đầy đủ tiêu chuẩn production).
- **Độ khó kỹ thuật:** Trung bình. Điểm hóc búa nhất là xử lý đồng bộ Augmentation cho mask/ảnh trong Albumentations, và tính toán số liệu Hausdorff Distance 95 (HD95) cho các ảnh mask trống (âm tính hoàn toàn).
- **Thời gian hoàn thành ước tính:** 1-2 phiên làm việc (nếu specs rõ ràng và cung cấp đủ thông số dataset).

---

## 4. Tiêu chí nghiệm thu (Acceptance Criteria)
1. **Model:** Forward pass chạy chuẩn shape bất kể kích thước ảnh không chia hết cho 16.
2. **Logic Train:** Overfit được 1 batch nhỏ (loss hội tụ gần 0).
3. **Pipeline:** `train.py` chạy qua ít nhất 1 epoch thực tế không báo lỗi; khả năng `resume` hoạt động trơn tru.
4. **Đánh giá:** `evaluate.py` xuất ra CSV đầy đủ và `predict.py` sinh ra ảnh overlay chuẩn màu.
5. **Notebook:** `train_on_colab.ipynb` chạy tự động tuần tự "một chạm".
6. **Không gian làm việc:** Toàn bộ code chỉ thao tác giới hạn trong thư mục `SOURCES/unet_baseline/`.

---

## 5. Câu hỏi cần User xác nhận (Cần cung cấp)
Để file `config.yaml` và `dataset.py` được thiết kế khớp 100% với dữ liệu thật của bạn, xin hãy cho tôi biết:
1. Bạn lưu dữ liệu ảnh/mask dưới **định dạng file** gì? (`.npy`, `.png`, `.jpg` hay `.tif`?)
2. Kích thước **kênh màu đầu vào**: Ảnh là 1 kênh (Grayscale) hay 3 kênh (RGB / 2.5D Stack)?
3. **Bài toán** này là Nhị phân (1 lớp dự đoán khối u vs nền) hay Đa lớp (dự đoán nhiều loại cơ quan khác nhau)?
4. (Tùy chọn) Kích thước `image_size` mặc định bạn muốn sử dụng là bao nhiêu? (Ví dụ 256x256).
