# ĐÁNH GIÁ CHUYÊN SÂU CÁC ĐỀ XUẤT ĐIỀU CHỈNH KẾ HOẠCH FINE-TUNING VÀ THIẾT KẾ LỘ TRÌNH THỰC NGHIỆM TỐI ƯU

* **Tài liệu đối chiếu**:
  * Kế hoạch gốc: [`MAMBA_UNET_REVIEW_AND_FIXES.md`](MAMBA_UNET_REVIEW_AND_FIXES.md)
  * Đề xuất điều chỉnh: [`STEP0_PLAN_ADJUSTMENT_PROPOSAL.md`](STEP0_PLAN_ADJUSTMENT_PROPOSAL.md)
* **Dữ liệu thực nghiệm căn cứ**:
  * Tổng hợp chỉ số Bước 0: [`docs/STEP0_MEASUREMENT_RESULTS/STEP0_FULL_METRICS_SUMMARY.md`](../STEP0_MEASUREMENT_RESULTS/STEP0_FULL_METRICS_SUMMARY.md)
  * Chi tiết 75 ca Validation: [`eval_results/step0_sweep/sweep_r0_patients.csv`](../../eval_results/step0_sweep/sweep_r0_patients.csv)
  * Chi tiết 203 nốt u Ground Truth: [`eval_results/step0_sweep/sweep_r0_nodules.csv`](../../eval_results/step0_sweep/sweep_r0_nodules.csv)
  * Báo cáo phiên chạy $R_0$: [`mamba_unet/runs/mamba_colab_500item_25epoch_extra_test/training_report.json`](../../mamba_unet/runs/mamba_colab_500item_25epoch_extra_test/training_report.json)
* **Mô hình cơ sở (Ground Baseline $R_0$)**: `best_model.pth` (500 ca LIDC-IDRI, 42 epochs, $\text{Val Dice}_{2D} = 0.7456$, $\text{Mean Dice}_{3D} = 0.3771$).

---

## 1. TỔNG QUAN BỐI CẢNH ĐỐI CHIẾU

Sau khi hoàn thành khâu đo lường chẩn đoán toàn diện tại **Bước 0**, hệ thống đã có đầy đủ dữ liệu thực nghiệm khách quan trên toàn bộ $75$ bệnh nhân Validation và $203$ nốt u Ground Truth (thay vì chỉ dựa trên các giả thuyết lý thuyết ban đầu).

Bản đề xuất [`STEP0_PLAN_ADJUSTMENT_PROPOSAL.md`](STEP0_PLAN_ADJUSTMENT_PROPOSAL.md) được xây dựng dựa trên các dữ liệu mới này. Dưới đây là bản đánh giá chi tiết, phân tách rạch ròi giữa:
1. **Các điểm chưa hợp lý, chứa mâu thuẫn kỹ thuật hoặc vi phạm nguyên tắc nghiên cứu khoa học** (cần bác bỏ hoặc hiệu chỉnh).
2. **Các điểm hoàn toàn hợp lý, sắc bén và được củng cố vững chắc bởi số liệu thực nghiệm** (cần tích hợp vào kế hoạch chính thức).
3. **Lộ trình thực nghiệm hoàn thiện tối ưu** phục vụ trực tiếp cho báo cáo và đồ án CT208E.

---

## 2. PHÂN TÍCH & CHỨNG MINH CÁC ĐỀ XUẤT CHƯA HỢP LÝ (KÈM SỐ LIỆU ĐỐI CHỨNG)

---

### 2.1 Chưa hợp lý #1: Gộp 7 biến độc lập vào Run 1 (Đề xuất 6) — Vi phạm nguyên tắc Ablation Study

#### a) Nội dung đề xuất:
Đề xuất 6 đề nghị hợp nhất toàn bộ các thay đổi vào ngay Phiên chạy số 1 (`R1_Data_Loss_Opt`):
* Kích thước ảnh: $512 \rightarrow 448$
* Batch size & Learning rate: $8 \rightarrow 12$, $\text{LR} = 1.2 \times 10^{-4}$
* Lọc nốt nhỏ: $\text{min\_train\_area} = 20.0 \rightarrow 5.0\text{ mm}^2$
* Tỷ lệ mẫu âm tính: $\text{neg\_ratio} = 0.25 \rightarrow 0.50$ (kèm vá ca không u)
* Trọng số hàm mất mát: $\text{pos\_weight} = 4.0 \rightarrow 2.0$
* Tối ưu hóa suy giảm trọng số: $\text{weight\_decay} = 10^{-4} \rightarrow 0.02$ (4 nhóm tham số)
* Làm mượt trọng số: Kích hoạt $\text{EMA} = 0.999$

#### b) Chứng minh sự không hợp lý & Rủi ro phương pháp luận:
1. **Mất kiểm soát biến độc lập (Loss of Scientific Interpretability)**:
   Trong một nghiên cứu khoa học hoặc đồ án chuyên ngành (CT208E), mục đích tối thượng của bảng thực nghiệm **Ablation Study** là định lượng xem *mỗi cải tiến đóng góp bao nhiêu phần trăm vào kết quả*.
   Nếu ném toàn bộ 7 biến vào cùng một lượt chạy và điểm $\text{Dice}_{3D}$ tăng từ $0.3771$ lên $0.5500$, bạn **hoàn toàn không có bằng chứng khoa học** để kết luận:
   * Mức tăng điểm đến từ việc cứu nốt nhỏ ($5\text{ mm}^2$) hay do giảm dương tính giả ($\text{pos\_weight} = 2.0$)?
   * Đóng góp của $\text{EMA}$ và $\text{Weight Decay}$ là bao nhiêu?
2. **Rủi ro bế tắc khi debug nếu mô hình phân kỳ hoặc suy giảm hiệu năng**:
   Việc tăng learning rate lên $1.2 \times 10^{-4}$ đồng thời tăng $\text{weight\_decay}$ gấp $200\text{ lần}$ (từ $10^{-4}$ lên $0.02$) trên backbone MambaVision có thể tạo ra tương tác không mong muốn. Nếu Run 1 bị lỗi, việc có quá nhiều biến thay đổi cùng lúc sẽ khiến ta không thể cô lập được nguyên nhân.
3. **Giải pháp chuẩn hóa**: Phải tách riêng ít nhất 2 nhóm biến độc lập:
   * **Nhóm 1 (Data & Loss)**: Sửa triệt để bản chất dữ liệu và mặt phẳng tối ưu (`min_train_area 5.0`, `neg_ratio 0.5`, `pos_weight 2.0`).
   * **Nhóm 2 (Regularization & Augmentation)**: Chống overfitting và làm mượt trọng số (`weight_decay 0.02`, `EMA 0.999`, `--aug-level 2`).

---

### 2.2 Chưa hợp lý #2: Gán cứng "con số ma thuật 16" cho ca không u (`neg_for_empty`) (Đề xuất 2)

#### a) Nội dung đề xuất:
Trong `common/data_pipeline/offline_builder.py`, đề xuất sửa nhánh `else` (khi ca bệnh không có lát dương tính nào) thành:
```python
neg_for_empty = max(1, int(round(16 * zero_sl_ratio))) if zero_sl_ratio > 0 else 0
num_neg_keep = min(len(neg_candidates), neg_for_empty)
```

#### b) Chứng minh sự không hợp lý bằng số liệu:
1. **Con số 16 không phản ánh đúng thực tế dữ liệu**:
   Trong tập huấn luyện $500$ ca thực tế của phiên $R_0$, có $4,069$ lát cắt dương tính phân bố trên $\approx 350$ ca bệnh có khối u. Số lát cắt dương tính trung bình thực tế trên mỗi ca có u là:
   $$\bar{N}_{\text{pos}} = \frac{4069}{350} \approx \mathbf{11.6\text{ lát/ca}} \quad (\text{không phải } 16)$$
2. **Gây ra nghịch lý lấy mẫu ngược (Sampling Paradox)**:
   * Với công thức trên, khi đặt $\text{zero\_sl\_ratio} = 0.5$:
     * Một ca bệnh **hoàn toàn không có u** ($n_{\text{gt}} = 0$) sẽ được trích xuất: $\text{round}(16 \times 0.5) = \mathbf{8\text{ lát âm tính}}$.
     * Nhưng một ca bệnh **có 1 nốt u nhỏ** chỉ xuất hiện trên đúng $1$ lát cắt duy nhất $\Rightarrow \text{num\_neg\_keep} = \text{round}(1 \times 0.5) = \mathbf{1\text{ lát âm tính}}$!
   * **Nghịch lý**: Ca có u nhỏ lại chỉ đóng góp vỏn vẹn $1$ lát âm tính để mô hình học cấu trúc xung quanh, trong khi ca không có u lại đóng góp gấp 8 lần ($8$ lát âm tính).
3. **Giải pháp chuẩn hóa**: Thay vì nhân với số 16 giả định, áp dụng cơ chế **Ngưỡng sàn mẫu âm tính tối thiểu (`min_neg_per_case`)** cho *toàn bộ mọi ca bệnh*:
   ```python
   # Đảm bảo mọi ca bệnh (dù có 0 lát u, 1 lát u hay nhiều lát u) đều đóng góp ít nhất min_neg_per_case lát âm tính
   num_neg_keep = max(min_neg_per_case, int(round(len(pos_slices) * zero_sl_ratio)))
   num_neg_keep = min(len(neg_candidates), num_neg_keep)
   ```
   *(Ví dụ đặt `min_neg_per_case = 4`: Ca rỗng lấy 4 lát âm; ca có 1 lát u lấy 4 lát âm; ca có 20 lát u lấy $\max(4, 10) = 10$ lát âm).*

---

### 2.3 Chưa hợp lý #3: Cắt giảm `--epochs 30` và `--patience 8` khi đang tăng Regularization (Đề xuất 6)

#### a) Nội dung đề xuất:
Đề xuất hạ số epoch tối đa từ 50 xuống 30 và giảm patience từ 12 xuống 8.

#### b) Chứng minh sự không hợp lý bằng quy luật tối ưu hóa & số liệu $R_0$:
1. **Số liệu từ lịch sử huấn luyện Baseline $R_0$ (`train.log`)**:
   * Checkpoint tốt nhất đạt được tại **Epoch 30** ($\text{Val Dice} = 0.7456$).
   * Quá trình huấn luyện tiếp tục diễn ra ổn định đến tận **Epoch 42** mới chạm ngưỡng dừng sớm (`patience = 12`).
2. **Quy luật vật lý của hàm mất mát khi tăng Regularization**:
   * Khi kích hoạt `weight_decay = 0.02` (phạt co cụm trọng số, ép mạng không bão hòa biên) và `EMA = 0.999` (lấy trung bình trượt trọng số qua $\approx 1.5$ epoch), mô hình **hội tụ chậm hơn và mượt mà hơn** so với khi dùng `wd = 1e-4`.
   * Nếu cắt số epoch xuống 30 và patience xuống 8, khi gặp pha dao động tự nhiên quanh Epoch 15–20, cơ chế Early Stopping sẽ ngắt ngang quá trình fit ở khoảng **Epoch 22–24**, trước khi mô hình kịp hội tụ đến điểm tối ưu toàn cục (**nguy cơ Underfitting rất cao**).
3. **Giải pháp chuẩn hóa**: Duy trì tối thiểu `--epochs 40` và `--patience 10`.

---

### 2.4 Chưa hợp lý #4: Sửa trực tiếp `mamba_unet/decoder.py` — Vi phạm Quy tắc Kiến trúc chuẩn Rule R1 (Đề xuất 4)

#### a) Nội dung đề xuất:
Đề xuất can thiệp và sửa đổi trực tiếp vào tệp [`mamba_unet/decoder.py`](../../mamba_unet/decoder.py) để tích hợp các nhánh Skip độ phân giải cao `up4`, `up5`, `hr_stem`.

#### b) Chứng minh sự không hợp lý:
1. **Vi phạm nguyên tắc bảo tồn Baseline (Rule R1)**:
   * Quy chuẩn kiến trúc của dự án quy định: Thư mục gốc `unet_base/` và tệp giải mã tiêu chuẩn `mamba_unet/decoder.py` phải được **giữ nguyên vẹn** làm mô hình tham chiếu chuẩn.
   * Nếu ghi đè trực tiếp các khối $H/2, H/1$ vào `UNetDecoder`, toàn bộ các bài kiểm thử hồi quy (`test_models.py`, `test_mamba_unet.py`) và khả năng tải lại các checkpoint cũ ($R_0$) sẽ bị lỗi cấu trúc `state_dict`.
2. **Giải pháp chuẩn hóa**:
   * Kế thừa lớp `UNetDecoder` thành một class mới hoặc tạo module riêng (ví dụ: `mamba_unet/hr_decoder.py`), hoặc sử dụng cờ khởi tạo `high_res_skip=False` (mặc định tắt hoàn toàn) như Plan gốc §6.1 đã thiết kế, đảm bảo khi chạy các Run 1 và Run 2 thì kiến trúc mạng vẫn giữ nguyên $100\%$.

---

### 2.5 Chưa hợp lý #5: Rủi ro bùng nổ Dương tính giả nếu hạ `--min-area 6.0` khi chưa huấn luyện lại (Đề xuất 1)

#### a) Nội dung đề xuất:
Hạ tham số lọc hậu xử lý `min_slice_area_mm2` từ $20.0$ xuống $6.0\text{ mm}^2$ ở khâu suy luận 3D.

#### b) Chứng minh rủi ro bằng số liệu:
1. **Số liệu dương tính giả tại Bước 0**:
   * Khi đang áp dụng ngưỡng lọc lớn $20.0\text{ mm}^2$, số cụm dương tính giả của $R_0$ đã ở mức rất cao: **$2.91\text{ FP/ca}$** ($Th = 0.6$) và **$2.97\text{ FP/ca}$** ($Th = 0.5$).
2. **Hậu quả nếu hạ ngưỡng lọc đơn thuần**:
   * Nếu hạ `--min-area 6.0` mà áp dụng lên mô hình chưa được sửa hàm loss, toàn bộ các đốm nhiễu nhỏ từ $6\text{--}19\text{ mm}^2$ (mạch máu rốn phổi, xẹp phổi rải rác) sẽ được giải phóng, đẩy lượng FP lên **$5\text{--}8\text{ FP/ca}$**, khiến điểm $\text{Dice}_{3D}$ sụp đổ nghiêm trọng hơn nữa.
3. **Giải pháp chuẩn hóa**:
   * Việc hạ `--min-area 6.0` **chỉ được áp dụng cho các mô hình từ Run 1 trở đi** (nơi đã có `neg_ratio 0.5` và `pos_weight 2.0` để kìm hãm FP).
   * Đồng thời, **bắt buộc phải kích hoạt điều kiện lọc đa lát cắt** `min_slices_span >= 2` (hoặc `min_volume_mm3 >= 10.0`) trong `postprocess_3d_mask` để loại bỏ các đốm nhiễu chỉ lóe lên trên $1$ lát cắt đơn lẻ.

---

## 3. PHÂN TÍCH & CỦNG CỐ CÁC ĐỀ XUẤT HOÀN TOÀN HỢP LÝ (BẰNG CHỨNG SỐ LIỆU VỮNG CHẮC)

Các điểm dưới đây trong bản đề xuất mới đã chỉ ra chính xác các khiếm khuyết chết người của Plan gốc và được chứng minh bằng số liệu thực nghiệm:

---

### 3.1 Hợp lý #1: Bãi bỏ mâu thuẫn "giữ 20 mm² ở hậu xử lý" của Plan gốc (Đề xuất 1) ⭐⭐⭐

* **Mâu thuẫn trong Plan gốc**:
  Tiêu đề mục §3.1 trong `MAMBA_UNET_REVIEW_AND_FIXES.md` viết: *"Cho phép hạ ngưỡng lát u khi dựng chỉ mục train (**giữ 20 mm² ở hậu xử lý**)"*.
* **Số liệu thực nghiệm chứng minh sự phi lý**:
  1. Theo `sweep_r0_nodules.csv`, trong $203$ nốt u Ground Truth của tập Validation, có tới **$59$ nốt ($29.06\%$)** có đường kính $d < 5.0\text{ mm}$ ($A_{\max} < 19.63\text{ mm}^2$).
  2. Tại Bước 0, độ nhạy của nhóm nốt này chỉ đạt **$3.39\%$ ($2/59$ nốt)**.
  3. Ở dải giáp ranh $5.05\text{--}6.00\text{ mm}$ ($A_{\max} \in [20, 28.3]\text{ mm}^2$), độ nhạy chỉ đạt **$25.93\%$ ($7/27$ nốt)**.
  4. Nếu Plan gốc giữ nguyên $20.0\text{ mm}^2$ ở hậu xử lý, thì dù mô hình mới có học giỏi đến mức dự đoán đúng $100\%$ các nốt nhỏ, **bộ lọc hậu xử lý vẫn sẽ xóa sạch $100\%$ các nốt này**! Trần độ nhạy tối đa của toàn hệ thống sẽ bị chặn cứng ở:
     $$\text{Sensitivity}_{\max} = \frac{203 - 59}{203} = \mathbf{70.94\%}$$
  * $\Rightarrow$ **Kết luận**: Đề xuất hạ đồng bộ `--min-train-area 5.0` lúc train và `--min-area 6.0` lúc eval (kèm `min_slices_span >= 2`) là phát hiện có giá trị sống còn.

---

### 3.2 Hợp lý #2: Phát hiện lỗi xung đột nhãn và Hủy bỏ Run R5 (`--tumor-union-p 0.0`) (Đề xuất 3) ⭐⭐⭐

* **Nội dung Plan gốc**: Chạy Run R5 với cờ `--tumor-union-p 0.0` (không hợp mask u vào mask phổi lúc train).
* **Số liệu Bước 0 chứng minh P3 không phải nguyên nhân chính**:
  * Kết quả `check_cache.py` trên $449$ ca bệnh:
    * Trung vị tỷ lệ u trong phổi: $\text{Median} = \mathbf{1.000}$
    * Phân vị thứ 5: $p_5 = \mathbf{0.994}$ ($95\%$ bệnh nhân giữ được $\ge 99.4\%$ thể tích u).
    * Chỉ có $8/449$ ca ($1.78\%$) có tỷ lệ $< 0.90$. Đặc biệt, **cả 5 ca xấu nhất (`0339, 0067, 0454, 0259, 0074`) đều nằm ở tập Train, không có ca nào nằm trong 75 ca tập Validation!**
* **Phát hiện lỗi logic rò rỉ ngược trong code Plan gốc (§3.2)**:
  * Trong `dataset.py`:
    ```python
    keep = lung_keep_mask(lung_stack, tumor_mask=tumor_for_keep)
    img_stack = apply_lung_mask(img_stack, keep, fill=HU_FLOOR) # Ngoài keep bị gán -1000 HU (đen)
    center_mask = (msk_stack[self.half_k : self.half_k + 1] > 0).astype(np.float32) # Nhãn giữ nguyên 1.0!
    ```
  * Nếu chạy Run R5 với `--tumor-union-p 0.0`, ở các ca có nốt u ngoài phổi như `LIDC-IDRI-0339` ($100\%$ u ngoài mask phổi), ảnh đầu vào tại vị trí khối u bị biến thành không khí $-1000\text{ HU}$, nhưng nhãn Ground Truth vẫn bắt mô hình phải dự đoán là Khối u! **Mạng bị ép học nhãn u trên nền không khí $\Rightarrow$ Phá hoại gradient huấn luyện.**
  * $\Rightarrow$ **Kết luận**: Quyết định hủy bỏ Run R5 và sửa code đồng bộ nhãn là hoàn toàn chính xác, tiết kiệm trực tiếp 3–4 giờ GPU Colab.

---

### 3.3 Hợp lý #3: Kích hoạt ngay `IMAGE_SIZE = 448` từ đầu và bỏ qua Shifted Window Stage 3–4 (Đề xuất 4) ⭐⭐⭐

* **Số liệu chứng minh an toàn tuyệt đối ($100\%$)**:
  * `check_cache.py` xác nhận: Mặc dù có $21/449$ ca có kích thước cơ thể $> 448\text{ mm}$, nhưng khi cắt tâm về $448 \times 448$, **tỷ lệ mất nhãn nốt u là chính xác $0/449$ ca ($0.00\%$)** (do phổi luôn nằm ở trung tâm lồng ngực).
* **Số liệu chứng minh tối ưu phần cứng**:
  * Tại $512 \times 512$: Stage 3 ($32 \times 32$, cửa sổ 14) bị pad lên $42 \times 42 = \mathbf{1,764\text{ tokens}}$.
  * Tại $448 \times 448$: Stage 3 ($28 \times 28$, cửa sổ 14) vừa khít $2 \times 2$ cửa sổ $= \mathbf{784\text{ tokens}}$ (**giảm $2.25\text{ lần}$** số token qua khối Mamba/Attention).
  * Thời gian mỗi epoch giảm từ $5\text{ phút }16\text{ giây}$ xuống $\approx \mathbf{3\text{ phút }50\text{ giây}}$ (tiết kiệm $\sim 25\%$ thời gian toàn bộ đồ án).
* **Số liệu chứng minh không cần sửa cửa sổ Stage 3–4 (§6.2)**:
  * Plan gốc lo ngại nốt lớn bị chia cắt bởi ranh giới cửa sổ nên đề xuất Shifted Window.
  * Tuy nhiên, số liệu Bước 0 chứng minh: Nốt trung bình ($10\text{--}20\text{ mm}$) đạt độ nhạy **$83.78\%$** ($31/37$ nốt), và nốt lớn ($\ge 20\text{ mm}$) đạt độ nhạy tuyệt đối **$100.0\%$** ($8/8$ nốt).
  * Mô hình nhận diện nốt lớn rất hoàn hảo. Điểm nghẽn duy nhất nằm ở việc nốt nhỏ $< 8\text{ mm}$ bị co lại thành $< 1\text{ pixel}$ ở tầng $H/4$ của decoder, nơi mà nhánh `--hr-skip` ($H/2, H/1$) giải quyết trực tiếp.
  * $\Rightarrow$ **Kết luận**: Đưa 448 lên làm kích thước chuẩn và tập trung vào `--hr-skip` là hoàn toàn đúng đắn.

---

### 3.4 Hợp lý #4: Giảm `pos_weight = 2.0` và nâng `--neg-ratio 0.5` (Đề xuất 2) ⭐⭐⭐

* **Số liệu chứng minh hiện tượng bão hòa xác suất (Logits Saturation)**:
  * Tại Bước 0: Ngưỡng $Th=0.5$ có $\text{Precision} = 0.4117 \approx \text{Recall} = 0.4039$, nhưng $\text{FP/ca} = 2.97$ (gấp 3 lần chuẩn lâm sàng).
  * Khi quét ngưỡng từ $Th=0.3$ lên tận $Th=0.7$, lượng dương tính giả **hầu như không suy chuyển** ($3.24 \rightarrow 2.80\text{ FP/ca}$) và $\text{Dice}_{3D}$ đi ngang ($0.3749 \rightarrow 0.3770$).
  * Điều này chứng minh các cụm FP đang bị phạt bởi `pos_weight = 4.0` đẩy lệch xác suất lên mức rất cao ($> 0.7$).
  * Khi hạ `pos_weight` xuống $2.0$, điểm cân bằng xác suất Bayes chuyển từ $p^* = \frac{1}{1+4} = 0.20$ về $p^* = \frac{1}{1+2} = 0.33$, giúp mạng tự tin triệt tiêu các đốm mô mềm không rõ ràng.
  * $\Rightarrow$ **Kết luận**: Hạ `pos_weight = 2.0` kết hợp `--neg-ratio 0.5` là bắt buộc để kéo FP/ca về $< 1.0$.

---

### 3.5 Hợp lý #5: Bộ 3 chống Overfitting `weight_decay = 0.02`, `EMA = 0.999`, `--aug-level 2` (Đề xuất 5) ⭐⭐

* **Số liệu chứng minh khoảng cách Overfitting $13\%$**:
  * Tại Epoch 42: $\text{Train Dice} = 0.8701$, trong khi $\text{Best Val Dice} = 0.7456 \Rightarrow$ **Chênh lệch tổng quát hóa lên tới $12.45\%\text{--}13.07\%$**.
  * $\text{Val Dice}$ giữa các epoch dao động nhiễu: Epoch 26 ($0.7262$) $\rightarrow$ Epoch 27 ($0.7396$) $\rightarrow$ Epoch 30 ($0.7456$) $\rightarrow$ Epoch 31 ($0.7391$).
  * Việc áp dụng `EMA = 0.999` giúp trung bình hóa trọng số qua $\sim 1,000$ bước cập nhật ($\approx 1.5$ epoch), giúp checkpoint chọn ra có tính bền vững cao nhất khi suy luận 3D.
  * $\Rightarrow$ **Kết luận**: Giữ nguyên toàn bộ thiết kế chống overfit của Plan gốc.

---

## 4. LỘ TRÌNH THỰC NGHIỆM 5 BƯỚC HOÀN THIỆN (FINAL REFINED ROADMAP)

Để vừa bảo đảm tính chặt chẽ của phương pháp khoa học cho đồ án CT208E, vừa khắc phục triệt để các rủi ro kỹ thuật đã phân tích, toàn bộ kế hoạch được chuẩn hóa thành **5 Phiên chạy chiến lược (Strategic Runs)**:

```
[Run 0: Baseline R0] (ĐÃ XONG)
       │
       ▼ (Áp dụng Kích thước 448 + Sửa Dữ liệu & Hàm Loss)
[Run 1: R1_Data_Loss]
       │
       ▼ (Thêm Regularization & Augmentation chống Overfit 13%)
[Run 2: R2_Regularization]
       │
       ▼ (Thêm Kiến trúc Skip độ phân giải cao H/2, H/1)
[Run 3: R3_Architecture_HR_Skip]
       │
       ▼ (Chạy Nghiệm thu Toàn bộ 1,010 Bệnh nhân LIDC-IDRI)
[Run 4: R4_Final_1010]
```

### Bảng cấu hình chi tiết 5 Runs:

| Phiên chạy (Run) | Cấu hình & Tham số kích hoạt | Mục trong Plan | Mục tiêu khoa học & Kỳ vọng định lượng |
| :--- | :--- | :---: | :--- |
| **Run 0 (`R0_Baseline`)**<br>*(Đã hoàn thành)* | • Ảnh $512$, `batch_size = 8`, `min_train_area = 20`, `neg_ratio = 0.25`, `pos_weight = 4.0`, `wd = 1e-4`. | §2 | • **Đã đo lường xong**:<br>  - $\text{Val Dice}_{2D} = 0.7456$, $\text{Dice}_{3D} = 0.3771$, $\text{FP/ca} = 2.91$.<br>  - Độ nhạy nốt $< 5\text{ mm}$: $3.39\%$; nốt $5\text{--}10\text{ mm}$: $57.58\%$. |
| **Run 1 (`R1_Data_Loss`)**<br>*(Sửa Dữ liệu & Loss)* | • `DUNEAI_IMAGE_SIZE=448 --batch-size 12`<br>• `--min-train-area 5.0 --neg-ratio 0.5`<br>• Vá ca không u: `min_neg_per_case = 4`<br>• `--loss-params '{"bce_weight": 0.3, "dice_weight": 0.7, "pos_weight": 2.0}'`<br>• `--epochs 40 --patience 10` *(giữ nguyên wd cũ 1e-4, chưa EMA)* | §3.1, §3.3, §5.3 | • **Mục tiêu**: Cô lập đo lường tác động của việc sửa dữ liệu nốt nhỏ và cân bằng lại hàm loss.<br>• **Kỳ vọng**: Khắc phục lỗi mù nốt nhỏ (G3, G4) và giảm FP trên ca rỗng (G2); $\text{Dice}_{3D}$ tăng lên vùng **$0.48 - 0.53$**; $\text{FP/ca}$ giảm xuống $\approx 1.5 - 1.8$. |
| **Run 2 (`R2_Regularization`)**<br>*(Chống Overfit & Ổn định)* | • Kế thừa toàn bộ cấu hình **Run 1**<br>• Thêm `--weight-decay 0.02` (4 param groups)<br>• Thêm `--ema 0.999`<br>• Thêm `--aug-level 2` (Affine xoay/scale/dịch $\pm 32\text{ px}$ + Gaussian jitter). | §3.4, §5.1, §5.2 | • **Mục tiêu**: Thu hẹp khoảng cách Overfitting $13\%$ giữa Train và Val; làm mượt trọng số kiểm định qua EMA.<br>• **Kỳ vọng**: Khoảng cách Train-Val thu hẹp còn $< 6\%$; $\text{FP/ca}$ giảm xuống $\le 1.1$; $\text{Dice}_{3D}$ tăng lên vùng **$0.53 - 0.57$**. |
| **Run 3 (`R3_HR_Skip`)**<br>*(Nâng cấp Kiến trúc)* | • Kế thừa toàn bộ cấu hình **Run 2**<br>• Thêm `--hr-skip` (Nhánh giải mã $H/2$ từ stem và $H/1$ từ ảnh gốc, tuân thủ Rule R1).<br>• *(Khởi tạo)*: `--init-weights` từ Run 2. | §3.6, §6.1 | • **Mục tiêu**: Bổ sung chi tiết biên độ nét cao cho nốt $< 8\text{ mm}$ thay vì phóng đại bilinear $\times 4$.<br>• **Kỳ vọng**: Độ nhạy nốt $< 5\text{ mm}$ đạt $> 45\%$; độ nhạy nốt $5\text{--}10\text{ mm}$ đạt $> 75\%$; $\text{Dice}_{3D}$ đạt đỉnh **$0.57 - 0.62$**. |
| **Run 4 (`R4_Final_1010`)**<br>*(Nghiệm thu toàn bộ dữ liệu)* | • Cấu hình tối ưu của **Run 3** chạy trên toàn bộ **$1,010$ bệnh nhân LIDC-IDRI** (`--epochs 45 --patience 12`, tách `--test-split 0.10`). | §5.6 | • **Mục tiêu**: Báo cáo kết quả chung cuộc của đề tài trên cả tập Validation ($75$ ca cố định) và tập Test độc lập ($101$ ca chưa từng thấy). |

---

## 5. KẾT LUẬN & HÀNH ĐỘNG TIẾP THEO

Bản tổng hợp này đã giải quyết trọn vẹn sự xung đột giữa Plan gốc và bản Đề xuất mới:
1. **Loại bỏ các rủi ro**: Không gộp quá nhiều biến vào Run 1; bỏ con số 16 hardcode; giữ đủ số epoch để mạng không underfitting; bảo toàn Rule R1.
2. **Khai thác tối đa các phát hiện đắt giá của Bước 0**: Áp dụng kích thước $448$; hạ ngưỡng nốt nhỏ $5\text{ mm}^2$ và hậu xử lý $6\text{ mm}^2$; hủy Run R5 lỗi nhãn; hạ `pos_weight = 2.0`.

File tài liệu này được lưu trữ cố định tại [`docs/FINE-TUNING-DOCS/STEP0_PROPOSALS_CRITIQUE_AND_FINAL_PLAN.md`](STEP0_PROPOSALS_CRITIQUE_AND_FINAL_PLAN.md) làm kim chỉ nam chính thức trước khi bắt đầu chỉnh sửa code và tiến hành huấn luyện Run 1.
