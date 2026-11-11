# ĐỀ XUẤT ĐIỀU CHỈNH KẾ HOẠCH CẢI TIẾN MÔ HÌNH UM-LOAN (MAMBA-UNET) DỰA TRÊN DỮ LIỆU CHẨN ĐOÁN BƯỚC 0

* **Tài liệu đối chiếu**: `MAMBA_UNET_REVIEW_AND_FIXES.md`, `MAMBA_UNET_SYSTEM_REPORT.md`
* **Dữ liệu bằng chứng**: `STEP0_FULL_METRICS_SUMMARY.md`, `STEP0_MEASUREMENT_AND_DIAGNOSTIC_REPORT.md`, `sweep_r0_patients.csv`, `sweep_r0_nodules.csv`
* **Mô hình cơ sở (Baseline $R_0$)**: `mamba_colab_500item_25epoch_extra_test/best_model.pth` (`mamba_vision_T`, $500$ ca LIDC-IDRI: $425$ Train / $75$ Val)

---

## 1. TỔNG QUAN CHẨN ĐOÁN BƯỚC 0: NHỮNG CON SỐ BIẾT NÓI

Trong quá trình huấn luyện 2D trên các lát cắt chọn lọc (`slices.csv`), mô hình đạt chỉ số rất khả quan với $\text{Train Dice}_{2D} = 0.8701$ (tại Epoch 42) và $\text{Best Val Dice}_{2D} = 0.7456$ (tại Epoch 30). Tuy nhiên, khi đưa vào quy trình suy luận trượt toàn khối 3D trên toàn bộ $75$ bệnh nhân tập Validation ($203$ nốt u Ground Truth), chỉ số $\text{Mean Dice}_{3D}$ giảm xuống mức **$0.3768$** (ở ngưỡng mặc định $Th = 0.5$) và đạt cực đại **$0.3771$** (ở ngưỡng tối ưu $Th = 0.6$).

Khi bóc tách dữ liệu từ tệp `sweep_r0_patients.csv` và `sweep_r0_nodules.csv` tại ngưỡng $Th = 0.6$, nguyên nhân của sự sụt giảm này được định vị chính xác qua **bảng phân tích 5 nhóm bệnh nhân** dưới đây:

### 1.1 Bảng phân rã hiệu năng 75 bệnh nhân Validation theo nhóm lâm sàng ($Th = 0.6$)

| Mã nhóm | Đặc điểm nhóm bệnh nhân | Số ca ($N=75$) | Tỷ trọng | $\text{Mean Dice}_{3D}$ | $\text{Mean Precision}_{3D}$ | $\text{Mean Recall}_{3D}$ | $\text{FP/ca}$ trung bình | Bản chất kỹ thuật gây sụt giảm điểm số |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Nhóm G1** | Không có u ($n_{\text{gt}} = 0$), đoán sạch ($\text{FP} = 0$) | $3$ ca | $4.0\%$ | **$1.0000$** | $1.0000$ | $1.0000$ | $0.00$ | Mô hình dự đoán hoàn hảo trên phổi rỗng (`0224, 0342, 0455`). |
| **Nhóm G2** | Không có u ($n_{\text{gt}} = 0$), báo động giả ($\text{FP} > 0$) | $8$ ca | $10.7\%$ | **$0.0000$** | $0.0000$ | $0.0000$ | $3.63$ | Khi $\text{GT} = 0$, chỉ cần $1$ cụm dương tính giả ($\ge 20\text{ mm}^2$) xuất hiện là công thức $\text{Dice}_{3D}$ lập tức rơi từ $1.0$ về $0.0$ (`0189, 0218, 0253, 0306, 0322, 0391, 0417, 0441`). |
| **Nhóm G3** | Chỉ chứa các nốt u siêu nhỏ ($A_{\max} < 20\text{ mm}^2$, $d < 5.05\text{ mm}$) | $8$ ca | $10.7\%$ | **$0.0230$** | $0.0195$ | $0.0279$ | $2.50$ | $7/8$ ca nhận $\text{Dice}_{3D} = 0.0000$ (`0038, 0178, 0212, 0247, 0280, 0317, 0328`) do mô hình chưa từng được học nốt $< 20\text{ mm}^2$ và bị bộ lọc hậu xử lý xóa bỏ. |
| **Nhóm G4** | Có nốt u $\ge 20\text{ mm}^2$ nhưng bỏ sót hoàn toàn ($n_{\text{det}} = 0$) | $9$ ca | $12.0\%$ | **$0.0000$** | $0.0000$ | $0.0000$ | $3.67$ | $100\%$ nốt u của $9$ ca này (`0093, 0165, 0200, 0291, 0370, 0371, 0376, 0399, 0405`) nằm ở vùng biên $d \in [5.05, 8.14]\text{ mm}$ ($A_{\max} \in [20, 52]\text{ mm}^2$). |
| **Nhóm G5** | Có nốt u $\ge 20\text{ mm}^2$ và phát hiện được $\ge 1$ nốt ($n_{\text{det}} \ge 1$) | $47$ ca | $62.7\%$ | **$0.5341$** *(Median: $0.5517$)* | $0.5858$ | $0.5653$ | $2.91$ | Trong $47$ ca này: $7$ ca không dính $\text{FP}$ đạt $\text{Mean Dice}_{3D} = \mathbf{0.6159}$ ($\text{Precision} = 0.9068$), còn $40$ ca dính $\text{FP}$ bị kéo tụt xuống $\text{Mean Dice}_{3D} = \mathbf{0.5198}$ ($\text{Precision} = 0.5296$). |

> **Kết luận từ bảng phân rã**: Kiến trúc UM-LOAN hoạt động rất tốt khi đã bắt được khối u (top $25\%$ bệnh nhân nhóm G5 đạt $\text{Dice}_{3D}$ từ $0.7685$ đến $0.9228$). Điểm trung bình toàn tập bị kéo xuống $0.3771$ là do **$24/75$ ca ($32.0\%$) bị điểm liệt $0.0000$** thuộc 3 nhóm lỗi: báo động giả trên ca không u (G2), mù nốt siêu nhỏ $< 5\text{ mm}$ (G3), và bỏ sót nốt vùng giáp ranh $5\text{--}8\text{ mm}$ (G4).

---

## 2. BẢNG ĐỐI CHIẾU TOÀN DIỆN VỚI KẾ HOẠCH `MAMBA_UNET_REVIEW_AND_FIXES.md`

Bản kế hoạch `MAMBA_UNET_REVIEW_AND_FIXES.md` được xây dựng trước khi có số liệu Bước 0 nên chứa nhiều nhánh điều kiện chờ kiểm chứng. Bảng dưới đây đối chiếu từng mục trong kế hoạch gốc với kết quả thực nghiệm Bước 0:

| Mục trong Plan gốc | Nội dung dự kiến trong `MAMBA_UNET_REVIEW_AND_FIXES.md` | Trạng thái đối chiếu từ Bước 0 | Quyết định điều chỉnh cho Kế hoạch |
| :--- | :--- | :---: | :--- |
| **§3.1 [P1]** | Hạ `--min-train-area` từ $20$ xuống $5\text{ mm}^2$ lúc dựng chỉ mục train, **nhưng giữ $20\text{ mm}^2$ ở hậu xử lý**. | ⚠️ **MÂU THUẪN NGHIÊM TRỌNG** | **Phải sửa ngay**: Nếu giữ $20\text{ mm}^2$ ở hậu xử lý 3D, toàn bộ $59/203$ nốt u $< 5\text{ mm}$ ($A_{\max} < 19.63\text{ mm}^2$) vẫn sẽ bị xóa sạch $100\%$ khi suy luận dù mô hình có học được. Phải hạ đồng bộ cả ngưỡng hậu xử lý (Xem Đề xuất 1). |
| **§3.1 [P1]** | Tăng `--neg-ratio` từ $0.25$ lên $0.5\text{--}1.0$ để giảm dương tính giả. | ⚠️ **THIẾU XỬ LÝ CA KHÔNG U** | **Phải bổ sung code**: Trong `offline_builder.py`, khi ca bệnh không có u (`len(pos_slices) == 0`), code đang gán cứng `num_neg_keep = 1` bất kể `--neg-ratio` bằng bao nhiêu. Cần sửa nhánh `else` này để cứu $8$ ca nhóm G2 (Xem Đề xuất 2). |
| **§3.2 [P3] & Run R5** | Thêm cờ `--tumor-union-p` và chạy Run R5 với `--tumor-union-p 0.0` (không hợp mask u vào mask phổi lúc train). | ❌ **LỖI LOGIC & KHÔNG CẦN THIẾT** | **Loại bỏ Run R5 & sửa logic**: Kết quả `check_cache.py` cho thấy $\text{Median} = 1.000, p_5 = 0.994$ (chỉ $8/449$ ca $< 0.90$). Hơn nữa, đặt `--tumor-union-p 0.0` mà không cắt `center_mask` sẽ ép mô hình học nhãn $u = 1$ trên nền không khí $-1000\text{ HU}$ (Xem Đề xuất 3). |
| **§4 (Hàm Loss) & Run R8** | Chờ kết quả `eval_sweep`: Giảm `pos_weight` xuống $2.0 / 1.0$ nếu $\text{FP}$ lớn; HOẶC đổi sang `focal_tversky` nếu $\text{Recall} \ll \text{Precision}$. | ✅ **ĐÃ CHỐT NHÁNH 1** | **Chọn giảm `pos_weight = 2.0`, bỏ `focal_tversky`**: Tại $Th=0.5$, $\text{Precision} (0.4117) \approx \text{Recall} (0.4039)$ nhưng $\text{FP/ca} = 2.97$ (gấp $3$ lần chuẩn lâm sàng). Đưa việc giảm `pos_weight` từ Run R8 lên ngay Run đầu tiên (Xem Đề xuất 2 & 6). |
| **§5.3 [P6] & Run R7** | Kiểm tra `check_cache.py` với kích thước $448$. Nếu không mất nhãn thì đặt `DUNEAI_IMAGE_SIZE=448` ở Run R7. | ✅ **XÁC NHẬN AN TOÀN $100\%$** | **Đưa từ Run R7 lên ngay Run 1**: Khảo sát chứng minh $100\%$ ca bệnh ($449/449$ ca, kể cả $21$ ca $\text{FOV} > 448$) có tỷ lệ mất nhãn $= 0.0\%$. Bật ngay từ đầu để giảm $2.25\times$ số token Stage 3 và tăng tốc train $20\text{--}25\%$ (Xem Đề xuất 4). |
| **§6.1 [P5] vs §6.2 [P6]** | §6.1 thêm nhánh `--hr-skip` ($H/2, H/1$). §6.2 bước 2–3 thêm Shifted Window hoặc Global Attention ở bottleneck nếu lỗi nằm ở nốt lớn. | ✅ **ĐÃ CHỐT ƯU TIÊN §6.1** | **Tập trung vào `--hr-skip` (§6.1), bỏ §6.2 bước 2–3**: Nốt trung bình và lớn ($\ge 10\text{ mm}$) đã đạt độ nhạy $83.8\%\text{--}100\%$. Điểm nghẽn nằm ở độ phân giải cho nốt nhỏ $< 8\text{ mm}$, nơi `--hr-skip` giải quyết trực tiếp (Xem Đề xuất 4). |
| **§3.4, §5.1, §5.2** | Sửa `weight_decay = 0.02` (4 param groups), thêm `EMA = 0.999`, và `aug-level 2` (Affine + Gaussian noise). | ✅ **HOÀN TOÀN PHÙ HỢP** | **Giữ nguyên thiết kế code trong Plan**: Giải quyết trực tiếp khoảng cách Overfitting $13\%$ và hiện tượng bão hòa xác suất Logits (Xem Đề xuất 5). |
| **§7 (Lộ trình Ablation)** | Chạy tuần tự 9 phiên (R1 $\rightarrow$ R9), hoặc rút gọn 6 phiên ($R_0 \rightarrow R_4 \rightarrow R_1+R_2 \rightarrow R_3 \rightarrow R_6 \rightarrow R_9$). | 🔄 **CẦN TÁI CẤU TRÚC** | **Rút gọn và sắp xếp lại thành 4 Runs chiến lược**: Đưa các bản vá lỗi dữ liệu/hàm loss/kích thước ảnh đã được Bước 0 chứng minh lên ngay Run 1, loại bỏ các Run thừa (Xem Đề xuất 6). |

---

## 3. CHI TIẾT 6 ĐỀ XUẤT ĐIỀU CHỈNH KẾ HOẠCH (KÈM SỐ LIỆU BẢO VỆ)

### ĐỀ XUẤT 1: Đồng bộ hóa ngưỡng diện tích huấn luyện (`--min-train-area 5.0`) VÀ hạ ngưỡng lọc hậu xử lý 3D (`--min-area 6.0` kèm điều kiện $\ge 2$ lát cắt)

#### 1. Nội dung điều chỉnh
* **Trong quá trình tạo chỉ mục huấn luyện (`§3.1`)**: Hạ ngưỡng `tumor_size_low_lim` (`--min-train-area`) từ $20.0\text{ mm}^2$ xuống **$5.0\text{ mm}^2$** (tương đương đường kính $d \ge 2.52\text{ mm}$). **Không hạ xuống $0.0\text{ mm}^2$** như ghi chú trong tệp `STEP0_MEASUREMENT_AND_DIAGNOSTIC_REPORT.md`.
* **Trong quá trình hậu xử lý và đánh giá 3D (`postprocess3d.py`, `evaluate.py`, `eval_sweep.py`)**: Bãi bỏ quy định *"giữ $20\text{ mm}^2$ ở hậu xử lý"* tại tiêu đề mục `§3.1` của `MAMBA_UNET_REVIEW_AND_FIXES.md`. Hạ tham số `min_slice_area_mm2` (`--min-area`) từ $20.0\text{ mm}^2$ xuống **$6.0\text{ mm}^2$** (hoặc $7.0\text{ mm}^2$, tương ứng chuẩn lâm sàng nốt u $d \ge 3.0\text{ mm}$ của LIDC-IDRI), đồng thời nâng tham số `min_slices_span` từ $1$ lên **$2$ lát cắt** (hoặc `min_volume_mm3 = 10.0`).

#### 2. Nguyên nhân & Số liệu chứng minh
* **Bằng chứng 1 — Sự xóa sổ nhóm nốt $< 5\text{ mm}$ ($29.1\%$ tổng số nốt)**:
  Theo dữ liệu `sweep_r0_nodules.csv`, trong $203$ nốt u Ground Truth của tập Validation, có tới **$59$ nốt ($29.06\%$)** có đường kính tương đương $d < 5.0\text{ mm}$ (diện tích mặt cắt ngang cực đại $A_{\max} = \frac{\pi d^2}{4} < 19.63\text{ mm}^2$, trung bình đạt $10.8\text{ mm}^2$). Do bộ lọc `TUMOR_SIZE_LOW_LIM = 20.0` loại bỏ toàn bộ $2,081$ lát cắt có diện tích u $< 20.0\text{ mm}^2$ khỏi `slices.csv`, mô hình bị mù hoàn toàn ở phân khúc này với độ nhạy chỉ đạt **$3.39\%$ ($2/59$ nốt)**, kéo $7$ bệnh nhân nhóm G3 về $\text{Dice}_{3D} = 0.0000$.
* **Bằng chứng 2 — Hiệu ứng "Bờ vực $20\text{ mm}^2$" (The $20\text{ mm}^2$ Cliff) tàn phá nhóm nốt $5.0\text{--}7.0\text{ mm}$**:
  Khi phân tách chi tiết $99$ nốt u thuộc dải $[5, 10)\text{ mm}$ trong `sweep_r0_nodules.csv` tại $Th = 0.6$, độ nhạy không hề đồng đều mà sụp đổ ngay sát ngưỡng $20\text{ mm}^2$:

| Dải đường kính chi tiết | Khoảng diện tích $A_{\max}$ ($\text{mm}^2$) | Số nốt GT | Số nốt phát hiện ($Th=0.6$) | Độ nhạy (Sensitivity) | Nguyên nhân vật lý |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **$0.00 - 5.05\text{ mm}$** | $1.0 - 19.6\text{ mm}^2$ | $59$ | $2$ | **$3.39\%$** | Bị loại $100\%$ khỏi `slices.csv` lúc train VÀ bị xóa $100\%$ bởi `postprocess_3d_mask(20.0)` lúc eval. |
| **$5.05 - 6.00\text{ mm}$** | $20.0 - 28.3\text{ mm}^2$ | $27$ | $7$ | **$25.93\%$** | Chỉ có $1$ lát tâm đủ $\ge 20\text{ mm}^2$ để vào tập train; các lát biên bị vứt bỏ. Khi suy luận, nếu mô hình dự đoán hơi co biên xuống $18\text{--}19\text{ mm}^2$, hậu xử lý $20\text{ mm}^2$ lập tức xóa mất nốt u. |
| **$6.00 - 7.00\text{ mm}$** | $28.3 - 38.5\text{ mm}^2$ | $28$ | $14$ | **$50.00\%$** | Nốt u bắt đầu có $2\text{--}3$ lát cắt vượt ngưỡng $20\text{ mm}^2$, nhưng vùng rìa cực trên/dưới vẫn bị khuyết trong tập train. |
| **$7.00 - 8.00\text{ mm}$** | $38.5 - 50.3\text{ mm}^2$ | $18$ | $12$ | **$66.67\%$** | Phần lớn thể tích nốt được học, độ nhạy phục hồi dần. |
| **$8.00 - 10.00\text{ mm}$** | $50.3 - 78.5\text{ mm}^2$ | $26$ | $23$ | **$88.46\%$** | Thoát hoàn toàn khỏi "bờ vực $20\text{ mm}^2$", độ nhạy ngang ngửa nhóm u lớn ($83.8\%\text{--}100\%$). |

* **Bằng chứng 3 — Tại sao chọn `min_area = 5.0 - 6.0` mà không chọn `20.0` hay `0.0`**:
  * Nếu giữ `min_slice_area_mm2 = 20.0` ở hậu xử lý: Hệ thống tự động chặn đứng trần độ nhạy tối đa trên toàn tập ở mức $(203 - 59) / 203 = \mathbf{70.9\%}$.
  * Nếu hạ `--min-train-area` xuống `5.0` và `--min-area` hậu xử lý xuống `6.0` ($d \ge 2.76\text{ mm}$): Trong $203$ nốt GT, có **$189/203$ nốt ($93.1\%$)** có $A_{\max} \ge 6.0\text{ mm}^2$ (chỉ loại bỏ $14$ vi nốt $1\text{--}5\text{ pixel}$ vốn là nhiễu bất đồng giữa các bác sĩ đọc phim). Kết hợp `min_slices_span = 2` giúp lọc sạch các đốm nhiễu dương tính giả chỉ lóe lên trên $1$ lát cắt đơn lẻ mà vẫn bảo toàn toàn bộ các nốt u thực sự từ $3\text{ mm}$ trở lên.

---

### ĐỀ XUẤT 2: Vá lỗ hổng lấy mẫu âm tính cho bệnh nhân không có u ($n_{\text{gt}} = 0$) trong `offline_builder.py`, nâng `--neg-ratio 0.5` và hạ `pos_weight = 2.0`

#### 1. Nội dung điều chỉnh
1. **Sửa mã nguồn `common/data_pipeline/offline_builder.py` (dòng 118–122)**: Thay đổi nhánh `else` khi bệnh nhân không có lát cắt dương tính nào (`len(pos_slices) == 0`). Thay vì chỉ lấy đúng $1$ lát cắt âm tính duy nhất, hệ thống cần lấy mẫu tương đương với một ca có trung bình $16$ lát dương tính: `num_neg_keep = min(len(neg_candidates), max(1, int(round(16 * zero_sl_ratio))))`.
2. **Cấu hình tham số dữ liệu & hàm mất mát ngay từ Run đầu tiên**:
   * Nâng `--neg-ratio` từ $0.25$ lên **$0.5$** (hoặc $0.6$).
   * Hạ `pos_weight` trong `--loss-params` từ $4.0$ xuống **$2.0$** (`'{"bce_weight": 0.3, "dice_weight": 0.7, "pos_weight": 2.0}'`).

#### 2. Nguyên nhân & Số liệu chứng minh
* **Bằng chứng 1 — Lỗ hổng `num_neg_keep = 1` giết chết $72.7\%$ số ca không u trong tập Validation**:
  Trong $75$ ca Validation, có **$11$ ca hoàn toàn không có khối u ($n_{\text{gt}} = 0$)**. Ở kết quả Bước 0 ($Th = 0.6$), có tới **$8/11$ ca ($72.7\%$)** bị mô hình dự đoán nhầm từ $1$ đến $10$ cụm dương tính giả (trung bình $3.63\text{ FP/ca}$), khiến cả $8$ ca này nhận $\text{Dice}_{3D} = 0.0000$ (Nhóm G2). Nếu cứu được $8$ ca này về $\text{FP} = 0$ ($\text{Dice}_{3D} = 1.0000$), điểm $\text{Mean Dice}_{3D}$ của toàn bộ $75$ bệnh nhân sẽ lập tức tăng thêm $\frac{8.0}{75} = \mathbf{+0.1067}$ (từ $0.3771$ vọt lên **$0.4838$**) mà chưa cần tính đến bất kỳ cải tiến nào khác!
  * *Nguyên nhân trong code*: Tại `offline_builder.py` (Tệp 27), dòng 121 viết:
    ```python
    if len(pos_slices) > 0:
        num_neg_keep = max(1, int(round(len(pos_slices) * zero_sl_ratio))) if zero_sl_ratio > 0 else 0
    else:
        num_neg_keep = 1 if len(neg_candidates) > 0 else 0
    ```
    Do nhánh `else` cố định bằng $1$, một bệnh nhân khỏe mạnh có $250$ lát cắt phổi chỉ đóng góp đúng **$1$ lát cắt** vào tập huấn luyện, bất chấp bạn tăng `--neg-ratio` lên bao nhiêu. Mô hình hoàn toàn thiếu khả năng nhận diện cấu trúc mạch máu rốn phổi và cơ hoành của những bệnh nhân không có u.
* **Bằng chứng 2 — Sự mất cân bằng giữa $75,620$ lát âm tính bị bỏ và hiện tượng bão hòa Logits do `pos_weight = 4.0`**:
  * Lúc huấn luyện, `ZERO_SL_RATIO = 0.25` đã loại bỏ **$75,620$ lát cắt âm tính dư thừa**, chỉ giữ lại $1,321$ lát âm tính trên tổng số $6,116$ lát ($21.6\%$). Nhưng khi suy luận 3D, mô hình phải quét qua $100\%$ thể tích phổi (nơi $> 95\%$ số lát cắt là âm tính).
  * Kết hợp với `pos_weight = 4.0` phạt lỗi bỏ sót nặng gấp $4$ lần lỗi báo động giả, mô hình bị đẩy vào trạng thái **bão hòa xác suất (Overconfident Logits)**: khi quét ngưỡng từ $Th = 0.3$ lên $Th = 0.7$, lượng dương tính giả hầu như không giảm ($3.24 \rightarrow 2.80\text{ FP/ca}$) và $\text{Dice}_{3D}$ đi ngang ($0.3749 \rightarrow 0.3770$). Điều này chứng minh các cụm $\text{FP}$ đang có xác suất dự đoán $> 0.7$, bắt buộc phải hạ `pos_weight` xuống $2.0$ và nâng `--neg-ratio` lên $0.5\text{--}0.6$ để cân bằng lại mặt phẳng tối ưu.

---

### ĐỀ XUẤT 3: Hủy bỏ phiên chạy `--tumor-union-p 0.0` (Run R5) và sửa lỗi xung đột nhãn tại Mục `§3.2`

#### 1. Nội dung điều chỉnh
* **Loại bỏ Run R5 (`--tumor-union-p 0.0`)** khỏi kế hoạch thực nghiệm ở Mục `§7`. Giữ mặc định `--tumor-union-p 1.0` trong suốt quá trình huấn luyện.
* **Sửa lỗi logic trong code đề xuất Mục `§3.2` (`dataset.py`)**: Nếu người dùng truyền `--tumor-union-p < 1.0` làm `use_union = False`, bắt buộc phải thêm dòng `center_mask = center_mask * (keep[self.half_k : self.half_k + 1] > 0).astype(np.float32)` để xóa phần nhãn khối u đã bị mặt nạ phổi che thành $-1000\text{ HU}$.
* **Giải pháp nhẹ cho 8 ca có nốt u bám màng phổi**: Trong hàm `lung_keep_mask` (`medical_prep.py` dòng 367), thêm tùy chọn giãn nhẹ `lung_mask` bằng hình thái học (`cv2.dilate` với kernel $5 \times 5$, $1$ vòng lặp — mở rộng biên phổi thêm $2\text{ mm}$) cho cả pha Train và Val/Inference.

#### 2. Nguyên nhân & Số liệu chứng minh
* **Bằng chứng từ `check_cache.py`**: Khảo sát trên $n = 449$ bệnh nhân cho thấy tỷ lệ bảo tồn khối u trong mặt nạ phổi đạt trung vị $\text{Median} = \mathbf{1.000}$ và phân vị thứ 5 đạt $p_5 = \mathbf{0.994}$ (tức $95\%$ bệnh nhân giữ được $\ge 99.4\%$ thể tích u). Chỉ có $8/449$ ca ($1.78\%$) có tỷ lệ $< 0.90$, và **cả $5$ ca tệ nhất (`0339, 0067, 0454, 0259, 0074`) đều nằm ở tập Train, không có ca nào nằm trong $75$ ca tập Validation**. Do đó, giả thuyết P3 không phải là nguyên nhân gây ra mức $\text{Dice}_{3D} = 0.3771$ trên tập Val.
* **Bằng chứng lỗi rò rỉ ngược trong code Mục `§3.2`**:
  Trong `dataset.py` (Tệp 24, dòng 142–150):
  ```python
  keep = lung_keep_mask(lung_stack, tumor_mask=tumor_for_keep)
  img_stack = apply_lung_mask(img_stack, keep, fill=HU_FLOOR) # Đặt vùng ngoài keep về -1000 HU
  ...
  center_mask = (msk_stack[self.half_k : self.half_k + 1] > 0).astype(np.float32) # Nhãn giữ nguyên!
  ```
  Nếu chạy Run R5 với `--tumor-union-p 0.0` (`tumor_for_keep = None`), ở ca `LIDC-IDRI-0339` ($100\%$ u nằm ngoài mặt nạ phổi) và `0067` ($79.54\%$ u nằm ngoài mặt nạ phổi), `img_stack` tại vị trí khối u bị gán cứng về $-1000\text{ HU}$ (đen tuyền như không khí), nhưng `center_mask` vẫn bằng $1.0$. Mô hình sẽ bị phạt nặng vì không dự đoán không khí $-1000\text{ HU}$ là khối u, dẫn đến nhiễu loạn gradient.

---

### ĐỀ XUẤT 4: Kích hoạt ngay `DUNEAI_IMAGE_SIZE=448` (`§5.3`) từ đầu và ưu tiên tuyệt đối `--hr-skip` (`§6.1`) thay vì sửa cửa sổ Stage 3–4 (`§6.2`)

#### 1. Nội dung điều chỉnh
* Chuyển việc áp dụng `DUNEAI_IMAGE_SIZE=448` từ Run R7 lên **ngay Run 1** (áp dụng cho toàn bộ các phiên huấn luyện và đánh giá tiếp theo), kết hợp nâng `--batch-size` từ $8$ lên **$12$ hoặc $16$**.
* Lưu khóa `"image_size": IMAGE_SIZE` vào `checkpoint_meta()` trong `trainer.py` và tự động đọc lại trong `predict_patient_volume` / `evaluate.py` để tránh rủi ro lệch kích thước giữa lúc train ($448$) và lúc eval ($512$).
* **Tập trung triển khai `--hr-skip` (Mục `§6.1`)** để lấy đặc trưng độ phân giải cao $H/2$ ($224 \times 224$) và $H/1$ ($448 \times 448$). **Bỏ qua Mục `§6.2` bước 2 và 3** (Shifted Window và Global Attention ở bottleneck).

#### 2. Nguyên nhân & Số liệu chứng minh
* **Bằng chứng an toàn và hiệu năng của `IMAGE_SIZE = 448`**:
  * Kết quả `check_cache.py` xác nhận: Mặc dù có $21/449$ ca có kích thước gốc sau chuẩn hóa $1\text{ mm}$ lớn hơn $448\text{ mm}$, nhưng khi cắt tâm về $448 \times 448$, **số ca bị mất nhãn khối u là $0/449$ ca ($0.00\%$)** (vì phổi luôn nằm ở trung tâm lồng ngực, phần bị cắt ở rìa ngoài chỉ là mô mỡ, da và thành ngực).
  * Về mặt toán học trong `mamba_vision.py` (Tệp 04): Ở kích thước $512 \times 512$, đặc trưng Stage 3 có kích thước $32 \times 32$ không chia hết cho `window_size = 14`, buộc hàm `F.pad` phải đệm lên $42 \times 42 = \mathbf{1,764\text{ tokens}}$; Stage 4 ($16 \times 16$, `window_size = 7`) phải đệm lên $21 \times 21 = \mathbf{441\text{ tokens}}$. Khi chuyển sang $448 \times 448$, Stage 3 có kích thước $28 \times 28 = \mathbf{784\text{ tokens}}$ (khớp khít $2 \times 2$ cửa sổ $14$) và Stage 4 là $14 \times 14 = \mathbf{196\text{ tokens}}$ (khớp khít $2 \times 2$ cửa sổ $7$). Số lượng token qua SSM/Attention giảm **$2.25\text{ lần}$**, triệt tiêu hoàn toàn hiện tượng biên giả (padding boundary artifacts) và rút ngắn thời gian mỗi epoch từ $5\text{ phút }16\text{ giây}$ xuống còn khoảng **$3\text{ phút }50\text{ giây}$**.
* **Bằng chứng ưu tiên `--hr-skip` (`§6.1`) thay vì `§6.2` bước 2–3**:
  * Ở các nốt u đòi hỏi ngữ cảnh rộng (dải $10\text{--}20\text{ mm}$ và $\ge 20\text{ mm}$), mô hình hiện tại đã đạt độ nhạy rất cao là **$83.78\%$ ($31/37$ nốt)** và **$100.0\%$ ($8/8$ nốt)**. Điều này chứng tỏ việc chia cửa sổ ở Stage 3–4 không làm giảm khả năng nhận diện nốt lớn.
  * Ngược lại, trong `decoder.py` (Tệp 06), đặc trưng mịn nhất từ Encoder là `f1` ở tỷ lệ $H/4$ ($1\text{ pixel} = 4\text{ mm} \times 4\text{ mm} = 16\text{ mm}^2$), sau đó đi qua `nn.Upsample(scale_factor=4, mode="bilinear")`. Với một nốt u đường kính $4\text{ mm}$, toàn bộ khối u bị thu nhỏ lại thành **chưa đầy $1\text{ pixel}$ duy nhất** ở tầng $H/4$ trước khi phóng đại nội suy tuyến tính $\times 4$. Do đó, việc nối thêm nhánh Skip Connection tại $H/2$ (`stem`) và $H/1$ (`hr_stem`) ở Mục `§6.1` là điều kiện tiên quyết để phân đoạn chính xác biên giới các nốt u nhỏ $< 8\text{ mm}$.

---

### ĐỀ XUẤT 5: Kích hoạt đồng bộ bộ 3 chống Overfitting: Phân nhóm `weight_decay = 0.02` (`§3.4`), `EMA = 0.999` (`§5.1`) và `--aug-level 2` (`§5.2`)

#### 1. Nội dung điều chỉnh
* Triển khai đúng code mục `§3.4` (tách 4 nhóm tham số trong `trainer.py` để loại trừ `bias`, `BatchNorm`, `LayerNorm`, `A_log`, `D` khỏi suy giảm trọng số) và nâng `--weight-decay` từ $10^{-4}$ lên **$0.02$**.
* Triển khai lớp `_EMA` (`decay = 0.999`) ở mục `§5.1` để làm mượt trọng số kiểm định.
* Triển khai `--aug-level 2` (biến đổi Affine xoay $\pm 15^\circ$, co giãn $0.85\text{--}1.15$, tịnh tiến $\pm 32\text{ px}$ + nhiễu Gaussian/Contrast) ở mục `§5.2`.

#### 2. Nguyên nhân & Số liệu chứng minh
* **Bằng chứng từ nhật ký huấn luyện 42 Epoch**:
  * Từ Epoch 24 đến Epoch 42, $\text{Train Loss}$ tiếp tục giảm sâu từ $0.1265$ xuống $0.1024$ và $\text{Train Dice}$ tăng từ $0.8390$ lên **$0.8701$**, nhưng $\text{Val Dice}$ đi ngang quanh ngưỡng $0.736\text{--}0.745$ (chênh lệch tổng quát hóa $\text{Generalization Gap} = \mathbf{13.07\%}$).
  * Giữa các epoch liên tiếp, $\text{Val Dice}$ dao động nhiễu $\pm 0.008\text{--}0.015$ (ví dụ: Epoch 26 tụt xuống $0.7262$, Epoch 27 lên $0.7396$, Epoch 30 vọt lên $0.7456$ rồi Epoch 31 tụt về $0.7391$). Việc áp dụng `EMA = 0.999` giúp lấy trung bình trượt trọng số qua $\sim 1,000$ bước cập nhật ($\approx 1.5\text{ epoch}$), giúp checkpoint `best_model.pth` hội tụ ổn định và không bị phụ thuộc vào may rủi của một epoch đơn lẻ.

---

### ĐỀ XUẤT 6: Tái cấu trúc Lộ trình Thực nghiệm Mục `§7` từ 9 Runs xuống 4 Runs Chiến Lược

#### 1. Nội dung điều chỉnh
Trong Mục `§7` của `MAMBA_UNET_REVIEW_AND_FIXES.md`, kế hoạch ban đầu chia nhỏ thành $9$ phiên chạy ($R_1 \rightarrow R_9$, tốn khoảng $25\text{--}30\text{ giờ}$ GPU T4). Dựa trên kết quả chẩn đoán Bước 0, các thay đổi mang tính **sửa lỗi hiển nhiên** (hạ ngưỡng u $5\text{ mm}^2$, nâng tỷ lệ âm tính $0.5$, sửa lỗi ca không u, hạ `pos_weight = 2.0`, kích thước ảnh $448$, `weight_decay = 0.02`, `EMA = 0.999`) cần được **hợp nhất ngay vào Phiên chạy số 1 (`R1`)** để làm đường cơ sở mới (New Baseline), giúp rút ngắn toàn bộ kế hoạch xuống **4 phiên chạy (Runs)** rõ ràng, phục vụ trực tiếp cho bảng Ablation Study trong báo cáo đồ án CT208E:

#### 2. Bảng Lộ Trình Thực Nghiệm 4 Bước Mới (Thay thế Bảng Mục `§7` cũ)

| Phiên chạy (Run) | Tổ hợp tham số & Cải tiến kích hoạt | Cần code ở mục | Mục tiêu kiểm chứng & Kỳ vọng định lượng |
| :--- | :--- | :---: | :--- |
| **Run 0 (`R0_Baseline`)** *(Đã hoàn thành)* | • Ảnh $512$, `batch_size = 8`, `min_train_area = 20`, `neg_ratio = 0.25`, `pos_weight = 4.0`, `wd = 1e-4`. | §2 | • $\text{Val Dice}_{2D} = 0.7456$, $\text{Dice}_{3D} = 0.3771$, $\text{FP/ca} = 2.91$.<br>• Độ nhạy nốt $< 5\text{ mm}: 3.4\%$; nốt $5\text{--}10\text{ mm}: 56.6\%$. |
| **Run 1 (`R1_Data_Loss_Opt`)** *(Hợp nhất R1+R2+R4+R7+R8 cũ)* | • `DUNEAI_IMAGE_SIZE=448 --batch-size 12`<br>• `--min-train-area 5.0 --neg-ratio 0.5` *(kèm vá lỗi ca không u)*<br>• `--loss-params '{"bce_weight": 0.3, "dice_weight": 0.7, "pos_weight": 2.0}'`<br>• `--weight-decay 0.02 --ema 0.999`<br>• `--epochs 30 --patience 8` | §3.1, §3.3, §3.4, §5.1, §5.3 | • **Mục tiêu**: Khắc phục lỗi mất nốt nhỏ (G3, G4) và lỗi báo động giả trên ca không u (G2).<br>• **Kỳ vọng**: Tốc độ train tăng $\sim 25\%$; $\text{FP/ca}$ giảm từ $2.91$ xuống $< 1.2$; độ nhạy nốt $< 5\text{ mm}$ tăng từ $3.4\%$ lên $> 35\%$ (khi đo với `--min-area 6.0`); $\text{Dice}_{3D}$ tăng vọt lên vùng **$0.52 - 0.58$**. |
| **Run 2 (`R2_Aug_Level2`)** *(Tương ứng R3 cũ)* | • Kế thừa toàn bộ cấu hình **Run 1**<br>• Thêm `--aug-level 2` (Affine xoay/scale/dịch $\pm 32\text{ px}$ + Intensity/Gaussian jitter). | §5.2 | • **Mục tiêu**: Thu hẹp khoảng cách Overfitting $13\%$ giữa Train và Val; dịch chuyển ngẫu nhiên lưới cửa sổ Stage 3–4.<br>• **Kỳ vọng**: Giảm thêm $\text{FP/ca}$ xuống $\le 1.0$; $\text{Dice}_{3D}$ tăng thêm $+0.02 - 0.04$. |
| **Run 3 (`R3_HR_Skip`)** *(Tương ứng R6 cũ)* | • Kế thừa toàn bộ cấu hình **Run 2**<br>• Thêm `--hr-skip` (Nhánh giải mã độ phân giải cao $H/2$ và $H/1$).<br>• *(Tùy chọn)* Khởi tạo `--init-weights` từ `best_model.pth` của Run 2 với `--freeze-backbone-epochs 0`. | §3.6, §6.1 | • **Mục tiêu**: Khôi phục chi tiết không gian sắc nét cho nốt u nhỏ $< 8\text{ mm}$ thay vì upsample bilinear $\times 4$.<br>• **Kỳ vọng**: Độ nhạy nốt $< 5\text{ mm}$ và $5\text{--}10\text{ mm}$ đạt mức cao nhất; biên khối u (HD95) giảm mạnh; $\text{Dice}_{3D}$ tăng thêm $+0.02 - 0.04$. |
| **Run 4 (`R4_Final_1010`)** *(Tương ứng R9 cũ)* | • Cấu hình tốt nhất (Run 3) chạy trên **toàn bộ $1,010$ bệnh nhân LIDC-IDRI** (`--epochs 40 --patience 10`, tách `--test-split 0.10`).<br>• *(Nếu $\text{FP/ca}$ ở Run 3 vẫn $> 1.0$)*: Chạy `tools/add_hard_negatives.py` trước khi fit. | §5.6 | • **Mục tiêu**: Triệt tiêu hoàn toàn giới hạn mẫu nhỏ ($500$ ca đầu), báo cáo kết quả cuối cùng của hệ thống **UM-LOAN** trên cả tập Validation ($75$ ca cố định) và tập Test độc lập. |

---

## 4. PHỤ LỤC: CÁC ĐOẠN MÃ NGUỒN CẦN CẬP NHẬT BỔ SUNG VÀO PLAN

Dưới đây là $3$ đoạn mã bổ sung/hiệu chỉnh trực tiếp cho các điểm chưa có trong `MAMBA_UNET_REVIEW_AND_FIXES.md`:

### 4.1 Bản vá `common/data_pipeline/offline_builder.py` (Bổ sung vào Mục `§3.1`)
Sửa đoạn lấy mẫu âm tính (dòng 118–122) để các bệnh nhân không có khối u (`has_tumor == 0` trên toàn bộ ca chụp) được lấy đủ số lát cắt âm tính tỷ lệ thuận với `zero_sl_ratio`:

```python
        # Cân bằng lát cắt âm tính (Negative Sampling) theo ZERO_SL_RATIO
        if len(pos_slices) > 0:
            num_neg_keep = max(1, int(round(len(pos_slices) * zero_sl_ratio))) if zero_sl_ratio > 0 else 0
        else:
            # SỬA LỖI BƯỚC 0: Ca không có khối u (n_gt = 0) lấy số lát âm tính tương đương
            # một ca có trung bình 16 lát dương tính (vd: zero_sl_ratio=0.5 -> lấy 8 lát âm tính)
            neg_for_empty = max(1, int(round(16 * zero_sl_ratio))) if zero_sl_ratio > 0 else 0
            num_neg_keep = min(len(neg_candidates), neg_for_empty)
```

### 4.2 Bản vá an toàn nhãn trong `common/data_pipeline/dataset.py` (Bổ sung vào Mục `§3.2`)
Đảm bảo nếu không hợp `tumor_mask` vào mặt nạ phổi thì nhãn `center_mask` cũng không giữ lại các pixel đã bị xóa thành $-1000\text{ HU}$:

```python
        if use_lung:
            img_stack, msk_stack, lung_stack = loaded
            use_union = (self.split == "train") and (random.random() < self.tumor_union_p)
            tumor_for_keep = msk_stack if use_union else None
            keep = lung_keep_mask(lung_stack, tumor_mask=tumor_for_keep)
            img_stack = apply_lung_mask(img_stack, keep, fill=HU_FLOOR)
            if not use_union:
                # Đồng bộ nhãn với vùng phổi giữ lại để tránh học mask=1 trên nền -1000 HU
                msk_stack = msk_stack & (keep > 0).astype(msk_stack.dtype)
        else:
            img_stack, msk_stack = loaded
```

### 4.3 Lệnh chạy chuẩn cho Phiên `R1_Data_Loss_Opt` và Đánh giá 3D đa ngưỡng diện tích
1. **Lệnh huấn luyện Run 1**:
```bash
DUNEAI_IMAGE_SIZE=448 DUNEAI_MAMBA_BACKEND=cuda python -u -m mamba_unet.train_main train \
  --cache-dir /content/data/cache_1mm \
  --backbone mamba_vision_T \
  --pretrained \
  --freeze-backbone-epochs 3 \
  --backbone-lr-mult 0.1 \
  --activation relu \
  --loss dice_bce \
  --loss-params '{"bce_weight": 0.3, "dice_weight": 0.7, "pos_weight": 2.0}' \
  --min-train-area 5.0 \
  --neg-ratio 0.5 \
  --weight-decay 0.02 \
  --ema 0.999 \
  --epochs 30 \
  --patience 8 \
  --batch-size 12 \
  --lr 0.00012 \
  --scheduler cosine \
  --grad-clip 1.0 \
  --workers 2 \
  --seed 42 \
  --run-name 'um_loan_r1_data_loss_opt_448' \
  --tag 'ablation_r1'
```

2. **Lệnh đánh giá 3D sau khi train xong Run 1 (Quét với `--min-area 6.0` để đo khả năng bắt nốt nhỏ $\ge 3\text{ mm}$)**:
```bash
DUNEAI_IMAGE_SIZE=448 DUNEAI_MAMBA_BACKEND=cuda python tools/eval_sweep.py \
  --cache-dir /content/data/cache_1mm \
  --checkpoint mamba_unet/runs/um_loan_r1_data_loss_opt_448/best_model.pth \
  --backbone mamba_vision_T \
  --split val \
  --thresholds 0.4 0.5 0.6 0.7 \
  --min-area 6.0 \
  --out-prefix sweep_r1_min6mm2
```