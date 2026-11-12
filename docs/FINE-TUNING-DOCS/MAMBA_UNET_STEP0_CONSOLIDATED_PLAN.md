# Đối chiếu Gemini 4 Argon và Gemini 3.8 Flash sau Bước 0, và hướng chỉnh `MAMBA_UNET_REVIEW_AND_FIXES.md`

*Viết tắt: **G4** = Gemini 4 Argon (`STEP0_PLAN_ADJUSTMENT_PROPOSAL.md`), **G3.8** = Gemini 3.8 Flash (`STEP0_PROPOSALS_CRITIQUE_AND_FINAL_PLAN.md`), **Plan gốc** = `MAMBA_UNET_REVIEW_AND_FIXES.md`. Mọi con số đã được tính lại trực tiếp từ `sweep_r0_patients.csv` và `sweep_r0_nodules.csv` (75 ca val, 203 nốt GT, ngưỡng 0.6 nếu không ghi khác).*

*Quy ước nhãn: **Ủng hộ** = tôi đồng ý; **Một phần** = đồng ý mục tiêu hoặc kết luận nhưng không đồng ý lý lẽ hoặc cách làm; **Phản bác** = tôi không đồng ý. File này **chưa sửa Plan gốc**. Mỗi điểm có phần "Chốt" mô tả sẽ sửa Plan gốc ở mục nào và sửa ra sao, để bạn duyệt trước.*

---

## 0. Bảng tổng hợp nhanh

| # | Điểm | G4 | G3.8 | Chốt cho Plan gốc | Độ chắc |
|---|---|---|---|---|---|
| 1 | Đồng bộ ngưỡng diện tích: train 5 mm², hậu xử lý 6 mm² | Ủng hộ | Ủng hộ | Sửa §3.1 (bỏ "giữ 20 mm²"), §4 | Cao |
| 2 | Lọc cụm nhỏ bằng `min_slices_span ≥ 2` | Phản bác | Phản bác (G3.8 đòi bật span ≥ 2) | Dùng lọc thể tích, quyết bằng R0b | Trung bình |
| 3 | Phạm vi áp dụng min-area 6 (chỉ từ Run 1) | Một phần | Phản bác | Đánh giá cả R0 bằng cùng bộ lọc (R0b) | Cao |
| 4 | Lấy mẫu âm tính (nhánh ca rỗng, số 16, `neg-ratio`) | Một phần | Một phần | Thêm §3.1b: lấy lại mẫu âm tính mỗi epoch | Cao (hướng), Trung bình (tham số) |
| 5 | `pos_weight 2.0` ngay Run 1 | Phản bác | Phản bác | §4: thành ablation đơn biến, có điều kiện | Trung bình |
| 6 | Hủy R5, vá nhãn khi `tumor-union-p < 1` | Ủng hộ | Ủng hộ | Sửa §3.2, bỏ R5, hạ P3 xuống thấp | Cao |
| 7 | `IMAGE_SIZE=448` ngay từ đầu | Ủng hộ (sai ước lượng thời gian) | Ủng hộ (lặp lại sai số đó) | Sửa §5.3 | Cao |
| 8 | Ưu tiên `--hr-skip`, bỏ §6.2; G3.8 bác sửa `decoder.py` | Ủng hộ (đặt sớm quá) | Ủng hộ §6.2; Phản bác phần decoder.py | Giữ §6.1, đặt sau R1; thêm test hồi quy | Cao |
| 9 | Bộ chống overfit (wd, EMA, aug 2) | Ủng hộ | Ủng hộ; Phản bác "wd ×200 nguy hiểm" | Giữ §3.4, §5.1, §5.2 | Cao |
| 10 | Gộp biến vào Run 1 và số lượng run | Một phần | Một phần | Sửa §7 | Trung bình |
| 11 | `--epochs` / `--patience` | Một phần | Ủng hộ kết luận, phản bác lý do | §4 và lệnh nền §7: 40/10 cho mọi run | Trung bình |
| 12 | Batch size và LR trong R1 | Phản bác | Một phần | §4: giữ bs 8, lr 1e-4 ở R1 | Trung bình |
| 13 | Khởi tạo nóng R3 từ R2 | Phản bác | Phản bác | §6.1(e), §7: không khởi tạo nóng trong dòng ablation | Cao |
| 14 | Dice dự kiến từng run | Phản bác | Phản bác | §7: đổi cột "Kỳ vọng" thành tiêu chí quyết định | Cao |
| 15 | "≤ 1 FP/ca là chuẩn lâm sàng", nguyên nhân FP, "nốt < 5 mm chưa từng học" | Phản bác | (không bàn, dùng lại) | Sửa lời văn §1, §2; thêm R0b | Cao |
| 16 | Cứu 8 ca G2 → Dice +0.107 | Một phần | (không bàn) | §2.1: đổi cách báo cáo metric | Cao |
| 17 | Bổ sung của tôi: R0b, val cố định, test, CI, ghép cặp | (không có) | (không có) | Thêm §2.3, §5.6 mở rộng | Cao |
| 18 | Lộ trình cuối | Một phần (4 run) | Một phần (5 run) | Thay bảng §7 | Trung bình |

**Ba chỗ cần bạn quyết định:** (a) có chạy R2 (`pos_weight 2`) hay không (phụ thuộc kết quả R0b); (b) `IMAGE_SIZE=448` gộp vào R1 hay chạy một control riêng; (c) ngân sách GPU: lộ trình đầy đủ 6 run hay lộ trình tiết kiệm 4 run (mục 3).

---

## 1. Số liệu nền đã kiểm lại

### 1.1 Tổng quan (ngưỡng 0.6)

| Chỉ số | Giá trị | Ghi chú |
|---|---|---|
| Dice 3D trung bình | **0.3771** | SD 0.349, SE 0.040, **CI 95% ±0.079** |
| Dice 3D chỉ trên 64 ca có GT | 0.3951 | 11 ca không GT kéo trung bình xuống |
| Ca không GT | 11 | 3 ca Dice = 1.0; **8 ca Dice = 0** (29 cụm FP, 3.63/ca) |
| Ca Dice = 0 | 24/75 | 8 ca không GT + **16/64 ca có GT (25%) bị bỏ sót hoàn toàn** |
| Độ nhạy theo nốt | 97/203 = 47.8% | |
| Độ nhạy nốt ≥ 5 mm | 95/144 = **66.0%** | CI 95% ±7.7% |
| Độ nhạy nốt < 5 mm | 2/59 = 3.4% | CI 95% ±4.6% |
| FP | 218 cụm | **9 ca chiếm 77 cụm (35.3%)** |
| Mức dùng dữ liệu lúc train | Dương tính 4795/6876 = 69.7% | **Âm tính 1321/76941 = 1.72%** |
| Phản ứng với ngưỡng | Chỉ **2 nốt** đổi trạng thái phát hiện khi Th 0.3 → 0.7 | FP tổng 243 → 210 (−13.6%) |

### 1.2 Độ nhạy theo diện tích lát lớn nhất A (1 px = 1 mm²)

| Dải A (mm²) | Số nốt | Phát hiện | Độ nhạy |
|---|---|---|---|
| < 6 | 12 | 0 | 0% |
| 6 – 10 | 9 | 0 | 0% |
| 10 – 20 | 38 | 2 | 5.3% |
| 20 – 28.3 | 27 | 7 | 25.9% |
| 28.3 – 38.5 | 28 | 14 | 50.0% |
| 38.5 – 50.3 | 18 | 12 | 66.7% |
| 50.3 – 78.5 | 26 | 23 | 88.5% |
| ≥ 78.5 | 45 | 39 | 86.7% |

Độ nhạy tăng đều theo A bắt đầu đúng tại mốc 20 mm².

### 1.3 Mức độ khớp với hai tài liệu

- Các số của G4 (nhóm G1–G5, bảng chi tiết theo đường kính, FP tập trung, 0.4838 nếu cứu 8 ca) khớp CSV.
- Lệch nhẹ: nốt có A ≥ 6 là **191/203 (94.1%)**, G4 ghi 189/203. Độ nhạy 5–10 mm tại Th = 0.6 là 56/99, không phải 57/99 (57 là giá trị ở Th ≤ 0.5).
- Dòng "spacing_z = 1.0 mm" trong báo cáo là dòng mẫu. `eval_sweep` lấy spacing thật từ `meta.json`.

### 1.4 Đính chính của tôi

Ở lượt trước tôi viết "âm tính chỉ chiếm 22% val (1321/6116)". Sai: con số đó tính trên cả 500 ca, không riêng val. Ở bản trước của file này tôi cũng ghi G3.8 R1 có đổi lr; thực tế bảng R1 của G3.8 chỉ nêu `--batch-size 12`, không nêu lr.

---

## 2. Từng điểm: lập trường, lý do, số liệu, chốt

### Điểm 1. Đồng bộ ngưỡng diện tích: train 5 mm², hậu xử lý 6 mm²

**Plan gốc nói gì.** §3.1: "hạ ngưỡng lát u khi dựng chỉ mục train xuống 5, *giữ 20 mm² ở hậu xử lý*". §4: `--min-train-area` 5.

**G4 (Đề xuất 1): Ủng hộ.** Đề xuất hạ cả hai ngưỡng, chọn 5 chứ không 0.

**G3.8 (§3.1): Ủng hộ.** Gọi đây là "phát hiện có giá trị sống còn".

**Ý kiến tôi: ủng hộ cả hai.** Đây cũng là lỗi của chính Plan gốc.
- **Số liệu.** Nốt có A ≥ 20: 144/203, nên giữ bộ lọc 20 mm² đặt trần độ nhạy ở **70.9%**. Nhóm A ∈ [6, 20) có **47 nốt (23.2% GT)**, chỉ bắt được 2.
- **Vì sao 5 mm² mà không 0.** Có 12 nốt A < 6 và không nốt nào được bắt. LIDC chỉ đánh dấu nốt ≥ 3 mm (d = 3 mm ứng với A ≈ 7.1 mm²), và GT ở đây là hợp của các reader nên các mảnh nhỏ hơn rất dễ là nhiễu bất đồng. `STEP0_EVALUATION_REPORT.md` (notebook tự sinh) gợi ý 0, G4 chọn 5 là đúng.
- **Hậu xử lý 6 mm²** giữ lại 191/203 nốt (94.1%).

**Chốt (Plan gốc cần sửa):**
- §3.1: đổi tiêu đề thành "hạ ngưỡng lát u khi dựng chỉ mục train; hậu xử lý hạ đồng bộ (xem Điểm 2, 3)". Xóa cụm "giữ 20 mm² ở hậu xử lý".
- §2.1 (`eval_sweep.py`): `--min-area` mặc định 20 chuyển thành danh sách quét (0, 6, 10, 20).
- §4, hàng `--min-train-area`: giữ 5, thêm lý do "12 nốt A < 6 nhiễu; d ≥ 3 mm ↔ A ≈ 7.1".
- §1: thêm P15 "bộ lọc hậu xử lý 20 mm² chặn trần độ nhạy ở 70.9%".

---

### Điểm 2. Lọc cụm nhỏ: `min_slices_span ≥ 2` hay lọc theo thể tích

**Plan gốc nói gì.** Không có (chỉ có lọc theo diện tích lát lớn nhất).

**G4 (Đề xuất 1): Phản bác.** Đề xuất nâng `min_slices_span` từ 1 lên 2 hoặc `min_volume_mm3 = 10`.

**G3.8 (§2.5): Phản bác.** Gọi `min_slices_span ≥ 2` là "bắt buộc phải kích hoạt".

**Ý kiến tôi: phản bác phần span; chấp nhận thể tích như một lựa chọn để thử.**
- **Lý do.** "2 lát" có ý nghĩa vật lý khác nhau giữa lát 0.6 mm và lát 2.5–5 mm. LIDC có độ dày lát từ khoảng 0.6 đến 5 mm. Với lát dày, nốt 3–5 mm thường chỉ nằm trên 1 lát và sẽ bị xóa oan.
- **Số liệu liên quan.** Thể tích 10 mm³ giữ được hình cầu từ khoảng 2.7 mm trở lên (nốt 3 mm có thể tích 14 mm³, nốt 2.5 mm là 8 mm³). Cả hai tài liệu đều chưa đo `span`. R0b ghi số lát của từng nốt GT để biết có bao nhiêu nốt thật chỉ nằm trên 1 lát.
- **Cần kiểm tra code.** G4 giả định `postprocess_3d_mask` có tham số `min_slices_span`. Tôi chưa thấy bằng chứng.

**Chốt:**
- §2.1: thêm tham số quét `--min-vols` (0, 10, 20 mm³), **không** thêm span.
- Quyết định giá trị `V_min` để sau R0b.

---

### Điểm 3. Phạm vi áp dụng min-area 6: chỉ từ Run 1 hay cả baseline

**G4: Một phần.** Đánh giá R1 bằng `--min-area 6.0` nhưng R0 vẫn bằng 20, nên chênh lệch lẫn bộ lọc với mô hình.

**G3.8 (§2.5): Phản bác.** Cho rằng hạ min-area xuống 6 sẽ đẩy FP lên "5–8 FP/ca", nên chỉ áp dụng cho mô hình từ Run 1.

**Ý kiến tôi: phản bác G3.8, ủng hộ một phần G4.**
- **Lý do.** Hạ min-area dùng cho **mọi** mô hình sẽ cho kết quả so được. Con số "5–8 FP/ca" là phỏng đoán: chỉ cần chạy R0 với min-area 6 là biết chính xác (R0b), không cần đoán.
- Cái G3.8 lo ngại có thật (FP tăng), nhưng cách xử lý là **đo**, không né.

**Chốt:**
- §7: thêm hàng R0b (đánh giá lại R0, không train) làm điều kiện đầu tiên. Mọi bảng so sánh về sau dùng R0b làm dòng baseline.
- §2.1: thay bằng `tools/eval_sweep_v2.py` (Phụ lục A).

---

### Điểm 4. Lấy mẫu âm tính

**Plan gốc nói gì.** §3.1: `--neg-ratio` 0.25 → 0.5–1.0, hard-negative mining ở §5.6.

**G4 (Đề xuất 2): Một phần.** Đúng mục tiêu, nhưng cách làm yếu:
- Trích đoạn `offline_builder.py` cho rằng ca không có u luôn lấy đúng 1 lát âm tính. Tôi không có file này, **chưa xác minh được**.
- Vá bằng `max(1, round(16 × ratio))` với số 16 là hằng số tự đặt.

**G3.8 (§2.2): Một phần.** Chỉ ra nghịch lý đúng (ca rỗng 8 lát, ca có 1 lát u chỉ 1 lát) và đề xuất `min_neg_per_case = 4`. Con số "4069 lát ÷ ~350 ca = 11.6" lấy từ `training_report.json` mà tôi không có, nên chưa kiểm được. Với tỷ lệ ca không u ở val (14.7%) tôi ước ra khoảng 11, nên chênh lệch giữa 16 và 11 không phải điểm quyết định.

**Ý kiến tôi: cả hai chạm vào triệu chứng nhưng bỏ sót nguyên nhân gốc.**
- **Số liệu quyết định.** Lúc train, mô hình chỉ thấy **1.72%** lát âm tính (1321/76941); lúc suy luận phải quét 100% lát phổi (khoảng 168 lát/ca). Tăng `neg-ratio` 0.25 → 0.5 chỉ đưa tỷ lệ đó lên khoảng 3–4%. `min_neg_per_case = 4` cho tối đa khoảng 1700 lát âm trên 425 ca, vẫn ở mức vài phần trăm.
- **Cách làm tốt hơn.** Giữ toàn bộ lát âm tính của tập train trong chỉ mục, và **mỗi epoch lấy mẫu lại ngẫu nhiên** `neg_ratio × số dương`. Chi phí mỗi epoch không đổi, độ phủ pool âm tính sau 40 epoch khoảng `1 − exp(−2.7k×40/65k) ≈ 81%` so với 1.7% hiện tại (Phụ lục B).
- **Val.** Lấy mẫu âm tính một lần, cố định theo seed, để val ổn định giữa các run.

**Chốt:**
- §3.1: thêm mục **3.1b** "Lấy lại mẫu âm tính mỗi epoch" (sampler Phụ lục B; `build_dataset_index` giữ toàn bộ âm tính của train; `--neg-ratio` hiểu là tỷ lệ **mỗi epoch**).
- §4, hàng `--neg-ratio`: 0.5 (cố định theo epoch). Thêm `min_neg_per_case` chỉ khi G3.8 §2.2 được xác minh trong code.
- §5.6: hard-negative mining giữ, đặt sau khi R1 có kết quả (R0 gần thuộc lòng tập train nên FP trên train sẽ ít hơn trên val).
- **Không** sửa nhánh ca rỗng bằng số 16.

---

### Điểm 5. `pos_weight 2.0` ngay Run 1

**Plan gốc nói gì.** §4: "Thử 2.0 / 1.0 nếu **precision ≪ recall**". §2.1: điều kiện đọc kết quả là "fp_comps lớn **và** precision ≪ recall".

**G4 (Đề xuất 2): Phản bác.** Đặt `pos_weight 2.0` vào R1, kỳ vọng FP/ca xuống dưới 1.2, lập luận bão hòa xác suất do `pos_weight = 4`.

**G3.8 (§3.4): Phản bác.** Ủng hộ G4, thêm phép tính Bayes: ngưỡng tối ưu `1/(1+w)` chuyển từ 0.20 sang 0.33.

**Ý kiến tôi: phản bác cả hai.**
- **Phần đúng.** Bão hòa xác suất có thật: chỉ 2/198 nốt đổi trạng thái giữa Th 0.3 và 0.7.
- **Phép tính của G3.8 chính là phản chứng.** Đầu ra BCE có trọng số liên hệ với xác suất thật bởi `p = p̃ / (w(1 − p̃) + p̃)`.

| Ngưỡng trên `p̃` (w = 4) | 0.3 | 0.5 | 0.7 |
|---|---|---|---|
| Xác suất thật tương ứng | 0.097 | 0.200 | 0.368 |

  Đổi w từ 4 sang 2 tại Th = 0.5 tương đương dịch ngưỡng của mô hình cũ lên khoảng **0.667**. Ở Th 0.6–0.7 mô hình cũ chỉ có 2.91–2.80 FP/ca. Theo ước lượng bậc nhất này, riêng việc hạ `pos_weight` giảm khoảng **0.1–0.15 FP/ca**, không thể đưa FP xuống dưới 1.2. (Thành phần Dice trong loss không chịu `pos_weight`, nên bão hòa còn do Dice loss và overfitting.)
- **Điều kiện của chính Plan gốc chưa thỏa.** Precision 0.4117 ≈ Recall 0.4039, không phải precision ≪ recall. G4 vẫn ghi "đã chốt nhánh giảm pos_weight" dù bảng của nó cho thấy P ≈ R.
- FP "tự tin" cần mẫu hình mới (Điểm 4, hard negative), không chỉ đổi trọng số.

**Chốt:**
- §4, hàng `pos_weight`: đổi thành "ablation đơn biến, chỉ chạy nếu R0b cho thấy FP có `max_prob` thấp đến trung bình hoặc kích thước nhỏ".
- §7 (R8 cũ): giữ làm run riêng, không gộp vào R1.
- §2.1: sửa bảng "Cách đọc kết quả": bỏ cụm "giảm pos_weight" khỏi dòng FP lớn khi P ≈ R.

---

### Điểm 6. Hủy R5 (`--tumor-union-p 0`) và vá nhãn

**Plan gốc nói gì.** §3.2: thêm `--tumor-union-p`; §7: R5 chạy `--tumor-union-p 0.0`.

**G4 (Đề xuất 3): Ủng hộ.**

**G3.8 (§3.2): Ủng hộ.**

**Ý kiến tôi: ủng hộ cả hai, và đây là lỗi thật của Plan gốc.**
- **Số liệu.** Trung vị `mask_in_lung_fraction` = 1.000, p5 = 0.994. Chỉ 8/449 ca (1.8%) có tỷ lệ dưới 0.9; 5 ca tệ nhất (0339, 0067, 0454, 0259, 0074) **đều không nằm trong val**. P3 không giải thích được Dice 3D thấp trên val.
- **Lỗi logic.** Với `tumor-union-p = 0`, ảnh bị gán −1000 HU ở vùng u nhưng `center_mask` vẫn bằng 1, nên mạng bị ép học "u trên nền không khí". Vá bằng `msk_stack & (keep > 0)` (G4 §4.2).
- Giãn mask phổi thêm 2 mm là ý phụ ưu tiên thấp.

**Chốt:**
- §3.2: thêm đoạn vá nhãn của G4 §4.2.
- §7: bỏ R5.
- §1: hạ P3 từ "Cao" xuống "Thấp" và ghi lý do.
- Mô tả "giãn mask phổi 2 mm" thêm vào §3.2 như tùy chọn thử sau cùng.

---

### Điểm 7. `IMAGE_SIZE = 448` ngay từ đầu

**Plan gốc nói gì.** §5.3: 448 là cải tiến mã nguồn; §7: chạy ở R7.

**G4 (Đề xuất 4): Ủng hộ quyết định, phản bác ước lượng thời gian.**

**G3.8 (§3.3): Ủng hộ quyết định, lặp lại sai số thời gian.**

**Ý kiến tôi: ủng hộ quyết định, phản bác ước lượng.**
- **Số liệu.** 0/449 ca mất nhãn khi cắt 448 (kể cả 21 ca có FOV lớn hơn 448). Token stage 3 giảm 42² = 1764 xuống 28² = 784 (**2.25 lần**), stage 4 từ 441 xuống 196, hết pad.
- **Sai số thời gian.** G4 tính epoch 5m16s → 3m50s chỉ dựa trên token, bỏ qua việc tập train **lớn lên** (hạ ngưỡng 5 mm² thêm lát dương tính, thêm lát âm tính). Số lát mỗi epoch tăng khoảng 1.5–1.7 lần (ước lượng), nên epoch 448 thực tế khoảng **6–7 phút**, mỗi run khoảng **4–4.5 giờ** (R0 là 3h38m).
- **Hạn chế kiểm tra.** Kiểm tra mất nhãn mới chạy trên 449 ca (trong 500), chưa chạy cho toàn bộ val.

**Chốt:**
- §5.3: thêm lưu `image_size` vào `checkpoint_meta` và đọc lại ở `evaluate.py`; chạy lại kiểm tra FOV cho đủ 500 ca; thay câu "nhanh hơn ~20%" bằng ước lượng "6–7 phút/epoch với tập train mới".
- §7: R1 gộp 448 và ghi rõ là "thay đổi hạ tầng" (nếu còn ngân sách, chạy một control R0 ở 448, 30 epoch).

---

### Điểm 8. `--hr-skip` ưu tiên, bỏ §6.2, và tranh luận về `decoder.py`

**Plan gốc nói gì.** §6.1: `--hr-skip` với cờ mặc định tắt; §6.2: shifted-window và global attention.

**G4 (Đề xuất 4): Ủng hộ về ưu tiên, nhưng đặt sớm quá.** Cho rằng `--hr-skip` giải quyết trực tiếp nốt < 8 mm.

**G3.8 (§3.3 và §2.4):** Ủng hộ bỏ §6.2; **Phản bác** G4 là "sửa `decoder.py` vi phạm Rule R1".

**Ý kiến tôi:**
- **Bỏ §6.2: ủng hộ cả hai.** Nốt ≥ 10 mm đạt 83.8–100%, không có bằng chứng lỗi do cửa sổ.
- **Ưu tiên `--hr-skip`: một phần.** Bảng 1.2 cho thấy độ nhạy tăng đều theo A từ mốc 20 mm², kiểu cắt cụt do train. Vì vậy chưa thể kết luận nốt 5–10 mm bị chặn bởi độ phân giải. `--hr-skip` chỉ đáng làm nếu độ nhạy 5–10 mm vẫn thấp **sau R1**.
- **G3.8 §2.4: phản bác.** G4 chỉ nói "tập trung `--hr-skip` (§6.1)". Thiết kế §6.1 có cờ `high_res_skip=False` mặc định tắt, khi tắt `state_dict` không đổi. Đây là phản bác vào thứ không được đề xuất. Cái cần thêm là một test hồi quy: tải checkpoint R0 với `strict=True` khi cờ tắt. (Tôi không xác minh được "Rule R1/`unet_base`" vì thiếu file.)

**Chốt:**
- §6.1: thêm test hồi quy; ghi rõ "chạy sau R1, chỉ khi độ nhạy 5–10 mm vẫn thấp".
- §6.2 bước 2–3: giữ trong tài liệu nhưng đánh dấu "không ưu tiên".
- §7: `--hr-skip` ở R4, không phải R6.

---

### Điểm 9. Bộ chống overfit (wd, EMA, aug 2) và tranh luận "wd ×200"

**Plan gốc nói gì.** §3.4 (wd 4 nhóm tham số), §5.1 (EMA), §5.2 (aug 2).

**G4 (Đề xuất 5): Ủng hộ.**

**G3.8 (§3.5): Ủng hộ.** **G3.8 (§2.1): Phản bác** nguy cơ "wd tăng 200 lần gây tương tác không mong muốn".

**Ý kiến tôi: ủng hộ cả hai về việc dùng bộ này; phản bác lập luận "wd ×200 nguy hiểm" của G3.8.**
- **Số liệu về overfit.** Train Dice 0.8701, val tốt nhất 0.7456: gap 0.1245 (so với epoch cuối 0.7394 là 0.1307). Val dao động ±0.008–0.015 giữa các epoch.
- **Tính toán wd.** R0 có khoảng 27.2k bước. Với AdamW decoupled, mức co trọng số tích lũy ≈ bước × lr × wd:
  - wd = 1e-4: **0.03%** (vô hiệu).
  - wd = 0.02: **2.7–5.3%** (cosine so với lr cố định). Nằm trong dải chuẩn 0.01–0.05, không có cơ sở lo phân kỳ.

**Chốt:**
- §3.4, §5.1, §5.2: giữ nguyên thiết kế.
- §4, hàng `--weight-decay`: ghi "0.02 (mức co tích lũy 2.7–5.3%)".
- Run chứa bộ này xếp sau R1 (Điểm 10).

---

### Điểm 10. Gộp biến vào Run 1 và số lượng run

**Plan gốc nói gì.** §7: R1–R9, thử từng biến một.

**G4 (Đề xuất 6): Một phần.** Gộp 7 biến vào R1 (448, bs 12, lr 1.2e-4, min-train-area, neg-ratio + vá ca rỗng, pos_weight, wd, EMA).

**G3.8 (§2.1, §4): Một phần.** Phản đối gộp, đề xuất tách thành "Data & Loss" và "Regularization", nhưng R1 của chính G3.8 vẫn gộp 448, `--batch-size 12`, min-train-area, neg-ratio + `min_neg_per_case`, `pos_weight 2.0`, epochs/patience.

**Ý kiến tôi.**
- G3.8 đúng về nguyên tắc (ablation cần cô lập đóng góp), nhưng không nhất quán khi R1 của nó vẫn bốn đến sáu biến.
- G4 gộp quá tay: wd, EMA, lr, bs, pos_weight cùng lúc làm không thể giải thích kết quả.
- Mỗi run tốn khoảng 4–4.5 giờ trên T4 nên không thể thử từng biến một như Plan gốc (9 run). Cách khả thi: **nhóm theo lý do kỹ thuật** (độ phủ dữ liệu; chống overfit; kiến trúc), và tách riêng những biến mà số liệu chưa ủng hộ (`pos_weight`).

**Chốt:** thay bảng §7 bằng lộ trình ở mục 3.

---

### Điểm 11. `--epochs` và `--patience`

**Plan gốc nói gì.** §7 lệnh nền: `--epochs 30 --patience 8`.

**G4 (Đề xuất 6): Một phần.** Giữ 30/8 (từ Plan gốc).

**G3.8 (§2.3): Ủng hộ kết luận, phản bác lý do.** Đề xuất 40/10 vì sợ underfitting khi tăng regularization.

**Ý kiến tôi.**
- R0 val đạt 0.7382 ở epoch 24, đỉnh 0.7456 ở epoch 30; chênh lệch sau epoch 24 nằm trong nhiễu. Lập luận "30 epoch gây underfitting" yếu.
- Aug mức 2 thường hội tụ chậm hơn, nên 40/10 hợp lý.
- **Quan trọng hơn con số:** cosine co theo `--epochs`, nên **mọi run R1–R5 phải dùng cùng cấu hình** thì mới so được.

**Chốt:** §4 (hàng epochs) và lệnh nền §7: `--epochs 40 --patience 10` cho mọi run.

---

### Điểm 12. Batch size và LR trong R1

**Plan gốc nói gì.** §4: `--batch-size` 12–16 kèm `--lr` ×1.4.

**G4 (R1): Phản bác.** `--batch-size 12`, `--lr 0.00012`.

**G3.8 (R1): Một phần.** Phê bình việc gộp (§2.1) nhưng R1 của nó vẫn có `--batch-size 12`.

**Ý kiến tôi: phản bác G4, một phần với G3.8.**
- Đổi bs và lr cùng lúc với đổi dữ liệu và độ phân giải thêm hai biến gây nhiễu ngay ở run quan trọng nhất.
- Có thể tăng bs ở run cuối nếu cần tăng tốc.

**Chốt:** §4, hàng `--batch-size`: "giữ 8 và lr 1e-4 trong R1–R4; tăng ở run cuối".

---

### Điểm 13. Khởi tạo nóng R3 từ R2

**Plan gốc nói gì.** §6.1(e): khởi tạo `--hr-skip` từ checkpoint cũ để "đỡ train lại".

**G4: Phản bác.** (R3 tùy chọn `--init-weights` từ Run 2.)

**G3.8: Phản bác.** (R3 `--init-weights` từ Run 2.)

**Ý kiến tôi: phản bác cả hai, và cả Plan gốc.** R3 khi đó được huấn luyện **thêm** so với R2; mức tăng có thể do thêm epoch, không phải do kiến trúc.

**Chốt:**
- §6.1(e): đổi thành "dòng ablation: train từ trọng số ImageNet cùng lịch trình; khởi tạo nóng chỉ dùng cho run cuối nếu cần tiết kiệm".
- §7: R6 cũ train từ ImageNet.

---

### Điểm 14. Dice dự kiến từng run

**G4:** R1 = 0.52–0.58; R2, R3 mỗi run cộng thêm 0.02–0.04. **G3.8:** R1 = 0.48–0.53, R2 = 0.53–0.57, R3 = 0.57–0.62. Cả hai: **Phản bác.**

**Lý do.**
- Không con số nào suy ra được từ dữ liệu.
- CI 95% của Dice 3D trên 75 ca là ±0.079, nên mức tăng +0.02–0.04 của một biến đơn lẻ **không phân biệt được với nhiễu** nếu so sánh không ghép cặp.
- Một lý do nữa: cột "Kỳ vọng" trong Plan gốc (+0.01–0.03...) cũng là ước lượng kinh nghiệm, cần được ghi nhãn như vậy.

**Chốt:**
- §7: đổi cột "Kỳ vọng" thành tiêu chí quyết định đo được (độ nhạy nốt ≥ 5 mm, FP/ca trên ca không GT, gap train–val), kiểm định ghép cặp (Phụ lục C).
- Các dòng "Kỳ vọng" trong §3–§6: thêm cụm "ước lượng kinh nghiệm, chưa kiểm chứng" và bỏ dòng "kỳ vọng tổng hợp 0.02–0.06".

---

### Điểm 15. "≤ 1 FP/ca là chuẩn lâm sàng", nguyên nhân FP, "nốt < 5 mm chưa từng học"

**G4 và báo cáo Step 0: Phản bác.** Ba phát biểu cần sửa:
1. "≤ 1.0 FP/ca là ngưỡng khuyến cáo lâm sàng": GT ở đây là **hợp** của các reader (gồm cả nốt chỉ một reader đánh dấu). Benchmark kiểu LUNA16 dùng đồng thuận và bỏ qua "phát hiện không liên quan". Một phần "FP" có thể là tổn thương thật không được đánh dấu. Không so trực tiếp được.
2. "FP do xương sườn, cơ hoành, mạch máu rốn phổi": chưa kiểm chứng; chưa có thống kê cụm FP. Con số "200–400 lát/ca" cũng chưa đúng với log (khoảng 168 lát phổi/ca).
3. "Nốt < 5 mm bị mù do chưa từng được học": chưa tách được khỏi bộ lọc hậu xử lý (ngưỡng 19.6 mm² của nhóm này đúng bằng ngưỡng lọc). Chỉ nói được là "bị chặn bởi cả hai".

**G3.8:** không bàn, dùng lại các phát biểu này.

**Chốt:**
- §1, §2: sửa lời văn các điểm trên.
- Thêm R0b (Điểm 17) để đo thật: thể tích, `max_prob`, vị trí cụm FP; độ nhạy < 5 mm với min-area 0 và 6.

---

### Điểm 16. Cứu 8 ca G2: Dice +0.107

**G4: Một phần.** Tính đúng (8/75 = +0.107, từ 0.3771 lên 0.4838), nhưng dùng nó như một "cải tiến".

**G3.8:** không bàn.

**Ý kiến tôi.** Mục tiêu đúng, nhưng đó là hiệu ứng của quy ước metric (GT = 0 và FP = 0 thì Dice = 1), không phải cải thiện phân đoạn.

**Chốt:** §2.1: báo cáo Dice 3D **chỉ trên ca có GT** (0.3951 cho R0), báo cáo FP/ca trên ca không GT riêng (3.63 với 8 ca có FP; 2.64 trung bình 11 ca).

---

### Điểm 17. Bổ sung của tôi (hai tài liệu không có)

**G4 và G3.8: không có phần nào tương ứng.**

**17a. R0b: đánh giá lại R0, không cần train.** Một lượt suy luận, lưu xác suất để quét miễn phí:
- min-area ∈ {0, 6, 10, 20} × V_min ∈ {0, 10, 20 mm³}: đo độ nhạy < 5 mm, 5–10 mm, FP/ca.
- Ngưỡng mở rộng 0.8, 0.9, 0.95, 0.99: kiểm tra FP có tự tin thật không.
- Thống kê từng cụm FP: `max_prob`, `mean_prob`, `vol_mm3`, `max_slice_area`, `z_slices`.
- `span` của từng nốt GT và phân bố `spacing_z`.
- Quy tắc đọc: nếu độ nhạy < 5 mm tăng rõ khi min-area giảm thì bộ lọc là nguyên nhân chính. Nếu cụm FP chủ yếu có `max_prob > 0.95` và thể tích lớn thì không kỳ vọng ở `pos_weight`, ưu tiên Điểm 4.

**17b. Val cố định và test riêng.** Chọn ngưỡng, min-area, V_min và model trên cùng 75 ca val rồi báo cáo trên val sẽ lạc quan. Cố định đúng 75 ID val hiện tại bằng `--val-patients-file` cho mọi run (đổi `--n-patients` hoặc `--select` làm đổi shuffle và val); test riêng 10% (khoảng 101 ca ở run cuối) chỉ chạm một lần.

**17c. Cách báo cáo.** Dice trên ca có GT; FP/ca tách theo ca có và không GT; FROC mức nốt tại 0.5/1/2/4 FP/ca cho nốt ≥ 5 mm và ≥ 3 mm; CI bootstrap theo ca (độ nhạy ≥ 5 mm là 66.0% ± 7.7%); so run bằng bootstrap ghép cặp theo ca và McNemar theo nốt (Phụ lục C).

**17d. Phân tầng theo độ đồng thuận reader (tùy chọn).** Mỗi ca có nhiều thư mục SEG theo từng reader. Nếu các nốt bị bỏ sót chủ yếu chỉ có 1 reader đánh dấu thì một phần "lỗi" là nhãn mơ hồ.

**Chốt:**
- Thêm §2.3 "R0b" (Phụ lục A).
- §5.6: thêm `--val-patients-file`; giữ `--test-split`.
- §3.3: giữ việc log P/R mỗi epoch (hai tài liệu không bàn).

---

### Điểm 18. Lộ trình cuối

**Plan gốc:** R1–R9.
**G4: Một phần (4 run, R1 gộp 7 biến).** **G3.8: Một phần (5 run).**
**Ý kiến tôi:** xem mục 3.

---

## 3. Lộ trình đề xuất thay bảng §7

Nguyên tắc: mọi run được đánh giá bằng **cùng hậu xử lý** và **cùng 75 ca val**; cấu hình huấn luyện chung (trừ biến đang thử): `--epochs 40 --patience 10`, bs 8, lr 1e-4, seed 42. Chi phí ước tính mỗi run: 4–4.5 giờ trên T4.

| Run | Thay đổi so với run trước | Cần code (Plan gốc) | Điều kiện / tiêu chí quyết định |
|---|---|---|---|
| **R0b** (không train) | Đánh giá lại R0 theo 17a | §2.3 mới, Phụ lục A | Xác định bộ lọc đích; nếu FP có `max_prob` cao thì ưu tiên R1, bỏ R2 |
| **R1** (độ phủ dữ liệu) | `--min-train-area 5`; lấy lại mẫu âm tính mỗi epoch (neg ratio 0.5); `IMAGE_SIZE=448`; vá nhãn Điểm 6. Giữ pos_weight 4, wd 1e-4 | §3.1, §3.1b, §3.2, §5.3 | Cùng hậu xử lý với R0b: độ nhạy nốt ≥ 5 mm tăng có ý nghĩa (McNemar); FP/ca trên ca không GT giảm; FP tổng không tăng |
| **R2** (đơn biến) | `pos_weight 2.0` | không | **Chỉ chạy nếu R0b không phủ nhận.** Ngược lại thay bằng hard-negative mining (§5.6) |
| **R3** (chống overfit) | wd 0.02 (4 nhóm) + EMA 0.999 + `--aug-level 2` | §3.4, §5.1, §5.2 | Gap train–val < 0.08; checkpoint ổn định giữa epoch. Muốn ablation đầy đủ thì tách (wd + EMA) và (aug 2) |
| **R4** (kiến trúc) | `--hr-skip`, train từ ImageNet cùng lịch trình | §6.1 | Độ nhạy 5–10 mm và HD95 cải thiện thêm so với R3. Nếu R1 đã đẩy 5–10 mm lên cao thì dừng ở R3 |
| **R5** (nghiệm thu) | Cấu hình tốt nhất trên 1010 ca, val 75 ca cố định, test 10% | §5.6 | Báo cáo duy nhất trên test, kèm CI |

**Lộ trình tiết kiệm (4 run):** R0b → R1 → R3 (gộp) → R4 → R5.

---

## 4. Tổng hợp chỉnh Plan gốc (để bạn duyệt)

| Mục Plan gốc | Hiện tại | Sửa thành | Từ điểm | Cần R0b |
|---|---|---|---|---|
| §1 bảng vấn đề | P1–P14 | Thêm P15 (hậu xử lý 20 mm² chặn trần 70.9%), P16 (âm tính chỉ 1.72% được thấy), P17 (quy ước metric ca không GT); P3 hạ xuống "Thấp" | 1, 4, 6, 16 | Không |
| §2.1 `eval_sweep.py` | `--min-area` mặc định 20; điều kiện "precision ≪ recall" | Quét min-area, V_min; báo cáo tách ca có/không GT; sửa bảng "Cách đọc kết quả" | 1, 2, 5, 16 | Không |
| §2.3 (mới) | Không có | R0b (Phụ lục A) | 3, 17 | Là chính nó |
| §3.1 | "giữ 20 mm² ở hậu xử lý" | Xóa; thêm 3.1b lấy lại mẫu âm tính | 1, 4 | Không |
| §3.2 | `--tumor-union-p` có lỗi nhãn | Thêm vá `msk_stack & (keep > 0)`; P3 ưu tiên thấp | 6 | Không |
| §3.4 | wd 0.02 | Giữ; ghi rõ mức co 2.7–5.3% | 9 | Không |
| §4 `pos_weight` | "2.0/1.0 nếu P ≪ R" | Ablation đơn biến, có điều kiện từ R0b | 5 | Có |
| §4 `--neg-ratio` | 0.5–1.0 | 0.5 theo epoch (3.1b) | 4 | Không |
| §4 `--epochs/--patience` | 30/8 | 40/10 cho mọi run | 11 | Không |
| §4 `--batch-size` | 12–16, lr ×1.4 | Giữ 8, lr 1e-4 ở R1–R4 | 12 | Không |
| §5.3 | 448 "nhanh hơn ~20%" | Thêm `image_size` vào meta; kiểm FOV đủ 500 ca; ước lượng 6–7 phút/epoch | 7 | Không |
| §5.6 | Hard negative, test split | Thêm `--val-patients-file`; hard negative đặt sau R1 | 4, 17 | Không |
| §6.1 | Khởi tạo nóng từ checkpoint cũ | Dòng ablation train từ ImageNet; thêm test hồi quy; chạy sau R1 | 8, 13 | Không |
| §6.2 | Bước 2–3 | Đánh dấu "không ưu tiên" | 8 | Không |
| §7 | R1–R9 | Bảng mục 3; cột "Kỳ vọng" → tiêu chí quyết định; bỏ R5 | 6, 10, 14, 18 | Có |
| §3–§6 "Kỳ vọng" | Các dòng +0.01–0.03 | Thêm nhãn "ước lượng kinh nghiệm"; bỏ "0.02–0.06" | 14 | Không |

---

## 5. Việc cần xác minh trong code trước khi chạy

1. `offline_builder.py`: nhánh `else` khi ca không có lát dương tính có thật sự đặt `num_neg_keep = 1`? Và `zero_sl_ratio` lớn có giữ toàn bộ ứng viên âm tính?
2. `postprocess_3d_mask`: có `min_slices_span` hay chỉ `min_volume_mm3`? Hàm có lấp lỗ/lọc theo phổi bên trong không (ảnh hưởng cách viết R0b)?
3. `BaseTrainer.fit` có gọi `on_epoch_start(epoch)` trước khi tạo iterator của DataLoader (cần để sampler lấy lại mẫu đúng epoch)?
4. `slices.csv`: tên cột (`patient_id`, `z`, `has_tumor`, `tumor_area_mm2`, `split`...) khớp với mã mẫu?
5. Checkpoint R0 tải được với `strict=True` khi `high_res_skip=False`.
6. Kiểm tra mất nhãn khi cắt 448 mới chạy trên 449 ca có mask phổi. Chạy lại cho đủ 500 ca, nhất là 75 ca val, trước khi chốt 448.

---

## 6. Thứ tự công việc (để khỏi rối)

Chỉ **Phụ lục A** là việc cần chạy ngay. Phụ lục B và C là công cụ dùng sau.

| Bước | Việc | Dùng phụ lục | Gửi lại tôi |
|---|---|---|---|
| 1 | Chọn cách chỉnh Plan: chỉnh ngay (để trống 2 chỗ phụ thuộc R0b) hoặc chờ R0b | không | Câu trả lời "cách 1" hoặc "cách 2" |
| 2 | (Tùy chọn) Gửi 3 file để tôi sửa Phụ lục A cho khớp code | không | `tools/eval_sweep.py`, `common/eval3d/postprocess3d.py`, đoạn chọn lát âm tính trong `offline_builder.py` |
| 3 | Chạy R0b, kèm phép kiểm tra hồi quy | **A** | 3 file CSV + kết quả hồi quy |
| 4 | Tôi điền các tham số còn trống vào Plan (V_min, có chạy R2 không, cột điều kiện §7) | không | không |
| 5 | Sửa code cho R1 (vá nhãn, §3.1, §3.1b, 448, `image_size`); xác minh mục 5 | **B** | Báo lỗi nếu gặp |
| 6 | Train R1, so với R0b | **C** | CSV của R1 |

---

## Phụ lục A: R0b, `tools/eval_sweep_v2.py` (bản phác, chưa chạy thử)

> **Khi nào chạy:** bây giờ, là việc đầu tiên, trước mọi thay đổi code train.
>
> **Chạy để làm gì:**
> 1. Tách "bộ lọc 20 mm² chặn nốt" khỏi "mô hình chưa học" cho nhóm nốt < 5 mm (Điểm 1, 15).
> 2. Chọn `V_min` cho lọc theo thể tích, và đếm bao nhiêu nốt thật chỉ nằm trên 1 lát (Điểm 2).
> 3. Đo FP có "tự tin" không (`max_prob`, thể tích) để quyết định có chạy R2 `pos_weight 2` hay không (Điểm 5).
> 4. Tạo dòng baseline R0 dưới **cùng hậu xử lý** với R1 để so sánh công bằng (Điểm 3).
>
> **Tại sao chạy trước:** không cần train, chỉ suy luận 75 ca (mỗi ca suy luận một lần, các ngưỡng quét trên cùng xác suất). Kết quả điền ba chỗ trống của Plan. Bỏ qua bước này thì R1 so với R0 sẽ lẫn tác động của bộ lọc.
>
> **Cần gì trước:** repo, checkpoint `best_model.pth` của R0, cache 1 mm. Đối chiếu các dòng gọi hàm với `tools/eval_sweep.py` trong repo (script đó đã chạy được trong notebook của bạn).
>
> **Cách chạy:**
> ```bash
> DUNEAI_MAMBA_BACKEND=cuda python tools/eval_sweep_v2.py \
>   --cache-dir /content/data/cache_1mm \
>   --checkpoint mamba_unet/runs/mamba_colab_500item_25epoch_extra_test/best_model.pth \
>   --split val --out-prefix sweep_r0b
> ```
>
> **Kiểm tra hồi quy trước khi tin kết quả:** lấy các dòng `amin=20, vmin=0, th=0.6` trong `sweep_r0b_patients.csv` rồi so với `sweep_r0_patients.csv`. `dice`, `fp` (so với `fp_comps`) phải khớp. Chỉ cần chạy thử trước với 3–5 ca. Nếu lệch, hàm hậu xử lý lọc theo thứ tự khác giả định của tôi (ví dụ lọc trước khi lấp lỗ), cần sửa script trước.
>
> **Thời gian:** phần suy luận tương đương lần `eval_sweep` trước; phần quét thêm tốn CPU (chưa đo).
>
> **Gửi lại tôi:** 3 file `sweep_r0b_patients.csv`, `sweep_r0b_nodules.csv`, `sweep_r0b_components.csv` và kết quả kiểm tra hồi quy.

```python
# Quét min-area x min-volume x ngưỡng; xuất thống kê cụm FP.
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.data_pipeline.medical_prep import lung_cache_path
from common.eval3d.postprocess3d import postprocess_3d_mask
from common.eval3d.volume_inference import predict_patient_volume
from mamba_unet.evaluate import load_mamba_unet_model

S26 = np.ones((3, 3, 3), np.uint8)

def comps(mask, prob, sz):
    lab, n = ndi.label(mask, structure=S26)
    rows = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == i
        per_slice = m.reshape(m.shape[0], -1).sum(1)
        rows.append(dict(id=i, vol_mm3=float(m.sum() * sz), max_area=float(per_slice.max()),
                         z_slices=int((per_slice > 0).sum()),
                         max_prob=float(prob[sl][m].max()), mean_prob=float(prob[sl][m].mean())))
    return lab, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val"); ap.add_argument("--backbone", default="mamba_vision_T")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    ap.add_argument("--min-areas", type=float, nargs="+", default=[0, 6, 10, 20])
    ap.add_argument("--min-vols", type=float, nargs="+", default=[0, 10, 20])
    ap.add_argument("--out-prefix", default="sweep_r0b")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mamba_unet_model(a.checkpoint, dev, backbone=a.backbone)
    cache = Path(a.cache_dir)
    df = pd.read_csv(cache / "slices.csv")
    pids = sorted(df[df["split"] == a.split]["patient_id"].unique())

    P, N, F = [], [], []
    for pid in pids:
        r = predict_patient_volume(model, cache / pid, device=dev)
        prob, gt, meta = r["prob_1mm"], r["gt_mask_1mm"], r["meta"]
        sz = float(meta["orig_spacing_xyz"][2])
        lp = lung_cache_path(cache / pid)
        lung = np.load(lp) if lp.exists() and meta.get("lung_ok", True) is not False else None

        gt_lab, n_gt = ndi.label(gt, structure=S26)
        gt_info = []
        for i, sl in enumerate(ndi.find_objects(gt_lab), 1):
            m = gt_lab[sl] == i
            ps = m.reshape(m.shape[0], -1).sum(1)
            gt_info.append(dict(id=i, diam=2 * np.sqrt(ps.max() / np.pi), span=int((ps > 0).sum())))

        for th in a.thresholds:
            # mask chưa lọc theo kích thước (giữ nguyên bước lấp lỗ/phổi của hàm có sẵn)
            base = postprocess_3d_mask(prob, lung_mask_zyx=lung, threshold=th,
                                       min_slice_area_mm2=0, spacing_z=sz) > 0
            lab, rows = comps(base, prob, sz)
            for amin in a.min_areas:
                for vmin in a.min_vols:
                    keep_ids = [c["id"] for c in rows if c["max_area"] >= amin and c["vol_mm3"] >= vmin]
                    pred = np.isin(lab, keep_ids)
                    inter = float((pred & (gt > 0)).sum())
                    dice = 2 * inter / max(1.0, float(pred.sum() + (gt > 0).sum()))
                    hit_ids = set(np.unique(lab[(gt > 0) & pred])) - {0}
                    fp = [c for c in rows if c["id"] in keep_ids and c["id"] not in hit_ids]
                    P.append(dict(pid=pid, th=th, amin=amin, vmin=vmin,
                                  dice=dice if (gt > 0).any() or pred.any() else 1.0,
                                  n_gt=n_gt, fp=len(fp)))
                    for g in gt_info:
                        det = bool((pred & (gt_lab == g["id"])).any())
                        N.append(dict(pid=pid, th=th, amin=amin, vmin=vmin,
                                      diam=g["diam"], span=g["span"], det=int(det)))
                    if amin == 0 and vmin == 0:
                        for c in rows:
                            F.append(dict(pid=pid, th=th, is_fp=int(c["id"] not in hit_ids), **c))
        print(pid, "xong")

    pd.DataFrame(P).to_csv(f"{a.out_prefix}_patients.csv", index=False)
    pd.DataFrame(N).to_csv(f"{a.out_prefix}_nodules.csv", index=False)
    pd.DataFrame(F).to_csv(f"{a.out_prefix}_components.csv", index=False)

if __name__ == "__main__":
    main()
```

Cách đọc nhanh: `components.csv` lọc `is_fp == 1` rồi thống kê `max_prob` và `vol_mm3`; `nodules.csv` nhóm theo `(amin, vmin, diam)` để so độ nhạy < 5 mm; cột `span` cho biết bao nhiêu nốt thật chỉ nằm trên 1 lát.

---

## Phụ lục B: lấy lại mẫu âm tính mỗi epoch (cho §3.1b)

> **Khi nào dùng:** không chạy riêng. Dùng ở bước 5, khi sửa code cho R1, sau khi bạn duyệt Plan. Hiện tại chỉ cần đọc.
>
> **Dùng để làm gì:** gắn vào train loader để mỗi epoch lấy lại một tập lát âm tính ngẫu nhiên khác (Điểm 4).
>
> **Tại sao:** lúc train mô hình chỉ thấy 1.72% lát âm tính (1321/76941), nên FP "tự tin" trên các cấu trúc nó chưa từng gặp. Sampler này phủ khoảng 81% pool âm tính sau 40 epoch mà thời gian mỗi epoch không đổi.
>
> **Đã kiểm tra:** chạy thử trên dữ liệu giả cùng quy mô (7950 lát/epoch, không trùng lặp, đúng theo epoch, phủ 81.2%).
>
> **Cần xác minh khi gắn vào repo:**
> 1. `has_tumor` truyền vào phải là cột của đúng tập train sau khi lọc `split == "train"` và `reset_index` (lệch chỉ số sẽ không báo lỗi).
> 2. Cột `has_tumor` phải theo ngưỡng mới sau khi hạ `--min-train-area` xuống 5.
> 3. `set_epoch` phải được gọi trước khi DataLoader tạo iterator của epoch đó (mục 3 trong danh sách xác minh).
>
> **Gửi lại tôi:** không cần, trừ khi gặp lỗi khi gắn.

```python
import numpy as np
from torch.utils.data import Sampler

class PosAllNegResample(Sampler):
    """Mỗi epoch: toàn bộ lát dương tính + neg_ratio * n_pos lát âm tính lấy ngẫu nhiên từ pool."""
    def __init__(self, has_tumor, neg_ratio=0.5, seed=42):
        has_tumor = np.asarray(has_tumor)
        self.pos = np.where(has_tumor == 1)[0]
        self.neg = np.where(has_tumor == 0)[0]
        self.k = min(len(self.neg), int(round(neg_ratio * len(self.pos))))
        self.seed, self.epoch = seed, 0

    def set_epoch(self, e):
        self.epoch = int(e)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = np.concatenate([self.pos, rng.choice(self.neg, self.k, replace=False)])
        rng.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self):
        return len(self.pos) + self.k
```

- Dựng `slices.csv` với `zero_sl_ratio` đủ lớn để giữ toàn bộ lát âm tính của **train**; với **val**, lấy mẫu âm tính một lần cố định theo seed (ví dụ 0.25–0.5) rồi lưu lại.
- Truyền sampler cho `DataLoader` train (bỏ `shuffle=True`) và gọi `sampler.set_epoch(epoch)` trong `on_epoch_start`.
- Mô hình thấy khoảng 2.7k lát âm tính khác nhau mỗi epoch; chi phí epoch không đổi so với `neg-ratio 0.5` cố định.

---

## Phụ lục C: so sánh ghép cặp giữa hai run

> **Khi nào dùng:** ở bước 6, sau khi R1 train xong và đã đánh giá bằng đúng hậu xử lý của R0b. Chưa cần chạy bây giờ.
>
> **Dùng để làm gì:** kiểm tra hai run có khác nhau thật không, theo ca (Dice) và theo nốt (McNemar).
>
> **Tại sao:** CI 95% của Dice 3D trên 75 ca là ±0.079, nên chênh lệch +0.02–0.04 nhìn bằng mắt không đáng tin. Ghép cặp theo ca cho CI hẹp hơn nhiều (ví dụ so Th 0.5 với 0.6: SD chênh lệch theo ca chỉ 0.0069).
>
> **Đã kiểm tra:** chạy được trên CSV thật.
>
> **Lưu ý khi dùng:** hai hàm nhận dữ liệu theo từng ca và từng nốt, nên chỉ dùng với CSV đầu ra của R0b/R1. McNemar yêu cầu hai bảng nốt **cùng thứ tự nốt**: khi so hai model khác nhau, hãy ghép bảng theo `pid` và `diam`, không dựa vào thứ tự dòng.
>
> **Gửi lại tôi:** CSV đầu ra của R1 khi có.

```python
import numpy as np, pandas as pd

def paired_bootstrap(a, b, col="dice", B=5000, seed=0):
    """a, b: DataFrame có cột pid và col (cùng tập ca). Trả về (Δ trung bình, CI95)."""
    d = (b.set_index("pid")[col] - a.set_index("pid")[col]).dropna().values
    rng = np.random.default_rng(seed)
    m = [rng.choice(d, len(d)).mean() for _ in range(B)]
    return d.mean(), np.percentile(m, [2.5, 97.5])

def mcnemar(nodA, nodB):
    """nodA, nodB: DataFrame (pid, diam, det) cùng thứ tự nốt.
    b = A bắt được mà B bỏ sót, c = ngược lại."""
    from scipy.stats import binomtest
    b = int(((nodA.det == 1) & (nodB.det == 0)).sum())
    c = int(((nodA.det == 0) & (nodB.det == 1)).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)
```
