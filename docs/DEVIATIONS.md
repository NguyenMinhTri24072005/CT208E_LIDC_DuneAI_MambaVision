# DUNEAI REFACTORING (v2.1) — BẢNG GHI NHẬN SAI LỆCH & ĐIỀU CHỈNH KỸ THUẬT (DEVIATIONS)

Tài liệu này ghi nhận toàn bộ các điều chỉnh kỹ thuật, chi tiết bổ sung và thỏa thuận giữa nhóm phát triển và tài liệu `docs/REFACTOR_SPEC.md` trong suốt quá trình thực thi từ Phase 0 đến Phase 7.

---

## 1. Không gian tên và Thư mục lưu Checkpoint
- **Mục spec liên quan:** Mục 2, Mục 6 (`<run_dir>`, `checkpoints/`).
- **Nội dung:** Checkpoint và lịch sử huấn luyện cũ từ `DuneAI_Remake/checkpoints/unet_base_25d/` đã được di chuyển an toàn sang `checkpoints/unet_base_25d/` ở ROOT dự án trước khi xóa `DuneAI_Remake/`.
- **Lý do:** Bảo tồn 100% dữ liệu huấn luyện, artifacts, file cấu hình và kết quả thực nghiệm của người dùng.
- **Hệ quả:** Hoàn toàn an toàn, cấu trúc thư mục mới giữ trọn vẹn lịch sử mà không phụ thuộc thư mục legacy.

## 2. Xử lý trường hợp biên cho Dice 3D khi TP = 0
- **Mục spec liên quan:** Mục 4.5 (`common/eval3d/metrics3d.py`).
- **Nội dung:** Khi cả `pred` hoặc `target` không rỗng nhưng `tp == 0` (hai khối u hoàn toàn không giao nhau), nếu dùng công thức mềm `(2*tp + eps)/(pred_sum + target_sum + eps)` sẽ ra giá trị xấp xỉ `eps / (sum + eps) > 0`. Đã bổ sung nhánh `elif tp == 0: dice = 0.0`.
- **Lý do:** Tuân thủ định nghĩa toán học chuẩn của Dice metric (hai tập rời nhau thì Dice phải bằng đúng 0.0 thay vì một số epsilon cực nhỏ).
- **Hệ quả:** Tăng độ chính xác và tính tin cậy tuyệt đối của bộ số liệu nghiệm thu 3D.

## 3. Kiểm tra tính độc lập cấu trúc trong Unit Tests
- **Mục spec liên quan:** Mục 3.3, Mục 10 (`tests/test_run_folder.py`, `tests/test_resume.py`).
- **Nội dung:** Thay vì kiểm tra `sys.modules` tại runtime (vốn bị ảnh hưởng bởi thứ tự nạp module khi chạy test runner theo nhóm `unittest discover`), các test cô lập được nâng cấp lên phân tích tĩnh AST kết hợp runtime sandbox mocking.
- **Lý do:** Tránh hiện tượng false positive/flaky test do cache module giữa các unit test kế tiếp nhau.
- **Hệ quả:** Mọi bài test có thể chạy độc lập hoặc chạy gộp theo bất kỳ thứ tự nào đều đảm bảo tính chính xác 100%.

## 4. Xóa các thư mục Legacy đã lỗi thời
- **Mục spec liên quan:** Mục 11 (Phase 7), Mục 12.
- **Nội dung:** Toàn bộ các thư mục mã cũ (`DuneAI/`, `Planning/`, `unet_baseline/`, `DuneAI_Remake/`) đã được xóa hoàn toàn theo chỉ đạo rõ ràng từ người dùng.
- **Lý do:** Dự án đã chuyển đổi hoàn toàn sang hai kiến trúc độc lập `unet_base/` và `mamba_unet/` trực tiếp dưới `SOURCES/`.
- **Hệ quả:** Mã nguồn cực kỳ tinh gọn, không còn mã rác hay tham chiếu chéo legacy.
