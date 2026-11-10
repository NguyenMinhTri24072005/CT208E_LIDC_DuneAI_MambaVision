# PROMPT ĐIỀU PHỐI CHO ANTIGRAVITY (GEMINI) — TÁI CẤU TRÚC DuneAI

> **v2.1:** khớp với `REFACTOR_SPEC.md` v2.1. Điểm sửa chính: hàm kích hoạt và hàm mất mát nằm **trong từng kiến trúc** (`unet_base/activations.py`, `unet_base/losses.py`, `mamba_unet/activations.py`, `mamba_unet/losses.py`), **không** có thư mục `activations/` hay `losses/` ở ROOT. Phase 1 không còn làm activation/loss; chúng được tạo ở Phase 3 (unet_base) và Phase 4 (mamba_unet). Khối tích chập (`DoubleConv`, `OutConv`) cũng nằm trong từng kiến trúc (`unet_base/blocks.py`, `mamba_unet/blocks.py`), không còn `common/nn_blocks.py`. Bạn phải dùng bản `REFACTOR_SPEC.md` v2.1 khi chép vào `SOURCES/docs/`.

## Trước khi bắt đầu (việc của bạn, ~5 phút)
1. **Sao lưu toàn bộ** thư mục `SOURCES/` (zip) và **sao lưu riêng ít nhất 1 checkpoint đã train** của mỗi kiến trúc (nếu có).
2. Chép `REFACTOR_SPEC.md` vào `SOURCES/docs/REFACTOR_SPEC.md` (tạo thư mục `docs/`).
3. Mở **thư mục `SOURCES/`** làm workspace trong Antigravity (không mở `DuneAI_Remake/`).
4. Tạo môi trường Python có `torch`, `numpy`, `pandas`, `scipy`, `opencv-python-headless`, `SimpleITK`, `scikit-image`, `tqdm`, `pyyaml`, `matplotlib`, `nbformat` để Agent chạy được test. (Không cần `timm`, `einops`, `mamba-ssm` — đó là chủ đích.)
5. Chạy từng phase **một lệnh một lần**: dán *PROMPT CHÍNH* → chờ báo cáo Phase 0 → đọc → dán *lệnh phase kế tiếp*.

> Mẹo: nếu Agent báo "PASS" mà không dán output lệnh thật, trả lời bằng **PROMPT KHI NGHI NGỜ** ở cuối file.

---

## PROMPT CHÍNH (dán một lần ở đầu cuộc trò chuyện)

```text
VAI TRÒ
Bạn là kỹ sư phần mềm cẩn thận, làm việc trong thư mục SOURCES/ (gốc dự án, đã mở sẵn). Bạn có Terminal và Git.
Nhiệm vụ: tái cấu trúc dự án học sâu phân đoạn khối u phổi 2.5D (LIDC-IDRI) theo đúng tài liệu docs/REFACTOR_SPEC.md.

BƯỚC 0 BẮT BUỘC
Đọc TOÀN BỘ docs/REFACTOR_SPEC.md trước khi làm bất cứ việc gì. Sau đó đọc cấu trúc thư mục hiện có của dự án và mã nguồn hiện tại.
Chưa đụng vào file code nào cho tới khi tôi ra lệnh "PHASE 0".

LUẬT SẮT (vi phạm = làm lại)
1. CHỈ DI CHUYỂN/TÁCH file logic đã chạy được. KHÔNG viết lại thuật toán, KHÔNG "dọn cho gọn", KHÔNG đổi hành vi, KHÔNG đổi tên thuộc tính/lớp con của mô hình (làm đổi khóa state_dict).
2. KHÔNG ĐOÁN. Nếu cần biết một tên hàm/khóa/tham số, hãy MỞ FILE ĐỌC. Nếu spec yêu cầu thứ mà code cũ không có, hoặc hai chỗ mâu thuẫn: DỪNG và hỏi tôi, đừng tự bịa.
3. Hai kiến trúc unet_base/ và mamba_unet/ TUYỆT ĐỐI không import nhau. Các gói dùng chung (common, configs) không bao giờ import hai kiến trúc.
4. Không đường dẫn cứng (cấm chuỗi "DuneAI_Remake", "SOURCES", "/content", "/kaggle", "D:\\" trong code). NGOẠI LỆ DUY NHẤT: hằng số nền tảng trong common/data_pipeline/runtime_data.py và giá trị mặc định trong ô cấu hình notebook do scripts/make_notebooks.py sinh ra (spec mục 8.6).
5. Không bao giờ in hay ghi token/secret.
6. Mọi lệnh test PHẢI được chạy thật trong Terminal. Khi báo cáo, DÁN NGUYÊN VĂN phần cuối output (ít nhất 5 dòng cuối) của từng lệnh. Cấm viết "đã pass" mà không dán output.
7. Chỉ làm ĐÚNG phase tôi giao. Xong phase: chạy test của phase, commit (git, nhánh refactor/split-architectures, message "P<n>: <tóm tắt>"), báo cáo theo mẫu, rồi DỪNG chờ lệnh. Không tự sang phase sau.
8. Không xóa file cũ cho tới PHASE 7. Trước đó chỉ TẠO file mới (copy/move có kiểm soát; ưu tiên copy ở P1–P6).
9. Mọi sai lệch so với spec, dù nhỏ, ghi vào docs/DEVIATIONS.md (file/mục spec, lý do, hệ quả).
10. Làm theo từng file một: sửa xong một file thì chạy kiểm tra cú pháp (python -m py_compile) hoặc import thử rồi mới sang file kế tiếp. Không sinh hàng loạt file rồi mới chạy thử.
11. MỌI THỨ THUỘC CẤU TRÚC MẠNG NẰM TRONG TỪNG KIẾN TRÚC. Mỗi kiến trúc có file RIÊNG: unet_base/blocks.py, unet_base/activations.py, unet_base/losses.py và mamba_unet/blocks.py, mamba_unet/activations.py, mamba_unet/losses.py (spec mục 4.1, 4.2, 4.3); activations và losses có registry riêng. TUYỆT ĐỐI KHÔNG tạo thư mục activations/ hoặc losses/ ở ROOT, KHÔNG tạo common/nn_blocks.py, KHÔNG đặt bất kỳ lớp nn.Module nào (khối conv, loss, activation, encoder, decoder) trong common/, KHÔNG 'khử trùng lặp' các bản giữa hai kiến trúc (trùng mã ở đây là CHỦ ĐÍCH để sửa bên này không ảnh hưởng bên kia). Metric đánh giá (compute_segmentation_metrics) là ngoại lệ duy nhất dùng chung: nằm ở common/metrics.py. Nếu bạn thấy chỗ nào trong spec hoặc trong lệnh của tôi còn nói activations/losses ở ROOT dùng chung: đó là sót, DỪNG và hỏi tôi.

ĐIỀU ĐẶC BIỆT QUAN TRỌNG (đọc kỹ spec mục 4.1, 5.3, 9)
- Hàm kích hoạt KHÔNG tham số đổi qua lại không làm đổi khóa state_dict; hàm CÓ tham số (star_relu, prelu) làm thêm khóa. Hàm kích hoạt phải đứng đúng chỉ số con trong nn.Sequential như bản cũ.
- mamba_unet phải chạy được khi KHÔNG có timm/einops/mamba_ssm/transformers (dùng bản dự phòng thuần PyTorch). Chọn nhân CUDA mamba_ssm chỉ khi import được VÀ có GPU CUDA.
- Golden state_dict phải sinh từ CODE CŨ ở Phase 0, trước mọi thay đổi. Tên khóa trong spec chỉ là ví dụ.
- common/ chỉ chứa: xử lý dữ liệu, vòng lặp train/quản lý run, đánh giá 3D, metric, đường dẫn. BaseTrainer nhận criterion đã dựng và metric_fn từ ngoài; cli_args nhận danh sách choices của --activation/--loss từ registry của kiến trúc gọi nó. common/ không có lớp nn.Module nào.
- Dữ liệu theo nền tảng (spec mục 8.6): Colab = zip cache_1mm trên Drive, giải nén CHỌN SỐ LƯỢNG bệnh nhân ra SSD; Kaggle = đọc /kaggle/input (chỉ đọc) qua thư mục symlink ghi được. Bố cục đã xác nhận: zip Colab có MỘT thư mục gốc cache_1mm/ chứa trực tiếp LIDC-IDRI-XXXX/; dữ liệu Kaggle lồng hai tầng .../cache_1mm/cache_1mm/LIDC-IDRI-XXXX/. Selftest dùng đúng các bố cục thật này làm ca chính. Mã vẫn dò theo mẫu LIDC-IDRI-<4 chữ số>, KHÔNG hard-code tên thư mục gốc hay độ sâu. Không sửa logic dataset.py/offline_builder.py để phục vụ phần này.
- Notebook phải được SINH bằng scripts/make_notebooks.py (nbformat), không viết JSON tay.

MẪU BÁO CÁO CUỐI MỖI PHASE
A. File đã tạo / sửa / xóa (danh sách đường dẫn)
B. Lệnh test đã chạy + output cuối (dán nguyên văn) + kết quả PASS/FAIL từng lệnh
C. Sai lệch so với spec (nếu có) — đã ghi vào docs/DEVIATIONS.md chưa
D. Câu hỏi/quyết định cần tôi trả lời
E. Commit hash của phase

Khi đã đọc xong spec và cấu trúc dự án, hãy trả lời NGẮN: (1) bạn hiểu mục tiêu thế nào bằng 5 dòng, (2) liệt kê bất cứ điểm nào trong spec mâu thuẫn hoặc không khớp với code hiện có mà bạn đã phát hiện, (3) hỏi những gì cần làm rõ. Chưa viết code.
```

---

## CÁC LỆNH PHASE (dán lần lượt, mỗi lần một lệnh, sau khi đã đọc báo cáo phase trước)

### PHASE 0 — Kiểm kê + golden (chưa di chuyển file)
```text
PHASE 0. Theo mục 11 và mục 9.1 của spec:
1) Tạo nhánh git refactor/split-architectures (nếu chưa có repo git ở đây thì DỪNG và báo tôi, đừng tự git init).
2) Kiểm kê toàn bộ code hiện có, viết docs/INVENTORY.md gồm: (a) bảng "file cũ -> file mới theo spec"; (b) với mỗi file: các hàm/lớp công khai; (c) mọi alias lớp mô hình (vd. DuneUNet, PureUNet, ClassicUNet, BaselineUNet, ModularUNet) KÈM danh sách nơi đang gọi chúng (grep); (d) toàn bộ tham số CLI cũ của từng lệnh; (e) hàm kích hoạt và hàm mất mát đang có + hàm mặc định của từng kiến trúc, ghi rõ hàm nào hiện đang được kiến trúc nào dùng (Phase 3 và 4 sẽ COPY đủ bộ vào file riêng của MỖI kiến trúc, nên không được bỏ sót hàm nào); liệt kê cả các khối conv (DoubleConv, OutConv, ConvBnAct, ...) và kiến trúc nào đang dùng khối nào; (f) mọi nơi nhắc tới use_stride4_stem; (g) về DỮ LIỆU: các tệp bắt buộc trong mỗi thư mục bệnh nhân, nơi code cũ đọc/ghi slices.csv (có ghi vào cache_dir không), cách chia train/val theo bệnh nhân và split_seed, cách code cũ tìm danh sách bệnh nhân (listdir hay đọc slices.csv).
3) Viết scripts/export_golden.py và CHẠY nó trên code HIỆN TẠI để sinh tests/golden/* theo mục 9.1 (kể cả legacy_prefixless_unet_keys.json nếu còn lớp U-Net không tiền tố).
4) KHÔNG sửa hay di chuyển bất kỳ file code cũ nào.
Tiêu chí xong: golden có đủ file; INVENTORY liệt kê mọi file code; git status chỉ có file mới. Báo cáo theo mẫu rồi DỪNG.
```

### PHASE 1 — Khung + paths + metrics
```text
PHASE 1. Theo mục 2, 4.6: tạo khung thư mục đích (chỉ __init__.py rỗng cho các gói chưa tới lượt), configs/, common/paths.py, common/metrics.py.
- KHÔNG tạo thư mục activations/ hay losses/ ở ROOT và KHÔNG tạo common/nn_blocks.py, ở phase này hay bất kỳ phase nào (luật 11). Khối conv, hàm kích hoạt, hàm mất mát sẽ nằm trong từng kiến trúc ở Phase 3 và 4.
- common/metrics.py: COPY compute_segmentation_metrics từ code cũ (giữ nguyên thân hàm, không thêm bớt), có khối if __name__ == "__main__" kèm bootstrap để `python -m common.metrics` chạy selftest.
- Mọi thư mục mã phải có __init__.py.
Tiêu chí xong: `python -m common.metrics` pass; `ls` ở ROOT không có activations/ hay losses/; common/ không chứa lớp nn.Module nào. Báo cáo rồi DỪNG.
```

### PHASE 2 — data_pipeline + engine
```text
PHASE 2. Theo mục 4.4, 6 và 8.6: tạo common/data_pipeline/ (chuyển medical_prep, offline_builder, dataset từ code cũ, sửa import cho đúng, giữ nguyên logic) và common/engine/ (BaseTrainer chuyển từ Trainer cũ — giữ nguyên MỌI tính năng; RunManager; run_config; checkpoint_io; cli_args; utils).
- common/data_pipeline/runtime_data.py (MÃ MỚI, theo đúng mục 8.6): CLI colab | kaggle | local | auto | selftest; ghi data_manifest.json; dò zip trên Drive; liệt kê bệnh nhân trong zip bằng zipfile (bố cục thật cache_1mm/LIDC-IDRI-XXXX/..., dò bằng regex, giải nén phẳng ra --dest/LIDC-IDRI-XXXX/); chọn first/random(seed)/patients-file xác định; giải nén chọn số lượng qua thư mục tạm + os.replace, idempotent; kiểm tra dung lượng; Kaggle: dò cache gốc từ hint (thư mục gốc dataset, thư mục cache hoặc đường dẫn một bệnh nhân) và dựng view-dir bằng symlink; xóa slices.csv cũ khi tập bệnh nhân đổi (sidecar slices.meta.json). Selftest đủ 9 ý ở mục 8.6, không dùng mạng, và tự phá thử 2 chỗ (bỏ os.replace tạm; bỏ sort trước khi chọn) để chứng minh test bắt được.
- BaseTrainer thêm các điểm mở rộng checkpoint_meta / on_epoch_start / build_param_groups; kiểm tra meta khi nạp (arch, model_config) với thông báo lỗi tiếng Việt rõ ràng.
- BaseTrainer nhận `criterion` (nn.Module đã dựng) và `metric_fn` (mặc định common.metrics.compute_segmentation_metrics) từ bên ngoài; KHÔNG import loss nào, không biết tên loss/activation nào.
- common/engine/cli_args.py: add_common_args(parser, *, activation_choices, loss_choices, default_activation, default_loss) — KHÔNG hard-code danh sách activation/loss; kiến trúc gọi sẽ truyền danh sách của riêng nó.
- RunManager tạo thư mục run theo mục 6, ghi run_config.json ngay lúc bắt đầu, cập nhật mỗi epoch, ghi trạng thái cuối kể cả khi bị ngắt/lỗi (atomic).
- Viết tests/test_run_folder.py và tests/test_resume.py (dùng mô hình giả và loss giả như nn.BCEWithLogitsLoss ngay trong test, KHÔNG phụ thuộc hai kiến trúc và KHÔNG import loss/activation của kiến trúc nào).
Tiêu chí xong: 4 selftest dữ liệu pass (medical_prep, offline_builder, dataset, runtime_data; `python -m common.data_pipeline.<module> selftest`), test_run_folder và test_resume pass. Báo cáo rồi DỪNG.
```

### PHASE 3 — unet_base
```text
PHASE 3. Theo mục 4.1, 4.2, 4.3, 5.1, 5.2: tạo unet_base/ đầy đủ (blocks, encoder, decoder, model, activations, losses, trainer, train_main, configs/default.yaml, thư mục notebooks/ và runs/ rỗng có .gitkeep).
- Giữ NGUYÊN tên con: encoder.inc/down1..4, decoder.up1..4/outc. UNetBase có self.encoder và self.decoder.
- Chưa xóa alias cũ ở code cũ (chỉ không xuất hiện trong code mới).
- unet_base/activations.py và unet_base/losses.py theo đúng giao diện mục 4.1, 4.2 (ACTIVATIONS/LOSSES, DEFAULT_ACTIVATION/DEFAULT_LOSS, get_activation/get_loss, list_activations/list_losses): COPY toàn bộ hàm kích hoạt và hàm mất mát có trong code cũ (giữ nguyên thân hàm), mỗi file CHỈ import torch + thư viện chuẩn, không import file nào khác của dự án. Mỗi file có khối if __name__ == "__main__" (kèm bootstrap) chạy selftest; thêm cờ --list để chỉ in tên hợp lệ. Không đưa metric vào losses.py.
- unet_base/blocks.py theo mục 4.3: COPY DoubleConv, OutConv (và mọi khối conv code cũ đang dùng) từ code cũ, giữ nguyên chỉ số Sequential (conv=0, bn=1, act=2, conv=3, bn=4, act=5); act là factory; chỉ import torch + unet_base.activations; có __main__ selftest. encoder.py và decoder.py import khối từ unet_base.blocks, KHÔNG từ common.
- build_unet_base(cfg) truyền get_activation(...) (của unet_base) vào các khối ở encoder và decoder, đúng chỉ số Sequential như bản cũ.
- train_main có selftest/overfit/train với đúng tham số ở mục 5.1 + --features; UNetBaseTrainer dựng criterion bằng unet_base.losses.get_loss và truyền cho BaseTrainer; --activation/--loss lấy choices từ list_activations()/list_losses() của unet_base; dùng RunManager; không sao chép argparse (dùng common/engine/cli_args.py).
- Viết/chạy tests/test_blocks.py phần unet_base (đổi activation không tham số giữ nguyên khóa, có tham số thì thêm khóa), tests/test_registries.py phần unet_base (thêm activation/loss giả vào registry của unet_base → chọn theo tên → ghi vào run_config), tests/test_state_dict_invariance.py phần unet_base (gồm mục 9.2 điểm 5: hàm không tham số giữ nguyên khóa, star_relu thì thêm khóa) và tests/test_independence.py (AST + chặn import mamba_unet/timm/einops/mamba_ssm/transformers theo mục 3.3; phải fail nếu tồn tại ROOT/activations hoặc ROOT/losses).
Tiêu chí xong: `python -m unet_base.blocks`, `python -m unet_base.activations`, `python -m unet_base.losses`, `python -m unet_base.train_main selftest`, test_registries (phần unet_base), invariance unet_base, independence pass. Tự phá thử 2 chỗ (bỏ nạp optimizer khi resume; đổi tên 1 khóa) để chứng minh test bắt được, rồi khôi phục. Báo cáo rồi DỪNG.
```

### PHASE 4 — mamba_unet
```text
PHASE 4. Theo mục 4.1, 4.2, 4.3, 5.1, 5.3: tạo mamba_unet/ đầy đủ: fallbacks.py, backbone/ (layers, mixer, blocks, mambavision, variants, pretrained — TÁCH từ file mamba_vision cũ, giữ nguyên logic và tên thuộc tính), decoder.py, model.py, blocks.py, activations.py, losses.py, trainer.py, train_main.py, configs/default.yaml, notebooks/ và runs/ có .gitkeep.
- mamba_unet/activations.py và mamba_unet/losses.py: bản RIÊNG của mamba_unet theo mục 4.1, 4.2, tạo bằng cách COPY toàn bộ hàm từ code cũ (KHÔNG import từ unet_base, KHÔNG copy-by-reference). Chỉ import torch + thư viện chuẩn; có __main__ selftest và --list. Đầu file activations.py ghi chú: activation chỉ áp dụng cho DECODER; hàm kích hoạt bên trong MambaVision backbone là cố định, không đưa vào registry.
- mamba_unet/blocks.py (ngoài backbone/, đây là khối CNN của DECODER) theo mục 4.3: bản RIÊNG, COPY từ code cũ (KHÔNG import từ unet_base, không hợp nhất với bản của unet_base dù giống hệt), giữ nguyên chỉ số Sequential; chỉ import torch + mamba_unet.activations; có __main__ selftest. Khác với mamba_unet/backbone/blocks.py (khối của MambaVision, giữ nguyên).
- Decoder của mamba_unet dùng khối từ mamba_unet.blocks với factory từ mamba_unet.activations.get_activation; MambaUNetTrainer dựng criterion bằng mamba_unet.losses.get_loss; --activation/--loss lấy choices từ registry của mamba_unet.
- fallbacks.py: selective_scan chọn nhân CUDA chỉ khi import được VÀ có CUDA; còn lại selective_scan_torch; hỗ trợ DUNEAI_MAMBA_BACKEND; mamba_unet phải chạy khi không có timm/einops/mamba_ssm/transformers.
- pretrained.py: đúng yêu cầu mục 5.3 (tempfile, weights_only có fallback, raise khi thiếu >10 khóa). Test bằng file trọng số GIẢ LẬP (không tải mạng).
- trainer.py: lịch đóng băng suy ra từ số epoch; BN backbone giữ eval khi đóng băng; 2 nhóm lr có lr_mult.
- Viết tests/test_scan_fallback.py (so khớp công thức đóng O(L^2) float64, sai khác <= 1e-4; backward hữu hạn); mở rộng tests/test_blocks.py cho mamba_unet (và kiểm tra: sửa/đổi hành vi DoubleConv ở bản này không làm đổi kết quả của bản unet_base); mở rộng tests/test_registries.py cho mamba_unet VÀ thêm kiểm tra cách ly: đăng ký một activation/loss giả vào registry của unet_base thì KHÔNG xuất hiện trong registry của mamba_unet và ngược lại; phần mamba của test_state_dict_invariance + test_independence (cấm import unet_base/timm/einops/mamba_ssm/transformers).
Tiêu chí xong: `python -m mamba_unet.blocks`, `python -m mamba_unet.activations`, `python -m mamba_unet.losses`, `python -m mamba_unet.train_main selftest`, test_registries (đủ hai bên), test_scan_fallback, invariance mamba, independence pass. Selftest phải kiểm tra: backbone KHÔNG đổi giá trị trong epoch đóng băng, và resume qua mốc mở băng vẫn đúng. Tự phá thử 2 chỗ như Phase 3. Báo cáo rồi DỪNG.
```

### PHASE 5 — Đánh giá 3D
```text
PHASE 5. Theo mục 4.5 và 7: tạo common/eval3d/ (chuyển toán 3D và vòng đánh giá từ postprocess cũ; evaluate_dataset_3d và predict_patient_volume NHẬN nn.Module đã nạp trọng số, tuyệt đối không đoán kiến trúc bằng if/elif) và unet_base/evaluate.py, mamba_unet/evaluate.py (selftest | eval | predict).
- Hàm load_*_from_run đọc theo thứ tự run_config.json -> meta checkpoint -> mặc định + bộ chuyển khóa cũ (mục 9.3, chỉ làm khi đã kiểm chứng bằng golden), luôn strict=True.
- eval ghi kết quả vào <run_dir>/eval/ gồm csv + json (tham số đánh giá, git commit, checkpoint dùng).
Tiêu chí xong: `python -m unet_base.evaluate selftest` và `python -m mamba_unet.evaluate selftest` pass; nạp checkpoint tiny qua load_*_from_run strict. Báo cáo rồi DỪNG.
```

### PHASE 6 — scripts, notebook, tài liệu
```text
PHASE 6. Theo mục 8: viết scripts/bootstrap_env.py, push_run_to_github.py (có --dry-run, che token, ngưỡng 95MB, KHÔNG đẩy ckpt có optimizer mặc định), compare_runs.py, make_notebooks.py (nbformat + validate), requirements/*.txt, .gitignore, .gitattributes, README.md (hướng dẫn Local / Colab / Kaggle).
- Sinh 6 notebook: <arch>/notebooks/{train_colab,train_kaggle,evaluate}.ipynb theo cấu trúc ô ở mục 8.2. Notebook chỉ GỌI lệnh Python, không chứa logic. Ô dữ liệu theo mục 8.6: Colab có ô mount Drive rồi gọi `runtime_data colab` (DRIVE_ZIP mặc định "auto"); Kaggle gọi `runtime_data kaggle` với KAGGLE_DATA_HINT mặc định /kaggle/input/datasets/minhtri2475/lidc-idri-1mm-cache-ct208e; evaluate.ipynb gọi `runtime_data auto --patients-file <run_dir>/data_manifest.json`; ô cấu hình có N_PATIENTS, SELECT, SELECT_SEED; CACHE_DIR đọc từ file JSON manifest. Tuyệt đối không viết logic giải nén hay dò đường dẫn trong notebook.
- Notebook mamba: ô cài đặt theo đúng mục 8.2 (KHÔNG pip install mambavision, KHÔNG causal-conv1d, KHÔNG --find-links tới file .whl; wheel mamba-ssm tùy chọn trong try/except; kiểm tra mã thoát pip trước khi in "cài xong").
- Kiểm thử push_run_to_github.py --dry-run bằng một repo git cục bộ giả (tạo trong thư mục tạm), gồm trường hợp file > 95MB.
Tiêu chí xong: `python scripts/make_notebooks.py --check`, dry-run push, `python scripts/compare_runs.py` trên 2 run giả lập pass. Báo cáo rồi DỪNG.
```

### PHASE 7 — Nghiệm thu và dọn dẹp
```text
PHASE 7. Theo mục 10 và 11:
1) Chạy `python tests/run_all.py` và toàn bộ danh sách lệnh ở mục 10; dán bảng kết quả thật.
2) Sửa mọi tham chiếu cũ (import, notebook, README). Sau đó mới XÓA các file cũ trùng lặp và thư mục DuneAI_Remake/ (chỉ khi run_all toàn PASS). Nếu còn nơi gọi alias cũ thì sửa nơi gọi, không giữ alias.
3) Kiểm tra bằng grep: không còn chuỗi "DuneAI_Remake", "from src.", "src/", "from activations", "from losses", "import activations", "import losses" trong code; và ROOT KHÔNG có thư mục activations/ hay losses/ (chỉ có unet_base/activations.py, unet_base/losses.py, mamba_unet/activations.py, mamba_unet/losses.py); không còn common/nn_blocks.py và không còn lớp nn.Module nào trong common/.
4) Chạy lại run_all.py sau khi xóa.
5) Nhắc tôi hai việc thủ công: chuyển gốc git lên SOURCES/ (mục 12.1) và quyết định LFS (mục 12.2).
Báo cáo cuối: tóm tắt cây thư mục, bảng test, toàn bộ DEVIATIONS.md.
```

---

## PROMPT KHI NGHI NGỜ (dán khi Agent báo pass mà thiếu bằng chứng, hoặc có dấu hiệu bịa)
```text
Dừng. Tôi chưa thấy bằng chứng. Hãy chạy lại các lệnh test của phase vừa rồi ngay bây giờ trong Terminal và dán nguyên văn 10 dòng cuối của MỖI lệnh, kèm exit code (echo $?). Sau đó chạy `git status` và `git diff --stat`. Không tóm tắt, không diễn giải trước khi dán output.
```

## PROMPT KHI TEST FAIL
```text
Test fail. Đừng sửa test cho khớp code. Hãy: (1) dán nguyên lỗi, (2) nêu NGUYÊN NHÂN GỐC bằng cách đọc code liên quan, (3) đối chiếu với spec và với golden/code cũ xem phần nào mới là đúng, (4) chỉ sửa ĐÚNG chỗ gây lỗi, (5) chạy lại và dán output. Nếu cách sửa làm khác spec hoặc đổi tên khóa state_dict thì DỪNG và hỏi tôi.
```
