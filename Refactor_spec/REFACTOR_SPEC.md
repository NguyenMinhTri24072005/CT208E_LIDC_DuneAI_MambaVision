# SPEC TÁI CẤU TRÚC DỰ ÁN DuneAI (v2.1) — HAI KIẾN TRÚC ĐỘC LẬP

> Tài liệu này thay thế hoàn toàn plan cũ. Đặt vào repo tại `docs/REFACTOR_SPEC.md`. AI Agent phải đọc toàn bộ trước khi sửa bất kỳ file nào.
> Quy ước: tên hàm/lớp/file viết tiếng Anh; chú thích và thông báo cho người dùng viết tiếng Việt.
>
> **v2.1 — sửa một hiểu nhầm của v2:** hàm kích hoạt (activation) và hàm mất mát (loss) **không** nằm ở ROOT dùng chung. **Mỗi kiến trúc có `activations.py` và `losses.py` riêng ngay trong thư mục của nó** (xem R11, mục 4.1–4.2). Chỉ có *metric đánh giá* là dùng chung (`common/metrics.py`) để hai kiến trúc so sánh bằng cùng một thước đo.
>
> **Nguyên tắc phân chia (chốt ở v2.1):** *mọi thứ thuộc cấu trúc mạng* — khối tích chập (`DoubleConv`, `OutConv`), encoder, decoder, backbone, mô hình, hàm kích hoạt, hàm mất mát — nằm **trong thư mục từng kiến trúc**, mỗi bên một bản riêng. `common/` chỉ chứa thứ **trước khi train** (xử lý dữ liệu), **trong khi train** (vòng lặp, quản lý run), **sau khi train** (đánh giá 3D, metric), và **cấu hình/đường dẫn**. `common/` không chứa bất kỳ lớp `nn.Module` nào.

---

## 0. KHÁC BIỆT SO VỚI PLAN CŨ (đọc để không lặp lại sai lầm)

| # | Plan cũ làm | Yêu cầu thực của chủ dự án | Spec này |
|---|---|---|---|
| 1 | Mọi thứ nằm trong `DuneAI_Remake/`, `common/` đứng cùng cấp hai kiến trúc | Hai kiến trúc **độc lập, ngang hàng, nằm trực tiếp dưới `SOURCES/`** (thư mục gốc mã nguồn) | `ROOT = SOURCES/`; `unet_base/` và `mamba_unet/` là thư mục cấp 1; thư mục `DuneAI_Remake/` **bị giải thể** |
| 2 | `Trainer` + CLI dùng chung đặt ở `common/`; `train.py` mỏng | **Mỗi** thư mục kiến trúc có `trainer.py`, `train_main.py` và thư mục `notebooks/` riêng | Vòng lặp lõi ở `common/engine/`; mỗi kiến trúc có `trainer.py` (phần đặc thù) + `train_main.py` + `notebooks/` |
| 3 | Notebook gom ở `notebooks/` gốc | Notebook nằm trong từng kiến trúc | `<arch>/notebooks/` |
| 4 | Hàm kích hoạt / mất mát nằm lẫn trong `common/`. *(Bản v2 đầu tiên hiểu nhầm là phải gom ra `activations/`, `losses/` ở ROOT dùng chung cho cả hai kiến trúc.)* | **Mỗi kiến trúc có hàm kích hoạt và hàm mất mát RIÊNG, nằm ngay trong thư mục kiến trúc**, để chỉnh/thử nghiệm bên này không ảnh hưởng bên kia | `unet_base/activations.py`, `unet_base/losses.py`, `mamba_unet/activations.py`, `mamba_unet/losses.py`; mỗi file tự có **registry** chọn theo tên. **Không** có `activations/`, `losses/` ở ROOT |
| 5 | `--save-dir` cố định, lần sau ghi đè lần trước | Mỗi lần train 1 thư mục riêng + file ghi đủ tham số để so sánh | `<arch>/runs/<run_id>/` + `run_config.json` + `scripts/compare_runs.py` |
| 6 | Không nói tới Colab/Kaggle/GitHub | Code trên PC → push GitHub → clone vào Colab/Kaggle; Colab lưu trọng số lên Drive, Kaggle push trọng số ngược về GitHub | Mục 8 (kèm xử lý giới hạn 100 MB/file của GitHub) |
| 7 | Tên khóa `state_dict` ghi tay trong plan | Trọng số đã train trên Kaggle phải nạp lại được | Mục 9: sinh **golden file từ code cũ** ở Phase 0, không đoán tên khóa |
| 8 | Đánh giá "đoán" kiến trúc / chỉ nhắc có CLI | Đánh giá viết bằng Python, notebook chỉ gọi | `<arch>/evaluate.py`; mô hình dựng lại **từ `run_config.json`**, không đoán |
| 9 | *(v2/v2.1 đầu)* `DoubleConv`, `OutConv` đặt ở `common/nn_blocks.py` dùng chung | Khối tích chập là **cấu trúc mạng** → mỗi bên phải tự chỉnh được mà không ảnh hưởng bên kia | `unet_base/blocks.py` và `mamba_unet/blocks.py`, mỗi bên một bản riêng; **không** còn `common/nn_blocks.py` |

---

## 1. NGUYÊN TẮC BẤT BIẾN (vi phạm = lỗi)

- **R1 Độc lập.** `unet_base/` và `mamba_unet/` **không bao giờ import lẫn nhau**. Cả hai chạy được khi thư mục kia bị xóa.
- **R2 Chiều phụ thuộc một chiều.** Kiến trúc → được import `common`, `configs` và **các file của chính nó** (kể cả `<arch>.activations`, `<arch>.losses`). Các gói dùng chung (`common`, `configs`) **không bao giờ import** `unet_base`/`mamba_unet`.
- **R3 Bảo toàn `state_dict`.** Tập khóa + shape của mô hình mới trùng 100% bản cũ; checkpoint cũ nạp `strict=True` và đầu ra `allclose(atol=1e-6)` (mục 9).
- **R4 Điều khiển bằng cấu hình.** Thứ tự ưu tiên: mặc định trong code < file YAML (`--config`) < tham số dòng lệnh. Cấu hình đã "giải" xong được ghi vào `run_config.json`.
- **R5 Luôn chạy từ ROOT.** Cách chuẩn: `python -m unet_base.train_main train ...`. Mọi file có `if __name__ == "__main__"` phải có đoạn bootstrap chèn `ROOT` vào `sys.path` để `python unet_base/train_main.py` cũng chạy được (mục 3.2).
- **R6 Không hard-code đường dẫn.** Cấm xuất hiện chuỗi `DuneAI_Remake`, `SOURCES`, `/content/...`, `/kaggle/...`, `D:\...` trong code. `ROOT` lấy từ `common/paths.py`. **Ngoại lệ duy nhất (chủ đích):** hằng số đặt tên cho nền tảng trong `common/data_pipeline/runtime_data.py`, và giá trị mặc định trong ô cấu hình của notebook do `make_notebooks.py` sinh ra (mục 8.6). Nơi khác không được.
- **R7 Không viết lại thuật toán.** Chỉ **di chuyển/tách file** logic đã chạy được. Mọi thay đổi hành vi (kể cả "dọn cho gọn") phải ghi vào `docs/DEVIATIONS.md` kèm lý do. Nếu spec yêu cầu thứ code cũ không có → **dừng và hỏi**, không tự bịa.
- **R8 Phụ thuộc nặng là tùy chọn.** `unet_base` chạy được khi **không** có `timm`, `einops`, `mamba_ssm`, `transformers`. `mamba_unet` cũng chạy được khi không có chúng (dùng bản dự phòng thuần PyTorch).
- **R9 Bí mật.** Không ghi/in token GitHub ở bất kỳ đâu (log, `run_config.json`, `.git/config`).
- **R10 Từng phase một.** Mỗi phase có tiêu chí hoàn thành; xong phải chạy test, commit riêng, báo cáo, **dừng chờ lệnh**.
- **R11 Mọi thứ thuộc cấu trúc mạng nằm trong từng kiến trúc.** Mỗi kiến trúc sở hữu riêng: `blocks.py` (khối tích chập), `activations.py`, `losses.py`, `encoder`, `decoder`, `model` — mỗi bên một bản. Cấm tạo thư mục `activations/`, `losses/` ở ROOT; cấm `common/` chứa **bất kỳ lớp `nn.Module` nào** (khối conv, loss, activation, encoder, decoder). Hai kiến trúc **được phép có mã trùng nhau** ở các file này (vd. cùng một `DoubleConv`) — đó là chủ đích (sửa một bên không làm lệch bên kia); **không** "khử trùng lặp" bằng cách gom ra chung.

---

## 2. CÂY THƯ MỤC MỤC TIÊU

```text
SOURCES/                               ← ROOT (cũng là gốc git). KHÔNG còn thư mục DuneAI_Remake/
├── README.md
├── .gitignore   .gitattributes
├── requirements/
│   ├── base.txt                       # torch, numpy, pandas, scipy, opencv-python-headless, SimpleITK,
│   │                                  # scikit-image, tqdm, pyyaml, matplotlib, nbformat
│   ├── unet_base.txt                  # -r base.txt
│   └── mamba_unet.txt                 # -r base.txt   (timm/einops/mamba-ssm chỉ là tùy chọn, ghi chú trong file)
│
├── configs/
│   └── config.py                      # HẰNG SỐ Y KHOA dùng chung (cửa sổ HU, IMAGE_SIZE, ngưỡng lát...). Không chứa siêu tham số thí nghiệm
│
├── common/                            # ── LÕI DÙNG CHUNG: tiền xử lý / vòng lặp train / đánh giá / đường dẫn. KHÔNG chứa lớp nn.Module nào ──
│   ├── __init__.py
│   ├── paths.py                       # REPO_ROOT, default_data_dir(), default_runs_root(arch)
│   ├── metrics.py                     # compute_segmentation_metrics — METRIC ĐÁNH GIÁ dùng chung (không phải loss)
│   ├── data_pipeline/
│   │   ├── medical_prep.py            # cửa sổ HU, tách phổi 3D, pad/crop 1mm, window_hu_torch
│   │   ├── offline_builder.py         # lọc lát cắt, xuất slices.csv
│   │   ├── dataset.py                 # LIDC25DDataset, create_dataloaders
│   │   └── runtime_data.py            # chuẩn bị DỮ LIỆU THEO NỀN TẢNG: Colab (giải nén chọn số lượng từ zip trên Drive), Kaggle (dò /kaggle/input + symlink), local (mục 8.6)
│   ├── engine/
│   │   ├── base_trainer.py            # BaseTrainer (vòng lặp lõi, chuyển từ Trainer cũ)
│   │   ├── run_manager.py             # tạo/đóng thư mục run, ghi run_config.json, tìm run để resume
│   │   ├── run_config.py              # lược đồ run_config + thu thập môi trường/git/dữ liệu
│   │   ├── checkpoint_io.py           # load_checkpoint_file, save atomic, meta kiểm tra khi nạp, export_weights_fp16
│   │   ├── cli_args.py                # các nhóm tham số argparse DÙNG CHUNG cho cả hai kiến trúc
│   │   └── utils.py                   # seed_everything, format_duration, atomic_csv_save
│   └── eval3d/
│       ├── postprocess3d.py           # unpad_or_uncrop_xy, resize_from_1mm_3d, postprocess_3d_mask
│       ├── metrics3d.py               # compute_3d_metrics, _compute_hd95_mm
│       ├── volume_inference.py        # predict_patient_volume (nhận nn.Module đã nạp trọng số)
│       ├── nrrd_io.py                 # save_prediction_nrrd
│       └── eval_runner.py             # evaluate_dataset_3d(model, ...)  — KHÔNG đoán kiến trúc
│
├── unet_base/                         # ══ KIẾN TRÚC 1: U-NET CNN THUẦN (5 tầng) ══
│   ├── __init__.py                    # xuất UNetBase, build_unet_base
│   ├── encoder.py                     # CNNBaselineEncoder
│   ├── decoder.py                     # UpBlock, ClassicUNetDecoder
│   ├── blocks.py                      # ── KHỐI CNN CỦA U-NET (chỉnh ở đây): DoubleConv, OutConv
│   ├── model.py                       # UNetBase, build_unet_base(cfg)
│   ├── activations.py                 # ── HÀM KÍCH HOẠT CỦA U-NET (chỉnh ở đây): StarReLU, SoftClampReLU... + ACTIVATIONS + get_activation
│   ├── losses.py                      # ── HÀM MẤT MÁT CỦA U-NET (chỉnh ở đây): Dice, Tversky, DiceBCE... + LOSSES + get_loss
│   ├── trainer.py                     # UNetBaseTrainer(BaseTrainer): dựng model/optimizer/meta riêng
│   ├── train_main.py                  # CLI: selftest | overfit | train
│   ├── evaluate.py                    # CLI: selftest | eval | predict   (Python thuần)
│   ├── configs/
│   │   ├── default.yaml
│   │   └── experiments/               # mỗi file = một thí nghiệm (vd. f32_gelu.yaml)
│   ├── notebooks/
│   │   ├── train_colab.ipynb
│   │   ├── train_kaggle.ipynb
│   │   └── evaluate.ipynb
│   └── runs/                          # mỗi lần train một thư mục (mục 6)
│
├── mamba_unet/                        # ══ KIẾN TRÚC 2: MAMBAVISION + U-NET ══
│   ├── __init__.py                    # xuất MambaUNet, build_mamba_unet
│   ├── fallbacks.py                   # chọn backend selective scan; DropPath, LayerNorm2d, trunc_normal_, rearrange/repeat dự phòng
│   ├── backbone/                      # (gói, KHÔNG phải một file khổng lồ)
│   │   ├── __init__.py                # MambaVision, build_mambavision_backbone, VARIANTS
│   │   ├── layers.py                  # window_partition/reverse, Downsample, PatchEmbed, ConvBlock
│   │   ├── mixer.py                   # MambaVisionMixer, Attention
│   │   ├── blocks.py                  # Block, MambaVisionLayer
│   │   ├── mambavision.py             # class MambaVision (forward → 4 feature map)
│   │   ├── variants.py                # mamba_vision_T/T2/S/B/L/L2 + bảng kênh cơ sở
│   │   └── pretrained.py              # tải + nạp trọng số NVIDIA (mục 5.3)
│   ├── decoder.py                     # UpBlock, UNetDecoder
│   ├── blocks.py                      # ── KHỐI CNN CỦA DECODER MAMBA-UNET (chỉnh ở đây): DoubleConv, OutConv — bản riêng, tách khỏi unet_base
│   ├── model.py                       # MambaUNet, build_mamba_unet(cfg)
│   ├── activations.py                 # ── HÀM KÍCH HOẠT CỦA DECODER MAMBA-UNET (chỉnh ở đây); backbone MambaVision giữ cố định
│   ├── losses.py                      # ── HÀM MẤT MÁT CỦA MAMBA-UNET (chỉnh ở đây) — bản riêng, tách khỏi unet_base
│   ├── trainer.py                     # MambaUNetTrainer: lịch đóng băng backbone, nhóm lr, meta
│   ├── train_main.py
│   ├── evaluate.py
│   ├── configs/   notebooks/   runs/  # như unet_base
│
├── scripts/
│   ├── bootstrap_env.py               # phát hiện Colab/Kaggle/local, cài requirements, in bảng môi trường
│   ├── push_run_to_github.py          # Kaggle: đẩy kết quả một lần train lên GitHub (mục 8.4)
│   ├── compare_runs.py                # gom mọi run_config.json + history.csv → bảng so sánh
│   ├── make_notebooks.py              # SINH các .ipynb từ mẫu bằng nbformat, rồi nbformat.validate
│   ├── export_golden.py               # Phase 0: xuất golden từ CODE CŨ
│   └── check_legacy_checkpoint.py     # nạp checkpoint cũ vào model mới (strict=True)
│
├── tests/
│   ├── golden/                        # sinh ở Phase 0
│   ├── test_independence.py
│   ├── test_state_dict_invariance.py
│   ├── test_registries.py
│   ├── test_blocks.py
│   ├── test_run_folder.py
│   ├── test_resume.py
│   ├── test_scan_fallback.py
│   └── run_all.py                     # chạy tất cả, in bảng PASS/FAIL, exit code ≠ 0 nếu có lỗi
│
├── docs/    (REFACTOR_SPEC.md, INVENTORY.md, DEVIATIONS.md — hai file sau do Agent tạo)
└── data/                              # (gitignored) cache_1mm/ ...   ← KHÁC với common/data_pipeline/ (đó là mã)
```


**Quy tắc đóng gói (đã kiểm chứng):** mọi thư mục chứa mã (`common`, `common/data_pipeline`, `common/engine`, `common/eval3d`, `unet_base`, `mamba_unet`, `mamba_unet/backbone`, `scripts`, `tests`) phải có `__init__.py`. Gói nào được chạy bằng `python -m <gói>` (không kèm tên module) **bắt buộc** có `__main__.py` — thiếu thì Python báo *"is a package and cannot be directly executed"*. Module chạy bằng `python -m <gói>.<module>` (vd. `python -m unet_base.blocks`, `python -m unet_base.activations`, `python -m unet_base.losses`, `python -m common.metrics`) thì module đó cần khối `if __name__ == "__main__":` (kèm bootstrap mục 3.2).

---

## 3. QUY TẮC IMPORT VÀ CÁCH CHẠY

### 3.1 Ma trận phụ thuộc
| Gói | Được import |
|---|---|
| `common` | `configs` (+ torch/thư viện chuẩn). **Không** import `unet_base`/`mamba_unet`; **không** định nghĩa lớp `nn.Module` nào |
| `unet_base` | `common`, `configs`, và các file **của chính nó** (`unet_base.activations`, `unet_base.losses`, ...) |
| `mamba_unet` | `common`, `configs`, và các file **của chính nó** (`mamba_unet.activations`, `mamba_unet.losses`, ...) |
| `<arch>/activations.py`, `<arch>/losses.py` | chỉ `torch` + thư viện chuẩn — **tự đủ**, không import file nào khác của dự án |
| `<arch>/blocks.py` | `torch` + thư viện chuẩn + `<arch>.activations` (chính kiến trúc đó) |
| `scripts`, `tests` | tất cả |

`tests/test_independence.py` quét AST và **fail** nếu có import vi phạm bảng trên. Ngoài ra phải **fail** nếu: (a) tồn tại thư mục `ROOT/activations` hoặc `ROOT/losses`; (b) có `import activations` / `from losses ...` kiểu cũ ở bất kỳ đâu; (c) `common/` định nghĩa bất kỳ lớp nào kế thừa `nn.Module` (quét AST) — khối conv, loss, activation, encoder, decoder đều không được ở đây.

### 3.2 Đoạn bootstrap bắt buộc ở đầu mọi file có `__main__`
```python
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]      # chỉnh số cấp cho đúng vị trí file; file ở ROOT/<arch>/x.py → parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
```

### 3.3 Kiểm thử độc lập (cách làm cụ thể)
Chạy selftest của một kiến trúc trong tiến trình **cấm** import kiến trúc kia và mọi thư viện tùy chọn:
```python
import sys, runpy
for name in ("mamba_unet", "timm", "einops", "mamba_ssm", "transformers"):
    sys.modules[name] = None          # import sẽ ném ImportError
sys.argv = ["x", "selftest"]
runpy.run_module("unet_base.train_main", run_name="__main__")
```
Làm tương tự cho `mamba_unet` (cấm `unet_base`, `timm`, `einops`, `mamba_ssm`, `transformers`).

---

## 4. ĐẶC TẢ: GIAO DIỆN CHUẨN CHO TỪNG KIẾN TRÚC (4.1–4.3) VÀ CÁC GÓI DÙNG CHUNG (4.4–4.6)

> **4.1, 4.2, 4.3 là bản mẫu giao diện, KHÔNG phải gói dùng chung.** Mỗi kiến trúc (`unet_base`, `mamba_unet`) tạo **một bản riêng** của ba file này trong thư mục của mình. Không có `activations/`, `losses/` ở ROOT và không có `common/nn_blocks.py`.

### 4.1 `<arch>/activations.py` — mỗi kiến trúc một file
```python
ACTIVATIONS: dict[str, ActivationSpec]          # ActivationSpec(factory, has_params: bool, doc: str)
DEFAULT_ACTIVATION: str                         # = hàm của code cũ cho kiến trúc này (Agent xác định ở Phase 0)
def get_activation(name: str, **kwargs) -> Callable[[], nn.Module]
    # trả về FACTORY không đối số: mỗi lần gọi tạo một nn.Module mới (để dùng cho nhiều lớp)
def list_activations() -> list[str]
```
- **Một file duy nhất** chứa cả các `nn.Module` tự viết (StarReLU, SoftClampReLU, ...) lẫn bảng đăng ký ở cuối file. **Thêm hàm kích hoạt mới = viết 1 lớp + thêm 1 dòng vào `ACTIVATIONS`**, không sửa nơi nào khác.
- **Khởi tạo:** khi tạo file cho từng kiến trúc (P3, P4), **sao chép** toàn bộ hàm kích hoạt có trong code cũ vào cả hai file (không mất khả năng nào); từ đó mỗi bên được thêm/bớt độc lập. Chỉ import `torch` + thư viện chuẩn.
- Tên tối thiểu: `relu`, `leaky_relu`, `gelu`, `silu`, `star_relu`, `soft_clamp_relu` (+ mọi hàm đang có trong code cũ; Agent đối chiếu ở Phase 0).
- **Quy tắc giữ khóa (đã kiểm chứng):** hàm **không tham số** (ReLU, GELU, SiLU, SoftClampReLU) đổi qua lại **không làm đổi** khóa `state_dict`. Hàm **có tham số học được** (`star_relu`, `prelu`) **thêm khóa** (vd. `2.scale`, `2.bias`) → checkpoint cũ không nạp được vào mô hình dùng chúng. Vì vậy:
  - `has_params=True` phải được khai báo trong registry;
  - activation phải đứng **đúng chỉ số con** trong `nn.Sequential` như bản cũ (không bọc thêm lớp làm lệch chỉ số);
  - tên activation được ghi vào `run_config.json` **và** `meta` của checkpoint; `evaluate.py` dựng lại đúng theo đó;
  - mặc định của mỗi kiến trúc = đúng hàm của code cũ (`DEFAULT_ACTIVATION`), checkpoint cũ không có `meta` thì giả định mặc định này.
- **Phạm vi:** `unet_base` — mọi `DoubleConv` trong `unet_base/blocks.py` (dùng ở encoder và decoder). `mamba_unet` — **chỉ decoder** (qua `mamba_unet/blocks.py`). Hàm kích hoạt **bên trong MambaVision backbone** (GELU/SiLU của kiến trúc gốc) là **cố định**, không cho chỉnh (đổi sẽ phá trọng số pretrained); ghi chú điều này ở đầu `mamba_unet/activations.py`.
- Chạy: `python -m <arch>.activations` = selftest; `python -m <arch>.activations --list` = chỉ in tên hợp lệ.

### 4.2 `<arch>/losses.py` — mỗi kiến trúc một file
```python
LOSSES: dict[str, LossSpec]                     # LossSpec(factory, doc)
DEFAULT_LOSS: str                               # = loss của code cũ cho kiến trúc này; DEFAULT_LOSS_PARAMS: dict
def get_loss(name: str, **params) -> nn.Module  # forward(logits, targets) -> scalar tensor
def list_losses() -> list[str]                  # ít nhất: dice, bce, dice_bce, tversky, focal_tversky (+ mọi loss đang có trong code cũ)
```
- **Một file duy nhất** chứa các lớp loss (DiceLoss, TverskyLoss, TverskyFocalLoss, DiceBCELoss, ...) và bảng `LOSSES`. **Thêm loss mới = 1 lớp + 1 dòng đăng ký**; **không** sửa Trainer hay `train_main`.
- **Khởi tạo:** như 4.1 — sao chép toàn bộ loss của code cũ vào cả hai file (P3, P4), sau đó mỗi bên chỉnh độc lập. Chỉ import `torch` + thư viện chuẩn. **Không** chứa metric (metric ở `common/metrics.py`, mục 4.6).
- Dùng từ CLI: `--loss dice_bce --loss-params '{"bce_weight":0.5,"dice_weight":0.5}'` hoặc YAML `loss: {name: dice_bce, params: {...}}`.
- Tên loss + tham số được ghi vào `run_config.json`.
- Tính toán nội bộ ép `logits.float()` (an toàn khi bật AMP).
- Chạy: `python -m <arch>.losses` = selftest; `--list` = chỉ in tên.

### 4.3 `<arch>/blocks.py` — mỗi kiến trúc một file
```python
class DoubleConv(nn.Module):  # (in, out, mid=None, act=<factory>)
class OutConv(nn.Module):     # (in, out)
```
- **Cấu trúc mạng là của từng kiến trúc**, nên khối conv không dùng chung. Mỗi bên có `blocks.py` riêng; muốn đổi kernel, thêm residual/attention, đổi norm, đổi số lớp conv... ở bên nào thì chỉ sửa file của bên đó.
- **Khởi tạo (P3, P4):** **sao chép** `DoubleConv`/`OutConv` (và bất kỳ khối conv nào code cũ đang dùng, vd. `ConvBnAct`) từ code cũ vào **cả hai** `blocks.py`. Agent **không** hợp nhất hai bản dù chúng giống hệt, và **không** import từ kiến trúc kia. Nếu code cũ của hai bên dùng khối khác nhau, giữ đúng khối của từng bên.
- **Bảo toàn khóa:** cấu trúc `Sequential` phải **giữ nguyên chỉ số** như bản cũ (conv=0, bn=1, act=2, conv=3, bn=4, act=5). `act` nhận **factory** (từ `<arch>.activations.get_activation`); giá trị mặc định = `DEFAULT_ACTIVATION` của chính kiến trúc.
- Import cho phép: `torch`, thư viện chuẩn, `<arch>.activations`. `UpBlock` vẫn nằm trong `decoder.py` của từng bên.
- Chạy: `python -m <arch>.blocks` = selftest (forward shape đúng; đổi activation không tham số giữ nguyên tập khóa; activation có tham số thì thêm khóa).

### 4.4 `common/engine/`
**`BaseTrainer`** — chuyển từ `Trainer` hiện có, **giữ nguyên** mọi tính năng: tự dùng `nn.DataParallel` khi nhiều GPU, AMP, `window_hu_torch` trên GPU, lưu atomic (`os.replace`), gradient clipping, thanh tiến trình, ETA, early stopping, bắt `KeyboardInterrupt`, `resume_from` (kể cả `"auto"`), đồng bộ `T_max` cosine khi tăng `--epochs`, kiểm tra `meta` khi nạp. Điểm mở rộng cho lớp con:
```python
class BaseTrainer:
    def checkpoint_meta(self) -> dict: ...          # lớp con trả {"arch":..., "model_config": {...}, ...}
    def on_epoch_start(self, epoch: int): ...       # lớp con tùy ý (vd. đóng/mở băng backbone)
    def build_param_groups(self, lr) -> list[dict]: ...   # mặc định 1 nhóm; mamba trả 2 nhóm
```
**Tiêm phụ thuộc (không import loss):** `BaseTrainer.__init__` nhận `criterion` (nn.Module đã dựng) và `metric_fn` (mặc định `common.metrics.compute_segmentation_metrics`). Lớp con kiến trúc dựng `criterion` bằng `<arch>.losses.get_loss` và dựng mô hình (qua `<arch>.blocks`, `<arch>.activations`) của chính nó. `BaseTrainer` không biết tên loss/activation nào.

Khóa `meta` bắt buộc khớp khi nạp: `arch`, `model_config` (đủ để dựng lại mô hình). Sai → `ValueError` nêu rõ khác ở đâu và gợi ý sửa tham số hoặc dùng run mới.

**`RunManager`**: `create_run(arch, runs_root, run_name=None, tag=None, resolved_config) -> RunHandle`; `find_resumable(arch, runs_root)`; `resolve_resume(arg, arch, runs_root)`; `finish(status, result)`. Ghi `run_config.json` **ngay khi bắt đầu** (`status="running"`), cập nhật `result` sau mỗi epoch, ghi trạng thái cuối khi kết thúc hoặc khi gặp ngoại lệ/ngắt. Mọi lần ghi đều atomic.

**`cli_args.py`**: các nhóm tham số dùng chung để hai CLI **giống hệt nhau** (mục 5.1). Cấm sao chép `argparse` vào từng kiến trúc. `--activation` và `--loss` **không** hard-code `choices` trong `cli_args.py`: hàm `add_common_args(parser, *, activation_choices, loss_choices, default_activation, default_loss)` nhận danh sách từ registry **của kiến trúc đang gọi** (`list_activations()`, `list_losses()`, `DEFAULT_*`).

### 4.5 `common/eval3d/`
Chỉ chứa toán học 3D và vòng đánh giá, **nhận `nn.Module` đã dựng và nạp trọng số**. Cấm `if/elif` đoán kiến trúc; cấm import `unet_base`/`mamba_unet`.

### 4.6 `common/metrics.py`
`compute_segmentation_metrics(logits, targets, threshold=0.5) -> dict` (intersection, pred_sum, target_sum, ...), chuyển **nguyên văn** từ code cũ (không thêm bớt gì; nếu code cũ chưa ép `logits.float()` thì ghi nhận vào `DEVIATIONS.md` chứ không tự sửa). 
**Vì sao metric dùng chung còn loss thì riêng:** metric là *thước đo* để so sánh hai kiến trúc nên phải giống hệt nhau; loss/activation là *đối tượng tinh chỉnh* nên mỗi kiến trúc tự sở hữu. Chạy `python -m common.metrics` = selftest.

---

## 5. ĐẶC TẢ HAI KIẾN TRÚC

### 5.1 CLI chung (cả hai kiến trúc)
Chế độ: `selftest` | `overfit` | `train` (+ `evaluate.py`: `selftest` | `eval` | `predict`).

| Nhóm | Tham số |
|---|---|
| Cấu hình | `--config path.yaml` |
| Dữ liệu | `--cache-dir`, `--data-manifest`, `--workers`, `--gpu-window`, `--no-lung-mask`, `--no-augment`, `--limit-train N`, `--limit-val N`, `--rebuild-index` |
| Mô hình chung | `--activation` |
| Loss | `--loss`, `--loss-params` (chuỗi JSON) |
| Tối ưu | `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--scheduler {cosine,plateau,step,none}`, `--grad-clip`, `--patience` |
| Chạy | `--runs-root`, `--run-name`, `--tag`, `--notes`, `--resume {auto \| <run_dir> \| <file.pth>}`, `--init-weights <file.pth>`, `--seed`, `--no-amp`, `--no-progress` |

- **Không** có tham số trùng nghĩa (bỏ `--auto-resume`; `--resume auto` đã đủ). Không alias.
- `--epochs` luôn là **tổng** số epoch đích (kể cả khi resume).
- `--cache-dir` là thư mục **đã chuẩn bị** do `runtime_data` trả về (mục 8.6). `--data-manifest` mặc định = `<cache-dir>/data_manifest.json` nếu có; dùng để điền `run_config.data.source` và chép vào thư mục run. Không có manifest → `source.kind = "local_dir"`.
- `--activation`, `--loss` được kiểm tra theo registry **của chính kiến trúc đang chạy** (`unet_base.train_main` chỉ thấy `unet_base.activations/losses`; `mamba_unet.train_main` chỉ thấy bản của nó). Tên hợp lệ có thể khác nhau giữa hai kiến trúc; `--help` liệt kê đúng danh sách của kiến trúc đó; tên sai → lỗi nêu rõ danh sách hợp lệ. Mặc định = `DEFAULT_ACTIVATION` / `DEFAULT_LOSS` trong file của kiến trúc.
- Mỗi kiến trúc thêm tham số riêng ở dưới.

### 5.2 `unet_base/`
- **Tham số riêng:** `--features` (mặc định `64`).
- `encoder.py`: `CNNBaselineEncoder` trả 5 feature map `[H, H/2, H/4, H/8, H/16]`; tên con giữ nguyên `inc, down1, down2, down3, down4`.
- `decoder.py`: `ClassicUNetDecoder`; tên con `up1, up2, up3, up4, outc`.
- `model.py`: duy nhất `UNetBase` với **hai thuộc tính `self.encoder` và `self.decoder`**; xóa alias (`DuneUNet`, `PureUNet`, `ClassicUNet`, `BaselineUNet`) **sau khi** `grep` mọi nơi gọi và sửa chúng.
- Nhánh `use_stride4_stem` chỉ được xóa khi: (a) không file/notebook/config nào truyền nó, (b) golden keys không đổi. Ghi vào `DEVIATIONS.md`.
- Mặc định `--save`: `<runs-root>/<run_id>`; `runs-root` mặc định `unet_base/runs`.
- `activations.py`, `losses.py`: theo mục 4.1, 4.2 (bản **riêng** của `unet_base`). `UNetBaseTrainer` dùng `unet_base.losses.get_loss`; `build_unet_base(cfg)` dùng `unet_base.activations.get_activation` truyền vào `unet_base.blocks.DoubleConv` (encoder + decoder). Không import gì từ `mamba_unet`.
- `blocks.py`: theo mục 4.3 (bản **riêng** của `unet_base`: `DoubleConv`, `OutConv`). `encoder.py` và `decoder.py` import khối từ `unet_base.blocks`, không từ `common`.

### 5.3 `mamba_unet/`
- **Tham số riêng:** `--backbone {mamba_vision_T,mamba_vision_T2,mamba_vision_S,mamba_vision_B}`, `--pretrained / --no-pretrained`, `--freeze-backbone-epochs N` (`0`=không đóng băng, `N>0`=N epoch đầu, `-1`=đóng băng suốt), `--backbone-lr-mult` (mặc định `0.1`, chỉ áp dụng khi pretrained; scratch dùng `1.0`).
- **Thuộc tính:** `MambaUNet.encoder` (`MambaVision`), `MambaUNet.decoder` (`UNetDecoder`). Backbone trả 4 map `[H/4, H/8, H/16, H/32]` với số kênh `dim·2^i` (T/T2: 80,160,320,640 · S: 96… · B: 128… · L/L2: 196…). Xóa nhánh `backbone in ("cnn","cnn_baseline")` và alias `ModularUNet`.
- **`fallbacks.py` — chọn backend selective scan** (đã kiểm chứng từ mã nguồn MambaVision: mô hình chỉ gọi **một hàm** `selective_scan_fn` và **không dùng** `causal-conv1d`):
  - dùng nhân CUDA `mamba_ssm` **chỉ khi** import được **và** `torch.cuda.is_available()`; ngược lại dùng `selective_scan_torch` thuần PyTorch;
  - biến môi trường `DUNEAI_MAMBA_BACKEND=torch|cuda` để ép; ép `cuda` mà không có GPU → `RuntimeError` rõ ràng;
  - `selective_scan_torch` tính bằng float32, hỗ trợ `B,C` dạng `(B,N,L)` và `(B,G,N,L)`, `z`, `delta_bias`, `delta_softplus`, `return_last_state`;
  - **tiêu chí nghiệm thu:** khớp công thức đóng O(L²) tính độc lập bằng float64 với sai khác ≤ 1e-4 (bản tham chiếu đã đo ~1e-6); backward cho gradient hữu hạn.
- **`backbone/pretrained.py`:**
  - tải trọng số NVIDIA vào `Path(tempfile.gettempdir())` (không dùng `/tmp` cứng — Windows không có);
  - đọc bằng `torch.load(..., weights_only=True)`; nếu bị từ chối (PyTorch ≥ 2.6 với file có đối tượng như `Namespace`) thì đọc lại `weights_only=False` kèm thông báo (file chính chủ NVIDIA);
  - lấy `state_dict` từ khóa `"state_dict"`/`"model"` nếu có; bỏ tiền tố `module.`; nạp `strict=False`, in số khóa thiếu/thừa, **raise** nếu thiếu > 10 khóa (sai kiến trúc).
- **`trainer.py`:** lịch đóng băng suy ra **chỉ từ số epoch** (resume ở đâu cũng đúng); khi đóng băng, BatchNorm của backbone giữ `eval()`; nhóm tham số: `decoder` (lr) và `backbone` (lr × mult), mỗi nhóm lưu `lr_mult` để đặt lại lr đúng tỉ lệ.
- Mặc định `runs-root`: `mamba_unet/runs`.
- `activations.py`, `losses.py`: theo mục 4.1, 4.2 (bản **riêng** của `mamba_unet`, tách hoàn toàn khỏi `unet_base`). `activations` chỉ áp dụng cho **decoder**; backbone MambaVision cố định. `MambaUNetTrainer` dùng `mamba_unet.losses.get_loss`. Không import gì từ `unet_base`.
- `blocks.py`: theo mục 4.3 (bản **riêng** của `mamba_unet`, sao chép từ code cũ, không import từ `unet_base`). `mamba_unet/decoder.py` dùng khối từ `mamba_unet.blocks` với factory từ `mamba_unet.activations`.

---

## 6. THƯ MỤC MỖI LẦN TRAIN VÀ `run_config.json`

```text
<runs-root>/<run_id>/
├── run_config.json          # BẮT BUỘC — toàn bộ thông tin để tái lập & so sánh
├── history.csv              # mỗi epoch: lr, train/val loss, dice, iou, thời gian
├── best_model.pth           # checkpoint đầy đủ (có optimizer) — dùng để resume/đánh giá
├── last_model.pth
├── best_weights_fp16.pth    # CHỈ state_dict fp16 + meta — nhẹ, để chia sẻ/push GitHub
├── metrics_final.json
├── data_manifest.json       # nguồn dữ liệu + danh sách bệnh nhân của run này (sinh từ runtime_data) — để đánh giá dựng lại đúng tập
├── train.log                # tee của stdout
└── eval/                    # kết quả đánh giá 3D (csv + json ghi tham số đánh giá)
```
**`run_id`** = `{YYYYmmdd-HHMMSS}_{arch}[_{variant}][_{tag}]`, ví dụ `20261109-153012_mamba_unet_T_lr1e-4`. Trùng thì thêm `-2`, `-3`.

**Lược đồ `run_config.json`** (`schema_version: 1`):
```json
{
  "schema_version": 1,
  "run_id": "...", "arch": "mamba_unet", "status": "running|completed|interrupted|failed",
  "notes": "", "started_at": "ISO8601", "finished_at": null, "duration_sec": null,
  "resume_events": [{"at": "ISO8601", "from_epoch": 30}],
  "init_weights": null,
  "git": {"commit": "...", "branch": "...", "dirty": false, "remote": "..."},
  "environment": {"platform": "colab|kaggle|local", "python": "...", "torch": "...", "cuda": "...",
                  "gpu": "...", "num_gpus": 1, "amp": true, "mamba_backend": "cuda|torch|n/a",
                  "packages": {"timm": null, "einops": null, "mamba_ssm": null}},
  "args": {},                       // toàn bộ tham số đã giải xong (YAML + CLI)
  "model": {"class": "MambaUNet", "config": {}, "activation": "gelu", "num_params": 0, "trainable_params": 0},
  "data": {"cache_dir": "...", "slices_csv_sha256": "...", "image_size": 512,
           "source": {"platform": "colab|kaggle|local", "kind": "zip|kaggle_input|local_dir", "origin": "...",
                      "n_available": 0, "n_selected": 0, "select": "first|random|file|all", "seed": 42,
                      "patient_ids_sha256": "..."},
           "train_slices": 0, "val_slices": 0, "train_patients": 0, "val_patients": 0,
           "split_seed": 42, "augment": true, "lung_mask": true},
  "loss": {"name": "dice_bce", "params": {}},
  "optimizer": {"name": "AdamW", "weight_decay": 0.0001,
                "param_groups": [{"name": "decoder", "lr": 1e-4}, {"name": "backbone", "lr": 1e-5}]},
  "scheduler": {"name": "cosine", "params": {}},
  "training": {"epochs_target": 50, "batch_size": 8, "grad_clip": 1.0, "patience": 12,
               "freeze_backbone_epochs": 0, "seed": 42},
  "result": {"epochs_completed": 0, "best_val_dice": null, "best_epoch": null,
             "last": {"val_loss": null, "val_dice": null, "val_iou": null}}
}
```
**`scripts/compare_runs.py`**: quét một hoặc nhiều `--runs-root`, đọc `run_config.json` + `history.csv`, xuất `runs_summary.csv` và bảng Markdown (cột: run_id, arch, n_patients, backbone, features, activation, loss, lr, epochs, best_val_dice, best_epoch, thời gian, git commit, notes); `--sort-by`, `--arch`, `--since`.

**Resume:** `--resume <run_dir>` tiếp tục **trong cùng thư mục** (ghi thêm `resume_events`); `--resume auto` = run mới nhất của kiến trúc đó có `status ≠ completed` (không có thì train mới, in thông báo); `--init-weights <file.pth>` = **run mới** chỉ nạp trọng số, ghi nguồn vào `init_weights`.

---

## 7. ĐÁNH GIÁ SAU KHI TRAIN (`<arch>/evaluate.py`)
- `eval --run-dir <dir> [--checkpoint best|last|<file>] --cache-dir ... --split val|train|all --threshold 0.5 --tta --batch-size N` → ghi `<run_dir>/eval/eval_3d_<split>_<timestamp>.csv` **và** `.json` (tham số đánh giá, git commit, checkpoint dùng).
- `predict --run-dir <dir> --patient-cache <dir> --out-nrrd <file> [--ref-nrrd <file>]`.
- `selftest`: dữ liệu giả lập, không cần mạng.
- Hàm nạp: `load_unet_base_from_run(path, device, which="best")` / `load_mamba_unet_from_run(...)`. Thứ tự: **(1)** `run_config.json` → **(2)** `meta` trong checkpoint → **(3)** checkpoint cũ không có meta: dùng mặc định + bộ chuyển khóa cũ (mục 9.3). Luôn `load_state_dict(..., strict=True)`.
- Mọi mã đánh giá là `.py`; notebook chỉ gọi lệnh (hoặc import hàm) và hiển thị kết quả.

---

## 8. QUY TRÌNH LÀM VIỆC: PC → GITHUB → COLAB / KAGGLE

### 8.1 Hạ tầng chung
- `scripts/bootstrap_env.py --arch {unet_base|mamba_unet}`: nhận diện nền tảng (Colab: `google.colab` import được; Kaggle: có `/kaggle/working`), cài `requirements/<arch>.txt`, in bảng môi trường (Python/torch/CUDA/GPU/backend scan). Đặt biến môi trường `DUNEAI_RUNS_ROOT`, `DUNEAI_DATA_DIR` nếu notebook truyền vào.
- `scripts/make_notebooks.py` **sinh** mọi `.ipynb` bằng `nbformat` và `nbformat.validate` (không viết JSON notebook bằng tay).

### 8.2 Notebook (cấu trúc ô — mỗi kiến trúc có bản Colab, Kaggle, Evaluate)
1. Ô cấu hình: `REPO_URL`, `BRANCH`, tham số thí nghiệm (EPOCHS, BATCH_SIZE, LR, ACTIVATION, LOSS, ...), `RUN_TAG`. Ghi chú trong ô: tên hợp lệ lấy bằng `python -m <arch>.activations --list` và `python -m <arch>.losses --list` (danh sách của **chính kiến trúc đó**).
2. Clone/pull repo (repo riêng tư: token từ Colab `userdata` / Kaggle Secrets, **không in**), `cd` vào ROOT.
3. `python scripts/bootstrap_env.py --arch <arch>`.
4. Dữ liệu (chi tiết mục 8.6): **Colab** — ô mount Drive, rồi `python -m common.data_pipeline.runtime_data colab --zip "$DRIVE_ZIP" --dest <SSD> --n-patients "$N_PATIENTS" --select "$SELECT" --seed "$SELECT_SEED" --out-json <file>`; **Kaggle** — `python -m common.data_pipeline.runtime_data kaggle --hint "$KAGGLE_DATA_HINT" --view-dir <thư mục ghi được> ... --out-json <file>`. Ô cấu hình có thêm `N_PATIENTS` (`None` = tất cả), `SELECT` (`first`|`random`), `SELECT_SEED`, và `DRIVE_ZIP` (Colab, mặc định `"auto"`) hoặc `KAGGLE_DATA_HINT` (Kaggle, mặc định `/kaggle/input/datasets/minhtri2475/lidc-idri-1mm-cache-ct208e`). Notebook đọc `CACHE_DIR` từ file JSON do lệnh ghi ra (không tách chuỗi từ stdout), in bảng tóm tắt (nền tảng, số bệnh nhân, thư mục) và truyền `--cache-dir "$CACHE_DIR"` cho các bước sau. `slices.csv` do chương trình dựng khi thiếu hoặc lệch (8.6).
5. `python -m <arch>.train_main selftest`.
6. Huấn luyện: lệnh dựng từ biến ở ô 1, `--cache-dir "$CACHE_DIR"`, `--runs-root` = Drive (Colab) hoặc `/kaggle/working/runs` (Kaggle), `--resume auto`. Notebook `evaluate.ipynb` gọi `runtime_data auto --patients-file <run_dir>/data_manifest.json` để dựng lại **đúng tập bệnh nhân** của run cần đánh giá.
7. Vẽ đường cong từ `history.csv` (gọi hàm trong module Python, không viết lại logic vẽ trong ô).
8. Đánh giá: `python -m <arch>.evaluate eval --run-dir <run_dir> ...`.
9. `python scripts/compare_runs.py --runs-root ...`.
10. **Kaggle**: ô cuối đẩy kết quả lên GitHub (8.4). **Colab**: không cần đẩy, trọng số đã nằm trên Drive.

Notebook **Mamba** — ô cài đặt (đã kiểm chứng nguyên nhân lỗi Colab trước đó):
- **Không** `pip install mambavision` (gói này ép `mamba-ssm==2.2.4`, `timm==1.0.15`, `transformers==4.50.0`; `mamba-ssm 2.2.4` chỉ có wheel dựng sẵn tới PyTorch 2.6 nên trên Colab mới sẽ rơi vào biên dịch từ nguồn và thất bại).
- **Không** cài `causal-conv1d` (mô hình không dùng).
- **Không** dùng `--find-links` trỏ vào file `.whl` (pip bỏ qua, quay về biên dịch). Nếu muốn nhân CUDA cho nhanh: `pip install --no-deps <URL wheel>` trong khối `try/except` không làm hỏng ô; thiếu wheel thì dùng bản PyTorch. Tên wheel `mamba-ssm 2.3.1`: `mamba_ssm-2.3.1+cu{12|11|13}torch{MAJOR.MINOR}cxx11abi{TRUE|FALSE}-cp{310..313}-cp{..}-linux_x86_64.whl` (có sẵn cho PyTorch 2.6–2.10; tự suy ra các thành phần từ `torch.__version__`, `torch.version.cuda`, `torch._C._GLIBCXX_USE_CXX11_ABI`, `sys.version_info`; kiểm tra URL tồn tại trước khi cài).
- Ô cuối phần cài đặt in rõ `mamba_backend` đang dùng, và **không** in "cài xong" nếu lệnh pip thất bại (kiểm tra mã thoát).

### 8.3 Nơi lưu theo nền tảng
| Nền tảng | Dữ liệu | `--runs-root` | Trọng số |
|---|---|---|---|
| Local | `data/cache_1mm` | `<arch>/runs` | trên máy |
| Colab | Zip `cache_1mm` trên Drive → giải nén **số lượng chọn được** ra SSD cục bộ (8.6) | thư mục trên **Drive** | tự nằm trên Drive |
| Kaggle | `/kaggle/input/...` (chỉ đọc) → thư mục symlink ghi được (8.6) | `/kaggle/working/runs` | **đẩy về GitHub** (8.4) |

### 8.4 `scripts/push_run_to_github.py` (Kaggle → GitHub)
- Đầu vào: `--run-dir`, `--repo`, `--branch` (mặc định `results`, **không đẩy lên `main`**), token từ biến môi trường `GITHUB_TOKEN`.
- Đẩy qua **bản clone tạm** (`--depth 1`) của nhánh `results` để không làm bẩn cây làm việc; đích: `<arch>/runs/<run_id>/`. Không lưu token vào `.git/config` (dùng URL `https://x-access-token:$TOKEN@...` cho riêng lệnh push, che token trong mọi log).
- Mặc định đẩy: `run_config.json`, `history.csv`, `metrics_final.json`, `data_manifest.json`, `eval/*`, `best_weights_fp16.pth`. **Không** đẩy `best_model.pth`/`last_model.pth` (có optimizer, rất nặng) trừ khi `--include-resume-ckpt`.
- **Giới hạn GitHub: tối đa 100 MB/file với git thường.** Số liệu thực tế: U-Net `features=64` ≈ 31,0 triệu tham số → fp16 ≈ 62 MB (vừa); riêng encoder MambaVision-T ≈ 31,2 triệu tham số → fp16 ≈ 62 MB, **cộng decoder sẽ gần hoặc vượt ngưỡng**. Quy tắc bắt buộc: file > 95 MB → thử `git lfs track` nếu có `git-lfs`; không có thì **bỏ qua file đó, cảnh báo rõ** và in hướng dẫn thay thế (Kaggle Dataset / Hugging Face Hub). Không bao giờ để lệnh push thất bại im lặng.
- Hạn mức LFS có giới hạn dung lượng/băng thông (xem trang giá GitHub hiện hành) → khuyến nghị chỉ đẩy `best_weights_fp16.pth`.

### 8.5 `.gitignore` / `.gitattributes`
- Bỏ qua: `data/`, `**/runs/**/last_model.pth`, `**/runs/**/best_model.pth`, `**/*.tmp`, `__pycache__/`, `.ipynb_checkpoints/`, `*.nrrd`, `desktop.ini`.
- Giữ lại (không ignore): `run_config.json`, `history.csv`, `metrics_final.json`, `data_manifest.json`, `eval/`, `best_weights_fp16.pth`.
- `.gitattributes`: `*.pth filter=lfs diff=lfs merge=lfs -text` chỉ bật khi chủ dự án quyết định dùng LFS (ghi chú sẵn, mặc định tắt).

### 8.6 Dữ liệu theo nền tảng (`common/data_pipeline/runtime_data.py`)

**Mục tiêu:** một chỗ duy nhất biết "dữ liệu nằm ở đâu trên từng nền tảng", biến nó thành một **thư mục cache đã chuẩn bị** (ghi được, chứa các thư mục `LIDC-IDRI-XXXX/`) rồi đưa cho `--cache-dir`. Việc này thuộc nhóm "xử lý trước train" nên đặt ở `common/`; notebook **chỉ gọi CLI** của nó, không tự viết logic dò đường dẫn hay giải nén.

**Quy ước dữ liệu gốc:** mỗi bệnh nhân là một thư mục `LIDC-IDRI-XXXX/` chứa `image_1mm.npy`, `lung_1mm.npy`, `mask_1mm.npy`, `meta.json` (Agent xác nhận danh sách tệp thật từ code cũ ở Phase 0). **Bố cục đã xác nhận của `cache_1mm.zip` (Colab):** một thư mục gốc `cache_1mm/` chứa trực tiếp các `LIDC-IDRI-XXXX/` (tức `cache_1mm/LIDC-IDRI-0001/image_1mm.npy`, ...). **Bố cục trên Kaggle:** `.../cache_1mm/cache_1mm/LIDC-IDRI-XXXX/` (lồng hai tầng cùng tên). Dù đã biết bố cục, mã vẫn dò theo regex `LIDC-IDRI-\d{4}` và **không hard-code** tên thư mục gốc hay độ sâu, để chịu được thay đổi nhỏ khi bạn đóng gói lại dữ liệu.

```text
python -m common.data_pipeline.runtime_data colab  --zip auto|<file.zip> [--drive-root <dir>] --dest <dir>
                       [--n-patients N] [--select first|random] [--seed 42] [--patients-file <json>] [--copy-zip-first] --out-json <file>
python -m common.data_pipeline.runtime_data kaggle --hint <dir> --view-dir <dir>
                       [--n-patients N] [--select first|random] [--seed 42] [--patients-file <json>] --out-json <file>
python -m common.data_pipeline.runtime_data local  --cache-dir <dir> [--out-json <file>]     # chỉ xác minh + ghi manifest, không chọn tập con
python -m common.data_pipeline.runtime_data auto   ...      # nhận diện nền tảng rồi gọi nhánh tương ứng (dùng cho evaluate.ipynb)
python -m common.data_pipeline.runtime_data selftest
```
`--out-json` ghi `data_manifest.json`: `platform, kind, origin, cache_dir` (thư mục đã chuẩn bị), `n_available, n_selected, selection{n_requested, mode, seed}, patient_ids, patient_ids_sha256`, và `zip{path, size_bytes}` nếu có. Dòng cuối của stdout là `CACHE_DIR=<đường dẫn>` cho người đọc; notebook lấy giá trị từ file JSON.

**Colab — zip trên Drive, chọn số lượng để giải nén**
1. Ô mount Drive nằm trong notebook (đặc thù nền tảng, không đặt trong `.py`).
2. `--zip auto`: tìm `cache_1mm.zip` (khớp mẫu `cache_1mm*.zip`) dưới `--drive-root` (độ sâu tối đa 4, ưu tiên đường dẫn người dùng đặt ở ô cấu hình `DRIVE_ZIP`). Nhiều ứng viên → dừng, liệt kê, yêu cầu đặt `DRIVE_ZIP`. Không thấy → lỗi tiếng Việt nêu rõ đã tìm ở đâu.
3. Đọc danh mục zip bằng `zipfile` (không giải nén), nhận ra thư mục bệnh nhân bằng regex (bố cục thật: `cache_1mm/LIDC-IDRI-XXXX/...`) → `n_available`. Chọn theo `--n-patients/--select/--seed` (`N` rỗng hoặc `0` = tất cả; `N > n_available` → dùng tất cả và cảnh báo).
4. Kiểm tra dung lượng trống của `--dest` ≥ tổng kích thước giải nén của phần được chọn + 5%; thiếu → lỗi rõ ràng, không giải nén dở.
5. Giải nén **chỉ** các bệnh nhân đã chọn vào `--dest/LIDC-IDRI-XXXX/` (bỏ tiền tố `cache_1mm/` của zip; `--dest` chính là thư mục cache trả về). Mỗi bệnh nhân giải nén vào thư mục tạm cùng cấp rồi `os.replace` → thư mục dở dang (ngắt giữa chừng) không bao giờ trông như hoàn chỉnh. Bệnh nhân đã đủ tệp và đúng kích thước thì bỏ qua → chạy lại ô an toàn (SSD Colab bị xóa mỗi phiên, nên đây là cách khôi phục). Có tiến trình `tqdm`.
6. Mặc định đọc trực tiếp từ zip trên Drive; `--copy-zip-first` chép zip về SSD trước (nhanh hơn khi giải nén gần hết, phí thời gian khi chỉ lấy ít). `--dest` phải là SSD cục bộ, không phải Drive.

**Kaggle — dữ liệu có sẵn ở `/kaggle/input` (chỉ đọc)**
1. Không chép, không giải nén. Nút "copy path" của Kaggle trả về đường dẫn **một bệnh nhân**, ví dụ `/kaggle/input/datasets/minhtri2475/lidc-idri-1mm-cache-ct208e/cache_1mm/cache_1mm/LIDC-IDRI-0001` ⇒ cache gốc là thư mục **cha**: `.../cache_1mm/cache_1mm`.
2. `--hint` nhận bất kỳ dạng nào: thư mục gốc dataset (`/kaggle/input/datasets/minhtri2475/lidc-idri-1mm-cache-ct208e`), thư mục cache, hoặc đường dẫn một bệnh nhân (tự lấy cha). Mã dò xuống tối đa 4 tầng, lấy thư mục đầu tiên có thư mục con khớp `LIDC-IDRI-\d{4}` chứa tệp `.npy`. Hint rỗng hoặc không tồn tại → dò dưới `/kaggle/input` (cả bố cục mới `/kaggle/input/datasets/<owner>/<slug>/` lẫn cũ `/kaggle/input/<slug>/`); nhiều ứng viên → dừng và liệt kê; không thấy → lỗi tiếng Việt hướng dẫn thêm dataset vào notebook (Add Input).
3. Vì nguồn chỉ đọc, dựng `--view-dir` (ghi được, mặc định nằm trong `/kaggle/working`) chứa **symlink** tới từng bệnh nhân được chọn, rồi trả view-dir làm cache dir. Lợi ích: (a) chọn tập con N bệnh nhân mà không phải sửa code dataset; (b) `slices.csv` và mọi tệp phụ ghi được vào view-dir. Không tạo được symlink → **dừng và báo**, không âm thầm chép dữ liệu (`/kaggle/working` rất hạn chế dung lượng).

**Quy tắc chung**
- Chọn bệnh nhân luôn **xác định** (cùng đầu vào → cùng tập): `first` = N đầu theo tên đã sắp xếp (mặc định); `random` = `random.Random(seed).sample` rồi sắp xếp lại; `--patients-file` = danh sách tường minh (mảng ID JSON hoặc một `data_manifest.json` cũ) — dùng để **dựng lại đúng tập bệnh nhân của một run** khi đánh giá.
- Train/val chia theo bệnh nhân với `split_seed`; đổi tập bệnh nhân thì tập val đổi, nên **các run khác `n_selected` không so sánh trực tiếp được**. Vì vậy `patient_ids_sha256` và `n_selected` được ghi vào `run_config.json` (`data.source`) và `compare_runs.py` hiện cột `n_patients`.
- **`slices.csv`:** không tin tệp có sẵn trong dataset gốc (có thể chứa đường dẫn tuyệt đối của máy khác). Cạnh `slices.csv` luôn ghi `slices.meta.json` (`patient_ids_sha256`); nếu `slices.csv` thiếu hoặc sidecar khác tập bệnh nhân hiện tại → **xóa `slices.csv` cũ** để chương trình dựng lại (`--rebuild-index` vẫn dùng như cũ). Agent đọc code cũ để xác nhận chỗ dataset đọc/ghi `slices.csv`; nếu code cũ ghi vào `cache_dir` thì không phải sửa gì (`--dest`/view-dir đã ghi được); nếu code cũ ghi chỗ khác → ghi `DEVIATIONS.md` và hỏi.
- Phần tìm/giải nén/symlink là **mã mới** (code cũ chưa có), nên không vi phạm R7; nhưng **không** được sửa logic trong `dataset.py`/`offline_builder.py` để phục vụ nó.
- `train_main` đọc `<cache-dir>/data_manifest.json` (hoặc `--data-manifest`), điền `run_config.data.source` và chép manifest vào thư mục run.
- **Selftest (không mạng):** (1) zip giả đúng **bố cục thật** `cache_1mm/LIDC-IDRI-000X/<tệp>` nhận diện đúng và giải nén ra `--dest/LIDC-IDRI-000X/`; thêm một ca phụ với zip phẳng (không có thư mục gốc) vẫn đúng; (2) `first` N=3 ra đúng 3 bệnh nhân đầu, `random` cùng seed hai lần ra cùng tập, khác seed ra tập khác; (3) chạy lại không giải nén lại, thư mục tạm dở dang được làm lại; (4) `N > n_available` → dùng tất cả + cảnh báo; (5) thiếu dung lượng (mô phỏng) → lỗi; (6) cây Kaggle giả lồng `cache_1mm/cache_1mm`, hint là thư mục gốc / thư mục cache / một bệnh nhân đều ra cùng cache dir, nguồn đặt chỉ-đọc vẫn dựng được view-dir symlink; (7) đổi tập bệnh nhân → `slices.csv` cũ bị xóa; (8) `--patients-file` từ manifest cũ dựng lại đúng tập; (9) không thấy zip/dataset → thông báo tiếng Việt và exit code ≠ 0. Phải **fail khi cố tình phá**: (a) bỏ bước thư mục tạm + `os.replace`; (b) bỏ sắp xếp trước khi chọn.

---

## 9. GIAO THỨC BẢO TOÀN `state_dict` (điều kiện số 1)

### 9.1 Phase 0 — sinh golden TỪ CODE CŨ (trước khi di chuyển bất cứ thứ gì)
`scripts/export_golden.py` chạy trên **repo ở trạng thái hiện tại** và ghi vào `tests/golden/`:
- `unet_base_keys.json`, `mamba_unet_keys.json`: `{khóa: shape}` của mô hình **kích thước thật** (`features=64`; `mamba_vision_T`, `pretrained=False`);
- `unet_base_tiny_state.pth`, `unet_base_tiny_out.pt`: mô hình nhỏ (`features=8`), seed cố định, `eval()`, đầu vào `randn(2,3,128,128)` với seed cố định → `state_dict` + đầu ra;
- `mamba_unet_tiny_*`: tương tự với cấu hình MambaVision **thu nhỏ** (vd. `dim=16, in_dim=8, depths=[1,1,2,1]`) để file nhỏ (state_dict của MambaVision-T đầy đủ ≈ 125 MB, không đưa vào git);
- `legacy_prefixless_unet_keys.json` nếu code cũ còn lớp U-Net **không tiền tố** (xem 9.3).
Tên khóa trong plan cũ chỉ là ví dụ minh họa — **nguồn sự thật là các file golden**.

### 9.2 Kiểm thử sau refactor (`tests/test_state_dict_invariance.py`)
1. Tập khóa và shape của mô hình mới == golden (cả hai kiến trúc, kích thước thật).
2. Nạp `*_tiny_state.pth` vào mô hình mới `strict=True`, đầu ra `allclose(atol=1e-6)` với `*_tiny_out.pt`.
3. `save → load` vòng tròn qua bộ nhớ tạm, `strict=True`, đầu ra khớp.
4. Nếu đặt biến môi trường `LEGACY_CKPT_UNET` / `LEGACY_CKPT_MAMBA` trỏ tới checkpoint thật đã train trên Kaggle: nạp bằng `load_*_from_run` và chạy một lượt forward; test **bỏ qua có thông báo** khi biến không đặt.
5. Đổi activation sang hàm không tham số vẫn cùng tập khóa; sang `star_relu` thì **phải** khác (kiểm tra `has_params` đúng) — chạy **riêng cho từng kiến trúc** bằng registry của nó.

### 9.3 Checkpoint cũ không tiền tố
Bản U-Net đời đầu đặt tên không có `encoder.`/`decoder.` (ví dụ `inc.double_conv.0.weight`, `down1.maxpool_conv.1.double_conv.0.weight`, `up1.up.weight`, `up1.conv.double_conv.0.weight`, `outc.conv.weight`). Giả thuyết ánh xạ: `inc.*`, `down{n}.*` → `encoder.…`; `up{n}.*`, `outc.*` → `decoder.…`. Agent **không được tin giả thuyết**: phải kiểm chứng bằng `legacy_prefixless_unet_keys.json` (từ code cũ) so với golden mới, viết `legacy_key_map` trong `unet_base/evaluate.py`, và có test nạp `strict=True`. Không tìm được ánh xạ khớp hoàn toàn → **dừng và báo cáo**.

---

## 10. KIỂM THỬ NGHIỆM THU
Tất cả chạy từ ROOT, mỗi lệnh phải exit code 0; `python tests/run_all.py` chạy toàn bộ và in bảng:

```bash
python -m common.metrics                    # selftest metric dùng chung
python -m unet_base.blocks                  # selftest khối CNN của unet_base
python -m mamba_unet.blocks                 # selftest khối CNN của mamba_unet
python -m unet_base.activations             # selftest registry kích hoạt của unet_base
python -m unet_base.losses                  # selftest registry loss của unet_base
python -m mamba_unet.activations            # selftest registry kích hoạt của mamba_unet
python -m mamba_unet.losses                 # selftest registry loss của mamba_unet
python -m common.data_pipeline.medical_prep selftest
python -m common.data_pipeline.offline_builder selftest
python -m common.data_pipeline.dataset selftest
python -m common.data_pipeline.runtime_data selftest
python -m unet_base.train_main selftest
python -m unet_base.evaluate selftest
python -m mamba_unet.train_main selftest
python -m mamba_unet.evaluate selftest
python tests/test_independence.py           # AST + chặn import (3.3)
python tests/test_state_dict_invariance.py
python tests/test_scan_fallback.py          # selective_scan_torch vs công thức đóng, backward hữu hạn
python tests/test_blocks.py                 # với MỖI kiến trúc: DoubleConv đổi activation không tham số giữ nguyên khóa, có tham số thì thêm khóa; chứng minh sửa blocks.py bên này không đổi bên kia
python tests/test_registries.py             # với MỖI kiến trúc: thêm activation/loss giả vào registry của nó → chọn theo tên → ghi vào run_config; và chứng minh đăng ký ở bên này KHÔNG xuất hiện ở bên kia
python tests/test_run_folder.py             # cấu trúc thư mục run + schema run_config.json
python tests/test_resume.py                 # ngắt giữa chừng → resume auto → liền mạch; đổi arch/features → ValueError rõ
python scripts/make_notebooks.py --check    # nbformat.validate mọi notebook
```
`selftest` của mỗi `train_main` phải gồm: train ngắn trên dữ liệu giả lập; ngắt bằng ngoại lệ rồi `--resume auto`; kiểm tra `run_config.json` đủ khóa; kiểm tra `best_weights_fp16.pth` nạp lại được; (mamba) lịch đóng băng đúng từng epoch và backbone không đổi giá trị khi đóng băng. Test mới phải **fail khi cố tình phá** (Agent tự thử ít nhất: bỏ lịch đóng băng, bỏ nạp optimizer khi resume, đổi tên một khóa) — ghi kết quả vào báo cáo.

---

## 11. CÁC PHASE THỰC THI (mỗi phase: làm → test → commit → báo cáo → DỪNG)

| Phase | Nội dung | Tiêu chí xong |
|---|---|---|
| **P0** | Nhánh `refactor/split-architectures`. Kiểm kê toàn bộ code hiện có → `docs/INVENTORY.md` (bảng *file cũ → file mới*, danh sách hàm/lớp công khai, alias, CLI cũ, nơi gọi từng alias). Viết & chạy `export_golden.py`. **Chưa di chuyển file nào** | Golden có đủ; INVENTORY liệt kê mọi file; không file code nào bị sửa |
| **P1** | Khung thư mục, `configs/`, `common/paths.py`, `common/metrics.py` (**không** tạo `activations/`, `losses/` ở ROOT, **không** tạo `common/nn_blocks.py`) | `python -m common.metrics` pass; `common/` không có lớp `nn.Module` nào |
| **P2** | `common/data_pipeline/` (kèm `runtime_data.py`, mục 8.6), `common/engine/` (BaseTrainer nhận `criterion` từ ngoài, RunManager, run_config, checkpoint_io, cli_args nhận choices từ ngoài, utils) | 4 selftest data pass (kể cả `runtime_data`); selftest `BaseTrainer` (resume/ngắt, dùng loss giả) và `test_run_folder` pass |
| **P3** | `unet_base/` đầy đủ (encoder, decoder, model, **blocks.py, activations.py, losses.py**, trainer, train_main) | `python -m unet_base.blocks`, `python -m unet_base.activations`, `python -m unet_base.losses`, `unet_base.train_main selftest` pass; `test_registries` phần unet_base; invariance của unet_base pass; độc lập pass |
| **P4** | `mamba_unet/` đầy đủ (fallbacks, backbone/, decoder, model, **blocks.py, activations.py, losses.py**, trainer, train_main) | `python -m mamba_unet.blocks`, `python -m mamba_unet.activations`, `python -m mamba_unet.losses`, `mamba_unet.train_main selftest` pass; `test_registries` đủ hai bên (kể cả kiểm tra cách ly registry); `test_scan_fallback` pass; invariance của mamba pass; độc lập pass |
| **P5** | `common/eval3d/` + `evaluate.py` hai kiến trúc | hai `evaluate selftest` pass; nạp checkpoint tiny qua `load_*_from_run` strict |
| **P6** | `scripts/`, `requirements/`, `.gitignore`, `.gitattributes`, `README.md`, sinh notebook | `make_notebooks.py --check` pass; `push_run_to_github.py --dry-run` pass với repo cục bộ giả |
| **P7** | `tests/run_all.py` toàn bộ; sửa mọi tham chiếu cũ; xóa file cũ; kiểm tra không còn chuỗi cấm (R6) và không có `ROOT/activations`, `ROOT/losses` (R11) | `run_all.py` toàn PASS; `grep` không còn `DuneAI_Remake`, `from src.`, `src/`, `from activations`, `from losses` |

Báo cáo cuối mỗi phase (bắt buộc): danh sách file tạo/sửa/xóa, lệnh test đã chạy kèm kết quả thực, mọi sai lệch so với spec (`DEVIATIONS.md`), câu hỏi còn treo.

---

## 12. RỦI RO & QUYẾT ĐỊNH CÒN MỞ (Agent KHÔNG tự quyết — hỏi chủ dự án)
1. **Gốc git:** nếu repo hiện tại có gốc ở `SOURCES/DuneAI_Remake/`, chủ dự án phải tự chuyển `.git` lên `SOURCES/` (hoặc khởi tạo repo mới) trước P7. Agent chỉ viết hướng dẫn, không tự di chuyển `.git`.
2. **Trọng số trên GitHub:** dùng Git LFS hay chỉ đẩy `best_weights_fp16.pth`? Muốn **resume giữa các phiên Kaggle** thì cần checkpoint đầy đủ (rất nặng) → LFS, Kaggle Dataset hoặc Hugging Face Hub.
3. **Hai `trainer.py` riêng:** theo yêu cầu mỗi kiến trúc có `trainer.py`, nhưng vòng lặp lõi dùng chung ở `common/engine/base_trainer.py` để không sao chép ~600 dòng và để hai bản không lệch nhau theo thời gian. Muốn sao chép nguyên vòng lặp vào từng kiến trúc thì phải nói rõ.
4. **Xóa `use_stride4_stem` và alias** chỉ khi chứng minh được không còn nơi dùng (mục 5.2).
5. **Mã trùng giữa hai `activations.py` / hai `losses.py`:** là chủ đích (R11). Muốn có một bản gốc để cả hai kế thừa thì phải nói rõ — Agent không tự gom.
6. **Khối CNN (`DoubleConv`/`OutConv`):** **đã chốt** là mỗi kiến trúc tự sở hữu (`<arch>/blocks.py`, mục 4.3), không dùng chung. Hệ quả chấp nhận: mã trùng giữa hai `blocks.py`; sửa lỗi ở khối phải sửa hai nơi nếu cả hai cùng cần.
7. **`BaseTrainer` ở `common/engine/`:** vòng lặp train (AMP, resume, early stopping, ...) không thuộc cấu trúc mạng nên vẫn dùng chung (xem mục 12.3). Nếu muốn mỗi kiến trúc tự có vòng lặp train riêng thì phải nói rõ.
8. **Dữ liệu (xác nhận khi chạy thật, Agent không đoán):** nơi code cũ ghi `slices.csv`; dataset Kaggle đổi slug khi tải lại (dò tự động đã xử lý, nhưng đổi chủ sở hữu thì sửa `KAGGLE_DATA_HINT`); chọn bệnh nhân mặc định là `first` hay `random` (hiện mặc định `first`).

## 13. NGOÀI PHẠM VI
Không đổi thuật toán/siêu tham số mặc định, không tối ưu hiệu năng, không thêm tính năng mới ngoài spec, không huấn luyện thật, không viết trực quan hóa/overlay (có thể làm sau).
