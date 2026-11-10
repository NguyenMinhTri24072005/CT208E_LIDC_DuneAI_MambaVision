# -*- coding: utf-8 -*-
"""
scripts/make_notebooks.py - Trình sinh tự động 6 Jupyter Notebooks cho 2 kiến trúc trên 3 nền tảng.

Các notebook được sinh ra:
  1. unet_base/notebooks/colab_train.ipynb
  2. unet_base/notebooks/kaggle_train.ipynb
  3. unet_base/notebooks/local_train.ipynb
  4. mamba_unet/notebooks/colab_train.ipynb
  5. mamba_unet/notebooks/kaggle_train.ipynb
  6. mamba_unet/notebooks/local_train.ipynb

Đặc tả chuẩn mực:
  - Colab : Mount Google Drive, nhận diện file zip cache_1mm, giải nén phẳng vào SSD /content
            với khả năng chọn số lượng bệnh nhân tùy ý, huấn luyện, đánh giá 3D, backup về Drive.
  - Kaggle: Tự động dò tìm dataset /kaggle/input (kể cả nested path dạng cache_1mm/cache_1mm),
            dựng symlink view-dir ghi được trong /kaggle/working, huấn luyện và đánh giá 3D.
  - Local : Chạy trực tiếp trên máy trạm với dữ liệu data/cache_1mm.
  - Chuẩn định dạng: nbformat 4.5 hợp lệ 100%.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _make_cell(cell_type: str, source: str) -> Dict[str, Any]:
    lines = [f"{line}\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    cell: Dict[str, Any] = {
        "id": uuid.uuid4().hex[:8],
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _build_notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# =============================================================================
# NỘI DUNG NOTEBOOK COLAB CHO MAMBA-UNET
# =============================================================================
def _make_colab_mamba_notebook() -> Dict[str, Any]:
    cells = [
        _make_cell("markdown", (
            "# 🩺 Huấn luyện Mamba-UNet (MambaVision Backbone) trên Google Colab\n\n"
            "Quy trình chuẩn hóa toàn diện:\n"
            "1. **Kiểm tra phần cứng GPU** (`!nvidia-smi`)\n"
            "2. **Cài đặt môi trường PyTorch 2.6.0 (cu126) & Triton 3.3.1** (tương thích chính xác với C++ CUDA ABI=True)\n"
            "3. **Mount Google Drive & Cài đặt 2 Wheels CUDA (`causal_conv1d` + `mamba_ssm`)** từ Drive\n"
            "4. **Khởi động lại Runtime** để nạp kernel C++/CUDA vào hệ thống\n"
            "5. **Kiểm tra xác minh CUDA Selective Scan** (fp32 + fp16, forward + backward)\n"
            "6. **Đồng bộ mã nguồn từ GitHub**\n"
            "7. **Giải nén dữ liệu từ Google Drive vào SSD cục bộ** (có tùy chọn số lượng ca bệnh nhân `NUM_PATIENTS`)\n"
            "8. **Huấn luyện mô hình với đầy đủ tham số tùy chỉnh** (Backbone, Pretrained, Activation, Loss, LR, Workers, Resume...)\n"
            "9. **Đánh giá 3D cấp độ bệnh nhân** trên tập Validation\n"
            "10. **Lưu kết quả & Checkpoints vào Google Drive theo từng thư mục phiên chạy kèm bản báo cáo Markdown/JSON** (tự động append nếu Resume)"
        )),
        _make_cell("code", "# 1. Kiểm tra phần cứng và GPU khả dụng\n!nvidia-smi"),
        _make_cell("code", (
            "# 2. Thiết lập môi trường PyTorch 2.6.0 (cu126) + Triton 3.3.1\n"
            "# (Chạy ở tiến trình riêng, không import torch trực tiếp trong kernel này)\n"
            "import subprocess, sys\n\n"
            "def probe(code):\n"
            "    r = subprocess.run([sys.executable, \"-c\", code], capture_output=True, text=True)\n"
            "    return r.stdout.strip() if r.returncode == 0 else None\n\n"
            "TORCH_PROBE = \"import torch;print(torch.__version__, int(torch._C._GLIBCXX_USE_CXX11_ABI))\"\n"
            "cur = probe(TORCH_PROBE)\n"
            "print(\"Torch hiện tại:\", cur)\n\n"
            "if cur and cur.startswith(\"2.6.0+cu126\") and cur.endswith(\" 1\"):\n"
            "    print(\"⏭️ Đã có torch 2.6.0+cu126 (ABI=True), bỏ qua bước cài torch.\")\n"
            "else:\n"
            "    print(\"🧹 Dọn dẹp và cài đặt PyTorch 2.6.0+cu126...\")\n"
            "    !rm -rf /usr/local/lib/python3.13/dist-packages/~* 2>/dev/null || true\n"
            "    !pip uninstall torch torchvision torchaudio causal-conv1d mamba-ssm -y -q 2>/dev/null || true\n"
            "    !pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 -q\n"
            "    cur = probe(TORCH_PROBE)\n"
            "    print(\"Torch sau khi cài:\", cur)\n\n"
            "assert cur and cur.startswith(\"2.6.0+cu126\") and cur.endswith(\" 1\"), \\\n"
            "    f\"❌ Torch chưa đúng (cần 2.6.0+cu126 và ABI=1, đang là: {cur}). Hãy thử chạy lại cell này.\"\n\n"
            "print(\"\\n⚙️ Cài đặt Triton 3.3.1 và các thư viện cần thiết (SimpleITK, etc.)...\")\n"
            "!python -m pip install triton==3.3.1 SimpleITK pandas scipy opencv-python-headless pyyaml matplotlib -q\n"
            "trt = probe(\"import triton;print(triton.__version__, hasattr(triton,'set_allocator'))\")\n"
            "print(\"Triton:\", trt)\n"
            "assert trt and trt.endswith(\"True\"), f\"❌ Triton chưa có set_allocator (đang là: {trt})\"\n\n"
            "print(\"\\n✅ CELL 2 XONG. Chạy tiếp cell 3 để mount Drive và cài 2 wheels.\")"
        )),
        _make_cell("code", (
            "# 3. Mount Google Drive & Cài đặt 2 Wheel CUDA (chọn theo ABI thật)\n"
            "import os, subprocess, sys\n"
            "from google.colab import drive\n\n"
            "print(\"🔗 Đang kết nối Google Drive...\")\n"
            "drive.mount('/content/drive')\n\n"
            "# Dò tìm thư mục wheels trên Google Drive (hỗ trợ cả cấu trúc mirror GG_1):\n"
            "WHEEL_CANDIDATES = [\n"
            "    \"/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/wheels\",\n"
            "    \"/content/drive/MyDrive/GG_1/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/wheels\",\n"
            "    \"/content/drive/MyDrive/wheels\",\n"
            "]\n"
            "WHEEL_DIR = None\n"
            "for d in WHEEL_CANDIDATES:\n"
            "    if os.path.isdir(d):\n"
            "        WHEEL_DIR = d\n"
            "        print(f\"🎯 Đã tìm thấy thư mục wheels tại: {WHEEL_DIR}\")\n"
            "        break\n\n"
            "assert WHEEL_DIR is not None, \"❌ Không tìm thấy thư mục wheels trên Google Drive! Hãy đảm bảo thư mục wheels đã được tải lên Drive.\"\n\n"
            "r = subprocess.run([sys.executable, \"-c\",\n"
            "                    \"import torch;print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')\"],\n"
            "                   capture_output=True, text=True)\n"
            "abi = r.stdout.strip()\n"
            "assert abi in (\"TRUE\", \"FALSE\"), f\"❌ Không đọc được ABI: {r.stderr}\"\n"
            "print(f\"🔍 ABI của torch đã cài: cxx11abi{abi}\")\n\n"
            "CAUSAL_WHL = f\"{WHEEL_DIR}/causal_conv1d-1.7.0+cu12torch2.6cxx11abi{abi}-cp313-cp313-linux_x86_64.whl\"\n"
            "MAMBA_WHL  = f\"{WHEEL_DIR}/mamba_ssm-2.3.2.post1+cu12torch2.6cxx11abi{abi}-cp313-cp313-linux_x86_64.whl\"\n\n"
            "for p in (CAUSAL_WHL, MAMBA_WHL):\n"
            "    assert os.path.exists(p), f\"❌ Không thấy wheel: {p}\"\n\n"
            "print(\"\\n🚀 Đang cài 2 wheel từ Drive...\")\n"
            "!python -m pip install \"{CAUSAL_WHL}\" --force-reinstall --no-deps -q\n"
            "!python -m pip install \"{MAMBA_WHL}\" --force-reinstall --no-deps -q\n\n"
            "print(\"\\n✅ CELL 3 XONG. Chạy tiếp cell 4 (khởi động lại runtime).\")"
        )),
        _make_cell("code", (
            "# 4. Khởi động lại Runtime để nạp thư viện C++/CUDA mới vào Python kernel\n"
            "# ⚠️ LƯU Ý QUAN TRỌNG:\n"
            "# Khi chạy ô này, Google Colab sẽ hiện thông báo đỏ: 'Your session crashed and has recovered'.\n"
            "# Đây là điều HOÀN TOÀN BÌNH THƯỜNG và cần thiết để khởi động lại kernel nạp C++ CUDA extensions.\n"
            "# Bạn chỉ cần ấn chạy tiếp Cell 5 ngay sau đó!\n"
            "import os\n"
            "os.kill(os.getpid(), 9)"
        )),
        _make_cell("code", (
            "# 5. Kiểm tra causal_conv1d + mamba_ssm (xuôi + ngược, fp32 + fp16)\n"
            "import sys, traceback, torch\n\n"
            "print(\"python :\", f\"cp{sys.version_info.major}{sys.version_info.minor}\")\n"
            "print(\"torch  :\", torch.__version__, \"| ABI =\", torch._C._GLIBCXX_USE_CXX11_ABI)\n"
            "print(\"GPU    :\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"KHÔNG CÓ\")\n"
            "import triton\n"
            "print(\"triton :\", triton.__version__, \"| set_allocator:\", hasattr(triton, \"set_allocator\"))\n\n"
            "ok = {}\n\n"
            "# --- Causal-Conv1d ---\n"
            "try:\n"
            "    import causal_conv1d, causal_conv1d_cuda\n"
            "    from causal_conv1d import causal_conv1d_fn\n"
            "    for dt in (torch.float32, torch.float16):\n"
            "        x = torch.randn(2, 16, 64, device=\"cuda\", dtype=dt, requires_grad=True)\n"
            "        w = torch.randn(16, 4, device=\"cuda\", dtype=dt, requires_grad=True)\n"
            "        y = causal_conv1d_fn(x, w, activation=\"silu\")\n"
            "        y.float().sum().backward()\n"
            "        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()\n"
            "    ok[\"causal_conv1d\"] = True\n"
            "    print(\"causal_conv1d : ✅ xuôi + ngược, fp32 + fp16\")\n"
            "except Exception:\n"
            "    ok[\"causal_conv1d\"] = False\n"
            "    print(\"causal_conv1d : ❌\"); traceback.print_exc()\n\n"
            "# --- Mamba-SSM ---\n"
            "try:\n"
            "    import mamba_ssm, selective_scan_cuda\n"
            "    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn\n"
            "    B, D, L, N = 2, 16, 64, 8\n"
            "    for dt in (torch.float32, torch.float16):\n"
            "        u     = torch.randn(B, D, L, device=\"cuda\", dtype=dt, requires_grad=True)\n"
            "        delta = torch.rand(B, D, L, device=\"cuda\", dtype=dt, requires_grad=True)\n"
            "        A     = (-torch.rand(D, N, device=\"cuda\")).requires_grad_()\n"
            "        Bm    = torch.randn(B, N, L, device=\"cuda\", dtype=dt, requires_grad=True)\n"
            "        Cm    = torch.randn(B, N, L, device=\"cuda\", dtype=dt, requires_grad=True)\n"
            "        Dv    = torch.randn(D, device=\"cuda\")\n"
            "        y = selective_scan_fn(u, delta, A, Bm, Cm, D=Dv, delta_softplus=True)\n"
            "        y.float().sum().backward()\n"
            "        assert torch.isfinite(y).all() and torch.isfinite(u.grad).all()\n"
            "    ok[\"mamba_ssm\"] = True\n"
            "    print(\"mamba_ssm     : ✅ selective_scan xuôi + ngược, fp32 + fp16\")\n"
            "except Exception:\n"
            "    ok[\"mamba_ssm\"] = False\n"
            "    print(\"mamba_ssm     : ❌\"); traceback.print_exc()\n\n"
            "assert all(ok.values()), \"❌ Kiểm tra mamba_ssm thất bại! Vui lòng kiểm tra lại log.\"\n"
            "print(\"\\n==> 🎉 MÔI TRƯỜNG CUDA MAMBA-SSM ĐÃ HOÀN TOÀN SẴN SÀNG!\")"
        )),
        _make_cell("code", (
            "# 6. Tải và đồng bộ mã nguồn từ GitHub\n"
            "import os, sys\n\n"
            "GITHUB_REPO = 'https://github.com/NguyenMinhTri24072005/CT208E_LIDC_DuneAI_MambaVision.git'\n"
            "BRANCH = 'main'  # Nhánh mặc định trên GitHub (đổi nếu bạn dùng nhánh khác)\n"
            "REPO_DIR = '/content/CT208E_LIDC_DuneAI_MambaVision'\n\n"
            "if os.path.exists('mamba_unet') and os.path.exists('common'):\n"
            "    print(f'✅ Đang chạy trực tiếp trong thư mục dự án: {os.getcwd()}')\n"
            "else:\n"
            "    if not os.path.exists(REPO_DIR):\n"
            "        print(f'📥 Đang clone repository từ GitHub (nhánh {BRANCH})...')\n"
            "        !git clone -b {BRANCH} {GITHUB_REPO} {REPO_DIR}\n"
            "    else:\n"
            "        print(f'🔄 Đang đồng bộ mã nguồn mới nhất từ GitHub...')\n"
            "        %cd {REPO_DIR}\n"
            "        !git fetch origin\n"
            "        !git checkout {BRANCH}\n"
            "        !git pull origin {BRANCH}\n"
            "    %cd {REPO_DIR}\n\n"
            "if os.getcwd() not in sys.path:\n"
            "    sys.path.insert(0, os.getcwd())\n\n"
            "print(f'🚀 Thư mục làm việc hiện tại: {os.getcwd()}')"
        )),
        _make_cell("code", (
            "# 7. Chuẩn bị dữ liệu từ Google Drive vào SSD cục bộ (/content/data/cache_1mm)\n"
            "# Thư mục GG_1 trên máy tính được mirror trực tiếp vào Google Drive.\n"
            "# Mã sẽ tự động tìm kiếm file zip theo đúng cấu trúc mirror của bạn:\n"
            "import os\n"
            "from common.data_pipeline.runtime_data import run_colab\n\n"
            "DRIVE_ZIP_CANDIDATES = [\n"
            "    # 1. Mirror trực tiếp thư mục GG_1 vào MyDrive:\n"
            "    '/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/data/cache_1mm.zip',\n"
            "    # 2. Mirror có thư mục cha GG_1 trong MyDrive:\n"
            "    '/content/drive/MyDrive/GG_1/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/data/cache_1mm.zip',\n"
            "    # 3. File zip tại thư mục KAGGLE_UPLOAD:\n"
            "    '/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/data/KAGGLE_UPLOAD/cache_1mm.zip',\n"
            "    # 4. File zip đặt ngay tại gốc MyDrive:\n"
            "    '/content/drive/MyDrive/cache_1mm.zip',\n"
            "]\n\n"
            "DRIVE_ZIP_PATH = None\n"
            "for cand in DRIVE_ZIP_CANDIDATES:\n"
            "    if os.path.exists(cand):\n"
            "        DRIVE_ZIP_PATH = cand\n"
            "        print(f'🎯 Đã tìm thấy file zip: {DRIVE_ZIP_PATH}')\n"
            "        break\n\n"
            "if DRIVE_ZIP_PATH is None:\n"
            "    print('⚠️ Chưa thấy ở các đường dẫn cố định, sẽ tự động dò tìm sâu dưới /content/drive...')\n"
            "    DRIVE_ZIP_PATH = 'auto'\n\n"
            "# NUM_PATIENTS:\n"
            "# - 0   : Giải nén toàn bộ 1010 bệnh nhân (cần ~40-60GB SSD Colab).\n"
            "# - > 0 : Chọn N bệnh nhân đầu tiên (vd: 50, 100, 200) để huấn luyện nhanh và tiết kiệm dung lượng đĩa.\n"
            "NUM_PATIENTS = 0\n\n"
            "cache_dir = run_colab(\n"
            "    zip_path=DRIVE_ZIP_PATH,\n"
            "    dest_dir='/content/data/cache_1mm',\n"
            "    n_patients=NUM_PATIENTS,\n"
            ")\n"
            "print(f'✅ Dữ liệu sẵn sàng tại: {cache_dir}')"
        )),
        _make_cell("code", (
            "# 8. Cấu hình chi tiết & Bắt đầu huấn luyện mô hình Mamba-UNet\n"
            "import os\n\n"
            "# =============================================================================\n"
            "# BẢNG THAM SỐ TÙY CHỈNH (Tùy chỉnh trực tiếp các giá trị bên dưới)\n"
            "# =============================================================================\n"
            "# 1. Kiến trúc mô hình & Backbone MambaVision\n"
            "BACKBONE = \"mamba_vision_T\"          # Lựa chọn: 'mamba_vision_T', 'mamba_vision_T2', 'mamba_vision_S', 'mamba_vision_B'\n"
            "PRETRAINED = True                   # True: Dùng trọng số ImageNet-1K pretrained; False: Khởi tạo ngẫu nhiên\n"
            "FREEZE_BACKBONE_EPOCHS = 5          # Số epoch đầu đóng băng backbone (-1: đóng băng suốt, 0: không đóng băng)\n"
            "BACKBONE_LR_MULT = 0.1              # Hệ số nhân learning rate cho backbone so với decoder\n\n"
            "# 2. Hàm kích hoạt Decoder (Activation)\n"
            "# Lựa chọn: 'relu', 'gelu', 'silu', 'mish', 'leaky_relu', 'star_relu', 'soft_clamp_relu', 'soft_clamp_silu', 'identity'\n"
            "ACTIVATION = \"relu\"\n\n"
            "# 3. Hàm mất mát (Loss Function)\n"
            "# Lựa chọn: 'dice_bce', 'dice', 'bce', 'tversky', 'focal_tversky', 'focal'\n"
            "LOSS = \"dice_bce\"\n"
            "LOSS_PARAMS = '{\"bce_weight\": 0.5, \"dice_weight\": 0.5}'  # Chuỗi JSON tham số loss, hoặc để trống ''\n\n"
            "# 4. Siêu tham số huấn luyện (Training Hyperparameters)\n"
            "EPOCHS = 50                         # Tổng số epoch huấn luyện\n"
            "BATCH_SIZE = 8                      # Kích thước batch (khuyến nghị 8 trên Colab T4 16GB)\n"
            "LR = 1e-4                           # Tốc độ học cơ sở (Learning rate)\n"
            "WEIGHT_DECAY = 1e-4                 # Hệ số suy giảm trọng số AdamW\n"
            "SCHEDULER = \"cosine\"                # Lựa chọn: 'cosine', 'plateau', 'step', 'none'\n"
            "GRAD_CLIP = 1.0                     # Ngưỡng kẹp gradient tối đa (0 = tắt)\n"
            "PATIENCE = 12                       # Số epoch chờ Early Stopping\n"
            "WORKERS = 2                         # Số luồng nạp dữ liệu DataLoader (Colab khuyến nghị 2-4)\n\n"
            "# 5. Quản lý phiên chạy & Khôi phục (Resume)\n"
            "# RESUME: None (chạy mới từ đầu), 'auto' (tiếp tục phiên gần nhất bị ngắt), hoặc đường dẫn checkpoint/run cụ thể\n"
            "RESUME = None                       # vd: 'auto' hoặc 'mamba_unet/runs/run_xxx/last_model.pth'\n"
            "INIT_WEIGHTS = None                 # Nạp trọng số ban đầu nhưng tạo run mới (None nếu không dùng)\n"
            "RUN_NAME = \"mamba_colab_run\"         # Tên phiên chạy (để trống '' để tự sinh theo timestamp)\n"
            "TAG = \"colab_mamba\"                 # Thẻ phân loại thí nghiệm\n"
            "NOTES = \"Huấn luyện Mamba-UNet với kernel CUDA native selective_scan trên Colab\"\n"
            "SEED = 42\n\n"
            "# 6. Tối ưu phần cứng & Chuẩn bị dữ liệu\n"
            "FORCE_CUDA_BACKEND = True           # Ép dùng kernel CUDA native (DUNEAI_MAMBA_BACKEND=cuda)\n"
            "USE_AMP = True                      # Bật Automatic Mixed Precision fp16 tăng tốc\n"
            "GPU_WINDOW = False                  # Chuyển HU int16 lên GPU chuẩn hóa cửa sổ\n"
            "NO_LUNG_MASK = False                # True: Không áp mặt nạ phổi ngoài lát cắt\n"
            "NO_AUGMENT = False                  # True: Tắt lật/xoay ngẫu nhiên\n"
            "LIMIT_TRAIN = 0                     # Giới hạn số lát cắt train (0 = toàn bộ)\n"
            "LIMIT_VAL = 0                       # Giới hạn số lát cắt val (0 = toàn bộ)\n\n"
            "# =============================================================================\n"
            "# TỔ HỢP LỆNH HUẤN LUYỆN\n"
            "# =============================================================================\n"
            "env_prefix = \"DUNEAI_MAMBA_BACKEND=cuda \" if FORCE_CUDA_BACKEND else \"\"\n"
            "cmd_parts = [\n"
            "    f\"{env_prefix}python -m mamba_unet.train_main train\",\n"
            "    f\"--cache-dir /content/data/cache_1mm\",\n"
            "    f\"--backbone {BACKBONE}\",\n"
            "    f\"{'--pretrained' if PRETRAINED else '--no-pretrained'}\",\n"
            "    f\"--freeze-backbone-epochs {FREEZE_BACKBONE_EPOCHS}\",\n"
            "    f\"--backbone-lr-mult {BACKBONE_LR_MULT}\",\n"
            "    f\"--activation {ACTIVATION}\",\n"
            "    f\"--loss {LOSS}\",\n"
            "    f\"--epochs {EPOCHS}\",\n"
            "    f\"--batch-size {BATCH_SIZE}\",\n"
            "    f\"--lr {LR}\",\n"
            "    f\"--weight-decay {WEIGHT_DECAY}\",\n"
            "    f\"--scheduler {SCHEDULER}\",\n"
            "    f\"--grad-clip {GRAD_CLIP}\",\n"
            "    f\"--patience {PATIENCE}\",\n"
            "    f\"--workers {WORKERS}\",\n"
            "    f\"--seed {SEED}\",\n"
            "]\n"
            "if LOSS_PARAMS and LOSS_PARAMS.strip():\n"
            "    cmd_parts.append(f\"--loss-params '{LOSS_PARAMS.strip()}'\")\n"
            "if RESUME:\n"
            "    cmd_parts.append(f\"--resume {RESUME}\")\n"
            "if INIT_WEIGHTS:\n"
            "    cmd_parts.append(f\"--init-weights '{INIT_WEIGHTS}'\")\n"
            "if RUN_NAME:\n"
            "    cmd_parts.append(f\"--run-name '{RUN_NAME}'\")\n"
            "if TAG:\n"
            "    cmd_parts.append(f\"--tag '{TAG}'\")\n"
            "if NOTES:\n"
            "    cmd_parts.append(f\"--notes '{NOTES}'\")\n"
            "if not USE_AMP:\n"
            "    cmd_parts.append(\"--no-amp\")\n"
            "if GPU_WINDOW:\n"
            "    cmd_parts.append(\"--gpu-window\")\n"
            "if NO_LUNG_MASK:\n"
            "    cmd_parts.append(\"--no-lung-mask\")\n"
            "if NO_AUGMENT:\n"
            "    cmd_parts.append(\"--no-augment\")\n"
            "if LIMIT_TRAIN > 0:\n"
            "    cmd_parts.append(f\"--limit-train {LIMIT_TRAIN}\")\n"
            "if LIMIT_VAL > 0:\n"
            "    cmd_parts.append(f\"--limit-val {LIMIT_VAL}\")\n\n"
            "full_train_cmd = \" \\\n    \".join(cmd_parts)\n"
            "print(\"🚀 Đang khởi động tiến trình huấn luyện...\")\n"
            "print(f\"Lệnh thực thi:\\n{full_train_cmd}\\n\" + \"=\" * 80)\n"
            "!{full_train_cmd}"
        )),
        _make_cell("code", (
            "# 9. Đánh giá 3D cấp độ bệnh nhân trên tập Validation\n"
            "!python -m mamba_unet.evaluate eval \\\n"
            "    --cache-dir /content/data/cache_1mm \\\n"
            "    --checkpoint checkpoints/best_model.pth \\\n"
            "    --split val"
        )),
        _make_cell("code", (
            "# 10. Lưu kết quả và Checkpoint vào Google Drive theo từng thư mục phiên chạy\n"
            "# Tự động tạo mới hoặc cập nhật nối thêm (append) file REPORT.md và JSON tổng kết\n"
            "import datetime, glob, json, os, shutil\n"
            "from pathlib import Path\n\n"
            "# 1. Xác định thư mục mirror Google Drive\n"
            "MIRROR_SOURCES_DIR = '/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES'\n"
            "if not os.path.exists(MIRROR_SOURCES_DIR):\n"
            "    MIRROR_SOURCES_DIR = '/content/drive/MyDrive/GG_1/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES'\n"
            "if not os.path.exists(MIRROR_SOURCES_DIR):\n"
            "    MIRROR_SOURCES_DIR = '/content/drive/MyDrive/DuneAI_Backups'\n\n"
            "# 2. Tìm thư mục phiên chạy gần nhất trong mamba_unet/runs\n"
            "local_runs = sorted(glob.glob(\"mamba_unet/runs/*\"), key=os.path.getmtime)\n"
            "if local_runs:\n"
            "    active_local_run = Path(local_runs[-1])\n"
            "    run_folder_name = active_local_run.name\n"
            "else:\n"
            "    run_folder_name = f\"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}\"\n"
            "    active_local_run = Path(f\"mamba_unet/runs/{run_folder_name}\")\n"
            "    active_local_run.mkdir(parents=True, exist_ok=True)\n\n"
            "# 3. Thư mục đích trên Google Drive\n"
            "drive_run_dir = Path(MIRROR_SOURCES_DIR) / \"mamba_unet\" / \"runs\" / run_folder_name\n"
            "drive_ckpt_dir = Path(MIRROR_SOURCES_DIR) / \"checkpoints\"\n"
            "drive_run_dir.mkdir(parents=True, exist_ok=True)\n"
            "drive_ckpt_dir.mkdir(parents=True, exist_ok=True)\n\n"
            "# 4. Trích xuất thông tin kết quả huấn luyện\n"
            "now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')\n"
            "best_val_dice = \"N/A\"\n"
            "best_epoch = \"N/A\"\n"
            "if (active_local_run / \"train_log.csv\").exists():\n"
            "    try:\n"
            "        import pandas as pd\n"
            "        df_log = pd.read_csv(active_local_run / \"train_log.csv\")\n"
            "        if \"val_dice\" in df_log.columns and len(df_log) > 0:\n"
            "            best_idx = df_log[\"val_dice\"].idxmax()\n"
            "            best_val_dice = f\"{df_log.loc[best_idx, 'val_dice']:.4f}\"\n"
            "            best_epoch = f\"{int(df_log.loc[best_idx, 'epoch'])}\"\n"
            "    except Exception:\n"
            "        pass\n\n"
            "# 5. Soạn thảo / Cập nhật file báo cáo REPORT.md\n"
            "report_md_path = drive_run_dir / \"REPORT.md\"\n"
            "is_resumed_session = bool(RESUME) and report_md_path.exists()\n\n"
            "if is_resumed_session:\n"
            "    print(f\"🔄 Đang ghi đè/nối thêm (append) thông tin phiên Resume vào: {report_md_path}\")\n"
            "    resume_block = f\"\"\"\n"
            "---\n"
            "### 🔄 Phiên Tiếp Tục Huấn Luyện (Resume Session) - {now_str}\n"
            "- **Khôi phục từ**: `{RESUME}`\n"
            "- **Số Epochs đích**: {EPOCHS} | **Batch Size**: {BATCH_SIZE} | **Learning Rate**: {LR}\n"
            "- **Scheduler**: {SCHEDULER} | **Workers**: {WORKERS}\n"
            "- **Best Val Dice hiện tại**: **{best_val_dice}** (tại Epoch {best_epoch})\n"
            "- **Ghi chú**: {NOTES}\n"
            "\"\"\"\n"
            "    with open(report_md_path, \"a\", encoding=\"utf-8\") as f:\n"
            "        f.write(resume_block)\n"
            "else:\n"
            "    print(f\"📝 Đang tạo mới file báo cáo tổng hợp: {report_md_path}\")\n"
            "    new_report = f\"\"\"# 📋 Báo Cáo Huấn Luyện Mamba-UNet: {run_folder_name}\n\n"
            "- **Thời gian khởi tạo**: {now_str}\n"
            "- **Kiến trúc mô hình**: Mamba-UNet\n"
            "  - **Backbone**: `{BACKBONE}` (Pretrained: `{PRETRAINED}`)\n"
            "  - **Đóng băng Backbone**: {FREEZE_BACKBONE_EPOCHS} epochs đầu (LR mult: {BACKBONE_LR_MULT})\n"
            "  - **Decoder Activation**: `{ACTIVATION}`\n"
            "- **Hàm mất mát (Loss)**: `{LOSS}` (Tham số: `{LOSS_PARAMS}`)\n"
            "- **Siêu tham số tối ưu**:\n"
            "  - **Optimizer**: AdamW (LR: `{LR}`, Weight Decay: `{WEIGHT_DECAY}`)\n"
            "  - **LR Scheduler**: `{SCHEDULER}`\n"
            "  - **Gradient Clip**: `{GRAD_CLIP}` | **Patience (Early Stopping)**: `{PATIENCE}`\n"
            "- **Cấu hình DataLoader**:\n"
            "  - **Batch Size**: {BATCH_SIZE} | **Workers**: {WORKERS} | **Seed**: {SEED}\n"
            "- **Backend Mamba**: CUDA Native Selective Scan (`mamba_ssm` + `causal_conv1d`)\n"
            "- **Kết quả huấn luyện**:\n"
            "  - **Best Val Dice**: **{best_val_dice}** (tại Epoch {best_epoch})\n"
            "- **Ghi chú**: {NOTES}\n"
            "\"\"\"\n"
            "    with open(report_md_path, \"w\", encoding=\"utf-8\") as f:\n"
            "        f.write(new_report)\n\n"
            "# Sao lưu báo cáo vào thư mục cục bộ và lưu thêm bản JSON\n"
            "shutil.copy2(report_md_path, active_local_run / \"REPORT.md\")\n"
            "report_json_data = {\n"
            "    \"run_folder\": run_folder_name,\n"
            "    \"timestamp\": now_str,\n"
            "    \"arch\": \"mamba_unet\",\n"
            "    \"backbone\": BACKBONE,\n"
            "    \"pretrained\": PRETRAINED,\n"
            "    \"activation\": ACTIVATION,\n"
            "    \"loss\": LOSS,\n"
            "    \"loss_params\": LOSS_PARAMS,\n"
            "    \"epochs\": EPOCHS,\n"
            "    \"batch_size\": BATCH_SIZE,\n"
            "    \"lr\": LR,\n"
            "    \"scheduler\": SCHEDULER,\n"
            "    \"workers\": WORKERS,\n"
            "    \"best_val_dice\": best_val_dice,\n"
            "    \"best_epoch\": best_epoch,\n"
            "    \"resume\": RESUME,\n"
            "    \"notes\": NOTES,\n"
            "}\n"
            "with open(drive_run_dir / \"training_report.json\", \"w\", encoding=\"utf-8\") as f:\n"
            "    json.dump(report_json_data, f, indent=2, ensure_ascii=False)\n"
            "shutil.copy2(drive_run_dir / \"training_report.json\", active_local_run / \"training_report.json\")\n\n"
            "# 6. Sao chép toàn bộ checkpoints và artifacts của phiên chạy vào Drive\n"
            "print(f\"🚀 Đang sao lưu toàn bộ files từ {active_local_run} sang {drive_run_dir}...\")\n"
            "!cp -ru {active_local_run}/* '{drive_run_dir}/' 2>/dev/null || true\n"
            "!cp -ru checkpoints/* '{drive_ckpt_dir}/' 2>/dev/null || true\n\n"
            "print(\"=\" * 80)\n"
            "print(f\"✅ HOÀN TẤT SAO LƯU!\")\n"
            "print(f\"📁 Thư mục phiên chạy tại Google Drive: {drive_run_dir}\")\n"
            "print(f\"📄 Báo cáo chi tiết: {report_md_path}\")\n"
            "print(f\"🎯 Google Drive Desktop sẽ tự động đồng bộ thư mục này về máy tính của bạn!\")\n"
            "print(\"=\" * 80)"
        )),
    ]
    return _build_notebook(cells)


# =============================================================================
# NỘI DUNG NOTEBOOK COLAB CHO UNET-BASE
# =============================================================================
def _make_colab_unet_notebook() -> Dict[str, Any]:
    cells = [
        _make_cell("markdown", (
            "# 🩺 Huấn luyện UNetBase (Classic CNN) trên Google Colab\n\n"
            "Quy trình chuẩn hóa toàn diện:\n"
            "1. **Kiểm tra phần cứng GPU** (`!nvidia-smi`)\n"
            "2. **Đồng bộ mã nguồn từ GitHub**\n"
            "3. **Cài đặt các thư viện y tế phụ thuộc** (SimpleITK, etc.)\n"
            "4. **Mount Google Drive**\n"
            "5. **Giải nén dữ liệu từ Google Drive vào SSD cục bộ** (có tùy chọn số lượng ca bệnh nhân `NUM_PATIENTS`)\n"
            "6. **Huấn luyện mô hình với đầy đủ tham số tùy chỉnh** (Features, Activation, Loss, LR, Workers, Resume...)\n"
            "7. **Đánh giá 3D cấp độ bệnh nhân** trên tập Validation\n"
            "8. **Lưu kết quả & Checkpoints vào Google Drive theo từng thư mục phiên chạy kèm bản báo cáo Markdown/JSON** (tự động append nếu Resume)"
        )),
        _make_cell("code", "# 1. Kiểm tra phần cứng và GPU khả dụng\n!nvidia-smi"),
        _make_cell("code", (
            "# 2. Tải và đồng bộ mã nguồn từ GitHub\n"
            "import os, sys\n\n"
            "GITHUB_REPO = 'https://github.com/NguyenMinhTri24072005/CT208E_LIDC_DuneAI_MambaVision.git'\n"
            "BRANCH = 'main'  # Nhánh mặc định trên GitHub (đổi nếu bạn dùng nhánh khác)\n"
            "REPO_DIR = '/content/CT208E_LIDC_DuneAI_MambaVision'\n\n"
            "if os.path.exists('unet_base') and os.path.exists('common'):\n"
            "    print(f'✅ Đang chạy trực tiếp trong thư mục dự án: {os.getcwd()}')\n"
            "else:\n"
            "    if not os.path.exists(REPO_DIR):\n"
            "        print(f'📥 Đang clone repository từ GitHub (nhánh {BRANCH})...')\n"
            "        !git clone -b {BRANCH} {GITHUB_REPO} {REPO_DIR}\n"
            "    else:\n"
            "        print(f'🔄 Đang đồng bộ mã nguồn mới nhất từ GitHub...')\n"
            "        %cd {REPO_DIR}\n"
            "        !git fetch origin\n"
            "        !git checkout {BRANCH}\n"
            "        !git pull origin {BRANCH}\n"
            "    %cd {REPO_DIR}\n\n"
            "if os.getcwd() not in sys.path:\n"
            "    sys.path.insert(0, os.getcwd())\n\n"
            "print(f'🚀 Thư mục làm việc hiện tại: {os.getcwd()}')"
        )),
        _make_cell("code", "# 3. Cài đặt các thư viện phụ thuộc\n!pip install -q SimpleITK pandas scipy opencv-python-headless pyyaml matplotlib"),
        _make_cell("code", "# 4. Kết nối với Google Drive\nfrom google.colab import drive\ndrive.mount('/content/drive')"),
        _make_cell("code", (
            "# 5. Chuẩn bị dữ liệu từ Google Drive vào SSD cục bộ (/content/data/cache_1mm)\n"
            "# Thư mục GG_1 trên máy tính được mirror trực tiếp vào Google Drive.\n"
            "# Mã sẽ tự động tìm kiếm file zip theo đúng cấu trúc mirror của bạn:\n"
            "import os\n"
            "from common.data_pipeline.runtime_data import run_colab\n\n"
            "DRIVE_ZIP_CANDIDATES = [\n"
            "    # 1. Mirror trực tiếp thư mục GG_1 vào MyDrive:\n"
            "    '/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/data/cache_1mm.zip',\n"
            "    # 2. Mirror có thư mục cha GG_1 trong MyDrive:\n"
            "    '/content/drive/MyDrive/GG_1/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/data/cache_1mm.zip',\n"
            "    # 3. File zip tại thư mục KAGGLE_UPLOAD:\n"
            "    '/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/data/KAGGLE_UPLOAD/cache_1mm.zip',\n"
            "    # 4. File zip đặt ngay tại gốc MyDrive:\n"
            "    '/content/drive/MyDrive/cache_1mm.zip',\n"
            "]\n\n"
            "DRIVE_ZIP_PATH = None\n"
            "for cand in DRIVE_ZIP_CANDIDATES:\n"
            "    if os.path.exists(cand):\n"
            "        DRIVE_ZIP_PATH = cand\n"
            "        print(f'🎯 Đã tìm thấy file zip: {DRIVE_ZIP_PATH}')\n"
            "        break\n\n"
            "if DRIVE_ZIP_PATH is None:\n"
            "    print('⚠️ Chưa thấy ở các đường dẫn cố định, sẽ tự động dò tìm sâu dưới /content/drive...')\n"
            "    DRIVE_ZIP_PATH = 'auto'\n\n"
            "# NUM_PATIENTS:\n"
            "# - 0   : Giải nén toàn bộ 1010 bệnh nhân (cần ~40-60GB SSD Colab).\n"
            "# - > 0 : Chọn N bệnh nhân đầu tiên (vd: 50, 100, 200) để huấn luyện nhanh và tiết kiệm dung lượng đĩa.\n"
            "NUM_PATIENTS = 0\n\n"
            "cache_dir = run_colab(\n"
            "    zip_path=DRIVE_ZIP_PATH,\n"
            "    dest_dir='/content/data/cache_1mm',\n"
            "    n_patients=NUM_PATIENTS,\n"
            ")\n"
            "print(f'✅ Dữ liệu sẵn sàng tại: {cache_dir}')"
        )),
        _make_cell("code", (
            "# 6. Cấu hình chi tiết & Bắt đầu huấn luyện mô hình UNetBase\n"
            "import os\n\n"
            "# =============================================================================\n"
            "# BẢNG THAM SỐ TÙY CHỈNH (Tùy chỉnh trực tiếp các giá trị bên dưới)\n"
            "# =============================================================================\n"
            "# 1. Kiến trúc mô hình UNetBase\n"
            "FEATURES = 64                       # Số kênh đặc trưng cơ sở (mặc định 64: [64, 128, 256, 512])\n\n"
            "# 2. Hàm kích hoạt (Activation)\n"
            "# Lựa chọn: 'relu', 'leaky_relu', 'gelu', 'silu', 'mish', 'identity'\n"
            "ACTIVATION = \"relu\"\n\n"
            "# 3. Hàm mất mát (Loss Function)\n"
            "# Lựa chọn: 'dice_bce', 'dice', 'bce', 'tversky', 'focal_tversky', 'focal'\n"
            "LOSS = \"dice_bce\"\n"
            "LOSS_PARAMS = '{\"bce_weight\": 0.5, \"dice_weight\": 0.5}'  # Chuỗi JSON tham số loss, hoặc để trống ''\n\n"
            "# 4. Siêu tham số huấn luyện (Training Hyperparameters)\n"
            "EPOCHS = 50                         # Tổng số epoch huấn luyện\n"
            "BATCH_SIZE = 8                      # Kích thước batch\n"
            "LR = 1e-4                           # Tốc độ học cơ sở (Learning rate)\n"
            "WEIGHT_DECAY = 1e-4                 # Hệ số suy giảm trọng số AdamW\n"
            "SCHEDULER = \"cosine\"                # Lựa chọn: 'cosine', 'plateau', 'step', 'none'\n"
            "GRAD_CLIP = 1.0                     # Ngưỡng kẹp gradient tối đa (0 = tắt)\n"
            "PATIENCE = 12                       # Số epoch chờ Early Stopping\n"
            "WORKERS = 2                         # Số luồng nạp dữ liệu DataLoader (Colab khuyến nghị 2-4)\n\n"
            "# 5. Quản lý phiên chạy & Khôi phục (Resume)\n"
            "# RESUME: None (chạy mới từ đầu), 'auto' (tiếp tục phiên gần nhất bị ngắt), hoặc đường dẫn checkpoint/run cụ thể\n"
            "RESUME = None                       # vd: 'auto' hoặc 'unet_base/runs/run_xxx/last_model.pth'\n"
            "INIT_WEIGHTS = None                 # Nạp trọng số ban đầu nhưng tạo run mới (None nếu không dùng)\n"
            "RUN_NAME = \"unet_colab_run\"         # Tên phiên chạy (để trống '' để tự sinh theo timestamp)\n"
            "TAG = \"colab_unet\"                 # Thẻ phân loại thí nghiệm\n"
            "NOTES = \"Huấn luyện UNetBase trên Colab\"\n"
            "SEED = 42\n\n"
            "# 6. Tối ưu phần cứng & Chuẩn bị dữ liệu\n"
            "USE_AMP = True                      # Bật Automatic Mixed Precision fp16 tăng tốc\n"
            "GPU_WINDOW = False                  # Chuyển HU int16 lên GPU chuẩn hóa cửa sổ\n"
            "NO_LUNG_MASK = False                # True: Không áp mặt nạ phổi ngoài lát cắt\n"
            "NO_AUGMENT = False                  # True: Tắt lật/xoay ngẫu nhiên\n"
            "LIMIT_TRAIN = 0                     # Giới hạn số lát cắt train (0 = toàn bộ)\n"
            "LIMIT_VAL = 0                       # Giới hạn số lát cắt val (0 = toàn bộ)\n\n"
            "# =============================================================================\n"
            "# TỔ HỢP LỆNH HUẤN LUYỆN\n"
            "# =============================================================================\n"
            "cmd_parts = [\n"
            "    \"python -m unet_base.train_main train\",\n"
            "    f\"--cache-dir /content/data/cache_1mm\",\n"
            "    f\"--features {FEATURES}\",\n"
            "    f\"--activation {ACTIVATION}\",\n"
            "    f\"--loss {LOSS}\",\n"
            "    f\"--epochs {EPOCHS}\",\n"
            "    f\"--batch-size {BATCH_SIZE}\",\n"
            "    f\"--lr {LR}\",\n"
            "    f\"--weight-decay {WEIGHT_DECAY}\",\n"
            "    f\"--scheduler {SCHEDULER}\",\n"
            "    f\"--grad-clip {GRAD_CLIP}\",\n"
            "    f\"--patience {PATIENCE}\",\n"
            "    f\"--workers {WORKERS}\",\n"
            "    f\"--seed {SEED}\",\n"
            "]\n"
            "if LOSS_PARAMS and LOSS_PARAMS.strip():\n"
            "    cmd_parts.append(f\"--loss-params '{LOSS_PARAMS.strip()}'\")\n"
            "if RESUME:\n"
            "    cmd_parts.append(f\"--resume {RESUME}\")\n"
            "if INIT_WEIGHTS:\n"
            "    cmd_parts.append(f\"--init-weights '{INIT_WEIGHTS}'\")\n"
            "if RUN_NAME:\n"
            "    cmd_parts.append(f\"--run-name '{RUN_NAME}'\")\n"
            "if TAG:\n"
            "    cmd_parts.append(f\"--tag '{TAG}'\")\n"
            "if NOTES:\n"
            "    cmd_parts.append(f\"--notes '{NOTES}'\")\n"
            "if not USE_AMP:\n"
            "    cmd_parts.append(\"--no-amp\")\n"
            "if GPU_WINDOW:\n"
            "    cmd_parts.append(\"--gpu-window\")\n"
            "if NO_LUNG_MASK:\n"
            "    cmd_parts.append(\"--no-lung-mask\")\n"
            "if NO_AUGMENT:\n"
            "    cmd_parts.append(\"--no-augment\")\n"
            "if LIMIT_TRAIN > 0:\n"
            "    cmd_parts.append(f\"--limit-train {LIMIT_TRAIN}\")\n"
            "if LIMIT_VAL > 0:\n"
            "    cmd_parts.append(f\"--limit-val {LIMIT_VAL}\")\n\n"
            "full_train_cmd = \" \\\n    \".join(cmd_parts)\n"
            "print(\"🚀 Đang khởi động tiến trình huấn luyện...\")\n"
            "print(f\"Lệnh thực thi:\\n{full_train_cmd}\\n\" + \"=\" * 80)\n"
            "!{full_train_cmd}"
        )),
        _make_cell("code", (
            "# 7. Đánh giá 3D cấp độ bệnh nhân trên tập Validation\n"
            "!python -m unet_base.evaluate eval \\\n"
            "    --cache-dir /content/data/cache_1mm \\\n"
            "    --checkpoint checkpoints/best_model.pth \\\n"
            "    --split val"
        )),
        _make_cell("code", (
            "# 8. Lưu kết quả và Checkpoint vào Google Drive theo từng thư mục phiên chạy\n"
            "# Tự động tạo mới hoặc cập nhật nối thêm (append) file REPORT.md và JSON tổng kết\n"
            "import datetime, glob, json, os, shutil\n"
            "from pathlib import Path\n\n"
            "# 1. Xác định thư mục mirror Google Drive\n"
            "MIRROR_SOURCES_DIR = '/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES'\n"
            "if not os.path.exists(MIRROR_SOURCES_DIR):\n"
            "    MIRROR_SOURCES_DIR = '/content/drive/MyDrive/GG_1/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES'\n"
            "if not os.path.exists(MIRROR_SOURCES_DIR):\n"
            "    MIRROR_SOURCES_DIR = '/content/drive/MyDrive/DuneAI_Backups'\n\n"
            "# 2. Tìm thư mục phiên chạy gần nhất trong unet_base/runs\n"
            "local_runs = sorted(glob.glob(\"unet_base/runs/*\"), key=os.path.getmtime)\n"
            "if local_runs:\n"
            "    active_local_run = Path(local_runs[-1])\n"
            "    run_folder_name = active_local_run.name\n"
            "else:\n"
            "    run_folder_name = f\"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}\"\n"
            "    active_local_run = Path(f\"unet_base/runs/{run_folder_name}\")\n"
            "    active_local_run.mkdir(parents=True, exist_ok=True)\n\n"
            "# 3. Thư mục đích trên Google Drive\n"
            "drive_run_dir = Path(MIRROR_SOURCES_DIR) / \"unet_base\" / \"runs\" / run_folder_name\n"
            "drive_ckpt_dir = Path(MIRROR_SOURCES_DIR) / \"checkpoints\"\n"
            "drive_run_dir.mkdir(parents=True, exist_ok=True)\n"
            "drive_ckpt_dir.mkdir(parents=True, exist_ok=True)\n\n"
            "# 4. Trích xuất thông tin kết quả huấn luyện\n"
            "now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')\n"
            "best_val_dice = \"N/A\"\n"
            "best_epoch = \"N/A\"\n"
            "if (active_local_run / \"train_log.csv\").exists():\n"
            "    try:\n"
            "        import pandas as pd\n"
            "        df_log = pd.read_csv(active_local_run / \"train_log.csv\")\n"
            "        if \"val_dice\" in df_log.columns and len(df_log) > 0:\n"
            "            best_idx = df_log[\"val_dice\"].idxmax()\n"
            "            best_val_dice = f\"{df_log.loc[best_idx, 'val_dice']:.4f}\"\n"
            "            best_epoch = f\"{int(df_log.loc[best_idx, 'epoch'])}\"\n"
            "    except Exception:\n"
            "        pass\n\n"
            "# 5. Soạn thảo / Cập nhật file báo cáo REPORT.md\n"
            "report_md_path = drive_run_dir / \"REPORT.md\"\n"
            "is_resumed_session = bool(RESUME) and report_md_path.exists()\n\n"
            "if is_resumed_session:\n"
            "    print(f\"🔄 Đang ghi đè/nối thêm (append) thông tin phiên Resume vào: {report_md_path}\")\n"
            "    resume_block = f\"\"\"\n"
            "---\n"
            "### 🔄 Phiên Tiếp Tục Huấn Luyện (Resume Session) - {now_str}\n"
            "- **Khôi phục từ**: `{RESUME}`\n"
            "- **Số Epochs đích**: {EPOCHS} | **Batch Size**: {BATCH_SIZE} | **Learning Rate**: {LR}\n"
            "- **Scheduler**: {SCHEDULER} | **Workers**: {WORKERS}\n"
            "- **Best Val Dice hiện tại**: **{best_val_dice}** (tại Epoch {best_epoch})\n"
            "- **Ghi chú**: {NOTES}\n"
            "\"\"\"\n"
            "    with open(report_md_path, \"a\", encoding=\"utf-8\") as f:\n"
            "        f.write(resume_block)\n"
            "else:\n"
            "    print(f\"📝 Đang tạo mới file báo cáo tổng hợp: {report_md_path}\")\n"
            "    new_report = f\"\"\"# 📋 Báo Cáo Huấn Luyện UNetBase: {run_folder_name}\n\n"
            "- **Thời gian khởi tạo**: {now_str}\n"
            "- **Kiến trúc mô hình**: UNetBase (Classic CNN)\n"
            "  - **Features**: {FEATURES} (Kênh cơ sở)\n"
            "  - **Decoder Activation**: `{ACTIVATION}`\n"
            "- **Hàm mất mát (Loss)**: `{LOSS}` (Tham số: `{LOSS_PARAMS}`)\n"
            "- **Siêu tham số tối ưu**:\n"
            "  - **Optimizer**: AdamW (LR: `{LR}`, Weight Decay: `{WEIGHT_DECAY}`)\n"
            "  - **LR Scheduler**: `{SCHEDULER}`\n"
            "  - **Gradient Clip**: `{GRAD_CLIP}` | **Patience (Early Stopping)**: `{PATIENCE}`\n"
            "- **Cấu hình DataLoader**:\n"
            "  - **Batch Size**: {BATCH_SIZE} | **Workers**: {WORKERS} | **Seed**: {SEED}\n"
            "- **Kết quả huấn luyện**:\n"
            "  - **Best Val Dice**: **{best_val_dice}** (tại Epoch {best_epoch})\n"
            "- **Ghi chú**: {NOTES}\n"
            "\"\"\"\n"
            "    with open(report_md_path, \"w\", encoding=\"utf-8\") as f:\n"
            "        f.write(new_report)\n\n"
            "# Sao lưu báo cáo vào thư mục cục bộ và lưu thêm bản JSON\n"
            "shutil.copy2(report_md_path, active_local_run / \"REPORT.md\")\n"
            "report_json_data = {\n"
            "    \"run_folder\": run_folder_name,\n"
            "    \"timestamp\": now_str,\n"
            "    \"arch\": \"unet_base\",\n"
            "    \"features\": FEATURES,\n"
            "    \"activation\": ACTIVATION,\n"
            "    \"loss\": LOSS,\n"
            "    \"loss_params\": LOSS_PARAMS,\n"
            "    \"epochs\": EPOCHS,\n"
            "    \"batch_size\": BATCH_SIZE,\n"
            "    \"lr\": LR,\n"
            "    \"scheduler\": SCHEDULER,\n"
            "    \"workers\": WORKERS,\n"
            "    \"best_val_dice\": best_val_dice,\n"
            "    \"best_epoch\": best_epoch,\n"
            "    \"resume\": RESUME,\n"
            "    \"notes\": NOTES,\n"
            "}\n"
            "with open(drive_run_dir / \"training_report.json\", \"w\", encoding=\"utf-8\") as f:\n"
            "    json.dump(report_json_data, f, indent=2, ensure_ascii=False)\n"
            "shutil.copy2(drive_run_dir / \"training_report.json\", active_local_run / \"training_report.json\")\n\n"
            "# 6. Sao chép toàn bộ checkpoints và artifacts của phiên chạy vào Drive\n"
            "print(f\"🚀 Đang sao lưu toàn bộ files từ {active_local_run} sang {drive_run_dir}...\")\n"
            "!cp -ru {active_local_run}/* '{drive_run_dir}/' 2>/dev/null || true\n"
            "!cp -ru checkpoints/* '{drive_ckpt_dir}/' 2>/dev/null || true\n\n"
            "print(\"=\" * 80)\n"
            "print(f\"✅ HOÀN TẤT SAO LƯU!\")\n"
            "print(f\"📁 Thư mục phiên chạy tại Google Drive: {drive_run_dir}\")\n"
            "print(f\"📄 Báo cáo chi tiết: {report_md_path}\")\n"
            "print(f\"🎯 Google Drive Desktop sẽ tự động đồng bộ thư mục này về máy tính của bạn!\")\n"
            "print(\"=\" * 80)"
        )),
    ]
    return _build_notebook(cells)


def make_colab_notebook(arch: str) -> Dict[str, Any]:
    if arch == "mamba_unet":
        return _make_colab_mamba_notebook()
    else:
        return _make_colab_unet_notebook()



# =============================================================================
# NỘI DUNG NOTEBOOK KAGGLE
# =============================================================================
def make_kaggle_notebook(arch: str) -> Dict[str, Any]:
    is_mamba = (arch == "mamba_unet")
    title = "Mamba-UNet (MambaVision Backbone)" if is_mamba else "UNetBase (Classic CNN)"
    train_cmd = (
        "!python -m mamba_unet.train_main train \\\n"
        "    --cache-dir /kaggle/working/data/cache_1mm \\\n"
        "    --backbone mamba_vision_T \\\n"
        "    --pretrained \\\n"
        "    --freeze-backbone-epochs 5 \\\n"
        "    --backbone-lr-mult 0.1 \\\n"
        "    --epochs 50 \\\n"
        "    --batch-size 8 \\\n"
        "    --lr 1e-4 \\\n"
        "    --tag kaggle_run"
    ) if is_mamba else (
        "!python -m unet_base.train_main train \\\n"
        "    --cache-dir /kaggle/working/data/cache_1mm \\\n"
        "    --features 64 \\\n"
        "    --epochs 50 \\\n"
        "    --batch-size 8 \\\n"
        "    --lr 1e-4 \\\n"
        "    --tag kaggle_run"
    )
    eval_cmd = (
        "!python -m mamba_unet.evaluate eval \\\n"
        "    --cache-dir /kaggle/working/data/cache_1mm \\\n"
        "    --checkpoint checkpoints/best_model.pth \\\n"
        "    --split val"
    ) if is_mamba else (
        "!python -m unet_base.evaluate eval \\\n"
        "    --cache-dir /kaggle/working/data/cache_1mm \\\n"
        "    --checkpoint checkpoints/best_model.pth \\\n"
        "    --split val"
    )

    cells = [
        _make_cell("markdown", f"# 🩺 Huấn luyện {title} trên Kaggle Environment\n\nQuy trình: Clone/đồng bộ mã nguồn GitHub -> Dò tìm dataset trong /kaggle/input (kể cả nested path cache_1mm/cache_1mm) -> Dựng symlink view-dir ghi được trong /kaggle/working -> Huấn luyện -> Đánh giá 3D."),
        _make_cell("code", "# 1. Kiểm tra GPU\n!nvidia-smi"),
        _make_cell("code", (
            "# 2. Tải và đồng bộ mã nguồn từ GitHub (Yêu cầu bật Internet: ON trong Settings bên phải)\n"
            "import os\n"
            "import sys\n\n"
            "GITHUB_REPO = 'https://github.com/NguyenMinhTri24072005/CT208E_LIDC_DuneAI_MambaVision.git'\n"
            "BRANCH = 'main'  # Nhánh mặc định trên GitHub (đổi nếu bạn dùng nhánh khác)\n"
            "REPO_DIR = '/kaggle/working/CT208E_LIDC_DuneAI_MambaVision'\n\n"
            f"if os.path.exists('{arch}') and os.path.exists('common'):\n"
            "    print(f'✅ Đang chạy trực tiếp trong thư mục dự án: {os.getcwd()}')\n"
            "else:\n"
            "    if not os.path.exists(REPO_DIR):\n"
            "        print(f'📥 Đang clone repository từ GitHub (nhánh {BRANCH})...')\n"
            "        !git clone -b {BRANCH} {GITHUB_REPO} {REPO_DIR}\n"
            "    else:\n"
            "        print(f'🔄 Đang đồng bộ mã nguồn mới nhất từ GitHub...')\n"
            "        %cd {REPO_DIR}\n"
            "        !git fetch origin\n"
            "        !git checkout {BRANCH}\n"
            "        !git pull origin {BRANCH}\n"
            "    %cd {REPO_DIR}\n\n"
            "if os.getcwd() not in sys.path:\n"
            "    sys.path.insert(0, os.getcwd())\n\n"
            "print(f'🚀 Thư mục làm việc hiện tại: {os.getcwd()}')"
        )),
        _make_cell("code", "# 3. Cài đặt các thư viện cần thiết\n!pip install -q SimpleITK pyyaml opencv-python-headless"),
        _make_cell("code", (
            "# 4. Chuẩn bị dữ liệu Kaggle Input qua Symlink View-Dir\n"
            "# Hỗ trợ đường dẫn trực tiếp hoặc tự động dò tìm dưới /kaggle/input\n"
            "from common.data_pipeline.runtime_data import run_kaggle\n\n"
            "# Đường dẫn dataset mẫu hoặc thư mục cha chứa dataset\n"
            "KAGGLE_HINT = '/kaggle/input/datasets/minhtri2475/lidc-idri-1mm-cache-ct208e/cache_1mm/cache_1mm'\n"
            "VIEW_DIR = '/kaggle/working/data/cache_1mm'\n\n"
            "cache_dir = run_kaggle(\n"
            "    hint=KAGGLE_HINT,\n"
            "    view_dir=VIEW_DIR,\n"
            "    n_patients=0,  # 0 = toàn bộ bệnh nhân có trong dataset\n"
            ")\n"
            "print(f'✅ Dữ liệu sẵn sàng tại: {cache_dir}')"
        )),
        _make_cell("code", f"# 5. Bắt đầu quá trình huấn luyện\n{train_cmd}"),
        _make_cell("code", f"# 6. Đánh giá 3D cấp độ bệnh nhân trên tập Validation\n{eval_cmd}"),
    ]
    return _build_notebook(cells)


# =============================================================================
# NỘI DUNG NOTEBOOK LOCAL
# =============================================================================
def make_local_notebook(arch: str) -> Dict[str, Any]:
    is_mamba = (arch == "mamba_unet")
    title = "Mamba-UNet (MambaVision Backbone)" if is_mamba else "UNetBase (Classic CNN)"
    train_cmd = (
        "!python -m mamba_unet.train_main train \\\n"
        "    --cache-dir data/cache_1mm \\\n"
        "    --backbone mamba_vision_T \\\n"
        "    --pretrained \\\n"
        "    --freeze-backbone-epochs 5 \\\n"
        "    --backbone-lr-mult 0.1 \\\n"
        "    --epochs 50 \\\n"
        "    --batch-size 8 \\\n"
        "    --lr 1e-4"
    ) if is_mamba else (
        "!python -m unet_base.train_main train \\\n"
        "    --cache-dir data/cache_1mm \\\n"
        "    --features 64 \\\n"
        "    --epochs 50 \\\n"
        "    --batch-size 8 \\\n"
        "    --lr 1e-4"
    )
    eval_cmd = (
        "!python -m mamba_unet.evaluate eval \\\n"
        "    --cache-dir data/cache_1mm \\\n"
        "    --checkpoint checkpoints/best_model.pth \\\n"
        "    --split val"
    ) if is_mamba else (
        "!python -m unet_base.evaluate eval \\\n"
        "    --cache-dir data/cache_1mm \\\n"
        "    --checkpoint checkpoints/best_model.pth \\\n"
        "    --split val"
    )

    cells = [
        _make_cell("markdown", f"# 🩺 Huấn luyện {title} trên Máy cục bộ (Local Workstation)\n\nSử dụng trực tiếp dữ liệu tại `data/cache_1mm`."),
        _make_cell("code", "# 1. Kiểm tra môi trường PyTorch và thiết bị tính toán\nimport torch\nprint(f'PyTorch version: {torch.__version__}')\nprint(f'CUDA Available: {torch.cuda.is_available()}')\nif torch.cuda.is_available():\n    print(f'Device: {torch.cuda.get_device_name(0)}')"),
        _make_cell("code", f"# 2. Huấn luyện mô hình\n{train_cmd}"),
        _make_cell("code", f"# 3. Đánh giá 3D cấp độ bệnh nhân\n{eval_cmd}"),
    ]
    return _build_notebook(cells)


# =============================================================================
# BỘ SINH TOÀN DIỆN VÀ CLI
# =============================================================================
ALL_CONFIGS = [
    ("unet_base", "colab", "colab_train.ipynb", make_colab_notebook),
    ("unet_base", "kaggle", "kaggle_train.ipynb", make_kaggle_notebook),
    ("unet_base", "local", "local_train.ipynb", make_local_notebook),
    ("mamba_unet", "colab", "colab_train.ipynb", make_colab_notebook),
    ("mamba_unet", "kaggle", "kaggle_train.ipynb", make_kaggle_notebook),
    ("mamba_unet", "local", "local_train.ipynb", make_local_notebook),
]


def generate_all_notebooks(root_dir: Path) -> List[Path]:
    """Sinh toàn bộ 6 file Jupyter Notebook tại đúng cấu trúc thư mục quy định."""
    created_paths = []
    for arch, platform, filename, builder_fn in ALL_CONFIGS:
        out_dir = root_dir / arch / "notebooks"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        nb_data = builder_fn(arch)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1, ensure_ascii=False)
        print(f"  ✓ Đã sinh notebook: {out_path.relative_to(root_dir)}")
        created_paths.append(out_path)
    return created_paths


def _selftest():
    print("🧪 Bắt đầu selftest scripts.make_notebooks...")
    generated = generate_all_notebooks(_ROOT)
    assert len(generated) == 6, f"Phải sinh đủ 6 notebooks, nhận được {len(generated)}"

    for nb_path in generated:
        assert nb_path.exists(), f"Không tìm thấy file: {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("nbformat") == 4, f"nbformat phải là 4: {nb_path}"
        assert len(data.get("cells", [])) >= 3, f"Số cells quá ít: {nb_path}"

        # Kiểm tra tính độc lập tuyệt đối giữa 2 kiến trúc trong các cell
        arch = nb_path.parent.parent.name
        code_text = "".join("".join(c.get("source", [])) for c in data["cells"])
        if arch == "unet_base":
            assert "mamba_unet" not in code_text, f"unet_base notebook chứa mamba_unet: {nb_path}"
        elif arch == "mamba_unet":
            assert "unet_base" not in code_text, f"mamba_unet notebook chứa unet_base: {nb_path}"

    print("✅ Selftest scripts.make_notebooks hoàn tất thành công (6/6 notebooks chuẩn xác)!")


def _check_notebooks(root_dir: Path):
    """Kiểm tra hợp lệ tính hợp chuẩn nbformat của tất cả notebooks."""
    print("🔍 Đang kiểm tra tính hợp chuẩn của các notebook...")
    try:
        import nbformat
        has_nbformat = True
    except ImportError:
        has_nbformat = False

    checked_count = 0
    for arch, platform, filename, _ in ALL_CONFIGS:
        nb_path = root_dir / arch / "notebooks" / filename
        if not nb_path.exists():
            print(f"  ⚠️ Chưa tìm thấy {filename}, đang sinh tự động...")
            generate_all_notebooks(root_dir)
            break

    for arch, platform, filename, _ in ALL_CONFIGS:
        nb_path = root_dir / arch / "notebooks" / filename
        assert nb_path.exists(), f"Không tìm thấy file: {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("nbformat") == 4, f"nbformat phải là 4: {nb_path}"
        assert len(data.get("cells", [])) >= 3, f"Số cells quá ít: {nb_path}"

        if has_nbformat:
            nb = nbformat.from_dict(data)
            nbformat.validate(nb)

        # Kiểm tra tính độc lập
        code_text = "".join("".join(c.get("source", [])) for c in data["cells"])
        if arch == "unet_base":
            assert "mamba_unet" not in code_text, f"unet_base notebook chứa mamba_unet: {nb_path}"
        elif arch == "mamba_unet":
            assert "unet_base" not in code_text, f"mamba_unet notebook chứa unet_base: {nb_path}"

        print(f"  ✓ Hợp lệ nbformat v4: {nb_path.relative_to(root_dir)}")
        checked_count += 1

    print(f"✅ Hoàn tất kiểm tra: Toàn bộ {checked_count}/6 notebooks đạt chuẩn nbformat v4!")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Sinh tự động và kiểm tra 6 Jupyter Notebooks cho dự án DuneAI")
    parser.add_argument("--check", action="store_true", help="Kiểm tra hợp lệ nbformat của toàn bộ notebooks")
    parser.add_argument("--selftest", action="store_true", help="Chạy kiểm thử tự động sinh và kiểm tra")
    args = parser.parse_args()

    if args.check:
        _check_notebooks(_ROOT)
    elif args.selftest:
        _selftest()
    else:
        print("📦 Đang sinh toàn bộ 6 notebooks chuẩn cho unet_base và mamba_unet...")
        generate_all_notebooks(_ROOT)
        print("🎉 Hoàn tất sinh notebooks!")


if __name__ == "__main__":
    main()
