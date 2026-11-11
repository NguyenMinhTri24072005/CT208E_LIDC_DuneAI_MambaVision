# -*- coding: utf-8 -*-
"""
scripts/make_step0_notebook.py - Sinh notebook colab_step0_measurement.ipynb cho Mamba-UNet.
"""
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]


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
                "name": "python",
                "version": "3.10.12",
            },
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_step0_notebook() -> Dict[str, Any]:
    cells = [
        _make_cell(
            "markdown",
            (
                "# 🔬 BƯỚC 0: Đo Lường & Đánh Giá 3D Chuyên Sâu (Không Cần Huấn Luyện Lại)\n\n"
                "**Tài liệu tham khảo**: `docs/FINE-TUNING-DOCS/MAMBA_UNET_REVIEW_AND_FIXES.md` (Mục 2: BƯỚC 0)\n\n"
                "### 🎯 Mục Tiêu Đo Lường\n"
                "Trước khi thực hiện chu kỳ fine-tuning mới, notebook này thực hiện đo lường khách quan mô hình hiện tại "
                "(ví dụ: checkpoint `mamba_colab_500item_25epoch_extra_test` với Val Dice 2D = 0.7456) qua 2 công cụ chẩn đoán chuyên sâu:\n\n"
                "1. **Kiểm tra chất lượng Cache & FOV (`tools/check_cache.py`)**:\n"
                "   - Đánh giá chỉ số `mask_in_lung_fraction`: Mặt nạ phổi tự động có vô tình cắt phạm vào khối u nằm sát màng phổi không?\n"
                "   - Đánh giá nguy cơ mất nhãn khối u khi pad/crop nếu hạ `IMAGE_SIZE` từ 512 xuống 448 để tăng tốc độ huấn luyện.\n\n"
                "2. **Quét ngưỡng xác suất & Đánh giá 3D mức bệnh nhân / nốt u (`tools/eval_sweep.py`)**:\n"
                "   - Trả lời: **Ngưỡng nhị phân tối ưu thực sự là bao nhiêu?** (0.3, 0.4, 0.5, 0.6 hay 0.7?)\n"
                "   - Định lượng: **Bao nhiêu dương tính giả trên mỗi ca bệnh (`fp_comps`)?**\n"
                "   - Phân tích theo kích thước: **Độ nhạy phát hiện theo đường kính tương đương** (0-5mm, 5-10mm, 10-20mm, $\\ge 20$mm).\n\n"
                "3. **Trực quan hóa & Lưu trữ kết quả**:\n"
                "   - Vẽ 4 biểu đồ then chốt: FROC curve, Precision-Recall, Dice 3D theo ngưỡng và độ nhạy theo kích thước nốt.\n"
                "   - Toàn bộ CSV, biểu đồ PNG và báo cáo Markdown được tự động đồng bộ sang Google Drive.\n\n"
                "---"
            ),
        ),
        _make_cell(
            "code",
            (
                "# 1. Kiểm tra phần cứng và GPU khả dụng\n"
                "!nvidia-smi"
            ),
        ),
        _make_cell(
            "code",
            (
                "# 2. Thiết lập môi trường PyTorch 2.6.0 (cu126) + Triton 3.3.1 & Thư viện y tế\n"
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
                "print(\"\\n⚙️ Cài đặt Triton 3.3.1 và các thư viện cần thiết...\")\n"
                "!python -m pip install triton==3.3.1 SimpleITK pandas scipy opencv-python-headless pyyaml matplotlib seaborn tabulate -q\n"
                "trt = probe(\"import triton;print(triton.__version__, hasattr(triton,'set_allocator'))\")\n"
                "print(\"Triton:\", trt)\n"
                "assert trt and trt.endswith(\"True\"), f\"❌ Triton chưa có set_allocator (đang là: {trt})\"\n\n"
                "print(\"\\n✅ CELL 2 XONG. Chạy tiếp cell 3 để mount Drive và nạp checkpoints.\")"
            ),
        ),
        _make_cell(
            "code",
            (
                "# 3. Mount Google Drive & Dò tìm thư mục dự án / Checkpoints / Wheels\n"
                "import os, subprocess, sys\n"
                "from pathlib import Path\n"
                "from google.colab import drive\n\n"
                "print(\"🔗 Đang kết nối Google Drive...\")\n"
                "drive.mount('/content/drive')\n\n"
                "# Dò tìm thư mục mirror trên Google Drive:\n"
                "SOURCES_CANDIDATES = [\n"
                "    \"/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES\",\n"
                "    \"/content/drive/MyDrive/GG_1/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES\",\n"
                "    \"/content/drive/MyDrive/CT208E/SOURCES\",\n"
                "    \"/content/drive/MyDrive/DuneAI_Backups\",\n"
                "]\n\n"
                "DRIVE_SOURCES_DIR = None\n"
                "for d in SOURCES_CANDIDATES:\n"
                "    if os.path.isdir(d):\n"
                "        DRIVE_SOURCES_DIR = d\n"
                "        print(f\"🎯 Đã tìm thấy thư mục dự án trên Drive: {DRIVE_SOURCES_DIR}\")\n"
                "        break\n\n"
                "if DRIVE_SOURCES_DIR is None:\n"
                "    print(\"⚠️ Chưa tìm thấy theo đường dẫn mặc định, đặt thư mục backup tại /content/drive/MyDrive/DuneAI_Backups\")\n"
                "    DRIVE_SOURCES_DIR = \"/content/drive/MyDrive/DuneAI_Backups\"\n"
                "    os.makedirs(DRIVE_SOURCES_DIR, exist_ok=True)\n\n"
                "# Tự động dò tìm thư mục wheels trên Google Drive để nạp CUDA kernel (nếu cần):\n"
                "WHEEL_CANDIDATES = [\n"
                "    f\"{DRIVE_SOURCES_DIR}/wheels\",\n"
                "    \"/content/drive/MyDrive/wheels\",\n"
                "]\n"
                "WHEEL_DIR = None\n"
                "for d in WHEEL_CANDIDATES:\n"
                "    if os.path.isdir(d):\n"
                "        WHEEL_DIR = d\n"
                "        print(f\"🎯 Đã tìm thấy thư mục wheels tại: {WHEEL_DIR}\")\n"
                "        break\n\n"
                "if WHEEL_DIR:\n"
                "    r = subprocess.run([sys.executable, \"-c\", \"import torch;print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')\"],\n"
                "                       capture_output=True, text=True)\n"
                "    abi = r.stdout.strip()\n"
                "    CAUSAL_WHL = f\"{WHEEL_DIR}/causal_conv1d-1.7.0+cu12torch2.6cxx11abi{abi}-cp313-cp313-linux_x86_64.whl\"\n"
                "    MAMBA_WHL  = f\"{WHEEL_DIR}/mamba_ssm-2.3.2.post1+cu12torch2.6cxx11abi{abi}-cp313-cp313-linux_x86_64.whl\"\n"
                "    if os.path.exists(CAUSAL_WHL) and os.path.exists(MAMBA_WHL):\n"
                "        print(\"🚀 Cài đặt 2 wheel causal_conv1d & mamba_ssm từ Drive...\")\n"
                "        !python -m pip install \"{CAUSAL_WHL}\" --force-reinstall --no-deps -q\n"
                "        !python -m pip install \"{MAMBA_WHL}\" --force-reinstall --no-deps -q\n"
                "        print(\"✅ Đã cài đặt xong CUDA wheels.\")"
            ),
        ),
        _make_cell(
            "code",
            (
                "# 4. Khởi động lại Runtime (Chỉ cần chạy nếu Cell 3 vừa cài mới wheels)\n"
                "# ⚠️ Lưu ý: Khi chạy cell này Colab sẽ hiện thông báo 'crashed and recovered' để nạp C++ extension.\n"
                "# Nếu môi trường đã có sẵn mamba_ssm, bạn có thể bỏ qua cell này và chạy luôn Cell 5.\n"
                "import os\n"
                "# Bỏ comment dòng dưới nếu cần restart runtime:\n"
                "# os.kill(os.getpid(), 9)"
            ),
        ),
        _make_cell(
            "code",
            (
                "# 5. Kiểm tra xác minh môi trường CUDA Mamba-SSM (Selective Scan)\n"
                "import sys, traceback, torch\n\n"
                "print(\"python :\", f\"cp{sys.version_info.major}{sys.version_info.minor}\")\n"
                "print(\"torch  :\", torch.__version__, \"| ABI =\", getattr(torch._C, \"_GLIBCXX_USE_CXX11_ABI\", False))\n"
                "print(\"GPU    :\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"KHÔNG CÓ\")\n\n"
                "try:\n"
                "    import mamba_ssm, selective_scan_cuda\n"
                "    print(\"mamba_ssm     : ✅ CUDA native selective_scan khả dụng!\")\n"
                "except Exception as e:\n"
                "    print(f\"ℹ️ mamba_ssm CUDA: {e} -> Sẽ sử dụng PyTorch scan fallback an toàn nếu cần.\")"
            ),
        ),
        _make_cell(
            "code",
            (
                "# 6. Tải và đồng bộ mã nguồn mới nhất từ GitHub\n"
                "import os, sys\n\n"
                "GITHUB_REPO = 'https://github.com/NguyenMinhTri24072005/CT208E_LIDC_DuneAI_MambaVision.git'\n"
                "BRANCH = 'main'\n"
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
            ),
        ),
        _make_cell(
            "code",
            (
                "# 7. Chuẩn bị dữ liệu Cache (cache_1mm) trên SSD Colab\n"
                "import os\n"
                "from pathlib import Path\n"
                "from common.data_pipeline.runtime_data import run_colab\n\n"
                "DRIVE_ZIP_CANDIDATES = [\n"
                "    f\"{DRIVE_SOURCES_DIR}/data/cache_1mm.zip\",\n"
                "    f\"{DRIVE_SOURCES_DIR}/data/KAGGLE_UPLOAD/cache_1mm.zip\",\n"
                "    \"/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/data/cache_1mm.zip\",\n"
                "    \"/content/drive/MyDrive/cache_1mm.zip\",\n"
                "]\n\n"
                "DRIVE_ZIP_PATH = None\n"
                "for cand in DRIVE_ZIP_CANDIDATES:\n"
                "    if os.path.exists(cand):\n"
                "        DRIVE_ZIP_PATH = cand\n"
                "        print(f'🎯 Đã tìm thấy file zip: {DRIVE_ZIP_PATH}')\n"
                "        break\n\n"
                "if DRIVE_ZIP_PATH is None:\n"
                "    DRIVE_ZIP_PATH = 'auto'\n\n"
                "# Nếu thư mục cache đã tồn tại sẵn trên SSD cục bộ và có slices.csv, ta bỏ qua bước giải nén\n"
                "local_cache = Path('/content/data/cache_1mm')\n"
                "if local_cache.exists() and (local_cache / 'slices.csv').exists():\n"
                "    cache_dir = str(local_cache)\n"
                "    print(f'⏭️ Đã có dữ liệu sẵn sàng trên SSD cục bộ: {cache_dir}')\n"
                "else:\n"
                "    print('📦 Đang giải nén dữ liệu từ Drive vào SSD cục bộ...')\n"
                "    cache_dir = run_colab(\n"
                "        zip_path=DRIVE_ZIP_PATH,\n"
                "        dest_dir='/content/data/cache_1mm',\n"
                "        n_patients=500,  # Đồng nhất với 500 ca bệnh của phiên chạy\n"
                "    )\n"
                "    print(f'✅ Dữ liệu sẵn sàng tại: {cache_dir}')"
            ),
        ),
        _make_cell(
            "markdown",
            (
                "### 2.2 Kiểm Tra Chất Lượng Cache & FOV (`tools/check_cache.py`)\n\n"
                "**Mục đích**:\n"
                "1. **Đánh giá tỷ lệ `mask_in_lung_fraction`**: Mặt nạ phổi có che mất một phần nốt u (đặc biệt nốt u nằm sát màng phổi) hay không. "
                "Nếu có nhiều ca `< 0.9`, đây là nguyên nhân gây giảm Precision/Recall cần xử lý ở P3.\n"
                "2. **Đánh giá FOV**: Nếu hạ kích thước ảnh huấn luyện `IMAGE_SIZE` từ 512 xuống 448 (để tăng tốc độ và tăng batch size), "
                "có bao nhiêu ca bị cắt mất khối u khi căn giữa (`pad_or_crop_xy`)."
            ),
        ),
        _make_cell(
            "code",
            (
                "# Chạy kiểm tra cache với kích thước 512 (chuẩn hiện tại) và 448 (phương án tối ưu hóa)\n"
                "print(\"--- 1. KIỂM TRA FOV & MASK PHỔI VỚI IMAGE_SIZE = 512 ---\")\n"
                "!python tools/check_cache.py /content/data/cache_1mm 512\n\n"
                "print(\"\\n--- 2. KIỂM TRA FOV & MASK PHỔI VỚI IMAGE_SIZE = 448 ---\")\n"
                "!python tools/check_cache.py /content/data/cache_1mm 448"
            ),
        ),
        _make_cell(
            "markdown",
            (
                "### 2.1 Quét Ngưỡng & Đánh Giá 3D Mức Bệnh Nhân / Nốt U (`tools/eval_sweep.py`)\n\n"
                "**Mục đích**:\n"
                "- Nạp checkpoint tốt nhất (`best_model.pth`) trực tiếp từ Google Drive.\n"
                "- Quét qua dải ngưỡng `[0.3, 0.4, 0.5, 0.6, 0.7]` trên tập Validation.\n"
                "- Ghi nhận:\n"
                "  - Dice 3D, Precision 3D, Recall 3D theo từng ngưỡng.\n"
                "  - Số lượng dương tính giả trên mỗi ca (`fp_comps`).\n"
                "  - Độ nhạy phát hiện nốt u theo từng dải kích thước (0-5mm, 5-10mm, 10-20mm, $\\ge 20$mm)."
            ),
        ),
        _make_cell(
            "code",
            (
                "import os, glob\n"
                "from pathlib import Path\n\n"
                "# =============================================================================\n"
                "# CẤU HÌNH THAM SỐ ĐÁNH GIÁ (BƯỚC 0)\n"
                "# =============================================================================\n"
                "# 1. Đường dẫn checkpoint trên Google Drive (hoặc cục bộ)\n"
                "CHECKPOINT_CANDIDATES = [\n"
                "    # Checkpoint phiên chạy 500 ca 25 epoch gần nhất:\n"
                "    f\"{DRIVE_SOURCES_DIR}/mamba_unet/runs/mamba_colab_500item_25epoch_extra_test/best_model.pth\",\n"
                "    f\"{DRIVE_SOURCES_DIR}/checkpoints/best_model.pth\",\n"
                "    \"mamba_unet/runs/mamba_colab_500item_25epoch_extra_test/best_model.pth\",\n"
                "]\n\n"
                "CHECKPOINT_PATH = None\n"
                "for cp in CHECKPOINT_CANDIDATES:\n"
                "    if os.path.exists(cp):\n"
                "        CHECKPOINT_PATH = cp\n"
                "        print(f\"🎯 Đã tìm thấy Checkpoint mục tiêu: {CHECKPOINT_PATH}\")\n"
                "        break\n\n"
                "if CHECKPOINT_PATH is None:\n"
                "    # Tìm kiếm tự động bất kỳ file best_model.pth nào trong thư mục runs trên Drive\n"
                "    found = glob.glob(f\"{DRIVE_SOURCES_DIR}/mamba_unet/runs/**/best_model.pth\", recursive=True)\n"
                "    if found:\n"
                "        CHECKPOINT_PATH = found[0]\n"
                "        print(f\"🎯 Tự động tìm thấy Checkpoint trên Drive: {CHECKPOINT_PATH}\")\n"
                "    else:\n"
                "        raise FileNotFoundError(f\"❌ Không tìm thấy checkpoint best_model.pth trên Drive ({DRIVE_SOURCES_DIR})!\")\n\n"
                "# 2. Tham số quét ngưỡng\n"
                "THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]  # Danh sách ngưỡng xác suất quét\n"
                "BACKBONE = \"mamba_vision_T\"\n"
                "MIN_AREA = 20.0                         # Lọc thành phần liên thông < 20 mm² (chuẩn hậu xử lý LIDC)\n"
                "SPLIT = \"val\"                           # Đánh giá trên tập Validation\n"
                "TTA = False                             # Test-Time Augmentation (True: flip TTA, đánh giá kỹ hơn)\n"
                "LIMIT_PATIENTS = 0                      # 0 = Đánh giá toàn bộ ca trong tập Val; N = Thử nhanh N ca đầu\n\n"
                "# 3. Thư mục xuất kết quả (lưu đồng thời tại cục bộ và Google Drive)\n"
                "EVAL_OUT_DIR = f\"{DRIVE_SOURCES_DIR}/eval_results/step0_sweep\"\n"
                "os.makedirs(EVAL_OUT_DIR, exist_ok=True)\n\n"
                "th_args = \" \".join(str(t) for t in THRESHOLDS)\n"
                "tta_flag = \"--tta\" if TTA else \"\"\n"
                "limit_flag = f\"--limit {LIMIT_PATIENTS}\" if LIMIT_PATIENTS > 0 else \"\"\n\n"
                "eval_cmd = (\n"
                "    f\"DUNEAI_MAMBA_BACKEND=cuda python tools/eval_sweep.py \"\n"
                "    f\"--cache-dir /content/data/cache_1mm \"\n"
                "    f\"--checkpoint \\\"{CHECKPOINT_PATH}\\\" \"\n"
                "    f\"--backbone {BACKBONE} \"\n"
                "    f\"--split {SPLIT} \"\n"
                "    f\"--thresholds {th_args} \"\n"
                "    f\"--min-area {MIN_AREA} \"\n"
                "    f\"{tta_flag} {limit_flag} \"\n"
                "    f\"--out-dir \\\"{EVAL_OUT_DIR}\\\" \"\n"
                "    f\"--out-prefix sweep_r0\"\n"
                ")\n\n"
                "print(\"🚀 Bắt đầu thực thi quét ngưỡng 3D...\")\n"
                "print(f\"Lệnh: {eval_cmd}\\n\" + \"=\" * 80)\n"
                "!{eval_cmd}"
            ),
        ),
        _make_cell(
            "markdown",
            (
                "### 2.3 Phân Tích Dữ Liệu & Vẽ Đồ Thị Y Tế Trực Quan\n\n"
                "Vẽ 4 biểu đồ then chốt:\n"
                "1. **Trade-off Dice 3D, Precision 3D & Recall 3D theo ngưỡng**: Tìm điểm cực trị Dice 3D.\n"
                "2. **Số lượng dương tính giả trên mỗi ca (`fp_comps`)**: Định lượng áp lực dương tính giả đối với bác sĩ chẩn đoán hình ảnh.\n"
                "3. **Độ nhạy phát hiện theo kích thước nốt u (mm)**: Bộc lộ rõ tình trạng bỏ sót nốt nhỏ (< 5mm, 5-10mm).\n"
                "4. **Đường cong FROC (Free-Response ROC)**: Tương quan giữa Recall nốt u và Dương tính giả trên mỗi ca bệnh."
            ),
        ),
        _make_cell(
            "code",
            (
                "import os\n"
                "import pandas as pd\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "from pathlib import Path\n\n"
                "# Đọc kết quả từ thư mục đã lưu\n"
                "out_dir = Path(EVAL_OUT_DIR)\n"
                "pat_csv = out_dir / \"sweep_r0_patients.csv\"\n"
                "nod_csv = out_dir / \"sweep_r0_nodules.csv\"\n\n"
                "assert pat_csv.exists() and nod_csv.exists(), f\"❌ Chưa tìm thấy file kết quả tại {out_dir}!\"\n\n"
                "P = pd.read_csv(pat_csv)\n"
                "N = pd.read_csv(nod_csv)\n\n"
                "print(\"=\" * 70)\n"
                "print(\"📊 BẢNG TỔNG HỢP HIỆU NĂNG 3D MỨC BỆNH NHÂN (TRUNG BÌNH THEO NGƯỠNG)\")\n"
                "print(\"=\" * 70)\n"
                "summary_df = P.groupby(\"th\")[[\"dice\", \"precision\", \"recall\", \"fp_comps\"]].mean().reset_index()\n"
                "summary_df.columns = [\"Ngưỡng (Th)\", \"Dice 3D\", \"Precision 3D\", \"Recall 3D\", \"FP / Ca (fp_comps)\"]\n"
                "display(summary_df.round(4))\n\n"
                "print(\"\\n\" + \"=\" * 70)\n"
                "print(\"🎯 BẢNG ĐỘ NHẠY PHÁT HIỆN NỐT U THEO DẢI ĐƯỜNG KÍNH (MM)\")\n"
                "print(\"=\" * 70)\n"
                "BINS = [(0, 5), (5, 10), (10, 20), (20, 1e9)]\n"
                "bin_names = [\"0 - 5 mm (Rất nhỏ)\", \"5 - 10 mm (Nhỏ)\", \"10 - 20 mm (Trung bình)\", \">= 20 mm (Lớn)\"]\n"
                "sens_rows = []\n"
                "for (lo, hi), bname in zip(BINS, bin_names):\n"
                "    sub = N[(N[\"diam\"] >= lo) & (N[\"diam\"] < hi)]\n"
                "    if len(sub) > 0:\n"
                "        row = {\"Dải kích thước\": bname, \"Số nốt GT\": len(sub) // len(P[\"th\"].unique())}\n"
                "        th_sens = sub.groupby(\"th\")[\"detected\"].mean().to_dict()\n"
                "        for th, val in th_sens.items():\n"
                "            row[f\"Th={th}\"] = round(val, 3)\n"
                "        sens_rows.append(row)\n\n"
                "sens_df = pd.DataFrame(sens_rows)\n"
                "display(sens_df)\n\n"
                "# =============================================================================\n"
                "# VẼ ĐỒ THỊ CHUYÊN SÂU\n"
                "# =============================================================================\n"
                "sns.set_theme(style=\"whitegrid\", font_scale=1.1)\n"
                "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n\n"
                "# 1. Dice, Precision, Recall vs Threshold\n"
                "ax1 = axes[0, 0]\n"
                "ax1.plot(summary_df[\"Ngưỡng (Th)\"], summary_df[\"Dice 3D\"], marker='o', linewidth=2.5, label=\"Dice 3D\", color=\"#1f77b4\")\n"
                "ax1.plot(summary_df[\"Ngưỡng (Th)\"], summary_df[\"Precision 3D\"], marker='s', linewidth=2.5, label=\"Precision 3D\", color=\"#2ca02c\")\n"
                "ax1.plot(summary_df[\"Ngưỡng (Th)\"], summary_df[\"Recall 3D\"], marker='^', linewidth=2.5, label=\"Recall 3D\", color=\"#ff7f0e\")\n"
                "best_th_idx = summary_df[\"Dice 3D\"].idxmax()\n"
                "best_th = summary_df.loc[best_th_idx, \"Ngưỡng (Th)\"]\n"
                "best_dice = summary_df.loc[best_th_idx, \"Dice 3D\"]\n"
                "ax1.axvline(best_th, color='red', linestyle='--', alpha=0.7, label=f\"Best Th = {best_th} (Dice={best_dice:.4f})\")\n"
                "ax1.set_title(\"1. Dice 3D, Precision & Recall theo Ngưỡng\", fontsize=14, fontweight=\"bold\")\n"
                "ax1.set_xlabel(\"Ngưỡng xác suất (Threshold)\")\n"
                "ax1.set_ylabel(\"Giá trị Score\")\n"
                "ax1.set_ylim([0.0, 1.0])\n"
                "ax1.legend()\n\n"
                "# 2. FP Comps / Patient vs Threshold\n"
                "ax2 = axes[0, 1]\n"
                "bars = ax2.bar(summary_df[\"Ngưỡng (Th)\"].astype(str), summary_df[\"FP / Ca (fp_comps)\"], color=\"#d62728\", alpha=0.8, width=0.4)\n"
                "ax2.axhline(1.0, color='gray', linestyle=':', label=\"Ngưỡng khuyến cáo lâm sàng (<= 1 FP/ca)\")\n"
                "ax2.set_title(\"2. Số Lượng Dương Tính Giả Trung Bình Trên Mỗi Ca (FP/Ca)\", fontsize=14, fontweight=\"bold\")\n"
                "ax2.set_xlabel(\"Ngưỡng xác suất (Threshold)\")\n"
                "ax2.set_ylabel(\"Số lượng FP components / Ca\")\n"
                "for bar in bars:\n"
                "    yval = bar.get_height()\n"
                "    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f\"{yval:.2f}\", ha='center', va='bottom', fontweight='bold')\n"
                "ax2.legend()\n\n"
                "# 3. Sensitivity by Nodule Size\n"
                "ax3 = axes[1, 0]\n"
                "for idx, row in sens_df.iterrows():\n"
                "    bname = row[\"Dải kích thước\"]\n"
                "    y_vals = [row[f\"Th={th}\"] for th in summary_df[\"Ngưỡng (Th)\"]]\n"
                "    ax3.plot(summary_df[\"Ngưỡng (Th)\"], y_vals, marker='o', linewidth=2, label=bname)\n"
                "ax3.set_title(\"3. Độ Nhạy Phát Hiện Nốt U Theo Dải Đường Kính\", fontsize=14, fontweight=\"bold\")\n"
                "ax3.set_xlabel(\"Ngưỡng xác suất (Threshold)\")\n"
                "ax3.set_ylabel(\"Độ nhạy (Sensitivity / Detection Rate)\")\n"
                "ax3.set_ylim([0.0, 1.05])\n"
                "ax3.legend()\n\n"
                "# 4. FROC Curve (Recall vs FP/case)\n"
                "ax4 = axes[1, 1]\n"
                "ax4.plot(summary_df[\"FP / Ca (fp_comps)\"], summary_df[\"Recall 3D\"], marker='D', color=\"#9467bd\", linewidth=2.5)\n"
                "for i, txt in enumerate(summary_df[\"Ngưỡng (Th)\"]):\n"
                "    ax4.annotate(f\"Th={txt}\", (summary_df.loc[i, \"FP / Ca (fp_comps)\"], summary_df.loc[i, \"Recall 3D\"]),\n"
                "                 textcoords=\"offset points\", xytext=(8, -8), ha='left', fontweight='bold')\n"
                "ax4.set_title(\"4. Đường Cong FROC (Recall 3D vs FP/Ca)\", fontsize=14, fontweight=\"bold\")\n"
                "ax4.set_xlabel(\"Dương tính giả trung bình trên mỗi ca (FP/ca)\")\n"
                "ax4.set_ylabel(\"Recall 3D (Mức voxel)\")\n"
                "ax4.grid(True, linestyle=\"--\", alpha=0.6)\n\n"
                "plt.tight_layout()\n"
                "chart_path = out_dir / \"step0_diagnostic_charts.png\"\n"
                "plt.savefig(chart_path, dpi=300, bbox_inches=\"tight\")\n"
                "plt.show()\n"
                "print(f\"📊 Đã lưu biểu đồ chẩn đoán độ phân giải cao tại: {chart_path}\")"
            ),
        ),
        _make_cell(
            "markdown",
            (
                "### 2.4 Kết Luận Đo Lường & Xuất Báo Cáo Y Tế (`STEP0_EVALUATION_REPORT.md`)\n\n"
                "**Bảng quy chuẩn kết luận lâm sàng**:\n"
                "| Quan sát từ kết quả | Kết luận lâm sàng | Hành động tương ứng cho các bước tiếp theo |\n"
                "|---|---|---|\n"
                "| `fp_comps` lớn (≫ 1/ca) & Precision ≪ Recall | Dương tính giả là nút thắt chính | Giảm `pos_weight` xuống 2.0-3.0, tăng tỷ lệ mẫu âm tính `neg-ratio` lên 0.5-1.0, nâng ngưỡng inference |\n"
                "| Độ nhạy nốt < 5 mm rất thấp (< 30%) | Mất nốt nhỏ do cắt ngưỡng 20mm² khi dựng cache | Hạ ngưỡng lọc train xuống 0 mm² (`--min-train-area 0`), chỉ lọc ở hậu xử lý |\n"
                "| Đỉnh Dice 3D đạt tại ngưỡng $\\ne 0.5$ | Ngưỡng bị lệch do tỷ trọng phạt mất mát | Chọn ngưỡng tối ưu trên tập Validation để áp dụng cho báo cáo Test |"
            ),
        ),
        _make_cell(
            "code",
            (
                "import datetime\n\n"
                "best_idx = summary_df[\"Dice 3D\"].idxmax()\n"
                "best_th = summary_df.loc[best_idx, \"Ngưỡng (Th)\"]\n"
                "best_dice = summary_df.loc[best_idx, \"Dice 3D\"]\n"
                "best_prec = summary_df.loc[best_idx, \"Precision 3D\"]\n"
                "best_rec = summary_df.loc[best_idx, \"Recall 3D\"]\n"
                "best_fp = summary_df.loc[best_idx, \"FP / Ca (fp_comps)\"]\n\n"
                "report_content = f\"\"\"# 📋 BÁO CÁO KẾT QUẢ ĐO LƯỜNG BƯỚC 0 (STEP 0 MEASUREMENT REPORT)\n"
                "**Mô hình đánh giá**: `{Path(CHECKPOINT_PATH).name}`\n"
                "**Thư mục checkpoint**: `{CHECKPOINT_PATH}`\n"
                "**Thời gian thực hiện**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                "**Tập dữ liệu**: Validation set ({P['pid'].nunique()} bệnh nhân)\n\n"
                "---\n\n"
                "### 1. Kết Quả Quét Ngưỡng 3D Mức Bệnh Nhân\n"
                "- **Ngưỡng tối ưu Dice 3D**: `Threshold = {best_th}`\n"
                "- **Hiệu năng tại ngưỡng tối ưu**:\n"
                "  - **Dice 3D**: `{best_dice:.4f}`\n"
                "  - **Precision 3D**: `{best_prec:.4f}`\n"
                "  - **Recall 3D**: `{best_rec:.4f}`\n"
                "  - **Dương tính giả trung bình**: `{best_fp:.2f} FP/ca`\n"
                "- **So sánh với ngưỡng mặc định 0.5**:\n"
                "  - Dice 3D tại 0.5: `{summary_df.loc[summary_df['Ngưỡng (Th)'] == 0.5, 'Dice 3D'].values[0] if 0.5 in summary_df['Ngưỡng (Th)'].values else 'N/A'}`\n\n"
                "---\n\n"
                "### 2. Độ Nhạy Theo Kích Thước Nốt U\n"
                "{sens_df.to_markdown(index=False)}\n\n"
                "---\n\n"
                "### 3. Khuyến Nghị Hành Động (Action Plan) Cho Các Bước Tiếp Theo\n"
                "1. **Dương tính giả**: {'⚠️ fp_comps > 1/ca, cần giảm pos_weight và tăng lát âm tính (neg_ratio >= 0.5)' if best_fp > 1.0 else '✅ Mức độ FP/ca nằm trong giới hạn kiểm soát tốt.'}\n"
                "2. **Nốt nhỏ (< 5mm)**: Khuyến nghị hạ ngưỡng diện tích train xuống 0 mm² lúc dựng chỉ mục để bảo tồn toàn vẹn nốt u nhỏ trong quá trình học.\n"
                "3. **Kích thước ảnh**: Giữ `IMAGE_SIZE = 512` hoặc cân nhắc `448` nếu không có ca bệnh nào bị cắt mất nhãn trong kiểm tra FOV.\n\n"
                "---\n"
                "*Báo cáo được tự động sinh bởi Colab Step 0 Measurement Notebook.*\n"
                "\"\"\"\n\n"
                "report_file = out_dir / \"STEP0_EVALUATION_REPORT.md\"\n"
                "with open(report_file, \"w\", encoding=\"utf-8\") as f:\n"
                "    f.write(report_content)\n\n"
                "print(f\"✅ ĐÃ XUẤT THÀNH CÔNG BÁO CÁO TẠI:\\n   {report_file}\")\n"
                "print(\"\\n\" + \"=\" * 70)\n"
                "print(report_content)\n"
                "print(\"=\" * 70)"
            ),
        ),
    ]
    return _build_notebook(cells)


def main():
    out_path = _ROOT / "mamba_unet" / "notebooks" / "colab_step0_measurement.ipynb"
    nb = make_step0_notebook()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"✅ Đã tạo thành công notebook: {out_path}")


if __name__ == "__main__":
    main()
