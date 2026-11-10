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
# NỘI DUNG NOTEBOOK COLAB
# =============================================================================
def make_colab_notebook(arch: str) -> Dict[str, Any]:
    is_mamba = (arch == "mamba_unet")
    title = "Mamba-UNet (MambaVision Backbone)" if is_mamba else "UNetBase (Classic CNN)"
    train_cmd = (
        "!python -m mamba_unet.train_main train \\\n"
        "    --cache-dir /content/data/cache_1mm \\\n"
        "    --backbone mamba_vision_T \\\n"
        "    --pretrained \\\n"
        "    --freeze-backbone-epochs 5 \\\n"
        "    --backbone-lr-mult 0.1 \\\n"
        "    --epochs 50 \\\n"
        "    --batch-size 8 \\\n"
        "    --lr 1e-4 \\\n"
        "    --tag colab_run"
    ) if is_mamba else (
        "!python -m unet_base.train_main train \\\n"
        "    --cache-dir /content/data/cache_1mm \\\n"
        "    --features 64 \\\n"
        "    --epochs 50 \\\n"
        "    --batch-size 8 \\\n"
        "    --lr 1e-4 \\\n"
        "    --tag colab_run"
    )
    eval_cmd = (
        "!python -m mamba_unet.evaluate eval \\\n"
        "    --cache-dir /content/data/cache_1mm \\\n"
        "    --checkpoint checkpoints/best_model.pth \\\n"
        "    --split val"
    ) if is_mamba else (
        "!python -m unet_base.evaluate eval \\\n"
        "    --cache-dir /content/data/cache_1mm \\\n"
        "    --checkpoint checkpoints/best_model.pth \\\n"
        "    --split val"
    )

    cells = [
        _make_cell("markdown", f"# 🩺 Huấn luyện {title} trên Google Colab\n\nQuy trình toàn diện: Clone mã nguồn từ GitHub -> Cài đặt thư viện -> Mount Drive -> Giải nén chọn số lượng ca bệnh nhân vào SSD -> Huấn luyện -> Đánh giá 3D -> Lưu Checkpoint về Drive."),
        _make_cell("code", "# 1. Kiểm tra phần cứng và GPU khả dụng\n!nvidia-smi"),
        _make_cell("code", (
            "# 2. Tải và đồng bộ mã nguồn từ GitHub\n"
            "import os\n"
            "import sys\n\n"
            "GITHUB_REPO = 'https://github.com/NguyenMinhTri24072005/CT208E_LIDC_DuneAI_MambaVision.git'\n"
            "BRANCH = 'main'  # Nhánh mặc định trên GitHub (đổi nếu bạn dùng nhánh khác)\n"
            "REPO_DIR = '/content/CT208E_LIDC_DuneAI_MambaVision'\n\n"
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
        _make_cell("code", "# 3. Cài đặt các thư viện phụ thuộc\n!pip install -q SimpleITK pandas scipy opencv-python-headless pyyaml matplotlib"),
    ]

    if is_mamba:
        cells.append(_make_cell("code", (
            "# 3b. (Tùy chọn) Cài đặt nhân CUDA mamba-ssm\n"
            "# Lưu ý: Mamba-UNet hỗ trợ tự động Fallback thuần PyTorch nếu không cài được mamba-ssm\n"
            "import torch\n"
            "if torch.cuda.is_available():\n"
            "    try:\n"
            "        print('⚡ Thử cài đặt causal-conv1d và mamba-ssm để tăng tốc GPU...')\n"
            "        !pip install -q causal-conv1d mamba-ssm\n"
            "        print('✅ Cài đặt mamba-ssm thành công!')\n"
            "    except Exception as e:\n"
            "        print('⚠️ Sẽ sử dụng PyTorch selective_scan fallback mặc định:', e)"
        )))

    cells.extend([
        _make_cell("code", "# 4. Kết nối với Google Drive\nfrom google.colab import drive\ndrive.mount('/content/drive')"),
        _make_cell("code", (
            "# 5. Chuẩn bị dữ liệu từ Google Drive vào SSD cục bộ (/content/data/cache_1mm)\n"
            "# Bạn có thể chỉnh sửa NUM_PATIENTS: 0 = giải nén toàn bộ, hoặc nhập số ca (vd: 50, 100)\n"
            "from common.data_pipeline.runtime_data import run_colab\n\n"
            "# Đặt đường dẫn file zip trong Google Drive của bạn (sửa lại theo đúng đường dẫn của bạn)\n"
            "DRIVE_ZIP_PATH = '/content/drive/MyDrive/cache_1mm.zip'  # Hoặc '/content/drive/MyDrive/DuneAI/cache_1mm.zip'\n"
            "NUM_PATIENTS = 0  # 0: Toàn bộ dataset, >0: Chọn N bệnh nhân đầu tiên để test nhanh\n\n"
            "cache_dir = run_colab(\n"
            "    zip_path=DRIVE_ZIP_PATH,\n"
            "    dest_dir='/content/data/cache_1mm',\n"
            "    n_patients=NUM_PATIENTS,\n"
            ")\n"
            "print(f'✅ Dữ liệu sẵn sàng tại: {cache_dir}')"
        )),
        _make_cell("code", f"# 6. Bắt đầu quá trình huấn luyện\n{train_cmd}"),
        _make_cell("code", f"# 7. Đánh giá 3D cấp độ bệnh nhân trên tập Validation\n{eval_cmd}"),
        _make_cell("code", (
            "# 8. Sao lưu kết quả và checkpoint về Google Drive an toàn\n"
            "BACKUP_DIR = '/content/drive/MyDrive/DuneAI_Backups'\n"
            "!mkdir -p {BACKUP_DIR}\n"
            f"!cp -ru {arch}/runs {{BACKUP_DIR}}/ 2>/dev/null || true\n"
            "!cp -ru checkpoints {BACKUP_DIR}/ 2>/dev/null || true\n"
            "print('✅ Đã sao lưu checkpoints và logs vào Google Drive!')"
        )),
    ])
    return _build_notebook(cells)


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
