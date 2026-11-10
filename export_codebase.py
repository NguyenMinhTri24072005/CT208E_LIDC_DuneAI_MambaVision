# -*- coding: utf-8 -*-
"""
export_codebase.py - Xuất toàn bộ cấu trúc thư mục và mã nguồn của dự án DuneAI.

Hỗ trợ kiến trúc mới (v2.1):
  - unet_base/
  - mamba_unet/
  - common/
  - configs/
  - scripts/
  - tests/
  - docs/
Xuất kết quả ra file: codebase_summary.txt
"""

import json
import os
import sys
from pathlib import Path

# Cấu hình UTF-8 cho console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = ROOT_DIR / "codebase_summary.txt"

TARGET_DIRS = [
    "common",
    "configs",
    "unet_base",
    "mamba_unet",
    "scripts",
    "tests",
    "docs",
]

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    ".vscode",
    ".ipynb_checkpoints",
    "runs",
    "checkpoints",
    "golden",
    "logs",
    "tb_logs",
    "wandb",
    "data",
}

EXCLUDE_FILES = {
    "desktop.ini",
    "Thumbs.db",
    ".DS_Store",
}

ALLOWED_EXTENSIONS = {".py", ".yaml", ".yml", ".sh", ".bat", ".md"}
NOTEBOOK_EXTENSIONS = {".ipynb"}


def generate_tree(dir_path: Path, prefix: str = "") -> str:
    tree_str = ""
    try:
        entries = sorted(os.listdir(dir_path))
    except (PermissionError, FileNotFoundError):
        return ""

    entries = [
        e for e in entries
        if e not in EXCLUDE_DIRS
        and e not in EXCLUDE_FILES
        and not e.startswith(".")
    ]

    for i, entry in enumerate(entries):
        path = dir_path / entry
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "

        if path.is_dir():
            tree_str += prefix + connector + entry + "/\n"
            extension = "    " if is_last else "│   "
            tree_str += generate_tree(path, prefix + extension)
        else:
            # Bỏ qua các file nhị phân lớn trong cây nếu có
            if entry.endswith((".pth", ".pt", ".npy", ".zip", ".tar", ".gz")):
                continue
            tree_str += prefix + connector + entry + "\n"
    return tree_str


def write_summary():
    print(f"📦 Đang quét và xuất mã nguồn dự án tại: {ROOT_DIR}")
    exported_files_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 1. CÂY THƯ MỤC DỰ ÁN
        f.write("=" * 80 + "\n")
        f.write("PROJECT DIRECTORY STRUCTURE (DuneAI Dual Architecture v2.1)\n")
        f.write("=" * 80 + "\n\n")

        for d in TARGET_DIRS:
            target_path = ROOT_DIR / d
            if target_path.exists():
                f.write(f"{d}/\n")
                f.write(generate_tree(target_path, prefix="  "))

        # Ghi chú về thư mục dữ liệu và checkpoints
        f.write("\nData & Artifacts Directory Summary:\n")
        if (ROOT_DIR / "data").exists():
            f.write("  ├── data/ (chứa cache_1mm và ảnh CT y tế - bỏ qua chi tiết mảng numpy)\n")
        if (ROOT_DIR / "checkpoints").exists():
            f.write("  └── checkpoints/ (chứa trọng số mô hình đã huấn luyện)\n")

        # 2. TOÀN BỘ NỘI DUNG MÃ NGUỒN
        f.write("\n" + "=" * 80 + "\n")
        f.write("SOURCE CODE & SPECIFICATION FILES\n")
        f.write("=" * 80 + "\n\n")

        # 2.1. Quét các file ở ROOT (README.md, requirements.txt, ...)
        root_files = ["README.md", "requirements.txt", ".gitignore"]
        for rf in root_files:
            rf_path = ROOT_DIR / rf
            if rf_path.exists():
                f.write(f"\n{'-' * 80}\n")
                f.write(f"FILE: {rf}\n")
                f.write(f"{'-' * 80}\n\n")
                try:
                    content = rf_path.read_text(encoding="utf-8")
                    f.write(content + "\n")
                    exported_files_count += 1
                except Exception as e:
                    f.write(f"Error reading {rf}: {e}\n")

        # 2.2. Quét các thư mục mã nguồn chính
        for d in TARGET_DIRS:
            target_path = ROOT_DIR / d
            if not target_path.exists():
                continue

            for root_str, dirs, files in os.walk(target_path):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
                dirs.sort()

                for file in sorted(files):
                    file_path = Path(root_str) / file
                    ext = file_path.suffix.lower()

                    if ext not in ALLOWED_EXTENSIONS and ext not in NOTEBOOK_EXTENSIONS:
                        continue

                    rel_path = file_path.relative_to(ROOT_DIR)
                    f.write(f"\n{'-' * 80}\n")
                    f.write(f"FILE: {rel_path}\n")
                    f.write(f"{'-' * 80}\n\n")

                    try:
                        content = file_path.read_text(encoding="utf-8")
                        if ext in NOTEBOOK_EXTENSIONS:
                            try:
                                nb = json.loads(content)
                                f.write("# [Jupyter Notebook Content Extraction]\n\n")
                                for i, cell in enumerate(nb.get("cells", [])):
                                    c_type = cell.get("cell_type", "unknown")
                                    src = "".join(cell.get("source", []))
                                    if c_type == "code":
                                        f.write(f"# --- [Cell {i+1} - Code] ---\n")
                                        f.write(src.strip() + "\n\n")
                                    elif c_type == "markdown":
                                        f.write(f"# --- [Cell {i+1} - Markdown] ---\n")
                                        for line in src.splitlines():
                                            f.write(f"# {line}\n")
                                        f.write("\n")
                            except Exception as e:
                                f.write(f"Error parsing notebook {rel_path}: {e}\n")
                        else:
                            f.write(content + "\n")

                        exported_files_count += 1
                    except Exception as e:
                        f.write(f"Error reading file {rel_path}: {e}\n")

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"✅ Đã xuất thành công {exported_files_count} files vào: {OUTPUT_FILE}")
    print(f"📊 Dung lượng tệp: {size_mb:.2f} MB")


if __name__ == "__main__":
    write_summary()
