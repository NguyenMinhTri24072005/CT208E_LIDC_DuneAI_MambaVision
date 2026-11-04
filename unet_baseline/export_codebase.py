#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script export toan bo cau truc thu muc va ma nguon cua unet_baseline ra 1 file txt
phuc vu cho viec tuong tac voi cac he thong AI ngoai (ChatGPT, Claude, Gemini, ...).

Cach su dung:
    python export_codebase.py
hoac:
    python export_codebase.py --output codebase_context.txt
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Dam bao terminal Windows khong bi loi UnicodeEncodeError khi print
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Cac thu muc can bo qua khong quet sau ma nguon (chua datasets, weights, logs, cache)
DEFAULT_IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".github",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "checkpoints",          # Chứa file weights (.pth) hàng trăm MB
    "data",                 # Chứa dữ liệu ảnh train/val/test
    "predictions",          # Chứa kết quả segmentation mask/ảnh
    "runs",                 # Chứa logs TensorBoard
    "evaluation_results",   # Chứa thư mục ảnh overlays
}

# Cac file he thong / file rac can bo qua
DEFAULT_IGNORE_FILES = {
    "desktop.ini",
    "Thumbs.db",
    ".DS_Store",
}

# Cac phan mo rong file ma nguon va tai lieu duoc phep trich xuat
ALLOWED_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".txt",
    ".sh",
    ".bat",
    ".toml",
    ".ini",
    ".cfg",
    ".ipynb",
    ".csv",
}

# Cac phan mo rong binary tuyet doi khong doc de tranh tran bo nho
BINARY_EXTENSIONS = {
    ".pth", ".pt", ".onnx", ".bin", ".h5", ".ckpt", ".pkl", ".pickle",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".pyc", ".pyd", ".exe", ".dll", ".so", ".tfevents"
}


def is_binary_file(filepath: Path) -> bool:
    """Kiem tra nhanh xem file co phai binary hay khong."""
    if filepath.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with open(filepath, "tr", encoding="utf-8") as f:
            f.read(1024)
            return False
    except (UnicodeDecodeError, PermissionError):
        return True


def format_size(num_bytes: int) -> str:
    """Dinh dang kich thuoc byte thanh B, KB, MB de doc."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def generate_tree(root_path: Path, output_filename: str, current_script_name: str) -> list:
    """Tao so do cay thu muc dang text dep mat va ro rang."""
    tree_lines = [f"{root_path.name}/"]

    def _walk_tree(dir_path: Path, prefix: str = ""):
        try:
            entries = sorted(
                list(dir_path.iterdir()),
                key=lambda x: (not x.is_dir(), x.name.lower())
            )
        except PermissionError:
            tree_lines.append(f"{prefix}└── [Permission Denied]")
            return

        filtered_entries = []
        for entry in entries:
            if entry.name in DEFAULT_IGNORE_FILES:
                continue
            if entry.name == "__pycache__":
                continue
            filtered_entries.append(entry)

        count = len(filtered_entries)
        for idx, entry in enumerate(filtered_entries):
            is_last = (idx == count - 1)
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")

            if entry.is_dir():
                if entry.name in DEFAULT_IGNORE_DIRS:
                    note = ""
                    if entry.name == "data":
                        note = " [dataset: train/val/test images omitted]"
                    elif entry.name == "checkpoints":
                        note = " [model weights: .pth omitted]"
                    elif entry.name == "runs":
                        note = " [tensorboard event logs omitted]"
                    elif entry.name == "predictions":
                        note = " [predicted visual outputs omitted]"
                    elif entry.name == "evaluation_results":
                        note = " [evaluation results & overlay images omitted]"

                    # Kiem tra neu co thu muc con
                    try:
                        subdirs = [s.name for s in entry.iterdir() if s.is_dir()]
                        if subdirs:
                            sub_info = f" (subfolders: {', '.join(subdirs)})"
                        else:
                            sub_info = ""
                    except Exception:
                        sub_info = ""

                    tree_lines.append(f"{prefix}{connector}{entry.name}/{note}{sub_info}")
                else:
                    tree_lines.append(f"{prefix}{connector}{entry.name}/")
                    _walk_tree(entry, child_prefix)
            else:
                extra = ""
                if entry.name == output_filename:
                    extra = " [output file]"
                elif entry.name == current_script_name:
                    extra = " [export script]"
                tree_lines.append(f"{prefix}{connector}{entry.name}{extra}")

    _walk_tree(root_path)
    return tree_lines


def format_ipynb_cell(cell: dict) -> str:
    """Trich xuat code va markdown tu notebook cell, bo qua hinh anh base64."""
    cell_type = cell.get("cell_type", "")
    source_lines = cell.get("source", [])
    if isinstance(source_lines, list):
        source_text = "".join(source_lines).strip()
    else:
        source_text = str(source_lines).strip()

    if not source_text:
        return ""

    if cell_type == "code":
        return f"# [CODE CELL]\n{source_text}\n"
    elif cell_type == "markdown":
        return f"# [MARKDOWN CELL]\n\"\"\"\n{source_text}\n\"\"\"\n"
    return ""


def read_file_content(filepath: Path, max_size_kb: int = 500) -> tuple:
    """
    Doc noi dung file an toan (UTF-8, xu ly notebook gon gang).
    Tra ve: (content_str, line_count, status_note)
    """
    size_bytes = filepath.stat().st_size
    if size_bytes > max_size_kb * 1024:
        return None, 0, f"[SKIPPED - File size too large ({format_size(size_bytes)} > {max_size_kb} KB)]"

    if is_binary_file(filepath):
        return None, 0, f"[SKIPPED - Binary file ({format_size(size_bytes)})]"

    # Xu ly Jupyter Notebook (.ipynb)
    if filepath.suffix.lower() == ".ipynb":
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                nb_data = json.load(f)
            cells = nb_data.get("cells", [])
            formatted_cells = []
            for idx, c in enumerate(cells, 1):
                cell_repr = format_ipynb_cell(c)
                if cell_repr:
                    formatted_cells.append(f"# --- Cell {idx} ---\n{cell_repr}")
            content = "\n".join(formatted_cells)
            line_count = len(content.splitlines())
            return content, line_count, f"[Clean Jupyter Notebook: {len(cells)} cells extracted]"
        except Exception:
            pass

    # Doc file text thong thuong
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            line_count = len(content.splitlines())
            return content, line_count, "OK"
    except Exception as e:
        return None, 0, f"[ERROR READING FILE: {e}]"


def collect_source_files(root_path: Path, output_filename: str, current_script_name: str, max_size_kb: int = 500) -> list:
    """Thu thap toan bo cac file ma nguon hop le can xuat."""
    collected = []

    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in DEFAULT_IGNORE_DIRS]

        for file in sorted(files):
            if file in DEFAULT_IGNORE_FILES:
                continue
            if file == output_filename:
                continue
            if file == current_script_name:
                continue

            file_path = Path(root) / file
            suffix = file_path.suffix.lower()

            if suffix in BINARY_EXTENSIONS:
                continue

            if suffix not in ALLOWED_EXTENSIONS and suffix != "":
                continue

            collected.append(file_path)

    return sorted(collected, key=lambda p: str(p.relative_to(root_path)).lower())


def export_codebase(root_dir: str = None, output_file: str = "codebase_context.txt", max_size_kb: int = 500):
    """Dieu phoi qua trinh quet va xuat ra file context cho AI."""
    if root_dir is None:
        root_path = Path(__file__).resolve().parent
    else:
        root_path = Path(root_dir).resolve()

    current_script_name = Path(__file__).name
    output_path = root_path / output_file if not Path(output_file).is_absolute() else Path(output_file)

    print("=" * 70)
    print(" SCANNING CODEBASE & STRUCTURE FOR AI CONTEXT")
    print(f" Root Directory: {root_path}")
    print(f" Output File:    {output_path}")
    print("=" * 70)

    # 1. Tao cay thu muc
    print("-> Generating directory tree...")
    tree_lines = generate_tree(root_path, output_path.name, current_script_name)
    tree_text = "\n".join(tree_lines)

    # 2. Thu thap danh sach file ma nguon
    print("-> Discovering source code files...")
    source_files = collect_source_files(root_path, output_path.name, current_script_name, max_size_kb)
    print(f"   Found {len(source_files)} valid source/config/doc files.")

    # 3. Ghi file xuat ra
    print("-> Writing output context file...")
    total_lines = 0
    exported_count = 0

    with open(output_path, "w", encoding="utf-8") as out:
        # Header danh cho AI
        out.write("=" * 80 + "\n")
        out.write("PROJECT CODEBASE CONTEXT & ARCHITECTURE\n")
        out.write(f"Project Name:      {root_path.name}\n")
        out.write(f"Root Directory:    {root_path}\n")
        out.write(f"Generated Date:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Total Code Files:  {len(source_files)}\n")
        out.write("Purpose:           AI Prompt Context (ChatGPT, Claude, Gemini, etc.)\n")
        out.write("=" * 80 + "\n\n")

        # Muc 1: So do cay thu muc
        out.write("=" * 80 + "\n")
        out.write("SECTION 1: DIRECTORY TREE STRUCTURE\n")
        out.write("=" * 80 + "\n\n")
        out.write("```text\n")
        out.write(tree_text + "\n")
        out.write("```\n\n")

        # Muc 2: Danh muc cac file
        out.write("=" * 80 + "\n")
        out.write("SECTION 2: SOURCE CODE INDEX\n")
        out.write("=" * 80 + "\n\n")
        for idx, file_path in enumerate(source_files, 1):
            rel_path = file_path.relative_to(root_path).as_posix()
            size_str = format_size(file_path.stat().st_size)
            out.write(f"{idx:02d}. [{rel_path}] ({size_str})\n")
        out.write("\n")

        # Muc 3: Chi tiet ma nguon tung file kem duong dan
        out.write("=" * 80 + "\n")
        out.write("SECTION 3: DETAILED SOURCE CODE BY FILE\n")
        out.write("=" * 80 + "\n\n")

        for idx, file_path in enumerate(source_files, 1):
            rel_path = file_path.relative_to(root_path).as_posix()
            abs_path = str(file_path.resolve())

            content, lines, note = read_file_content(file_path, max_size_kb)

            out.write("=" * 80 + "\n")
            out.write(f"FILE #{idx:02d}: {rel_path}\n")
            out.write(f"RELATIVE PATH: {rel_path}\n")
            out.write(f"ABSOLUTE PATH: {abs_path}\n")
            out.write(f"FILE SIZE:     {format_size(file_path.stat().st_size)}\n")
            out.write(f"STATUS:        {note}\n")
            if content is not None:
                out.write(f"TOTAL LINES:   {lines}\n")
            out.write("=" * 80 + "\n")

            if content is not None:
                total_lines += lines
                exported_count += 1
                lang = file_path.suffix.lstrip(".").lower()
                if lang == "ipynb":
                    lang = "python"
                elif not lang:
                    lang = "text"

                out.write(f"```{lang}\n")
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
                out.write("```\n\n")
            else:
                out.write(f"\n<<< {note} >>>\n\n")

        # Ket thuc
        out.write("=" * 80 + "\n")
        out.write("END OF CODEBASE CONTEXT\n")
        out.write(f"Total Files Exported: {exported_count}/{len(source_files)}\n")
        out.write(f"Total Lines of Code:  {total_lines:,}\n")
        out.write("=" * 80 + "\n")

    output_size_str = format_size(output_path.stat().st_size)
    print("=" * 70)
    print(" EXPORT COMPLETED SUCCESSFULLY!")
    print(f" - Total files exported: {exported_count} files")
    print(f" - Total lines of code:  {total_lines:,} lines")
    print(f" - Output file size:     {output_size_str}")
    print(f" - Output file path:     {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Export toan bo ma nguon va so do thu muc unet_baseline ra 1 file txt cho AI."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="codebase_context.txt",
        help="Ten file hoac duong dan file xuat ra (mac dinh: codebase_context.txt)"
    )
    parser.add_argument(
        "--root", "-r",
        type=str,
        default=None,
        help="Duong dan thu muc goc (mac dinh: thu muc chua script nay)"
    )
    parser.add_argument(
        "--max-size-kb",
        type=int,
        default=500,
        help="Gioi han dung luong toi da (KB) cho 1 file ma nguon (mac dinh: 500 KB)"
    )

    args = parser.parse_args()
    export_codebase(root_dir=args.root, output_file=args.output, max_size_kb=args.max_size_kb)


if __name__ == "__main__":
    main()
