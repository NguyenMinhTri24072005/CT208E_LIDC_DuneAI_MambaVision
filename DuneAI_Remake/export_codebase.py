import os
import json

# Tự động lấy thư mục gốc của project DuneAI (nơi chứa script này)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(ROOT_DIR, "codebase_summary.txt")

# Các thư mục và tệp cần bỏ qua khi quét
EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "logs", "tb_logs", ".gemini", ".system_generated"}
EXCLUDE_FILES = {"codebase_summary.txt", "export_codebase.py", "desktop.ini"}

# Các định dạng mã nguồn trong DuneAI (Python, R, Jupyter Notebook)
ALLOWED_EXTENSIONS = {".py", ".r", ".R", ".ipynb"}

def generate_tree(dir_path, prefix=""):
    """Tạo cây thư mục trực quan của toàn bộ project."""
    tree_str = ""
    try:
        entries = os.listdir(dir_path)
    except PermissionError:
        return ""
    
    entries = [e for e in entries if e not in EXCLUDE_DIRS and e not in EXCLUDE_FILES]
    entries.sort(key=lambda s: (not os.path.isdir(os.path.join(dir_path, s)), s.lower()))
    
    for i, entry in enumerate(entries):
        path = os.path.join(dir_path, entry)
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        
        if os.path.isdir(path):
            tree_str += prefix + connector + entry + "/\n"
            extension = "    " if is_last else "│   "
            tree_str += generate_tree(path, prefix + extension)
        else:
            tree_str += prefix + connector + entry + "\n"
    return tree_str

def write_summary():
    """Trích xuất cây thư mục và toàn bộ mã nguồn của DuneAI vào file codebase_summary.txt."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DUNEAI - PROJECT DIRECTORY STRUCTURE\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(os.path.basename(ROOT_DIR) + "/\n")
        f.write(generate_tree(ROOT_DIR))
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("SOURCE CODE FILES (Python, R, Jupyter Notebooks)\n")
        f.write("=" * 80 + "\n\n")
        
        # Thu thập toàn bộ file mã nguồn
        code_files = []
        for root, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                if file in EXCLUDE_FILES:
                    continue
                ext = os.path.splitext(file)[1]
                if ext.lower() in ALLOWED_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, ROOT_DIR)
                    code_files.append((rel_path, full_path, ext.lower()))
        
        # Sắp xếp danh sách file theo đường dẫn
        code_files.sort(key=lambda x: x[0].lower())
        
        for rel_path, full_path, ext in code_files:
            f.write(f"\n{'-'*80}\n")
            f.write(f"FILE: {rel_path}\n")
            f.write(f"{'-'*80}\n\n")
            
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as code_file:
                    content = code_file.read()
                    
                    if ext == ".ipynb":
                        try:
                            nb = json.loads(content)
                            f.write("# Extracting code and descriptions from Jupyter Notebook:\n\n")
                            for idx, cell in enumerate(nb.get("cells", [])):
                                cell_type = cell.get("cell_type")
                                source_lines = "".join(cell.get("source", [])).strip()
                                
                                if not source_lines:
                                    continue
                                    
                                if cell_type == "markdown":
                                    f.write(f"# [Markdown Description - Section {idx+1}]:\n")
                                    for line in source_lines.splitlines():
                                        f.write(f"# {line}\n")
                                    f.write("\n")
                                elif cell_type == "code":
                                    f.write(f"# --- Code Cell {idx+1} ---\n")
                                    f.write(source_lines + "\n\n")
                        except Exception as e:
                            f.write(f"# Error parsing notebook {rel_path}: {e}\n")
                    else:
                        f.write(content.strip() + "\n")
            except Exception as e:
                f.write(f"# Error reading file {rel_path}: {e}\n")

if __name__ == "__main__":
    write_summary()
    print(f"Codebase summary generated successfully at: {OUTPUT_FILE}")

