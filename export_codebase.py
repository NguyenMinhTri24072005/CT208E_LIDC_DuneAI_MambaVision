import os
import json

ROOT_DIR = r"D:\GG_1\COMPUTER SCIENCE\NAM4_HK1_(2026-2027)\CT208E\SOURCES"
OUTPUT_FILE = os.path.join(ROOT_DIR, "codebase_summary.txt")
TARGET_DIRS = ["model", "data"]
EXCLUDE_DIRS = ["__pycache__", ".git", ".ipynb_checkpoints", "logs", "tb_logs"]
ALLOWED_EXTENSIONS = [".py", ".yaml", ".yml"]

def generate_tree(dir_path, prefix=""):
    tree_str = ""
    try:
        entries = os.listdir(dir_path)
    except PermissionError:
        return ""
    
    entries.sort()
    entries = [e for e in entries if e not in EXCLUDE_DIRS]
    
    for i, entry in enumerate(entries):
        path = os.path.join(dir_path, entry)
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        
        # Tránh in ra hàng ngàn thư mục con của data/raw và data/processed/volumes
        if "data\\raw" in path and os.path.isdir(path):
            tree_str += prefix + connector + entry + "/ (omitted internal contents)\n"
            continue
        if "data\\processed" in path and entry == "volumes":
            tree_str += prefix + connector + entry + "/ (omitted internal contents)\n"
            continue
            
        if os.path.isdir(path):
            tree_str += prefix + connector + entry + "/\n"
            extension = "    " if is_last else "│   "
            tree_str += generate_tree(path, prefix + extension)
        else:
            tree_str += prefix + connector + entry + "\n"
    return tree_str

def write_summary():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("PROJECT DIRECTORY STRUCTURE\n")
        f.write("================================================================================\n\n")
        
        for d in TARGET_DIRS:
            target_path = os.path.join(ROOT_DIR, d)
            if os.path.exists(target_path):
                f.write(f"{d}/\n")
                f.write(generate_tree(target_path))
        
        f.write("\n================================================================================\n")
        f.write("SOURCE CODE FILES\n")
        f.write("================================================================================\n\n")
        
        model_dir = os.path.join(ROOT_DIR, "model")
        for root, dirs, files in os.walk(model_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                # Include Python files and configs. Include train.ipynb separately for cell source
                if any(file.endswith(ext) for ext in ALLOWED_EXTENSIONS) or file.endswith(".ipynb"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, ROOT_DIR)
                    f.write(f"\n{'-'*80}\n")
                    f.write(f"FILE: {rel_path}\n")
                    f.write(f"{'-'*80}\n\n")
                    try:
                        with open(file_path, "r", encoding="utf-8") as code_file:
                            content = code_file.read()
                            if file.endswith(".ipynb"):
                                try:
                                    nb = json.loads(content)
                                    f.write("# Extracting Python code cells from Jupyter Notebook:\n\n")
                                    for i, cell in enumerate(nb.get("cells", [])):
                                        if cell.get("cell_type") == "code":
                                            f.write(f"# --- Cell {i+1} ---\n")
                                            f.write("".join(cell.get("source", [])) + "\n\n")
                                except Exception as e:
                                    f.write(f"Error parsing notebook: {e}\n")
                            else:
                                f.write(content + "\n")
                    except Exception as e:
                        f.write(f"Error reading file: {e}\n")

if __name__ == "__main__":
    write_summary()
    print(f"Codebase summary generated successfully at: {OUTPUT_FILE}")
