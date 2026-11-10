# -*- coding: utf-8 -*-
"""
tests/run_all.py - Bộ chạy toàn bộ kiểm thử nghiệm thu (Phase 7 - Refactor Spec v2.1).

Chạy tất cả các bài kiểm tra theo mục 10 của REFACTOR_SPEC.md:
  1. Selftest metric dùng chung (common.metrics)
  2. Selftest khối CNN unet_base (unet_base.blocks)
  3. Selftest khối CNN mamba_unet (mamba_unet.blocks)
  4. Selftest registry kích hoạt unet_base (unet_base.activations)
  5. Selftest registry loss unet_base (unet_base.losses)
  6. Selftest registry kích hoạt mamba_unet (mamba_unet.activations)
  7. Selftest registry loss mamba_unet (mamba_unet.losses)
  8. Selftest medical_prep (common.data_pipeline.medical_prep)
  9. Selftest offline_builder (common.data_pipeline.offline_builder)
 10. Selftest dataset (common.data_pipeline.dataset)
 11. Selftest runtime_data (common.data_pipeline.runtime_data)
 12. Selftest train_main unet_base (unet_base.train_main)
 13. Selftest evaluate unet_base (unet_base.evaluate)
 14. Selftest train_main mamba_unet (mamba_unet.train_main)
 15. Selftest evaluate mamba_unet (mamba_unet.evaluate)
 16. Unit test test_independence (tests/test_independence.py)
 17. Unit test test_state_dict_invariance (tests/test_state_dict_invariance.py)
 18. Unit test test_scan_fallback (tests/test_scan_fallback.py)
 19. Unit test test_blocks (tests/test_blocks.py)
 20. Unit test test_registries (tests/test_registries.py)
 21. Unit test test_run_folder (tests/test_run_folder.py)
 22. Unit test test_resume (tests/test_resume.py)
 23. Unit test test_eval3d (tests/test_eval3d.py)
 24. Unit test test_notebooks (tests/test_notebooks.py)
 25. Kiểm tra tính hợp lệ notebook (scripts/make_notebooks.py --check)
 26. Quét tĩnh sự sạch sẽ & chuỗi cấm (Cleanliness & Static Audit)

Xuất bảng kết quả và trả về mã thoát 0 nếu toàn bộ PASS, 1 nếu có bài FAIL.
"""

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

# Cấu hình UTF-8 cho console stdout/stderr trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]


def run_cleanliness_audit():
    """Kiểm tra tĩnh tính sạch sẽ toàn dự án:

    - Không có activations/ hoặc losses/ ở ROOT
    - Không có common/nn_blocks.py
    - Không có lớp con nn.Module trong common/
    - Không có DuneAI_Remake ở ROOT
    - Không có chuỗi cấm trong mã nguồn
    """
    errors = []

    # 1. ROOT directory checks
    if (ROOT / "activations").exists():
        errors.append("Tồn tại thư mục cấm 'activations/' ở ROOT")
    if (ROOT / "losses").exists():
        errors.append("Tồn tại thư mục cấm 'losses/' ở ROOT")
    if (ROOT / "DuneAI_Remake").exists():
        errors.append("Tồn tại thư mục cấm 'DuneAI_Remake/' ở ROOT")
    if (ROOT / "common" / "nn_blocks.py").exists():
        errors.append("Tồn tại file cấm 'common/nn_blocks.py'")

    # 2. common/ contains 0 nn.Module subclasses
    common_dir = ROOT / "common"
    for py_path in common_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        id_name = getattr(base, "id", None) or getattr(base, "attr", None)
                        if id_name in ("Module", "nn.Module"):
                            errors.append(f"Lớp {node.name} trong {py_path.name} kế thừa nn.Module")
        except Exception as e:
            errors.append(f"Lỗi parse AST {py_path}: {e}")

    # 3. Forbidden string patterns in active source code
    patterns = [
        "DuneAI_Remake",
        "from src.",
        "src/",
        "from activations",
        "from losses",
        "import activations",
        "import losses",
    ]
    ignore_dirs = {".git", ".gemini", "__pycache__", "golden", "checkpoints", "runs", "node_modules", "data", "files"}
    for root_str, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
        parts = Path(root_str).relative_to(ROOT).parts
        if any(p in ignore_dirs or p.startswith(".") for p in parts):
            continue
        # Skip docs and refactor specs
        if parts and parts[0] in ("docs", "Refactor_spec"):
            continue

        for f in files:
            if not f.endswith((".py", ".yaml", ".yml", ".json", ".sh", ".bat")):
                continue
            # Skip test files that deliberately mention strings for test assertions
            if f in ("test_independence.py", "run_all.py", "scan_cleanliness.py"):
                continue
            fp = Path(root_str) / f
            try:
                content = fp.read_text(encoding="utf-8")
                for pat in patterns:
                    if pat in content:
                        errors.append(f"Phát hiện chuỗi cấm '{pat}' trong {fp.relative_to(ROOT)}")
            except Exception as e:
                pass

    if errors:
        return False, "\n".join(errors)
    return True, "All structural & cleanliness criteria passed."


def main():
    print("=" * 90)
    print(" DUNEAI REFACTOR SPEC v2.1 — TOÀN BỘ KIỂM THỬ NGHIỆM THU (PHASE 7)")
    print("=" * 90)

    py = sys.executable

    test_specs = [
        ("01. Metrics dùng chung", [py, "-m", "common.metrics"]),
        ("02. Blocks unet_base", [py, "-m", "unet_base.blocks"]),
        ("03. Blocks mamba_unet", [py, "-m", "mamba_unet.blocks"]),
        ("04. Activations unet_base", [py, "-m", "unet_base.activations"]),
        ("05. Losses unet_base", [py, "-m", "unet_base.losses"]),
        ("06. Activations mamba_unet", [py, "-m", "mamba_unet.activations"]),
        ("07. Losses mamba_unet", [py, "-m", "mamba_unet.losses"]),
        ("08. Data medical_prep", [py, "-m", "common.data_pipeline.medical_prep", "selftest"]),
        ("09. Data offline_builder", [py, "-m", "common.data_pipeline.offline_builder", "selftest"]),
        ("10. Data dataset", [py, "-m", "common.data_pipeline.dataset", "selftest"]),
        ("11. Data runtime_data", [py, "-m", "common.data_pipeline.runtime_data", "selftest"]),
        ("12. Train unet_base", [py, "-m", "unet_base.train_main", "selftest"]),
        ("13. Eval unet_base", [py, "-m", "unet_base.evaluate", "selftest"]),
        ("14. Train mamba_unet", [py, "-m", "mamba_unet.train_main", "selftest"]),
        ("15. Eval mamba_unet", [py, "-m", "mamba_unet.evaluate", "selftest"]),
        ("16. Test independence", [py, "tests/test_independence.py"]),
        ("17. Test invariance", [py, "tests/test_state_dict_invariance.py"]),
        ("18. Test scan fallback", [py, "tests/test_scan_fallback.py"]),
        ("19. Test blocks", [py, "tests/test_blocks.py"]),
        ("20. Test registries", [py, "tests/test_registries.py"]),
        ("21. Test run folder", [py, "tests/test_run_folder.py"]),
        ("22. Test resume", [py, "tests/test_resume.py"]),
        ("23. Test eval3d", [py, "tests/test_eval3d.py"]),
        ("24. Test notebooks", [py, "tests/test_notebooks.py"]),
        ("25. Check notebooks nbformat", [py, "scripts/make_notebooks.py", "--check"]),
    ]

    results = []
    overall_start = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"

    for idx, (name, cmd) in enumerate(test_specs, 1):
        cmd_str = " ".join(cmd[1:])
        print(f"[{idx:02d}/{len(test_specs)+1:02d}] Đang chạy: {cmd_str} ...", end="", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        elapsed = time.perf_counter() - t0
        passed = (proc.returncode == 0)
        status_str = "PASS" if passed else "FAIL"
        print(f" [{status_str}] ({elapsed:.2f}s)")
        results.append({
            "num": idx,
            "name": name,
            "cmd": cmd_str,
            "status": status_str,
            "duration": elapsed,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        })

    # Test 26: Cleanliness & static audit
    idx = len(test_specs) + 1
    print(f"[{idx:02d}/{len(test_specs)+1:02d}] Đang chạy: Cleanliness & Static Isolation Audit ...", end="", flush=True)
    t0 = time.perf_counter()
    passed, audit_msg = run_cleanliness_audit()
    elapsed = time.perf_counter() - t0
    status_str = "PASS" if passed else "FAIL"
    print(f" [{status_str}] ({elapsed:.2f}s)")
    results.append({
        "num": idx,
        "name": "26. Cleanliness & Static Audit",
        "cmd": "internal AST & structure audit",
        "status": status_str,
        "duration": elapsed,
        "stdout": audit_msg if passed else "",
        "stderr": audit_msg if not passed else "",
    })

    total_time = time.perf_counter() - overall_start

    # In bảng tổng kết
    print("\n" + "=" * 90)
    print(f"{'#':<3} | {'Tên Kiểm Thử':<30} | {'Lệnh':<36} | {'Kết quả':<7} | {'Thời gian':<8}")
    print("-" * 90)
    all_passed = True
    for r in results:
        status_display = r["status"]
        if r["status"] != "PASS":
            all_passed = False
        print(f"{r['num']:02d}  | {r['name']:<30} | {r['cmd']:<36} | {status_display:<7} | {r['duration']:>6.2f}s")
    print("=" * 90)

    failed_runs = [r for r in results if r["status"] != "PASS"]
    if failed_runs:
        print(f"\n❌ CÓ {len(failed_runs)} BÀI KIỂM THỬ THẤT BẠI:")
        for fr in failed_runs:
            print(f"\n--- [FAIL] {fr['name']} ({fr['cmd']}) ---")
            if fr["stderr"]:
                print(f"STDERR:\n{fr['stderr'].strip()}")
            if fr["stdout"]:
                print(f"STDOUT:\n{fr['stdout'].strip()[-1000:]}")
        print(f"\nTổng thời gian: {total_time:.2f}s. KẾT QUẢ: THẤT BẠI.")
        sys.exit(1)
    else:
        print(f"\n✅ TOÀN BỘ {len(results)}/{len(results)} BÀI KIỂM THỬ ĐỀU PASS HOÀN TOÀN!")
        print(f"Tổng thời gian thực thi: {total_time:.2f}s.")
        sys.exit(0)


if __name__ == "__main__":
    main()
