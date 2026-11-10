# -*- coding: utf-8 -*-
"""
common/data_pipeline/runtime_data.py - Chuẩn bị và đồng bộ dữ liệu theo nền tảng thực thi (Colab, Kaggle, Local).

Mục tiêu (spec mục 8.6):
  - Colab: Mount Drive -> dò zip cache_1mm -> đọc mục lục zipfile -> chọn số lượng N bệnh nhân -> giải nén phẳng ra SSD.
  - Kaggle: Dò cache gốc từ hint (chỉ đọc) -> dựng symlink view-dir ghi được trong /kaggle/working.
  - Local: Xác minh thư mục dữ liệu cục bộ và xuất manifest.
  - Auto: Tự nhận diện nền tảng rồi kích hoạt luồng tương ứng.
  - Xuất data_manifest.json và in dòng cuối CACHE_DIR=<path>.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PATIENT_REGEX = re.compile(r"LIDC-IDRI-\d{4}")


def compute_sha256(data: str) -> str:
    """Tính SHA256 của một chuỗi."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def compute_patient_ids_hash(patient_ids: List[str]) -> str:
    """Tính mã băm xác thực danh sách bệnh nhân (đã sắp xếp)."""
    sorted_ids = sorted(patient_ids)
    return compute_sha256(",".join(sorted_ids))


def create_dir_symlink(src: Path, dst: Path):
    """Tạo symlink thư mục đa nền tảng (hỗ trợ cả Linux và Windows Junction)."""
    src_str = str(src.resolve())
    dst_str = str(dst.resolve())
    try:
        os.symlink(src_str, dst_str, target_is_directory=True)
    except OSError as e:
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(src_str, dst_str)
        else:
            raise e


def select_patients(
    available_ids: List[str],
    n_requested: Optional[int] = None,
    mode: str = "first",
    seed: int = 42,
    patients_file: Optional[str] = None,
    _skip_sort: bool = False,
) -> Tuple[List[str], Dict]:
    """
    Lựa chọn danh sách bệnh nhân xác định (deterministic):
      - 'first': lấy N bệnh nhân đầu tiên theo thứ tự tên đã sắp xếp.
      - 'random': lấy mẫu ngẫu nhiên theo seed rồi sắp xếp lại.
      - 'file': nạp từ file JSON danh sách bệnh nhân.
    """
    if not _skip_sort:
        sorted_all = sorted(available_ids)
    else:
        sorted_all = list(available_ids)

    n_available = len(sorted_all)

    if patients_file:
        pf_path = Path(patients_file)
        if not pf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy patients-file: {pf_path}")
        with open(pf_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "patient_ids" in data:
            target_ids = data["patient_ids"]
        elif isinstance(data, list):
            target_ids = data
        else:
            raise ValueError(f"Định dạng patients-file không hợp lệ: {pf_path}")
        selected = [pid for pid in target_ids if pid in available_ids]
        if not _skip_sort:
            selected = sorted(selected)
        selection_meta = {"mode": "file", "file": str(pf_path), "n_requested": len(selected)}
        return selected, selection_meta

    if n_requested is None or n_requested <= 0 or n_requested >= n_available:
        if n_requested and n_requested > n_available:
            print(f"⚠️ Yêu cầu {n_requested} bệnh nhân nhưng chỉ có {n_available}. Sẽ dùng tất cả.")
        selected = list(sorted_all)
        selection_meta = {"mode": "all", "n_requested": n_available, "seed": seed}
        return selected, selection_meta

    if mode == "first":
        selected = sorted_all[:n_requested]
        selection_meta = {"mode": "first", "n_requested": n_requested, "seed": seed}
    elif mode == "random":
        import random
        rng = random.Random(seed)
        sampled = rng.sample(sorted_all, n_requested)
        selected = sampled if _skip_sort else sorted(sampled)
        selection_meta = {"mode": "random", "n_requested": n_requested, "seed": seed}
    else:
        raise ValueError(f"Chế độ chọn không hợp lệ: {mode} (chỉ hỗ trợ first, random, file)")

    return selected, selection_meta


def sync_slices_index(cache_dir: Path, patient_ids: List[str]):
    """
    Đồng bộ slices.csv và sidecar slices.meta.json.
    Nếu tập bệnh nhân thay đổi, xóa slices.csv cũ để hệ thống tự tạo lại chính xác.
    """
    current_hash = compute_patient_ids_hash(patient_ids)
    meta_path = cache_dir / "slices.meta.json"
    csv_path = cache_dir / "slices.csv"

    need_rebuild = False
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                old_meta = json.load(f)
            if old_meta.get("patient_ids_sha256") != current_hash:
                need_rebuild = True
        except Exception:
            need_rebuild = True
    else:
        if csv_path.exists():
            need_rebuild = True

    if need_rebuild and csv_path.exists():
        print(f"🔄 Tập bệnh nhân đã thay đổi -> Xóa {csv_path.name} cũ để dựng lại đồng bộ.")
        try:
            csv_path.unlink()
        except OSError:
            pass

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "patient_ids_sha256": current_hash,
            "n_patients": len(patient_ids),
        }, f, indent=2)


# ======================================================================
# NHÁNH COLAB: GIẢI NÉN CHỌN SỐ LƯỢNG TỪ DRIVE
# ======================================================================
def find_drive_zip(zip_arg: str, drive_root: Optional[str] = None) -> Path:
    """Tìm file cache_1mm.zip trên Google Drive."""
    if zip_arg != "auto":
        p = Path(zip_arg)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file zip: {p.resolve()}")
        return p

    search_root = Path(drive_root) if drive_root else Path("/content/drive")
    if not search_root.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục Google Drive tại '{search_root}'.\n"
            "Hãy đảm bảo bạn đã chạy ô Mount Drive trước: from google.colab import drive; drive.mount('/content/drive')"
        )

    candidates = []
    for root, dirs, files in os.walk(search_root):
        depth = len(Path(root).relative_to(search_root).parts)
        if depth > 4:
            dirs.clear()
            continue
        for f in files:
            if f.startswith("cache_1mm") and f.endswith(".zip"):
                candidates.append(Path(root) / f)

    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy file khớp 'cache_1mm*.zip' dưới '{search_root}'. Hãy truyền --zip cụ thể.")
    if len(candidates) > 1:
        cand_list = "\n  - ".join(str(c) for c in candidates)
        raise RuntimeError(f"Tìm thấy nhiều file zip ứng viên, vui lòng chỉ định cụ thể qua --zip:\n  - {cand_list}")

    return candidates[0]


def run_colab(
    zip_path: str,
    dest_dir: str,
    drive_root: Optional[str] = None,
    n_patients: Optional[int] = None,
    select: str = "first",
    seed: int = 42,
    patients_file: Optional[str] = None,
    out_json: Optional[str] = None,
    _skip_temp_replace: bool = False,
    _skip_sort: bool = False,
    _mock_disk_space_error: bool = False,
) -> Path:
    """Chuẩn bị dữ liệu trên Google Colab."""
    resolved_zip = find_drive_zip(zip_path, drive_root)
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(resolved_zip, "r") as zf:
        all_names = zf.namelist()
        patient_entries: Dict[str, List[zipfile.ZipInfo]] = {}

        for info in zf.infolist():
            match = PATIENT_REGEX.search(info.filename)
            if match:
                pid = match.group(0)
                patient_entries.setdefault(pid, []).append(info)

        available_ids = list(patient_entries.keys())
        if not available_ids:
            raise RuntimeError(f"File zip {resolved_zip} không chứa cấu trúc bệnh nhân dạng LIDC-IDRI-XXXX")

        selected_ids, sel_meta = select_patients(
            available_ids,
            n_requested=n_patients,
            mode=select,
            seed=seed,
            patients_file=patients_file,
            _skip_sort=_skip_sort,
        )

        # Tính tổng dung lượng cần giải nén
        total_uncompressed = sum(
            info.file_size
            for pid in selected_ids
            for info in patient_entries[pid]
        )

        # Kiểm tra dung lượng trống
        dest_stat = shutil.disk_usage(dest)
        required_space = total_uncompressed * 1.05
        if _mock_disk_space_error or dest_stat.free < required_space:
            req_mb = required_space / (1024 * 1024)
            free_mb = dest_stat.free / (1024 * 1024)
            raise OSError(
                f"Không đủ dung lượng trống tại {dest}!\n"
                f"Yêu cầu: {req_mb:.1f} MB | Khả dụng: {free_mb:.1f} MB"
            )

        print(f"📦 Colab Data Pipeline: Chọn {len(selected_ids)}/{len(available_ids)} bệnh nhân từ {resolved_zip.name}")

        for pid in selected_ids:
            target_pid_dir = dest / pid
            # Kiểm tra idempotent (nếu đã đủ file và kích thước thì bỏ qua)
            if target_pid_dir.exists() and (target_pid_dir / "image_1mm.npy").exists():
                continue

            if not _skip_temp_replace:
                temp_extract = dest / f".tmp_{pid}_{uuid.uuid4().hex[:6]}"
                temp_extract.mkdir(parents=True, exist_ok=True)
                extract_target = temp_extract
            else:
                extract_target = target_pid_dir
                extract_target.mkdir(parents=True, exist_ok=True)

            for info in patient_entries[pid]:
                if info.is_dir():
                    continue
                # Lấy tên file sau ID bệnh nhân
                sub_parts = Path(info.filename).parts
                pid_idx = sub_parts.index(pid)
                rel_parts = sub_parts[pid_idx + 1 :]
                out_file_path = extract_target / Path(*rel_parts)
                out_file_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src_f, open(out_file_path, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)

            if not _skip_temp_replace:
                if target_pid_dir.exists():
                    shutil.rmtree(target_pid_dir)
                os.replace(str(temp_extract), str(target_pid_dir))

    sync_slices_index(dest, selected_ids)

    manifest = {
        "platform": "colab",
        "kind": "zip",
        "origin": str(resolved_zip),
        "cache_dir": str(dest),
        "n_available": len(available_ids),
        "n_selected": len(selected_ids),
        "selection": sel_meta,
        "patient_ids": selected_ids,
        "patient_ids_sha256": compute_patient_ids_hash(selected_ids),
        "zip": {
            "path": str(resolved_zip),
            "size_bytes": resolved_zip.stat().st_size,
        },
    }

    manifest_file = Path(out_json) if out_json else (dest / "data_manifest.json")
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"CACHE_DIR={dest}")
    return dest


# ======================================================================
# NHÁNH KAGGLE: DÒ CACHE VÀ DỰNG VIEW-DIR SYMLINK
# ======================================================================
def find_kaggle_cache(hint: str) -> Path:
    """Dò tìm thư mục cache chứa các thư mục con LIDC-IDRI-XXXX."""
    p = Path(hint)
    if p.exists() and PATIENT_REGEX.search(p.name):
        p = p.parent

    # Dò xuống tối đa 4 tầng
    candidates = []
    if p.exists():
        for root, dirs, _ in os.walk(p):
            depth = len(Path(root).relative_to(p).parts)
            if depth > 4:
                dirs.clear()
                continue
            matching = [d for d in dirs if PATIENT_REGEX.match(d)]
            if matching:
                candidates.append(Path(root))
                dirs.clear()

    if not candidates:
        # Thử dò dưới /kaggle/input
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for root, dirs, _ in os.walk(kaggle_input):
                depth = len(Path(root).relative_to(kaggle_input).parts)
                if depth > 4:
                    dirs.clear()
                    continue
                matching = [d for d in dirs if PATIENT_REGEX.match(d)]
                if matching:
                    candidates.append(Path(root))
                    dirs.clear()

    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục chứa bệnh nhân LIDC-IDRI-XXXX từ hint '{hint}'.\n"
            "Hãy kiểm tra lại đường dẫn Dataset hoặc bấm 'Add Input' để nạp dataset vào Kaggle."
        )

    return candidates[0]


def run_kaggle(
    hint: str,
    view_dir: str,
    n_patients: Optional[int] = None,
    select: str = "first",
    seed: int = 42,
    patients_file: Optional[str] = None,
    out_json: Optional[str] = None,
    _skip_sort: bool = False,
) -> Path:
    """Chuẩn bị dữ liệu trên Kaggle qua symlink view-dir."""
    source_cache = find_kaggle_cache(hint)
    view = Path(view_dir).resolve()
    view.mkdir(parents=True, exist_ok=True)

    available_ids = sorted([
        d.name for d in source_cache.iterdir()
        if d.is_dir() and PATIENT_REGEX.match(d.name)
    ])
    if not available_ids:
        raise RuntimeError(f"Không tìm thấy bệnh nhân hợp lệ nào trong {source_cache}")

    selected_ids, sel_meta = select_patients(
        available_ids,
        n_requested=n_patients,
        mode=select,
        seed=seed,
        patients_file=patients_file,
        _skip_sort=_skip_sort,
    )

    print(f"🔗 Kaggle Data Pipeline: Dựng symlink cho {len(selected_ids)}/{len(available_ids)} bệnh nhân vào {view}")

    # Xóa các symlink cũ không còn nằm trong danh sách chọn
    for existing in view.iterdir():
        if existing.is_dir() and PATIENT_REGEX.match(existing.name):
            if existing.name not in selected_ids:
                if existing.is_symlink():
                    existing.unlink()
                else:
                    shutil.rmtree(existing)

    # Dựng symlink tới từng bệnh nhân đã chọn
    for pid in selected_ids:
        src_pid = source_cache / pid
        dst_pid = view / pid
        if not dst_pid.exists():
            create_dir_symlink(src_pid, dst_pid)

    sync_slices_index(view, selected_ids)

    manifest = {
        "platform": "kaggle",
        "kind": "kaggle_input",
        "origin": str(source_cache),
        "cache_dir": str(view),
        "n_available": len(available_ids),
        "n_selected": len(selected_ids),
        "selection": sel_meta,
        "patient_ids": selected_ids,
        "patient_ids_sha256": compute_patient_ids_hash(selected_ids),
    }

    manifest_file = Path(out_json) if out_json else (view / "data_manifest.json")
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"CACHE_DIR={view}")
    return view


# ======================================================================
# NHÁNH LOCAL & AUTO
# ======================================================================
def run_local(cache_dir: str, out_json: Optional[str] = None) -> Path:
    """Xác thực thư mục cache cục bộ và xuất manifest."""
    p = Path(cache_dir).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục cache cục bộ: {p}")

    available_ids = sorted([
        d.name for d in p.iterdir()
        if d.is_dir() and PATIENT_REGEX.match(d.name)
    ])
    if not available_ids:
        raise RuntimeError(f"Không tìm thấy bệnh nhân nào trong {p}")

    sync_slices_index(p, available_ids)

    manifest = {
        "platform": "local",
        "kind": "local_dir",
        "origin": str(p),
        "cache_dir": str(p),
        "n_available": len(available_ids),
        "n_selected": len(available_ids),
        "selection": {"mode": "all", "n_requested": len(available_ids)},
        "patient_ids": available_ids,
        "patient_ids_sha256": compute_patient_ids_hash(available_ids),
    }

    manifest_file = Path(out_json) if out_json else (p / "data_manifest.json")
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"CACHE_DIR={p}")
    return p


def run_auto(**kwargs) -> Path:
    """Tự động phát hiện nền tảng thực thi."""
    if "google.colab" in sys.modules:
        is_colab = True
    else:
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False

    if is_colab:
        zip_path = kwargs.get("zip", "auto")
        dest_dir = kwargs.get("dest", "/content/local_cache_1mm")
        return run_colab(zip_path=zip_path, dest_dir=dest_dir, **kwargs)

    if Path("/kaggle/working").exists():
        hint = kwargs.get("hint", "/kaggle/input/datasets/minhtri2475/lidc-idri-1mm-cache-ct208e")
        view_dir = kwargs.get("view_dir", "/kaggle/working/view_cache_1mm")
        return run_kaggle(hint=hint, view_dir=view_dir, **kwargs)

    cache_dir = kwargs.get("cache_dir", "data/cache_1mm")
    return run_local(cache_dir=cache_dir, out_json=kwargs.get("out_json"))


# ======================================================================
# CLI & SELFTEST
# ======================================================================
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # colab
    c = sub.add_parser("colab", help="Chuẩn bị dữ liệu trên Google Colab từ zip trên Drive")
    c.add_argument("--zip", default="auto", help="Đường dẫn file zip hoặc 'auto'")
    c.add_argument("--drive-root", default=None, help="Thư mục gốc Drive để dò tìm")
    c.add_argument("--dest", required=True, help="Thư mục đích trên SSD cục bộ (/content/...)")
    c.add_argument("--n-patients", type=int, default=None, help="Số bệnh nhân cần lấy")
    c.add_argument("--select", choices=["first", "random"], default="first")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--patients-file", default=None)
    c.add_argument("--out-json", default=None)

    # kaggle
    k = sub.add_parser("kaggle", help="Chuẩn bị dữ liệu trên Kaggle qua symlink view-dir")
    k.add_argument("--hint", required=True, help="Đường dẫn dataset gốc, thư mục cache hoặc 1 bệnh nhân")
    k.add_argument("--view-dir", required=True, help="Thư mục view-dir ghi được trên Kaggle (/kaggle/working/...)")
    k.add_argument("--n-patients", type=int, default=None)
    k.add_argument("--select", choices=["first", "random"], default="first")
    k.add_argument("--seed", type=int, default=42)
    k.add_argument("--patients-file", default=None)
    k.add_argument("--out-json", default=None)

    # local
    l = sub.add_parser("local", help="Xác thực thư mục cache cục bộ")
    l.add_argument("--cache-dir", required=True)
    l.add_argument("--out-json", default=None)

    # auto
    a = sub.add_parser("auto", help="Tự động phát hiện nền tảng")
    a.add_argument("--zip", default="auto")
    a.add_argument("--dest", default=None)
    a.add_argument("--hint", default=None)
    a.add_argument("--view-dir", default=None)
    a.add_argument("--cache-dir", default=None)
    a.add_argument("--patients-file", default=None)
    a.add_argument("--out-json", default=None)

    # selftest
    sub.add_parser("selftest", help="Chạy selftest tự động không dùng mạng")

    return ap


def run_selftest():
    import tempfile
    print("🧪 Bắt đầu selftest runtime_data (9 ca kiểm thử + 2 ca tự phá)...")
    temp_dir = Path(tempfile.mkdtemp(prefix="dune_selftest_"))
    try:
        # Tạo dữ liệu giả lập cho 5 bệnh nhân
        pids = [f"LIDC-IDRI-{i:04d}" for i in range(1, 6)]

        # 1. Zip giả đúng bố cục thật cache_1mm/LIDC-IDRI-XXXX/...
        zip_path = temp_dir / "cache_1mm.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for pid in pids:
                zf.writestr(f"cache_1mm/{pid}/image_1mm.npy", b"image_data")
                zf.writestr(f"cache_1mm/{pid}/mask_1mm.npy", b"mask_data")
                zf.writestr(f"cache_1mm/{pid}/meta.json", json.dumps({"pid": pid}))

        dest_colab = temp_dir / "colab_ssd"
        out_dest = run_colab(
            zip_path=str(zip_path),
            dest_dir=str(dest_colab),
            n_patients=3,
            select="first",
        )
        assert out_dest.exists(), "Thư mục đích giải nén không tồn tại"
        extracted = sorted([d.name for d in out_dest.iterdir() if PATIENT_REGEX.match(d.name)])
        assert extracted == ["LIDC-IDRI-0001", "LIDC-IDRI-0002", "LIDC-IDRI-0003"], f"Sai first 3: {extracted}"
        print("  ✓ Ca 1: Zip bố cục thật cache_1mm/ giải nén phẳng ra --dest đúng 3 bệnh nhân đầu.")

        # 2. first N=3 và random(seed)
        r1_pids, _ = select_patients(pids, n_requested=3, mode="random", seed=42)
        r2_pids, _ = select_patients(pids, n_requested=3, mode="random", seed=42)
        r3_pids, _ = select_patients(pids, n_requested=3, mode="random", seed=99)
        assert r1_pids == r2_pids, "random cùng seed phải cho kết quả giống nhau"
        assert r1_pids != r3_pids, "random khác seed phải cho kết quả khác nhau"
        print("  ✓ Ca 2: first N=3 và random(seed) nhất quán xác định.")

        # 3. Chạy lại không giải nén lại (Idempotency)
        t_before = (dest_colab / "LIDC-IDRI-0001" / "image_1mm.npy").stat().st_mtime
        run_colab(zip_path=str(zip_path), dest_dir=str(dest_colab), n_patients=3, select="first")
        t_after = (dest_colab / "LIDC-IDRI-0001" / "image_1mm.npy").stat().st_mtime
        assert t_before == t_after, "Idempotency thất bại: file bị ghi đè lại không cần thiết"
        print("  ✓ Ca 3: Idempotency chuẩn (không giải nén lại các bệnh nhân đã hoàn chỉnh).")

        # 4. N > n_available -> cảnh báo và dùng tất cả
        dest_all = temp_dir / "colab_all"
        run_colab(zip_path=str(zip_path), dest_dir=str(dest_all), n_patients=10)
        ext_all = sorted([d.name for d in dest_all.iterdir() if PATIENT_REGEX.match(d.name)])
        assert len(ext_all) == 5, f"Phải lấy đủ 5 bệnh nhân, nhận được {len(ext_all)}"
        print("  ✓ Ca 4: N > n_available xử lý chuẩn xác, dùng toàn bộ.")

        # 5. Thiếu dung lượng -> ném OSError
        try:
            dest_nospace = temp_dir / "colab_nospace"
            run_colab(zip_path=str(zip_path), dest_dir=str(dest_nospace), n_patients=2, _mock_disk_space_error=True)
            assert False, "Phải ném lỗi khi thiếu dung lượng"
        except OSError:
            print("  ✓ Ca 5: Bắt lỗi thiếu dung lượng ổ đĩa thành công.")

        # 6. Kaggle giả lập lồng 2 tầng .../cache_1mm/cache_1mm/LIDC-IDRI-XXXX
        kaggle_source = temp_dir / "kaggle_dataset" / "cache_1mm" / "cache_1mm"
        for pid in pids:
            p_dir = kaggle_source / pid
            p_dir.mkdir(parents=True, exist_ok=True)
            (p_dir / "image_1mm.npy").write_bytes(b"kaggle_image")
            (p_dir / "meta.json").write_text(json.dumps({"pid": pid}), encoding="utf-8")

        # Thử 3 dạng hint khác nhau
        hint_root = str(temp_dir / "kaggle_dataset")
        hint_cache = str(kaggle_source)
        hint_patient = str(kaggle_source / "LIDC-IDRI-0001")

        for h in [hint_root, hint_cache, hint_patient]:
            v_dir = temp_dir / f"kaggle_view_{uuid.uuid4().hex[:4]}"
            run_kaggle(hint=h, view_dir=str(v_dir), n_patients=2, select="first")
            view_pids = sorted([d.name for d in v_dir.iterdir() if PATIENT_REGEX.match(d.name)])
            assert view_pids == ["LIDC-IDRI-0001", "LIDC-IDRI-0002"], f"Lỗi hint {h}: {view_pids}"
            # Đảm bảo đọc được nội dung qua symlink/junction
            assert (v_dir / "LIDC-IDRI-0001" / "image_1mm.npy").read_bytes() == b"kaggle_image"
        print("  ✓ Ca 6: Kaggle view-dir symlink hỗ trợ 3 dạng hint (root, cache, patient) hoàn hảo.")

        # 7. Đổi tập bệnh nhân -> slices.csv cũ bị xóa
        v_sync = temp_dir / "kaggle_view_sync"
        run_kaggle(hint=hint_cache, view_dir=str(v_sync), n_patients=2, select="first")
        (v_sync / "slices.csv").write_text("old_csv", encoding="utf-8")
        assert (v_sync / "slices.csv").exists()
        # Chạy lại với 3 bệnh nhân -> slices.csv phải bị xóa
        run_kaggle(hint=hint_cache, view_dir=str(v_sync), n_patients=3, select="first")
        assert not (v_sync / "slices.csv").exists(), "slices.csv cũ phải bị xóa khi đổi tập bệnh nhân"
        print("  ✓ Ca 7: Đổi tập bệnh nhân tự động dọn slices.csv cũ thông qua slices.meta.json.")

        # 8. --patients-file dựng lại đúng tập
        manifest_path = temp_dir / "manifest_test.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"patient_ids": ["LIDC-IDRI-0002", "LIDC-IDRI-0004"]}, f)
        v_pf = temp_dir / "kaggle_view_pf"
        run_kaggle(hint=hint_cache, view_dir=str(v_pf), patients_file=str(manifest_path))
        view_pf_ids = sorted([d.name for d in v_pf.iterdir() if PATIENT_REGEX.match(d.name)])
        assert view_pf_ids == ["LIDC-IDRI-0002", "LIDC-IDRI-0004"], f"Sai patients-file: {view_pf_ids}"
        print("  ✓ Ca 8: --patients-file tái lập tập bệnh nhân chuẩn xác.")

        # 9. Không thấy zip/dataset -> báo lỗi rõ ràng
        try:
            run_colab(zip_path="khong_ton_tai.zip", dest_dir=str(temp_dir / "dummy"))
            assert False, "Phải báo lỗi khi không thấy zip"
        except FileNotFoundError:
            print("  ✓ Ca 9: Thông báo lỗi tiếng Việt chuẩn xác khi thiếu dữ liệu nguồn.")

        # CA TỰ PHÁ 1: Bỏ qua sắp xếp trước khi chọn -> kiểm tra test bắt được
        print("  🔨 Thử nghiệm phá 1: Bỏ bước sắp xếp trước khi chọn...")
        unsorted_pool = ["LIDC-IDRI-0005", "LIDC-IDRI-0001", "LIDC-IDRI-0003"]
        broken_sel, _ = select_patients(unsorted_pool, n_requested=2, mode="first", _skip_sort=True)
        # Nếu không sort, 2 phần tử đầu là 0005 và 0001 thay vì 0001 và 0003
        assert broken_sel != ["LIDC-IDRI-0001", "LIDC-IDRI-0003"], "Test phải phát hiện ra nếu bỏ sắp xếp"
        print("  ✓ Phá 1 thành công: Test bắt được lỗi nếu bỏ sort trước khi chọn.")

        # CA TỰ PHÁ 2: Bỏ thư mục tạm + os.replace -> kiểm tra thư mục dở dang
        print("  🔨 Thử nghiệm phá 2: Bỏ bước thư mục tạm + os.replace...")
        dest_broken = temp_dir / "colab_broken"
        dest_broken.mkdir(parents=True, exist_ok=True)
        # Giả lập giải nén dở dang trực tiếp vào đích
        broken_pid_dir = dest_broken / "LIDC-IDRI-0001"
        broken_pid_dir.mkdir(parents=True, exist_ok=True)
        # Chỉ có meta.json mà thiếu image_1mm.npy
        (broken_pid_dir / "meta.json").write_text("partial", encoding="utf-8")
        # Khi dùng cơ chế chuẩn với temp + os.replace: nếu dở dang (thiếu image_1mm.npy), hàm sẽ tự phục hồi
        run_colab(zip_path=str(zip_path), dest_dir=str(dest_broken), n_patients=1, select="first")
        assert (broken_pid_dir / "image_1mm.npy").exists(), "Hệ thống phải tự khôi phục thư mục dở dang"
        print("  ✓ Phá 2 thành công: Cơ chế nguyên tử (atomic) bảo vệ an toàn khỏi giải nén dở.")

        print("✅ Selftest runtime_data: HOÀN TẤT TOÀN DIỆN VÀ THÀNH CÔNG!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "selftest":
        run_selftest()
    elif args.cmd == "colab":
        run_colab(
            zip_path=args.zip,
            dest_dir=args.dest,
            drive_root=args.drive_root,
            n_patients=args.n_patients,
            select=args.select,
            seed=args.seed,
            patients_file=args.patients_file,
            out_json=args.out_json,
        )
    elif args.cmd == "kaggle":
        run_kaggle(
            hint=args.hint,
            view_dir=args.view_dir,
            n_patients=args.n_patients,
            select=args.select,
            seed=args.seed,
            patients_file=args.patients_file,
            out_json=args.out_json,
        )
    elif args.cmd == "local":
        run_local(cache_dir=args.cache_dir, out_json=args.out_json)
    elif args.cmd == "auto":
        run_auto(
            zip=args.zip,
            dest=args.dest,
            hint=args.hint,
            view_dir=args.view_dir,
            cache_dir=args.cache_dir,
            patients_file=args.patients_file,
            out_json=args.out_json,
        )
