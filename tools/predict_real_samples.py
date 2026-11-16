# -*- coding: utf-8 -*-
"""
tools/predict_real_samples.py - Công cụ nạp mô hình đã train, lọc mẫu chưa train từ slices.csv,
suy luận 3D theo tiêu chí ngưỡng (độ tin cậy, đường kính mm, diện tích mm², thể tích mm³),
so sánh chi tiết từng lát cắt Predict vs GT và xuất hình ảnh trực quan hóa y khoa.

Các chế độ CLI:
  1. prepare : Nạp checkpoint + run_config.json + slices*.csv của đợt train về workspace,
               lọc bỏ toàn bộ ca đã train, copy đúng N ca chưa train (Val / Unseen) vào cache cục bộ.
  2. predict : Chạy suy luận 3D trên các ca mục tiêu với đầy đủ bộ tiêu chí lọc,
               tính % chuẩn trên từng lát cắt / từng khối u / toàn khối 3D, vẽ mask so sánh & xuất báo cáo.
  3. --selftest : Kiểm thử tự động toàn bộ chu trình trên dữ liệu 3D giả lập nhẹ (chạy nhanh trên CPU).
"""

import argparse
import dataclasses
import datetime
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.data_pipeline.medical_prep import (
    IMAGE_SIZE,
    clip_and_window_hu,
    load_cached,
    lung_cache_path,
    lung_keep_mask,
)
from common.data_pipeline.runtime_data import (
    PATIENT_REGEX,
    compute_patient_ids_hash,
    select_patients,
)
from common.engine.checkpoint_io import load_checkpoint_file
from common.eval3d.metrics3d import compute_3d_metrics, compute_binary_overlap_stats
from common.eval3d.volume_inference import predict_patient_volume

S26 = np.ones((3, 3, 3), dtype=np.uint8)
S8_2D = np.ones((3, 3), dtype=np.uint8)
NODULE_BINS: List[Tuple[float, float, str]] = [
    (0.0, 5.0, "< 5 mm (Rất nhỏ)"),
    (5.0, 10.0, "5 - 10 mm (Nhỏ)"),
    (10.0, 20.0, "10 - 20 mm (Vừa)"),
    (20.0, 1e9, ">= 20 mm (Lớn)"),
]

RUN_PRESETS: Dict[str, str] = {
    "r2bc": "mamba_colab_500item_50epoch_R2bc_LossBCE3_PosW15_Aug2",
    "r2a": "mamba_colab_500item_50epoch_R2a_WD002_EMA0999",
    "r2b": "mamba_colab_500item_50epoch_R2b_LossBCE3_PosW15",
    "r2c": "mamba_colab_500item_50epoch_R2c_AugLevel2",
    "r1c": "mamba_colab_500item_50epoch_R1c_StaticNeg_PosW2",
    "r1b": "mamba_colab_500item_50epoch_R1b_NegDynamic",
    "r1a": "mamba_colab_500item_25epoch_extra_test_R1a_Img448_Area5",
    "r0b": "mamba_colab_500item_25epoch_extra_test_R0_Baseline",
}


def _get_tqdm():
    """Trả về tqdm nếu thư viện sẵn sàng, ngược lại trả về iterator giả tương thích API."""
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        class _DummyTqdm:
            def __init__(self, iterable=None, *args, **kwargs):
                self._it = iterable if iterable is not None else []
            def __iter__(self):
                return iter(self._it)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n=1):
                pass
            def set_postfix_str(self, s=""):
                pass
        return _DummyTqdm


def _eq_diam_mm(area_mm2: float) -> float:
    """Quy đổi diện tích (mm²) sang đường kính tương đương hình tròn (mm)."""
    if area_mm2 <= 0.0:
        return 0.0
    return 2.0 * float(np.sqrt(float(area_mm2) / np.pi))


def _diam_bin_label(diam_mm: float) -> str:
    """Phân loại đường kính tương đương vào dải kích thước chuẩn LIDC."""
    for lo, hi, label in NODULE_BINS:
        if lo <= diam_mm < hi:
            return label
    return ">= 20 mm (Lớn)"


def _compute_assd_mm(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """Tính Average Symmetric Surface Distance (ASSD, mm) giữa 2 mask 3D."""
    pred_any = bool(pred_bin.any())
    gt_any = bool(gt_bin.any())
    if not pred_any and not gt_any:
        return 0.0
    if not pred_any or not gt_any:
        return float("nan")

    pred_border = pred_bin ^ ndi.binary_erosion(pred_bin)
    gt_border = gt_bin ^ ndi.binary_erosion(gt_bin)
    if not pred_border.any():
        pred_border = pred_bin
    if not gt_border.any():
        gt_border = gt_bin

    dt_gt = ndi.distance_transform_edt(~gt_border, sampling=spacing_zyx)
    dt_pred = ndi.distance_transform_edt(~pred_border, sampling=spacing_zyx)
    d1 = dt_gt[pred_border]
    d2 = dt_pred[gt_border]
    return float((float(d1.mean()) + float(d2.mean())) / 2.0)


@dataclasses.dataclass
class FilterCriteria:
    """Bộ tiêu chí lọc ngưỡng khi suy luận mẫu thật."""
    threshold: float = 0.5
    min_comp_max_prob: float = 0.0
    min_diameter_mm: float = 0.0
    min_area_mm2: float = 5.0
    min_volume_mm3: float = 0.0
    min_slices_span: int = 1
    filter_scope: str = "3d_component"  # '3d_component' | 'per_slice' | 'both'
    fill_holes: bool = True


# ==============================================================================
# 1. NẠP MÔ HÌNH, RUN_CONFIG & SLICES.CSV ĐỂ CHỐNG RÒ RỈ DỮ LIỆU TRAIN
# ==============================================================================
def _copy_file_with_progress(src: Path, dst: Path, desc: Optional[str] = None) -> None:
    """Copy file nguyên tử có thanh tiến trình tqdm (hữu ích cho file checkpoint > 400MB trên Drive)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = src.stat().st_size
    tmp = dst.parent / f".tmp_{dst.name}_{os.getpid()}"
    tqdm_cls = _get_tqdm()
    chunk_size = 4 * 1024 * 1024  # 4 MB
    show_bar = total_bytes >= 10 * 1024 * 1024

    try:
        with open(src, "rb") as f_in, open(tmp, "wb") as f_out:
            pbar = (
                tqdm_cls(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc or f"  Chép {src.name}",
                    dynamic_ncols=True,
                    leave=False,
                )
                if show_bar
                else None
            )
            try:
                while True:
                    buf = f_in.read(chunk_size)
                    if not buf:
                        break
                    f_out.write(buf)
                    if pbar is not None:
                        pbar.update(len(buf))
            finally:
                if pbar is not None:
                    pbar.close()
        os.replace(str(tmp), str(dst))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def find_run_artifacts(
    checkpoint_or_run_dir: Optional[Union[str, Path]] = None,
    drive_sources_dir: Optional[Union[str, Path]] = None,
    run_preset: Optional[str] = None,
    ckpt_name: str = "best_model.pth",
    arch: str = "mamba_unet",
) -> Dict[str, Optional[Path]]:
    """Tìm chính xác thư mục run nguồn, file checkpoint, run_config.json và file slices*.csv đi kèm."""
    run_folder_name = None
    if run_preset and run_preset != "custom":
        if run_preset not in RUN_PRESETS:
            raise ValueError(f"Preset '{run_preset}' không hợp lệ. Hỗ trợ: {sorted(RUN_PRESETS.keys())}")
        run_folder_name = RUN_PRESETS[run_preset]

    search_roots: List[Path] = []
    if drive_sources_dir:
        search_roots.append(Path(drive_sources_dir))
    search_roots.append(_ROOT)

    src_ckpt: Optional[Path] = None
    src_run_dir: Optional[Path] = None

    if checkpoint_or_run_dir:
        p = Path(checkpoint_or_run_dir)
        if not p.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy checkpoint hoặc thư mục run: {p}")
        if p.is_file():
            src_ckpt = p
            src_run_dir = p.parent
        else:
            src_run_dir = p
            cand = src_run_dir / ckpt_name
            if not cand.exists():
                for alt in ("best_model.pth", "best_weights_fp16.pth", "last_model.pth"):
                    if (src_run_dir / alt).exists():
                        cand = src_run_dir / alt
                        break
            if not cand.exists():
                raise FileNotFoundError(f"❌ Không thấy file trọng số ({ckpt_name}) trong {src_run_dir}")
            src_ckpt = cand
    elif run_folder_name:
        for base in search_roots:
            cand_dir = base / arch / "runs" / run_folder_name
            if cand_dir.is_dir():
                src_run_dir = cand_dir
                for cname in (ckpt_name, "best_model.pth", "best_weights_fp16.pth", "last_model.pth"):
                    if (cand_dir / cname).exists():
                        src_ckpt = cand_dir / cname
                        break
                if src_ckpt is not None:
                    break
        if src_ckpt is None or src_run_dir is None:
            searched = ", ".join(str(b / arch / "runs" / run_folder_name) for b in search_roots)
            raise FileNotFoundError(f"❌ Không tìm thấy run preset '{run_preset}' tại: {searched}")
    else:
        raise ValueError("Phải cung cấp --checkpoint/--run-dir hoặc --run-preset hợp lệ.")

    run_config_path = src_run_dir / "run_config.json"
    slices_csv_path = _find_slices_csv_in_dir(src_run_dir, run_config_path if run_config_path.exists() else None)
    return {
        "run_dir": src_run_dir,
        "checkpoint": src_ckpt,
        "run_config": run_config_path if run_config_path.exists() else None,
        "slices_csv": slices_csv_path,
    }


def _find_slices_csv_in_dir(run_dir: Path, run_config_path: Optional[Path] = None) -> Optional[Path]:
    """Dò file slices*.csv của chính đợt train bên trong thư mục run."""
    if not run_dir or not run_dir.is_dir():
        return None

    # Ưu tiên đọc tên file từ run_config.json
    if run_config_path and run_config_path.exists():
        try:
            cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
            csv_str = cfg.get("args", {}).get("slices_csv") or cfg.get("data", {}).get("slices_csv")
            if csv_str:
                csv_name = Path(str(csv_str)).name
                cand = run_dir / csv_name
                if cand.exists():
                    return cand
        except Exception:
            pass

    # Fallback tìm các file slices*.csv nằm trực tiếp trong thư mục run
    candidates = sorted(run_dir.glob("slices*.csv"))
    if candidates:
        return candidates[0]
    return None


def stage_model_to_workspace(
    artifacts: Dict[str, Optional[Path]],
    dest_model_dir: Union[str, Path],
    overwrite: bool = False,
) -> Dict[str, Optional[Path]]:
    """Copy checkpoint, run_config.json và slices*.csv của đợt train lên thư mục cục bộ (/content)."""
    src_ckpt = artifacts["checkpoint"]
    src_run_dir = artifacts["run_dir"]
    assert src_ckpt is not None and src_run_dir is not None

    dest_dir = Path(dest_model_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Nếu nguồn và đích đã trùng nhau thì không cần copy
    if src_run_dir.resolve() == dest_dir:
        return artifacts

    staged_ckpt = dest_dir / src_ckpt.name
    if overwrite or not staged_ckpt.exists() or staged_ckpt.stat().st_size != src_ckpt.stat().st_size:
        print(f"📥 Nạp checkpoint lên workspace: {src_ckpt.name} -> {dest_dir}")
        _copy_file_with_progress(src_ckpt, staged_ckpt, desc=f"  Chép {src_ckpt.name}")
    else:
        print(f"⏭️ Đã có sẵn checkpoint cục bộ: {staged_ckpt}")

    # Copy toàn bộ file metadata nhỏ đi kèm trong thư mục run (run_config.json, slices*.csv, REPORT.md...)
    small_files = ["run_config.json", "training_report.json", "REPORT.md", "DATA_SPEC_AND_CONFIG.md"]
    for p in src_run_dir.glob("slices*.csv"):
        small_files.append(p.name)

    staged_cfg = None
    staged_csv = None
    for fname in dict.fromkeys(small_files):
        sp = src_run_dir / fname
        if sp.is_file():
            dp = dest_dir / fname
            if overwrite or not dp.exists() or dp.stat().st_size != sp.stat().st_size:
                shutil.copy2(sp, dp)
            if fname == "run_config.json":
                staged_cfg = dp
            elif fname.startswith("slices") and fname.endswith(".csv") and staged_csv is None:
                staged_csv = dp

    if staged_csv is None and artifacts.get("slices_csv"):
        sp = artifacts["slices_csv"]
        if sp and sp.exists():
            dp = dest_dir / sp.name
            shutil.copy2(sp, dp)
            staged_csv = dp

    return {
        "run_dir": dest_dir,
        "checkpoint": staged_ckpt,
        "run_config": staged_cfg,
        "slices_csv": staged_csv,
    }


def discover_run_splits(
    run_slices_csv: Optional[Union[str, Path]],
    run_config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Đọc file slices.csv của đợt train để xác định chính xác tập train_ids và val_ids."""
    train_ids: set = set()
    val_ids: set = set()
    val_tumor_ids: set = set()
    val_neg_ids: set = set()
    csv_used: Optional[Path] = Path(run_slices_csv) if run_slices_csv else None

    if csv_used and csv_used.exists():
        df = pd.read_csv(csv_used)
        if "split" not in df.columns or "patient_id" not in df.columns:
            raise ValueError(f"❌ File {csv_used} thiếu cột 'split' hoặc 'patient_id'.")
        train_ids = set(df[df["split"] == "train"]["patient_id"].astype(str).unique())
        df_val = df[df["split"] == "val"]
        val_ids = set(df_val["patient_id"].astype(str).unique())

        if "has_tumor" in df_val.columns:
            grp = df_val.groupby("patient_id")["has_tumor"].max()
            val_tumor_ids = set(grp[grp > 0].index.astype(str))
            val_neg_ids = set(grp[grp == 0].index.astype(str))
        else:
            val_tumor_ids = set(val_ids)
    else:
        csv_used = None

    run_info: Dict[str, Any] = {}
    if run_config_path and Path(run_config_path).exists():
        try:
            run_info = json.loads(Path(run_config_path).read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "slices_csv": str(csv_used) if csv_used else None,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "val_tumor_ids": sorted(val_tumor_ids),
        "val_neg_ids": sorted(val_neg_ids),
        "run_info": run_info,
    }


def scan_available_patients(source_path: Union[str, Path]) -> Tuple[str, List[str]]:
    """Quét danh sách ID bệnh nhân LIDC-IDRI-XXXX có trong thư mục hoặc file .zip."""
    p = Path(source_path)
    if not p.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy nguồn dữ liệu: {p}")

    if p.is_file() and p.suffix.lower() == ".zip":
        found = set()
        with zipfile.ZipFile(p, "r") as zf:
            for info in zf.infolist():
                m = PATIENT_REGEX.search(info.filename)
                if m:
                    found.add(m.group(0))
        return "zip", sorted(found)

    if p.is_dir():
        found = sorted(
            d.name for d in p.iterdir()
            if d.is_dir() and PATIENT_REGEX.fullmatch(d.name)
        )
        return "dir", found

    raise ValueError(f"❌ Nguồn dữ liệu không hợp lệ (cần thư mục hoặc .zip): {p}")


def _load_tumor_hint_map(source_path: Union[str, Path], split_info: Dict[str, Any]) -> Dict[str, bool]:
    """Thu thập nhanh gợi ý ca nào có u / không u (không mở từng file .npy trên Drive)."""
    hints: Dict[str, bool] = {}
    for pid in split_info.get("val_tumor_ids", []):
        hints[pid] = True
    for pid in split_info.get("val_neg_ids", []):
        hints[pid] = False

    # Nếu nguồn là thư mục và có file slices.csv tổng quát trên cache gốc (chỉ dùng để biết ca nào có u, KHÔNG dùng split)
    p = Path(source_path)
    if p.is_dir() and (p / "slices.csv").is_file():
        try:
            df = pd.read_csv(p / "slices.csv", usecols=["patient_id", "has_tumor"])
            grp = df.groupby("patient_id")["has_tumor"].max()
            for pid, has_t in grp.items():
                pid_s = str(pid)
                if pid_s not in hints:
                    hints[pid_s] = bool(has_t > 0)
        except Exception:
            pass
    return hints


def select_non_train_patients(
    available_ids: Sequence[str],
    split_info: Dict[str, Any],
    source_mode: str = "val",
    selection_mode: str = "mixed",
    num_samples: int = 5,
    seed: int = 42,
    custom_patient_ids: Optional[Sequence[str]] = None,
    allow_train_leak: bool = False,
    tumor_hints: Optional[Dict[str, bool]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Lọc bỏ ca đã train và chọn đúng danh sách N mẫu phù hợp để predict:
      - source_mode='val'      : Chỉ lấy các ca thuộc split=='val' của đợt train.
      - source_mode='unseen'   : Chỉ lấy các ca có trong kho dữ liệu nhưng KHÔNG nằm trong slices.csv đợt train.
      - source_mode='non_train': Lấy cả 'val' và 'unseen' (loại sạch 'train').
      - source_mode='custom'   : Dùng danh sách custom_patient_ids (chặn ca 'train' trừ khi allow_train_leak=True).
    """
    avail_set = set(available_ids)
    train_set = set(split_info["train_ids"])
    val_set = set(split_info["val_ids"])
    unseen_set = avail_set - train_set - val_set

    status_map: Dict[str, str] = {}
    for pid in available_ids:
        if pid in train_set:
            status_map[pid] = "TRAIN"
        elif pid in val_set:
            status_map[pid] = "VAL"
        else:
            status_map[pid] = "UNSEEN"

    if not train_set and not val_set and source_mode == "val":
        raise RuntimeError(
            "❌ Không có file slices*.csv của đợt train nên không biết ca nào thuộc tập 'val'! "
            "Vui lòng truyền --slices-csv hoặc dùng --source custom / non_train."
        )

    if source_mode == "custom":
        req = list(dict.fromkeys(custom_patient_ids or []))
        if not req:
            raise ValueError("❌ Chế độ --source custom yêu cầu truyền danh sách --patient-ids.")
        missing = [pid for pid in req if pid not in avail_set]
        if missing:
            raise FileNotFoundError(f"❌ Không tìm thấy bệnh nhân trong nguồn dữ liệu: {missing}")
        leaked = [pid for pid in req if pid in train_set]
        if leaked and not allow_train_leak:
            raise ValueError(
                f"🚫 PHÁT HIỆN RÒ RỈ TẬP TRAIN! Các ca sau ĐÃ ĐƯỢC TRAIN trong đợt này: {leaked}.\n"
                f"Nếu bạn thực sự cố tình kiểm tra lại trên ca train, hãy thêm cờ --allow-train-leak."
            )
        if leaked:
            print(f"⚠️ CẢNH BÁO RÒ RỈ TRAIN: Bạn đang ép predict trên ca TRAIN: {leaked}")
        return req, {pid: status_map[pid] for pid in req}

    if source_mode == "val":
        candidates = sorted(avail_set & val_set)
    elif source_mode == "unseen":
        candidates = sorted(unseen_set)
    elif source_mode == "non_train":
        candidates = sorted((avail_set & val_set) | unseen_set)
    else:
        raise ValueError(f"❌ --source không hợp lệ: {source_mode}")

    if not candidates:
        raise RuntimeError(
            f"❌ Không tìm thấy bệnh nhân nào phù hợp cho source='{source_mode}' "
            f"(Tổng kho={len(available_ids)}, Train={len(train_set)}, Val={len(val_set)}, Unseen={len(unseen_set)})."
        )

    if num_samples <= 0 or num_samples >= len(candidates):
        selected = candidates
        return selected, {pid: status_map[pid] for pid in selected}

    hints = tumor_hints or {}
    if selection_mode == "first":
        selected = candidates[:num_samples]
    elif selection_mode == "random":
        import random
        rng = random.Random(seed)
        selected = sorted(rng.sample(candidates, num_samples))
    elif selection_mode in ("tumor_first", "mixed"):
        pos_cands = [p for p in candidates if hints.get(p) is True]
        neg_cands = [p for p in candidates if hints.get(p) is False]
        unk_cands = [p for p in candidates if p not in hints]

        import random
        rng = random.Random(seed)
        rng.shuffle(pos_cands)
        rng.shuffle(neg_cands)
        rng.shuffle(unk_cands)

        if selection_mode == "mixed" and num_samples >= 4 and neg_cands:
            n_neg = max(1, min(len(neg_cands), max(1, num_samples // 5)))
            n_pos = num_samples - n_neg
            chosen = pos_cands[:n_pos] + neg_cands[:n_neg]
            if len(chosen) < num_samples:
                rem_pool = [p for p in (pos_cands + unk_cands + neg_cands) if p not in chosen]
                chosen.extend(rem_pool[: num_samples - len(chosen)])
            selected = sorted(chosen[:num_samples])
        else:
            pool = pos_cands + unk_cands + neg_cands
            selected = sorted(pool[:num_samples])
    else:
        raise ValueError(f"❌ --select không hợp lệ: {selection_mode}")

    return selected, {pid: status_map[pid] for pid in selected}


def stage_patient_samples(
    source_kind: str,
    source_path: Union[str, Path],
    selected_ids: Sequence[str],
    dest_cache_dir: Union[str, Path],
    local_fast_cache_dir: Optional[Union[str, Path]] = None,
    workers: int = 4,
) -> Path:
    """Copy chỉ đúng các bệnh nhân được chọn về ổ SSD cục bộ với thanh tiến trình tqdm."""
    dest = Path(dest_cache_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for tmp_p in dest.glob(".tmp_*"):
        shutil.rmtree(tmp_p, ignore_errors=True)

    fast_dir = Path(local_fast_cache_dir).resolve() if local_fast_cache_dir else None
    pending: List[str] = []
    already_ok = 0
    for pid in selected_ids:
        target_dir = dest / pid
        if (target_dir / "image_1mm.npy").is_file() and (target_dir / "meta.json").is_file():
            already_ok += 1
        else:
            pending.append(pid)

    print(
        f"📦 Chuẩn bị dữ liệu Predict tại {dest}: "
        f"Chọn {len(selected_ids)} ca | Đã có sẵn: {already_ok} | Cần chép thêm: {len(pending)}"
    )
    if not pending:
        return dest

    tqdm_cls = _get_tqdm()
    src = Path(source_path)

    if source_kind == "zip":
        wanted = set(pending)
        entries: Dict[str, List[zipfile.ZipInfo]] = {pid: [] for pid in pending}
        with zipfile.ZipFile(src, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                m = PATIENT_REGEX.search(info.filename)
                if m and m.group(0) in wanted:
                    entries[m.group(0)].append(info)

            for pid in tqdm_cls(pending, desc="📦 Giải nén mẫu từ Drive zip", dynamic_ncols=True):
                # Ưu tiên lấy từ local SSD cache nếu đã có
                if fast_dir and (fast_dir / pid / "image_1mm.npy").is_file():
                    _copy_patient_dir(fast_dir / pid, dest / pid)
                    continue
                tmp = dest / f".tmp_{pid}_{uuid.uuid4().hex[:6]}"
                tmp.mkdir(parents=True, exist_ok=True)
                try:
                    for info in entries[pid]:
                        parts = Path(info.filename).parts
                        pid_idx = parts.index(pid)
                        rel_parts = parts[pid_idx + 1 :]
                        out_f = tmp / Path(*rel_parts)
                        out_f.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as sf, open(out_f, "wb") as df:
                            shutil.copyfileobj(sf, df)
                    if (dest / pid).exists():
                        shutil.rmtree(dest / pid)
                    os.replace(str(tmp), str(dest / pid))
                except Exception:
                    shutil.rmtree(tmp, ignore_errors=True)
                    raise
    else:
        def _copy_job(pid: str) -> Tuple[str, Optional[str]]:
            try:
                if fast_dir and (fast_dir / pid / "image_1mm.npy").is_file():
                    _copy_patient_dir(fast_dir / pid, dest / pid)
                    return pid, "local-ssd"
                _copy_patient_dir(src / pid, dest / pid)
                return pid, "drive"
            except Exception as e:
                return pid, f"ERR:{e!r}"

        errors = []
        n_workers = max(1, min(int(workers), len(pending)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_copy_job, pid) for pid in pending]
            for fut in tqdm_cls(as_completed(futs), total=len(futs), desc="📥 Chép mẫu về SSD", dynamic_ncols=True):
                pid, status = fut.result()
                if status and status.startswith("ERR:"):
                    errors.append((pid, status))
        if errors:
            raise RuntimeError(f"❌ Lỗi khi copy mẫu từ Drive: {errors[:5]}")

    return dest


def _copy_patient_dir(src_pid_dir: Path, dst_pid_dir: Path) -> None:
    """Copy nguyên tử 1 thư mục bệnh nhân cache_1mm/<pid>."""
    if src_pid_dir.resolve() == dst_pid_dir.resolve():
        return
    tmp = dst_pid_dir.parent / f".tmp_{dst_pid_dir.name}_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        for f in src_pid_dir.iterdir():
            if f.is_file() and f.name.lower() != "desktop.ini":
                shutil.copyfile(f, tmp / f.name)
        if dst_pid_dir.exists():
            shutil.rmtree(dst_pid_dir)
        os.replace(str(tmp), str(dst_pid_dir))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


# ==============================================================================
# 2. TỰ ĐỘNG NHẬN DIỆN KIẾN TRÚC, IMAGE_SIZE VÀ NẠP TRỌNG SỐ MÔ HÌNH
# ==============================================================================
def resolve_model_metadata(
    checkpoint_path: Union[str, Path],
    run_config_path: Optional[Union[str, Path]] = None,
    explicit_arch: Optional[str] = None,
    explicit_backbone: Optional[str] = None,
    explicit_image_size: int = 0,
) -> Dict[str, Any]:
    """Đọc đầy đủ metadata từ checkpoint + run_config.json để nạp đúng 100% mô hình."""
    ckpt_p = Path(checkpoint_path)
    ckpt = load_checkpoint_file(ckpt_p, map_location="cpu")
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

    cfg: Dict[str, Any] = {}
    cfg_p = Path(run_config_path) if run_config_path else (ckpt_p.parent / "run_config.json")
    if cfg_p.is_file():
        try:
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

    arch = (
        explicit_arch
        or meta.get("arch")
        or cfg.get("arch")
        or ("unet_base" if "unet_base" in str(ckpt_p) else "mamba_unet")
    )
    backbone = (
        explicit_backbone
        or meta.get("backbone")
        or cfg.get("model", {}).get("backbone")
        or cfg.get("args", {}).get("backbone")
        or "mamba_vision_T"
    )
    activation = (
        meta.get("activation")
        or cfg.get("model", {}).get("activation")
        or cfg.get("args", {}).get("activation")
        or "relu"
    )
    image_size = int(
        explicit_image_size
        if (explicit_image_size and explicit_image_size > 0)
        else (
            meta.get("image_size")
            or cfg.get("model", {}).get("image_size")
            or cfg.get("data", {}).get("image_size")
            or cfg.get("args", {}).get("duneai_image_size")
            or IMAGE_SIZE
        )
    )
    return {
        "arch": str(arch),
        "backbone": str(backbone),
        "activation": str(activation),
        "image_size": image_size,
        "epoch": int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0,
        "best_val_dice": float(ckpt.get("best_val_dice", 0.0)) if isinstance(ckpt, dict) else 0.0,
        "is_ema": bool(ckpt.get("is_ema", False)) if isinstance(ckpt, dict) else False,
        "run_id": cfg.get("run_id", ckpt_p.parent.name),
    }


def load_model_for_predict(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    meta_info: Dict[str, Any],
) -> torch.nn.Module:
    """Nạp mô hình theo kiến trúc (mamba_unet hoặc unet_base) tương thích CUDA hoặc CPU fallback."""
    arch = meta_info["arch"]
    os.environ["DUNEAI_IMAGE_SIZE"] = str(meta_info["image_size"])

    if device.type != "cuda":
        os.environ.setdefault("DUNEAI_MAMBA_BACKEND", "pytorch")

    if arch == "unet_base":
        from unet_base.evaluate import load_unet_base_model
        return load_unet_base_model(
            checkpoint_path=checkpoint_path,
            device=device,
            activation=meta_info["activation"],
        )

    from mamba_unet.evaluate import load_mamba_unet_model
    return load_mamba_unet_model(
        checkpoint_path=checkpoint_path,
        device=device,
        backbone=meta_info["backbone"],
        activation=meta_info["activation"],
    )


# ==============================================================================
# 3. LỌC NGƯỠNG ĐỘ TIN CẬY, ĐƯỜNG KÍNH (MM), DIỆN TÍCH (MM²), THỂ TÍCH (MM³)
# ==============================================================================
def _filter_2d_slices(
    mask_zyx: np.ndarray,
    min_area_mm2: float,
    min_diameter_mm: float,
) -> np.ndarray:
    """Lọc các đảo kết nối 2D trên từng lát cắt theo diện tích (mm²) và đường kính (mm)."""
    if min_area_mm2 <= 0.0 and min_diameter_mm <= 0.0:
        return mask_zyx

    out = np.zeros_like(mask_zyx, dtype=np.uint8)
    for z in np.where(mask_zyx.sum(axis=(1, 2)) > 0)[0]:
        lab2d, n2d = ndi.label(mask_zyx[z], structure=S8_2D)
        for cid in range(1, n2d + 1):
            comp = lab2d == cid
            area = float(comp.sum())
            diam = _eq_diam_mm(area)
            if area >= min_area_mm2 and diam >= min_diameter_mm:
                out[z][comp] = 1
    return out


def filter_and_analyze_components(
    prob_1mm: np.ndarray,
    gt_1mm: np.ndarray,
    lung_1mm: Optional[np.ndarray],
    spacing_z: float,
    criteria: FilterCriteria,
) -> Dict[str, Any]:
    """
    Áp bộ tiêu chí ngưỡng lên bản đồ xác suất 3D và phân tích chi tiết:
      - Tách riêng cụm ĐƯỢC GIỮ LẠI (clean_mask) và cụm BỊ LOẠI BỎ (rejected_mask) kèm lý do.
      - Đối chiếu khớp từng nốt GT thật (26-connectivity) theo dải đường kính (<5mm, 5-10mm, ...).
    """
    keep_lung = lung_keep_mask(lung_1mm, tumor_mask=None) if lung_1mm is not None else None
    raw_mask = (np.asarray(prob_1mm) >= float(criteria.threshold)).astype(np.uint8)
    if keep_lung is not None:
        raw_mask &= keep_lung

    work_mask = raw_mask.copy()
    if criteria.fill_holes and work_mask.sum() > 0:
        for z in np.where(work_mask.sum(axis=(1, 2)) > 0)[0]:
            work_mask[z] = ndi.binary_fill_holes(work_mask[z]).astype(np.uint8)

    if criteria.filter_scope in ("per_slice", "both") and work_mask.sum() > 0:
        work_mask = _filter_2d_slices(work_mask, criteria.min_area_mm2, criteria.min_diameter_mm)

    # Nhãn hóa khối GT 3D
    gt_b = np.asarray(gt_1mm) > 0
    gt_lab, n_gt = ndi.label(gt_b, structure=S26)
    gt_objs = ndi.find_objects(gt_lab) if n_gt > 0 else []
    gt_vox_counts = np.bincount(gt_lab.ravel()) if n_gt > 0 else np.zeros(1, dtype=np.int64)

    # Nhãn hóa cụm Predict 3D
    pred_lab, n_pred = ndi.label(work_mask > 0, structure=S26) if work_mask.sum() > 0 else (np.zeros_like(work_mask, dtype=np.int32), 0)
    pred_objs = ndi.find_objects(pred_lab) if n_pred > 0 else []

    clean_mask = np.zeros_like(work_mask, dtype=np.uint8)
    rejected_mask = np.zeros_like(work_mask, dtype=np.uint8)
    pred_components: List[Dict[str, Any]] = []

    for cid, slc in enumerate(pred_objs, start=1):
        if slc is None:
            continue
        sub = pred_lab[slc] == cid
        vox = int(sub.sum())
        if vox == 0:
            continue

        z0, z1 = int(slc[0].start), int(slc[0].stop - 1)
        z_span = int(sub.shape[0])
        slice_areas = sub.reshape(z_span, -1).sum(axis=1)
        max_area = float(slice_areas.max())
        max_diam = _eq_diam_mm(max_area)
        vol_mm3 = float(vox) * float(spacing_z)

        probs_sub = prob_1mm[slc][sub]
        max_prob = float(probs_sub.max())
        mean_prob = float(probs_sub.mean())

        coords = np.argwhere(sub)
        cy = float(coords[:, 1].mean() + slc[1].start)
        cx = float(coords[:, 2].mean() + slc[2].start)
        cz = float(coords[:, 0].mean() + slc[0].start)

        reasons: List[str] = []
        if max_prob < criteria.min_comp_max_prob:
            reasons.append(f"MaxProb={max_prob:.2f} < {criteria.min_comp_max_prob:.2f}")
        if z_span < criteria.min_slices_span:
            reasons.append(f"Span={z_span} < {criteria.min_slices_span}")
        if max_area < criteria.min_area_mm2:
            reasons.append(f"Area={max_area:.1f} < {criteria.min_area_mm2:.1f} mm²")
        if max_diam < criteria.min_diameter_mm:
            reasons.append(f"Diam={max_diam:.1f} < {criteria.min_diameter_mm:.1f} mm")
        if vol_mm3 < criteria.min_volume_mm3:
            reasons.append(f"Vol={vol_mm3:.1f} < {criteria.min_volume_mm3:.1f} mm³")

        kept = len(reasons) == 0
        if kept:
            clean_mask[slc][sub] = 1
        else:
            rejected_mask[slc][sub] = 1

        tp_vox = int(gt_b[slc][sub].sum())
        hit_ids = sorted(int(x) for x in np.unique(gt_lab[slc][sub]) if x > 0) if tp_vox > 0 else []
        verdict = ("TP (Trúng u thật)" if hit_ids else "FP (Dương tính giả)") if kept else "REJECTED (Đã lọc bỏ)"
        hit_gt_total_vox = sum(int(gt_vox_counts[hid]) for hid in hit_ids)

        pred_components.append({
            "comp_type": "PRED",
            "comp_id": cid,
            "kept": kept,
            "filter_reason": "ĐẠT NGƯỠNG" if kept else "; ".join(reasons),
            "verdict": verdict,
            "z_start": z0,
            "z_end": z1,
            "z_span": z_span,
            "centroid_zyx": f"({cz:.1f}, {cy:.1f}, {cx:.1f})",
            "max_area_mm2": round(max_area, 2),
            "max_diam_mm": round(max_diam, 2),
            "size_bin": _diam_bin_label(max_diam),
            "vol_mm3": round(vol_mm3, 2),
            "voxels": vox,
            "max_prob": round(max_prob, 4),
            "mean_prob": round(mean_prob, 4),
            "overlap_gt_voxels": tp_vox,
            "matched_gt_ids": ",".join(str(x) for x in hit_ids) if hit_ids else "-",
            "overlap_dice_pct": round(100.0 * (2.0 * tp_vox) / (vox + hit_gt_total_vox) if hit_ids else 0.0, 2),
        })

    # Phân tích từng nốt GT thật đối chiếu với clean_mask đã giữ lại
    gt_nodules: List[Dict[str, Any]] = []
    for gid, slc in enumerate(gt_objs, start=1):
        if slc is None:
            continue
        sub = gt_lab[slc] == gid
        vox = int(sub.sum())
        if vox == 0:
            continue
        z0, z1 = int(slc[0].start), int(slc[0].stop - 1)
        z_span = int(sub.shape[0])
        max_area = float(sub.reshape(z_span, -1).sum(axis=1).max())
        max_diam = _eq_diam_mm(max_area)
        vol_mm3 = float(vox) * float(spacing_z)

        coords = np.argwhere(sub)
        cy = float(coords[:, 1].mean() + slc[1].start)
        cx = float(coords[:, 2].mean() + slc[2].start)
        cz = float(coords[:, 0].mean() + slc[0].start)

        pred_in_gt = clean_mask[slc][sub] > 0
        tp_vox = int(pred_in_gt.sum())
        detected = tp_vox > 0
        recall_pct = 100.0 * tp_vox / max(1, vox)

        # Tính xác suất trung bình & cực đại của model ngay trên vùng GT này (kể cả khi bị miss ngưỡng!)
        probs_gt = prob_1mm[slc][sub]
        gt_max_prob = float(probs_gt.max())
        gt_mean_prob = float(probs_gt.mean())

        # Tính Dice cục bộ trong bounding box mở rộng nhẹ quanh nốt GT hoặc theo cụm pred khớp
        pred_touch_ids = set(int(x) for x in np.unique(pred_lab[slc][sub]) if x > 0)
        # Chỉ tính union của những cụm pred được giữ lại mà chạm vào GT này
        kept_touch_mask = np.zeros_like(clean_mask, dtype=bool)
        for pc in pred_components:
            if pc["kept"] and str(gid) in pc["matched_gt_ids"].split(","):
                kept_touch_mask |= (pred_lab == pc["comp_id"])
        if kept_touch_mask.any():
            gt_full_g = (gt_lab == gid)
            tp_full = int((kept_touch_mask & gt_full_g).sum())
            p_full = int(kept_touch_mask.sum())
            g_full = int(gt_full_g.sum())
            nod_dice = 100.0 * (2.0 * tp_full) / (p_full + g_full)
            nod_iou = 100.0 * tp_full / (p_full + g_full - tp_full)
        else:
            nod_dice = 0.0
            nod_iou = 0.0

        gt_nodules.append({
            "comp_type": "GT",
            "comp_id": gid,
            "kept": True,
            "filter_reason": "-",
            "verdict": "DETECTED (Bắt được)" if detected else "MISSED (Bỏ sót)",
            "z_start": z0,
            "z_end": z1,
            "z_span": z_span,
            "centroid_zyx": f"({cz:.1f}, {cy:.1f}, {cx:.1f})",
            "max_area_mm2": round(max_area, 2),
            "max_diam_mm": round(max_diam, 2),
            "size_bin": _diam_bin_label(max_diam),
            "vol_mm3": round(vol_mm3, 2),
            "voxels": vox,
            "max_prob": round(gt_max_prob, 4),
            "mean_prob": round(gt_mean_prob, 4),
            "overlap_gt_voxels": tp_vox,
            "matched_gt_ids": str(gid),
            "overlap_dice_pct": round(nod_dice, 2),
            "nodule_recall_pct": round(recall_pct, 2),
            "nodule_iou_pct": round(nod_iou, 2),
        })

    return {
        "raw_mask_1mm": raw_mask,
        "clean_mask_1mm": clean_mask,
        "rejected_mask_1mm": rejected_mask,
        "keep_lung": keep_lung,
        "gt_nodules": gt_nodules,
        "pred_components": pred_components,
    }


# ==============================================================================
# 4. ĐỐI CHIẾU TỪNG LÁT CẮT PREDICT VS GT & TÍNH ĐẦY ĐỦ THÔNG SỐ % CHUẨN
# ==============================================================================
def compute_slice_and_patient_stats(
    pid: str,
    split_status: str,
    prob_1mm: np.ndarray,
    raw_mask_1mm: np.ndarray,
    clean_mask_1mm: np.ndarray,
    gt_mask_1mm: np.ndarray,
    gt_nodules: List[Dict[str, Any]],
    pred_components: List[Dict[str, Any]],
    meta: Dict[str, Any],
    elapsed_sec: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Tính toàn bộ chỉ số 3D, chỉ số mức nốt và bảng so sánh chi tiết từng lát cắt."""
    spacing_z = float(meta.get("orig_spacing_xyz", [1.0, 1.0, 1.0])[2])
    m_raw = compute_3d_metrics(raw_mask_1mm, gt_mask_1mm, spacing_z=spacing_z, compute_hd95=False)
    m_post = compute_3d_metrics(clean_mask_1mm, gt_mask_1mm, spacing_z=spacing_z, compute_hd95=True)
    assd_mm = _compute_assd_mm(clean_mask_1mm > 0, gt_mask_1mm > 0, spacing_zyx=(spacing_z, 1.0, 1.0))

    gt_z_set = set(int(z) for z in np.where(gt_mask_1mm.reshape(gt_mask_1mm.shape[0], -1).sum(axis=1) > 0)[0])
    pred_z_set = set(int(z) for z in np.where(clean_mask_1mm.reshape(clean_mask_1mm.shape[0], -1).sum(axis=1) > 0)[0])
    compare_slices = sorted(gt_z_set | pred_z_set)

    slice_rows: List[Dict[str, Any]] = []
    pred_slice_dices: List[float] = []
    compare_slice_dices: List[float] = []
    pred_slices_tp_sum = 0.0
    pred_slices_p_sum = 0.0
    pred_slices_g_sum = 0.0
    n_pred_hit_gt_loc = 0
    n_pred_same_z_only = 0

    for z in compare_slices:
        g = gt_mask_1mm[z] > 0
        p = clean_mask_1mm[z] > 0
        prob_z = prob_1mm[z]

        g_area = float(g.sum())
        p_area = float(p.sum())
        tp_px = float(np.logical_and(p, g).sum())
        fp_px = float(np.logical_and(p, ~g).sum())
        fn_px = float(np.logical_and(~p, g).sum())

        stats_2d = compute_binary_overlap_stats(tp_px, p_area, g_area)
        dice_pct = stats_2d["dice_3d"] * 100.0
        iou_pct = stats_2d["iou_3d"] * 100.0
        prec_pct = stats_2d["precision_3d"] * 100.0
        rec_pct = stats_2d["recall_3d"] * 100.0

        if p_area > 0 and g_area > 0:
            if tp_px > 0:
                status = "TP (Khớp GT)"
                n_pred_hit_gt_loc += 1
            else:
                status = "MISS_LOC (Cùng lát z nhưng lệch chỗ)"
            n_pred_same_z_only += 1
        elif p_area > 0:
            status = "FP (Đoán nhầm - GT không có u)"
        else:
            status = "FN (Bỏ sót - GT có u)"

        if p_area > 0:
            max_prob_z = float(prob_z[p].max())
            mean_prob_z = float(prob_z[p].mean())
            pred_slice_dices.append(dice_pct)
            pred_slices_tp_sum += tp_px
            pred_slices_p_sum += p_area
            pred_slices_g_sum += g_area
        else:
            max_prob_z = float(prob_z[g].max()) if g_area > 0 else float(prob_z.max())
            mean_prob_z = float(prob_z[g].mean()) if g_area > 0 else 0.0

        compare_slice_dices.append(dice_pct)
        slice_rows.append({
            "patient_id": pid,
            "z": z,
            "num_slices": int(gt_mask_1mm.shape[0]),
            "has_gt": int(g_area > 0),
            "has_pred": int(p_area > 0),
            "status": status,
            "gt_area_mm2": round(g_area, 1),
            "pred_area_mm2": round(p_area, 1),
            "gt_diam_mm": round(_eq_diam_mm(g_area), 2),
            "pred_diam_mm": round(_eq_diam_mm(p_area), 2),
            "max_prob": round(max_prob_z, 4),
            "mean_prob": round(mean_prob_z, 4),
            "tp_px": int(tp_px),
            "fp_px": int(fp_px),
            "fn_px": int(fn_px),
            "slice_dice_pct": round(dice_pct, 2),
            "slice_iou_pct": round(iou_pct, 2),
            "slice_precision_pct": round(prec_pct, 2),
            "slice_recall_pct": round(rec_pct, 2),
        })

    # Tổng hợp % chuẩn trên tập các lát ĐƯỢC PREDICT
    if pred_slices_p_sum > 0:
        pooled_pred_slices_dice = (2.0 * pred_slices_tp_sum) / (pred_slices_p_sum + pred_slices_g_sum) * 100.0
        mean_pred_slices_dice = float(np.mean(pred_slice_dices))
        slice_prec_pct = 100.0 * n_pred_hit_gt_loc / len(pred_z_set)
    elif len(gt_z_set) == 0:
        pooled_pred_slices_dice = 100.0
        mean_pred_slices_dice = 100.0
        slice_prec_pct = 100.0
    else:
        pooled_pred_slices_dice = 0.0
        mean_pred_slices_dice = 0.0
        slice_prec_pct = 0.0

    slice_rec_pct = (100.0 * n_pred_hit_gt_loc / len(gt_z_set)) if len(gt_z_set) > 0 else (100.0 if len(pred_z_set) == 0 else 0.0)
    mean_compare_dice = float(np.mean(compare_slice_dices)) if compare_slice_dices else 100.0

    gt_vol = float(m_post["gt_volume_mm3"])
    pred_vol = float(m_post["pred_volume_mm3"])
    vol_sim_pct = (
        100.0 if (gt_vol == 0 and pred_vol == 0)
        else (1.0 - abs(pred_vol - gt_vol) / (pred_vol + gt_vol)) * 100.0
    )
    vol_ratio = (pred_vol / gt_vol) if gt_vol > 0 else (0.0 if pred_vol == 0 else float("inf"))

    kept_preds = [c for c in pred_components if c["kept"]]
    rejected_preds = [c for c in pred_components if not c["kept"]]
    tp_preds = [c for c in kept_preds if c["matched_gt_ids"] != "-"]
    fp_preds = [c for c in kept_preds if c["matched_gt_ids"] == "-"]
    detected_gt = [g for g in gt_nodules if g["overlap_gt_voxels"] > 0]

    # Phân loại kết quả chung của ca bệnh
    if len(gt_nodules) == 0 and len(kept_preds) == 0:
        case_outcome = "CLEAN_NEG (Âm tính sạch 0 FP)"
    elif len(gt_nodules) == 0 and len(kept_preds) > 0:
        case_outcome = "FALSE_POS (Đoán nhầm trên ca sạch)"
    elif len(detected_gt) == len(gt_nodules) and len(fp_preds) == 0:
        case_outcome = "EXACT_HIT (Bắt trọn tất cả nốt, 0 FP)"
    elif len(detected_gt) > 0:
        case_outcome = "PARTIAL_HIT" if len(detected_gt) < len(gt_nodules) else "HIT_WITH_FP"
    else:
        case_outcome = "MISSED_ALL (Bỏ sót toàn bộ u GT)"

    patient_summary = {
        "patient_id": pid,
        "split_status": split_status,
        "case_outcome": case_outcome,
        "num_slices": int(gt_mask_1mm.shape[0]),
        "spacing_z_mm": round(spacing_z, 3),
        "n_gt_nodules": len(gt_nodules),
        "n_gt_detected": len(detected_gt),
        "nodule_sens_pct": round(100.0 * len(detected_gt) / len(gt_nodules), 2) if gt_nodules else 100.0,
        "n_pred_kept": len(kept_preds),
        "n_pred_tp": len(tp_preds),
        "n_pred_fp": len(fp_preds),
        "n_pred_filtered_out": len(rejected_preds),
        "n_gt_slices": len(gt_z_set),
        "n_pred_slices": len(pred_z_set),
        "n_pred_hit_slices": n_pred_hit_gt_loc,
        "slice_prec_pct": round(slice_prec_pct, 2),
        "slice_rec_pct": round(slice_rec_pct, 2),
        "pooled_dice_on_pred_slices_pct": round(pooled_pred_slices_dice, 2),
        "mean_dice_on_pred_slices_pct": round(mean_pred_slices_dice, 2),
        "mean_dice_on_compare_slices_pct": round(mean_compare_dice, 2),
        "dice_3d_raw_pct": round(float(m_raw["dice_3d"]) * 100.0, 2),
        "dice_3d_post_pct": round(float(m_post["dice_3d"]) * 100.0, 2),
        "iou_3d_post_pct": round(float(m_post["iou_3d"]) * 100.0, 2),
        "precision_3d_pct": round(float(m_post["precision_3d"]) * 100.0, 2),
        "recall_3d_pct": round(float(m_post["recall_3d"]) * 100.0, 2),
        "vol_similarity_pct": round(vol_sim_pct, 2),
        "gt_vol_mm3": gt_vol,
        "pred_vol_mm3": pred_vol,
        "vol_ratio": round(vol_ratio, 3) if np.isfinite(vol_ratio) else "inf",
        "hd95_mm": m_post["hd95_mm"],
        "assd_mm": round(assd_mm, 3) if np.isfinite(assd_mm) else float("nan"),
        "time_sec": round(elapsed_sec, 2),
    }
    return patient_summary, slice_rows


# ==============================================================================
# 5. TRỰC QUAN HÓA MASK SO SÁNH TỪNG LÁT CẮT & TOÀN CẢNH 3D (MATPLOTLIB AGG)
# ==============================================================================
def _compute_zoom_bbox(gt_2d: np.ndarray, pred_2d: np.ndarray, h: int, w: int, min_box: int = 96, pad: int = 24) -> Tuple[int, int, int, int]:
    """Tính khung zoom ROI bao quanh hợp của GT và Predict trên lát cắt 2D."""
    union = (gt_2d > 0) | (pred_2d > 0)
    if not union.any():
        cy, cx = h // 2, w // 2
        half = min_box // 2
        return max(0, cy - half), min(h, cy + half), max(0, cx - half), min(w, cx + half)

    ys, xs = np.where(union)
    y0, y1 = int(ys.min()) - pad, int(ys.max()) + 1 + pad
    x0, x1 = int(xs.min()) - pad, int(xs.max()) + 1 + pad

    cur_h, cur_w = y1 - y0, x1 - x0
    side = max(min_box, cur_h, cur_w)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = side // 2
    ry0 = max(0, min(h - side, cy - half)) if side < h else 0
    rx0 = max(0, min(w - side, cx - half)) if side < w else 0
    ry1 = min(h, ry0 + side)
    rx1 = min(w, rx0 + side)
    return ry0, ry1, rx0, rx1


def render_patient_visualizations(
    pid: str,
    patient_cache_dir: Path,
    prob_1mm: np.ndarray,
    clean_mask_1mm: np.ndarray,
    gt_mask_1mm: np.ndarray,
    lung_mask_1mm: Optional[np.ndarray],
    patient_summary: Dict[str, Any],
    slice_rows: List[Dict[str, Any]],
    threshold: float,
    figures_dir: Path,
    slices_per_page: int = 6,
    max_vis_slices: int = 0,
) -> List[Path]:
    """Vẽ bộ ảnh so sánh trực quan GT vs Predict cho toàn bộ lát cắt mục tiêu + tổng quan 3D."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figures_dir.mkdir(parents=True, exist_ok=True)
    img_1mm, _, meta = load_cached(patient_cache_dir, mmap=True)
    num_slices, h, w = img_1mm.shape
    saved_paths: List[Path] = []

    # 1. Chọn danh sách lát cắt cần vẽ
    rows_to_draw = list(slice_rows)
    if max_vis_slices and max_vis_slices > 0 and len(rows_to_draw) > max_vis_slices:
        # Ưu tiên giữ các lát TP hoặc FP rõ nhất, lấy đều
        idx_pick = np.linspace(0, len(rows_to_draw) - 1, max_vis_slices, dtype=int)
        rows_to_draw = [rows_to_draw[int(i)] for i in idx_pick]

    # Nếu ca âm tính sạch hoàn toàn (0 lát GT & 0 lát Pred), chọn lát giữa vùng phổi để xác nhận trực quan
    if not rows_to_draw:
        if lung_mask_1mm is not None and lung_mask_1mm.sum() > 0:
            active = np.where(lung_mask_1mm.reshape(num_slices, -1).sum(axis=1) > 0)[0]
            mid_z = int(active[len(active) // 2])
        else:
            mid_z = num_slices // 2
        rows_to_draw = [{
            "z": mid_z,
            "status": "CLEAN_NEG (0 GT | 0 Pred)",
            "gt_area_mm2": 0.0,
            "pred_area_mm2": 0.0,
            "gt_diam_mm": 0.0,
            "pred_diam_mm": 0.0,
            "max_prob": float(prob_1mm[mid_z].max()),
            "slice_dice_pct": 100.0,
            "slice_iou_pct": 100.0,
        }]

    # Precompute lung-window base image chỉ cho những lát cần vẽ
    z_indices = sorted(set(int(r["z"]) for r in rows_to_draw))
    img_win_map = {z: clip_and_window_hu(np.asarray(img_1mm[z])) for z in z_indices}

    # Chia trang vẽ so sánh chi tiết từng lát (4 cột/lát)
    n_pages = int(np.ceil(len(rows_to_draw) / slices_per_page))
    for page_idx in range(n_pages):
        chunk = rows_to_draw[page_idx * slices_per_page : (page_idx + 1) * slices_per_page]
        n_rows = len(chunk)
        fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.1 * n_rows), squeeze=False)

        fig.suptitle(
            f"{pid} [{patient_summary['split_status']}] — So sánh Predict vs GT (Trang {page_idx + 1}/{n_pages})\n"
            f"3D Dice: {patient_summary['dice_3d_post_pct']:.1f}% | "
            f"Dice trên các lát Predict: {patient_summary['pooled_dice_on_pred_slices_pct']:.1f}% | "
            f"Nốt GT bắt được: {patient_summary['n_gt_detected']}/{patient_summary['n_gt_nodules']} | "
            f"Cụm FP: {patient_summary['n_pred_fp']}",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )

        for row_i, sr in enumerate(chunk):
            z = int(sr["z"])
            base = img_win_map[z]
            gt_2d = gt_mask_1mm[z] > 0
            pr_2d = clean_mask_1mm[z] > 0
            prob_2d = prob_1mm[z]

            # Cột 1: CT Lung Window + biên phổi nhẹ
            ax0 = axes[row_i, 0]
            ax0.imshow(base, cmap="gray", vmin=0.0, vmax=1.0)
            if lung_mask_1mm is not None and lung_mask_1mm[z].any():
                ax0.contour(lung_mask_1mm[z] > 0, levels=[0.5], colors=["#00bcd4"], linewidths=0.7, alpha=0.55)
            ax0.set_title(f"Lát z={z}/{num_slices}\n[{sr['status']}]", fontsize=10, fontweight="bold")
            ax0.axis("off")

            # Cột 2: Ground Truth Mask
            ax1 = axes[row_i, 1]
            ax1.imshow(base, cmap="gray", vmin=0.0, vmax=1.0)
            if gt_2d.any():
                gt_ov = np.ma.masked_where(~gt_2d, np.ones_like(base))
                ax1.imshow(gt_ov, cmap="Greens", vmin=0.0, vmax=1.5, alpha=0.65)
                ax1.contour(gt_2d, levels=[0.5], colors=["#00ff00"], linewidths=1.3)
            ax1.set_title(
                f"Ground Truth (Thật)\nArea={sr['gt_area_mm2']:.1f} mm² | d={sr['gt_diam_mm']:.1f} mm",
                fontsize=10,
                color="#1b5e20" if gt_2d.any() else "black",
            )
            ax1.axis("off")

            # Cột 3: Predict Mask + Heatmap xác suất
            ax2 = axes[row_i, 2]
            ax2.imshow(base, cmap="gray", vmin=0.0, vmax=1.0)
            prob_show = np.ma.masked_where(prob_2d < max(0.15, threshold * 0.5), prob_2d)
            ax2.imshow(prob_show, cmap="hot", vmin=0.0, vmax=1.0, alpha=0.55)
            if pr_2d.any():
                ax2.contour(pr_2d, levels=[0.5], colors=["#00e5ff"], linewidths=1.3)
            ax2.set_title(
                f"Predict (Th>={threshold})\nArea={sr['pred_area_mm2']:.1f} mm² | d={sr['pred_diam_mm']:.1f} mm | Pmax={sr['max_prob']:.2f}",
                fontsize=10,
                color="#b71c1c" if pr_2d.any() else "black",
            )
            ax2.axis("off")

            # Cột 4: Zoom ROI Overlay TP / FP / FN
            ax3 = axes[row_i, 3]
            ry0, ry1, rx0, rx1 = _compute_zoom_bbox(gt_2d, pr_2d, h, w)
            crop_base = base[ry0:ry1, rx0:rx1]
            crop_gt = gt_2d[ry0:ry1, rx0:rx1]
            crop_pr = pr_2d[ry0:ry1, rx0:rx1]

            # Tạo ảnh RGB blend trực quan: CT xám + TP (Xanh lá) + FP (Đỏ) + FN (Xanh dương)
            rgb = np.stack([crop_base, crop_base, crop_base], axis=-1).copy()
            tp_c = crop_gt & crop_pr
            fp_c = (~crop_gt) & crop_pr
            fn_c = crop_gt & (~crop_pr)

            alpha = 0.58
            rgb[tp_c] = (1 - alpha) * rgb[tp_c] + alpha * np.array([0.0, 0.95, 0.25])
            rgb[fp_c] = (1 - alpha) * rgb[fp_c] + alpha * np.array([1.0, 0.15, 0.15])
            rgb[fn_c] = (1 - alpha) * rgb[fn_c] + alpha * np.array([0.15, 0.55, 1.0])
            ax3.imshow(np.clip(rgb, 0.0, 1.0))
            if crop_gt.any():
                ax3.contour(crop_gt, levels=[0.5], colors=["lime"], linewidths=1.1, linestyles="-")
            if crop_pr.any():
                ax3.contour(crop_pr, levels=[0.5], colors=["red"], linewidths=1.1, linestyles="--")

            legend_handles = [
                Patch(facecolor="#00f240", label="TP (Đúng)"),
                Patch(facecolor="#ff2626", label="FP (Thừa)"),
                Patch(facecolor="#268cff", label="FN (Sót)"),
            ]
            ax3.legend(handles=legend_handles, loc="lower right", fontsize=7.5, framealpha=0.82)
            ax3.set_title(
                f"Zoom ROI So sánh | Chuẩn Dice={sr['slice_dice_pct']:.1f}% | IoU={sr['slice_iou_pct']:.1f}%",
                fontsize=10,
                fontweight="bold",
            )
            ax3.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        page_path = figures_dir / f"{pid}_slices_p{page_idx + 1:02d}.png"
        fig.savefig(page_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(page_path)

    # Vẽ ảnh tổng quan 3D MIP + Đồ thị diện tích/Dice dọc trục Z
    overview_path = _render_3d_overview(
        pid=pid,
        img_1mm=img_1mm,
        clean_mask_1mm=clean_mask_1mm,
        gt_mask_1mm=gt_mask_1mm,
        slice_rows=slice_rows,
        patient_summary=patient_summary,
        out_path=figures_dir / f"{pid}_3d_overview.png",
    )
    saved_paths.append(overview_path)
    return saved_paths


def _render_3d_overview(
    pid: str,
    img_1mm: np.ndarray,
    clean_mask_1mm: np.ndarray,
    gt_mask_1mm: np.ndarray,
    slice_rows: List[Dict[str, Any]],
    patient_summary: Dict[str, Any],
    out_path: Path,
) -> Path:
    """Vẽ tổng quan 3 mặt phẳng MIP (Axial, Coronal, Sagittal) và đồ thị hồ sơ lát Z."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    img_win = clip_and_window_hu(np.asarray(img_1mm))
    gt_b = gt_mask_1mm > 0
    pr_b = clean_mask_1mm > 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"Tổng quan 3D: {pid} ({patient_summary['split_status']}) — {patient_summary['case_outcome']}\n"
        f"3D Dice={patient_summary['dice_3d_post_pct']:.1f}% | IoU={patient_summary['iou_3d_post_pct']:.1f}% | "
        f"Vol GT={patient_summary['gt_vol_mm3']:.1f} mm³ vs Pred={patient_summary['pred_vol_mm3']:.1f} mm³ | "
        f"HD95={patient_summary['hd95_mm']} mm",
        fontsize=12.5,
        fontweight="bold",
    )

    views = [
        ("Chiếu Axial MIP (X-Y)", 0, (0, 0)),
        ("Chiếu Coronal MIP (Z-X)", 1, (0, 1)),
        ("Chiếu Sagittal MIP (Z-Y)", 2, (1, 0)),
    ]
    for title, axis, pos in views:
        ax = axes[pos]
        bg = img_win.mean(axis=axis)
        g_proj = gt_b.any(axis=axis)
        p_proj = pr_b.any(axis=axis)
        norm_bg = (bg - bg.min()) / max(1e-5, (bg.max() - bg.min()))
        rgb = np.stack([norm_bg, norm_bg, norm_bg], axis=-1)
        tp = g_proj & p_proj
        fp = (~g_proj) & p_proj
        fn = g_proj & (~p_proj)
        rgb[tp] = [0.0, 0.95, 0.25]
        rgb[fp] = [1.0, 0.15, 0.15]
        rgb[fn] = [0.15, 0.55, 1.0]
        ax.imshow(rgb, aspect="auto" if axis > 0 else "equal", origin="lower" if axis > 0 else "upper")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(
            handles=[
                Patch(facecolor="#00f240", label="TP (Khớp)"),
                Patch(facecolor="#ff2626", label="FP (Thừa)"),
                Patch(facecolor="#268cff", label="FN (Sót)"),
            ],
            loc="lower right",
            fontsize=8,
            framealpha=0.8,
        )
        ax.axis("off")

    # Panel (1, 1): Đồ thị diện tích & Dice theo lát Z
    ax_p = axes[1, 1]
    z_all = np.arange(gt_b.shape[0])
    gt_areas = gt_b.reshape(gt_b.shape[0], -1).sum(axis=1)
    pr_areas = pr_b.reshape(pr_b.shape[0], -1).sum(axis=1)
    ax_p.plot(z_all, gt_areas, color="#2e7d32", linewidth=2.0, label="Diện tích GT (mm²)")
    ax_p.plot(z_all, pr_areas, color="#d32f2f", linewidth=1.8, linestyle="--", label="Diện tích Predict (mm²)")
    ax_p.fill_between(z_all, gt_areas, alpha=0.18, color="#2e7d32")
    ax_p.fill_between(z_all, pr_areas, alpha=0.15, color="#d32f2f")
    ax_p.set_title("Phân bố diện tích u & độ khớp dọc theo trục Z", fontsize=11, fontweight="bold")
    ax_p.set_xlabel("Chỉ số lát cắt (z)")
    ax_p.set_ylabel("Diện tích (mm²)")
    ax_p.grid(True, linestyle="--", alpha=0.5)
    ax_p.legend(loc="upper left")

    if slice_rows:
        ax_d = ax_p.twinx()
        sz = [r["z"] for r in slice_rows]
        sd = [r["slice_dice_pct"] for r in slice_rows]
        ax_d.scatter(sz, sd, color="#1565c0", s=24, zorder=5, label="Slice Dice (%)")
        ax_d.set_ylabel("Slice Dice (%)", color="#1565c0")
        ax_d.set_ylim(-5, 108)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ==============================================================================
# 6. CHẠY PREDICT END-TO-END, IN BẢNG CHI TIẾT & XUẤT BÁO CÁO
# ==============================================================================
def run_predict_pipeline(
    model: torch.nn.Module,
    cache_dir: Path,
    patient_ids: Sequence[str],
    status_map: Dict[str, str],
    meta_info: Dict[str, Any],
    criteria: FilterCriteria,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 8,
    tta: bool = False,
    use_amp: bool = True,
    show_progress: bool = True,
    render_figures: bool = True,
    slices_per_page: int = 6,
    max_vis_slices: int = 0,
    drive_out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Thực thi predict trên các bệnh nhân đã chọn, in thông số, vẽ mask & lưu báo cáo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    tqdm_cls = _get_tqdm()

    patient_summaries: List[Dict[str, Any]] = []
    all_nodules: List[Dict[str, Any]] = []
    all_slices: List[Dict[str, Any]] = []
    patient_figures: Dict[str, List[str]] = {}

    print("=" * 95)
    print(
        f"🔬 BẮT ĐẦU PREDICT MẪU THẬT | Mô hình: {meta_info['run_id']} ({meta_info['arch']}/{meta_info['backbone']})\n"
        f"📐 Image size: {meta_info['image_size']} | Thiết bị: {device} | TTA: {tta} | Số ca: {len(patient_ids)}\n"
        f"🎯 Tiêu chí lọc: Conf >= {criteria.threshold} | MaxCompProb >= {criteria.min_comp_max_prob} | "
        f"Diam >= {criteria.min_diameter_mm} mm | Area >= {criteria.min_area_mm2} mm² | "
        f"Vol >= {criteria.min_volume_mm3} mm³ | Span >= {criteria.min_slices_span} | Scope={criteria.filter_scope}"
    )
    print("=" * 95)

    p_iter = (
        tqdm_cls(patient_ids, desc="🩻 Predict từng bệnh nhân", dynamic_ncols=True)
        if show_progress
        else patient_ids
    )

    for idx, pid in enumerate(p_iter, start=1):
        p_cache = cache_dir / pid
        t0 = time.time()
        inf_res = predict_patient_volume(
            model=model,
            patient_cache_dir=p_cache,
            device=device,
            batch_size=batch_size,
            image_size=meta_info["image_size"],
            apply_lung=True,
            tta=tta,
            use_amp=use_amp,
            compute_post_and_metrics=False,
            show_progress=show_progress,
            progress_desc=f"   ↳ Lát cắt {pid}",
        )
        prob_1mm = inf_res["prob_1mm"]
        gt_1mm = inf_res["gt_mask_1mm"]
        meta = inf_res["meta"]
        spacing_z = float(meta.get("orig_spacing_xyz", [1.0, 1.0, 1.0])[2])

        lung_p = lung_cache_path(p_cache)
        lung_1mm = np.asarray(np.load(lung_p, mmap_mode="r")) if lung_p.exists() else None

        filt_res = filter_and_analyze_components(
            prob_1mm=prob_1mm,
            gt_1mm=gt_1mm,
            lung_1mm=lung_1mm,
            spacing_z=spacing_z,
            criteria=criteria,
        )
        elapsed = time.time() - t0

        p_sum, s_rows = compute_slice_and_patient_stats(
            pid=pid,
            split_status=status_map.get(pid, "UNKNOWN"),
            prob_1mm=prob_1mm,
            raw_mask_1mm=filt_res["raw_mask_1mm"],
            clean_mask_1mm=filt_res["clean_mask_1mm"],
            gt_mask_1mm=gt_1mm,
            gt_nodules=filt_res["gt_nodules"],
            pred_components=filt_res["pred_components"],
            meta=meta,
            elapsed_sec=elapsed,
        )
        patient_summaries.append(p_sum)
        for row in filt_res["gt_nodules"] + filt_res["pred_components"]:
            all_nodules.append({"patient_id": pid, "split_status": p_sum["split_status"], **row})
        all_slices.extend(s_rows)

        if render_figures:
            paths = render_patient_visualizations(
                pid=pid,
                patient_cache_dir=p_cache,
                prob_1mm=prob_1mm,
                clean_mask_1mm=filt_res["clean_mask_1mm"],
                gt_mask_1mm=gt_1mm,
                lung_mask_1mm=filt_res["keep_lung"],
                patient_summary=p_sum,
                slice_rows=s_rows,
                threshold=criteria.threshold,
                figures_dir=fig_dir,
                slices_per_page=slices_per_page,
                max_vis_slices=max_vis_slices,
            )
            patient_figures[pid] = [str(p) for p in paths]

        # In thông tin chi tiết ngay cho từng ca bệnh
        print(
            f"\n[{idx:02d}/{len(patient_ids):02d}] {pid} [{p_sum['split_status']}] | {p_sum['case_outcome']} ({elapsed:.1f}s)\n"
            f"   • Khớp 3D: Dice={p_sum['dice_3d_post_pct']:.2f}% (Raw={p_sum['dice_3d_raw_pct']:.2f}%) | "
            f"IoU={p_sum['iou_3d_post_pct']:.2f}% | Prec={p_sum['precision_3d_pct']:.2f}% | Rec={p_sum['recall_3d_pct']:.2f}%\n"
            f"   • Thể tích & Biên: GT={p_sum['gt_vol_mm3']:.1f} mm³ vs Pred={p_sum['pred_vol_mm3']:.1f} mm³ "
            f"(Sim={p_sum['vol_similarity_pct']:.1f}%) | HD95={p_sum['hd95_mm']} mm | ASSD={p_sum['assd_mm']} mm\n"
            f"   • Đối chiếu lát cắt: GT={p_sum['n_gt_slices']} lát | Pred={p_sum['n_pred_slices']} lát | "
            f"Trúng={p_sum['n_pred_hit_slices']} lát | Chuẩn gộp trên các lát Predict={p_sum['pooled_dice_on_pred_slices_pct']:.2f}%\n"
            f"   • Nốt 3D: GT bắt được {p_sum['n_gt_detected']}/{p_sum['n_gt_nodules']} ({p_sum['nodule_sens_pct']:.1f}%) | "
            f"Pred giữ lại={p_sum['n_pred_kept']} (TP={p_sum['n_pred_tp']}, FP={p_sum['n_pred_fp']}) | "
            f"Đã lọc bỏ theo ngưỡng={p_sum['n_pred_filtered_out']}"
        )

    df_pat = pd.DataFrame(patient_summaries)
    df_nod = pd.DataFrame(all_nodules)
    df_slc = pd.DataFrame(all_slices)

    pat_csv = out_dir / "predict_patients_summary.csv"
    nod_csv = out_dir / "predict_nodules_detail.csv"
    slc_csv = out_dir / "predict_slices_detail.csv"
    df_pat.to_csv(pat_csv, index=False)
    df_nod.to_csv(nod_csv, index=False)
    df_slc.to_csv(slc_csv, index=False)

    overall = _compute_overall_report(df_pat, df_nod, df_slc, meta_info, criteria)
    report_json = out_dir / "predict_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": overall,
                "patients": patient_summaries,
                "figures": patient_figures,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    report_md = out_dir / "PREDICT_REPORT.md"
    _write_markdown_report(report_md, overall, df_pat, df_nod, df_slc, meta_info, criteria)

    _print_console_summary(overall, df_pat, df_nod)
    print(
        f"\n💾 ĐÃ LƯU TOÀN BỘ KẾT QUẢ TẠI: {out_dir}\n"
        f"   - Bảng bệnh nhân : {pat_csv.name}\n"
        f"   - Bảng khối u 3D : {nod_csv.name}\n"
        f"   - Bảng lát cắt   : {slc_csv.name}\n"
        f"   - Báo cáo MD/JSON: {report_md.name}, {report_json.name}\n"
        f"   - Ảnh mask PNG   : {fig_dir}"
    )

    if drive_out_dir:
        drv = Path(drive_out_dir)
        if drv.resolve() != out_dir.resolve():
            drv.mkdir(parents=True, exist_ok=True)
            all_out_files = [f for f in out_dir.rglob("*") if f.is_file()]
            for f in tqdm_cls(all_out_files, desc="☁️ Đồng bộ kết quả sang Drive", dynamic_ncols=True):
                rel = f.relative_to(out_dir)
                target = drv / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
            print(f"✅ Đã sao lưu toàn bộ kết quả sang Google Drive: {drv}")

    return {
        "out_dir": str(out_dir),
        "overall": overall,
        "patients_df": df_pat,
        "nodules_df": df_nod,
        "slices_df": df_slc,
        "figures": patient_figures,
    }


def _compute_overall_report(
    df_pat: pd.DataFrame,
    df_nod: pd.DataFrame,
    df_slc: pd.DataFrame,
    meta_info: Dict[str, Any],
    criteria: FilterCriteria,
) -> Dict[str, Any]:
    """Tổng hợp số liệu trung bình trên toàn bộ đợt Predict."""
    pos_pats = df_pat[df_pat["n_gt_nodules"] > 0]
    neg_pats = df_pat[df_pat["n_gt_nodules"] == 0]
    valid_hd95 = [v for v in df_pat["hd95_mm"] if np.isfinite(v)]
    valid_assd = [v for v in df_pat["assd_mm"] if np.isfinite(v)]

    gt_nod_df = df_nod[df_nod["comp_type"] == "GT"] if len(df_nod) else pd.DataFrame()
    pred_kept_df = df_nod[(df_nod["comp_type"] == "PRED") & (df_nod["kept"] == True)] if len(df_nod) else pd.DataFrame()
    pred_rej_df = df_nod[(df_nod["comp_type"] == "PRED") & (df_nod["kept"] == False)] if len(df_nod) else pd.DataFrame()

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": meta_info,
        "criteria": dataclasses.asdict(criteria),
        "total_patients": int(len(df_pat)),
        "pos_patients": int(len(pos_pats)),
        "neg_patients": int(len(neg_pats)),
        "clean_neg_patients": int((neg_pats["n_pred_fp"] == 0).sum()) if len(neg_pats) else 0,
        "mean_3d_dice_all_pct": round(float(df_pat["dice_3d_post_pct"].mean()), 2),
        "mean_3d_dice_pos_pct": round(float(pos_pats["dice_3d_post_pct"].mean()), 2) if len(pos_pats) else None,
        "mean_3d_iou_all_pct": round(float(df_pat["iou_3d_post_pct"].mean()), 2),
        "mean_pooled_pred_slices_dice_pct": round(float(df_pat["pooled_dice_on_pred_slices_pct"].mean()), 2),
        "mean_hd95_mm": round(float(np.mean(valid_hd95)), 3) if valid_hd95 else None,
        "mean_assd_mm": round(float(np.mean(valid_assd)), 3) if valid_assd else None,
        "total_gt_nodules": int(len(gt_nod_df)),
        "total_gt_detected": int((gt_nod_df["overlap_gt_voxels"] > 0).sum()) if len(gt_nod_df) else 0,
        "total_pred_kept": int(len(pred_kept_df)),
        "total_pred_fp": int((pred_kept_df["matched_gt_ids"] == "-").sum()) if len(pred_kept_df) else 0,
        "total_pred_filtered_out": int(len(pred_rej_df)),
        "total_compare_slices": int(len(df_slc)),
    }


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Chuyển DataFrame thành bảng GFM Markdown an toàn kể cả khi chưa cài tabulate."""
    if df is None or len(df) == 0:
        return "_(Không có dữ liệu)_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = [str(c) for c in df.columns]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows_md = [
            "| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |"
            for row in df.itertuples(index=False, name=None)
        ]
        return "\n".join([header, sep, *rows_md])


def _write_markdown_report(
    out_path: Path,
    overall: Dict[str, Any],
    df_pat: pd.DataFrame,
    df_nod: pd.DataFrame,
    df_slc: pd.DataFrame,
    meta_info: Dict[str, Any],
    criteria: FilterCriteria,
) -> None:
    """Ghi báo cáo Markdown đầy đủ về đợt Predict mẫu thật."""
    lines = [
        f"# 🩻 BÁO CÁO KẾT QUẢ PREDICT MẪU THẬT ({meta_info['run_id']})",
        "",
        f"- **Thời gian**: {overall['timestamp']}",
        f"- **Kiến trúc**: `{meta_info['arch']}` | Backbone: `{meta_info['backbone']}` | Image Size: `{meta_info['image_size']}` | EMA: `{meta_info['is_ema']}`",
        f"- **Tiêu chí lọc ngưỡng**: Confidence `>= {criteria.threshold}`, MaxCompProb `>= {criteria.min_comp_max_prob}`, "
        f"Đường kính `>= {criteria.min_diameter_mm} mm`, Diện tích `>= {criteria.min_area_mm2} mm²`, "
        f"Thể tích `>= {criteria.min_volume_mm3} mm³`, Span `>= {criteria.min_slices_span}`, Scope=`{criteria.filter_scope}`",
        "",
        "## 1. Tổng hợp toàn đợt Predict",
        f"- **Số bệnh nhân**: {overall['total_patients']} ({overall['pos_patients']} ca có u thật | {overall['neg_patients']} ca không u)",
        f"- **Mean 3D Dice (Tất cả ca)**: `{overall['mean_3d_dice_all_pct']}%` | **Mean 3D Dice (Ca có u)**: `{overall['mean_3d_dice_pos_pct']}%`",
        f"- **Mean Dice trên tập các lát ĐƯỢC PREDICT**: `{overall['mean_pooled_pred_slices_dice_pct']}%`",
        f"- **Độ nhạy mức nốt u 3D**: `{overall['total_gt_detected']}/{overall['total_gt_nodules']}` | "
        f"**Cụm FP giữ lại**: `{overall['total_pred_fp']}` | **Cụm đã lọc bỏ**: `{overall['total_pred_filtered_out']}`",
        f"- **Mean HD95**: `{overall['mean_hd95_mm']} mm` | **Mean ASSD**: `{overall['mean_assd_mm']} mm`",
        "",
        "## 2. Bảng tổng hợp theo từng bệnh nhân",
        _df_to_markdown(df_pat),
        "",
    ]
    if len(df_nod):
        lines.extend([
            "## 3. Phân tích từng nốt u 3D (GT thật vs Cụm Predict)",
            _df_to_markdown(df_nod),
            "",
        ])
    if len(df_slc):
        lines.extend([
            "## 4. Chi tiết đối chiếu từng lát cắt (50 dòng đầu)",
            _df_to_markdown(df_slc.head(50)),
            "",
        ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _print_console_summary(overall: Dict[str, Any], df_pat: pd.DataFrame, df_nod: pd.DataFrame) -> None:
    """In tóm tắt tổng hợp toàn đợt ra console."""
    print("\n" + "=" * 95)
    print("📊 TỔNG HỢP TOÀN ĐỢT PREDICT MẪU THẬT")
    print("=" * 95)
    print(
        f"  • Tổng số ca: {overall['total_patients']} ({overall['pos_patients']} có u | {overall['neg_patients']} sạch, sạch hoàn toàn {overall['clean_neg_patients']} ca)\n"
        f"  • Mean 3D Dice (Tất cả / Có u): {overall['mean_3d_dice_all_pct']}% / {overall['mean_3d_dice_pos_pct']}%\n"
        f"  • Mean Chuẩn Dice trên các lát Predict: {overall['mean_pooled_pred_slices_dice_pct']}%\n"
        f"  • Nốt 3D: Bắt được {overall['total_gt_detected']}/{overall['total_gt_nodules']} | "
        f"Cụm FP: {overall['total_pred_fp']} | Đã lọc bỏ bởi ngưỡng kích thước/độ tin cậy: {overall['total_pred_filtered_out']}\n"
        f"  • HD95 TB: {overall['mean_hd95_mm']} mm | ASSD TB: {overall['mean_assd_mm']} mm"
    )
    if len(df_nod):
        gt_df = df_nod[df_nod["comp_type"] == "GT"]
        if len(gt_df):
            print("\n🎯 ĐỘ NHẠY PHÁT HIỆN NỐT GT THEO KÍCH THƯỚC (MM):")
            for _, _, lbl in NODULE_BINS:
                sub = gt_df[gt_df["size_bin"] == lbl]
                if len(sub):
                    hits = int((sub["overlap_gt_voxels"] > 0).sum())
                    print(f"  - {lbl:<22}: {hits}/{len(sub)} ({100.0 * hits / len(sub):.1f}%)")


# ==============================================================================
# 7. CLI SUBCOMMANDS & SELFTEST
# ==============================================================================
def _resolve_prepare_data_source(data_source: str, drive_sources_dir: Optional[str]) -> Path:
    """Tự động dò thư mục cache_1mm hoặc cache_1mm.zip từ Drive hoặc local."""
    if data_source != "auto":
        p = Path(data_source)
        if not p.exists():
            raise FileNotFoundError(f"❌ Không thấy --data-source: {p}")
        return p

    cands: List[Path] = []
    if drive_sources_dir:
        d = Path(drive_sources_dir)
        cands.extend([d / "data" / "cache_1mm", d / "data" / "cache_1mm.zip", d / "data" / "KAGGLE_UPLOAD" / "cache_1mm.zip"])
    cands.extend([
        Path("/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/data/cache_1mm"),
        Path("/content/drive/MyDrive/GG_1/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/data/cache_1mm"),
        Path("/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/data/cache_1mm.zip"),
        _ROOT / "data" / "cache_1mm",
    ])
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError("❌ Không tìm thấy nguồn cache_1mm (thư mục hoặc .zip). Hãy truyền --data-source cụ thể.")


def cmd_prepare(args) -> None:
    """Lệnh CLI prepare: nạp model + slices.csv + copy đúng N ca chưa train về workspace."""
    artifacts = find_run_artifacts(
        checkpoint_or_run_dir=args.checkpoint or args.run_dir,
        drive_sources_dir=args.drive_sources_dir,
        run_preset=args.run_preset,
        ckpt_name=args.ckpt_name,
        arch=args.arch,
    )
    run_name = artifacts["run_dir"].name
    model_dest = Path(args.model_dest) if args.model_dest else (_ROOT / args.arch / "runs" / run_name)
    staged_model = stage_model_to_workspace(artifacts, model_dest, overwrite=args.overwrite_model)

    slices_csv = Path(args.slices_csv) if args.slices_csv else staged_model["slices_csv"]
    split_info = discover_run_splits(slices_csv, staged_model["run_config"])
    print(
        f"📄 Đối chiếu phân chia đợt train ({split_info['slices_csv'] or 'KHÔNG CÓ'}): "
        f"Train={len(split_info['train_ids'])} ca | Val={len(split_info['val_ids'])} ca "
        f"(Val có u: {len(split_info['val_tumor_ids'])}, Val sạch: {len(split_info['val_neg_ids'])})"
    )

    src_path = _resolve_prepare_data_source(args.data_source, args.drive_sources_dir)
    src_kind, avail_ids = scan_available_patients(src_path)
    print(f"🔎 Nguồn dữ liệu ({src_kind}): {src_path} | Tổng số ca có sẵn: {len(avail_ids)}")

    hints = _load_tumor_hint_map(src_path, split_info)
    selected_ids, status_map = select_non_train_patients(
        available_ids=avail_ids,
        split_info=split_info,
        source_mode=args.source,
        selection_mode=args.select,
        num_samples=args.num_samples,
        seed=args.seed,
        custom_patient_ids=args.patient_ids,
        allow_train_leak=args.allow_train_leak,
        tumor_hints=hints,
    )
    print(f"✅ Đã lọc chọn {len(selected_ids)} ca (mode={args.source}, select={args.select}):")
    for pid in selected_ids:
        t_hint = "Có u GT" if hints.get(pid) is True else ("Sạch (0 u)" if hints.get(pid) is False else "Chưa rõ")
        print(f"   - {pid} | Trạng thái đợt train: {status_map[pid]} | Gợi ý nhãn: {t_hint}")

    dest_cache = stage_patient_samples(
        source_kind=src_kind,
        source_path=src_path,
        selected_ids=selected_ids,
        dest_cache_dir=args.dest_cache,
        local_fast_cache_dir=args.local_fast_cache,
        workers=args.workers,
    )

    manifest = {
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_checkpoint": str(staged_model["checkpoint"]),
        "run_config": str(staged_model["run_config"]) if staged_model["run_config"] else None,
        "run_slices_csv": str(slices_csv) if slices_csv else None,
        "cache_dir": str(dest_cache),
        "source_mode": args.source,
        "selection_mode": args.select,
        "selected_ids": selected_ids,
        "status_map": status_map,
        "patient_ids_sha256": compute_patient_ids_hash(selected_ids),
    }
    manifest_path = dest_cache / "predict_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n🎯 ĐÃ SẴN SÀNG! Manifest lưu tại: {manifest_path}")


def cmd_predict(args) -> None:
    """Lệnh CLI predict: chạy suy luận theo ngưỡng và xuất báo cáo + hình ảnh so sánh."""
    manifest: Dict[str, Any] = {}
    if args.manifest and Path(args.manifest).exists():
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    elif args.cache_dir and (Path(args.cache_dir) / "predict_manifest.json").exists():
        manifest = json.loads((Path(args.cache_dir) / "predict_manifest.json").read_text(encoding="utf-8"))

    checkpoint = args.checkpoint or manifest.get("model_checkpoint")
    cache_dir_str = args.cache_dir or manifest.get("cache_dir")
    if not checkpoint or not cache_dir_str:
        raise argparse.ArgumentError(None, "❌ Cần truyền --manifest hoặc đầy đủ (--checkpoint và --cache-dir).")

    cache_dir = Path(cache_dir_str)
    ckpt_p = Path(checkpoint)
    slices_csv = args.slices_csv or manifest.get("run_slices_csv")
    if not slices_csv:
        auto_csv = _find_slices_csv_in_dir(ckpt_p.parent)
        slices_csv = str(auto_csv) if auto_csv else None

    split_info = discover_run_splits(slices_csv, manifest.get("run_config") or (ckpt_p.parent / "run_config.json"))
    _, avail_ids = scan_available_patients(cache_dir)

    if args.patient_ids:
        target_ids = list(args.patient_ids)
    elif manifest.get("selected_ids"):
        target_ids = [pid for pid in manifest["selected_ids"] if pid in avail_ids]
    else:
        target_ids = avail_ids

    train_set = set(split_info["train_ids"])
    val_set = set(split_info["val_ids"])
    status_map: Dict[str, str] = {}
    filtered_ids: List[str] = []
    for pid in target_ids:
        if pid in train_set:
            if not args.allow_train_leak and args.patient_ids:
                raise ValueError(
                    f"🚫 {pid} nằm trong tập TRAIN của đợt train này! Thêm --allow-train-leak nếu cố tình kiểm tra."
                )
            elif not args.allow_train_leak:
                print(f"⏭️ Tự động bỏ qua {pid} vì nằm trong tập TRAIN (tránh rò rỉ dữ liệu).")
                continue
            status_map[pid] = "TRAIN (⚠️ RÒ RỈ)"
        elif pid in val_set:
            status_map[pid] = "VAL (Chưa train)"
        else:
            status_map[pid] = "UNSEEN (Mới 100%)"
        filtered_ids.append(pid)

    if args.max_patients > 0:
        filtered_ids = filtered_ids[: args.max_patients]
    if not filtered_ids:
        raise RuntimeError(f"❌ Không có bệnh nhân hợp lệ nào trong {cache_dir} để predict.")

    meta_info = resolve_model_metadata(
        checkpoint_path=ckpt_p,
        run_config_path=manifest.get("run_config"),
        explicit_arch=args.arch,
        explicit_backbone=args.backbone,
        explicit_image_size=args.image_size,
    )
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    model = load_model_for_predict(ckpt_p, device=device, meta_info=meta_info)

    criteria = FilterCriteria(
        threshold=float(args.threshold),
        min_comp_max_prob=float(args.min_comp_max_prob),
        min_diameter_mm=float(args.min_diameter_mm),
        min_area_mm2=float(args.min_area_mm2),
        min_volume_mm3=float(args.min_vol_mm3),
        min_slices_span=int(args.min_slices_span),
        filter_scope=args.filter_scope,
        fill_holes=not args.no_fill_holes,
    )

    out_dir = Path(args.out_dir) if args.out_dir else (_ROOT / "eval_results" / "predict_samples" / meta_info["run_id"])
    run_predict_pipeline(
        model=model,
        cache_dir=cache_dir,
        patient_ids=filtered_ids,
        status_map=status_map,
        meta_info=meta_info,
        criteria=criteria,
        out_dir=out_dir,
        device=device,
        batch_size=args.batch_size,
        tta=args.tta,
        use_amp=not args.no_amp,
        show_progress=not args.no_progress,
        render_figures=not args.no_figures,
        slices_per_page=args.slices_per_page,
        max_vis_slices=args.max_vis_slices,
        drive_out_dir=args.drive_out_dir,
    )


def _selftest() -> None:
    """Kiểm thử nhanh toàn bộ luồng prepare + predict + visualization trên CPU 2 nhân."""
    print("🧪 Bắt đầu selftest tools/predict_real_samples.py (CPU lightweight)...")
    tmp = Path(tempfile.mkdtemp())
    try:
        drive_src = tmp / "fake_drive"
        run_dir = drive_src / "mamba_unet" / "runs" / "tiny_run"
        cache_src = drive_src / "data" / "cache_1mm"
        run_dir.mkdir(parents=True)
        cache_src.mkdir(parents=True)

        # Dựng 3 bệnh nhân giả (64x64): ca 0001 (TRAIN), ca 0002 (VAL có u), ca 0003 (UNSEEN sạch)
        for pid, has_u in [("LIDC-IDRI-0001", True), ("LIDC-IDRI-0002", True), ("LIDC-IDRI-0003", False)]:
            p = cache_src / pid
            p.mkdir(parents=True)
            img = np.full((10, 64, 64), -800, dtype=np.int16)
            msk = np.zeros((10, 64, 64), dtype=np.uint8)
            lung = np.zeros((10, 64, 64), dtype=np.uint8)
            lung[1:9, 8:56, 8:56] = 1
            if has_u:
                msk[3:6, 26:34, 26:34] = 1  # 64 mm2 -> diam ~9.03 mm
                img[3:6, 26:34, 26:34] = 50
            np.save(p / "image_1mm.npy", img)
            np.save(p / "mask_1mm.npy", msk)
            np.save(p / "lung_1mm.npy", lung)
            (p / "meta.json").write_text(
                json.dumps({"orig_shape_zyx": [10, 64, 64], "orig_spacing_xyz": [0.8, 0.8, 1.5], "lung_ok": True}),
                encoding="utf-8",
            )

        pd.DataFrame([
            {"patient_id": "LIDC-IDRI-0001", "z": 4, "num_slices": 10, "has_tumor": 1, "tumor_area_mm2": 64.0, "has_lung_mask": 1, "split": "train"},
            {"patient_id": "LIDC-IDRI-0002", "z": 4, "num_slices": 10, "has_tumor": 1, "tumor_area_mm2": 64.0, "has_lung_mask": 1, "split": "val"},
        ]).to_csv(run_dir / "slices_test.csv", index=False)

        from mamba_unet.backbone.mamba_vision import MambaVision
        from mamba_unet.model import MambaUNet

        tiny_bb = MambaVision(dim=16, in_dim=8, depths=[1, 1, 2, 1], window_size=[8, 8, 14, 7], num_heads=[2, 2, 4, 4], in_chans=3)
        model = MambaUNet(in_channels=3, num_classes=1, backbone=tiny_bb, act="relu")
        ckpt_p = run_dir / "best_model.pth"
        torch.save({"model_state_dict": model.state_dict(), "meta": {"arch": "mamba_unet", "image_size": 64, "activation": "relu"}}, ckpt_p)
        (run_dir / "run_config.json").write_text(json.dumps({"run_id": "tiny_run", "arch": "mamba_unet", "model": {"image_size": 64}}), encoding="utf-8")

        # Test stage_model + select_non_train + stage_samples
        staged_dir = tmp / "ws_model"
        artifacts = find_run_artifacts(checkpoint_or_run_dir=run_dir)
        staged = stage_model_to_workspace(artifacts, staged_dir)
        split_info = discover_run_splits(staged["slices_csv"], staged["run_config"])
        assert split_info["train_ids"] == ["LIDC-IDRI-0001"]
        assert split_info["val_ids"] == ["LIDC-IDRI-0002"]

        kind, avail = scan_available_patients(cache_src)
        sel, st_map = select_non_train_patients(avail, split_info, source_mode="non_train", selection_mode="first", num_samples=5)
        assert sel == ["LIDC-IDRI-0002", "LIDC-IDRI-0003"]
        assert "LIDC-IDRI-0001" not in sel

        dest_cache = stage_patient_samples(kind, cache_src, sel, tmp / "pred_cache", workers=1)
        out_dir = tmp / "out"
        res = run_predict_pipeline(
            model=model,
            cache_dir=dest_cache,
            patient_ids=sel,
            status_map=st_map,
            meta_info={"arch": "mamba_unet", "backbone": "tiny", "activation": "relu", "image_size": 64, "is_ema": False, "run_id": "tiny_run"},
            criteria=FilterCriteria(threshold=0.5, min_diameter_mm=0.0, min_area_mm2=5.0),
            out_dir=out_dir,
            device=torch.device("cpu"),
            batch_size=2,
            use_amp=False,
            show_progress=False,
            render_figures=True,
        )
        assert (out_dir / "predict_patients_summary.csv").exists()
        assert (out_dir / "PREDICT_REPORT.md").exists()
        assert len(res["figures"]["LIDC-IDRI-0002"]) >= 2
        print("✅ Selftest tools/predict_real_samples.py hoàn tất thành công!")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Nạp model + chọn mẫu chưa train + Predict & so sánh trực quan GT vs Predict.")
    ap.add_argument("--selftest", action="store_true", help="Chạy kiểm thử nhanh toàn bộ pipeline trên CPU.")
    sub = ap.add_subparsers(dest="cmd")

    # 1. prepare
    p_prep = sub.add_parser("prepare", help="Nạp checkpoint + slices.csv + copy mẫu chưa train từ Drive về local SSD.")
    p_prep.add_argument("--drive-sources-dir", default=None, help="Thư mục gốc dự án trên Google Drive.")
    p_prep.add_argument("--run-preset", default=None, help=f"Tên preset run ({', '.join(RUN_PRESETS.keys())}, hoặc custom).")
    p_prep.add_argument("--run-dir", default=None, help="Đường dẫn thư mục run nguồn cụ thể.")
    p_prep.add_argument("--checkpoint", default=None, help="Đường dẫn file checkpoint cụ thể (.pth).")
    p_prep.add_argument("--ckpt-name", default="best_model.pth", help="Tên file checkpoint ưu tiên (mặc định best_model.pth).")
    p_prep.add_argument("--arch", default="mamba_unet", choices=["mamba_unet", "unet_base"])
    p_prep.add_argument("--model-dest", default=None, help="Thư mục đích chứa checkpoint & slices.csv trên local /content.")
    p_prep.add_argument("--overwrite-model", action="store_true")
    p_prep.add_argument("--slices-csv", default=None, help="Chỉ định file slices*.csv tùy chọn nếu không nằm trong run dir.")
    p_prep.add_argument("--data-source", default="auto", help="Thư mục cache_1mm hoặc file cache_1mm.zip nguồn.")
    p_prep.add_argument("--local-fast-cache", default=None, help="Cache SSD local có sẵn (nếu có) để chép tức thì.")
    p_prep.add_argument("--dest-cache", default="/content/data/predict_cache_1mm", help="Thư mục đích chứa các mẫu được chọn.")
    p_prep.add_argument("--source", default="val", choices=["val", "unseen", "non_train", "custom"], help="Nguồn mẫu chưa từng train.")
    p_prep.add_argument("--select", default="mixed", choices=["mixed", "tumor_first", "first", "random"], help="Chiến lược chọn N mẫu.")
    p_prep.add_argument("--num-samples", type=int, default=5, help="Số ca bệnh cần lấy (0 = toàn bộ ca hợp lệ).")
    p_prep.add_argument("--patient-ids", nargs="+", default=None, help="Danh sách ID bệnh nhân khi dùng --source custom.")
    p_prep.add_argument("--allow-train-leak", action="store_true", help="Cho phép predict cả ca train khi dùng custom.")
    p_prep.add_argument("--workers", type=int, default=4)
    p_prep.add_argument("--seed", type=int, default=42)
    p_prep.set_defaults(func=cmd_prepare)

    # 2. predict
    p_pred = sub.add_parser("predict", help="Chạy predict 3D theo tiêu chí ngưỡng, so sánh từng lát cắt & vẽ mask.")
    p_pred.add_argument("--manifest", default=None, help="Đường dẫn predict_manifest.json sinh ra từ bước prepare.")
    p_pred.add_argument("--checkpoint", default=None, help="Đường dẫn file checkpoint (.pth).")
    p_pred.add_argument("--cache-dir", default=None, help="Thư mục cache chứa các ca cần predict.")
    p_pred.add_argument("--slices-csv", default=None, help="File slices*.csv của đợt train để loại ca train.")
    p_pred.add_argument("--patient-ids", nargs="+", default=None, help="Chỉ định danh sách ID bệnh nhân cần predict.")
    p_pred.add_argument("--max-patients", type=int, default=0, help="Giới hạn số ca predict tối đa (0 = tất cả).")
    p_pred.add_argument("--allow-train-leak", action="store_true")
    p_pred.add_argument("--arch", default=None, choices=["mamba_unet", "unet_base"])
    p_pred.add_argument("--backbone", default=None)
    p_pred.add_argument("--image-size", type=int, default=0, help="0 = tự động đọc từ checkpoint / run_config.json.")
    p_pred.add_argument("--batch-size", type=int, default=8)
    p_pred.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng độ tin cậy pixel [0..1].")
    p_pred.add_argument("--min-comp-max-prob", type=float, default=0.0, help="Ngưỡng xác suất đỉnh tối thiểu của khối u.")
    p_pred.add_argument("--min-diameter-mm", type=float, default=0.0, help="Đường kính tối thiểu (mm) để giữ nốt (vd: 5.0 hoặc 10.0).")
    p_pred.add_argument("--min-area-mm2", type=float, default=5.0, help="Diện tích lát cắt tối thiểu (mm²).")
    p_pred.add_argument("--min-vol-mm3", type=float, default=0.0, help="Thể tích 3D tối thiểu (mm³).")
    p_pred.add_argument("--min-slices-span", type=int, default=1, help="Số lát cắt liên tiếp tối thiểu.")
    p_pred.add_argument("--filter-scope", default="3d_component", choices=["3d_component", "per_slice", "both"], help="Phạm vi áp ngưỡng kích thước.")
    p_pred.add_argument("--no-fill-holes", action="store_true")
    p_pred.add_argument("--tta", action="store_true", help="Bật Test-Time Augmentation (lật ngang/dọc).")
    p_pred.add_argument("--no-cuda", action="store_true")
    p_pred.add_argument("--no-amp", action="store_true")
    p_pred.add_argument("--no-progress", action="store_true")
    p_pred.add_argument("--no-figures", action="store_true")
    p_pred.add_argument("--slices-per-page", type=int, default=6)
    p_pred.add_argument("--max-vis-slices", type=int, default=0, help="0 = vẽ toàn bộ lát cắt được predict/GT.")
    p_pred.add_argument("--out-dir", default=None, help="Thư mục lưu CSV, Markdown và hình ảnh PNG.")
    p_pred.add_argument("--drive-out-dir", default=None, help="Thư mục Google Drive để đồng bộ toàn bộ kết quả sau khi xong.")
    p_pred.set_defaults(func=cmd_predict)

    return ap


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = build_argparser()
    args = parser.parse_args()
    if args.selftest:
        _selftest()
        return
    if not getattr(args, "cmd", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
