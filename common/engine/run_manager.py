# -*- coding: utf-8 -*-
"""
common/engine/run_manager.py - Quản lý cấu trúc thư mục Run độc lập cho từng lần huấn luyện.

Mỗi lần train được cấp một thư mục riêng: <runs-root>/<run_id>/
Bao gồm:
  - run_config.json: Tái lập 100% tham số, môi trường, git, kiến trúc
  - history.csv: Log metrics từng epoch
  - train.log: Ghi nhật ký đầy đủ
  - best_model.pth, last_model.pth, best_weights_fp16.pth
"""

import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.paths import default_runs_root
from common.engine.run_config import build_initial_run_config
from common.engine.utils import atomic_save_json


class RunHandle:
    """Quản lý vòng đời của một phiên chạy (Run)."""

    def __init__(self, run_dir: Path, config: Dict[str, Any]):
        self.run_dir = Path(run_dir).resolve()
        self.config = config
        self.run_id = config.get("run_id", self.run_dir.name)
        self.config_path = self.run_dir / "run_config.json"
        self.log_path = self.run_dir / "train.log"
        self.history_path = self.run_dir / "history.csv"
        self._start_time = time.time()

        # Tạo thư mục và ghi run_config.json ban đầu ngay lập tức (status="running")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

    def _save_config(self):
        atomic_save_json(self.config_path, self.config)

    def log(self, message: str, echo: bool = True):
        """Ghi thông điệp đồng thời ra stdout và train.log."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        if echo:
            try:
                print(formatted)
            except UnicodeEncodeError:
                encoding = sys.stdout.encoding or "utf-8"
                print(formatted.encode(encoding, errors="replace").decode(encoding, errors="replace"))
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
        except Exception:
            pass

    def record_resume_event(self, from_epoch: int):
        """Ghi nhận sự kiện tiếp tục huấn luyện (resume)."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        events = self.config.setdefault("resume_events", [])
        events.append({"at": now_iso, "from_epoch": from_epoch})
        self._save_config()

    def update_result(
        self,
        epoch: int,
        best_val_dice: Optional[float] = None,
        best_epoch: Optional[int] = None,
        last_val_loss: Optional[float] = None,
        last_val_dice: Optional[float] = None,
        last_val_iou: Optional[float] = None,
    ):
        """Cập nhật tiến độ sau mỗi epoch."""
        res = self.config.setdefault("result", {})
        res["epochs_completed"] = epoch
        if best_val_dice is not None:
            res["best_val_dice"] = float(best_val_dice)
        if best_epoch is not None:
            res["best_epoch"] = int(best_epoch)

        last = res.setdefault("last", {})
        if last_val_loss is not None:
            last["val_loss"] = float(last_val_loss)
        if last_val_dice is not None:
            last["val_dice"] = float(last_val_dice)
        if last_val_iou is not None:
            last["val_iou"] = float(last_val_iou)

        # Cập nhật thời lượng chạy
        elapsed = time.time() - self._start_time
        self.config["duration_sec"] = round(elapsed, 1)
        self._save_config()

    def finish(self, status: str = "completed", extra_metrics: Optional[Dict[str, Any]] = None):
        """Đóng phiên chạy và ghi nhận trạng thái cuối cùng."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.config["status"] = status
        self.config["finished_at"] = now_iso
        self.config["duration_sec"] = round(time.time() - self._start_time, 1)
        if extra_metrics:
            self.config.setdefault("result", {}).update(extra_metrics)
        self._save_config()
        self.log(f"[FINISH] Phiên chạy '{self.run_id}' kết thúc với trạng thái: {status.upper()}")


def generate_run_id(arch: str, variant: Optional[str] = None, tag: Optional[str] = None) -> str:
    """Sinh run_id dạng {YYYYmmdd-HHMMSS}_{arch}[_{variant}][_{tag}]."""
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [now_str, arch]
    if variant:
        parts.append(variant)
    if tag:
        parts.append(tag)
    return "_".join(parts)


def create_run(
    arch: str,
    runs_root: Optional[Union[str, Path]] = None,
    run_name: Optional[str] = None,
    tag: Optional[str] = None,
    variant: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    model_meta: Optional[Dict[str, Any]] = None,
    data_meta: Optional[Dict[str, Any]] = None,
    loss_meta: Optional[Dict[str, Any]] = None,
    optimizer_meta: Optional[Dict[str, Any]] = None,
    scheduler_meta: Optional[Dict[str, Any]] = None,
    training_meta: Optional[Dict[str, Any]] = None,
    notes: str = "",
    init_weights: Optional[str] = None,
) -> RunHandle:
    """Khởi tạo một Run mới hoàn chỉnh kèm run_config.json."""
    root = Path(runs_root).resolve() if runs_root else default_runs_root(arch)
    root.mkdir(parents=True, exist_ok=True)

    if run_name:
        run_id = run_name
    else:
        run_id = generate_run_id(arch, variant=variant, tag=tag)

    # Đảm bảo run_id là duy nhất (nếu trùng thêm -2, -3)
    target_dir = root / run_id
    counter = 2
    base_id = run_id
    while target_dir.exists():
        run_id = f"{base_id}-{counter}"
        target_dir = root / run_id
        counter += 1

    config = build_initial_run_config(
        run_id=run_id,
        arch=arch,
        args=args or {},
        model_meta=model_meta or {},
        data_meta=data_meta or {},
        loss_meta=loss_meta or {},
        optimizer_meta=optimizer_meta or {},
        scheduler_meta=scheduler_meta or {},
        training_meta=training_meta or {},
        notes=notes,
        init_weights=init_weights,
    )

    return RunHandle(target_dir, config)


def find_resumable(arch: str, runs_root: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Tìm run mới nhất của kiến trúc đó chưa hoàn thành (status != 'completed')."""
    root = Path(runs_root).resolve() if runs_root else default_runs_root(arch)
    if not root.exists():
        return None

    candidate_runs = []
    for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "run_config.json").exists() and (d / "last_model.pth").exists():
            try:
                with open(d / "run_config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if cfg.get("arch") == arch and cfg.get("status") != "completed":
                    candidate_runs.append(d)
            except Exception:
                pass

    return candidate_runs[0] if candidate_runs else None


def resolve_resume(
    resume_arg: Optional[str],
    arch: str,
    runs_root: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Phân giải tham số --resume:
      - 'auto': Tự tìm run dở dang mới nhất
      - Đường dẫn thư mục run: Nạp last_model.pth trong thư mục đó
      - Đường dẫn file checkpoint .pth cụ thể
    Trả về: (resolved_run_dir, resolved_checkpoint_path)
    """
    if not resume_arg:
        return None, None

    root = Path(runs_root).resolve() if runs_root else default_runs_root(arch)

    if resume_arg.lower() == "auto":
        resumable_dir = find_resumable(arch, root)
        if resumable_dir:
            return resumable_dir, resumable_dir / "last_model.pth"
        return None, None

    p = Path(resume_arg).resolve()
    if p.is_dir():
        ckpt = p / "last_model.pth"
        if not ckpt.exists():
            ckpt = p / "best_model.pth"
        return p, (ckpt if ckpt.exists() else None)

    if p.is_file():
        run_d = p.parent
        return run_d, p

    return None, None
