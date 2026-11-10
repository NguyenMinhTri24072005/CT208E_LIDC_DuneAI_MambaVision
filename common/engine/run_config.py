# -*- coding: utf-8 -*-
"""
common/engine/run_config.py - Lược đồ và hàm thu thập siêu dữ liệu thí nghiệm (run_config.json).

Quản lý thông tin:
  - Môi trường chạy (platform, python, torch, CUDA, GPU)
  - Thông tin Git (commit hash, branch, dirty)
  - Dữ liệu và manifest nguồn
  - Kết quả và sự kiện resume
"""

import datetime
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.engine.utils import atomic_save_json

SCHEMA_VERSION = 1


def get_git_info() -> Dict[str, Any]:
    """Lấy thông tin git commit hiện tại (nếu có)."""
    git_info = {"commit": "unknown", "branch": "unknown", "dirty": False, "remote": "unknown"}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_info["commit"] = commit
    except Exception:
        pass

    try:
        branch = subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_info["branch"] = branch
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "-C", str(_ROOT), "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_info["dirty"] = bool(status)
    except Exception:
        pass

    try:
        remote = subprocess.check_output(
            ["git", "-C", str(_ROOT), "config", "--get", "remote.origin.url"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        # Loại bỏ token nếu có trong URL remote để bảo mật tuyệt đối (R9)
        if "@" in remote:
            remote = "https://" + remote.split("@")[-1]
        git_info["remote"] = remote
    except Exception:
        pass

    return git_info


def get_environment_info(amp: bool = True, mamba_backend: Optional[str] = None) -> Dict[str, Any]:
    """Thu thập thông tin hệ thống và các gói thư viện phụ thuộc."""
    import torch

    if "google.colab" in sys.modules:
        plat = "colab"
    elif Path("/kaggle/working").exists():
        plat = "kaggle"
    else:
        plat = "local"

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    num_gpus = torch.cuda.device_count() if cuda_available else 0

    packages = {}
    for pkg in ("timm", "einops", "mamba_ssm"):
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg] = None

    return {
        "platform": plat,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda if cuda_available else None,
        "gpu": gpu_name,
        "num_gpus": num_gpus,
        "amp": bool(amp and cuda_available),
        "mamba_backend": mamba_backend or ("cuda" if (packages.get("mamba_ssm") and cuda_available) else "torch"),
        "packages": packages,
    }


def build_initial_run_config(
    run_id: str,
    arch: str,
    args: Dict[str, Any],
    model_meta: Dict[str, Any],
    data_meta: Dict[str, Any],
    loss_meta: Dict[str, Any],
    optimizer_meta: Dict[str, Any],
    scheduler_meta: Dict[str, Any],
    training_meta: Dict[str, Any],
    notes: str = "",
    init_weights: Optional[str] = None,
) -> Dict[str, Any]:
    """Khởi tạo tài liệu run_config.json theo schema v1."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "arch": arch,
        "status": "running",
        "notes": notes,
        "started_at": now_iso,
        "finished_at": None,
        "duration_sec": None,
        "resume_events": [],
        "init_weights": init_weights,
        "git": get_git_info(),
        "environment": get_environment_info(
            amp=not args.get("no_amp", False),
            mamba_backend=model_meta.get("mamba_backend"),
        ),
        "args": args,
        "model": model_meta,
        "data": data_meta,
        "loss": loss_meta,
        "optimizer": optimizer_meta,
        "scheduler": scheduler_meta,
        "training": training_meta,
        "result": {
            "epochs_completed": 0,
            "best_val_dice": None,
            "best_epoch": None,
            "last": {
                "val_loss": None,
                "val_dice": None,
                "val_iou": None,
            },
        },
    }
