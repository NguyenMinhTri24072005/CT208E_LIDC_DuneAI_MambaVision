# -*- coding: utf-8 -*-
"""
common/paths.py - Quản lý đường dẫn tập trung cho toàn bộ dự án DuneAI.

Quy tắc bất biến:
  - Tuyệt đối không hard-code đường dẫn tuyệt đối (Không dùng tên repo cũ, SOURCES, /content, /kaggle)
  - REPO_ROOT tự động suy ra từ vị trí file common/paths.py (2 cấp cha)
"""

import os
from pathlib import Path

# REPO_ROOT: Thư mục gốc của repository (SOURCES/)
REPO_ROOT = Path(__file__).resolve().parent.parent


def default_data_dir() -> Path:
    """Trả về thư mục dữ liệu cache mặc định (SOURCES/data/cache_1mm)."""
    env_dir = os.environ.get("DUNEAI_DATA_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return REPO_ROOT / "data" / "cache_1mm"


def default_runs_root(arch: str) -> Path:
    """
    Trả về thư mục gốc chứa các thí nghiệm huấn luyện theo từng kiến trúc:
      - arch="unet_base"   -> SOURCES/unet_base/runs
      - arch="mamba_unet"  -> SOURCES/mamba_unet/runs
    """
    env_runs = os.environ.get("DUNEAI_RUNS_ROOT")
    if env_runs:
        return Path(env_runs).resolve()
    return REPO_ROOT / arch / "runs"
