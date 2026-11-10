# -*- coding: utf-8 -*-
"""
common/engine/utils.py - Các hàm tiện ích hỗ trợ huấn luyện:
  - seed_everything: Cố định hạt giống ngẫu nhiên đa thư viện
  - format_duration: Định dạng thời gian chạy thân thiện
  - atomic_save_json: Ghi JSON nguyên tử qua file tạm + os.replace
"""

import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Cố định seed ngẫu nhiên cho python random, numpy và torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_duration(seconds: float) -> str:
    """Chuyển số giây sang định dạng giờ:phút:giây (vd: '01h 24m 35s' hoặc '04m 12s')."""
    if seconds is None:
        return "N/A"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def atomic_save_json(path: Union[str, Path], data: Any, indent: int = 2):
    """Lưu dữ liệu JSON nguyên tử (atomic) tránh bị hỏng file khi bị ngắt đột ngột."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_file = p.parent / f".tmp_{p.name}_{os.getpid()}"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    os.replace(str(temp_file), str(p))
