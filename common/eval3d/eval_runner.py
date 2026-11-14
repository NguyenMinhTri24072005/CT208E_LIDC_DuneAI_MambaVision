# -*- coding: utf-8 -*-
"""
common/eval3d/eval_runner.py - Vòng lặp đánh giá 3D trên tập dữ liệu cấp độ bệnh nhân.

Nhận instance model đã nạp sẵn trọng số, duyệt qua danh sách bệnh nhân validation
hoặc toàn tập, tính toán và xuất báo cáo kết quả chi tiết từng ca và tổng hợp.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import TUMOR_SIZE_LOW_LIM
from common.eval3d.volume_inference import predict_patient_volume


def evaluate_dataset_3d(
    model: nn.Module,
    cache_dir: Union[str, Path],
    split: str = "val",
    batch_size: int = 8,
    threshold: float = 0.5,
    min_slice_area_mm2: float = TUMOR_SIZE_LOW_LIM,
    min_volume_mm3: float = 0.0,
    tta: bool = False,
    out_csv: Optional[Union[str, Path]] = None,
    out_json: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    slices_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Chạy suy luận 3D và hậu xử lý cho toàn bộ bệnh nhân thuộc `split` ('val', 'train', hoặc 'all').
    So sánh trực tiếp Dice 3D trước và sau hậu xử lý.
    """
    cache_dir = Path(cache_dir)
    csv_path = Path(slices_csv) if slices_csv else (cache_dir / "slices.csv")
    if slices_csv and not csv_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy file --slices-csv: {csv_path}")

    if split != "all":
        if not csv_path.exists():
            raise FileNotFoundError(
                f"❌ Không tìm thấy file chỉ mục '{csv_path}' để lọc split='{split}'. "
                f"Vui lòng truyền --slices-csv <đường_dẫn_csv> hoặc dùng --split all."
            )
        df_slices = pd.read_csv(csv_path)
        if "split" not in df_slices.columns or "patient_id" not in df_slices.columns:
            raise ValueError(f"❌ File chỉ mục '{csv_path}' thiếu cột 'split' hoặc 'patient_id'.")
        pids = sorted(df_slices[df_slices["split"] == split]["patient_id"].unique().tolist())
    else:
        pids = sorted([
            p.name for p in cache_dir.iterdir()
            if p.is_dir() and (p / "meta.json").exists() and (p / "image_1mm.npy").exists()
        ])

    if not pids:
        raise RuntimeError(f"❌ Không tìm thấy bệnh nhân nào cho split='{split}' trong {cache_dir}")

    device = device or next(model.parameters()).device

    print("=" * 90)
    print(
        f"🩻 ĐÁNH GIÁ 3D CẤP ĐỘ BỆNH NHÂN | Split: {split} ({len(pids)} ca) | "
        f"Thiết bị: {device} | Ngưỡng lọc u: >= {min_slice_area_mm2} mm²"
    )
    print("=" * 90)

    rows: List[Dict[str, Any]] = []
    for idx, pid in enumerate(pids, start=1):
        t0 = time.time()
        res = predict_patient_volume(
            model=model,
            patient_cache_dir=cache_dir / pid,
            device=device,
            batch_size=batch_size,
            threshold=threshold,
            min_slice_area_mm2=min_slice_area_mm2,
            min_volume_mm3=min_volume_mm3,
            tta=tta,
            use_amp=use_amp,
        )
        dt = time.time() - t0
        m_raw = res["metrics_raw"]
        m_post = res["metrics_post"]

        row = {
            "patient_id": pid,
            "dice_3d_raw": round(m_raw["dice_3d"], 4),
            "dice_3d_post": round(m_post["dice_3d"], 4),
            "iou_3d_post": round(m_post["iou_3d"], 4),
            "precision_3d": round(m_post["precision_3d"], 4),
            "recall_3d": round(m_post["recall_3d"], 4),
            "hd95_mm": m_post["hd95_mm"],
            "gt_vol_mm3": m_post["gt_volume_mm3"],
            "pred_vol_mm3": m_post["pred_volume_mm3"],
            "time_sec": round(dt, 2),
        }
        rows.append(row)
        diff_dice = m_post["dice_3d"] - m_raw["dice_3d"]
        print(
            f"[{idx:02d}/{len(pids):02d}] {pid} ({dt:.1f}s) | "
            f"3D Dice: {m_raw['dice_3d']:.4f} -> {m_post['dice_3d']:.4f} ({diff_dice:+.4f}) | "
            f"IoU: {m_post['iou_3d']:.4f} | HD95: {m_post['hd95_mm']} mm | "
            f"Vol(GT/Pred): {m_post['gt_volume_mm3']}/{m_post['pred_volume_mm3']} mm³"
        )

    df_res = pd.DataFrame(rows)
    mean_raw = float(df_res["dice_3d_raw"].mean())
    mean_post = float(df_res["dice_3d_post"].mean())
    mean_iou = float(df_res["iou_3d_post"].mean())

    valid_hd95 = [v for v in df_res["hd95_mm"] if np.isfinite(v)]
    mean_hd95 = float(np.mean(valid_hd95)) if valid_hd95 else float("nan")

    print("=" * 90)
    print(
        f"📊 TỔNG KẾT ({len(pids)} bệnh nhân) | "
        f"Mean 3D Dice (Raw): {mean_raw:.4f} -> Mean 3D Dice (Post): {mean_post:.4f} "
        f"({mean_post - mean_raw:+.4f}) | Mean 3D IoU: {mean_iou:.4f} | Mean HD95: {mean_hd95:.2f} mm"
    )
    print("=" * 90)

    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(out_csv, index=False)
        print(f"✅ Đã lưu bảng báo cáo đánh giá 3D tại: {out_csv}")

    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "total_patients": len(pids),
            "split": split,
            "mean_dice_3d_raw": round(mean_raw, 4),
            "mean_dice_3d_post": round(mean_post, 4),
            "mean_iou_3d_post": round(mean_iou, 4),
            "mean_hd95_mm": round(mean_hd95, 3) if np.isfinite(mean_hd95) else None,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Đã lưu tóm tắt JSON tại: {out_json}")

    return df_res
