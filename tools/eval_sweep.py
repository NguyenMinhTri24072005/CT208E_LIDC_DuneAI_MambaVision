# -*- coding: utf-8 -*-
"""tools/eval_sweep.py - Quét ngưỡng + đánh giá 3D mức bệnh nhân và mức nốt."""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.data_pipeline.medical_prep import lung_cache_path
from common.eval3d.metrics3d import compute_3d_metrics
from common.eval3d.postprocess3d import postprocess_3d_mask
from common.eval3d.volume_inference import predict_patient_volume
from mamba_unet.evaluate import load_mamba_unet_model

S26 = np.ones((3, 3, 3), np.uint8)
BINS = [(0, 5), (5, 10), (10, 20), (20, 1e9)]          # đường kính tương đương (mm)


def eq_diam(sub):                                       # sub: mask bool (z,y,x) của 1 nốt
    area = sub.reshape(sub.shape[0], -1).sum(1).max()   # diện tích lát lớn nhất (1 px = 1 mm²)
    return 2.0 * float(np.sqrt(area / np.pi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--backbone", default="mamba_vision_T")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--min-area", type=float, default=20.0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn số ca bệnh nhân cần đánh giá (0 = tất cả)")
    ap.add_argument("--out-dir", default=None, help="Thư mục lưu kết quả CSV (mặc định lưu thư mục hiện tại)")
    ap.add_argument("--out-prefix", default="eval_sweep")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mamba_unet_model(a.checkpoint, dev, backbone=a.backbone)
    cache = Path(a.cache_dir)
    df = pd.read_csv(cache / "slices.csv")
    pids = sorted(df[df["split"] == a.split]["patient_id"].unique())
    if a.limit > 0:
        pids = pids[: a.limit]
    print(f"🔬 Đánh giá quét ngưỡng trên {len(pids)} bệnh nhân tập '{a.split}'...")

    pat_rows, nod_rows = [], []
    for pid in pids:
        res = predict_patient_volume(model, cache / pid, device=dev, tta=a.tta)
        prob, gt, meta = res["prob_1mm"], res["gt_mask_1mm"], res["meta"]
        sz = float(meta.get("orig_spacing_xyz", [1, 1, 1])[2])
        lp = lung_cache_path(cache / pid)
        lung = np.load(lp) if (lp.exists() and meta.get("lung_ok", True) is not False) else None

        gt_lab, n_gt = ndi.label(gt, structure=S26)
        gt_objs = ndi.find_objects(gt_lab)
        diams = [eq_diam(gt_lab[sl] == i + 1) for i, sl in enumerate(gt_objs)]

        for th in a.thresholds:
            pred = postprocess_3d_mask(prob, lung_mask_zyx=lung, threshold=th,
                                       min_slice_area_mm2=a.min_area, spacing_z=sz)
            m = compute_3d_metrics(pred, gt, spacing_z=sz, compute_hd95=False)
            pr_lab, n_pr = ndi.label(pred, structure=S26)
            hit = np.unique(pr_lab[gt > 0]); hit = hit[hit > 0]
            detected = 0
            for i, sl in enumerate(gt_objs):
                ok = bool((pred[sl][gt_lab[sl] == i + 1] > 0).any())
                detected += ok
                nod_rows.append(dict(pid=pid, th=th, diam=diams[i], detected=int(ok)))
            pat_rows.append(dict(pid=pid, th=th, dice=m["dice_3d"], precision=m["precision_3d"],
                                 recall=m["recall_3d"], fp_comps=int(n_pr - len(hit)),
                                 n_gt=n_gt, n_detected=detected))
        print(f"[{pid}] xong")

    P, N = pd.DataFrame(pat_rows), pd.DataFrame(nod_rows)
    out_dir = Path(a.out_dir) if a.out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    pat_csv = out_dir / f"{a.out_prefix}_patients.csv"
    nod_csv = out_dir / f"{a.out_prefix}_nodules.csv"
    P.to_csv(pat_csv, index=False)
    N.to_csv(nod_csv, index=False)
    print(f"\n💾 Đã lưu kết quả tại:\n  - {pat_csv}\n  - {nod_csv}")

    print("\n=== THEO NGƯỠNG (trung bình trên ca) ===")
    print(P.groupby("th")[["dice", "precision", "recall", "fp_comps"]].mean().round(4))
    print("\n=== ĐỘ NHẠY THEO NỐT & KÍCH THƯỚC (đường kính tương đương, mm) ===")
    for lo, hi in BINS:
        sub = N[(N["diam"] >= lo) & (N["diam"] < hi)]
        if len(sub):
            print(f"[{lo:>4}-{hi if hi < 1e8 else 'inf'}) n={len(sub)//len(a.thresholds):4d} ",
                  sub.groupby("th")["detected"].mean().round(3).to_dict())


if __name__ == "__main__":
    main()