# -*- coding: utf-8 -*-
"""tools/eval_sweep_v2.py - Đánh giá đối chứng R0b đa ngưỡng diện tích lát và thể tích 3D."""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.data_pipeline.medical_prep import IMAGE_SIZE, lung_cache_path
from common.engine.checkpoint_io import load_checkpoint_file
from common.eval3d.metrics3d import compute_3d_metrics
from common.eval3d.postprocess3d import postprocess_3d_mask
from common.eval3d.volume_inference import predict_patient_volume
from mamba_unet.evaluate import load_mamba_unet_model

S26 = np.ones((3, 3, 3), np.uint8)
BINS = [(0, 5), (5, 10), (10, 20), (20, 1e9)]  # đường kính tương đương (mm)


def eq_diam(sub: np.ndarray) -> float:
    """Tính đường kính tương đương lớn nhất (mm) từ mặt cắt 2D rộng nhất của khối u 3D."""
    area = sub.reshape(sub.shape[0], -1).sum(1).max()
    return 2.0 * float(np.sqrt(area / np.pi))


def _resolve_image_size(checkpoint_path: str, explicit_size: int) -> int:
    """Ưu tiên --image-size chỉ định, sau đó đọc từ meta của checkpoint, cuối cùng dùng IMAGE_SIZE."""
    if explicit_size and explicit_size > 0:
        return int(explicit_size)
    try:
        ckpt = load_checkpoint_file(checkpoint_path, map_location="cpu")
        meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        if meta.get("image_size"):
            return int(meta["image_size"])
    except Exception:
        pass
    return int(IMAGE_SIZE)


def _ensure_slices_df(cache: Path, checkpoint_path: str) -> pd.DataFrame:
    """Đọc slices.csv trong cache; tự động copy từ thư mục run hoặc dựng mới nếu thiếu."""
    slices_file = cache / "slices.csv"
    if not slices_file.exists():
        ckpt_slices = Path(checkpoint_path).resolve().parent / "slices.csv"
        if ckpt_slices.exists():
            print(f"ℹ️ Phát hiện bản sao chỉ mục tại {ckpt_slices}, đang nạp vào cache...")
            shutil.copy2(ckpt_slices, slices_file)
        else:
            print(f"⚙️ Chưa tìm thấy {slices_file.name}, đang tự động quét và dựng chỉ mục...")
            from common.data_pipeline.offline_builder import build_dataset_index
            build_dataset_index(cache)
    return pd.read_csv(slices_file)


def run_eval_sweep_v2(
    model: torch.nn.Module,
    cache_dir: Path,
    pids: Sequence[str],
    device: torch.device,
    thresholds: Sequence[float] = (0.5, 0.6, 0.7),
    min_areas: Sequence[float] = (6.0, 10.0, 20.0),
    min_vols: Sequence[float] = (0.0, 10.0, 20.0),
    min_span: int = 1,
    batch_size: int = 8,
    image_size: int = IMAGE_SIZE,
    tta: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy suy luận 3D một lần mỗi ca, sau đó quét tổ hợp (th, min_area, min_vol)."""
    pat_rows, nod_rows = [], []

    for pid in pids:
        res = predict_patient_volume(
            model,
            cache_dir / pid,
            device=device,
            batch_size=batch_size,
            image_size=image_size,
            tta=tta,
        )
        prob, gt, meta = res["prob_1mm"], res["gt_mask_1mm"], res["meta"]
        sz = float(meta.get("orig_spacing_xyz", [1.0, 1.0, 1.0])[2])
        lp = lung_cache_path(cache_dir / pid)
        lung = np.load(lp) if (lp.exists() and meta.get("lung_ok", True) is not False) else None

        gt_lab, n_gt = ndi.label(gt, structure=S26)
        gt_objs = ndi.find_objects(gt_lab)
        valid_gt = [(idx, sl) for idx, sl in enumerate(gt_objs, start=1) if sl is not None]
        diams = [eq_diam(gt_lab[sl] == idx) for idx, sl in valid_gt]

        for th in thresholds:
            for ma in min_areas:
                for mv in min_vols:
                    pred = postprocess_3d_mask(
                        prob,
                        lung_mask_zyx=lung,
                        threshold=th,
                        min_slice_area_mm2=ma,
                        min_volume_mm3=mv,
                        spacing_z=sz,
                        min_slices_span=min_span,
                    )
                    m = compute_3d_metrics(pred, gt, spacing_z=sz, compute_hd95=False)
                    pr_lab, n_pr = ndi.label(pred, structure=S26)
                    hit = np.unique(pr_lab[gt > 0])
                    hit = hit[hit > 0]

                    detected = 0
                    for (idx, sl), diam in zip(valid_gt, diams):
                        ok = bool((pred[sl][gt_lab[sl] == idx] > 0).any())
                        detected += int(ok)
                        nod_rows.append(
                            dict(
                                pid=pid,
                                th=float(th),
                                min_area=float(ma),
                                min_vol=float(mv),
                                diam=float(diam),
                                detected=int(ok),
                            )
                        )

                    pat_rows.append(
                        dict(
                            pid=pid,
                            th=float(th),
                            min_area=float(ma),
                            min_vol=float(mv),
                            dice=float(m["dice_3d"]),
                            precision=float(m["precision_3d"]),
                            recall=float(m["recall_3d"]),
                            fp_comps=int(n_pr - len(hit)),
                            n_gt=int(n_gt),
                            n_detected=int(detected),
                        )
                    )
        if verbose:
            print(f"[{pid}] Đã quét xong.")

    return pd.DataFrame(pat_rows), pd.DataFrame(nod_rows)


def print_sweep_summary(P: pd.DataFrame, N: pd.DataFrame) -> None:
    """In bảng tổng hợp theo (th, min_area, min_vol) và độ nhạy theo dải đường kính."""
    if len(P) == 0:
        print("⚠️ Không có dữ liệu bệnh nhân để tổng hợp.")
        return

    print("\n=== TỔNG HỢP THEO CẤU HÌNH HẬU XỬ LÝ (trung bình trên ca) ===")
    grp = P.groupby(["th", "min_area", "min_vol"])[["dice", "precision", "recall", "fp_comps"]].mean().round(4)
    print(grp.to_string())

    best_idx = grp["dice"].idxmax()
    best_row = grp.loc[best_idx]
    print(
        f"\n⭐ Đỉnh Dice 3D: {best_row['dice']:.4f} tại Th={best_idx[0]}, "
        f"min_area={best_idx[1]} mm², min_vol={best_idx[2]} mm³ "
        f"(Prec={best_row['precision']:.4f}, Rec={best_row['recall']:.4f}, FP/ca={best_row['fp_comps']:.2f})"
    )

    if len(N) == 0:
        return

    best_th, best_ma, best_mv = best_idx
    n_best = N[(N["th"] == best_th) & (N["min_area"] == best_ma) & (N["min_vol"] == best_mv)]
    print(f"\n=== ĐỘ NHẠY THEO DẢI ĐƯỜNG KÍNH TẠI ĐỈNH (Th={best_th}, A>={best_ma}, V>={best_mv}) ===")
    for lo, hi in BINS:
        sub = n_best[(n_best["diam"] >= lo) & (n_best["diam"] < hi)]
        if len(sub):
            sens = float(sub["detected"].mean())
            hits = int(sub["detected"].sum())
            print(f"[{lo:>4}-{hi if hi < 1e8 else 'inf'}) n={len(sub):4d} | bắt được {hits:4d}/{len(sub):<4d} ({sens * 100:5.1f}%)")


def _selftest() -> None:
    """Kiểm thử tự động logic eval_sweep_v2 trên dữ liệu 3D giả lập (CPU)."""
    print("🧪 Bắt đầu selftest tools/eval_sweep_v2.py...")
    tmp = Path(tempfile.mkdtemp())
    try:
        cache = tmp / "cache"
        # Ca 1: Có 1 nốt nhỏ (< 5mm) và 1 nốt trung bình (~11mm)
        p1 = cache / "LIDC-IDRI-0001"
        p1.mkdir(parents=True)
        img1 = np.full((10, 64, 64), -800, dtype=np.int16)
        msk1 = np.zeros((10, 64, 64), dtype=np.uint8)
        lung1 = np.zeros((10, 64, 64), dtype=np.uint8)
        lung1[1:9, 8:56, 8:56] = 1
        msk1[2:4, 15:18, 15:18] = 1   # 9 mm² -> d ≈ 3.38 mm (< 5mm)
        msk1[5:8, 30:40, 30:40] = 1   # 100 mm² -> d ≈ 11.28 mm (10-20mm)
        img1[msk1 > 0] = 50
        for p_dir, img, msk, lung in [
            (p1, img1, msk1, lung1),
            # Ca 2: Ca không u (n_gt = 0)
            (cache / "LIDC-IDRI-0002", np.full((8, 64, 64), -800, dtype=np.int16), np.zeros((8, 64, 64), dtype=np.uint8), np.ones((8, 64, 64), dtype=np.uint8)),
        ]:
            p_dir.mkdir(parents=True, exist_ok=True)
            np.save(p_dir / "image_1mm.npy", img)
            np.save(p_dir / "mask_1mm.npy", msk)
            np.save(p_dir / "lung_1mm.npy", lung)
            with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "orig_shape_zyx": list(img.shape),
                        "orig_spacing_xyz": [0.7, 0.7, 1.0],
                        "shape_1mm_zyx": list(img.shape),
                        "lung_ok": True,
                    },
                    f,
                )

        from mamba_unet.backbone.mamba_vision import MambaVision
        from mamba_unet.model import MambaUNet

        tiny_bb = MambaVision(
            dim=16,
            in_dim=8,
            depths=[1, 1, 2, 1],
            window_size=[8, 8, 14, 7],
            num_heads=[2, 2, 4, 4],
            in_chans=3,
        )
        model = MambaUNet(in_channels=3, num_classes=1, backbone=tiny_bb, act="relu")
        ckpt_path = tmp / "tiny_r0.pth"
        torch.save(
            {"model_state_dict": model.state_dict(), "meta": {"arch": "mamba_unet", "image_size": 64}},
            ckpt_path,
        )
        assert _resolve_image_size(str(ckpt_path), 0) == 64

        P, N = run_eval_sweep_v2(
            model=model,
            cache_dir=cache,
            pids=["LIDC-IDRI-0001", "LIDC-IDRI-0002"],
            device=torch.device("cpu"),
            thresholds=[0.5, 0.6],
            min_areas=[6.0, 20.0],
            min_vols=[0.0, 10.0],
            batch_size=2,
            image_size=64,
            tta=False,
            verbose=False,
        )
        # 2 patients * 2 th * 2 min_area * 2 min_vol = 16 rows
        assert len(P) == 16, f"Số dòng patient phải là 16, nhận {len(P)}"
        # 2 GT nodules (only on patient 1) * 8 grid combos = 16 rows
        assert len(N) == 16, f"Số dòng nodule phải là 16, nhận {len(N)}"
        assert {"pid", "th", "min_area", "min_vol", "dice", "precision", "recall", "fp_comps", "n_gt", "n_detected"} <= set(P.columns)
        assert {"pid", "th", "min_area", "min_vol", "diam", "detected"} <= set(N.columns)

        print_sweep_summary(P, N)
        print("✅ Selftest tools/eval_sweep_v2.py hoàn tất thành công!")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Đánh giá đối chứng R0b đa ngưỡng diện tích và thể tích 3D.")
    ap.add_argument("--selftest", action="store_true", help="Chạy kiểm thử tự động trên dữ liệu giả lập (CPU)")
    ap.add_argument("--cache-dir", default=None, help="Thư mục dữ liệu cache_1mm")
    ap.add_argument("--checkpoint", default=None, help="Đường dẫn file checkpoint (.pth)")
    ap.add_argument("--split", default="val", help="Tập đánh giá (val, train, test, all)")
    ap.add_argument("--backbone", default="mamba_vision_T")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    ap.add_argument("--min-areas", type=float, nargs="+", default=[6.0, 10.0, 20.0])
    ap.add_argument("--min-vols", type=float, nargs="+", default=[0.0, 10.0, 20.0])
    ap.add_argument("--min-span", type=int, default=1, help="Số lát cắt liên tiếp tối thiểu của một cụm 3D")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=0, help="Kích thước ảnh suy luận (0 = tự đọc từ checkpoint)")
    ap.add_argument("--tta", action="store_true", help="Bật Test-Time Augmentation (lật ngang/dọc)")
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn số ca bệnh nhân (0 = tất cả)")
    ap.add_argument("--out-dir", default=None, help="Thư mục lưu kết quả CSV (mặc định thư mục hiện tại)")
    ap.add_argument("--out-prefix", default="sweep_r0b")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return

    if not a.cache_dir or not a.checkpoint:
        ap.error("--cache-dir và --checkpoint là bắt buộc khi không chạy --selftest.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = _resolve_image_size(a.checkpoint, a.image_size)
    model = load_mamba_unet_model(a.checkpoint, dev, backbone=a.backbone)

    cache = Path(a.cache_dir)
    df = _ensure_slices_df(cache, a.checkpoint)
    sub_df = df if a.split == "all" else df[df["split"] == a.split]
    pids = sorted(sub_df["patient_id"].unique())
    if a.limit > 0:
        pids = pids[: a.limit]

    print(
        f"🔬 Đánh giá đối chứng v2 trên {len(pids)} bệnh nhân tập '{a.split}' "
        f"(image_size={img_size}, Th={a.thresholds}, A={a.min_areas}, V={a.min_vols}, span>={a.min_span})..."
    )

    P, N = run_eval_sweep_v2(
        model=model,
        cache_dir=cache,
        pids=pids,
        device=dev,
        thresholds=a.thresholds,
        min_areas=a.min_areas,
        min_vols=a.min_vols,
        min_span=a.min_span,
        batch_size=a.batch_size,
        image_size=img_size,
        tta=a.tta,
        verbose=True,
    )

    out_dir = Path(a.out_dir) if a.out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    pat_csv = out_dir / f"{a.out_prefix}_patients.csv"
    nod_csv = out_dir / f"{a.out_prefix}_nodules.csv"
    P.to_csv(pat_csv, index=False)
    N.to_csv(nod_csv, index=False)
    print(f"\n✅ Hoàn tất lưu kết quả tại:\n  - {pat_csv}\n  - {nod_csv}")

    print_sweep_summary(P, N)


if __name__ == "__main__":
    main()
