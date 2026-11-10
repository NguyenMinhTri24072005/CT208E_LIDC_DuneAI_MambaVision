#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiền xử lý LIDC-IDRI (bản DICOM-SEG) -> dataset .npy để train MambaVision + UNet.

Mỗi bệnh nhân:  CT (nhiều .dcm) + SEG (mỗi SEG = 1 bác sĩ vẽ 1 nốt)
           ->  volumes/<pid>_img.npy    (D,H,W) float16, đã resample, clip HU, chuẩn hóa [0,1]
               volumes/<pid>_votes.npy  (D,H,W) uint8, số bác sĩ đánh dấu mỗi voxel (0-4)
               meta/<pid>.json          thông tin ca (đánh dấu ca đã xong)
Sau khi chạy: cases.csv, slices.csv (chia train/val/test THEO BỆNH NHÂN), log_failed.csv, checks/*.png

Mặc định chỉ chạy THỬ 3 ca đầu. Ví dụ:
    python preprocess_lidc.py                       # thử 3 ca đầu
    python preprocess_lidc.py --n 10 --jobs 2       # thử 10 ca, 2 tiến trình song song
    python preprocess_lidc.py --pids LIDC-IDRI-0001 LIDC-IDRI-0005
    python preprocess_lidc.py --all --jobs 2        # chạy toàn bộ (chỉ khi thử đã đúng)
Chạy lại được: ca nào đã xong sẽ bị bỏ qua (dùng --overwrite để làm lại).

Thư viện cần: pip install pydicom numpy scipy pandas matplotlib
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage

# ====================== ĐƯỜNG DẪN MẶC ĐỊNH (Windows) ======================
BASE = Path(r"D:\GG_1\COMPUTER SCIENCE\NAM4_KH1_(2026-2027)\CT208E\SOURCES")
DEFAULT_DATA = BASE / "DATASET"                        # dữ liệu gốc
DEFAULT_OUT = BASE / "Data Mining" / "processed"      # dữ liệu sau xử lý
# ==========================================================================

DEFAULT_TARGET = (1.5, 0.8, 0.8)   # spacing đích (z, y, x) mm
HU_MIN, HU_MAX = -1000, 400


# ------------------------------------------------------------------ đọc dữ liệu
def load_ct(ct_dir: Path):
    """Đọc CT -> volume HU (D,H,W) float32, dict SOPInstanceUID->chỉ số lát, spacing (z,y,x)."""
    files = sorted(ct_dir.glob("*.dcm"))
    if len(files) < 2:
        raise ValueError(f"thư mục CT chỉ có {len(files)} file .dcm")
    sl = [pydicom.dcmread(str(f)) for f in files]
    sl.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    zs = np.array([float(s.ImagePositionPatient[2]) for s in sl])
    dz = np.diff(zs)
    if dz.min() <= 1e-3:
        raise ValueError("có lát trùng vị trí z")
    if np.ptp(dz) > 0.5:
        raise ValueError(f"lát không đều/thiếu lát (dz {dz.min():.2f}-{dz.max():.2f} mm)")
    rows, cols = int(sl[0].Rows), int(sl[0].Columns)
    vol = np.empty((len(sl), rows, cols), np.float32)
    for i, s in enumerate(sl):
        if (int(s.Rows), int(s.Columns)) != (rows, cols):
            raise ValueError("các lát có kích thước khác nhau")
        vol[i] = (s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1))
                  + float(getattr(s, "RescaleIntercept", 0)))
    py, px = map(float, sl[0].PixelSpacing)
    sop2idx = {s.SOPInstanceUID: i for i, s in enumerate(sl)}
    return vol, sop2idx, (float(np.median(dz)), py, px)


def load_votes(seg_paths, sop2idx, shape):
    """Mỗi file SEG = 1 phiếu. Trả về votes (D,H,W) uint8 và số frame không khớp lát CT nào."""
    votes = np.zeros(shape, np.uint8)
    missing = 0
    for p in seg_paths:
        ds = pydicom.dcmread(str(p))
        arr = ds.pixel_array
        if arr.ndim == 2:
            arr = arr[None]
        if arr.shape[1:] != tuple(shape[1:]):
            raise ValueError(f"SEG {arr.shape[1:]} khác CT {tuple(shape[1:])}")
        m = np.zeros(shape, bool)
        for i, pf in enumerate(ds.PerFrameFunctionalGroupsSequence):
            sop = pf.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
            k = sop2idx.get(sop)
            if k is None:
                missing += 1
                continue
            m[k] |= arr[i] > 0
        votes += m
    return votes, missing


# ------------------------------------------------------------------ tiện ích
def save_npy(path: Path, arr):
    """Ghi qua file tạm rồi đổi tên, để lỡ bị ngắt giữa chừng không sinh file hỏng."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def save_check_png(img, vt, path: Path, pid: str):
    """Ảnh kiểm tra bằng mắt: lát có nhiều nốt nhất, toàn lát và crop quanh nốt (đúng dữ liệu sau xử lý)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    area = vt.reshape(vt.shape[0], -1).sum(1)
    z = int(area.argmax()) if area.max() > 0 else img.shape[0] // 2
    ys, xs = np.where(vt[z] > 0)
    H, W = img.shape[1:]
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (H // 2, W // 2)
    c = 48
    y0, x0 = int(np.clip(cy - c, 0, max(H - 2 * c, 0))), int(np.clip(cx - c, 0, max(W - 2 * c, 0)))
    sl = (slice(y0, y0 + 2 * c), slice(x0, x0 + 2 * c))
    base = img[z].astype(np.float32)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    ov = dict(cmap="autumn_r", vmin=1, vmax=4, alpha=.6)
    ax[0].imshow(base, cmap="gray", vmin=0, vmax=1)
    ax[0].imshow(np.ma.masked_equal(vt[z], 0), **ov)
    ax[0].set_title(f"{pid} | lát z={z}/{img.shape[0]}")
    ax[1].imshow(base[sl], cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("crop quanh nốt (không mask)")
    ax[2].imshow(base[sl], cmap="gray", vmin=0, vmax=1)
    ax[2].imshow(np.ma.masked_equal(vt[z][sl], 0), **ov)
    ax[2].set_title(f"votes (max={int(vt.max())})")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ------------------------------------------------------------------ xử lý 1 ca
def process(pid, data: Path, out: Path, target, overwrite, make_png):
    t0 = time.time()
    meta_path = out / "meta" / f"{pid}.json"
    if meta_path.exists() and not overwrite:
        return dict(pid=pid, status="skip", sec=0.0)
    try:
        root = data / pid
        ct_dirs = sorted(p for p in root.glob("CT*") if p.is_dir())
        if len(ct_dirs) != 1:
            raise ValueError(f"có {len(ct_dirs)} thư mục CT (cần đúng 1)")
        seg_paths = sorted(root.glob("SEG*/*.dcm"))

        vol, sop2idx, sp = load_ct(ct_dirs[0])
        votes, missing = load_votes(seg_paths, sop2idx, vol.shape)

        f = (sp[0] / target[0], sp[1] / target[1], sp[2] / target[2])
        img = ndimage.zoom(vol, f, order=1)          # ảnh: nội suy tuyến tính
        vt = ndimage.zoom(votes, f, order=0)         # votes: nearest (giữ số nguyên)
        orig_shape = list(vol.shape)
        del vol, votes
        if img.shape != vt.shape:
            raise ValueError(f"shape ảnh {img.shape} khác votes {vt.shape} sau resample")
        img = ((np.clip(img, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float16)

        save_npy(out / "volumes" / f"{pid}_img.npy", img)
        save_npy(out / "volumes" / f"{pid}_votes.npy", vt)
        if make_png:
            try:
                save_check_png(img, vt, out / "checks" / f"{pid}.png", pid)
            except Exception as e:  # không để lỗi vẽ hình làm hỏng ca
                print(f"  (không vẽ được ảnh kiểm tra {pid}: {e})")

        D = img.shape[0]
        case = dict(patient_id=pid, orig_shape=orig_shape, orig_spacing=[round(x, 4) for x in sp],
                    new_shape=list(img.shape), n_seg=len(seg_paths), max_votes=int(vt.max()),
                    missing_frames=int(missing), seconds=round(time.time() - t0, 2),
                    vox_ge1=(vt >= 1).reshape(D, -1).sum(1).tolist(),
                    vox_ge2=(vt >= 2).reshape(D, -1).sum(1).tolist())
        with open(meta_path, "w", encoding="utf-8") as fh:   # ghi cuối cùng = dấu hiệu ca đã xong
            json.dump(case, fh)
        return dict(pid=pid, status="ok", sec=time.time() - t0, shape=list(img.shape),
                    n_seg=len(seg_paths), max_votes=int(vt.max()), missing=int(missing))
    except Exception as e:
        return dict(pid=pid, status="fail", sec=time.time() - t0, error=f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ CSV chia tập
def build_index(out: Path, all_pids, seed):
    """Chia train/val/test THEO BỆNH NHÂN (70/15/15) trên toàn bộ danh sách bệnh nhân trong DATASET."""
    n = len(all_pids)
    perm = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(.7 * n), int(.15 * n)
    split = {}
    for rank, idx in enumerate(perm):
        split[all_pids[idx]] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")

    cases, slices = [], []
    for p in sorted((out / "meta").glob("*.json")):
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)
        pid = c["patient_id"]
        v1, v2 = c.pop("vox_ge1"), c.pop("vox_ge2")
        c["split"] = split.get(pid, "test")
        cases.append(c)
        slices.append(pd.DataFrame(dict(patient_id=pid, z=np.arange(len(v1)), vox_ge1=v1, vox_ge2=v2, split=c["split"])))
    if not cases:
        return None, None
    cases = pd.DataFrame(cases)
    for col in ("orig_shape", "new_shape", "orig_spacing"):
        cases[col] = cases[col].apply(lambda x: "x".join(str(v) for v in x))
    slices = pd.concat(slices, ignore_index=True)
    slices["has_nodule"] = slices.vox_ge1 > 0
    cases.to_csv(out / "cases.csv", index=False, encoding="utf-8-sig")
    slices.to_csv(out / "slices.csv", index=False, encoding="utf-8-sig")
    return cases, slices


def fmt_time(sec):
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


# ------------------------------------------------------------------ main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Tiền xử lý LIDC-IDRI (chạy thử vài ca theo mặc định)")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="thư mục chứa LIDC-IDRI-XXXX")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="thư mục lưu dữ liệu sau xử lý")
    ap.add_argument("--n", type=int, default=3, help="số ca chạy thử (mặc định 3, lấy các ca đầu danh sách)")
    ap.add_argument("--pids", nargs="+", help="chỉ định cụ thể, ví dụ LIDC-IDRI-0001 LIDC-IDRI-0002")
    ap.add_argument("--all", action="store_true", help="chạy toàn bộ bệnh nhân")
    ap.add_argument("--jobs", type=int, default=1, help="số tiến trình song song (i3 2 nhân: 1 hoặc 2)")
    ap.add_argument("--target", type=float, nargs=3, default=DEFAULT_TARGET, metavar=("Z", "Y", "X"),
                    help="spacing đích mm (mặc định 1.5 0.8 0.8)")
    ap.add_argument("--overwrite", action="store_true", help="xử lý lại cả ca đã xong")
    ap.add_argument("--no-png", action="store_true", help="không vẽ ảnh kiểm tra")
    ap.add_argument("--seed", type=int, default=42, help="seed chia train/val/test")
    a = ap.parse_args()

    if not a.data.is_dir():
        sys.exit(f"Không thấy thư mục dữ liệu: {a.data}")
    all_pids = sorted(p.name for p in a.data.iterdir() if p.is_dir() and p.name.startswith("LIDC-IDRI-"))
    if not all_pids:
        sys.exit(f"Không có thư mục LIDC-IDRI-* trong {a.data}")

    if a.pids:
        bad = [p for p in a.pids if p not in all_pids]
        if bad:
            sys.exit(f"Không thấy bệnh nhân: {bad}")
        pids = list(a.pids)
    elif a.all:
        pids = all_pids
    else:
        pids = all_pids[:a.n]

    for sub in ("volumes", "meta", "checks"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(a.out).free / 1e9
    print(f"Dữ liệu gốc : {a.data}  ({len(all_pids)} bệnh nhân)")
    print(f"Output      : {a.out}  (trống {free_gb:.0f} GB)")
    print(f"Chạy {len(pids)} ca | jobs={a.jobs} | CPU={os.cpu_count()} luồng | target={tuple(a.target)} mm\n")

    results = []
    args = (a.data, a.out, tuple(a.target), a.overwrite, not a.no_png)

    def report(i, r):
        results.append(r)
        if r["status"] == "ok":
            print(f"[{i}/{len(pids)}] {r['pid']}  OK   {r['sec']:.1f}s  shape={r['shape']}  "
                  f"SEG={r['n_seg']}  max_votes={r['max_votes']}"
                  + (f"  CẢNH BÁO: {r['missing']} frame SEG không khớp lát" if r["missing"] else ""))
        elif r["status"] == "skip":
            print(f"[{i}/{len(pids)}] {r['pid']}  bỏ qua (đã xử lý)")
        else:
            print(f"[{i}/{len(pids)}] {r['pid']}  LỖI  {r['error']}")

    t_start = time.time()
    if a.jobs <= 1:
        for i, pid in enumerate(pids, 1):
            report(i, process(pid, *args))
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(process, pid, *args) for pid in pids]
            for i, fu in enumerate(as_completed(futs), 1):
                report(i, fu.result())
    wall = time.time() - t_start

    # ---- log lỗi
    fails = [r for r in results if r["status"] == "fail"]
    if fails:
        logp = a.out / "log_failed.csv"
        df = pd.DataFrame([dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), patient_id=r["pid"], error=r["error"]) for r in fails])
        df.to_csv(logp, mode="a", header=not logp.exists(), index=False, encoding="utf-8-sig")
        print(f"\nCó {len(fails)} ca lỗi, đã ghi vào {logp}")

    # ---- CSV chia tập
    cases, slices = build_index(a.out, all_pids, a.seed)
    if cases is not None:
        print(f"\nĐã ghi cases.csv ({len(cases)} ca) và slices.csv ({len(slices)} lát) tại {a.out}")
        print("Số ca theo tập:", cases.split.value_counts().to_dict())
        print("(Chia theo bệnh nhân trên toàn bộ danh sách trong DATASET, cố định theo --seed.)")

    # ---- ước lượng cho toàn bộ
    oks = [r for r in results if r["status"] == "ok"]
    if oks:
        mean_sec = float(np.mean([r["sec"] for r in oks]))
        sizes = [((a.out / "volumes" / f"{r['pid']}_img.npy").stat().st_size
                  + (a.out / "volumes" / f"{r['pid']}_votes.npy").stat().st_size) for r in oks]
        mean_mb = float(np.mean(sizes)) / 1e6
        N = len(all_pids)
        print(f"\nTổng thời gian lần chạy: {fmt_time(wall)} | trung bình {mean_sec:.1f}s/ca (mỗi tiến trình)")
        print(f"Ước lượng thô cho {N} ca: ~{fmt_time(mean_sec * N / max(a.jobs, 1))} với jobs={a.jobs}, "
              f"dung lượng ~{mean_mb * N / 1000:.0f} GB")
        print("(Chỉ là ước lượng từ vài ca: ca nhiều lát sẽ lâu và nặng hơn. Đo trên ≥5 ca sẽ chính xác hơn.)")
    print(f"\nẢnh kiểm tra bằng mắt: {a.out / 'checks'}")


if __name__ == "__main__":      # bắt buộc trên Windows khi dùng nhiều tiến trình
    main()
