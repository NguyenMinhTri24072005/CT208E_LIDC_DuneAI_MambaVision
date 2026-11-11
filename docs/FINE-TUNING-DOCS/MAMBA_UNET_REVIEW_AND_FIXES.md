# Mamba-UNet (MambaVision-T + UNet) — Tổng hợp vấn đề, nguyên nhân, cách sửa và cải tiến

*Dựa trên: `MAMBA_UNET_SYSTEM_REPORT.md`, `medical_prep.py` và nhật ký huấn luyện `mamba_colab_500item_25epoch_extra_test` (50 epoch đặt, dừng sớm ở epoch 42, best Val Dice 0.7456 ở epoch 30, 500 ca LIDC / 1010).*

> **Quy ước:** mọi thay đổi bên dưới đều có **giá trị mặc định = hành vi hiện tại** (cờ mới mặc định tắt) để không phá vỡ code, test và golden keys đang có. Các con số "kỳ vọng" là ước lượng kinh nghiệm để xếp thứ tự ưu tiên, **không phải cam kết** — phải kiểm chứng bằng ablation (mục 7).

---

## 0. Tóm tắt

- Val Dice 0.7456, Train Dice 0.87 → khoảng cách ~0.13 và val phẳng từ epoch ~24 dù LR giảm 10 lần ⇒ **đã chạm trần tổng quát hóa của cấu hình hiện tại** (không phải vấn đề tối ưu hóa).
- Run mẫu 25 epoch (0.7412) và run 50 epoch (0.7456) chỉ lệch 0.004 ⇒ tăng thêm epoch **không** giúp.
- Nguyên nhân chính nằm ở: (1) pipeline dữ liệu làm val 2D lạc quan và lệch so với suy luận 3D, (2) decoder chỉ ra feature H/4 rồi upsample ×4, (3) ít dữ liệu + regularization yếu (weight decay vô hiệu, augmentation nghèo), (4) chưa có đo lường đúng (3D, theo nốt, P/R).
- **Làm trước (không tốn công train):** chạy `tools/eval_sweep.py` (mục 2) để biết lỗi nằm ở nốt nhỏ, dương tính giả hay ngưỡng.

---

## 1. Bảng tổng hợp vấn đề

| ID | Vấn đề | Vị trí | Nguyên nhân | Ảnh hưởng tới kết quả | Mức | Sửa ở |
|---|---|---|---|---|---|---|
| P1 | Val 2D lạc quan hơn thực tế | `offline_builder.py` `build_dataset_index`, `configs/config.py` | 2081 lát u < 20 mm² (nốt nhỏ + lát biên nốt) bị loại khỏi cả train lẫn val; âm tính chỉ 22% (1321/6116), 75620 lát âm bị bỏ; ngưỡng 20 mm² và quantize 25 HU kế thừa DuneAI (u NSCLC lớn), không hợp nốt LIDC | Mô hình không học nốt nhỏ/lát biên; 3D Dice thật thấp hơn val 2D; dương tính giả nhiều khi chạy cả khối | Cao | §3.1, §5.2 |
| P2 | Chưa đo 3D / theo nốt / P-R trong quá trình chọn model | `base_trainer.py`, `evaluate.py` | Chỉ log Dice gộp pixel; chọn best bằng val Dice 2D | Không biết lỗi do FP hay FN, nốt nhỏ hay lớn; ngưỡng 0.5 chưa chắc tối ưu (do `pos_weight=4`) | Cao | §2, §3.3 |
| P3 | Lệch train ↔ val/inference ở mặt nạ phổi | `dataset.py` `__getitem__` (`lung_keep_mask`) | Train hợp mask phổi với mask u đã dilate; val/inference chỉ dùng mask phổi. Nốt dính màng phổi bị tách phổi bỏ sót → train thấy "đảo mô mềm quanh u" (lộ nhãn), val bị xóa mất một phần u | Gap train/val; nốt sát thành ngực bị cắt khi inference | Cao | §2 (kiểm tra), §3.2 |
| P4 | Overfit (gap 0.13) | toàn pipeline | Ít nốt độc lập (5186 lát tương quan), aug chỉ flip/rot90, weight decay 1e-4 (vô hiệu), drop_path cố định 0.2 | Val dừng ở ~0.74 trong khi train → 0.87 | Cao | §3.4, §4, §5 |
| P5 | Độ phân giải đầu ra thấp | `decoder.py` (`final_up`) | Feature mịn nhất H/4 (128×128), upsample bilinear ×4, không có skip H/2, H/1. 1 mm/pixel nên nốt 4 mm ≈ 1 pixel ở H/4 | Biên mờ, bỏ sót nốt nhỏ; trần Dice thấp | Cao | §6.1 |
| P6 | Cửa sổ stage 3–4 bị pad + cô lập | `mamba_vision.py` `MambaVisionLayer`, `configs/config.py` | 512 → feature 32×32 (cửa sổ 14) pad lên 42×42; stage 4 16×16 (cửa sổ 7) pad lên 21×21; không trao đổi giữa các cửa sổ; aug không có dịch chuyển nên lưới cửa sổ luôn cố định so với ảnh | Tốn ~1.7× token ở stage 3–4; artefact ở biên cửa sổ; nốt nằm vắt qua biên cửa sổ | Trung bình | §5.3, §6.2 |
| P7 | BN backbone không `eval()` khi đóng băng như spec | `base_trainer.run_epoch` gọi `model.train()` **sau** `on_epoch_start` | Hook đặt BN eval bị ghi đè | Thực tế BN vẫn cập nhật thống kê CT trong 3 epoch freeze (có thể có lợi!). Chỉ là sai spec | Thấp | §3.5 |
| P8 | Weight decay vô hiệu và áp cả lên norm/bias/`A_log`/`D` | `trainer.build_param_groups`, `train_main.py` | AdamW wd=1e-4 ⇒ sau ~27k bước chỉ co trọng số ~0.03%; không dùng cờ `_no_weight_decay` | Gần như không có regularization | Trung bình | §3.4 |
| P9 | Chuẩn hóa đầu vào lệch ImageNet | `model.py` | Đầu vào [0,1] từ cửa sổ HU, không mean/std; backbone pretrained quen RGB-ImageNet | Lệch miền không cần thiết (BN stem tự thích nghi một phần) | Thấp | §6.3 |
| P10 | Khoảng cách z không chuẩn hóa | `dataset.py` `_neighbor_indices`, `volume_inference.py` | z±1 = 0.6 mm ở ca này, 3–5 mm ở ca khác | "Ngữ cảnh 2.5D" khác nghĩa vật lý giữa các ca | Thấp–TB | §5.4 |
| P11 | Nội suy `nearest` khi resize 1 mm | `medical_prep.py` `IMAGE_INTERP` | Kế thừa tác giả; spacing LIDC 0.5–0.98 mm ⇒ đang *downsample*, nearest tạo răng cưa | Nhiễu trên nốt nhỏ | Thấp (tốn công rebuild) | §5.5 |
| P12 | Chọn 500 ca đầu + chưa có test set | `runtime_data.select_patients` (`first`), `offline_builder` | Ca LIDC-IDRI đầu có thể thiên về vài máy/độ dày lát; model chọn bằng val rồi báo cáo luôn val | Thiên lệch chọn mẫu; best_val_dice lạc quan nhẹ | Trung bình | §5.6 |
| P13 | Val nhiễu, không dùng trung bình trọng số | `base_trainer.fit` | 75 ca val ⇒ dao động ±0.005–0.01/epoch; không EMA | "Best epoch" ngẫu nhiên; kết quả kém ổn định | Trung bình | §5.1 |
| P14 | Lỗi/bất nhất nhỏ | xem §3.6 | — | Gây hiểu nhầm | Thấp | §3.6 |

---

## 2. BƯỚC 0 — Đo lường (làm trước, không cần train lại)

### 2.1 Tạo `tools/eval_sweep.py`: quét ngưỡng + đánh giá 3D theo từng nốt

Trả lời các câu hỏi: ngưỡng tốt nhất là bao nhiêu? bao nhiêu dương tính giả mỗi ca? độ nhạy theo kích thước nốt?

```python
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
    ap.add_argument("--out-prefix", default="eval_sweep")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_mamba_unet_model(a.checkpoint, dev, backbone=a.backbone)
    cache = Path(a.cache_dir)
    df = pd.read_csv(cache / "slices.csv")
    pids = sorted(df[df["split"] == a.split]["patient_id"].unique())

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
    P.to_csv(f"{a.out_prefix}_patients.csv", index=False)
    N.to_csv(f"{a.out_prefix}_nodules.csv", index=False)

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
```

Chạy (Colab):

```bash
python tools/eval_sweep.py --cache-dir /content/data/cache_1mm \
  --checkpoint mamba_unet/runs/mamba_colab_500item_25epoch_extra_test/best_model.pth \
  --split val --out-prefix sweep_r0
```

**Cách đọc kết quả**

| Quan sát | Kết luận | Hành động |
|---|---|---|
| `fp_comps` lớn (≫ 1/ca) và precision ≪ recall | Dương tính giả là nút thắt | §4 (`pos_weight`↓), §5.2 (neg-ratio↑, hard negative), ngưỡng↑ |
| Độ nhạy nốt < 5 mm rất thấp | Mất nốt nhỏ (do P1, P5) | §3.1, §6.1 |
| Dice 3D tăng rõ khi đổi ngưỡng ≠ 0.5 | Ngưỡng lệch do `pos_weight` | Chọn ngưỡng theo val, báo cáo trên test |
| Dice 3D ≈ Dice val 2D | Val 2D không quá lạc quan | Ưu tiên chống overfit (§3.4, §5) |

### 2.2 Tạo `tools/check_cache.py`: kiểm tra P3 (mask phổi cắt u) và FOV khi đổi `IMAGE_SIZE`

```python
# -*- coding: utf-8 -*-
"""tools/check_cache.py <cache_dir> [image_size=448]"""
import json, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]); size = int(sys.argv[2]) if len(sys.argv) > 2 else 448
fr, lost_fov = [], []
for mp in sorted(root.glob("LIDC-IDRI-*/meta.json")):
    m = json.loads(mp.read_text(encoding="utf-8"))
    f = m.get("mask_in_lung_fraction")
    if f is not None:
        fr.append((float(f), mp.parent.name))
    z, h, w = m["shape_1mm_zyx"]
    if h > size or w > size:                                   # sẽ bị crop giữa khi pad_or_crop_xy
        proj = np.load(mp.parent / "mask_1mm.npy", mmap_mode="r")
        proj = np.asarray(proj).any(axis=0)
        if proj.sum() == 0:
            continue
        sy, sx = max(0, (h - size) // 2), max(0, (w - size) // 2)
        inside = proj[sy:sy + min(h, size), sx:sx + min(w, size)].sum()
        lost_fov.append((1 - inside / proj.sum(), mp.parent.name))

if fr:
    v = np.array([x for x, _ in fr])
    print(f"mask_in_lung_fraction: n={len(v)} | <0.9: {(v<0.9).sum()} | <0.7: {(v<0.7).sum()} | "
          f"median={np.median(v):.3f} | p5={np.percentile(v,5):.3f}")
    print("5 ca tệ nhất:", sorted(fr)[:5])
else:
    print("Không có mask_in_lung_fraction trong meta.json (cache build thiếu --extract-lungs?)")
print(f"Ca có FOV > {size}: {len(lost_fov)} | ca mất nhãn do crop: {sum(x>0 for x,_ in lost_fov)}")
print("Top mất nhãn do crop:", sorted(lost_fov, reverse=True)[:5])
```

- Nếu nhiều ca có `mask_in_lung_fraction < 0.9` ⇒ P3 nghiêm trọng, ưu tiên §3.2.
- Nếu có ca mất nhãn do crop ⇒ cân nhắc trước khi đổi `IMAGE_SIZE` xuống 448 (§5.3).

---

## 3. Sửa vấn đề (chi tiết theo file)

### 3.1 [P1] Cho phép hạ ngưỡng lát u khi **dựng chỉ mục train** (giữ 20 mm² ở hậu xử lý)

`offline_builder.build_dataset_index` **đã có** tham số `tumor_size_low_lim` và `zero_sl_ratio`; chỉ cần đưa chúng ra CLI.

**a) `common/engine/cli_args.py`** — thêm vào `data_group`:

```python
    data_group.add_argument("--min-train-area", type=float, default=20.0,
                            help="Ngưỡng diện tích u (mm²) tối thiểu của lát khi dựng slices.csv (0 = giữ mọi lát có u)")
    data_group.add_argument("--neg-ratio", type=float, default=0.25,
                            help="Tỷ lệ lát âm tính / lát dương tính khi dựng slices.csv (ZERO_SL_RATIO)")
    data_group.add_argument("--aug-level", type=int, choices=[1, 2], default=1,
                            help="1: flip/rot90 (hiện tại); 2: + affine + nhiễu/tương phản")
    data_group.add_argument("--tumor-union-p", type=float, default=1.0,
                            help="Xác suất hợp mask u vào vùng giữ phổi lúc train (1.0 = hiện tại, 0 = đồng nhất với val)")
    data_group.add_argument("--neighbor-mm", type=float, default=0.0,
                            help="Khoảng cách vật lý (mm) giữa các lát 2.5D (0 = lát kề theo chỉ số như hiện tại)")
```

**b) `common/data_pipeline/dataset.py`** — import và `build_dataloaders`:

```python
import json                                  # (đã có ở đầu file)
import cv2                                   # THÊM
from configs.config import IMAGE_SIZE, TUMOR_SIZE_LOW_LIM, ZERO_SL_RATIO   # THÊM 2 hằng số

def build_dataloaders(
    cache_dir, batch_size=8, num_workers=2, in_channels=3, use_gpu_window=False,
    lung_mask=True, augment=True, limit_train=0, limit_val=0, rebuild_index=False, seed=42,
    tumor_size_low_lim=TUMOR_SIZE_LOW_LIM, zero_sl_ratio=ZERO_SL_RATIO,      # THÊM
    aug_level=1, tumor_union_p=1.0, neighbor_mm=0.0,                         # THÊM
    **kwargs,
):
    from pathlib import Path
    cache_path = Path(cache_dir).resolve()

    # --- Tự dựng lại slices.csv nếu tham số chỉ mục đổi (tránh dùng chỉ mục cũ trong im lặng) ---
    params = {"version": 2, "tumor_size_low_lim": float(tumor_size_low_lim), "zero_sl_ratio": float(zero_sl_ratio)}
    sidecar = cache_path / "slices.params.json"
    old = {"version": 1, "tumor_size_low_lim": 20.0, "zero_sl_ratio": 0.25}   # mặc định của chỉ mục cũ
    if sidecar.exists():
        try:
            old = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            pass
    if rebuild_index or old != params or not (cache_path / "slices.csv").exists():
        build_dataset_index(cache_path, tumor_size_low_lim=tumor_size_low_lim, zero_sl_ratio=zero_sl_ratio)
        sidecar.write_text(json.dumps(params), encoding="utf-8")

    common = dict(cache_dir=cache_path, in_channels=in_channels, apply_lung=lung_mask,
                  window_on_cpu=not use_gpu_window, neighbor_mm=neighbor_mm)
    train_ds = LIDC25DDataset(split="train", augment=augment, aug_level=aug_level,
                              tumor_union_p=tumor_union_p, **common)
    val_ds = LIDC25DDataset(split="val", augment=False, **common)
    # ... phần limit_train / limit_val / DataLoader giữ nguyên ...
```

**c) `mamba_unet/train_main.py`** — truyền cờ vào lời gọi `build_dataloaders(...)` trong `train_action`:

```python
        tumor_size_low_lim=args.min_train_area,
        zero_sl_ratio=args.neg_ratio,
        aug_level=args.aug_level,
        tumor_union_p=args.tumor_union_p,
        neighbor_mm=args.neighbor_mm,
```

> ⚠️ Đổi chỉ mục ⇒ **tập val đổi** (có thêm lát nhỏ/âm tính) ⇒ Val Dice 2D mới **không so trực tiếp** được với 0.7456. So sánh bằng `eval_sweep.py` (3D) trên cùng tập ca.

**Giá trị khuyến nghị:** `--min-train-area 5 --neg-ratio 0.5`. Không nên 0 tuyệt đối: nhãn là *hợp* của các reader nên các mảnh 1–3 pixel là nhiễu nhãn.
**Kỳ vọng:** recall nốt nhỏ ↑, 3D Dice ↑ (có thể Val Dice 2D mới lại *giảm* vì val khó hơn — đó là bình thường).

---

### 3.2 [P3] Đồng nhất cách che phổi giữa train và val

**`common/data_pipeline/dataset.py`** — class `LIDC25DDataset`:

1. `__init__`: thêm tham số và lưu:

```python
    def __init__(self, cache_dir, csv_path=None, split="train", in_channels=3, image_size=IMAGE_SIZE,
                 apply_lung=True, window_on_cpu=True, quantize=True, augment=False,
                 aug_level=1, tumor_union_p=1.0, neighbor_mm=0.0):       # THÊM 3 tham số cuối
        ...
        self.aug_level = int(aug_level)
        self.tumor_union_p = float(tumor_union_p)
        self.neighbor_mm = float(neighbor_mm)
```

2. `__getitem__`: thay đoạn `tumor_for_keep`:

```python
            # TRƯỚC: tumor_for_keep = msk_stack if self.split == "train" else None
            use_union = (self.split == "train") and (random.random() < self.tumor_union_p)
            tumor_for_keep = msk_stack if use_union else None
```

**Thực nghiệm:** `--tumor-union-p 0.0` (đồng nhất hoàn toàn với val) và `0.5` (trộn). Nếu P3 thật sự nghiêm trọng (xem `check_cache.py`), nên thấy gap train–val giảm và nốt sát thành ngực khá hơn. Nếu `mask_in_lung_fraction` ≈ 1 ở hầu hết ca thì tác động nhỏ.
**Giải pháp bổ sung (nếu nhiều ca bị cắt):** làm lớn mask phổi (closing/convex hull theo lát) trong `extract_lungs_mask` rồi rebuild cache — tốn công nên chỉ làm khi số liệu cho thấy cần thiết.

---

### 3.3 [P2, P13] Log Precision/Recall mỗi epoch + sửa log LR

**`common/engine/base_trainer.py`**

a) cuối `run_epoch`, thay khối return:

```python
        eps = 1e-6
        epoch_loss = total_loss / max(1, total_samples)
        epoch_dice = (2.0 * inter_sum + eps) / (pred_sum + target_sum + eps)
        union_sum = pred_sum + target_sum - inter_sum
        epoch_iou = (inter_sum + eps) / (union_sum + eps)
        epoch_prec = (inter_sum + eps) / (pred_sum + eps)      # THÊM
        epoch_rec = (inter_sum + eps) / (target_sum + eps)     # THÊM
        return {"loss": epoch_loss, "dice": epoch_dice, "iou": epoch_iou,
                "precision": epoch_prec, "recall": epoch_rec}
```

b) trong `fit`, dict `row` thêm 2 cột (sau `val_iou`):

```python
                    "val_precision": round(val_m["precision"], 5),
                    "val_recall": round(val_m["recall"], 5),
```

c) `log_msg`: đổi phần Val thành
`f"Val [Loss: {val_m['loss']:.4f}, Dice: {val_m['dice']:.4f}, P: {val_m['precision']:.3f}, R: {val_m['recall']:.3f}] | "`

d) sửa LR được log là LR *của epoch vừa chạy* (hiện đang in LR sau `scheduler.step()`): chuyển dòng

```python
                lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
```
lên **trước** khối `if self.scheduler is not None:`.

`history.csv` cũ vẫn đọc/resume được (cột mới là NaN ở các epoch cũ).

---

### 3.4 [P4, P8] Weight decay thật sự có tác dụng, loại norm/bias/`A_log`/`D`

**`mamba_unet/trainer.py`** — sửa `__init__` và `build_param_groups`:

```python
class MambaUNetTrainer(BaseTrainer):
    def __init__(self, model, criterion, backbone_name="mamba_vision_T", activation_name="relu",
                 in_channels=3, num_classes=1, freeze_backbone_epochs=0, backbone_lr_mult=0.1,
                 weight_decay: float = 1e-4,                     # THÊM
                 **kwargs):
        super().__init__(model=model, criterion=criterion, **kwargs)
        ...
        self.weight_decay = float(weight_decay)

    @staticmethod
    def _no_decay(name: str, p: torch.nn.Parameter) -> bool:
        return (p.ndim <= 1 or name.endswith(".bias") or "A_log" in name
                or name.endswith(".D") or getattr(p, "_no_weight_decay", False))

    def build_param_groups(self, lr: float) -> List[Dict[str, Any]]:
        m = self._unwrap_model()
        groups = []
        for part, mod, mult in (("decoder", m.decoder, 1.0), ("backbone", m.encoder, self.backbone_lr_mult)):
            dec, nodec = [], []
            for n, p in mod.named_parameters():
                if not p.requires_grad:
                    continue
                (nodec if self._no_decay(n, p) else dec).append(p)
            if dec:
                groups.append({"params": dec, "lr": lr * mult, "weight_decay": self.weight_decay, "name": part})
            if nodec:
                groups.append({"params": nodec, "lr": lr * mult, "weight_decay": 0.0, "name": part + "_nodecay"})
        return groups
```

Nhóm `decoder` vẫn đứng đầu ⇒ `lr` log (`param_groups[0]`) không đổi ý nghĩa.

**`mamba_unet/train_main.py`** — truyền `weight_decay=args.weight_decay` vào `MambaUNetTrainer(...)`.

**Cập nhật `_selftest` trong `trainer.py`:** nhóm tham số giờ là 4:

```python
    groups = trainer.build_param_groups(1e-4)
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {"decoder", "decoder_nodecay", "backbone", "backbone_nodecay"}
    assert by_name["decoder"]["lr"] == 1e-4 and abs(by_name["backbone"]["lr"] - 1e-5) < 1e-9
    assert by_name["decoder_nodecay"]["weight_decay"] == 0.0
```

> ⚠️ Resume một run **cũ** (2 nhóm) vào optimizer mới (4 nhóm) sẽ in cảnh báo "Không nạp được optimizer state" (code đã `try/except`) và chạy tiếp với optimizer mới — chấp nhận được; run mới thì không ảnh hưởng.

**Giá trị chạy:** `--weight-decay 0.02` (thử 0.01–0.05).
**Kỳ vọng:** giảm gap, +0.003–0.01 Dice (nhỏ, nhưng gần như miễn phí).

---

### 3.5 [P7] BN backbone khi đóng băng: làm đúng ý đồ hoặc ghi rõ

Hiện tại BN vẫn ở chế độ train trong các epoch freeze (xem bảng P7). Vì đầu vào CT lệch phân phối so với ImageNet, hành vi hiện tại có thể **có lợi**. Đề xuất: thêm cờ để *so sánh*, mặc định giữ nguyên.

**`common/engine/base_trainer.py`** — thêm 2 hook rỗng và gọi chúng:

```python
    def on_mode_set(self, is_train: bool):
        """Hook chạy ngay sau model.train(is_train) trong run_epoch."""
        pass

    def on_step_end(self):
        """Hook chạy sau mỗi optimizer step."""
        pass
```

trong `run_epoch`:

```python
        self.model.train(is_train)
        self.on_mode_set(is_train)                      # THÊM ngay sau dòng này
        ...
                if is_train and self.optimizer:
                    if self.scaler is not None and self.use_amp:
                        ...
                    else:
                        ...
                    self.on_step_end()                  # THÊM, cùng mức thụt lề với khối if/else ở trên
```

**`mamba_unet/trainer.py`:**

```python
    # __init__: thêm freeze_bn: bool = False  ->  self.freeze_bn = bool(freeze_bn); self._frozen_now = False
    # on_epoch_start: sau khi tính should_freeze ->  self._frozen_now = bool(should_freeze)

    def on_mode_set(self, is_train: bool):
        if is_train and self.freeze_bn and self._frozen_now:
            for m in self._unwrap_model().encoder.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
```

**`train_main.py`** — thêm `--freeze-bn` (store_true) vào `_add_mamba_specific_args` và truyền `freeze_bn=args.freeze_bn`.

Nếu thử `--freeze-bn`, nên đi kèm chuẩn hóa đầu vào (§6.3), vì BN stem dùng thống kê ImageNet sẽ lệch nặng với ảnh [0,1].

---

### 3.6 [P14] Các lỗi/bất nhất nhỏ

| Việc | File | Sửa |
|---|---|---|
| `load_state_dict(strict=False)` nuốt lỗi thiếu/thừa khóa | `mamba_unet/evaluate.py` `load_mamba_unet_model` | `missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)` rồi `print` nếu khác rỗng (hoặc `strict=True`) |
| Warm-start (`--init-weights`) luôn strict | `base_trainer.load_checkpoint` | `self._unwrap_model().load_state_dict(sd, strict=resume_training)`; khi không resume, in `missing/unexpected` (cần cho §6.1) |
| Docstring "264 keys" vs selftest 300 keys | `mamba_unet/model.py` đầu file | Sửa thành 300 (250 encoder + 50 decoder) |
| `evaluate.py` không cho `--split test` | `mamba_unet/evaluate.py` | `choices=["val","train","test","all"]` (cần §5.6) |
| Báo cáo mô tả sai code | `MAMBA_UNET_SYSTEM_REPORT.md` | Xem mục 8 |

---

## 4. Cải tiến bằng **tham số** (không đổi code, hoặc chỉ đổi cờ đã thêm ở trên)

| Tham số | Hiện tại | Thử | Lý do | Kỳ vọng |
|---|---|---|---|---|
| `--weight-decay` | 1e-4 | 0.01–0.05 (sau §3.4) | wd cũ vô hiệu với AdamW | Giảm gap; +0.003–0.01 |
| `--min-train-area` | 20 | 5 | Giữ lát biên/nốt nhỏ | Recall nốt nhỏ ↑; 3D Dice ↑ |
| `--neg-ratio` | 0.25 | 0.5–1.0 | Phân phối âm tính giống inference hơn | FP ↓ (kiểm tra bằng `fp_comps`) |
| `--aug-level` | 1 | 2 (sau §5.2) | Chống overfit, phá lưới cửa sổ | +0.01–0.02 |
| `--drop-path` (sau §6.4) | 0.2 cố định | 0.1 / 0.3 | Regularizer duy nhất đang chạy | ±0.005 (cần thử) |
| `--backbone-lr-mult` | 0.1 | 0.3 | LR backbone 1e-5 có thể quá thấp cho thích nghi miền CT | Thử; có thể tăng overfit |
| `--freeze-backbone-epochs` | 3 | 2–3 | Epoch 1–3 đã tốt; ít ảnh hưởng | ≈0 |
| `--loss-params pos_weight` | 4.0 | 2.0 / 1.0 nếu precision ≪ recall | `pos_weight=4` đẩy mô hình dự đoán thừa | P↑, FP↓; chọn theo `eval_sweep` |
| `--loss` | dice_bce | `focal_tversky` (α=0.3, β=0.7, γ=0.75) nếu recall ≪ precision | Phạt bỏ sót | Cải thiện recall nốt khó |
| `--epochs` / `--patience` | 50 / 12 | 30 / 8 | Plateau từ epoch ~24, chạy ablation nhanh hơn (cosine co theo `--epochs`) | Tiết kiệm ~40% thời gian |
| `--batch-size` | 8 (~6.5 GB) | 12–16 + `--lr` ×1.4 | Còn dư VRAM trên T4 | Ổn định BN/Dice-loss gộp batch |
| `--workers` | 2 | đo thử 2 vs 4 | Epoch freeze 3m19s với chỉ decoder backward — có thể nghẽn CPU (đọc mmap + `lung_keep_mask` bằng cv2) | Nhanh hơn nếu Colab có ≥ 4 vCPU |
| `--select` | `first` | `random` + `--seed 42` | Tránh lệch máy chụp ở các ID đầu | Giảm thiên lệch |
| Số ca | 500 | **1010 ca** | Đòn bẩy lớn nhất với dữ liệu ít | +0.01–0.03 (kỳ vọng thận trọng) |

Chạy toàn bộ 1010 ca (không truyền `--n-patients`):

```bash
python -m common.data_pipeline.runtime_data colab --zip auto --dest /content/data/cache_1mm
```

(500 ngẫu nhiên: thêm `--n-patients 500 --select random --seed 42`.) Thời gian mỗi epoch ~gấp đôi; Colab có thể ngắt phiên ⇒ dùng `--resume auto`.

---

## 5. Cải tiến bằng **mã nguồn** (không đổi cấu trúc mạng)

### 5.1 EMA trọng số (giảm nhiễu chọn "best epoch", thường tăng nhẹ Dice)

**`mamba_unet/trainer.py`** — thêm lớp ở đầu file:

```python
class _EMA:
    """EMA tham số; BN running stats và buffer nguyên/số đếm thì sao chép."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point and not (k.endswith("running_mean") or k.endswith("running_var")):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def swap_in(self, model: nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def swap_out(self, model: nn.Module):
        model.load_state_dict(self.backup)
        self.backup = None
```

trong `MambaUNetTrainer`:

```python
    # __init__: thêm ema_decay: float = 0.0  ->  self.ema_decay = float(ema_decay); self.ema = None

    def on_epoch_start(self, epoch):
        ...  # giữ nguyên phần cũ
        if self.ema_decay > 0 and self.ema is None:
            self.ema = _EMA(self._unwrap_model(), self.ema_decay)

    def on_step_end(self):
        if self.ema is not None:
            self.ema.update(self._unwrap_model())

    def run_epoch(self, loader, is_train=True, **kw):            # val dùng trọng số EMA
        if (not is_train) and self.ema is not None:
            m = self._unwrap_model()
            self.ema.swap_in(m)
            try:
                return super().run_epoch(loader, is_train=False, **kw)
            finally:
                self.ema.swap_out(m)
        return super().run_epoch(loader, is_train=is_train, **kw)

    def save_checkpoint(self, filename, epoch, val_metrics):     # checkpoint lưu trọng số EMA
        if self.ema is None:
            return super().save_checkpoint(filename, epoch, val_metrics)
        m = self._unwrap_model()
        self.ema.swap_in(m)
        try:
            return super().save_checkpoint(filename, epoch, val_metrics)
        finally:
            self.ema.swap_out(m)
```

`train_main.py`: thêm `--ema` (float, mặc định 0.0 = tắt) và truyền `ema_decay=args.ema`.
Lưu ý: `last_model.pth` chứa trọng số EMA nên resume sẽ tiếp tục từ trọng số EMA (lệch nhẹ so với optimizer state) — chấp nhận được. Với ~650 step/epoch, `decay=0.999` có "bán kỳ" ≈ 1 epoch.
**Kỳ vọng:** +0.003–0.01 Dice, đường cong val mượt hơn.

### 5.2 Augmentation mức 2: affine + tương phản/nhiễu

**`common/data_pipeline/dataset.py`** — thêm 2 phương thức vào `LIDC25DDataset`:

```python
    def _affine_pair(self, img_chw, mask_1hw):
        """Xoay ±15°, scale 0.85–1.15, dịch ±32 px (đồng bộ ảnh 3 kênh và mask)."""
        h, w = img_chw.shape[1:]
        ang, sc = random.uniform(-15, 15), random.uniform(0.85, 1.15)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, sc)
        M[:, 2] += (random.uniform(-32, 32), random.uniform(-32, 32))
        air = 0.0 if self.window_on_cpu else -1000          # không khí: 0 (sau cửa sổ) hoặc -1000 HU
        img = np.stack([cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=air) for c in img_chw])
        msk = cv2.warpAffine(mask_1hw[0], M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)[None]
        return np.ascontiguousarray(img), np.ascontiguousarray(msk)

    @staticmethod
    def _intensity_jitter(img):
        """Chỉ cho ảnh đã cửa sổ [0,1]: nhân tương phản, thêm nhiễu Gaussian (không đụng vùng không khí = 0)."""
        img = img * random.uniform(0.9, 1.1)
        sigma = random.uniform(0.0, 0.01)
        if sigma > 0:
            img = img + np.random.normal(0.0, sigma, img.shape).astype(np.float32) * (img > 0)
        return np.clip(img, 0.0, 1.0).astype(np.float32)
```

trong `__getitem__`, thay khối `if self.augment:`:

```python
        if self.augment:
            img_out, center_mask = self._augment_pair(img_out, center_mask)
            if self.aug_level >= 2:
                if random.random() < 0.5:
                    img_out, center_mask = self._affine_pair(img_out, center_mask)
                if self.window_on_cpu:
                    img_out = self._intensity_jitter(img_out)
```

Dịch chuyển ±32 px cũng làm lưới cửa sổ stage 3–4 trượt so với giải phẫu (giảm P6). **Kỳ vọng:** +0.01–0.02 Dice, gap giảm; có thể cần nhiều epoch hơn một chút để hội tụ.

### 5.3 Giảm lãng phí pad: `IMAGE_SIZE` = 448 (bội của 224 = 14 × 16 px)

Với 448: stage 3 là 28×28 = đúng 2×2 cửa sổ 14; stage 4 là 14×14 = đúng 2×2 cửa sổ 7 ⇒ **không pad**, số token stage 3 giảm 1764 → 784 (×2.25), toàn mạng nhanh hơn ~20–25%.

**`configs/config.py`:**

```python
import os
IMAGE_SIZE = int(os.environ.get("DUNEAI_IMAGE_SIZE", 512))     # mặc định giữ 512
```

Chạy train **và** eval cùng biến môi trường: `DUNEAI_IMAGE_SIZE=448 python -u -m mamba_unet.train_main ...` và `DUNEAI_IMAGE_SIZE=448 python -m mamba_unet.evaluate ...` (lệch giá trị giữa train và eval sẽ cho kết quả sai; nếu muốn an toàn hơn có thể lưu `image_size` vào `checkpoint_meta` rồi đọc lại ở `evaluate.py`).
Trước khi dùng, chạy `tools/check_cache.py <cache> 448`: nếu có ca bị mất nhãn do crop giữa thì không dùng 448 hoặc cắt theo bbox phổi.
**Kỳ vọng:** tốc độ ↑ ~20%; Dice ≈ không đổi (có thể +0.00–0.01 nhờ cửa sổ không pad).

### 5.4 Chuẩn hóa khoảng cách z cho 2.5D (`--neighbor-mm`)

**`common/data_pipeline/offline_builder.py`** — thêm cột `spacing_z` vào mỗi dòng `rows.append({...})`:

```python
                "spacing_z": round(float(meta.get("orig_spacing_xyz", [1.0, 1.0, 1.0])[2]), 3),
```

**`common/data_pipeline/dataset.py`:**

```python
    def _neighbor_indices(self, z, num_slices, step=1):
        return [min(max(0, z + off * step), num_slices - 1) for off in range(-self.half_k, self.half_k + 1)]

    # trong __getitem__, thay dòng z_indices = ...
        step = 1
        if self.neighbor_mm > 0 and "spacing_z" in row.index and float(row["spacing_z"]) > 0:
            step = max(1, int(round(self.neighbor_mm / float(row["spacing_z"]))))
        z_indices = self._neighbor_indices(z, num_slices, step)
```

**`common/eval3d/volume_inference.py`** — thêm tham số `neighbor_mm: float = 0.0` vào `predict_patient_volume` (và chuyển tiếp từ `evaluate_dataset_3d`, hai CLI):

```python
    step = max(1, int(round(neighbor_mm / spacing_z))) if neighbor_mm > 0 else 1
    ...
            z_idx = [min(max(0, z + off * step), num_slices - 1) for off in range(-half_k, half_k + 1)]
```

Với `--neighbor-mm 2.0`: lát 0.6 mm → bước 3; 1.25 mm → 2; ≥ 2.5 mm → 1. Train và eval **phải dùng cùng giá trị**. **Kỳ vọng:** +0.00–0.01 (cải thiện nhất quán, hiệu quả rõ hơn nếu tập có nhiều độ dày lát khác nhau).

### 5.5 Resize ảnh về 1 mm bằng `area` thay `nearest` (tuỳ chọn, tốn công)

```bash
python common/data_pipeline/medical_prep.py build --nrrd-dir <nrrd> --cache-dir <cache_new> \
  --interp area --extract-lungs --overwrite --workers 4
```

**Bắt buộc `--extract-lungs`:** `build_patient_cache` ghi lại `meta.json`; thiếu cờ này mất khóa `lung_ok` và có thể làm mask phổi không còn đồng bộ (lung zeros ⇒ ảnh toàn -1000 HU). Tách phổi chậm; chỉ làm khi đã xong các cải tiến rẻ hơn. **Kỳ vọng:** nhỏ (+0.00–0.01), ảnh nốt nhỏ sạch hơn.

### 5.6 Tách tập test riêng (và hard-negative)

**`offline_builder.build_dataset_index`** — thêm `test_split=0.0` vào chữ ký và thay khối chia tập:

```python
    if len(shuffled_pids) == 1:
        val_patients, test_patients, train_patients = set(), set(), set(shuffled_pids)
    else:
        val_size = max(1, int(round(len(shuffled_pids) * val_split)))
        test_size = int(round(len(shuffled_pids) * test_split)) if test_split > 0 else 0
        val_patients = set(shuffled_pids[:val_size])                      # val giữ NGUYÊN như cũ
        test_patients = set(shuffled_pids[val_size:val_size + test_size])
        train_patients = set(shuffled_pids[val_size + test_size:])
    ...
        split = "val" if pid in val_patients else ("test" if pid in test_patients else "train")
```

Val giữ đúng tập ca cũ (cùng seed và `val_size`), chỉ phần train bị rút bớt. Dùng `--split test` trong `evaluate.py` / `eval_sweep.py` và **chỉ báo cáo test một lần** ở cuối.

**Hard-negative mining (sau khi có model, mục 2.1):** thêm dòng âm tính vào `slices.csv` cho các lát model dự đoán dương tính giả trên tập train:

```python
# tools/add_hard_negatives.py  (rút gọn)
import sys; from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.eval3d.volume_inference import predict_patient_volume
from mamba_unet.evaluate import load_mamba_unet_model

cache, ckpt, MAXN = Path(sys.argv[1]), sys.argv[2], 10
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_mamba_unet_model(ckpt, dev)
df = pd.read_csv(cache / "slices.csv")
new = []
for pid in sorted(df[df.split == "train"].patient_id.unique()):
    r = predict_patient_volume(model, cache / pid, device=dev)
    fp = (r["raw_mask_1mm"] > 0) & (r["gt_mask_1mm"] == 0)
    cnt = fp.reshape(fp.shape[0], -1).sum(1)
    have = set(df[df.patient_id == pid].z)
    zs = [int(z) for z in np.argsort(-cnt) if cnt[z] >= 20 and int(z) not in have][:MAXN]
    for z in zs:
        new.append(dict(patient_id=pid, z=z, num_slices=int(fp.shape[0]), has_tumor=0,
                        tumor_area_mm2=0.0, has_lung_mask=1, split="train",
                        spacing_z=float(r["meta"].get("orig_spacing_xyz", [1, 1, 1])[2])))
pd.concat([df, pd.DataFrame(new)]).to_csv(cache / "slices.csv", index=False)
print("đã thêm", len(new), "lát âm tính khó")
```

Sao lưu `slices.csv` trước. Lưu ý `build_dataloaders` (§3.1) dựng lại chỉ mục khi tham số đổi ⇒ chạy script này **sau** khi tham số chỉ mục đã ổn định, và không đổi `--min-train-area/--neg-ratio` nữa. **Kỳ vọng:** FP/ca ↓ rõ rệt; Dice 3D ↑ nếu FP là nút thắt.

---

## 6. Cải tiến **kiến trúc**

### 6.1 Skip độ phân giải cao (H/2 và H/1) — thay đổi kiến trúc quan trọng nhất

Mục tiêu: sửa P5 mà **không đụng backbone pretrained** (250 khóa giữ nguyên), mặc định tắt nên golden keys/test cũ vẫn đúng.

**a) `mamba_unet/backbone/mamba_vision.py`**

Trong `MambaVision.__init__`, thêm (sau `self.in_chans = in_chans`): `self.return_stem_feat = False`. Thay `forward_features`:

```python
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.return_stem_feat:
            stem = self.patch_embed.conv_down[:3](x)            # (B, in_dim, H/2, W/2): Conv-BN-ReLU đầu tiên
            x = self.patch_embed.conv_down[3:](stem)            # phần còn lại của PatchEmbed (dùng chung trọng số)
        else:
            stem = None
            x = self.patch_embed(x)
        features = []
        for level in self.levels:
            x, feat = level(x, return_feature_before_downsample=True)
            features.append(feat)
        return ([stem] + features) if stem is not None else features
```

**b) `mamba_unet/decoder.py`** — `UNetDecoder`:

```python
    def __init__(self, encoder_channels=None, num_classes=1, final_channels=32, act=None, init_bias=-4.0,
                 high_res_skip: bool = False, stem_channels: int = 32, in_channels: int = 3):
        super().__init__()
        ...  # giữ nguyên c1..c4, up1, up2, up3
        self.high_res_skip = bool(high_res_skip)
        if self.high_res_skip:
            mid = max(final_channels, 48)
            self.up4 = UpBlock(c1, stem_channels, mid, use_conv_transpose=False, act=act)       # H/4 -> H/2
            self.hr_stem = DoubleConv(in_channels, 16, act=act)                                  # nhánh H/1 từ ảnh gốc
            self.up5 = UpBlock(mid, 16, final_channels, use_conv_transpose=False, act=act)      # H/2 -> H/1
        else:
            self.final_up = nn.Sequential(
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
                DoubleConv(c1, final_channels, act=act),
            )
        self.outc = OutConv(final_channels, num_classes, init_bias=init_bias)

    def forward(self, features: List[torch.Tensor], x_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.high_res_skip:
            f0, f1, f2, f3, f4 = features
        else:
            f1, f2, f3, f4 = features
        d1 = self.up1(f4, f3)
        d2 = self.up2(d1, f2)
        d3 = self.up3(d2, f1)
        if self.high_res_skip:
            x = self.up5(self.up4(d3, f0), self.hr_stem(x_in))
        else:
            x = self.final_up(d3)
        return self.outc(x)
```

**c) `mamba_unet/model.py`** — `MambaUNet`:

```python
    def __init__(self, in_channels=3, num_classes=1, backbone="mamba_vision_T", pretrained=False,
                 final_channels=32, act=None, init_bias=-4.0,
                 high_res_skip: bool = False, drop_path_rate: float = 0.2,
                 input_norm: Optional[Tuple[float, float]] = None, **kwargs):
        ...
        if isinstance(backbone, str):
            self.encoder = build_mambavision_backbone(name=backbone, pretrained=pretrained,
                                                      in_chans=in_channels, drop_path_rate=drop_path_rate)
        ...
        self.high_res_skip = bool(high_res_skip)
        if self.high_res_skip:
            self.encoder.return_stem_feat = True
        self.decoder = UNetDecoder(
            encoder_channels=encoder_channels, num_classes=num_classes, final_channels=final_channels,
            act=act, init_bias=init_bias, high_res_skip=self.high_res_skip,
            stem_channels=getattr(self.encoder, "in_dim", 32), in_channels=in_channels,
        )
        # chuẩn hóa đầu vào (xem §6.3); buffer không persistent => không thêm khóa vào state_dict
        self.input_norm = input_norm is not None
        mean, std = input_norm if self.input_norm else (0.0, 1.0)
        self.register_buffer("_in_mean", torch.tensor(float(mean)), persistent=False)
        self.register_buffer("_in_std", torch.tensor(float(std)), persistent=False)

    def forward(self, x):
        if self.input_norm:
            x = (x - self._in_mean) / self._in_std
        features = self.encoder(x)
        return self.decoder(features, x)
```

(thêm `Tuple` vào `from typing import ...`). Trong `build_mamba_unet` đọc thêm `high_res_skip`, `drop_path_rate`, `input_norm` bằng `kwargs.get(...)` giống các tham số hiện có.

**d) `train_main.py` / `trainer.py` / `evaluate.py`:**

- `_add_mamba_specific_args`: thêm `--hr-skip` (store_true), `--drop-path` (float, 0.2), `--input-norm` (`nargs=2, type=float`, mặc định None).
- `train_action`: `build_mamba_unet(..., high_res_skip=args.hr_skip, drop_path_rate=args.drop_path, input_norm=tuple(args.input_norm) if args.input_norm else None)`; ghi các giá trị này vào `model_meta` của `create_run`.
- `MambaUNetTrainer.checkpoint_meta()`: thêm `"high_res_skip"`, `"input_norm"` (list hoặc None) vào dict (nhận từ `__init__`, ví dụ `high_res_skip: bool = False, input_norm=None`).
- `evaluate.load_mamba_unet_model`: dựng model với `high_res_skip=bool(meta.get("high_res_skip", False))` và `input_norm=tuple(meta["input_norm"]) if meta.get("input_norm") else None`. **Nếu thiếu bước này, model HR sẽ được dựng sai ở eval.**

**e) Khởi động từ checkpoint cũ (đỡ phải train từ đầu):**

```bash
... --hr-skip --init-weights mamba_unet/runs/<run_cũ>/best_model.pth --freeze-backbone-epochs 0
```

nhờ `strict=resume_training` ở §3.6, các khóa `up1–up3`, `outc` và toàn bộ backbone được nạp; `up4`, `up5`, `hr_stem` khởi tạo ngẫu nhiên.

**Chi phí:** `hr_stem` + `up5` chạy ở độ phân giải đầy đủ nhưng kênh nhỏ (16/32); tổng FLOPs tương đương `final_up` cũ (DoubleConv 80→32 ở 512²), thêm `up4` ở 256². Bộ nhớ tăng nhẹ.
**Kỳ vọng:** biên nét hơn, recall nốt nhỏ ↑ rõ nhất ở nhóm < 10 mm; +0.01–0.03 Dice lát, và 3D recall nốt nhỏ ↑. Đây là cải tiến có kỳ vọng lớn nhất về kiến trúc nhưng vẫn cần ablation.

### 6.2 Cửa sổ cô lập ở stage 3–4 (P6) — các mức từ rẻ đến đắt

1. **Rẻ:** dịch chuyển ngẫu nhiên (aug-level 2, §5.2) + `IMAGE_SIZE=448` (§5.3).
2. **Trung bình:** dịch lưới cửa sổ (shifted window) ở các block lẻ: trong `MambaVisionLayer.forward` với `not self.conv`, dùng `torch.roll(x, (-ws//2, -ws//2), (2, 3))` trước `window_partition` cho block có chỉ số lẻ rồi roll ngược; **không cần mask** nếu nhận chấp nhận ghép vòng (cyclic) ở biên. Cần sắp xếp lại vòng lặp `for blk in self.blocks` thành `for j, blk in enumerate(...)` và partition/reverse **từng block** (tốn thêm một lần partition/reverse mỗi block). Thay đổi này không làm đổi khóa tham số ⇒ vẫn nạp được pretrained, nhưng *thay đổi hành vi so với pretrain* nên cần fine-tune đủ lâu.
3. **Đắt:** thêm một khối attention toàn cục nhẹ ở bottleneck (H/32 = 14×14 token với 448) sau stage 4; thêm khóa mới (decoder/bottleneck).

Chỉ làm bước 2–3 nếu `eval_sweep` cho thấy lỗi ở nốt lớn/ngữ cảnh rộng (vd. phân biệt mạch máu – nốt), vì với nốt nhỏ thì §6.1 hiệu quả hơn.

### 6.3 Chuẩn hóa đầu vào (`--input-norm MEAN STD`)

Tính mean/std trên vài trăm lát train đã qua cửa sổ HU + che phổi:

```python
from common.data_pipeline.dataset import LIDC25DDataset
import numpy as np, random
ds = LIDC25DDataset("/content/data/cache_1mm", split="train", augment=False)
xs = np.stack([ds[i]["image"].numpy() for i in random.sample(range(len(ds)), 300)])
print(xs.mean(), xs.std())
```

Rồi chạy `--input-norm <mean> <std>`. Chuẩn hóa nằm *trong* `MambaUNet.forward` nên train, eval và TTA tự nhất quán (buffer không persistent ⇒ không đổi `state_dict`/golden keys; `input_norm` phải nằm trong `checkpoint_meta`, §6.1d). Đặc biệt nên dùng cùng `--freeze-bn`. **Kỳ vọng:** nhỏ (+0.00–0.01), thấp rủi ro.

### 6.4 Các hướng nâng cao hơn (làm sau, khi đã có số liệu)

- **Deep supervision:** thêm đầu ra phụ 1×1 trên `d1`, `d2` (upsample về kích thước nhãn), loss phụ trọng số 0.3/0.15. Giúp gradient tới decoder sâu và nốt nhỏ ổn định hơn; thêm khóa mới (head phụ), cần sửa `forward` trả về tuple khi train.
- **Nhãn mềm theo số reader:** hiện mask là *hợp* của các reader (biên mờ, có vùng chỉ 1 reader đánh dấu). Nếu export lại từng reader từ DICOM-SEG, dùng tỷ lệ đồng thuận (0, 0.25, …, 1) làm nhãn mềm cho BCE và ngưỡng ≥ 0.5 cho Dice. Giảm nhiễu nhãn, nhưng cần xây lại cache mask.
- **Huấn luyện theo crop quanh nốt:** crop 224/448 với ~50% crop chứa u (cho phép batch lớn hơn, nhiều biến thiên hơn), suy luận toàn khung. Tương thích với `IMAGE_SIZE=448`.
- **Hậu xử lý:** thay lọc theo "diện tích lát lớn nhất ≥ 20 mm²" bằng lọc theo xác suất trung bình của cụm + thể tích; điều chỉnh sau khi biết phân bố FP từ `eval_sweep`.

---

## 7. Kế hoạch thực nghiệm (từng thay đổi một, mỗi cấu hình ≥ 2 seed nếu có thể)

Nhiễu của val ±0.005–0.01 ⇒ chênh < 0.01 Dice **không** nên coi là cải thiện nếu chỉ 1 seed. Dùng `--epochs 30 --patience 8` cho ablation, rồi chạy bản cuối dài hơn. Lệnh cơ sở (từ nhật ký của bạn), chỉ thay phần in đậm theo từng bước:

```bash
DUNEAI_MAMBA_BACKEND=cuda python -u -m mamba_unet.train_main train \
  --cache-dir /content/data/cache_1mm --backbone mamba_vision_T --pretrained \
  --freeze-backbone-epochs 3 --backbone-lr-mult 0.1 --activation relu --loss dice_bce \
  --loss-params '{"bce_weight": 0.3, "dice_weight": 0.7, "pos_weight": 4.0}' \
  --epochs 30 --patience 8 --batch-size 8 --lr 0.0001 --scheduler cosine --grad-clip 1.0 \
  --workers 2 --seed 42 --run-name <TÊN_RUN> --tag colab_mamba \
  <CỜ_THAY_ĐỔI>
```

| Run | `<CỜ_THAY_ĐỔI>` | Cần code ở | Kiểm tra bằng |
|---|---|---|---|
| R0 | *(baseline đã chạy)* + `eval_sweep` | §2 | 3D Dice, FP/ca, độ nhạy theo kích thước |
| R1 | `--weight-decay 0.02` | §3.4 | gap train–val, Dice |
| R2 | `--weight-decay 0.02 --ema 0.999` | §5.1 | độ mượt val, Dice |
| R3 | `... --aug-level 2` | §5.2 | gap, Dice |
| R4 | `... --min-train-area 5 --neg-ratio 0.5` | §3.1 | **`eval_sweep` 3D** (val 2D không so được) |
| R5 | `... --tumor-union-p 0.0` | §3.2 | nốt dính thành ngực, gap |
| R6 | `... --hr-skip --init-weights <best R4> --freeze-backbone-epochs 0` | §6.1 | độ nhạy nốt < 10 mm |
| R7 | `DUNEAI_IMAGE_SIZE=448 ...` | §5.3 | thời gian/epoch, Dice |
| R8 | chỉnh `pos_weight`/`--loss` theo P/R | — | P, R, `fp_comps` |
| R9 | bản tốt nhất + **toàn bộ 1010 ca**, `--epochs 40` | — | **test set** (§5.6) |

Thứ tự gợi ý nếu thời gian eo hẹp (Colab T4, mỗi run ~2.5–3.5 giờ): **R0 → R4 → R1+R2 → R3 → R6 → R9**.

**Kỳ vọng tổng hợp (rất thô, chỉ để định hướng):** gộp các cải tiến có thể đưa Dice 3D/val lên vài điểm phần trăm (khoảng 0.02–0.06) và giảm đáng kể FP; mức cụ thể phụ thuộc vào kết quả `eval_sweep` của R0.

---

## 8. Chỗ báo cáo `MAMBA_UNET_SYSTEM_REPORT.md` cần chỉnh cho khớp code

| Vị trí trong báo cáo | Báo cáo ghi | Thực tế trong code |
|---|---|---|
| Mục 1.3 (MambaVisionMixer) | Quét hai chiều (bi-directional scan) | `selective_scan_fn` gọi **một lần, một chiều**; nhánh `z` là conv1d + SiLU **không qua SSM**, ghép với `y` bằng `torch.cat` (cách MambaVision bù cho tính nhân quả) |
| Mục 1.2 và bảng `model.py` | "4 skip connections", UpBlock dùng ConvTranspose2d | 3 skip (`f1, f2, f3`; `f4` là bottleneck), upsample **bilinear** (`use_conv_transpose=False`); `final_up` = Upsample ×4 + DoubleConv |
| Mục 1.4 | Bias -4 (≈1.8%) "tương ứng tần suất thực" | Mục 1.1 nói pixel dương tính chỉ 0.1–0.5%; 1.8% cao hơn prior thật. Bỏ "triệt để ngăn gradient bùng nổ" (chỉ ổn định giai đoạn đầu) |
| Mục 1.5 | ~3.9 it/s, ~31 ảnh/s | Chỉ đúng khi backbone **đóng băng** (3m19s/epoch). Mở băng: 5m16s/epoch (~2.2 it/s) |
| Mục 4.2 | Lọc cụm có thể tích < 20 mm³; loại từng lát < 20 mm² | `min_volume_mm3` mặc định **0**; code giữ cụm có **diện tích lát lớn nhất ≥ 20 mm²** (không cắt từng lát). Đơn vị 20 mm² vs mm³ bị lẫn |
| Mục 4.3 và bảng tệp 32 | ASD, Specificity, "làm mịn biên" | Code chỉ có Dice, IoU, Precision, Recall, thể tích, HD95; hậu xử lý chỉ có lấp lỗ + lọc thành phần liên thông |
| Mục 3.2 (`run_config.json` mẫu) | best 0.7412 ở epoch 21 | Run thực này: best **0.7456** ở epoch **30**, dừng sớm epoch 42 (`--epochs 50`) |
| Mục 3.5 | best_model ~250 MB, fp16 ~60 MB | Ước lượng ~36M tham số ⇒ fp16 ~70 MB, checkpoint đầy đủ (kèm AdamW + history) lớn hơn nhiều; nên **đo lại** bằng `ls -lh` |
| Mục 1.3/2.2 (`weight-decay`) | "giúp chống overfit" | Với AdamW wd=1e-4 gần như không tác dụng (xem P8) |
| `model.py` docstring | 264 keys | selftest và báo cáo là 300 keys |

Nên bổ sung vào báo cáo (đây là phần giá trị cho phản biện): phân tích gap train–val, ý nghĩa của val phẳng khi LR giảm, hạn chế của val 2D (P1), và bảng ablation R0–R9 ở mục 7.
