"""
check_native_pipeline.py  — chạy 1 lần trên Colab TRƯỚC khi train (mất ~1 phút):

    !python check_native_pipeline.py --config configs/config_native.yaml --data.root_dir=/content/segmented_slices

Kiểm tra: (1) batch train là crop 192, (2) batch val/test giữ kích thước gốc (pad bội số 32),
(3) mask nhị phân và crop thật sự chứa nốt, (4) 1 bước forward/backward chạy được, (5) không còn
cảnh báo Affine. In kích thước ảnh trên đĩa để bạn thấy chúng KHÔNG phải 256x256.
"""
import argparse
import warnings

import numpy as np
import torch
import yaml

from models import build_model
from utils import get_dataloaders, build_loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_native.yaml")
    args, unknown = ap.parse_known_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    for a in unknown:  # override kiểu --data.root_dir=...
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            d = cfg
            ks = k.split(".")
            for kk in ks[:-1]:
                d = d.setdefault(kk, {})
            d[ks[-1]] = v
    cfg["train"]["limit_train_samples"] = 256
    cfg["train"]["limit_val_samples"] = 64
    cfg["train"]["limit_test_samples"] = 64
    cfg["train"]["num_workers"] = 2

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        train_loader, val_loader, _ = get_dataloaders(cfg)
    bad = [str(x.message) for x in w if "not valid" in str(x.message)]
    print("Cảnh báo tham số không hợp lệ:", bad or "không có")

    x, y = next(iter(train_loader))
    print(f"[train] images {tuple(x.shape)} masks {tuple(y.shape)} | mask unique {torch.unique(y).tolist()}")
    has_pos = (y.flatten(1).sum(1) > 0).float().mean().item()
    print(f"[train] tỉ lệ crop có nốt trong batch: {has_pos:.2f} (kỳ vọng ~0.7–0.85)")

    sizes = set()
    for xv, yv in val_loader:
        sizes.add(tuple(xv.shape[-2:]))
        assert xv.shape[-2] % 32 == 0 and xv.shape[-1] % 32 == 0
    print(f"[val] các kích thước batch (đã pad bội số 32): {sorted(sizes)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(dev)
    crit = build_loss(cfg).to(dev)
    out = model(x.to(dev))
    loss = crit(out, y.to(dev))
    loss.backward()
    print(f"[fwd/bwd] out {tuple(out.shape)} loss {loss.item():.4f}  -> OK")


if __name__ == "__main__":
    main()
