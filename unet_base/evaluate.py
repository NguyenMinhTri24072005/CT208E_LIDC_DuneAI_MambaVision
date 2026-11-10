# -*- coding: utf-8 -*-
"""
unet_base/evaluate.py - CLI suy luận và đánh giá 3D cấp độ bệnh nhân cho kiến trúc UNetBase.

Các chế độ hoạt động:
  - selftest: Kiểm thử tự động chu trình suy luận 3D, hậu xử lý và xuất file NRRD.
  - predict : Suy luận 3D toàn khối cho 1 bệnh nhân và xuất file mặt nạ NRRD chuẩn y tế.
  - eval    : Đánh giá 3D trên tập dữ liệu validation hoặc toàn tập, xuất bảng CSV/JSON.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import TUMOR_SIZE_LOW_LIM
from common.engine.checkpoint_io import load_checkpoint_file
from common.eval3d.eval_runner import evaluate_dataset_3d
from common.eval3d.nrrd_io import load_nrrd_mask, save_prediction_nrrd
from common.eval3d.volume_inference import predict_patient_volume
from unet_base.model import UNetBase


def load_unet_base_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    features: int = 64,
    activation: str = "relu",
    in_channels: int = 3,
    classes: int = 1,
) -> UNetBase:
    """Nạp trọng số checkpoint an toàn vào mô hình UNetBase."""
    ckpt = load_checkpoint_file(checkpoint_path, map_location=str(device))
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    meta = ckpt.get("meta", {})

    saved_feat = meta.get("features", None)
    if saved_feat:
        features = int(saved_feat)
    else:
        for k in ("encoder.inc.double_conv.0.weight", "inc.double_conv.0.weight"):
            if k in sd:
                features = int(sd[k].shape[0])
                in_channels = int(sd[k].shape[1])
                break

    saved_act = meta.get("activation", activation)
    model = UNetBase(in_channels=in_channels, classes=classes, features=features, act=saved_act)

    cleaned_sd = {}
    for k, v in sd.items():
        k_clean = k[7:] if k.startswith("module.") else k
        if k_clean.startswith(("inc.", "down1.", "down2.", "down3.", "down4.")):
            k_clean = "encoder." + k_clean
        elif k_clean.startswith(("up1.", "up2.", "up3.", "up4.", "outc.")):
            k_clean = "decoder." + k_clean
        cleaned_sd[k_clean] = v

    model.load_state_dict(cleaned_sd, strict=False)
    model.to(device)
    model.eval()
    return model


def predict_action(args):
    """Suy luận 3D cho 1 ca bệnh nhân và tùy chọn xuất file NRRD."""
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    model = load_unet_base_model(
        checkpoint_path=args.checkpoint,
        device=device,
        features=args.features,
    )
    res = predict_patient_volume(
        model=model,
        patient_cache_dir=args.patient_cache,
        device=device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        min_slice_area_mm2=args.min_area,
        tta=args.tta,
        use_amp=not args.no_amp,
    )
    m_raw, m_post = res["metrics_raw"], res["metrics_post"]
    print("=" * 80)
    print(f"🩻 KẾT QUẢ SUY LUẬN 3D (UNetBase): {Path(args.patient_cache).name}")
    print("=" * 80)
    print(f"  - 3D Dice (Trước hậu xử lý): {m_raw['dice_3d']:.4f} ({m_raw['pred_volume_mm3']} mm³)")
    print(f"  - 3D Dice (Sau hậu xử lý)  : {m_post['dice_3d']:.4f} ({m_post['pred_volume_mm3']} mm³)")
    print(f"  - Thể tích u thực tế (GT)  : {m_post['gt_volume_mm3']} mm³ | HD95: {m_post['hd95_mm']} mm")

    if args.out_nrrd:
        save_prediction_nrrd(res["clean_mask_1mm"], res["meta"], args.out_nrrd, args.ref_nrrd)
        print(f"💾 Đã xuất file mặt nạ 3D NRRD tại: {args.out_nrrd}")


def eval_action(args):
    """Đánh giá 3D trên danh sách bệnh nhân và xuất báo cáo CSV/JSON."""
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    model = load_unet_base_model(
        checkpoint_path=args.checkpoint,
        device=device,
        features=args.features,
    )
    evaluate_dataset_3d(
        model=model,
        cache_dir=args.cache_dir,
        split=args.split,
        batch_size=args.batch_size,
        threshold=args.threshold,
        min_slice_area_mm2=args.min_area,
        tta=args.tta,
        out_csv=args.out_csv,
        out_json=args.out_json,
        device=device,
        use_amp=not args.no_amp,
    )


def selftest_action(_):
    """Kiểm thử tự động chu trình suy luận 3D cho UNetBase."""
    print("🧪 Bắt đầu selftest unet_base.evaluate...")
    tmp = Path(tempfile.mkdtemp())
    try:
        p_dir = tmp / "LIDC-IDRI-0001"
        p_dir.mkdir(parents=True)
        img = np.full((8, 64, 64), -800, dtype=np.int16)
        msk = np.zeros((8, 64, 64), dtype=np.uint8)
        lung_c = np.zeros((8, 64, 64), dtype=np.uint8)
        lung_c[1:7, 10:54, 10:54] = 1
        msk[3:5, 25:35, 25:35] = 1
        img[3:5, 25:35, 25:35] = 50

        np.save(p_dir / "image_1mm.npy", img)
        np.save(p_dir / "mask_1mm.npy", msk)
        np.save(p_dir / "lung_1mm.npy", lung_c)
        with open(p_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "orig_shape_zyx": [8, 64, 64],
                "orig_spacing_xyz": [0.78, 0.78, 1.5],
                "shape_1mm_zyx": [8, 64, 64],
                "lung_ok": True,
            }, f)

        model = UNetBase(in_channels=3, classes=1, features=8)
        ckpt_path = tmp / "tiny_unet.pth"
        torch.save({"model_state_dict": model.state_dict(), "meta": {"arch": "unet_base", "features": 8}}, ckpt_path)

        loaded_model = load_unet_base_model(ckpt_path, device=torch.device("cpu"), features=8)
        res = predict_patient_volume(
            loaded_model, p_dir, batch_size=2, image_size=64, tta=True, use_amp=False
        )
        assert res["prob_1mm"].shape == (8, 64, 64)
        assert res["clean_mask_1mm"].shape == (8, 64, 64)
        assert "dice_3d" in res["metrics_post"]

        # Kiểm tra xuất và đọc file NRRD
        out_nrrd = tmp / "test_out.nrrd"
        save_prediction_nrrd(res["clean_mask_1mm"], res["meta"], out_nrrd)
        assert out_nrrd.exists(), "File NRRD không được tạo!"
        arr_read, sp = load_nrrd_mask(out_nrrd)
        assert arr_read.shape == (8, 64, 64)

        print("✅ Selftest unet_base.evaluate hoàn tất thành công!")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Đánh giá 3D cấp độ bệnh nhân (UNetBase)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # 1. selftest
    selftest_p = subparsers.add_parser("selftest", help="Kiểm thử tự động pipeline suy luận 3D")
    selftest_p.set_defaults(func=selftest_action)

    # 2. predict
    predict_p = subparsers.add_parser("predict", help="Suy luận 3D cho 1 ca bệnh nhân và xuất file NRRD")
    predict_p.add_argument("--patient-cache", required=True, help="Thư mục cache_1mm/<patient_id>")
    predict_p.add_argument("--checkpoint", required=True, help="File trọng số checkpoint (.pth)")
    predict_p.add_argument("--out-nrrd", default=None, help="Đường dẫn xuất file mặt nạ 3D (.nrrd)")
    predict_p.add_argument("--ref-nrrd", default=None, help="Đường dẫn file ảnh CT gốc để copy header tọa độ")
    predict_p.add_argument("--features", type=int, default=64, help="Số kênh đặc trưng cơ sở UNetBase")
    predict_p.add_argument("--batch-size", type=int, default=8)
    predict_p.add_argument("--threshold", type=float, default=0.5)
    predict_p.add_argument("--min-area", type=float, default=TUMOR_SIZE_LOW_LIM)
    predict_p.add_argument("--tta", action="store_true", help="Bật Test-Time Augmentation (lật ngang/dọc)")
    predict_p.add_argument("--no-cuda", action="store_true")
    predict_p.add_argument("--no-amp", action="store_true")
    predict_p.set_defaults(func=predict_action)

    # 3. eval
    eval_p = subparsers.add_parser("eval", help="Đánh giá 3D trên toàn bộ tập dữ liệu")
    eval_p.add_argument("--cache-dir", default="data/cache_1mm", help="Thư mục dữ liệu cache_1mm")
    eval_p.add_argument("--checkpoint", required=True, help="File trọng số checkpoint (.pth)")
    eval_p.add_argument("--split", choices=["val", "train", "all"], default="val")
    eval_p.add_argument("--features", type=int, default=64)
    eval_p.add_argument("--batch-size", type=int, default=8)
    eval_p.add_argument("--threshold", type=float, default=0.5)
    eval_p.add_argument("--min-area", type=float, default=TUMOR_SIZE_LOW_LIM)
    eval_p.add_argument("--tta", action="store_true")
    eval_p.add_argument("--out-csv", default=None)
    eval_p.add_argument("--out-json", default=None)
    eval_p.add_argument("--no-cuda", action="store_true")
    eval_p.add_argument("--no-amp", action="store_true")
    eval_p.set_defaults(func=eval_action)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
