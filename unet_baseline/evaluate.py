"""
evaluate.py
-----------
Đánh giá chi tiết mô hình trên tập Test:
- Tải trọng số từ checkpoint (best.pth)
- Hỗ trợ Mixed Precision (autocast) & memory_format=channels_last tăng tốc độ
- Tùy chọn lọc nhiễu diện tích nhỏ (--min_area) bằng Connected Components
- Phân tầng thống kê Dice theo 4 nhóm kích thước nốt u (Ground Truth)
- Xuất bảng kết quả chi tiết ra file test_metrics.csv và test_summary.json
- Tùy chọn lưu ảnh trực quan hóa overlay (--save_overlay) với giới hạn (--max_overlays)
"""

import os
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Any, Tuple, List

import yaml
import numpy as np
import cv2
import torch
from tqdm import tqdm

from models import build_model
from utils import get_dataloaders, calculate_dice, calculate_iou, calculate_hd95, MetricTracker


def remove_small_components(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Lọc bỏ các connected components có diện tích < min_area pixels."""
    if min_area <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 1
    return clean_mask


def get_tumor_size_group(area: int) -> str:
    """Phân nhóm kích thước nốt u theo diện tích pixel (trên lát cắt 256x256)."""
    if area <= 0:
        return "empty"
    elif area <= 24:
        return "<=24 (rất nhỏ)"
    elif area <= 57:
        return "25-57 (nhỏ)"
    elif area <= 177:
        return "58-177 (trung bình)"
    else:
        return ">177 (lớn)"


def parse_args_and_config() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình U-Net trên tập Test")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pth",
        help="Đường dẫn tới file checkpoint .pth",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Đường dẫn tới file cấu hình yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Thư mục xuất file báo cáo metrics và ảnh overlay",
    )
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        help="Lưu ảnh trực quan hóa so sánh GT và Pred",
    )
    parser.add_argument(
        "--max_overlays",
        type=int,
        default=50,
        help="Số lượng ảnh overlay tối đa cần lưu (mặc định 50, tránh nghẽn I/O Drive)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng nhị phân hóa xác suất sigmoid (mặc định 0.5)",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=0,
        help="Lọc bỏ các đốm dự đoán nhỏ hơn ngưỡng pixel này (vd: 10 hoặc 20, mặc định 0)",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu test nếu muốn kiểm tra nhanh (mặc định None = chạy đủ 2006 mẫu)",
    )
    args, unknown = parser.parse_known_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint/config tại: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Đảm bảo đánh giá mặc định chạy toàn bộ 100% mẫu kiểm thử (2,006 mẫu)
    if "train" in cfg:
        cfg["train"]["limit_test_samples"] = args.limit_samples

    # Cho phép override cấu hình từ CLI (vd: --data.root_dir=/content/quick_data)
    for arg in unknown:
        if arg.startswith("--") and "=" in arg:
            key, val = arg[2:].split("=", 1)
            if val.lower() == "true":
                parsed_val = True
            elif val.lower() == "false":
                parsed_val = False
            elif val.lower() in ("null", "none"):
                parsed_val = None
            else:
                try:
                    parsed_val = int(val)
                except ValueError:
                    try:
                        parsed_val = float(val)
                    except ValueError:
                        parsed_val = val

            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = parsed_val

    return args, cfg


@torch.no_grad()
def evaluate() -> None:
    args, cfg = parse_args_and_config()

    # 1. Kiểm tra checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Đánh giá checkpoint: {ckpt_path} trên thiết bị: {device}")

    # 2. Khởi tạo mô hình và nạp weights
    model = build_model(cfg).to(device)
    use_channels_last = cfg.get("train", {}).get("channels_last", True) and device.type == "cuda"
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Lấy Test DataLoader
    _, _, test_loader = get_dataloaders(cfg)
    test_dataset = test_loader.dataset
    print(f"📊 Tổng số mẫu kiểm thử: {len(test_dataset)}")
    if args.min_area > 0:
        print(f"🧹 Bật bộ lọc diện tích nhỏ: loại bỏ các đốm < {args.min_area} px")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        print(f"🖼️ Lưu tối đa {args.max_overlays} ảnh overlay vào: {overlay_dir}")

    tracker = MetricTracker()
    rows = []
    n_classes = cfg["model"].get("n_classes", 1)

    # Khởi tạo bộ đếm phân tầng diện tích nốt u
    stratified_groups = ["<=24 (rất nhỏ)", "25-57 (nhỏ)", "58-177 (trung bình)", ">177 (lớn)", "empty"]
    stratified_data = {
        grp: {"dices": [], "pred_empty": 0}
        for grp in stratified_groups
    }
    saved_overlays = 0
    hd95_valid_count = 0
    hd95_nan_count = 0

    print("⏳ Đang chạy suy luận trên tập Test...")
    idx = 0
    for images, masks in tqdm(test_loader, desc="[Evaluate]"):
        if use_channels_last:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
        else:
            images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)

        if n_classes == 1:
            raw_preds = (torch.sigmoid(logits) > args.threshold).cpu().numpy().astype(np.uint8)
            targets = masks.cpu().numpy().astype(np.uint8)

            for b in range(raw_preds.shape[0]):
                p_i = raw_preds[b, 0]
                t_i = targets[b, 0]

                # Lọc diện tích nhỏ nếu được chỉ định
                if args.min_area > 0:
                    p_i = remove_small_components(p_i, min_area=args.min_area)

                img_path, _ = test_dataset.samples[idx]
                stem = img_path.stem

                gt_area = int(np.sum(t_i))
                pred_area = int(np.sum(p_i))
                size_grp = get_tumor_size_group(gt_area)

                dice = calculate_dice(p_i, t_i)
                iou = calculate_iou(p_i, t_i)
                hd95 = calculate_hd95(p_i, t_i)

                tracker.update(dice=dice, iou=iou, hd95=hd95)
                if not np.isnan(hd95):
                    hd95_valid_count += 1
                else:
                    hd95_nan_count += 1

                # Ghi nhận phân tầng
                if size_grp in stratified_data:
                    stratified_data[size_grp]["dices"].append(dice)
                    if pred_area == 0 and gt_area > 0:
                        stratified_data[size_grp]["pred_empty"] += 1

                rows.append({
                    "sample_id": stem,
                    "gt_area": gt_area,
                    "pred_area": pred_area,
                    "size_group": size_grp,
                    "dice": round(dice, 4),
                    "iou": round(iou, 4),
                    "hd95": round(hd95, 4) if not np.isnan(hd95) else "NaN",
                })

                # Tạo ảnh overlay nếu bật cờ và chưa quá giới hạn max_overlays
                if args.save_overlay and saved_overlays < args.max_overlays:
                    orig_img = images[b].cpu().permute(1, 2, 0).numpy()
                    mean = np.array(cfg["data"].get("mean", [0.485, 0.456, 0.406]))
                    std = np.array(cfg["data"].get("std", [0.229, 0.224, 0.225]))
                    disp_img = np.clip((orig_img * std + mean) * 255.0, 0, 255).astype(np.uint8)
                    if disp_img.shape[-1] == 1:
                        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2BGR)

                    overlay = disp_img.copy()
                    gt_mask = t_i > 0
                    pred_mask = p_i > 0
                    overlap = np.logical_and(gt_mask, pred_mask)
                    overlay[gt_mask] = [0, 255, 0]        # Green: GT
                    overlay[pred_mask] = [0, 0, 255]      # Red: Pred
                    overlay[overlap] = [0, 255, 255]      # Yellow: Trùng khớp
                    blended = cv2.addWeighted(disp_img, 0.65, overlay, 0.35, 0)

                    gt_vis = cv2.cvtColor((t_i * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    pred_vis = cv2.cvtColor((p_i * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

                    combined = np.hstack([disp_img, gt_vis, pred_vis, blended])
                    cv2.imwrite(str(overlay_dir / f"{stem}_eval.png"), combined)
                    saved_overlays += 1

                idx += 1

    # 4. Tính toán thống kê tổng kết
    summary = tracker.summary()
    dice_mean, dice_std = summary["dice"]
    iou_mean, iou_std = summary["iou"]
    hd95_mean, hd95_std = summary["hd95"]

    hd95_str = f"{hd95_mean:.4f} ± {hd95_std:.4f}" if not np.isnan(hd95_mean) else "N/A"

    print("\n" + "=" * 65)
    print("📋 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print(f"  - Tổng số mẫu đánh giá : {len(test_dataset)}")
    print(f"  - Dice Coefficient      : {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"  - IoU (Jaccard)         : {iou_mean:.4f} ± {iou_std:.4f}")
    print(f"  - HD95 (Hausdorff)      : {hd95_str} (Hợp lệ: {hd95_valid_count}/{len(test_dataset)}, NaN: {hd95_nan_count})")
    print("-" * 65)
    print("📊 PHÂN TẦNG HIỆU NĂNG THEO DIỆN TÍCH NỐT U (GROUND TRUTH):")
    print(f"  {'Nhóm kích thước':<22} | {'Số mẫu':<8} | {'Dice (Mean ± Std)':<20} | {'Dự đoán rỗng (sót)':<18}")
    print("  " + "-" * 75)

    stratified_summary = {}
    for grp in ["<=24 (rất nhỏ)", "25-57 (nhỏ)", "58-177 (trung bình)", ">177 (lớn)"]:
        d_list = stratified_data[grp]["dices"]
        n_samples = len(d_list)
        empty_cnt = stratified_data[grp]["pred_empty"]
        if n_samples > 0:
            m_dice = float(np.mean(d_list))
            s_dice = float(np.std(d_list))
            dice_str = f"{m_dice:.4f} ± {s_dice:.4f}"
        else:
            m_dice, s_dice = 0.0, 0.0
            dice_str = "N/A (0 mẫu)"
        print(f"  {grp:<22} | {n_samples:<8} | {dice_str:<20} | {empty_cnt:<18}")
        stratified_summary[grp] = {
            "num_samples": n_samples,
            "dice_mean": round(m_dice, 4),
            "dice_std": round(s_dice, 4),
            "pred_empty_count": empty_cnt,
        }
    print("=" * 65)

    # 5. Xuất file test_metrics.csv
    csv_path = output_dir / "test_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "gt_area", "pred_area", "size_group", "dice", "iou", "hd95"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        # Thêm dòng Mean ± Std ở cuối
        writer.writerow({
            "sample_id": "MEAN ± STD",
            "gt_area": "",
            "pred_area": "",
            "size_group": "ALL",
            "dice": f"{dice_mean:.4f} ± {dice_std:.4f}",
            "iou": f"{iou_mean:.4f} ± {iou_std:.4f}",
            "hd95": hd95_str,
        })

    # 6. Xuất file test_summary.json
    json_path = output_dir / "test_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "num_samples": len(test_dataset),
                "threshold": args.threshold,
                "min_area": args.min_area,
                "dice": {"mean": round(float(dice_mean), 4), "std": round(float(dice_std), 4)},
                "iou": {"mean": round(float(iou_mean), 4), "std": round(float(iou_std), 4)},
                "hd95": {
                    "mean": round(float(hd95_mean), 4) if not np.isnan(hd95_mean) else None,
                    "std": round(float(hd95_std), 4) if not np.isnan(hd95_std) else None,
                    "valid_count": hd95_valid_count,
                    "nan_count": hd95_nan_count,
                },
                "stratified_metrics": stratified_summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"📁 Đã lưu báo cáo tại: {output_dir}/")


if __name__ == "__main__":
    evaluate()
