"""
diagnose.py
-----------
Bộ công cụ chẩn đoán chuyên sâu cho mô hình U-Net Baseline trên dữ liệu LIDC-IDRI:
1. Chẩn đoán A (mode: sweep): Quét ngưỡng Sigmoid (threshold) và lọc diện tích nhỏ (min_area)
   để tìm điểm tối ưu Dice/IoU mà không cần train lại.
2. Chẩn đoán B (mode: size): Phân tích hiệu năng Dice chi tiết theo 4 tầng kích thước nốt u
   (<=24 px, 25-57 px, 58-177 px, >177 px) từ file test_metrics.csv.
3. Chẩn đoán C (mode: channels): Kiểm tra bản chất dữ liệu 2.5D (ảnh 3 lát cắt liên tiếp Z-1, Z, Z+1
   hay ảnh grayscale nhân bản 3 kênh giống nhau).
"""

import os
import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
import numpy as np
import cv2
import torch
from tqdm import tqdm

from models import build_model
from utils import get_dataloaders, calculate_dice, calculate_iou


def remove_small_components(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Lọc bỏ các connected components có diện tích < min_area."""
    if min_area <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 1
    return clean_mask


# ==============================================================================
# CHẨN ĐOÁN A: QUÉT NGƯỠNG VÀ LỌC DIỆN TÍCH NHỎ (THRESHOLD & MIN_AREA SWEEP)
# ==============================================================================
@torch.no_grad()
def diagnose_sweep(cfg: Dict[str, Any], ckpt_path: Path, limit_samples: int = 500) -> None:
    print("\n" + "=" * 70)
    print("🔍 [CHẨN ĐOÁN A] QUÉT TÌM NGƯỠNG TỐI ƯU (THRESHOLD & MIN_AREA SWEEP)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    if not ckpt_path.exists():
        print(f"❌ Không tìm thấy checkpoint tại: {ckpt_path}")
        return

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Dùng Val Loader hoặc Test Loader giới hạn mẫu để quét nhanh
    if limit_samples:
        cfg["train"]["limit_val_samples"] = limit_samples
    _, val_loader, _ = get_dataloaders(cfg)

    print(f"⏳ Thu thập xác suất dự đoán (Sigmoid) trên {len(val_loader.dataset)} mẫu...")
    all_probs = []
    all_targets = []

    for images, masks in tqdm(val_loader, desc="[Inference]"):
        images = images.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_targets.append(masks.numpy().astype(np.uint8))

    probs_arr = np.concatenate(all_probs, axis=0)[:, 0]  # Shape: (N, H, W)
    targets_arr = np.concatenate(all_targets, axis=0)[:, 0]

    thresholds = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    min_areas = [0, 5, 10, 20]

    print(f"\n📊 BẢNG KẾT QUẢ QUÉT SIÊU THAM SỐ HẬU XỬ LÝ (POST-PROCESSING):")
    print(f"{'Threshold':<10} | {'Min Area (px)':<14} | {'Mean Dice':<12} | {'Mean IoU':<12}")
    print("-" * 55)

    best_dice = -1.0
    best_config = (0.5, 0)

    for th in thresholds:
        bin_masks = (probs_arr > th).astype(np.uint8)
        for ma in min_areas:
            dices = []
            ious = []
            for i in range(len(bin_masks)):
                p = bin_masks[i]
                t = targets_arr[i]
                if ma > 0:
                    p = remove_small_components(p, min_area=ma)
                dices.append(calculate_dice(p, t))
                ious.append(calculate_iou(p, t))

            mean_d = float(np.mean(dices))
            mean_i = float(np.mean(ious))

            mark = ""
            if mean_d > best_dice:
                best_dice = mean_d
                best_config = (th, ma)
                mark = " ⭐ (Best)"

            print(f"{th:<10.2f} | {ma:<14} | {mean_d:<12.4f} | {mean_i:<12.4f}{mark}")

    print("-" * 55)
    print(f"🎯 Điểm tối ưu nhất: Threshold = {best_config[0]}, Min Area = {best_config[1]} px -> Dice: {best_dice:.4f}")
    print("💡 Gợi ý: Bạn có thể truyền các tham số này vào evaluate.py:")
    print(f"   python evaluate.py --threshold={best_config[0]} --min_area={best_config[1]}")


# ==============================================================================
# CHẨN ĐOÁN B: PHÂN TÍCH THEO 4 TẦNG KÍCH THƯỚC NỐT U (TUMOR SIZE BREAKDOWN)
# ==============================================================================
def diagnose_size_breakdown(csv_path: Path) -> None:
    print("\n" + "=" * 70)
    print("📊 [CHẨN ĐOÁN B] PHÂN TÍCH HIỆU NĂNG THEO 4 TẦNG KÍCH THƯỚC NỐT U")
    print("=" * 70)

    if not csv_path.exists():
        print(f"❌ Không tìm thấy file kết quả đánh giá: {csv_path}")
        print("💡 Vui lòng chạy evaluate.py trước để tạo file test_metrics.csv")
        return

    groups = {
        "<=24 (rất nhỏ)": [],
        "25-57 (nhỏ)": [],
        "58-177 (trung bình)": [],
        ">177 (lớn)": [],
        "empty": [],
    }
    empty_preds = {k: 0 for k in groups}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("sample_id") == "MEAN ± STD":
                continue
            dice = float(row["dice"])
            gt_area = int(row.get("gt_area", 0)) if row.get("gt_area") else None
            pred_area = int(row.get("pred_area", 0)) if row.get("pred_area") else 0
            size_grp = row.get("size_group")

            # Nếu file CSV cũ chưa có cột size_group, tự suy luận từ gt_area
            if not size_grp:
                if gt_area is None or gt_area == 0:
                    size_grp = "empty"
                elif gt_area <= 24:
                    size_grp = "<=24 (rất nhỏ)"
                elif gt_area <= 57:
                    size_grp = "25-57 (nhỏ)"
                elif gt_area <= 177:
                    size_grp = "58-177 (trung bình)"
                else:
                    size_grp = ">177 (lớn)"

            if size_grp in groups:
                groups[size_grp].append(dice)
                if pred_area == 0 and (gt_area is not None and gt_area > 0):
                    empty_preds[size_grp] += 1

    print(f"{'Nhóm kích thước':<22} | {'Số mẫu':<8} | {'Dice (Mean ± Std)':<20} | {'Dự đoán rỗng (sót u)':<20}")
    print("-" * 75)
    for grp in ["<=24 (rất nhỏ)", "25-57 (nhỏ)", "58-177 (trung bình)", ">177 (lớn)"]:
        d_list = groups[grp]
        n_samples = len(d_list)
        n_empty = empty_preds[grp]
        if n_samples > 0:
            m_dice = float(np.mean(d_list))
            s_dice = float(np.std(d_list))
            rate_empty = (n_empty / n_samples) * 100
            print(f"{grp:<22} | {n_samples:<8} | {m_dice:.4f} ± {s_dice:.4f}{'':<4} | {n_empty} ({rate_empty:.1f}%)")
        else:
            print(f"{grp:<22} | 0        | N/A                  | 0")
    print("=" * 75)
    print("📌 Nhận định y khoa:")
    print("  - Nốt u nhỏ (<=24 px) thường chiếm 25-30% nhưng có Dice thấp nhất vì chỉ cần lệch vài pixel")
    print("    là Dice tụt dốc nghiêm trọng. Nếu tỉ lệ dự đoán rỗng cao -> Cần bổ sung Focal Tumor Cropping.")


# ==============================================================================
# CHẨN ĐOÁN C: KIỂM TRA BẢN CHẤT DỮ LIỆU 2.5D (COLOR CHANNEL CHECK)
# ==============================================================================
def diagnose_channels(data_dir: Path, num_samples: int = 20) -> None:
    print("\n" + "=" * 70)
    print("🔬 [CHẨN ĐOÁN C] KIỂM TRA ĐỘ LỆCH KÊNH MÀU (XÁC NHẬN ẢNH 2.5D)")
    print("=" * 70)

    img_dir = data_dir / "images"
    if not img_dir.exists():
        # Tìm các thư mục con khả dĩ
        possible = list(data_dir.glob("**/images"))
        if possible:
            img_dir = possible[0]
        else:
            print(f"❌ Không tìm thấy thư mục images tại: {data_dir}")
            return

    img_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.npy"))
    if not img_files:
        print(f"❌ Không tìm thấy file ảnh nào trong: {img_dir}")
        return

    np.random.seed(42)
    selected_files = np.random.choice(img_files, size=min(num_samples, len(img_files)), replace=False)

    diffs_01 = []
    diffs_12 = []

    print(f"Kiểm tra ngẫu nhiên {len(selected_files)} ảnh trong: {img_dir.name}/")
    for f in selected_files:
        if f.suffix == ".npy":
            arr = np.load(f)
            if arr.ndim == 3 and arr.shape[0] == 3:
                c0, c1, c2 = arr[0], arr[1], arr[2]
            elif arr.ndim == 3 and arr.shape[2] == 3:
                c0, c1, c2 = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            else:
                continue
        else:
            bgr = cv2.imread(str(f))
            if bgr is None:
                continue
            c0, c1, c2 = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

        d01 = float(np.mean(np.abs(c0.astype(float) - c1.astype(float))))
        d12 = float(np.mean(np.abs(c1.astype(float) - c2.astype(float))))
        diffs_01.append(d01)
        diffs_12.append(d12)

    mean_diff_01 = np.mean(diffs_01) if diffs_01 else 0.0
    mean_diff_12 = np.mean(diffs_12) if diffs_12 else 0.0

    print(f"\n📈 Kết quả đo chênh lệch cường độ pixel giữa các kênh:")
    print(f"  - Sai khác trung bình |Kênh 0 - Kênh 1|: {mean_diff_01:.4f}")
    print(f"  - Sai khác trung bình |Kênh 1 - Kênh 2|: {mean_diff_12:.4f}")

    print("\n🔍 KẾT LUẬN KIỂM ĐỊNH:")
    if mean_diff_01 < 1e-4 and mean_diff_12 < 1e-4:
        print("  ⚠️ ẢNH 2D NHÂN BẢN (Fake 2.5D): Cả 3 kênh R, G, B hoàn toàn giống hệt nhau!")
        print("     -> Mô hình U-Net không nhận được thông tin ngữ cảnh trục Z từ các lát lân cận.")
        print("     -> Đây là cơ hội tăng vọt Dice: Khi chuyển sang trích xuất 2.5D thật (Z-1, Z, Z+1),")
        print("        đặc trưng biên khối u sẽ rõ ràng hơn rất nhiều.")
    else:
        print("  ✅ ẢNH 2.5D CHUẨN: Có sự khác biệt rõ rệt giữa các kênh lát cắt liên tiếp (Z-1, Z, Z+1).")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Công cụ chẩn đoán chuyên sâu U-Net Baseline")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "sweep", "size", "channels"],
        help="Chế độ chẩn đoán: sweep (ngưỡng), size (phân tầng u), channels (kiểm tra 2.5D), all (tất cả)",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--metrics_csv", type=str, default="evaluation_results/test_metrics.csv")
    parser.add_argument("--data_dir", type=str, default="../data/segmented_slices")
    parser.add_argument("--limit_samples", type=int, default=500)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    if args.mode in ("channels", "all"):
        data_path = Path(args.data_dir)
        if not data_path.exists() and "data" in cfg and "root_dir" in cfg["data"]:
            data_path = Path(cfg["data"]["root_dir"])
        diagnose_channels(data_path)

    if args.mode in ("size", "all"):
        diagnose_size_breakdown(Path(args.metrics_csv))

    if args.mode in ("sweep", "all"):
        ckpt = Path(args.checkpoint)
        if ckpt.exists() and cfg:
            diagnose_sweep(cfg, ckpt, limit_samples=args.limit_samples)
        else:
            print(f"\n⚠️ Bỏ qua Chẩn đoán A (Sweep) vì chưa có file checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
