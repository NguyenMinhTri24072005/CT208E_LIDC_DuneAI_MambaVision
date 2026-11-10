"""
analyze_nodules.py
------------------
Công cụ khảo sát và phân tích toàn diện đặc tính nốt u phổi trong tập dữ liệu segmented_slices (LIDC-IDRI):
- Đo đạc tỷ lệ diện tích mask nốt u trên toàn bộ lát cắt (% Mask Ratio).
- Quy đổi diện tích nốt u khi đưa vào tensor huấn luyện 256x256.
- Tính toán kích thước hình học y tế (diện tích mm2, đường kính tương đương mm).
- Phân loại nốt u theo các tầng kích thước: Rất nhỏ, Nhỏ, Vừa, Lớn, Rất lớn.
- Báo cáo chi tiết theo từng tập Train, Val, Test và toàn bộ Dataset.
- Tùy chọn xuất file CSV chi tiết, JSON tổng kết và biểu đồ trực quan hóa PNG.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

# Đảm bảo in tiếng Việt chuẩn Unicode trên console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Khảo sát nốt u phổi và tỷ lệ diện tích mask trong segmented_slices"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="segmented_slices",
        help="Đường dẫn đến thư mục segmented_slices (mặc định: segmented_slices)",
    )
    parser.add_argument(
        "--cases_csv",
        type=str,
        default="../data/processed/cases.csv",
        help="Đường dẫn đến file cases.csv chứa kích thước lát cắt và spacing (mặc định: ../data/processed/cases.csv hoặc processed/cases.csv)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Tập dữ liệu cần khảo sát: all, train, val, test (mặc định: all)",
    )
    parser.add_argument(
        "--export_csv",
        type=str,
        default="nodule_statistics_detailed.csv",
        help="Tên file CSV xuất dữ liệu chi tiết cho 13,048 mẫu (để trống nếu không muốn xuất)",
    )
    parser.add_argument(
        "--export_json",
        type=str,
        default="nodule_statistics_summary.json",
        help="Tên file JSON tóm tắt thống kê (mặc định: nodule_statistics_summary.json)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Vẽ và lưu biểu đồ phân phối nốt u thành file nodule_distribution_charts.png (mặc định: True)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_false",
        dest="plot",
        help="Tắt chức năng vẽ biểu đồ",
    )
    return parser.parse_args()


def load_dataset(data_dir: Path, cases_csv_path: Path) -> pd.DataFrame:
    """Nạp metadata của tập dữ liệu segmented_slices và kết hợp với cases.csv."""
    # 1. Tìm file positive_slices_summary.csv
    summary_csv = data_dir / "positive_slices_summary.csv"
    if not summary_csv.exists():
        # Thử tìm ở thư mục cha hoặc con
        alt_paths = [
            Path("segmented_slices/positive_slices_summary.csv"),
            Path("data/segmented_slices/positive_slices_summary.csv"),
            data_dir.parent / "segmented_slices" / "positive_slices_summary.csv",
        ]
        found = False
        for p in alt_paths:
            if p.exists():
                summary_csv = p
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Không tìm thấy positive_slices_summary.csv tại: {data_dir}")

    df_slices = pd.read_csv(summary_csv)
    df_slices.columns = df_slices.columns.str.strip()

    # 2. Tìm file cases.csv để lấy kích thước lát cắt chính xác và spacing mm
    candidate_cases = [
        cases_csv_path,
        Path("processed/cases.csv"),
        Path("../data/processed/cases.csv"),
        data_dir.parent / "processed" / "cases.csv",
        Path("D:/GG_1/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/data/processed/cases.csv"),
    ]
    df_cases = None
    for p in candidate_cases:
        if p.exists():
            df_cases = pd.read_csv(p)
            df_cases.columns = df_cases.columns.str.strip()
            break

    if df_cases is not None:
        merged = df_slices.merge(
            df_cases[["patient_id", "new_shape", "orig_spacing", "orig_shape"]],
            on="patient_id",
            how="left",
        )
    else:
        merged = df_slices.copy()
        merged["new_shape"] = "222x450x450"
        merged["orig_spacing"] = "2.5x0.7x0.7"

    # Trích xuất chiều cao (H) và chiều rộng (W) của lát cắt
    def parse_hw(shape_str):
        try:
            parts = str(shape_str).split("x")
            if len(parts) == 3:
                return int(parts[1]), int(parts[2])
        except Exception:
            pass
        return 450, 450

    hws = [parse_hw(s) for s in merged["new_shape"]]
    merged["slice_h"] = [x[0] for x in hws]
    merged["slice_w"] = [x[1] for x in hws]
    merged["slice_area"] = merged["slice_h"] * merged["slice_w"]

    # 3. Tính tỷ lệ % diện tích mask trên toàn bộ ảnh
    # Ratio (%) = (positive_pixels / slice_area) * 100
    merged["mask_ratio_percent"] = (merged["positive_pixels"] / merged["slice_area"]) * 100.0

    # 4. Quy đổi số pixel tương đương khi đưa vào mô hình Deep Learning (Tensor 256x256)
    # Tensor 256x256 có tổng số 65,536 pixels
    merged["pixels_in_256"] = merged["positive_pixels"] * (256 * 256 / merged["slice_area"])

    # 5. Tính diện tích mm2 và đường kính tương đương mm
    def parse_spacing(sp_str):
        try:
            parts = str(sp_str).split("x")
            if len(parts) == 3:
                return float(parts[1]), float(parts[2])
        except Exception:
            pass
        return 0.7031, 0.7031

    spacings = [parse_spacing(s) for s in merged["orig_spacing"]]
    merged["pixel_area_mm2"] = [x[0] * x[1] for x in spacings]
    merged["nodule_area_mm2"] = merged["positive_pixels"] * merged["pixel_area_mm2"]
    # Đường kính tương đương hình tròn D = 2 * sqrt(Area / pi)
    merged["nodule_diameter_mm"] = 2.0 * np.sqrt(merged["nodule_area_mm2"] / np.pi)
    merged["diameter_mm"] = merged["nodule_diameter_mm"]

    # 6. Gán nhãn phân tầng kích thước
    # Cách 1: Phân tầng theo Pixel (Chuẩn phân vị Quartiles LIDC-IDRI)
    def assign_pixel_group(p):
        if p <= 24:
            return "1. Rất nhỏ (<= 24 px)"
        elif p <= 57:
            return "2. Nhỏ (25 - 57 px)"
        elif p <= 177:
            return "3. Vừa (58 - 177 px)"
        elif p <= 500:
            return "4. Lớn (178 - 500 px)"
        else:
            return "5. Rất lớn (> 500 px)"

    merged["group_pixel"] = merged["positive_pixels"].apply(assign_pixel_group)

    # Cách 2: Phân tầng theo Tỷ lệ diện tích Mask (% trên lát cắt)
    def assign_ratio_group(r):
        if r < 0.02:
            return "1. Cực nhỏ (< 0.02%)"
        elif r < 0.05:
            return "2. Nhỏ (0.02% - 0.05%)"
        elif r < 0.15:
            return "3. Vừa (0.05% - 0.15%)"
        elif r < 0.30:
            return "4. Lớn (0.15% - 0.30%)"
        else:
            return "5. Rất lớn (>= 0.30%)"

    merged["group_ratio"] = merged["mask_ratio_percent"].apply(assign_ratio_group)

    # Cách 3: Phân tầng theo Đường kính y khoa mm (Fleischner Society / Lung-RADS)
    def assign_diameter_group(d):
        if d < 4.0:
            return "1. Vi nốt (< 4 mm)"
        elif d < 6.0:
            return "2. Nốt nhỏ (4 - 6 mm)"
        elif d < 8.0:
            return "3. Nốt vừa (6 - 8 mm)"
        elif d < 15.0:
            return "4. Nốt lớn (8 - 15 mm)"
        else:
            return "5. Khối u (> 15 mm)"

    merged["group_diameter"] = merged["diameter_mm"].apply(assign_diameter_group)

    return merged


def print_statistics(df: pd.DataFrame, split_name: str):
    """In bảng thống kê chi tiết ra màn hình terminal."""
    total_samples = len(df)
    n_patients = df["patient_id"].nunique()

    print("\n" + "=" * 80)
    print(f"📊 KẾT QUẢ KHẢO SÁT NỐT U PHỔI TRONG TẬP: [{split_name.upper()}]")
    print(f"   • Tổng số lát cắt phân tích : {total_samples:,} mẫu")
    print(f"   • Số lượng bệnh nhân duy nhất: {n_patients:,} ca")
    print("=" * 80)

    # 1. Thống kê số lượng Pixel nốt u
    p_desc = df["positive_pixels"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    r_desc = df["mask_ratio_percent"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    d_desc = df["nodule_diameter_mm"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    p256_desc = df["pixels_in_256"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

    print("\n📈 1. BẢNG CHỈ SỐ THỐNG KÊ MÔ TẢ (DESCRIPTIVE STATISTICS):")
    print("-" * 80)
    print(f"{'Chỉ số Thống kê':<20} | {'Pixel Gốc':<14} | {'Pixel (256x256)':<16} | {'Tỷ lệ Mask (%)':<16} | {'Đường kính (mm)':<15}")
    print("-" * 80)
    print(f"{'Nhỏ nhất (Min)':<20} | {p_desc['min']:<14.0f} | {p256_desc['min']:<16.2f} | {r_desc['min']:<16.5f}% | {d_desc['min']:<15.2f}")
    print(f"{'Phân vị 10% (P10)':<20} | {p_desc['10%']:<14.0f} | {p256_desc['10%']:<16.2f} | {r_desc['10%']:<16.5f}% | {d_desc['10%']:<15.2f}")
    print(f"{'Phân vị 25% (Q1)':<20} | {p_desc['25%']:<14.0f} | {p256_desc['25%']:<16.2f} | {r_desc['25%']:<16.5f}% | {d_desc['25%']:<15.2f}")
    print(f"{'Trung vị (Median / P50)':<20} | {p_desc['50%']:<14.0f} | {p256_desc['50%']:<16.2f} | {r_desc['50%']:<16.5f}% | {d_desc['50%']:<15.2f}")
    print(f"{'Trung bình (Mean)':<20} | {p_desc['mean']:<14.1f} | {p256_desc['mean']:<16.2f} | {r_desc['mean']:<16.5f}% | {d_desc['mean']:<15.2f}")
    print(f"{'Độ lệch chuẩn (Std)':<20} | {p_desc['std']:<14.1f} | {p256_desc['std']:<16.2f} | {r_desc['std']:<16.5f}% | {d_desc['std']:<15.2f}")
    print(f"{'Phân vị 75% (Q3)':<20} | {p_desc['75%']:<14.0f} | {p256_desc['75%']:<16.2f} | {r_desc['75%']:<16.5f}% | {d_desc['75%']:<15.2f}")
    print(f"{'Phân vị 90% (P90)':<20} | {p_desc['90%']:<14.0f} | {p256_desc['90%']:<16.2f} | {r_desc['90%']:<16.5f}% | {d_desc['90%']:<15.2f}")
    print(f"{'Phân vị 95% (P95)':<20} | {p_desc['95%']:<14.0f} | {p256_desc['95%']:<16.2f} | {r_desc['95%']:<16.5f}% | {d_desc['95%']:<15.2f}")
    print(f"{'Phân vị 99% (P99)':<20} | {p_desc['99%']:<14.0f} | {p256_desc['99%']:<16.2f} | {r_desc['99%']:<16.5f}% | {d_desc['99%']:<15.2f}")
    print(f"{'Lớn nhất (Max)':<20} | {p_desc['max']:<14.0f} | {p256_desc['max']:<16.2f} | {r_desc['max']:<16.5f}% | {d_desc['max']:<15.2f}")
    print("-" * 80)

    # 2. Bảng phân bố theo nhóm diện tích Pixel (Quartiles)
    print("\n🎯 2. PHÂN BỐ SỐ LƯỢNG MẪU THEO DIỆN TÍCH PIXEL NỐT U:")
    print("-" * 80)
    print(f"{'Nhóm kích thước Pixel':<28} | {'Số mẫu':<10} | {'Tỷ lệ (%)':<12} | {'Tỷ lệ Mask TB (%)':<18} | {'Đường kính TB (mm)':<16}")
    print("-" * 80)
    for grp, sub in df.groupby("group_pixel"):
        cnt = len(sub)
        pct = (cnt / total_samples) * 100.0
        m_ratio = sub["mask_ratio_percent"].mean()
        m_diam = sub["nodule_diameter_mm"].mean()
        print(f"{grp:<28} | {cnt:<10,d} | {pct:<12.2f}% | {m_ratio:<18.4f}% | {m_diam:<16.2f}")
    print("-" * 80)

    # 3. Bảng phân bố theo Tỷ lệ diện tích Mask (%)
    print("\n🔬 3. PHÂN BỐ SỐ LƯỢNG MẪU THEO TỶ LỆ MASK TRÊN MỘT MẪU (% DIỆN TÍCH LÁT CẮT):")
    print("-" * 80)
    print(f"{'Nhóm Tỷ lệ Mask (%)':<28} | {'Số mẫu':<10} | {'Tỷ lệ (%)':<12} | {'Khoảng Pixel Gốc':<20} | {'Pixel trên 256x256':<16}")
    print("-" * 80)
    for grp, sub in df.groupby("group_ratio"):
        cnt = len(sub)
        pct = (cnt / total_samples) * 100.0
        p_min, p_max = int(sub["positive_pixels"].min()), int(sub["positive_pixels"].max())
        p256_mean = sub["pixels_in_256"].mean()
        print(f"{grp:<28} | {cnt:<10,d} | {pct:<12.2f}% | {f'{p_min} - {p_max} px':<20} | {p256_mean:<16.1f}")
    print("-" * 80)

    # 4. Bảng phân bố theo Tiêu chuẩn Y tế Quốc tế (Đường kính mm)
    print("\n🩺 4. PHÂN BỐ THEO TIÊU CHUẨN ĐƯỜNG KÍNH Y KHOA (FLEISCHNER / LUNG-RADS):")
    print("-" * 80)
    print(f"{'Phân loại Y khoa':<28} | {'Số mẫu':<10} | {'Tỷ lệ (%)':<12} | {'Đánh giá Lâm sàng':<35}")
    print("-" * 80)
    clinical_notes = {
        "1. Vi nốt (< 4 mm)": "Nguy cơ ác tính rất thấp (<1%)",
        "2. Nốt nhỏ (4 - 6 mm)": "Cần theo dõi định kỳ sau 12 tháng",
        "3. Nốt vừa (6 - 8 mm)": "Theo dõi sau 6-12 tháng (nguy cơ 1-2%)",
        "4. Nốt lớn (8 - 15 mm)": "Nguy cơ cao (nghi ngờ ác tính >5%)",
        "5. Khối u (> 15 mm)": "Nguy cơ ác tính rất cao (sinh thiết/PET)",
    }
    for grp, sub in df.groupby("group_diameter"):
        cnt = len(sub)
        pct = (cnt / total_samples) * 100.0
        note = clinical_notes.get(grp, "")
        print(f"{grp:<28} | {cnt:<10,d} | {pct:<12.2f}% | {note:<35}")
    print("-" * 80)

    # 5. Phân bố chéo giữa các Split (nếu xem all)
    if split_name == "all" and "split" in df.columns:
        print("\n📂 5. PHÂN BỐ CHI TIẾT SỐ LƯỢNG MẪU THEO SPLIT (TRAIN / VAL / TEST):")
        print("-" * 80)
        ct = pd.crosstab(df["group_pixel"], df["split"], margins=True, margins_name="Tổng cộng")
        print(ct.to_string())
        print("-" * 80)

    # 6. Top 5 mẫu lớn nhất và Top 5 mẫu nhỏ nhất
    print("\n🔍 6. TOP 5 MẪU CÓ NỐT U LỚN NHẤT & NHỎ NHẤT:")
    print("-" * 80)
    top_largest = df.nlargest(5, "positive_pixels")[["patient_id", "z", "split", "positive_pixels", "mask_ratio_percent", "nodule_diameter_mm"]]
    top_smallest = df.nsmallest(5, "positive_pixels")[["patient_id", "z", "split", "positive_pixels", "mask_ratio_percent", "nodule_diameter_mm"]]

    print("• TOP 5 MẪU LỚN NHẤT:")
    for _, r in top_largest.iterrows():
        print(f"   [{r['split'].upper():5s}] {r['patient_id']} (z={r['z']:3d}) -> {r['positive_pixels']:4d} px | Mask: {r['mask_ratio_percent']:.4f}% | Đường kính: {r['nodule_diameter_mm']:.2f} mm")

    print("\n• TOP 5 MẪU NHỎ NHẤT:")
    for _, r in top_smallest.iterrows():
        print(f"   [{r['split'].upper():5s}] {r['patient_id']} (z={r['z']:3d}) -> {r['positive_pixels']:4d} px | Mask: {r['mask_ratio_percent']:.5f}% | Đường kính: {r['nodule_diameter_mm']:.2f} mm")
    print("=" * 80)


def plot_distribution_charts(df: pd.DataFrame, out_path: Path):
    """Vẽ và xuất bộ biểu đồ phân phối 4 panel ra file ảnh PNG."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("⚠️ Chưa cài đặt matplotlib, bỏ qua bước vẽ biểu đồ.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panel 1: Histogram tỷ lệ % Mask trên lát cắt
    ax1 = axes[0, 0]
    ax1.hist(df["mask_ratio_percent"], bins=60, color="#2b5c8f", edgecolor="black", alpha=0.85)
    median_r = df["mask_ratio_percent"].median()
    mean_r = df["mask_ratio_percent"].mean()
    ax1.axvline(median_r, color="red", linestyle="--", linewidth=1.5, label=f"Median: {median_r:.4f}%")
    ax1.axvline(mean_r, color="orange", linestyle=":", linewidth=1.5, label=f"Mean: {mean_r:.4f}%")
    ax1.set_title("1. Phân phối Tỷ lệ % Mask trên Lát cắt (Mask Ratio %)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Tỷ lệ diện tích nốt u / toàn bộ ảnh (%)", fontsize=10)
    ax1.set_ylabel("Số lượng lát cắt", fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Phân bố số lượng mẫu theo Nhóm Kích thước Pixel
    ax2 = axes[0, 1]
    pixel_counts = df["group_pixel"].value_counts().sort_index()
    colors = ["#4a90e2", "#50e3c2", "#f5a623", "#e056fd", "#d0021b"]
    bars = ax2.bar(range(len(pixel_counts)), pixel_counts.values, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_xticks(range(len(pixel_counts)))
    labels = [k.split(". ")[-1] for k in pixel_counts.index]
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:,}\n({h/len(df)*100:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_title("2. Số lượng mẫu theo Tầng Kích thước Pixel", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Số lượng mẫu", fontsize=10)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Phân bố theo Đường kính Lâm sàng (mm)
    ax3 = axes[1, 0]
    ax3.hist(df["nodule_diameter_mm"], bins=50, color="#16a085", edgecolor="black", alpha=0.85)
    median_d = df["nodule_diameter_mm"].median()
    ax3.axvline(median_d, color="red", linestyle="--", linewidth=1.5, label=f"Median: {median_d:.1f} mm")
    ax3.axvline(8.0, color="purple", linestyle="--", linewidth=1.2, label="Ngưỡng cao: 8 mm")
    ax3.axvline(15.0, color="crimson", linestyle="--", linewidth=1.2, label="Khối u: 15 mm")
    ax3.set_title("3. Phân phối Đường kính Nốt u Tương đương (mm)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Đường kính hình học tương đương (mm)", fontsize=10)
    ax3.set_ylabel("Số lượng lát cắt", fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Phân bố số lượng mẫu qua các tập Train / Val / Test
    ax4 = axes[1, 1]
    if "split" in df.columns:
        split_ct = pd.crosstab(df["group_pixel"], df["split"])
        # Đảm bảo thứ tự cột
        cols = [c for c in ["train", "val", "test"] if c in split_ct.columns]
        split_ct = split_ct[cols]
        split_ct.plot(kind="bar", stacked=True, ax=ax4, colormap="viridis", edgecolor="black", alpha=0.85)
        ax4.set_title("4. Phân tầng Kích thước qua Train / Val / Test", fontsize=12, fontweight="bold")
        ax4.set_xlabel("")
        ax4.set_xticklabels([k.split(". ")[-1] for k in split_ct.index], rotation=15, ha="right", fontsize=9)
        ax4.set_ylabel("Số lượng mẫu", fontsize=10)
        ax4.legend(title="Tập dữ liệu", fontsize=10)
        ax4.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(f"BÁO CÁO PHÂN TÍCH ĐẶC TÍNH NỐT U PHỔI (LIDC-IDRI - {len(df):,} SAMPLES)", fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Đã vẽ và lưu biểu đồ phân phối tại: {out_path.resolve()}")


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (script_dir / data_dir).resolve()

    cases_csv_path = Path(args.cases_csv)
    if not cases_csv_path.is_absolute():
        cases_csv_path = (script_dir / cases_csv_path).resolve()

    print(f"🚀 Bắt đầu phân tích tập dữ liệu: {data_dir}")
    df = load_dataset(data_dir, cases_csv_path)

    # Lọc split nếu có yêu cầu
    if args.split != "all":
        df = df[df["split"] == args.split].copy().reset_index(drop=True)

    # 1. In báo cáo
    print_statistics(df, args.split)

    # 2. Xuất file CSV chi tiết
    if args.export_csv:
        csv_out = script_dir / args.export_csv
        cols_to_export = [
            "patient_id", "z", "split", "positive_pixels", "slice_h", "slice_w", "slice_area",
            "mask_ratio_percent", "pixels_in_256", "nodule_area_mm2", "nodule_diameter_mm",
            "group_pixel", "group_ratio", "group_diameter", "image_path", "mask_path"
        ]
        available_cols = [c for c in cols_to_export if c in df.columns]
        df[available_cols].to_csv(csv_out, index=False, encoding="utf-8-sig")
        print(f"\n💾 Đã lưu dữ liệu chi tiết 13,048 mẫu tại: {csv_out.resolve()}")

    # 3. Xuất file JSON tóm tắt
    if args.export_json:
        json_out = script_dir / args.export_json
        summary_dict = {
            "total_samples": len(df),
            "total_patients": int(df["patient_id"].nunique()),
            "positive_pixels": {
                "min": int(df["positive_pixels"].min()),
                "p25": float(df["positive_pixels"].quantile(0.25)),
                "median": float(df["positive_pixels"].median()),
                "mean": round(float(df["positive_pixels"].mean()), 2),
                "std": round(float(df["positive_pixels"].std()), 2),
                "p75": float(df["positive_pixels"].quantile(0.75)),
                "p90": float(df["positive_pixels"].quantile(0.90)),
                "max": int(df["positive_pixels"].max()),
            },
            "mask_ratio_percent": {
                "min": round(float(df["mask_ratio_percent"].min()), 6),
                "p25": round(float(df["mask_ratio_percent"].quantile(0.25)), 5),
                "median": round(float(df["mask_ratio_percent"].median()), 5),
                "mean": round(float(df["mask_ratio_percent"].mean()), 5),
                "p75": round(float(df["mask_ratio_percent"].quantile(0.75)), 5),
                "p90": round(float(df["mask_ratio_percent"].quantile(0.90)), 5),
                "max": round(float(df["mask_ratio_percent"].max()), 5),
            },
            "nodule_diameter_mm": {
                "min": round(float(df["nodule_diameter_mm"].min()), 2),
                "median": round(float(df["nodule_diameter_mm"].median()), 2),
                "mean": round(float(df["nodule_diameter_mm"].mean()), 2),
                "max": round(float(df["nodule_diameter_mm"].max()), 2),
            },
            "counts_by_pixel_group": df["group_pixel"].value_counts().to_dict(),
            "counts_by_ratio_group": df["group_ratio"].value_counts().to_dict(),
            "counts_by_diameter_group": df["group_diameter"].value_counts().to_dict(),
        }
        with open(json_out, "w", encoding="utf-8") as jf:
            json.dump(summary_dict, jf, indent=2, ensure_ascii=False)
        print(f"📋 Đã lưu tóm tắt JSON tại: {json_out.resolve()}")

    # 4. Vẽ biểu đồ nếu bật cờ
    if args.plot:
        chart_out = script_dir / "nodule_distribution_charts.png"
        plot_distribution_charts(df, chart_out)


if __name__ == "__main__":
    main()
