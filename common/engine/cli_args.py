# -*- coding: utf-8 -*-
"""
common/engine/cli_args.py - Các nhóm tham số dòng lệnh dùng chung cho toàn bộ dự án.

Đảm bảo giao diện dòng lệnh giữa hai kiến trúc unet_base và mamba_unet hoàn toàn đồng nhất.
Tuyệt đối không hard-code danh sách choices của activation và loss; các kiến trúc sẽ tự
truyền danh sách choices từ registry riêng của mình (R11, mục 4.4, 5.1).
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    activation_choices: List[str],
    loss_choices: List[str],
    default_activation: str = "relu",
    default_loss: str = "dice_bce",
    default_batch_size: int = 8,
    default_epochs: int = 50,
    default_lr: float = 1e-4,
    default_seed: int = 42,
):
    """
    Gắn các nhóm tham số chuẩn hóa vào ArgumentParser của lệnh train:
      1. Cấu hình
      2. Dữ liệu
      3. Mô hình chung
      4. Hàm mất mát (Loss)
      5. Tối ưu hóa (Optimizer & Scheduler)
      6. Quản lý phiên chạy (Run & Resume)
    """
    # 1. Cấu hình
    cfg_group = parser.add_argument_group("Cấu hình thí nghiệm")
    cfg_group.add_argument("--config", default=None, help="Đường dẫn file cấu hình YAML ghi đè tham số")

    # 2. Dữ liệu (giá trị mặc định tái lập đúng hành vi R0)
    data_group = parser.add_argument_group("Dữ liệu")
    data_group.add_argument("--cache-dir", default="data/cache_1mm", help="Thư mục dữ liệu cache_1mm riêng cho từng Run")
    data_group.add_argument("--slices-csv", default=None, help="Đường dẫn file slices.csv tùy chỉnh (mặc định: <cache-dir>/slices.csv)")
    data_group.add_argument("--data-manifest", default=None, help="Đường dẫn data_manifest.json (nếu có)")
    data_group.add_argument("--min-train-area", type=float, default=20.0, help="Ngưỡng diện tích u tối thiểu (mm²) khi lập chỉ mục (mặc định R0: 20.0)")
    data_group.add_argument("--neg-ratio", type=float, default=0.25, help="Tỷ lệ lát âm tính trên lát dương tính (mặc định R0: 0.25)")
    data_group.add_argument("--min-neg-per-case", type=int, default=1, help="Số lát âm tính tối thiểu mỗi ca bệnh, gồm cả ca không u (mặc định R0: 1)")
    data_group.add_argument("--dynamic-neg", action="store_true", help="Bật lấy mẫu âm tính động PosAllNegResample mỗi epoch (mặc định R0: tắt)")
    data_group.add_argument("--keep-all-train-neg", action="store_true", help="Giữ toàn bộ lát âm phổi tập train trong slices.csv để cấp kho cho --dynamic-neg (mặc định R0: tắt)")
    data_group.add_argument("--tumor-union-p", type=float, default=1.0, help="Xác suất hợp mask u vào mask phổi khi train (mặc định R0: 1.0)")
    data_group.add_argument("--aug-level", type=int, default=1, choices=[1, 2], help="Cấp độ augment dữ liệu: 1=Flip/Rot90 (mặc định R0), 2=Affine+Jitter (Run 2)")
    data_group.add_argument("--workers", type=int, default=2, help="Số luồng nạp dữ liệu DataLoader")
    data_group.add_argument("--gpu-window", action="store_true", help="Chuyển mảng HU int16 lên GPU để chuẩn hóa cửa sổ")
    data_group.add_argument("--no-lung-mask", action="store_true", help="Không áp mặt nạ phổi ngoài lát cắt")
    data_group.add_argument("--no-augment", action="store_true", help="Tắt tăng cường dữ liệu lật/xoay ngẫu nhiên")
    data_group.add_argument("--limit-train", type=int, default=0, help="Giới hạn số lát train (0 = toàn bộ)")
    data_group.add_argument("--limit-val", type=int, default=0, help="Giới hạn số lát val (0 = toàn bộ)")
    data_group.add_argument("--rebuild-index", action="store_true", help="Bắt buộc tạo lại slices.csv")

    # 3. Mô hình chung
    model_group = parser.add_argument_group("Mô hình chung")
    model_group.add_argument(
        "--activation",
        choices=activation_choices,
        default=default_activation,
        help=f"Hàm kích hoạt (lựa chọn của kiến trúc này: {activation_choices})",
    )

    # 4. Loss
    loss_group = parser.add_argument_group("Hàm mất mát")
    loss_group.add_argument(
        "--loss",
        choices=loss_choices,
        default=default_loss,
        help=f"Hàm mất mát (lựa chọn của kiến trúc này: {loss_choices})",
    )
    loss_group.add_argument("--loss-params", default=None, help="Tham số bổ sung cho loss dưới dạng chuỗi JSON")

    # 5. Tối ưu
    opt_group = parser.add_argument_group("Tối ưu hóa")
    opt_group.add_argument("--epochs", type=int, default=default_epochs, help="Tổng số epoch huấn luyện đích")
    opt_group.add_argument("--batch-size", type=int, default=default_batch_size, help="Kích thước batch")
    opt_group.add_argument("--lr", type=float, default=default_lr, help="Tốc độ học cơ sở (learning rate)")
    opt_group.add_argument("--weight-decay", type=float, default=1e-4, help="Hệ số phạt suy giảm trọng số (mặc định R0: 1e-4)")
    opt_group.add_argument("--ema", type=float, default=0.0, help="Hệ số trượt Exponential Moving Average (mặc định R0: 0.0 = tắt, Run 2 bật 0.999)")
    opt_group.add_argument(
        "--scheduler",
        choices=["cosine", "plateau", "step", "none"],
        default="cosine",
        help="Chiến lược điều chỉnh tốc độ học (LR Scheduler)",
    )
    opt_group.add_argument("--grad-clip", type=float, default=1.0, help="Ngưỡng kẹp gradient tối đa (0 = tắt)")
    opt_group.add_argument("--patience", type=int, default=12, help="Số epoch chờ trước khi dừng sớm Early Stopping")

    # 6. Chạy & Quản lý Run
    run_group = parser.add_argument_group("Quản lý phiên chạy (Run)")
    run_group.add_argument("--runs-root", default=None, help="Thư mục gốc lưu trữ các lần chạy")
    run_group.add_argument(
        "--drive-sync-dir",
        default=None,
        help="Đường dẫn thư mục Google Drive để tự động đồng bộ checkpoints và logs thời gian thực sau mỗi epoch",
    )
    run_group.add_argument("--run-name", default=None, help="Tên tùy chỉnh cho phiên chạy")
    run_group.add_argument("--tag", default=None, help="Thẻ phân loại thí nghiệm (vd: baseline, ablation)")
    run_group.add_argument("--notes", default="", help="Ghi chú chi tiết cho thí nghiệm")
    run_group.add_argument(
        "--resume",
        default=None,
        help="Tiếp tục huấn luyện: 'auto' (run mới nhất dở dang), đường dẫn run_dir hoặc file checkpoint cụ thể",
    )
    run_group.add_argument("--init-weights", default=None, help="Nạp trọng số ban đầu từ checkpoint (khởi tạo run mới)")
    run_group.add_argument("--seed", type=int, default=default_seed, help="Hạt giống ngẫu nhiên")
    run_group.add_argument("--no-amp", action="store_true", help="Tắt Mixed Precision (tự động bật nếu có GPU)")
    run_group.add_argument("--no-progress", action="store_true", help="Tắt thanh tiến trình tqdm chi tiết")


def validate_experiment_flags(args: argparse.Namespace) -> None:
    """Chặn sớm các cờ thuộc Run 2/Run 3 nếu được bật trước khi triển khai."""
    if float(getattr(args, "ema", 0.0)) > 0.0:
        raise NotImplementedError("Cờ --ema > 0 thuộc Giai đoạn Run 2 và chưa được kích hoạt ở Run 1.")
    if int(getattr(args, "aug_level", 1)) != 1:
        raise NotImplementedError("Cờ --aug-level != 1 thuộc Giai đoạn Run 2 và chưa được kích hoạt ở Run 1.")
    if bool(getattr(args, "hr_skip", False)):
        raise NotImplementedError("Cờ --hr-skip thuộc Giai đoạn Run 3 và chưa được kích hoạt ở Run 1.")


def build_run_data_meta(args: argparse.Namespace, cache_dir: Path, image_size: int) -> Dict[str, Any]:
    """Tạo từ điển data_meta đầy đủ để ghi vào run_config.json."""
    slices_csv = getattr(args, "slices_csv", None)
    return {
        "cache_dir": str(cache_dir),
        "slices_csv": str(Path(slices_csv).resolve()) if slices_csv else str((cache_dir / "slices.csv")),
        "image_size": int(image_size),
        "min_train_area": float(getattr(args, "min_train_area", 20.0)),
        "neg_ratio": float(getattr(args, "neg_ratio", 0.25)),
        "min_neg_per_case": int(getattr(args, "min_neg_per_case", 1)),
        "dynamic_neg": bool(getattr(args, "dynamic_neg", False)),
        "keep_all_train_neg": bool(getattr(args, "keep_all_train_neg", False)),
        "tumor_union_p": float(getattr(args, "tumor_union_p", 1.0)),
        "aug_level": int(getattr(args, "aug_level", 1)),
    }
