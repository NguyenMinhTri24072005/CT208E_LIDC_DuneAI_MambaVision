# -*- coding: utf-8 -*-
"""
common/engine/cli_args.py - Các nhóm tham số dòng lệnh dùng chung cho toàn bộ dự án.

Đảm bảo giao diện dòng lệnh giữa hai kiến trúc unet_base và mamba_unet hoàn toàn đồng nhất.
Tuyệt đối không hard-code danh sách choices của activation và loss; các kiến trúc sẽ tự
truyền danh sách choices từ registry riêng của mình (R11, mục 4.4, 5.1).
"""

import argparse
from typing import List, Optional


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

    # 2. Dữ liệu
    data_group = parser.add_argument_group("Dữ liệu")
    data_group.add_argument("--cache-dir", default="data/cache_1mm", help="Thư mục dữ liệu cache_1mm đã chuẩn bị")
    data_group.add_argument("--data-manifest", default=None, help="Đường dẫn data_manifest.json (nếu có)")
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
    opt_group.add_argument("--weight-decay", type=float, default=1e-4, help="Hệ số phạt suy giảm trọng số")
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
