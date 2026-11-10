# -*- coding: utf-8 -*-
"""
common package - Lõi dùng chung cho toàn bộ dự án DuneAI:
  - data_pipeline/: Tiền xử lý ảnh CT, trích xuất chỉ mục và chuẩn bị runtime dữ liệu
  - engine/: Vòng lặp huấn luyện lõi BaseTrainer, quản lý Run, checkpoint I/O, cli_args
  - eval3d/: Các phép biến đổi 3D, tính toán chỉ số hình thái 3D và suy luận thể tích
  - metrics.py: Các hàm đo lường phân đoạn lát cắt 2.5D (Dice, IoU, ...)
  - paths.py: Quản lý đường dẫn gốc và thư mục mặc định

TUYỆT ĐỐI KHÔNG chứa bất kỳ lớp nn.Module nào (R11).
"""
