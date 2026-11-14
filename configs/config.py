# -*- coding: utf-8 -*-
"""
configs/config.py - Hằng số y tế và tham số tiền xử lý ảnh CT chuẩn của dự án DuneAI.

Giữ nguyên vẹn các hằng số không gian & chuẩn hóa lát cắt từ phiên bản gốc.
Không chứa siêu tham số huấn luyện của từng thí nghiệm (được quản lý qua YAML và CLI).
"""

import os

# ==========================================
# THÔNG SỐ KHÔNG GIAN & VẬT LÝ Y TẾ
# ==========================================
IMAGE_SIZE = int(os.environ.get("DUNEAI_IMAGE_SIZE", 512))

# Cửa sổ Hounsfield cho mô phổi (Lung Window)
WINDOW_WIDTH = 1500
WINDOW_LEVEL = -600
RESAMPLING_STEP = 25

# ==========================================
# THÔNG SỐ LỌC DỮ LIỆU BỆNH NHÂN
# ==========================================
# Ngưỡng diện tích nhỏ nhất để giữ lại lát cắt có khối u (mm^2)
TUMOR_SIZE_LOW_LIM = 20.0

# Tỷ lệ lấy mẫu các lát cắt âm tính (không có khối u) để cân bằng dữ liệu
ZERO_SL_RATIO = 0.25

# ==========================================
# THÔNG SỐ CHIA TẬP MẶC ĐỊNH
# ==========================================
VAL_SPLIT = 0.15
SEED = 42

# Giá trị mặc định hỗ trợ tương thích ngược cho các module cũ
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
