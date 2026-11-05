# configs/config.py

# ==========================================
# THÔNG SỐ KHÔNG GIAN & VẬT LÝ Y TẾ
# ==========================================
IMAGE_SIZE = 512

# Cửa sổ Hounsfield cho mô phổi (Lung Window)
WINDOW_WIDTH = 1500
WINDOW_LEVEL = -600
RESAMPLING_STEP = 25

# ==========================================
# THÔNG SỐ LỌC DỮ LIỆU BỆNH NHÂN
# ==========================================
# Ngưỡng diện tích nhỏ nhất để giữ lại lát cắt có khối u
TUMOR_SIZE_LOW_LIM = 20.0  # mm^2

# Tỷ lệ lấy mẫu các lát cắt âm tính (không có khối u) để cân bằng dữ liệu
ZERO_SL_RATIO = 0.25

# ==========================================
# THÔNG SỐ HUẤN LUYỆN
# ==========================================
VAL_SPLIT = 0.15
SEED = 42

BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4