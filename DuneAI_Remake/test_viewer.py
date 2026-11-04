import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import sys

def load_nrrd(file_path):
    """Hàm đọc file NRRD thành mảng Numpy"""
    try:
        sitk_img = sitk.ReadImage(file_path)
        # SimpleITK trả về mảng (Z, Y, X), ta giữ nguyên
        volume = sitk.GetArrayFromImage(sitk_img)
        spacing = sitk_img.GetSpacing()
        return volume, spacing
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        sys.exit(1)

def view_slices_interactive(volume, mask=None, title="CT Viewer"):
    """Công cụ kéo thanh trượt để xem lát cắt 3D"""
    num_slices = volume.shape[0]
    
    # Thiết lập giao diện đồ họa
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2) # Chừa chỗ cho thanh trượt
    
    # Hiển thị lát cắt đầu tiên (nằm giữa)
    init_slice = num_slices // 2
    img_display = ax.imshow(volume[init_slice], cmap='gray')
    
    # Nếu có truyền mặt nạ, vẽ đè lên với màu đỏ hơi trong suốt (alpha=0.4)
    mask_display = None
    if mask is not None:
        # Tạo colormap đè (chỉ hiện màu đỏ ở pixel có giá trị > 0)
        mask_display = ax.imshow(mask[init_slice], cmap='Reds', alpha=0.4, vmin=0, vmax=1)
        # Ẩn mặt nạ ở những chỗ pixel = 0
        mask_display.set_data(np.ma.masked_where(mask[init_slice] == 0, mask[init_slice]))

    ax.set_title(f"{title} - Lát cắt: {init_slice}/{num_slices-1}")
    ax.axis('off')

    # Tạo thanh trượt (Slider)
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Z-Axis',
        valmin=0,
        valmax=num_slices - 1,
        valinit=init_slice,
        valstep=1
    )

    # Hàm cập nhật hình ảnh khi kéo thanh trượt
    def update(val):
        idx = int(slider.val)
        img_display.set_data(volume[idx])
        
        # Chỉnh dải màu (contrast) linh hoạt theo từng lát cắt
        img_display.set_clim(vmin=volume[idx].min(), vmax=volume[idx].max())
        
        if mask is not None:
            mask_display.set_data(np.ma.masked_where(mask[idx] == 0, mask[idx]))
            
        ax.set_title(f"{title} - Lát cắt: {idx}/{num_slices-1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    # ==========================================
    # CÁCH TEST: Thay đường dẫn tới file NRRD của bạn vào đây
    # ==========================================
    # Đảm bảo bạn đã tải 1 bệnh nhân LIDC-IDRI-0001 về máy cục bộ
    IMAGE_PATH = r"D:\GG_1\COMPUTER SCIENCE\NAM4_HK1_(2026-2027)\CT208E\SOURCES\data\duneai_nrrd\LIDC-IDRI-0002\image.nrrd"
    MASK_PATH = r"D:\GG_1\COMPUTER SCIENCE\NAM4_HK1_(2026-2027)\CT208E\SOURCES\data\duneai_nrrd\LIDC-IDRI-0002\mask.nrrd"

    print("Đang nạp dữ liệu CT...")
    vol_img, spacing_img = load_nrrd(IMAGE_PATH)
    print(f"Ảnh CT gốc: {vol_img.shape} | Spacing: {spacing_img}")
    
    # Nạp mặt nạ (Nếu có)
    vol_mask, spacing_mask = load_nrrd(MASK_PATH)
    
    # HIỂN THỊ:
    # 1. Nếu chỉ muốn xem ảnh trắng đen thô:
    # view_slices_interactive(vol_img, title="Ảnh CT Thô (Chưa xử lý)")
    
    # 2. Nếu muốn xem ảnh kèm khối u đè lên (Bỏ comment dòng dưới):
    view_slices_interactive(vol_img, mask=vol_mask, title="Ảnh CT kèm Khối U")