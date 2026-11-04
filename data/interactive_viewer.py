import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
import os

def main(pid="LIDC-IDRI-0006"):
    print(f"Đang tải dữ liệu 3D của bệnh nhân {pid}...")
    
    # Đường dẫn tương đối từ thư mục Data Mining
    img_path = f"processed/volumes/{pid}_img.npy"
    votes_path = f"processed/volumes/{pid}_votes.npy"
    
    if not os.path.exists(img_path) or not os.path.exists(votes_path):
        print(f"LỖI: Không tìm thấy file dữ liệu cho {pid}.")
        print(f"Hãy chắc chắn file {img_path} tồn tại.")
        sys.exit(1)
        
    # Nạp nguyên khối 3D vào RAM
    img_vol = np.load(img_path)
    votes_vol = np.load(votes_path)
    
    D, H, W = img_vol.shape
    print(f"Tải thành công! Kích thước khối 3D: Z={D}, Y={H}, X={W}")

    # Tạo giao diện cửa sổ
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2) # Chừa khoảng trống bên dưới cho thanh trượt
    
    # Tìm tự động lát cắt chứa khối u to nhất để hiển thị đầu tiên
    area = votes_vol.reshape(D, -1).sum(1)
    init_z = int(area.argmax()) if area.max() > 0 else D // 2
    
    # Hiển thị ảnh CT lồng ngực (trắng đen)
    img_display = ax.imshow(img_vol[init_z], cmap='gray', vmin=0, vmax=1)
    
    # Đè lớp màu (khối u) lên trên ảnh CT
    # np.ma.masked_where giúp làm trong suốt những chỗ không có khối u (giá trị 0)
    overlay_data = np.ma.masked_where(votes_vol[init_z] == 0, votes_vol[init_z])
    vote_display = ax.imshow(overlay_data, cmap='autumn_r', vmin=1, vmax=4, alpha=0.6)
    
    ax.set_title(f"Bệnh nhân: {pid} | Lát cắt Z = {init_z}/{D-1}")
    ax.axis('off')
    
    # Vẽ thanh trượt (Slider) để cuộn theo trục Z
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Cuộn trục Z', 0, D - 1, valinit=init_z, valstep=1)
    
    # Hàm cập nhật lại hình ảnh mỗi khi bạn kéo thanh trượt
    def update(val):
        z = int(slider.val)
        img_display.set_data(img_vol[z])
        
        new_overlay = np.ma.masked_where(votes_vol[z] == 0, votes_vol[z])
        vote_display.set_data(new_overlay)
        
        ax.set_title(f"Bệnh nhân: {pid} | Lát cắt Z = {z}/{D-1}")
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    
    print("Mở cửa sổ giao diện... (Bạn có thể kéo thanh trượt bên dưới ảnh để đi xuyên qua lồng ngực)")
    plt.show()

if __name__ == "__main__":
    # Cho phép truyền tên bệnh nhân từ dòng lệnh (mặc định là 1002)
    target_pid = sys.argv[1] if len(sys.argv) > 1 else "LIDC-IDRI-1002"
    main(target_pid)
