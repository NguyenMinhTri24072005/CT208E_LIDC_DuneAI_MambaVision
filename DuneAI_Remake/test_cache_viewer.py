# -*- coding: utf-8 -*-
"""
test_cache_viewer.py - Trình xem trực quan dữ liệu cache_1mm đúng như đầu vào của mạng DuneUNet.

Hỗ trợ bật/tắt trực tiếp:
  1. Mask U (Đỏ)
  2. Viền Phổi (Xanh)
  3. Cửa sổ HU [-1000, 150] -> [0, 1]
  4. Cắt ngoài phổi (AI Input): Áp dụng đúng lung_keep_mask + apply_lung_mask để xóa sạch da, thịt, xương, bàn chụp.
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
for p in (str(THIS_DIR), str(THIS_DIR / "src"), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.medical_prep import (
    HU_FLOOR,
    apply_lung_mask,
    clip_and_window_hu,
    lung_keep_mask,
)


class CacheViewer:
    def __init__(self, patient_cache_dir):
        self.p_dir = Path(patient_cache_dir)
        print(f"Đang nạp dữ liệu Cache từ: {self.p_dir}...")

        self.img = np.load(self.p_dir / "image_1mm.npy", mmap_mode="r")
        self.msk = np.load(self.p_dir / "mask_1mm.npy", mmap_mode="r")

        lung_path = self.p_dir / "lung_1mm.npy"
        self.lung = np.load(lung_path, mmap_mode="r") if lung_path.exists() else None

        with open(self.p_dir / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.n_slices = self.img.shape[0]
        # Tự động nhảy tới lát cắt có khối u lớn nhất để kiểm tra ngay khi mở
        tumor_areas = (np.asarray(self.msk) > 0).reshape(self.n_slices, -1).sum(axis=1)
        self.idx = int(np.argmax(tumor_areas)) if tumor_areas.max() > 0 else self.n_slices // 2

        # Trạng thái hiển thị mặc định: Bật sẵn Cửa sổ HU và Cắt ngoài phổi (đúng ảnh AI nhìn thấy)
        self.show_tumor = True
        self.show_lung_overlay = False
        self.apply_window = True
        self.crop_outside_lung = True

        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.fig.canvas.manager.set_window_title(f"DuneAI Cache Viewer - {self.p_dir.name}")
        plt.subplots_adjust(bottom=0.20, left=0.25)
        self.ax.axis("off")

        ax_chk = self.fig.add_axes([0.02, 0.35, 0.21, 0.26])
        ax_chk.set_title("Tùy chọn hiển thị", fontsize=10, fontweight="bold")
        self.labels = [
            "1. Mask U (Đỏ)",
            "2. Phủ Phổi (Xanh)",
            "3. Cửa sổ HU",
            "4. Cắt ngoài phổi (AI)",
        ]
        self.check = CheckButtons(
            ax_chk,
            self.labels,
            [self.show_tumor, self.show_lung_overlay, self.apply_window, self.crop_outside_lung],
        )
        self.check.on_clicked(self.on_check)

        ax_sl = self.fig.add_axes([0.25, 0.08, 0.65, 0.03])
        self.slider = Slider(ax_sl, "Lát cắt Z", 0, self.n_slices - 1, valinit=self.idx, valstep=1)
        self.slider.on_changed(self.on_slide)

        h, w = self.img.shape[1], self.img.shape[2]
        self.img_display = self.ax.imshow(np.zeros((h, w), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
        self.msk_display = self.ax.imshow(np.zeros((h, w), dtype=np.float32), cmap="Reds", alpha=0.45, vmin=0, vmax=1)
        self.lung_display = (
            self.ax.imshow(np.zeros((h, w), dtype=np.float32), cmap="Blues", alpha=0.22, vmin=0, vmax=1)
            if self.lung is not None
            else None
        )

        self.update_view()

    def on_check(self, label):
        if label == self.labels[0]:
            self.show_tumor = not self.show_tumor
        elif label == self.labels[1]:
            self.show_lung_overlay = not self.show_lung_overlay
        elif label == self.labels[2]:
            self.apply_window = not self.apply_window
        elif label == self.labels[3]:
            self.crop_outside_lung = not self.crop_outside_lung
        self.update_view()

    def on_slide(self, val):
        self.idx = int(val)
        self.update_view()

    def update_view(self):
        z = self.idx
        img_slice = np.asarray(self.img[z : z + 1])  # (1, H, W)
        msk_slice = np.asarray(self.msk[z : z + 1])  # (1, H, W)

        # Bước 1: Nếu bật "4. Cắt ngoài phổi (AI)", chạy đúng lung_keep_mask + apply_lung_mask như dataset.py
        if self.crop_outside_lung and self.lung is not None:
            lung_slice = np.asarray(self.lung[z : z + 1])
            keep = lung_keep_mask(lung_slice, tumor_mask=msk_slice)
            img_slice = apply_lung_mask(img_slice, keep, fill=HU_FLOOR)

        img_2d = img_slice[0]
        msk_2d = (msk_slice[0] > 0).astype(np.float32)

        # Bước 2: Áp cửa sổ HU [-1000, 150] -> [0, 1]
        if self.apply_window:
            img_2d = clip_and_window_hu(img_2d)
            self.img_display.set_clim(vmin=0.0, vmax=1.0)
        else:
            self.img_display.set_clim(vmin=-1000.0, vmax=400.0)

        self.img_display.set_data(img_2d)

        # Bước 3: Lớp phủ khối u (Đỏ)
        if self.show_tumor:
            self.msk_display.set_data(np.ma.masked_where(msk_2d == 0, msk_2d))
        else:
            self.msk_display.set_data(np.ma.masked_all_like(msk_2d))

        # Bước 4: Lớp phủ vùng phổi (Xanh)
        if self.lung_display is not None:
            if self.show_lung_overlay:
                l_2d = (np.asarray(self.lung[z]) > 0).astype(np.float32)
                self.lung_display.set_data(np.ma.masked_where(l_2d == 0, l_2d))
            else:
                self.lung_display.set_data(np.ma.masked_all_like(msk_2d))

        area_mm2 = int(msk_2d.sum())
        lung_status = "Đã cắt da/xương" if (self.crop_outside_lung and self.lung is not None) else "Ảnh gốc 1mm"
        self.ax.set_title(
            f"Lát cắt Z: {z}/{self.n_slices - 1} | Shape: {img_2d.shape[0]}x{img_2d.shape[1]} | "
            f"U: {area_mm2} mm² | Chế độ: {lung_status}",
            fontsize=11,
            fontweight="bold",
        )
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    # Tự động tìm thư mục data/cache_1mm (nằm cùng cấp với DuneAI_Remake hoặc bên trong)
    candidates = [
        ROOT_DIR / "data" / "cache_1mm",
        THIS_DIR / "data" / "cache_1mm",
    ]
    CACHE_DIR = next((p for p in candidates if p.exists()), candidates[0])

    if not CACHE_DIR.exists():
        print(f"❌ Không tìm thấy thư mục Cache: {CACHE_DIR}")
        sys.exit(1)

    patients = sorted([p for p in CACHE_DIR.iterdir() if p.is_dir() and (p / "image_1mm.npy").exists()])
    if not patients:
        print(f"❌ Thư mục {CACHE_DIR} chưa có bệnh nhân nào hoàn tất!")
        sys.exit(1)

    PATIENT_DIR = patients[0]
    viewer = CacheViewer(PATIENT_DIR)
    plt.show()