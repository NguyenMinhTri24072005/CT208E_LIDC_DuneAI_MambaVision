import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import numpy as np
import sys
from pathlib import Path

# Nạp các hàm từ src.medical_prep
sys.path.append(str(Path(__file__).resolve().parent))
from src.medical_prep import (load_nrrd, resize_and_pad_3d, clip_and_window_hu,
                              extract_lungs_mask, lung_keep_mask, apply_lung_mask)

# Thứ tự đúng với pipeline của tác giả: resize + pad trên HU thô, rồi mới kẹp cửa sổ.
STAGE_RAW    = "1. Ảnh Thô Nguyên Bản"
STAGE_RESIZE = "2. Resize 1mm + Pad (HU)"
STAGE_HU     = "3. Kẹp Cửa Sổ (HU)"
STAGE_LUNG   = "4. Tách Phổi + Cửa Sổ (chậm)"
STAGES = [STAGE_RAW, STAGE_RESIZE, STAGE_HU, STAGE_LUNG]
MASK_LABEL = "Hiện mask nốt"


class LazyViewer:
    """Mới mở chỉ có ảnh gốc. Mỗi bước chỉ được tính khi bạn tích chọn, và được cache lại."""

    def __init__(self, image_path, mask_path=None, title="Tiền xử lý CT"):
        print("Đang nạp ảnh CT thô...")
        self.vol_raw, self.spacing = load_nrrd(image_path)
        self.mask_raw = None
        if mask_path and Path(mask_path).exists():
            self.mask_raw, _ = load_nrrd(mask_path)
        else:
            print("(Không có mask.nrrd, bỏ qua lớp mặt nạ)")

        self.title = title
        self.vols = {STAGE_RAW: self.vol_raw}       # cache ảnh theo giai đoạn
        self.masks = {STAGE_RAW: self.mask_raw}     # cache mask nốt theo giai đoạn
        self.state = {s: (s == STAGE_RAW) for s in STAGES}   # mới đầu chỉ ảnh gốc
        self.show_mask = False
        self.n0 = self.vol_raw.shape[0]
        self.idx = self.n0 // 2
        self.axes, self.displays = [], []

        self.fig = plt.figure(figsize=(18, 7))
        self.fig.canvas.manager.set_window_title(title)

        # Ô tích chọn giai đoạn (+ mask)
        labels = STAGES + ([MASK_LABEL] if self.mask_raw is not None else [])
        actives = [self.state[s] for s in STAGES] + ([False] if self.mask_raw is not None else [])
        ax_chk = self.fig.add_axes([0.01, 0.36, 0.19, 0.28])
        ax_chk.set_title("Chọn bước hiển thị", fontsize=10, fontweight="bold")
        self.check = CheckButtons(ax_chk, labels, actives)
        self.check.on_clicked(self.on_check)

        # Thanh trượt Z
        ax_sl = self.fig.add_axes([0.25, 0.04, 0.65, 0.03])
        self.slider = Slider(ax_sl, "Lát cắt Z", 0, self.n0 - 1, valinit=self.idx, valstep=1)
        self.slider.on_changed(self.on_slide)

        self.redraw()

    # ---------- tính toán lười (chỉ chạy khi được chọn) ----------
    def get_vol(self, stage):
        if stage in self.vols:
            return self.vols[stage]
        if stage == STAGE_RESIZE:
            # resize trên HU thô; ảnh được pad bằng -1000 HU (mặc định của resize_and_pad_3d)
            self.vols[stage] = resize_and_pad_3d(self.vol_raw, self.spacing)
        elif stage == STAGE_HU:
            self.vols[stage] = clip_and_window_hu(self.get_vol(STAGE_RESIZE))
        elif stage == STAGE_LUNG:
            # tách phổi trên HU thô ở lưới gốc, resize mask về 1 mm, che ảnh bằng -1000, rồi cửa sổ.
            # Không hợp mask nốt vào (đó chỉ dành cho huấn luyện, dùng ở đây là rò rỉ nhãn).
            lung = extract_lungs_mask(np.maximum(self.vol_raw, -1000))
            if lung is None:
                raise RuntimeError("không tìm thấy phổi")
            lung_r = resize_and_pad_3d(lung, self.spacing, is_mask=True)
            self.vols[stage] = clip_and_window_hu(apply_lung_mask(self.get_vol(STAGE_RESIZE), lung_r))
        return self.vols[stage]

    def get_mask(self, stage):
        if self.mask_raw is None:
            return None
        if stage in self.masks:
            return self.masks[stage]
        # mọi bước sau bước 1 dùng chung mask nốt đã resize (nearest)
        if STAGE_RESIZE not in self.masks:
            self.masks[STAGE_RESIZE] = resize_and_pad_3d(self.mask_raw, self.spacing, is_mask=True)
        self.masks[stage] = self.masks[STAGE_RESIZE]
        return self.masks[stage]

    # ---------- vẽ ----------
    @staticmethod
    def _slice(arr, idx, n0):
        j = int(round(idx / max(n0 - 1, 1) * (arr.shape[0] - 1)))   # ánh xạ nếu số lát khác nhau
        return arr[j]

    def _status(self, text):
        self.fig.suptitle(text, fontsize=13)
        self.fig.canvas.draw()
        plt.pause(0.01)

    def redraw(self):
        for ax in self.axes:
            ax.remove()
        self.axes, self.displays = [], []

        active = [s for s in STAGES if self.state[s]]
        n = len(active)
        left, right, gap = 0.22, 0.99, 0.01
        w = (right - left - gap * (n - 1)) / n

        for i, stage in enumerate(active):
            self._status(f"Đang xử lý: {stage} ...")
            try:
                vol = self.get_vol(stage)
                mask = self.get_mask(stage) if self.show_mask else None
            except Exception as e:
                print(f"❌ Lỗi ở bước '{stage}': {e}")
                self._deactivate(stage)
                return self.redraw()

            ax = self.fig.add_axes([left + i * (w + gap), 0.12, w, 0.76])
            img = ax.imshow(self._slice(vol, self.idx, self.n0), cmap="gray")
            m_disp = None
            if mask is not None:
                m = self._slice(mask, self.idx, self.n0)
                m_disp = ax.imshow(np.ma.masked_where(m == 0, m), cmap="Reds",
                                   alpha=0.4, vmin=0, vmax=1)
            z, h, wd = vol.shape
            ax.set_title(f"{stage}\n(Shape: {z}x{h}x{wd})", fontsize=10, fontweight="bold")
            ax.axis("off")
            self.axes.append(ax)
            self.displays.append((vol, mask, img, m_disp))

        self.update_frames()

    def update_frames(self):
        for vol, mask, img, m_disp in self.displays:
            s = self._slice(vol, self.idx, self.n0)
            img.set_data(s)
            img.set_clim(vmin=s.min(), vmax=s.max())
            if m_disp is not None:
                m = self._slice(mask, self.idx, self.n0)
                m_disp.set_data(np.ma.masked_where(m == 0, m))
        self.fig.suptitle(f"{self.title} - Z: {self.idx}/{self.n0 - 1}", fontsize=13)
        self.fig.canvas.draw_idle()

    # ---------- sự kiện ----------
    def _deactivate(self, stage):
        self.state[stage] = False
        i = STAGES.index(stage)
        self.check.eventson = False       # bỏ tích mà không kích hoạt callback
        self.check.set_active(i)
        self.check.eventson = True

    def on_check(self, label):
        if label == MASK_LABEL:
            self.show_mask = not self.show_mask
        else:
            self.state[label] = not self.state[label]
            if not any(self.state.values()):      # luôn giữ ít nhất 1 ảnh
                self.state[STAGE_RAW] = True
                self.check.eventson = False
                self.check.set_active(0)
                self.check.eventson = True
        self.redraw()

    def on_slide(self, val):
        self.idx = int(val)
        self.update_frames()


if __name__ == "__main__":
    NRRD_DIR   = Path(r"D:\GG_1\COMPUTER SCIENCE\NAM4_HK1_(2026-2027)\CT208E\SOURCES\data\duneai_raw_nrrd")
    PATIENT_ID = "LIDC-IDRI-0002"

    viewer = LazyViewer(
        NRRD_DIR / PATIENT_ID / "image.nrrd",
        NRRD_DIR / PATIENT_ID / "mask.nrrd",
        title=f"Tiền xử lý CT - {PATIENT_ID}",
    )
    plt.show()
