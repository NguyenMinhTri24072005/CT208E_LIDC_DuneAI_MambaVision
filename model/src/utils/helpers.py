"""
helpers.py
Các hàm tiện ích dùng chung trong toàn bộ pipeline huấn luyện:
  - set_seed: Khóa random seed để tái lập kết quả
  - save_checkpoint / load_checkpoint: Lưu/tải trọng số model
  - count_parameters: Đếm số tham số model
  - plot_predictions: Vẽ ảnh CT, mask thật, mask dự đoán cạnh nhau
  - EarlyStopping: Class theo dõi và dừng train sớm
"""

import os
import random
import numpy as np
import torch
import matplotlib
try:
    # Trên Colab / Jupyter: dùng backend inline (tự động)
    # Trên server không GUI: fallback về Agg
    import matplotlib.pyplot as plt
    plt.figure()  # Thử tạo figure
    plt.close()
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# 1. SET SEED — Tái lập kết quả
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Khóa toàn bộ nguồn random để đảm bảo kết quả tái lập được
    khi chạy lại cùng code và cùng config.

    Args:
        seed: Giá trị seed nguyên dương. Mặc định 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Buộc CuDNN dùng thuật toán cố định (chậm hơn 1 chút nhưng tái lập được)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Đã khóa random seed = {seed}")


# =============================================================================
# 2. CHECKPOINT — Lưu và tải trọng số
# =============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_dice: float,
    config: dict,
    is_best: bool = False,
) -> None:
    """
    Lưu trạng thái model, optimizer, epoch và val_dice vào file .pth.

    Args:
        model:      Model PyTorch cần lưu.
        optimizer:  Optimizer cần lưu (để tiếp tục train sau).
        epoch:      Số epoch hiện tại.
        val_dice:   Điểm Dice trên tập validation tại epoch này.
        config:     Dict cấu hình (đọc từ YAML) — lưu lại để biết thí nghiệm nào.
        is_best:    True = đây là model tốt nhất từ trước đến nay.
    """
    # Nếu cấu hình tắt lưu (enabled: false), bỏ qua không ghi file
    if not config.get("SAVE", {}).get("enabled", True):
        return

    checkpoint_dir = Path(config["SAVE"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    exp_name = config["SAVE"].get("experiment_name", "experiment")
    filename = checkpoint_dir / f"{exp_name}_epoch{epoch:03d}_dice{val_dice:.4f}.pth"

    state = {
        "epoch": epoch,
        "val_dice": val_dice,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    torch.save(state, filename)
    print(f"[Checkpoint] Đã lưu: {filename}")

    # Nếu là model tốt nhất, lưu thêm bản "best.pth" dễ tìm
    if is_best:
        best_path = checkpoint_dir / f"{exp_name}_best.pth"
        torch.save(state, best_path)
        print(f"[Checkpoint] Model tốt nhất → {best_path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None,
) -> dict:
    """
    Tải trọng số từ file .pth vào model (và optimizer nếu muốn tiếp tục train).

    Args:
        path:      Đường dẫn tới file .pth.
        model:     Model cần nạp trọng số vào.
        optimizer: Optimizer cần nạp trạng thái vào (None = chỉ tải model).
        device:    Thiết bị đích (cuda / cpu). None = tự detect.

    Returns:
        Dict chứa {'epoch', 'val_dice', 'config'} từ checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    val_dice = checkpoint.get("val_dice", 0.0)
    print(f"[Checkpoint] Đã tải từ: {path}  (epoch={epoch}, val_dice={val_dice:.4f})")

    return {
        "epoch": epoch,
        "val_dice": val_dice,
        "config": checkpoint.get("config", {}),
    }


# =============================================================================
# 3. COUNT PARAMETERS — Đếm số tham số model
# =============================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """
    Đếm tổng số tham số có thể huấn luyện (trainable) trong model.

    Args:
        model: Model PyTorch.

    Returns:
        Tổng số tham số (int).
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Tổng số tham số huấn luyện: {total:,}")
    return total


# =============================================================================
# 4. PLOT PREDICTIONS — Vẽ kết quả trực quan
# =============================================================================

def plot_predictions(
    image: np.ndarray,
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    save_path: str = None,
    title: str = "",
) -> None:
    """
    Vẽ 3 ảnh cạnh nhau: Ảnh CT gốc | Mask thật | Mask AI dự đoán.

    Args:
        image:      Ảnh CT, shape (H, W) hoặc (3, H, W). Kênh giữa được hiển thị.
        mask_true:  Mask nhị phân thật, shape (H, W) hoặc (1, H, W).
        mask_pred:  Mask nhị phân dự đoán, shape (H, W) hoặc (1, H, W).
        save_path:  Nếu truyền vào, lưu ảnh ra file thay vì hiển thị.
        title:      Tiêu đề cho figure (ví dụ: "Epoch 10 | Val Dice: 0.72").
    """
    # Chuẩn hóa về (H, W)
    if image.ndim == 3:
        img_show = image[1]  # Lấy lát cắt giữa (kênh Z)
    else:
        img_show = image

    if mask_true.ndim == 3:
        mask_true = mask_true[0]
    if mask_pred.ndim == 3:
        mask_pred = mask_pred[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(img_show, cmap="gray")
    axes[0].set_title("Ảnh CT (lát Z)")
    axes[0].axis("off")

    axes[1].imshow(mask_true, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Mask thật (Ground Truth)")
    axes[1].axis("off")

    axes[2].imshow(mask_pred, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("Mask dự đoán (AI)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Đã lưu ảnh: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# =============================================================================
# 5. EARLY STOPPING — Dừng train sớm khi model không cải thiện
# =============================================================================

class EarlyStopping:
    """
    Theo dõi val_dice qua từng epoch. Nếu không cải thiện sau
    `patience` epoch liên tiếp, đặt cờ `self.early_stop = True`.

    Cách dùng trong train loop:
        early_stop = EarlyStopping(patience=15)
        for epoch in range(epochs):
            val_dice = validate(...)
            early_stop(val_dice)
            if early_stop.early_stop:
                print("Dừng sớm!")
                break
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        """
        Args:
            patience:  Số epoch chờ trước khi dừng.
            min_delta: Mức cải thiện tối thiểu để tính là "có cải thiện".
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_dice: float) -> bool:
        """
        Gọi sau mỗi epoch với val_dice mới.

        Returns:
            True nếu đây là điểm tốt nhất (model nên được lưu lại).
        """
        is_best = False

        if self.best_score is None:
            self.best_score = val_dice
            is_best = True
        elif val_dice > self.best_score + self.min_delta:
            self.best_score = val_dice
            self.counter = 0
            is_best = True
        else:
            self.counter += 1
            print(f"[EarlyStopping] Không cải thiện {self.counter}/{self.patience} epoch")
            if self.counter >= self.patience:
                self.early_stop = True

        return is_best
