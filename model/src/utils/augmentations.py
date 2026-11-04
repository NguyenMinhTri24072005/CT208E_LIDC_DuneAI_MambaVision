"""
augmentations.py
----------------
Module Data Augmentation cho ảnh y tế CT (ảnh grayscale lưu dưới dạng numpy array).

Quy ước:
    - Ảnh đầu vào : numpy array shape (H, W) hoặc (3, H, W) – 3 lát cắt 2.5D
    - Mask đầu vào: numpy array shape (H, W) hoặc (1, H, W) – nhị phân {0, 1}
    - Mỗi transform đều nhận (image, mask) và trả về (image, mask) đã biến đổi.
    - Tất cả phép biến đổi hình học đều áp dụng ĐỒNG THỜI cho cả ảnh lẫn mask
      bằng cách dùng cùng một random seed / tham số ngẫu nhiên.

Tác giả  : CT208E – Deep Learning Medical Imaging
Ngày     : 2026-10-01
"""

import numpy as np
from scipy.ndimage import rotate as nd_rotate


# ---------------------------------------------------------------------------
# Các hàm tiện ích nội bộ
# ---------------------------------------------------------------------------

def _is_multichannel(arr: np.ndarray) -> bool:
    """Kiểm tra xem array có nhiều kênh (C, H, W) hay không (H, W)."""
    return arr.ndim == 3


def _apply_on_channels(func, arr: np.ndarray, **kwargs):
    """
    Áp dụng hàm `func` lên từng kênh của array.
    Nếu arr.shape == (H, W)    → áp dụng trực tiếp.
    Nếu arr.shape == (C, H, W) → áp dụng từng kênh rồi stack lại.
    """
    if _is_multichannel(arr):
        channels = [func(arr[c], **kwargs) for c in range(arr.shape[0])]
        return np.stack(channels, axis=0)
    return func(arr, **kwargs)


# ---------------------------------------------------------------------------
# 1. RandomHorizontalFlip
# ---------------------------------------------------------------------------

class RandomHorizontalFlip:
    """
    Lật ngang (flip theo trục W) với xác suất p.
    Áp dụng ĐỒNG THỜI cho cả image và mask.
    """

    def __init__(self, p: float = 0.5):
        """
        Args:
            p (float): Xác suất thực hiện phép lật. Mặc định 0.5.
        """
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """
        Args:
            image: numpy array shape (H, W) hoặc (3, H, W).
            mask : numpy array shape (H, W) hoặc (1, H, W).
        Returns:
            (image, mask) đã được lật ngang (hoặc giữ nguyên).
        """
        if np.random.random() < self.p:
            # np.flip trên trục cuối cùng (axis=-1) tương đương lật ngang
            image = np.flip(image, axis=-1).copy()
            mask  = np.flip(mask,  axis=-1).copy()
        return image, mask


# ---------------------------------------------------------------------------
# 2. RandomVerticalFlip
# ---------------------------------------------------------------------------

class RandomVerticalFlip:
    """
    Lật dọc (flip theo trục H) với xác suất p.
    Áp dụng ĐỒNG THỜI cho cả image và mask.
    """

    def __init__(self, p: float = 0.5):
        """
        Args:
            p (float): Xác suất thực hiện phép lật. Mặc định 0.5.
        """
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """
        Args:
            image: numpy array shape (H, W) hoặc (3, H, W).
            mask : numpy array shape (H, W) hoặc (1, H, W).
        Returns:
            (image, mask) đã được lật dọc (hoặc giữ nguyên).
        """
        if np.random.random() < self.p:
            # Axis -2 tương ứng với chiều H (hàng)
            image = np.flip(image, axis=-2).copy()
            mask  = np.flip(mask,  axis=-2).copy()
        return image, mask


# ---------------------------------------------------------------------------
# 3. RandomRotation
# ---------------------------------------------------------------------------

class RandomRotation:
    """
    Xoay ngẫu nhiên trong khoảng [-degrees, +degrees] với xác suất p.
    Dùng scipy.ndimage.rotate; vùng ngoài biên được lấp bằng 0.
    Áp dụng CÙNG GÓC XOAY cho cả image và mask.
    """

    def __init__(self, degrees: float = 15, p: float = 0.3):
        """
        Args:
            degrees (float): Góc xoay tối đa (độ). Mặc định 15.
            p       (float): Xác suất thực hiện phép xoay. Mặc định 0.3.
        """
        self.degrees = degrees
        self.p = p

    def _rotate_arr(self, arr: np.ndarray, angle: float, order: int) -> np.ndarray:
        """
        Xoay array 2-D hoặc (C, H, W).
        Args:
            arr  : numpy array.
            angle: góc xoay (độ), dương = ngược chiều kim đồng hồ.
            order: bậc nội suy (1 = tuyến tính cho ảnh, 0 = nearest cho mask).
        """
        if _is_multichannel(arr):
            channels = [
                nd_rotate(arr[c], angle, axes=(0, 1),
                          reshape=False, order=order, cval=0.0)
                for c in range(arr.shape[0])
            ]
            return np.stack(channels, axis=0)
        return nd_rotate(arr, angle, axes=(0, 1),
                         reshape=False, order=order, cval=0.0)

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """
        Args:
            image: numpy array shape (H, W) hoặc (3, H, W).
            mask : numpy array shape (H, W) hoặc (1, H, W).
        Returns:
            (image, mask) đã được xoay cùng một góc (hoặc giữ nguyên).
        """
        if np.random.random() < self.p:
            # Lấy ngẫu nhiên một góc trong khoảng [-degrees, +degrees]
            angle = np.random.uniform(-self.degrees, self.degrees)

            # Ảnh: nội suy bậc 1 (tuyến tính) để giữ chất lượng tốt
            image = self._rotate_arr(image, angle, order=1)

            # Mask nhị phân: nội suy bậc 0 (nearest-neighbor) để giữ giá trị {0, 1}
            mask  = self._rotate_arr(mask,  angle, order=0)
            mask  = (mask > 0.5).astype(mask.dtype)   # đảm bảo nhị phân

        return image, mask


# ---------------------------------------------------------------------------
# 4. RandomCrop
# ---------------------------------------------------------------------------

class RandomCrop:
    """
    Cắt ngẫu nhiên một vùng kích thước (size x size).
    Vị trí cắt GIỐNG NHAU cho cả image và mask.
    """

    def __init__(self, size: int = 256):
        """
        Args:
            size (int): Kích thước cạnh của vùng cắt (vuông). Mặc định 256.
        """
        self.size = size

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """
        Args:
            image: numpy array shape (H, W) hoặc (3, H, W).
            mask : numpy array shape (H, W) hoặc (1, H, W).
        Returns:
            (image_crop, mask_crop) kích thước (size, size) hoặc (C, size, size).
        Raises:
            ValueError: Nếu ảnh nhỏ hơn kích thước cắt yêu cầu.
        """
        # Lấy H, W từ hai chiều cuối cùng
        H, W = image.shape[-2], image.shape[-1]

        if H < self.size or W < self.size:
            raise ValueError(
                f"RandomCrop: Kích thước ảnh ({H}x{W}) nhỏ hơn "
                f"kích thước cắt ({self.size}x{self.size})."
            )

        # Chọn điểm góc trên-trái ngẫu nhiên
        top  = np.random.randint(0, H - self.size + 1)
        left = np.random.randint(0, W - self.size + 1)

        # Cắt trên hai chiều cuối cùng
        image_crop = image[..., top:top + self.size, left:left + self.size]
        mask_crop  = mask [..., top:top + self.size, left:left + self.size]

        return image_crop.copy(), mask_crop.copy()


# ---------------------------------------------------------------------------
# 5. CenterCrop
# ---------------------------------------------------------------------------

class CenterCrop:
    """
    Cắt chính giữa ảnh một vùng kích thước (size x size).
    Thường dùng khi validation / test để có kết quả ổn định.
    """

    def __init__(self, size: int = 256):
        """
        Args:
            size (int): Kích thước cạnh của vùng cắt (vuông). Mặc định 256.
        """
        self.size = size

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """
        Args:
            image: numpy array shape (H, W) hoặc (3, H, W).
            mask : numpy array shape (H, W) hoặc (1, H, W).
        Returns:
            (image_crop, mask_crop) kích thước (size, size) hoặc (C, size, size).
        Raises:
            ValueError: Nếu ảnh nhỏ hơn kích thước cắt yêu cầu.
        """
        H, W = image.shape[-2], image.shape[-1]

        if H < self.size or W < self.size:
            raise ValueError(
                f"CenterCrop: Kích thước ảnh ({H}x{W}) nhỏ hơn "
                f"kích thước cắt ({self.size}x{self.size})."
            )

        # Tính điểm góc trên-trái để cắt chính giữa
        top  = (H - self.size) // 2
        left = (W - self.size) // 2

        image_crop = image[..., top:top + self.size, left:left + self.size]
        mask_crop  = mask [..., top:top + self.size, left:left + self.size]

        return image_crop.copy(), mask_crop.copy()


# ---------------------------------------------------------------------------
# 6. GaussianNoise
# ---------------------------------------------------------------------------

class GaussianNoise:
    """
    Thêm nhiễu Gaussian vào ảnh với xác suất p.
    CHỈ áp dụng lên ảnh, KHÔNG thêm nhiễu vào mask (mask là nhị phân).
    """

    def __init__(self, sigma: float = 0.01, p: float = 0.2):
        """
        Args:
            sigma (float): Độ lệch chuẩn của phân phối Gaussian. Mặc định 0.01.
            p     (float): Xác suất thêm nhiễu. Mặc định 0.2.
        """
        self.sigma = sigma
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """
        Args:
            image: numpy array shape (H, W) hoặc (3, H, W).
            mask : numpy array shape (H, W) hoặc (1, H, W) – giữ nguyên.
        Returns:
            (image_noisy, mask) trong đó mask không thay đổi.
        """
        if np.random.random() < self.p:
            # Sinh nhiễu cùng kích thước với ảnh, mean=0, std=sigma
            noise = np.random.normal(loc=0.0, scale=self.sigma, size=image.shape)
            image = image + noise.astype(image.dtype)

        # mask giữ nguyên, không thêm nhiễu
        return image, mask


# ---------------------------------------------------------------------------
# 7. Compose
# ---------------------------------------------------------------------------

class Compose:
    """
    Kết hợp nhiều transform thành một pipeline.
    Mỗi transform sẽ được áp dụng tuần tự lên (image, mask).
    """

    def __init__(self, transforms: list):
        """
        Args:
            transforms (list): Danh sách các transform cần áp dụng theo thứ tự.
        """
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """
        Áp dụng tuần tự từng transform trong pipeline.

        Args:
            image: numpy array shape (H, W) hoặc (3, H, W).
            mask : numpy array shape (H, W) hoặc (1, H, W).
        Returns:
            (image, mask) sau khi đã qua toàn bộ pipeline.
        """
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

    def __repr__(self) -> str:
        lines = [self.__class__.__name__ + "("]
        for t in self.transforms:
            lines.append(f"    {t.__class__.__name__},")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Factory functions
# ---------------------------------------------------------------------------

def get_train_transforms(config: dict) -> Compose:
    """
    Tạo pipeline augmentation cho tập TRAIN từ dict cấu hình.

    Các key được đọc từ config (tất cả đều có giá trị mặc định):
        crop_size        (int)   : Kích thước crop. Mặc định 256.
        hflip_p          (float) : Xác suất lật ngang. Mặc định 0.5.
        vflip_p          (float) : Xác suất lật dọc. Mặc định 0.5.
        rotation_degrees (float) : Góc xoay tối đa. Mặc định 15.
        rotation_p       (float) : Xác suất xoay. Mặc định 0.3.
        noise_sigma      (float) : Độ lệch chuẩn nhiễu. Mặc định 0.01.
        noise_p          (float) : Xác suất thêm nhiễu. Mặc định 0.2.

    Args:
        config (dict): Từ điển cấu hình (thường đọc từ file YAML/JSON).

    Returns:
        Compose: Pipeline augmentation cho training.
    """
    # Lấy image_size từ config['DATA'] nếu có, mặc định 256
    data_cfg = config.get("DATA", {})
    crop_size        = data_cfg.get("image_size", 256)
    
    # Các tham số augmentation có thể thêm vào AUGMENTATION hoặc DATA
    aug_cfg          = config.get("AUGMENTATION", data_cfg)
    hflip_p          = aug_cfg.get("hflip_p",          0.5)
    vflip_p          = aug_cfg.get("vflip_p",          0.5)
    rotation_degrees = aug_cfg.get("rotation_degrees", 15)
    rotation_p       = aug_cfg.get("rotation_p",       0.3)
    noise_sigma      = aug_cfg.get("noise_sigma",      0.01)
    noise_p          = aug_cfg.get("noise_p",          0.2)

    return Compose([
        RandomHorizontalFlip(p=hflip_p),
        RandomVerticalFlip(p=vflip_p),
        RandomRotation(degrees=rotation_degrees, p=rotation_p),
        RandomCrop(size=crop_size),
        GaussianNoise(sigma=noise_sigma, p=noise_p),
    ])


def get_val_transforms(config: dict) -> Compose:
    """
    Tạo pipeline augmentation cho tập VALIDATION / TEST từ dict cấu hình.

    Với val/test chỉ dùng CenterCrop (không flip, xoay hay thêm nhiễu)
    để đảm bảo tính tái lập (reproducibility).

    Các key được đọc từ config:
        crop_size (int): Kích thước crop. Mặc định 256.

    Args:
        config (dict): Từ điển cấu hình.

    Returns:
        Compose: Pipeline cho validation / test.
    """
    data_cfg  = config.get("DATA", {})
    crop_size = data_cfg.get("image_size", 256)

    return Compose([
        CenterCrop(size=crop_size),
    ])
