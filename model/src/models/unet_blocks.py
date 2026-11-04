"""
unet_blocks.py
--------------
Các khối xây dựng (building blocks) cho nhánh Decoder của mạng UNet
kết hợp với MambaVision Encoder.

Kiến trúc Decoder tổng quan:
    D1: (B, 640,  8,  8) + Skip C3 (320) → (B, 320, 16, 16)
    D2: (B, 320, 16, 16) + Skip C2 (160) → (B, 160, 32, 32)
    D3: (B, 160, 32, 32) + Skip C1  (80) → (B,  80, 64, 64)
    D4: (B,  80, 64, 64) + Skip C0  (80) → (B,  32,128,128)
    Head: (B, 32, 128, 128)             → (B,   1,256,256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------

def _get_activation(name: str) -> nn.Module:
    """
    Trả về module activation tương ứng với tên.

    Args:
        name (str): Tên activation — 'GELU', 'ReLU', hoặc 'SiLU'.

    Returns:
        nn.Module: Module activation.

    Raises:
        ValueError: Nếu tên activation không được hỗ trợ.
    """
    name = name.upper()
    activations = {
        "GELU": nn.GELU(),
        "RELU": nn.ReLU(inplace=True),
        "SILU": nn.SiLU(inplace=True),
    }
    if name not in activations:
        raise ValueError(
            f"Activation '{name}' không được hỗ trợ. "
            f"Chọn một trong: {list(activations.keys())}"
        )
    return activations[name]


# ---------------------------------------------------------------------------
# ConvBnAct — Khối Conv + BatchNorm + Activation
# ---------------------------------------------------------------------------

class ConvBnAct(nn.Module):
    """
    Khối tích chập cơ bản: Conv2d → BatchNorm2d → Activation.

    Được dùng làm đơn vị xây dựng trong các DecoderBlock để
    trích xuất đặc trưng sau khi ghép nối (concatenation).

    Args:
        in_channels  (int): Số kênh đầu vào.
        out_channels (int): Số kênh đầu ra.
        activation   (str): Tên hàm kích hoạt ('GELU' | 'ReLU' | 'SiLU').
                            Mặc định là 'GELU'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "GELU",
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            # Conv 3×3, không dùng bias vì có BatchNorm ngay sau
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            _get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Feature map đầu vào, shape (B, C_in, H, W).

        Returns:
            Tensor: Feature map đầu ra, shape (B, C_out, H, W).
        """
        return self.block(x)


# ---------------------------------------------------------------------------
# DecoderBlock — Khối giải mã UNet
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Khối giải mã (Decoder Block) cho UNet.

    Luồng xử lý:
        1. Upsample x lên gấp đôi bằng ConvTranspose2d.
        2. Căn chỉnh kích thước không gian với skip connection
           (xử lý lệch 1 pixel nếu có).
        3. Concatenate [x_up, skip] theo chiều kênh.
        4. Hai lớp ConvBnAct để tinh chỉnh đặc trưng.

    Args:
        in_channels   (int): Số kênh của feature map x (đầu vào từ tầng sâu hơn).
        skip_channels (int): Số kênh của skip connection từ encoder.
        out_channels  (int): Số kênh đầu ra sau khi giải mã.
        activation    (str): Tên hàm kích hoạt. Mặc định là 'GELU'.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        activation: str = "GELU",
    ) -> None:
        super().__init__()

        # ConvTranspose2d: upsample 2× (stride=2, kernel=2)
        # Giữ nguyên số kênh — việc giảm kênh sẽ do ConvBnAct đảm nhiệm
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
        )

        # Sau khi concat: số kênh = in_channels + skip_channels
        concat_channels = in_channels + skip_channels

        # Hai lớp ConvBnAct để học đặc trưng sau khi ghép nối
        self.conv1 = ConvBnAct(concat_channels, out_channels, activation)
        self.conv2 = ConvBnAct(out_channels, out_channels, activation)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    (Tensor): Feature map từ tầng sâu hơn, shape (B, C_in, H, W).
            skip (Tensor): Skip connection từ encoder,
                           shape (B, C_skip, H*2, W*2).

        Returns:
            Tensor: Feature map đã giải mã, shape (B, C_out, H*2, W*2).
        """
        # Bước 1: Upsample x lên gấp đôi
        x_up = self.upsample(x)

        # Bước 2: Căn chỉnh kích thước không gian
        # ConvTranspose2d đôi khi tạo ra kích thước lệch 1 pixel
        # so với skip connection → dùng F.interpolate để align chính xác
        if x_up.shape[-2:] != skip.shape[-2:]:
            x_up = F.interpolate(
                x_up,
                size=skip.shape[-2:],   # (H_skip, W_skip)
                mode="bilinear",
                align_corners=False,
            )

        # Bước 3: Ghép nối theo chiều kênh
        x_cat = torch.cat([x_up, skip], dim=1)  # (B, C_in + C_skip, H, W)

        # Bước 4: Hai lớp ConvBnAct
        out = self.conv1(x_cat)
        out = self.conv2(out)

        return out


# ---------------------------------------------------------------------------
# SegmentationHead — Đầu ra phân đoạn
# ---------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """
    Đầu ra phân đoạn (Segmentation Head) cuối cùng của UNet.

    Luồng xử lý:
        1. Upsample 2× bằng nn.Upsample (bilinear).
        2. Conv2d 1×1 để chiếu xuống `num_classes` kênh.
        3. Sigmoid để đưa đầu ra về khoảng [0, 1].

    Ví dụ:
        Input:  (B, 32, 128, 128)
        Output: (B,  1, 256, 256)  — với num_classes=1 (binary segmentation)

    Args:
        in_channels (int): Số kênh đầu vào.
        num_classes (int): Số lớp phân đoạn. Mặc định là 1 (binary).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            # Upsample 2× — bilinear, không học tham số
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # Conv 1×1: chiếu kênh xuống num_classes
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            # Sigmoid: đưa xác suất về [0, 1]
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Feature map đầu vào, shape (B, C_in, H, W).

        Returns:
            Tensor: Bản đồ xác suất phân đoạn,
                    shape (B, num_classes, H*2, W*2).
        """
        return self.head(x)


# ---------------------------------------------------------------------------
# Smoke test nhanh (chỉ chạy khi gọi trực tiếp file này)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Smoke test: unet_blocks.py")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- ConvBnAct ---
    cba = ConvBnAct(64, 32, activation="GELU").to(device)
    t_in = torch.randn(2, 64, 32, 32, device=device)
    t_out = cba(t_in)
    assert t_out.shape == (2, 32, 32, 32), f"ConvBnAct FAILED: {t_out.shape}"
    print(f"[OK] ConvBnAct  : {tuple(t_in.shape)} → {tuple(t_out.shape)}")

    # --- DecoderBlock (D1) ---
    # Bottleneck: (2, 640, 8, 8) + Skip C3: (2, 320, 16, 16) → (2, 320, 16, 16)
    db1 = DecoderBlock(640, 320, 320, activation="GELU").to(device)
    x_bot = torch.randn(2, 640, 8, 8, device=device)
    s3    = torch.randn(2, 320, 16, 16, device=device)
    d1    = db1(x_bot, s3)
    assert d1.shape == (2, 320, 16, 16), f"D1 FAILED: {d1.shape}"
    print(f"[OK] DecoderBlock D1: {tuple(x_bot.shape)} + {tuple(s3.shape)} → {tuple(d1.shape)}")

    # --- DecoderBlock (D2) ---
    db2 = DecoderBlock(320, 160, 160, activation="GELU").to(device)
    s2  = torch.randn(2, 160, 32, 32, device=device)
    d2  = db2(d1, s2)
    assert d2.shape == (2, 160, 32, 32), f"D2 FAILED: {d2.shape}"
    print(f"[OK] DecoderBlock D2: {tuple(d1.shape)} + {tuple(s2.shape)} → {tuple(d2.shape)}")

    # --- DecoderBlock (D3) ---
    db3 = DecoderBlock(160, 80, 80, activation="GELU").to(device)
    s1  = torch.randn(2, 80, 64, 64, device=device)
    d3  = db3(d2, s1)
    assert d3.shape == (2, 80, 64, 64), f"D3 FAILED: {d3.shape}"
    print(f"[OK] DecoderBlock D3: {tuple(d2.shape)} + {tuple(s1.shape)} → {tuple(d3.shape)}")

    # --- DecoderBlock (D4) ---
    db4 = DecoderBlock(80, 80, 32, activation="GELU").to(device)
    s0  = torch.randn(2, 80, 128, 128, device=device)
    d4  = db4(d3, s0)
    assert d4.shape == (2, 32, 128, 128), f"D4 FAILED: {d4.shape}"
    print(f"[OK] DecoderBlock D4: {tuple(d3.shape)} + {tuple(s0.shape)} → {tuple(d4.shape)}")

    # --- SegmentationHead ---
    head  = SegmentationHead(32, num_classes=1).to(device)
    seg   = head(d4)
    assert seg.shape == (2, 1, 256, 256), f"SegHead FAILED: {seg.shape}"
    assert seg.min() >= 0.0 and seg.max() <= 1.0, "SegHead: giá trị ngoài [0,1]!"
    print(f"[OK] SegmentationHead: {tuple(d4.shape)} → {tuple(seg.shape)}")

    print("\nTất cả smoke test PASSED ✓")
    sys.exit(0)
