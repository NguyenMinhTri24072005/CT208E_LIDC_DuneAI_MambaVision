"""
unet_parts.py
-------------
Các khối cấu trúc (building blocks) cơ bản cho mạng U-Net:
- DoubleConv: (Conv3x3 -> BatchNorm2d -> ReLU) x 2
- Down: MaxPool2d(2) -> DoubleConv
- Up: Upsample/ConvTranspose2d -> Padding căn chỉnh shape -> Concat -> DoubleConv
- OutConv: Conv2d 1x1 ra số lượng classes
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Convolution => [BN] => ReLU) * 2 kèm Residual Shortcut giúp chống mất mát đặc trưng nốt u nhỏ.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return self.act(out)


class Down(nn.Module):
    """
    Downscaling với MaxPool2d(2) rồi đi qua DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Shape: (B, C_in, H, W) -> (B, C_out, H // 2, W // 2)"""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling sau đó nối với skip connection và đi qua DoubleConv.
    Hỗ trợ cả bilinear=True và ConvTranspose2d.
    Tự động căn chỉnh kích thước bằng F.pad khi H/W lẻ.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Tensor từ tầng dưới cần upsample (B, C1, H1, W1)
            x2: Tensor skip connection từ Encoder (B, C2, H2, W2)
        Returns:
            Tensor sau khi concat và conv (B, out_channels, H2, W2)
        """
        x1 = self.up(x1)

        # Căn chỉnh kích thước nếu có sự chênh lệch do kích thước lẻ (H, W không chia hết cho 2)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        # F.pad nhận định dạng: [pad_left, pad_right, pad_top, pad_bottom]
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        # Ghép đặc trưng theo chiều kênh (dim=1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Lớp tích chập 1x1 ánh xạ kênh đặc trưng về số lượng lớp dự đoán (n_classes).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Shape: (B, C_in, H, W) -> (B, n_classes, H, W)"""
        return self.conv(x)
