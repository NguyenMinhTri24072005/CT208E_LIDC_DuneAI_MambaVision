"""
unet_model.py
-------------
Kiến trúc hoàn chỉnh của mạng U-Net chuẩn (Ronneberger et al., 2015).
Đầu ra trả về raw logits (chưa qua activation).
Hỗ trợ:
- Kaiming Normal initialization
- Gradient checkpointing tùy chọn để tiết kiệm VRAM
- Factory function build_model(cfg)
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """
    Kiến trúc U-Net chuẩn gồm 4 tầng Down và 4 tầng Up.

    Args:
        n_channels: Số kênh ảnh đầu vào (1 cho Grayscale, 3 cho RGB).
        n_classes: Số lớp phân đoạn (1 cho Binary segmentation, >1 cho Multi-class).
        bilinear: Sử dụng Bilinear Upsampling thay cho ConvTranspose2d nếu True.
        base_channels: Số lượng kênh cơ sở tại tầng đầu tiên (mặc định 64).
        use_checkpoint: Sử dụng gradient checkpointing để tiết kiệm VRAM nếu True.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        bilinear: bool = False,
        base_channels: int = 64,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channels = base_channels
        self.use_checkpoint = use_checkpoint

        factor = 2 if bilinear else 1

        # Encoder (Nhánh trích xuất đặc trưng)
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, (base_channels * 16) // factor)

        # Decoder (Nhánh khôi phục độ phân giải)
        self.up1 = Up(base_channels * 16, (base_channels * 8) // factor, bilinear)
        self.up2 = Up(base_channels * 8, (base_channels * 4) // factor, bilinear)
        self.up3 = Up(base_channels * 4, (base_channels * 2) // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        # Output head
        self.outc = OutConv(base_channels, n_classes)

        # Khởi tạo trọng số
        self.init_weights()

    def init_weights(self) -> None:
        """Khởi tạo trọng số mô hình theo phân phối Kaiming Normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của mạng U-Net.
        Args:
            x: Tensor đầu vào shape (B, n_channels, H, W)
        Returns:
            Logits thô shape (B, n_classes, H, W)
        """
        if self.use_checkpoint and x.requires_grad:
            x1 = checkpoint(self.inc, x, use_reentrant=False)
            x2 = checkpoint(self.down1, x1, use_reentrant=False)
            x3 = checkpoint(self.down2, x2, use_reentrant=False)
            x4 = checkpoint(self.down3, x3, use_reentrant=False)
            x5 = checkpoint(self.down4, x4, use_reentrant=False)

            x = checkpoint(self.up1, x5, x4, use_reentrant=False)
            x = checkpoint(self.up2, x, x3, use_reentrant=False)
            x = checkpoint(self.up3, x, x2, use_reentrant=False)
            x = checkpoint(self.up4, x, x1, use_reentrant=False)
            logits = self.outc(x)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)

        return logits


def build_model(cfg: Dict[str, Any]) -> UNet:
    """
    Factory function xây dựng mô hình UNet từ cấu hình dict.
    Args:
        cfg: Dict chứa thông tin cấu hình (nhóm 'model')
    Returns:
        Instance UNet
    """
    model_cfg = cfg.get("model", {})
    return UNet(
        n_channels=model_cfg.get("n_channels", 3),
        n_classes=model_cfg.get("n_classes", 1),
        bilinear=model_cfg.get("bilinear", False),
        base_channels=model_cfg.get("base_channels", 64),
        use_checkpoint=model_cfg.get("use_checkpoint", False),
    )


if __name__ == "__main__":
    # Test shape & xử lý kích thước không chia hết cho 16
    print("[UNet Test] Kiểm tra khởi tạo và kích thước shape...")
    test_model = UNet(n_channels=3, n_classes=1, bilinear=False, base_channels=32)

    # Test với kích thước thông thường 256x256
    dummy_input = torch.randn(2, 3, 256, 256)
    out = test_model(dummy_input)
    print(f"Input: {dummy_input.shape} -> Output: {out.shape}")
    assert out.shape == (2, 1, 256, 256), f"Lỗi shape: {out.shape}"

    # Test với kích thước không chia hết cho 16 (ví dụ: 251x239)
    dummy_odd = torch.randn(1, 3, 251, 239)
    out_odd = test_model(dummy_odd)
    print(f"Odd Input: {dummy_odd.shape} -> Output: {out_odd.shape}")
    assert out_odd.shape == (1, 1, 251, 239), f"Lỗi odd shape: {out_odd.shape}"
    print("[UNet Test] Đạt mọi tiêu chí shape!")
