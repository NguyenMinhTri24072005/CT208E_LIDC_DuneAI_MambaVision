# -*- coding: utf-8 -*-
"""
dune_unet.py - Kiến trúc mạng nơ-ron cho dự án DuneAI.
Hỗ trợ đầu vào 2.5D (3 kênh: lát z-1, z, z+1) và đầu ra 1 kênh (mặt nạ khối u).
Được thiết kế theo dạng Module để dễ dàng chuyển đổi qua lại giữa:
  - Baseline U-Net thuần CNN (`backbone="none"`, mặc định)
  - Hybrid MambaVision-UNet (`backbone="mambavision_t"`, `"mambavision_s"`, `"mambavision_mock"`, ...)

Dùng:
  python DuneAI_Remake/src/models/dune_unet.py selftest
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
for p in (str(THIS_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.custom_backbone import MambaVisionEncoder, is_mambavision_backbone


class DoubleConv(nn.Module):
    """Khối tích chập kép cơ bản của U-Net: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Khối thu nhỏ (Encoder): MaxPool2d để giảm kích thước ảnh một nửa, sau đó dùng DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Khối phóng to (Decoder) và kết nối chéo (Skip Connection)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Dùng ConvTranspose2d để phóng to ảnh lên gấp đôi
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # Phóng to x1 (từ lớp dưới lên)
        x1 = self.up(x1)

        # Xử lý trường hợp kích thước ảnh bị lẻ (padding để khớp với x2)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        # Kết nối chéo: Nối ma trận đặc trưng x2 (từ Encoder) với x1 (vừa phóng to)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Lớp xuất cuối cùng: Chuyển số lượng kênh đặc trưng về số lượng lớp nhãn (1 kênh)"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DuneUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        classes=1,
        features=64,
        backbone="none",
        pretrained=True,
        freeze_backbone=False,
        use_mock=False,
    ):
        """
        Khởi tạo mạng DuneUNet cho chuẩn 2.5D.
        - in_channels: 3 (Lát cắt Z-1, Z, Z+1).
        - classes: 1 (Dự đoán pixel có phải là khối u hay không).
        - features: Số lượng bộ lọc (filter) cơ sở.
        - backbone: "none" (mặc định: U-Net thuần CNN) hoặc tên MambaVision ("mambavision_t", "mambavision_mock", ...).
        - pretrained: Có nạp trọng số Pre-trained khi dùng MambaVision hay không.
        - freeze_backbone: Khởi tạo ở trạng thái đóng băng Backbone nếu True.
        - use_mock: Dùng _MockMambaVisionBackbone thuần PyTorch để kiểm thử trên CPU.
        """
        super(DuneUNet, self).__init__()
        self.in_channels = int(in_channels)
        self.classes = int(classes)
        self.features = int(features)
        self.has_backbone = is_mambavision_backbone(backbone)
        self.backbone_name = str(backbone).strip().lower() if self.has_backbone else "none"
        self.pretrained = bool(pretrained)

        if not self.has_backbone:
            # =================================================================
            # CHẾ ĐỘ 1 (MẶC ĐỊNH): BASELINE U-NET THUẦN CNN
            # Giữ nguyên 100% tên tầng để tương thích với mọi checkpoint cũ
            # =================================================================
            self.encoder = None
            self.inc = DoubleConv(in_channels, features)
            self.down1 = Down(features, features * 2)
            self.down2 = Down(features * 2, features * 4)
            self.down3 = Down(features * 4, features * 8)
            self.down4 = Down(features * 8, features * 16)

            self.up1 = Up(features * 16, features * 8)
            self.up2 = Up(features * 8, features * 4)
            self.up3 = Up(features * 4, features * 2)
            self.up4 = Up(features * 2, features)
            self.outc = OutConv(features, classes)
        else:
            # =================================================================
            # CHẾ ĐỘ 2: HYBRID MAMBAVISION BACKBONE + U-NET DECODER
            # =================================================================
            is_mock = bool(use_mock or self.backbone_name == "mambavision_mock")
            self.encoder = MambaVisionEncoder(
                model_name=self.backbone_name,
                in_channels=in_channels,
                pretrained=pretrained and not is_mock,
                freeze_backbone=False,
                use_mock=is_mock,
            )
            c1, c2, c3, c4 = self.encoder.out_channels  # VD MambaVision-T: [80, 160, 320, 640]

            # Nhánh Stem giữ chi tiết biên độ phân giải cao ở (H, W) và (H/2, W/2)
            self.inc = DoubleConv(in_channels, features)
            self.down1 = Down(features, features)

            # Tầng khớp kênh 1x1 đưa 4 tầng đặc trưng MambaVision (H/4..H/32) về [2f, 4f, 8f, 16f]
            self.proj1 = nn.Sequential(
                nn.Conv2d(c1, features * 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(features * 2),
                nn.ReLU(inplace=True),
            )
            self.proj2 = nn.Sequential(
                nn.Conv2d(c2, features * 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(features * 4),
                nn.ReLU(inplace=True),
            )
            self.proj3 = nn.Sequential(
                nn.Conv2d(c3, features * 8, kernel_size=1, bias=False),
                nn.BatchNorm2d(features * 8),
                nn.ReLU(inplace=True),
            )
            self.proj4 = nn.Sequential(
                nn.Conv2d(c4, features * 16, kernel_size=1, bias=False),
                nn.BatchNorm2d(features * 16),
                nn.ReLU(inplace=True),
            )

            # Nhánh Giải mã (Decoder) 5 bậc từ H/32 lên H
            self.up1 = Up(features * 16, features * 8)  # H/32 -> H/16
            self.up2 = Up(features * 8, features * 4)   # H/16 -> H/8
            self.up3 = Up(features * 4, features * 2)   # H/8  -> H/4
            self.up4 = Up(features * 2, features * 2)   # H/4  -> H/2 (nối với down1 có f kênh)
            self.up5 = Up(features * 2, features)       # H/2  -> H   (nối với inc có f kênh)
            self.outc = OutConv(features, classes)

        # Khởi tạo bias âm để mô hình bắt đầu với xác suất nền cao (u rất nhỏ so với ảnh)
        nn.init.constant_(self.outc.conv.bias, -4.0)

        if freeze_backbone and self.has_backbone:
            self.freeze_backbone()

    # =========================================================================
    # CƠ CHẾ ĐÓNG / MỞ KHÓA BACKBONE & PHÂN NHÓM TỐC ĐỘ HỌC (PARAM GROUPS)
    # =========================================================================
    @property
    def encoder_frozen(self) -> bool:
        """Trả về True nếu đang dùng MambaVision và nhánh encoder đang bị đóng băng."""
        return bool(self.has_backbone and self.encoder is not None and self.encoder.is_frozen)

    def freeze_backbone(self):
        """Đóng băng trọng số MambaVision (nếu đang bật backbone)."""
        if self.has_backbone and self.encoder is not None:
            self.encoder.freeze()
            for p in self.encoder.backbone.parameters():
                p.grad = None
        return self

    def unfreeze_backbone(self):
        """Mở khóa trọng số MambaVision để huấn luyện tinh chỉnh toàn mạng."""
        if self.has_backbone and self.encoder is not None:
            self.encoder.unfreeze()
        return self

    def set_backbone_frozen(self, frozen: bool):
        """Công tắc đóng/mở khóa Backbone: True = Đóng băng (Freeze), False = Mở khóa (Unfreeze)."""
        return self.freeze_backbone() if frozen else self.unfreeze_backbone()

    def param_groups(self, base_lr: float, backbone_lr_mult: float = 0.1):
        """
        Tạo danh sách nhóm tham số cho Optimizer:
          - Khi chạy Baseline (`has_backbone=False`): 1 nhóm duy nhất với `lr = base_lr, lr_mult = 1.0`.
          - Khi chạy MambaVision (`has_backbone=True`):
              + Nhóm 0 (Decoder, Stem, Projections, Input Adapter): `lr = base_lr, lr_mult = 1.0`
              + Nhóm 1 (Lõi MambaVision Backbone): `lr = base_lr * backbone_lr_mult, lr_mult = backbone_lr_mult`
        """
        base_lr = float(base_lr)
        if not self.has_backbone or self.encoder is None:
            return [{"params": list(self.parameters()), "lr": base_lr, "lr_mult": 1.0}]

        bb_params = list(self.encoder.backbone.parameters())
        bb_ids = {id(p) for p in bb_params}
        head_params = [p for p in self.parameters() if id(p) not in bb_ids]
        mult = float(backbone_lr_mult)
        return [
            {"params": head_params, "lr": base_lr, "lr_mult": 1.0},
            {"params": bb_params, "lr": base_lr * mult, "lr_mult": mult},
        ]

    def forward(self, x):
        if not self.has_backbone:
            # Đi xuống (Encoder thuần CNN)
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)

            # Đáy (Bottleneck)
            x5 = self.down4(x4)

            # Đi lên (Decoder + Skip Connections)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            return self.outc(x)
        else:
            # 1. Nhánh chi tiết biên độ phân giải cao (H và H/2)
            s0 = self.inc(x)
            s1 = self.down1(s0)

            # 2. Nhánh ngữ cảnh toàn cục MambaVision (H/4, H/8, H/16, H/32)
            f1, f2, f3, f4 = self.encoder(x)
            p1 = self.proj1(f1)
            p2 = self.proj2(f2)
            p3 = self.proj3(f3)
            p4 = self.proj4(f4)

            # 3. Đi lên (Decoder + Skip Connections đa tầng)
            d = self.up1(p4, p3)
            d = self.up2(d, p2)
            d = self.up3(d, p1)
            d = self.up4(d, s1)
            d = self.up5(d, s0)
            return self.outc(d)


def _cmd_selftest(_=None):
    print("=" * 60)
    print("🧠 KIỂM THỬ KIẾN TRÚC DUNE U-NET 2.5D (BASELINE & MAMBAVISION)")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 128, 128, device=device)

    # 1. Kiểm thử Baseline U-Net (backbone="none")
    m_base = DuneUNet(in_channels=3, classes=1, features=16, backbone="none").to(device)
    assert m_base.has_backbone is False and m_base.backbone_name == "none"
    with torch.no_grad():
        y_base = m_base(x)
    assert y_base.shape == (2, 1, 128, 128), f"Sai shape Baseline: {y_base.shape}"
    assert len(m_base.param_groups(1e-4)) == 1

    # 2. Kiểm thử MambaVision-UNet (backbone="mambavision_mock") + Đóng/Mở khóa
    m_mamba = DuneUNet(
        in_channels=3, classes=1, features=16, backbone="mambavision_mock", freeze_backbone=True
    ).to(device)
    assert m_mamba.has_backbone is True and m_mamba.encoder_frozen is True
    groups = m_mamba.param_groups(1e-3, backbone_lr_mult=0.1)
    assert len(groups) == 2 and groups[0]["lr_mult"] == 1.0 and groups[1]["lr_mult"] == 0.1

    m_mamba.train()
    y_mamba = m_mamba(x)
    assert y_mamba.shape == (2, 1, 128, 128), f"Sai shape MambaUNet: {y_mamba.shape}"
    y_mamba.mean().backward()
    assert all(p.grad is None for p in m_mamba.encoder.backbone.parameters()), "Backbone đang freeze không được nhận grad"
    assert any(p.grad is not None for p in m_mamba.up1.parameters()), "Decoder phải nhận grad"

    m_mamba.zero_grad(set_to_none=True)
    m_mamba.unfreeze_backbone()
    assert m_mamba.encoder_frozen is False
    m_mamba(x).mean().backward()
    assert any(p.grad is not None for p in m_mamba.encoder.backbone.parameters()), "Sau khi unfreeze, Backbone phải nhận grad"

    print("✅ selftest dune_unet.py: OK! (Cả chế độ Baseline và MambaVision đều hoạt động hoàn hảo)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=False)
    s = sub.add_parser("selftest", help="Kiểm thử tự động kiến trúc DuneUNet")
    s.set_defaults(fn=_cmd_selftest)
    args = ap.parse_args()
    _cmd_selftest(args)