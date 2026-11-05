# -*- coding: utf-8 -*-
"""
dune_unet.py - Kiến trúc mạng nơ-ron cho dự án DuneAI.
Hỗ trợ đầu vào 2.5D (3 kênh: lát z-1, z, z+1) và đầu ra 1 kênh (mặt nạ khối u).
Được thiết kế theo dạng Module để dễ dàng cắm rút và nâng cấp lên Mamba-UNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Khối thu nhỏ (Encoder): MaxPool2d để giảm kích thước ảnh một nửa, sau đó dùng DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
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
    def __init__(self, in_channels=3, classes=1, features=64):
        """
        Khởi tạo mạng DuneUNet cho chuẩn 2.5D.
        - in_channels: 3 (Lát cắt Z-1, Z, Z+1).
        - classes: 1 (Dự đoán pixel có phải là khối u hay không).
        - features: Số lượng bộ lọc (filter) cơ sở. Càng lớn mạng càng thông minh nhưng ngốn VRAM.
        """
        super(DuneUNet, self).__init__()
        self.in_channels = in_channels
        self.classes = classes

        # 1. ENCODER (Khối chiết xuất đặc trưng - Dần dần thu nhỏ ảnh, tăng số kênh)
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        
        # 2. BOTTLENECK (Đáy chữ U - Điểm sâu nhất của mạng)
        # NƠI CẤY GHÉP MAMBA: Sau này bạn có thể thay thế DoubleConv ở đây bằng các khối MambaVision 
        # để tăng khả năng nắm bắt bối cảnh toàn cục (Global Context).
        self.down4 = Down(features * 8, features * 16)
        
        # 3. DECODER (Khối giải mã - Dần dần phóng to ảnh, giảm số kênh)
        self.up1 = Up(features * 16, features * 8)
        self.up2 = Up(features * 8, features * 4)
        self.up3 = Up(features * 4, features * 2)
        self.up4 = Up(features * 2, features)
        
        # Lớp đầu ra (Logits)
        self.outc = OutConv(features, classes)
        
        # Khởi tạo bias âm để mô hình bắt đầu với xác suất nền cao (u rất nhỏ so với ảnh)
        nn.init.constant_(self.outc.conv.bias, -4.0)

    def forward(self, x):
        # Đi xuống (Encoder)
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
        
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    import time
    
    print("="*50)
    print("🧠 KIỂM THỬ KIẾN TRÚC DUNE U-NET 2.5D")
    print("="*50)
    
    # 1. Khởi tạo mô hình
    model = DuneUNet(in_channels=3, classes=1)
    
    # Kích hoạt GPU nếu có (hoặc dùng CPU nếu PyTorch đang cài chưa có CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Mô hình đang chạy trên thiết bị: {device}")
    
    # Đếm số lượng tham số học được (Parameters)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số của mô hình: {total_params:,}")
    
    # 2. Tạo một Tensor giả lập Batch đầu vào (Kích thước giống hệt DataLoader)
    # Batch size = 4, Channels = 3, H=512, W=512
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 512, 512).to(device)
    print(f"Kích thước Tensor đầu vào  : {dummy_input.shape}")
    
    # 3. Chạy qua mạng nơ-ron (Forward pass)
    start_time = time.time()
    with torch.no_grad(): # Tắt tính đạo hàm vì chỉ đang test tốc độ forward
        output = model(dummy_input)
    end_time = time.time()
    
    print(f"Kích thước Tensor đầu ra   : {output.shape}")
    print(f"Thời gian xử lý 1 batch    : {(end_time - start_time)*1000:.2f} ms")
    
    # 4. Nghiệm thu
    assert output.shape == (batch_size, 1, 512, 512), "LỖI: Kích thước đầu ra không đúng chuẩn!"
    print("✅ KIỂM THỬ THÀNH CÔNG: Kiến trúc mạng hoàn toàn hợp lệ.")