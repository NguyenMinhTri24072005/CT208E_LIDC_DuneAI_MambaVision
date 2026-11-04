"""
mamba_unet.py
Class chính MambaUNet: Ghép MambaVision Encoder (từ third_party) với UNet Decoder (từ unet_blocks).

Kiến trúc:
    Input (B, 3, 256, 256)
        │
        ├── PatchEmbed ──────────────────────────── Skip C0 (B, 80, 64, 64)
        │       │
        ├── Stage 1 (ConvBlock ×1) ──────────────── Skip C1 (B, 80, 64, 64)
        │       │ Downsample
        ├── Stage 2 (ConvBlock ×3) ──────────────── Skip C2 (B, 160, 32, 32)
        │       │ Downsample
        ├── Stage 3 (MambaBlock+Attn ×8) ─────────── Skip C3 (B, 320, 16, 16)
        │       │ Downsample
        └── Stage 4 (MambaBlock+Attn ×4) ─────────── Bottleneck (B, 640, 8, 8)
                │
        DecoderBlock1 ← Skip C3  →  (B, 320, 16, 16)
                │
        DecoderBlock2 ← Skip C2  →  (B, 160, 32, 32)
                │
        DecoderBlock3 ← Skip C1  →  (B, 80, 64, 64)
                │
        DecoderBlock4 ← Skip C0  →  (B, 32, 128, 128)
                │
        SegmentationHead          →  (B, 1, 256, 256)

Ghi chú kỹ thuật quan trọng:
    - MambaVision gốc chỉ expose forward_features() chạy tuần tự qua 4 Stage,
      không trả về các đặc trưng trung gian. Ta phải tự lặp qua encoder.levels
      để lưu skip features SAU mỗi bộ blocks TRƯỚC khi Downsample.
    - Stage 3 & 4 dùng window attention → cần padding để kích thước chia hết
      cho window_size. Logic padding này đã nằm bên trong MambaVisionLayer.forward(),
      nên khi ta gọi level(x) trực tiếp thì nó tự xử lý đúng.
    - Để tránh sửa file third_party, ta gọi level.blocks tuần tự rồi gọi
      level.downsample riêng, thay vì gọi level(x) một lần.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Thêm đường dẫn tới third_party để import được MambaVision
_THIRD_PARTY = Path(__file__).parent.parent.parent / "third_party" / "MambaVision"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from mambavision.models.mamba_vision import MambaVision, mamba_vision_T, mamba_vision_T2, mamba_vision_S
from .unet_blocks import DecoderBlock, SegmentationHead


# =============================================================================
# Bảng tra cứu: tên backbone → hàm khởi tạo MambaVision tương ứng
# =============================================================================
_BACKBONE_REGISTRY = {
    "mamba_vision_T":  mamba_vision_T,
    "mamba_vision_T2": mamba_vision_T2,
    "mamba_vision_S":  mamba_vision_S,
}

# Số kênh đầu ra tại mỗi Stage theo backbone
# (dim × 2^i, với dim=80 cho T/T2/S)
_ENCODER_CHANNELS = {
    "mamba_vision_T":  [32, 80, 160, 320, 640],  # [C0(128x128), C1(64x64), C2(32x32), C3(16x16), C4(8x8)]
    "mamba_vision_T2": [32, 80, 160, 320, 640],
    "mamba_vision_S":  [64, 96, 192, 384, 768],
}


# =============================================================================
# MambaUNet — Kiến trúc chính
# =============================================================================

class MambaUNet(nn.Module):
    """
    Kiến trúc UNet với MambaVision làm Encoder và Decoder tích chập làm nhánh đi lên.

    Args:
        config (dict): Dict cấu hình đọc từ train_config.yaml. Các key sử dụng:
            - config["MODEL"]["backbone"]:          Tên biến thể MambaVision
            - config["MODEL"]["pretrained"]:        Tải trọng số ImageNet hay không
            - config["MODEL"]["num_classes"]:       Số kênh đầu ra (thường = 1)
            - config["MODEL"]["decoder_channels"]:  Danh sách số kênh Decoder [d1,d2,d3,d4]
            - config["MODEL"]["decoder_activation"]: Tên hàm kích hoạt (GELU/ReLU/SiLU)
    """

    def __init__(self, config: dict):
        super().__init__()

        # ---- Đọc cấu hình ----
        backbone_name = config["MODEL"]["backbone"]
        pretrained    = config["MODEL"].get("pretrained", False)
        num_classes   = config["MODEL"].get("num_classes", 1)
        dec_channels  = config["MODEL"].get("decoder_channels", [320, 160, 80, 32])
        activation    = config["MODEL"].get("decoder_activation", "GELU")

        if backbone_name not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Backbone '{backbone_name}' không được hỗ trợ. "
                f"Chọn một trong: {list(_BACKBONE_REGISTRY.keys())}"
            )

        enc_ch = _ENCODER_CHANNELS[backbone_name]
        # enc_ch = [C0, C1, C2, C3, C4]
        # C0 = PatchEmbed output, C1-C3 = Stage output trước Downsample, C4 = Bottleneck

        # ---- Khởi tạo Encoder (MambaVision) ----
        build_fn = _BACKBONE_REGISTRY[backbone_name]
        self.encoder: MambaVision = build_fn(pretrained=pretrained)

        # Xóa bỏ phần Classification Head gốc (không cần cho Segmentation)
        if hasattr(self.encoder, "head"):
            del self.encoder.head
        if hasattr(self.encoder, "avgpool"):
            del self.encoder.avgpool
        if hasattr(self.encoder, "norm"):
            del self.encoder.norm

        # ---- Khởi tạo Decoder ----
        # D1: Bottleneck(C4) + Skip C3 → dec_channels[0]
        self.decoder1 = DecoderBlock(
            in_channels   = enc_ch[4],
            skip_channels = enc_ch[3],
            out_channels  = dec_channels[0],
            activation    = activation,
        )
        # D2: D1_out + Skip C2 → dec_channels[1]
        self.decoder2 = DecoderBlock(
            in_channels   = dec_channels[0],
            skip_channels = enc_ch[2],
            out_channels  = dec_channels[1],
            activation    = activation,
        )
        # D3: D2_out + Skip C1 → dec_channels[2]
        self.decoder3 = DecoderBlock(
            in_channels   = dec_channels[1],
            skip_channels = enc_ch[1],
            out_channels  = dec_channels[2],
            activation    = activation,
        )
        # D4: D3_out + Skip C0 → dec_channels[3]
        self.decoder4 = DecoderBlock(
            in_channels   = dec_channels[2],
            skip_channels = enc_ch[0],
            out_channels  = dec_channels[3],
            activation    = activation,
        )

        # ---- Segmentation Head ----
        self.seg_head = SegmentationHead(
            in_channels = dec_channels[3],
            num_classes = num_classes,
        )
        
        # Chuẩn hóa theo thống kê ImageNet (cần khi dùng pretrained)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Bias head về "nền" để BCE không phải học từ xác suất 0.5
        nn.init.constant_(self.seg_head.head[1].bias, -4.0)

    # -------------------------------------------------------------------------
    def _encode(self, x: torch.Tensor):
        """
        Chạy qua toàn bộ Encoder và thu thập skip features.

        Thay vì gọi encoder.forward_features(x) một lần (bỏ qua skip),
        ta tự lặp qua encoder.levels để lưu đặc trưng trung gian.

        Returns:
            bottleneck: Tensor (B, C4, H4, W4) — đầu ra cuối của Encoder
            skips: List[Tensor] — [C0, C1, C2, C3] dùng cho Skip Connections
        """
        skips = []

        # Bước 0: PatchEmbed gồm 2 lần Conv stride 2:
        # - Lớp 0-2: Conv(3, in_dim, 3, 2, 1) + BN + ReLU -> (B, in_dim, 128, 128) [Skip C0]
        # - Lớp 3-5: Conv(in_dim, dim, 3, 2, 1) + BN + ReLU -> (B, dim, 64, 64)
        c0 = self.encoder.patch_embed.conv_down[:3](x)  # (B, 32, 128, 128)
        skips.append(c0)                                # Skip C0 (128×128)
        x = self.encoder.patch_embed.conv_down[3:](c0)  # (B, 80, 64, 64)

        # Bước 1-4: Lặp qua 4 Stage của MambaVision
        for level in self.encoder.levels:
            # --- Chạy qua các block trong Stage (CNN hoặc Mamba+Attn) ---
            if level.transformer_block:
                # Stage 3 & 4: Cần xử lý window partition
                _, _, H, W = x.shape
                pad_r = (level.window_size - W % level.window_size) % level.window_size
                pad_b = (level.window_size - H % level.window_size) % level.window_size
                if pad_r > 0 or pad_b > 0:
                    x = F.pad(x, (0, pad_r, 0, pad_b))
                    _, _, Hp, Wp = x.shape
                else:
                    Hp, Wp = H, W

                from mambavision.models.mamba_vision import MambaVision, mamba_vision_T, mamba_vision_T2, mamba_vision_S, window_partition, window_reverse
                x = window_partition(x, level.window_size)
                for blk in level.blocks:
                    x = blk(x)
                x = window_reverse(x, level.window_size, Hp, Wp)
                if pad_r > 0 or pad_b > 0:
                    x = x[:, :, :H, :W].contiguous()
            else:
                # Stage 1 & 2: Thuần CNN, không cần window partition
                for blk in level.blocks:
                    x = blk(x)

            # Chỉ lưu skip cho các Stage CÓ Downsample (Stage 1, 2, 3)
            # Stage 4 (cuối cùng) KHÔNG có downsample → output là bottleneck
            if level.downsample is not None:
                skips.append(x)            # C1, C2, C3
                x = level.downsample(x)    # Thu nhỏ để sang Stage tiếp theo

        # skips = [C0, C1, C2, C3]  (4 phần tử)
        # x hiện tại = Bottleneck (B, 640, 8, 8) — output Stage 4
        bottleneck = x
        return bottleneck, skips

    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Luồng dữ liệu từ đầu vào đến đầu ra mặt nạ.

        Args:
            x: Tensor (B, 3, H, W) — ảnh CT 2.5D, giá trị [0, 1]

        Returns:
            Tensor (B, 1, H, W) — Mặt nạ xác suất khối u, giá trị [0, 1]
        """
        # === ENCODER ===
        x = (x - self.img_mean) / self.img_std
        bottleneck, skips = self._encode(x)
        # skips[0] = C0 (B, 32, 128, 128) - từ lớp 1 của PatchEmbed
        # skips[1] = C1 (B, 80, 64, 64)   - sau Stage 1
        # skips[2] = C2 (B, 160, 32, 32)  - sau Stage 2
        # skips[3] = C3 (B, 320, 16, 16)  - sau Stage 3
        # bottleneck  = C4 (B, 640, 8, 8)  - đáy chữ U (Stage 4)

        # === DECODER ===
        d = self.decoder1(bottleneck, skips[3])  # (B,640,8,8)   + Skip C3(16x16)  → (B,320,16,16)
        d = self.decoder2(d, skips[2])           # (B,320,16,16) + Skip C2(32x32)  → (B,160,32,32)
        d = self.decoder3(d, skips[1])           # (B,160,32,32) + Skip C1(64x64)  → (B,80,64,64)
        d = self.decoder4(d, skips[0])           # (B,80,64,64)  + Skip C0(128x128)→ (B,32,128,128)

        # === SEGMENTATION HEAD ===
        out = self.seg_head(d)                   # (B,32,128,128) → (B,1,256,256)
        return out

    # -------------------------------------------------------------------------
    def freeze_encoder(self) -> None:
        """
        Đóng băng toàn bộ tham số của Encoder (dùng khi muốn chỉ train Decoder).
        Hữu ích khi fine-tuning với dataset nhỏ để tránh overfitting.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("[MambaUNet] Đã đóng băng Encoder. Chỉ train Decoder.")

    def unfreeze_encoder(self) -> None:
        """Mở băng toàn bộ Encoder để train cùng với Decoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("[MambaUNet] Đã mở băng Encoder. Train toàn bộ mạng.")


# =============================================================================
# Hàm factory — Tạo model từ config dict
# =============================================================================

def build_model(config: dict) -> MambaUNet:
    """
    Hàm tiện ích tạo MambaUNet từ config dict.

    Args:
        config: Dict cấu hình đọc từ train_config.yaml.

    Returns:
        Instance MambaUNet đã khởi tạo, chuyển lên device phù hợp.
    """
    model = MambaUNet(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    backbone = config["MODEL"]["backbone"]
    pretrained = config["MODEL"].get("pretrained", False)
    print(f"[MambaUNet] Khởi tạo thành công: backbone={backbone}, pretrained={pretrained}")
    print(f"[MambaUNet] Thiết bị: {device}")

    return model
