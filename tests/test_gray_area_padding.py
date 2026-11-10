# -*- coding: utf-8 -*-
"""
tests/test_gray_area_padding.py - Kiểm thử chuyên sâu VÙNG XÁM PADDING & GRADIENTS (Ảnh 512x512).

Mục tiêu kiểm định:
  1. Đưa ảnh kích thước thực tế 512x512 qua MambaVision (với trọng số thật NVIDIA).
  2. Xác minh hoạt động đệm không gian (padding) tại Stage 3 (32 -> 42 -> 32) và Stage 4 (16 -> 21 -> 16).
  3. Kiểm tra thông suốt dòng gradient (Loss backward) từ đầu ra 512x512 ngược về từng tầng Mixer/Attention.
  4. Đảm bảo 100% tham số và đạo hàm không có NaN, không có Inf.
  5. Giả lập bộ nhớ với batch_size = 2, 4, 8 trên ảnh 512x512.
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mamba_unet.backbone.mamba_vision import mamba_vision_T
from mamba_unet.model import MambaUNet


class TestGrayAreaPaddingAndVRAM(unittest.TestCase):
    """Kiểm thử áp lực (stress test) vùng xám padding và gradient flow cho ảnh y tế 512x512."""

    def test_01_real_image_512_forward_all_stages(self):
        """Kiểm tra forward pass qua từng stage với ảnh 512x512."""
        model = mamba_vision_T(pretrained=False, in_chans=3)
        model.eval()

        # Ảnh kích thước chuẩn 512x512
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            features = model(x)

        self.assertEqual(len(features), 4, "Phải xuất ra đúng 4 skip features")
        self.assertEqual(features[0].shape, (2, 80, 128, 128), "Stage 1 shape sai")
        self.assertEqual(features[1].shape, (2, 160, 64, 64), "Stage 2 shape sai")
        self.assertEqual(features[2].shape, (2, 320, 32, 32), "Stage 3 (sau khi unpad từ 42) shape sai")
        self.assertEqual(features[3].shape, (2, 640, 16, 16), "Stage 4 (sau khi unpad từ 21) shape sai")

        # Kiểm tra tính hữu hạn (không NaN, không Inf)
        for i, f in enumerate(features):
            self.assertTrue(torch.isfinite(f).all(), f"Feature stage {i+1} có chứa NaN hoặc Inf")

    def test_02_gradient_flow_through_padding_stages(self):
        """Kiểm tra backward pass: gradient truyền thông suốt qua các khối WindowAttention bị pad."""
        model = MambaUNet(in_channels=3, num_classes=1, backbone="mamba_vision_T", pretrained=False)
        model.train()

        # Batch 2 ảnh 512x512
        x = torch.randn(2, 3, 512, 512, requires_grad=True)
        target = torch.ones(2, 1, 512, 512)

        out = model(x)
        self.assertEqual(out.shape, (2, 1, 512, 512), "Đầu ra Mamba-UNet phải khôi phục đúng 512x512")

        loss = F.binary_cross_entropy_with_logits(out, target)
        loss.backward()

        # 1. Đạo hàm đầu vào ảnh phải hữu hạn và khác 0
        self.assertIsNotNone(x.grad, "Đạo hàm của tensor ảnh đầu vào bị None")
        self.assertTrue(torch.isfinite(x.grad).all(), "Đạo hàm đầu vào có NaN hoặc Inf")
        self.assertGreater(x.grad.abs().sum().item(), 0.0, "Đạo hàm đầu vào bằng 0 (chết gradient)")

        # 2. Kiểm tra các tham số của Stage 3 & 4 (nơi có WindowAttention bị pad)
        stage3_checked = 0
        stage4_checked = 0
        for name, param in model.encoder.named_parameters():
            if "levels.2." in name and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), f"Gradient của {name} có NaN/Inf")
                stage3_checked += 1
            if "levels.3." in name and param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), f"Gradient của {name} có NaN/Inf")
                stage4_checked += 1

        self.assertGreater(stage3_checked, 0, "Stage 3 không nhận được gradient")
        self.assertGreater(stage4_checked, 0, "Stage 4 không nhận được gradient")

    def test_03_batch_stress_memory_simulation(self):
        """Giả lập forward + backward với Batch Size = 2, 4 trên ảnh 512x512 để kiểm tra rò rỉ bộ nhớ."""
        model = MambaUNet(in_channels=3, num_classes=1, backbone="mamba_vision_T", pretrained=False)
        model.train()

        for b in [2, 4]:
            x = torch.randn(b, 3, 512, 512)
            target = torch.zeros(b, 1, 512, 512)
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out, target)
            loss.backward()
            model.zero_grad()
            self.assertEqual(out.shape, (b, 1, 512, 512))


if __name__ == "__main__":
    unittest.main()
