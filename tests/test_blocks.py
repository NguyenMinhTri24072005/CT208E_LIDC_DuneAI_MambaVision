# -*- coding: utf-8 -*-
"""
tests/test_blocks.py - Kiểm thử các khối tích chập chuẩn DoubleConv và OutConv.

Yêu cầu kiểm thử (spec mục 10, 4.3):
  - Khối DoubleConv bảo tồn thứ tự chỉ số lớp con: 0: Conv, 1: BN, 2: Act, 3: Conv, 4: BN, 5: Act.
  - Đổi activation không tham số (relu, gelu, silu, ...) GIỮ NGUYÊN 100% tập khóa state_dict.
  - Đổi activation có tham số (star_relu_learnable) THÊM đúng các khóa tham số học được.
  - Kiểm tra riêng biệt cho unet_base và mamba_unet.
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unet_base.blocks import DoubleConv as UNetDoubleConv, OutConv as UNetOutConv
from mamba_unet.blocks import DoubleConv as MambaDoubleConv, OutConv as MambaOutConv


class TestBlocksUNetBase(unittest.TestCase):
    def test_double_conv_forward_and_indices(self):
        """Kiểm tra forward pass và chỉ số lớp con chuẩn xác của DoubleConv (unet_base)."""
        block = UNetDoubleConv(in_channels=16, out_channels=32, act="relu")
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 32, 32, 32))

        # Kiểm tra chỉ số lớp con
        seq = block.double_conv
        self.assertIsInstance(seq[0], nn.Conv2d)
        self.assertIsInstance(seq[1], nn.BatchNorm2d)
        self.assertIsInstance(seq[2], nn.ReLU)
        self.assertIsInstance(seq[3], nn.Conv2d)
        self.assertIsInstance(seq[4], nn.BatchNorm2d)
        self.assertIsInstance(seq[5], nn.ReLU)

    def test_parameterless_activations_preserve_state_dict_keys(self):
        """Đổi activation không tham số KHÔNG làm đổi tập khóa state_dict."""
        base_keys = set(UNetDoubleConv(16, 32, act="relu").state_dict().keys())

        paramless_acts = ["gelu", "silu", "mish", "soft_clamp_relu", "soft_clamp_silu", "identity"]
        for act_name in paramless_acts:
            b = UNetDoubleConv(16, 32, act=act_name)
            curr_keys = set(b.state_dict().keys())
            self.assertEqual(
                base_keys,
                curr_keys,
                f"Activation '{act_name}' làm lệch tập khóa state_dict! Khác biệt: {curr_keys ^ base_keys}",
            )

    def test_learnable_activation_adds_expected_keys(self):
        """star_relu phải thêm đúng 4 khóa tham số học được (2 scale, 2 bias)."""
        base_keys = set(UNetDoubleConv(16, 32, act="relu").state_dict().keys())
        learnable_block = UNetDoubleConv(16, 32, act="star_relu")
        learnable_keys = set(learnable_block.state_dict().keys())

        added_keys = learnable_keys - base_keys
        expected_added = {
            "double_conv.2.scale",
            "double_conv.2.bias",
            "double_conv.5.scale",
            "double_conv.5.bias",
        }
        self.assertEqual(added_keys, expected_added)

    def test_out_conv(self):
        """Kiểm tra OutConv forward và khởi tạo bias."""
        oc = UNetOutConv(in_channels=32, out_channels=1, init_bias=-4.0)
        x = torch.randn(2, 32, 16, 16)
        out = oc(x)
        self.assertEqual(out.shape, (2, 1, 16, 16))
        self.assertAlmostEqual(float(oc.conv.bias.data[0]), -4.0, places=5)


class TestBlocksMambaUNet(unittest.TestCase):
    def test_double_conv_forward_and_indices(self):
        """Kiểm tra forward pass và chỉ số lớp con chuẩn xác của DoubleConv (mamba_unet)."""
        block = MambaDoubleConv(in_channels=16, out_channels=32, act="relu")
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 32, 32, 32))

        seq = block.double_conv
        self.assertIsInstance(seq[0], nn.Conv2d)
        self.assertIsInstance(seq[1], nn.BatchNorm2d)
        self.assertIsInstance(seq[2], nn.ReLU)
        self.assertIsInstance(seq[3], nn.Conv2d)
        self.assertIsInstance(seq[4], nn.BatchNorm2d)
        self.assertIsInstance(seq[5], nn.ReLU)

    def test_parameterless_activations_preserve_state_dict_keys(self):
        """Đổi activation không tham số KHÔNG làm đổi tập khóa state_dict trong mamba_unet."""
        base_keys = set(MambaDoubleConv(16, 32, act="relu").state_dict().keys())

        paramless_acts = ["gelu", "silu", "mish", "soft_clamp_relu", "soft_clamp_silu", "identity"]
        for act_name in paramless_acts:
            b = MambaDoubleConv(16, 32, act=act_name)
            curr_keys = set(b.state_dict().keys())
            self.assertEqual(
                base_keys,
                curr_keys,
                f"Activation '{act_name}' làm lệch tập khóa state_dict! Khác biệt: {curr_keys ^ base_keys}",
            )

    def test_learnable_activation_adds_expected_keys(self):
        """star_relu trong mamba_unet phải thêm đúng 4 khóa tham số học được."""
        base_keys = set(MambaDoubleConv(16, 32, act="relu").state_dict().keys())
        learnable_block = MambaDoubleConv(16, 32, act="star_relu")
        learnable_keys = set(learnable_block.state_dict().keys())

        added_keys = learnable_keys - base_keys
        expected_added = {
            "double_conv.2.scale",
            "double_conv.2.bias",
            "double_conv.5.scale",
            "double_conv.5.bias",
        }
        self.assertEqual(added_keys, expected_added)

    def test_out_conv(self):
        """Kiểm tra OutConv forward và khởi tạo bias trong mamba_unet."""
        oc = MambaOutConv(in_channels=32, out_channels=1, init_bias=-4.0)
        x = torch.randn(2, 32, 16, 16)
        out = oc(x)
        self.assertEqual(out.shape, (2, 1, 16, 16))
        self.assertAlmostEqual(float(oc.conv.bias.data[0]), -4.0, places=5)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    unittest.main()
