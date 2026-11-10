# -*- coding: utf-8 -*-
"""
tests/test_state_dict_invariance.py - Kiểm tra bất biến 100% tập khóa và tensor đầu ra so với golden file.

Quy tắc bất biến (R3, spec mục 9):
  - unet_base: 118 keys và shapes trùng 100% tests/golden/unet_base_keys.json.
  - unet_base snapshot: nạp tiny_state strict=True, sai khác tensor đầu ra atol <= 1e-6.
  - mamba_unet: 264 keys và shapes trùng 100% tests/golden/mamba_unet_keys.json.
  - mamba_unet snapshot: nạp tiny_state strict=True, sai khác tensor đầu ra atol <= 1e-6.
  - Ánh xạ tiền tố encoder./decoder. cho checkpoint legacy prefixless khớp 100%.
"""

import json
import sys
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unet_base.model import UNetBase
from mamba_unet.backbone.mamba_vision import MambaVision
from mamba_unet.model import MambaUNet


class TestStateDictInvarianceUNetBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden_dir = _ROOT / "tests" / "golden"
        with open(cls.golden_dir / "unet_base_keys.json", "r", encoding="utf-8") as f:
            cls.expected_keys_shapes = json.load(f)

    def test_unet_base_keys_and_shapes_match_golden(self):
        """Kiểm tra 118 keys và shapes của UNetBase(features=64) trùng khớp 100% golden."""
        model = UNetBase(in_channels=3, classes=1, features=64)
        sd = model.state_dict()

        self.assertEqual(len(sd), len(self.expected_keys_shapes), f"Số lượng keys khác nhau! {len(sd)} != {len(self.expected_keys_shapes)}")

        for key, expected_shape in self.expected_keys_shapes.items():
            self.assertIn(key, sd, f"Thiếu key '{key}' trong UNetBase!")
            self.assertEqual(
                list(sd[key].shape),
                expected_shape,
                f"Shape của key '{key}' không khớp! {list(sd[key].shape)} != {expected_shape}",
            )

    def test_unet_base_tiny_snapshot_numerical_match(self):
        """Nạp tiny state (features=8) strict=True và kiểm tra đầu ra khớp golden tensor."""
        tiny_state_path = self.golden_dir / "unet_base_tiny_state.pth"
        tiny_out_path = self.golden_dir / "unet_base_tiny_out.pt"

        self.assertTrue(tiny_state_path.exists(), f"Không tìm thấy {tiny_state_path}")
        self.assertTrue(tiny_out_path.exists(), f"Không tìm thấy {tiny_out_path}")

        model = UNetBase(in_channels=3, classes=1, features=8)
        state_dict = torch.load(tiny_state_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        torch.manual_seed(123)
        dummy_in = torch.randn(2, 3, 128, 128)

        with torch.no_grad():
            out = model(dummy_in)

        golden_out = torch.load(tiny_out_path, map_location="cpu")
        torch.testing.assert_close(
            out,
            golden_out,
            rtol=1e-5,
            atol=1e-6,
            msg="Tensor đầu ra của UNetBase không khớp chính xác với unet_base_tiny_out.pt!",
        )

    def test_legacy_prefixless_mapping(self):
        """Kiểm tra bộ chuyển đổi tiền tố cho checkpoint cũ không có encoder./decoder."""
        legacy_keys_path = self.golden_dir / "legacy_prefixless_unet_keys.json"
        if not legacy_keys_path.exists():
            self.skipTest("Không có legacy_prefixless_unet_keys.json")

        with open(legacy_keys_path, "r", encoding="utf-8") as f:
            legacy_keys = json.load(f)

        def convert_legacy_key(k: str) -> str:
            if k.startswith(("inc.", "down1.", "down2.", "down3.", "down4.")):
                return f"encoder.{k}"
            elif k.startswith(("up1.", "up2.", "up3.", "up4.", "outc.")):
                return f"decoder.{k}"
            return k

        converted_keys = {convert_legacy_key(k): shape for k, shape in legacy_keys.items()}
        self.assertEqual(set(converted_keys.keys()), set(self.expected_keys_shapes.keys()))


class TestStateDictInvarianceMambaUNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden_dir = _ROOT / "tests" / "golden"
        with open(cls.golden_dir / "mamba_unet_keys.json", "r", encoding="utf-8") as f:
            cls.expected_keys_shapes = json.load(f)

    def test_mamba_unet_keys_and_shapes_match_golden(self):
        """Kiểm tra 264 keys và shapes của MambaUNet(backbone='mamba_vision_T') trùng khớp 100% golden."""
        model = MambaUNet(in_channels=3, num_classes=1, backbone="mamba_vision_T", pretrained=False, act="relu")
        sd = model.state_dict()

        self.assertEqual(len(sd), len(self.expected_keys_shapes), f"Số lượng keys khác nhau! {len(sd)} != {len(self.expected_keys_shapes)}")

        for key, expected_shape in self.expected_keys_shapes.items():
            self.assertIn(key, sd, f"Thiếu key '{key}' trong MambaUNet!")
            self.assertEqual(
                list(sd[key].shape),
                expected_shape,
                f"Shape của key '{key}' không khớp! {list(sd[key].shape)} != {expected_shape}",
            )

    def test_mamba_unet_tiny_snapshot_numerical_match(self):
        """Nạp tiny state strict=True và kiểm tra đầu ra khớp golden tensor atol <= 1e-6."""
        tiny_state_path = self.golden_dir / "mamba_unet_tiny_state.pth"
        tiny_out_path = self.golden_dir / "mamba_unet_tiny_out.pt"

        self.assertTrue(tiny_state_path.exists(), f"Không tìm thấy {tiny_state_path}")
        self.assertTrue(tiny_out_path.exists(), f"Không tìm thấy {tiny_out_path}")

        tiny_bb = MambaVision(
            dim=16,
            in_dim=8,
            depths=[1, 1, 2, 1],
            window_size=[8, 8, 14, 7],
            num_heads=[2, 2, 4, 4],
            in_chans=3,
        )
        model = MambaUNet(in_channels=3, num_classes=1, backbone=tiny_bb, act="relu")
        state_dict = torch.load(tiny_state_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        torch.manual_seed(123)
        dummy_in = torch.randn(2, 3, 128, 128)

        with torch.no_grad():
            out = model(dummy_in)

        golden_out = torch.load(tiny_out_path, map_location="cpu")
        torch.testing.assert_close(
            out,
            golden_out,
            rtol=1e-5,
            atol=1e-6,
            msg="Tensor đầu ra của MambaUNet không khớp chính xác với mamba_unet_tiny_out.pt!",
        )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    unittest.main()
