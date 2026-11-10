# -*- coding: utf-8 -*-
"""
tests/test_registries.py - Kiểm thử hệ thống Registry cho Activations và Losses.

Yêu cầu kiểm thử (spec mục 10, 4.1, 4.2):
  - Tra cứu hàm kích hoạt và hàm loss hợp lệ trả về module đúng.
  - Tra cứu tên không hợp lệ ném ValueError nêu rõ danh sách được hỗ trợ.
  - Thêm một entry giả lập vào registry unet_base -> chọn được qua tên.
  - Cách ly tuyệt đối: Đăng ký ở unet_base KHÔNG xuất hiện ở mamba_unet và ngược lại (R11).
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unet_base.activations import (
    ACTIVATIONS as UNET_ACTIVATIONS,
    ActivationSpec as UNetActivationSpec,
    get_activation as unet_get_activation,
    list_activations as unet_list_activations,
)
from unet_base.losses import (
    LOSSES as UNET_LOSSES,
    LossSpec as UNetLossSpec,
    get_loss as unet_get_loss,
    list_losses as unet_list_losses,
)

from mamba_unet.activations import (
    ACTIVATIONS as MAMBA_ACTIVATIONS,
    ActivationSpec as MambaActivationSpec,
    get_activation as mamba_get_activation,
    list_activations as mamba_list_activations,
)
from mamba_unet.losses import (
    LOSSES as MAMBA_LOSSES,
    LossSpec as MambaLossSpec,
    get_loss as mamba_get_loss,
    list_losses as mamba_list_losses,
)


class TestRegistriesUNetBase(unittest.TestCase):
    def test_activation_registry_lookup(self):
        """Kiểm tra tra cứu activation hợp lệ và không hợp lệ (unet_base)."""
        factory = unet_get_activation("relu")
        mod = factory()
        self.assertIsInstance(mod, nn.ReLU)

        with self.assertRaises(ValueError) as ctx:
            unet_get_activation("non_existent_activation_xyz")
        self.assertIn("Không hỗ trợ activation", str(ctx.exception))

    def test_custom_activation_registration(self):
        """Thêm một activation mới vào registry unet_base."""
        custom_key = "custom_test_activation_unet"
        try:
            UNET_ACTIVATIONS[custom_key] = UNetActivationSpec(
                factory=lambda **kw: nn.ELU(alpha=1.0),
                has_params=False,
                doc="Custom ELU test activation",
            )
            self.assertIn(custom_key, unet_list_activations())
            factory = unet_get_activation(custom_key)
            mod = factory()
            self.assertIsInstance(mod, nn.ELU)
        finally:
            UNET_ACTIVATIONS.pop(custom_key, None)

    def test_loss_registry_lookup(self):
        """Kiểm tra tra cứu loss hợp lệ và không hợp lệ (unet_base)."""
        loss_mod = unet_get_loss("dice_bce")
        self.assertTrue(isinstance(loss_mod, nn.Module))

        with self.assertRaises(ValueError) as ctx:
            unet_get_loss("non_existent_loss_xyz")
        self.assertIn("Không hỗ trợ loss", str(ctx.exception))


class TestRegistriesMambaUNet(unittest.TestCase):
    def test_activation_registry_lookup(self):
        """Kiểm tra tra cứu activation hợp lệ và không hợp lệ (mamba_unet)."""
        factory = mamba_get_activation("relu")
        mod = factory()
        self.assertIsInstance(mod, nn.ReLU)

        with self.assertRaises(ValueError) as ctx:
            mamba_get_activation("non_existent_activation_xyz")
        self.assertIn("Không hỗ trợ activation", str(ctx.exception))

    def test_loss_registry_lookup(self):
        """Kiểm tra tra cứu loss hợp lệ và không hợp lệ (mamba_unet)."""
        loss_mod = mamba_get_loss("dice_bce")
        self.assertTrue(isinstance(loss_mod, nn.Module))

        with self.assertRaises(ValueError) as ctx:
            mamba_get_loss("non_existent_loss_xyz")
        self.assertIn("Không hỗ trợ loss", str(ctx.exception))


class TestRegistryIsolation(unittest.TestCase):
    def test_activation_registries_are_completely_isolated(self):
        """Đăng ký activation vào unet_base không xuất hiện trong mamba_unet và ngược lại."""
        key_unet = "temp_isolation_act_unet"
        key_mamba = "temp_isolation_act_mamba"

        try:
            UNET_ACTIVATIONS[key_unet] = UNetActivationSpec(
                factory=lambda **kw: nn.ReLU(),
                has_params=False,
                doc="unet only",
            )
            MAMBA_ACTIVATIONS[key_mamba] = MambaActivationSpec(
                factory=lambda **kw: nn.GELU(),
                has_params=False,
                doc="mamba only",
            )

            # unet_base có key_unet nhưng KHÔNG có key_mamba
            self.assertIn(key_unet, unet_list_activations())
            self.assertNotIn(key_mamba, unet_list_activations())

            # mamba_unet có key_mamba nhưng KHÔNG có key_unet
            self.assertIn(key_mamba, mamba_list_activations())
            self.assertNotIn(key_unet, mamba_list_activations())

        finally:
            UNET_ACTIVATIONS.pop(key_unet, None)
            MAMBA_ACTIVATIONS.pop(key_mamba, None)

    def test_loss_registries_are_completely_isolated(self):
        """Đăng ký loss vào unet_base không xuất hiện trong mamba_unet và ngược lại."""
        key_unet = "temp_isolation_loss_unet"
        key_mamba = "temp_isolation_loss_mamba"

        try:
            UNET_LOSSES[key_unet] = UNetLossSpec(
                factory=lambda **kw: nn.L1Loss(),
                doc="unet only loss",
            )
            MAMBA_LOSSES[key_mamba] = MambaLossSpec(
                factory=lambda **kw: nn.MSELoss(),
                doc="mamba only loss",
            )

            self.assertIn(key_unet, unet_list_losses())
            self.assertNotIn(key_mamba, unet_list_losses())

            self.assertIn(key_mamba, mamba_list_losses())
            self.assertNotIn(key_unet, mamba_list_losses())

        finally:
            UNET_LOSSES.pop(key_unet, None)
            MAMBA_LOSSES.pop(key_mamba, None)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    unittest.main()
