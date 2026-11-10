# -*- coding: utf-8 -*-
"""
tests/test_independence.py - Kiểm tra tính độc lập cấu trúc, quy tắc import một chiều (R1, R2, R11).

Kiểm thử bằng AST và Runtime Isolation:
  - unet_base KHÔNG BAO GIỜ import mamba_unet (R1).
  - mamba_unet KHÔNG BAO GIỜ import unet_base (R1).
  - common KHÔNG BAO GIỜ import unet_base hoặc mamba_unet (R2).
  - common KHÔNG CHỨA bất kỳ lớp nào kế thừa nn.Module (R11).
  - KHÔNG TỒN TẠI activations/ hoặc losses/ ở ROOT, KHÔNG TỒN TẠI common/nn_blocks.py.
  - Chạy unet_base trong môi trường cấm tuyệt đối mamba_unet (sys.modules["mamba_unet"] = None).
  - Chạy mamba_unet trong môi trường cấm tuyệt đối unet_base (sys.modules["unet_base"] = None).
"""

import ast
import os
import sys
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class TestIndependence(unittest.TestCase):
    def test_root_cleanliness(self):
        """Đảm bảo không có thư mục cấm ở ROOT."""
        self.assertFalse((_ROOT / "activations").exists(), "Thư mục activations/ ở ROOT vi phạm R11!")
        self.assertFalse((_ROOT / "losses").exists(), "Thư mục losses/ ở ROOT vi phạm R11!")
        self.assertFalse((_ROOT / "common" / "nn_blocks.py").exists(), "common/nn_blocks.py vi phạm R11!")

    def test_common_has_no_nn_modules(self):
        """common/ phải có đúng 0 lớp kế thừa nn.Module (quét AST)."""
        common_dir = _ROOT / "common"
        violations = []
        for py_path in common_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        id_name = getattr(base, "id", None) or getattr(base, "attr", None)
                        if id_name in ("Module", "nn.Module"):
                            violations.append((py_path.name, node.name))

        self.assertEqual(violations, [], f"Phát hiện lớp kế thừa nn.Module trong common/: {violations}")

    def test_common_does_not_import_architectures(self):
        """common/ không bao giờ import unet_base hay mamba_unet."""
        common_dir = _ROOT / "common"
        forbidden = {"unet_base", "mamba_unet"}
        violations = []

        for py_path in common_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in forbidden:
                            violations.append((py_path.name, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        if top in forbidden:
                            violations.append((py_path.name, node.module))

        self.assertEqual(violations, [], f"common/ import vi phạm kiến trúc: {violations}")

    def test_unet_base_does_not_import_mamba_unet(self):
        """unet_base không bao giờ import mamba_unet (quét AST)."""
        unet_dir = _ROOT / "unet_base"
        violations = []

        for py_path in unet_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if "mamba_unet" in alias.name:
                            violations.append((py_path.name, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.module and "mamba_unet" in node.module:
                        violations.append((py_path.name, node.module))

        self.assertEqual(violations, [], f"unet_base import vi phạm mamba_unet: {violations}")

    def test_mamba_unet_does_not_import_unet_base(self):
        """mamba_unet không bao giờ import unet_base (quét AST)."""
        mamba_dir = _ROOT / "mamba_unet"
        violations = []

        for py_path in mamba_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if "unet_base" in alias.name:
                            violations.append((py_path.name, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.module and "unet_base" in node.module:
                        violations.append((py_path.name, node.module))

        self.assertEqual(violations, [], f"mamba_unet import vi phạm unet_base: {violations}")

    def test_runtime_isolation_unet_base(self):
        """Kiểm tra unet_base chạy độc lập hoàn hảo khi mamba_unet bị vô hiệu hóa hoàn toàn."""
        saved_mamba = sys.modules.get("mamba_unet", None)
        try:
            sys.modules["mamba_unet"] = None

            from unet_base.model import UNetBase
            model = UNetBase(features=8)
            x = torch.randn(1, 3, 32, 32)
            out = model(x)
            self.assertEqual(out.shape, (1, 1, 32, 32))
        finally:
            if saved_mamba is not None:
                sys.modules["mamba_unet"] = saved_mamba
            else:
                sys.modules.pop("mamba_unet", None)

    def test_runtime_isolation_mamba_unet(self):
        """Kiểm tra mamba_unet chạy độc lập hoàn hảo khi unet_base bị vô hiệu hóa hoàn toàn."""
        saved_unet = sys.modules.get("unet_base", None)
        try:
            sys.modules["unet_base"] = None

            from mamba_unet.backbone.mamba_vision import MambaVision
            from mamba_unet.model import MambaUNet
            tiny_bb = MambaVision(
                dim=16,
                in_dim=8,
                depths=[1, 1, 2, 1],
                window_size=[8, 8, 14, 7],
                num_heads=[2, 2, 4, 4],
                in_chans=3,
            )
            model = MambaUNet(in_channels=3, num_classes=1, backbone=tiny_bb, act="relu")
            x = torch.randn(1, 3, 64, 64)
            out = model(x)
            self.assertEqual(out.shape, (1, 1, 64, 64))
        finally:
            if saved_unet is not None:
                sys.modules["unet_base"] = saved_unet
            else:
                sys.modules.pop("unet_base", None)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    unittest.main()
