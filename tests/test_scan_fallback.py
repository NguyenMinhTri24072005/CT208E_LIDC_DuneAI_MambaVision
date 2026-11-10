# -*- coding: utf-8 -*-
"""
tests/test_scan_fallback.py - Kiểm thử thuật toán Selective Scan Fallback PyTorch (R10).

Quy tắc nghiệm thu:
  1. Sai khác số học giữa selective_scan_torch và tham chiếu float64 <= 1e-4.
  2. Lan truyền ngược (backward pass) sinh ra gradients hữu hạn cho toàn bộ biến đầu vào.
  3. Cơ chế chọn backend qua biến môi trường DUNEAI_MAMBA_BACKEND hoạt động đúng đặc tả.
"""

import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mamba_unet.fallbacks import get_scan_backend, selective_scan_fn, selective_scan_torch


def _float64_reference_scan(u, delta, A, B, C, D=None):
    """
    Hàm tham chiếu chuẩn giải tích bằng số thực kép (float64).
    Công thức hồi quy Mamba SSM:
        h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * u_t
        y_t = h_t * C_t + D * u_t
    """
    u_f64 = u.detach().to(torch.float64)
    delta_f64 = delta.detach().to(torch.float64)
    A_f64 = A.detach().to(torch.float64)
    B_f64 = B.detach().to(torch.float64)
    C_f64 = C.detach().to(torch.float64)
    D_f64 = D.detach().to(torch.float64) if D is not None else None

    b, d, l = u_f64.shape
    n = A_f64.shape[1]

    h = torch.zeros(b, d, n, dtype=torch.float64, device=u_f64.device)
    ys = []

    for t in range(l):
        dt = delta_f64[:, :, t]
        u_t = u_f64[:, :, t]
        b_t = B_f64[:, :, t]
        c_t = C_f64[:, :, t]

        dA = torch.exp(dt.unsqueeze(-1) * A_f64.unsqueeze(0))
        dB = dt.unsqueeze(-1) * b_t.unsqueeze(1)

        h = dA * h + dB * u_t.unsqueeze(-1)
        y_t = (h * c_t.unsqueeze(1)).sum(dim=-1)
        if D_f64 is not None:
            y_t = y_t + D_f64.unsqueeze(0) * u_t
        ys.append(y_t)

    return torch.stack(ys, dim=-1)


class TestScanFallback(unittest.TestCase):
    """Kiểm tra độ chính xác và tính ổn định của fallback Selective Scan."""

    def setUp(self):
        torch.manual_seed(42)
        self.B = 2
        self.D = 8
        self.L = 16
        self.N = 4

    def test_numerical_accuracy_vs_float64(self):
        """Kiểm tra sai khác số học giữa hàm torch và tham chiếu float64 <= 1e-4."""
        u = torch.randn(self.B, self.D, self.L)
        delta = F.softplus(torch.randn(self.B, self.D, self.L))
        A = -torch.rand(self.D, self.N) - 0.1
        B = torch.randn(self.B, self.N, self.L)
        C = torch.randn(self.B, self.N, self.L)
        D = torch.randn(self.D)

        out_torch = selective_scan_torch(u, delta, A, B, C, D)
        out_ref = _float64_reference_scan(u, delta, A, B, C, D).to(torch.float32)

        max_diff = torch.max(torch.abs(out_torch - out_ref)).item()
        self.assertLessEqual(
            max_diff,
            1e-4,
            f"Sai lệch selective_scan_torch ({max_diff:.2e}) vượt ngưỡng cho phép 1e-4!",
        )

    def test_backward_pass_gradients_finite(self):
        """Kiểm tra lan truyền ngược: toàn bộ biến đầu vào đều có gradient hữu hạn."""
        u = torch.randn(self.B, self.D, self.L, requires_grad=True)
        delta = F.softplus(torch.randn(self.B, self.D, self.L)).requires_grad_(True)
        A = (-torch.rand(self.D, self.N) - 0.1).requires_grad_(True)
        B = torch.randn(self.B, self.N, self.L, requires_grad=True)
        C = torch.randn(self.B, self.N, self.L, requires_grad=True)
        D = torch.randn(self.D, requires_grad=True)

        out = selective_scan_torch(u, delta, A, B, C, D)
        loss = out.sum()
        loss.backward()

        for name, tensor in [("u", u), ("delta", delta), ("A", A), ("B", B), ("C", C), ("D", D)]:
            self.assertIsNotNone(tensor.grad, f"Gradient của {name} bị None!")
            self.assertTrue(
                torch.isfinite(tensor.grad).all(),
                f"Gradient của {name} chứa NaN hoặc Inf!",
            )

    def test_backend_selection_env(self):
        """Kiểm tra cơ chế lựa chọn backend qua biến môi trường DUNEAI_MAMBA_BACKEND."""
        old_val = os.environ.get("DUNEAI_MAMBA_BACKEND")
        try:
            # 1. Ép buộc torch backend
            os.environ["DUNEAI_MAMBA_BACKEND"] = "torch"
            self.assertEqual(get_scan_backend(), "torch")
            u = torch.randn(2, 4, 8)
            delta = torch.randn(2, 4, 8)
            A = -torch.rand(4, 2)
            B = torch.randn(2, 2, 8)
            C = torch.randn(2, 2, 8)
            out = selective_scan_fn(u, delta, A, B, C)
            self.assertEqual(out.shape, (2, 4, 8))

            # 2. Ép buộc cuda backend khi không có GPU / CUDA mamba
            os.environ["DUNEAI_MAMBA_BACKEND"] = "cuda"
            self.assertEqual(get_scan_backend(), "cuda")
            if not torch.cuda.is_available():
                with self.assertRaises(RuntimeError):
                    selective_scan_fn(u, delta, A, B, C)
        finally:
            if old_val is not None:
                os.environ["DUNEAI_MAMBA_BACKEND"] = old_val
            else:
                os.environ.pop("DUNEAI_MAMBA_BACKEND", None)


if __name__ == "__main__":
    unittest.main()
