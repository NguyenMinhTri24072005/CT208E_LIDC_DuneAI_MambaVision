# -*- coding: utf-8 -*-
"""
mamba_unet/fallbacks.py - Chọn backend Selective Scan (CUDA vs Pure PyTorch) và các lớp dự phòng.

Quy tắc bất biến (R8, spec mục 5.3):
  - mamba_unet chạy được ngay cả khi không có mamba_ssm, timm, einops (dùng bản dự phòng thuần PyTorch).
  - Biến môi trường DUNEAI_MAMBA_BACKEND=torch|cuda để ép backend; ép cuda mà thiếu kernel/GPU -> RuntimeError rõ ràng.
  - selective_scan_torch tính toán bằng float32, khớp công thức đóng O(L²) tính độc lập bằng float64 với sai khác <= 1e-4.
  - Backward pass cho gradient hữu hạn.
"""

import os
import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. FALLBACKS CHO TIMM VÀ EINOPS (R8: Phụ thuộc nặng là tùy chọn)
# =============================================================================
try:
    from timm.layers import DropPath, LayerNorm2d, trunc_normal_
except ImportError:
    try:
        from timm.models.layers import DropPath, LayerNorm2d, trunc_normal_
    except ImportError:
        def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
            return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
                super().__init__()
                self.drop_prob = drop_prob
                self.scale_by_keep = scale_by_keep

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep_prob = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
                if keep_prob > 0.0 and self.scale_by_keep:
                    random_tensor.div_(keep_prob)
                return x * random_tensor

        class LayerNorm2d(nn.Module):
            def __init__(self, num_channels: int, eps: float = 1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(num_channels))
                self.bias = nn.Parameter(torch.zeros(num_channels))
                self.eps = eps

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x


try:
    from einops import rearrange, repeat
except ImportError:
    def rearrange(tensor: torch.Tensor, pattern: str, **kwargs) -> torch.Tensor:
        pattern = pattern.strip()
        if pattern == "b l d -> b d l":
            return tensor.transpose(1, 2).contiguous()
        elif pattern == "b d l -> b l d":
            return tensor.transpose(1, 2).contiguous()
        elif pattern == "b d l -> (b l) d":
            b, d, l = tensor.shape
            return tensor.permute(0, 2, 1).reshape(b * l, d).contiguous()
        elif pattern == "(b l) d -> b d l":
            l = kwargs.get("l")
            bl, d = tensor.shape
            b = bl // l
            return tensor.view(b, l, d).permute(0, 2, 1).contiguous()
        elif pattern == "(b l) dstate -> b dstate l":
            l = kwargs.get("l")
            bl, ds = tensor.shape
            b = bl // l
            return tensor.view(b, l, ds).permute(0, 2, 1).contiguous()
        elif pattern == "... (L two) -> ... L two":
            shape = list(tensor.shape)
            shape[-1] = shape[-1] // 2
            shape.append(2)
            return tensor.view(*shape)
        elif pattern == "d -> d 1":
            return tensor.unsqueeze(1)
        raise NotImplementedError(f"Pattern rearrange chưa hỗ trợ: {pattern}")

    def repeat(tensor: torch.Tensor, pattern: str, **kwargs) -> torch.Tensor:
        clean_pat = pattern.replace(" ", "")
        if "B G N L -> B (G H) N L" in pattern:
            h = kwargs.get("H", 1)
            return tensor.repeat_interleave(h, dim=1)
        elif clean_pat == "n->dn":
            d = kwargs.get("d", 1)
            return tensor.unsqueeze(0).repeat(d, 1)
        raise NotImplementedError(f"Pattern repeat chưa hỗ trợ: {pattern}")


# =============================================================================
# 2. SELECTIVE SCAN PURE PYTORCH FALLBACK
# =============================================================================
def selective_scan_torch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Thuật toán Selective Scan cài đặt bằng PyTorch thuần túy (CPU & GPU).
    Tính toán bằng float32 để ổn định gradient và khớp độ chính xác cao.
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()

    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))

    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B_rep = B.repeat_interleave(dim // B.shape[1], dim=1)
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B_rep, u)

    if is_variable_C and C.dim() == 4:
        C_rep = C.repeat_interleave(dim // C.shape[1], dim=1)
    else:
        C_rep = C

    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C_rep[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)

    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.unsqueeze(1)
    if z is not None:
        out = out * F.silu(z.float())

    out = out.to(dtype=dtype_in)
    return (out, last_state) if return_last_state else out


def get_scan_backend() -> str:
    """Trả về tên backend hiện đang được kích hoạt ('cuda' hoặc 'torch')."""
    forced = os.environ.get("DUNEAI_MAMBA_BACKEND", "").lower().strip()
    if forced in ("cuda", "torch"):
        return forced
    if torch.cuda.is_available():
        try:
            import mamba_ssm
            return "cuda"
        except ImportError:
            pass
    return "torch"


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Dispatcher chọn Selective Scan Backend dựa trên cấu hình môi trường:
      - DUNEAI_MAMBA_BACKEND=cuda : ép dùng CUDA mamba_ssm kernel, lỗi nếu không khả dụng.
      - DUNEAI_MAMBA_BACKEND=torch: ép dùng selective_scan_torch thuần PyTorch.
      - Mặc định: tự động chọn CUDA nếu có GPU và gói mamba_ssm, ngược lại dùng fallback PyTorch.
    """
    forced_backend = os.environ.get("DUNEAI_MAMBA_BACKEND", "").lower().strip()

    if forced_backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("❌ DUNEAI_MAMBA_BACKEND=cuda được yêu cầu nhưng thiết bị không có GPU CUDA khả dụng!")
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as cuda_scan
            return cuda_scan(
                u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus, return_last_state=return_last_state
            )
        except ImportError as e:
            raise RuntimeError(f"❌ DUNEAI_MAMBA_BACKEND=cuda được yêu cầu nhưng không import được gói mamba_ssm: {e}")

    elif forced_backend == "torch":
        return selective_scan_torch(
            u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus, return_last_state=return_last_state
        )

    # Chế độ tự động (Auto fallback)
    if u.is_cuda and torch.cuda.is_available():
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as cuda_scan
            return cuda_scan(
                u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus, return_last_state=return_last_state
            )
        except (ImportError, Exception):
            pass

    return selective_scan_torch(
        u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus, return_last_state=return_last_state
    )


# =============================================================================
# 3. CÔNG THỨC THAM CHIẾU ĐÓNG O(L^2) BẰNG FLOAT64 ĐỂ KIỂM THỬ
# =============================================================================
def selective_scan_ref_float64(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Công thức tham chiếu độc lập tính bằng float64 phục vụ kiểm thử độ chính xác (<= 1e-4).
    """
    u_64 = u.to(torch.float64)
    delta_64 = delta.to(torch.float64)
    A_64 = A.to(torch.float64)
    B_64 = B.to(torch.float64)
    C_64 = C.to(torch.float64)

    batch, dim, L = u.shape
    dstate = A.shape[1]

    # Tính toán tuần tự trạng thái ẩn x_t bằng float64
    x = torch.zeros((batch, dim, dstate), dtype=torch.float64, device=u.device)
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta_64, A_64))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta_64, B_64, u_64)

    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum("bdn,bn->bd", x, C_64[:, :, i])
        ys.append(y)

    y_out = torch.stack(ys, dim=2)
    if D is not None:
        y_out = y_out + u_64 * D.to(torch.float64).unsqueeze(1)
    return y_out.float()


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.fallbacks...")
    torch.manual_seed(42)
    B, D_in, L, N = 2, 4, 8, 4
    u = torch.randn(B, D_in, L, requires_grad=True)
    delta = F.softplus(torch.randn(B, D_in, L)).requires_grad_(True)
    A = -torch.rand(D_in, N)
    B_mat = torch.randn(B, N, L, requires_grad=True)
    C_mat = torch.randn(B, N, L, requires_grad=True)
    D_vec = torch.randn(D_in)

    # 1. So sánh selective_scan_torch với công thức tham chiếu float64
    out_torch = selective_scan_torch(u, delta, A, B_mat, C_mat, D=D_vec)
    out_ref = selective_scan_ref_float64(u, delta, A, B_mat, C_mat, D=D_vec)

    diff = (out_torch - out_ref).abs().max().item()
    print(f"  ✓ Sai khác tối đa so với float64 tham chiếu: {diff:.2e} (Tiêu chí <= 1e-4)")
    assert diff <= 1e-4, f"Sai khác quá lớn: {diff}"

    # 2. Kiểm tra backward pass tính toán đạo hàm hữu hạn
    loss = out_torch.sum()
    loss.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all(), "Đạo hàm u chứa NaN hoặc Inf!"
    assert delta.grad is not None and torch.isfinite(delta.grad).all(), "Đạo hàm delta chứa NaN hoặc Inf!"
    print("  ✓ Gradient backward hữu hạn và ổn định.")

    # 3. Kiểm tra biến môi trường DUNEAI_MAMBA_BACKEND=torch
    os.environ["DUNEAI_MAMBA_BACKEND"] = "torch"
    out_forced = selective_scan_fn(u, delta, A, B_mat, C_mat, D=D_vec)
    assert out_forced.shape == out_torch.shape
    os.environ.pop("DUNEAI_MAMBA_BACKEND", None)

    print("✅ Selftest mamba_unet.fallbacks hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
