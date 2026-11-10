# -*- coding: utf-8 -*-
"""
unet_base/activations.py - Registry và các hàm kích hoạt (Activation Functions) riêng của unet_base.

Quy tắc bất biến (R1, R11):
  - File tự đủ: CHỈ import torch và thư viện chuẩn, không import bất kỳ file nào khác của dự án.
  - Mỗi hàm kích hoạt trả về một FACTORY không đối số: Callable[[], nn.Module]
    để mỗi lần gọi sẽ tạo một instance nn.Module mới (dùng cho nhiều lớp khác nhau).
  - Khai báo rõ ràng has_params (True nếu có tham số học được làm đổi tập khóa state_dict).
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. CÁC LỚP ACTIVATION TỰ VIẾT
# =============================================================================
class SoftClampReLU(nn.Module):
    """
    ReLU với hàm kẹp mềm ở ngưỡng trên tau:
        f(x) = tau * tanh(relu(x) / tau)
    Không có tham số học được (has_params=False).
    """
    def __init__(self, tau: float = 10.0, inplace: bool = False):
        super().__init__()
        self.tau = float(tau)
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = F.relu(x, inplace=self.inplace)
        return self.tau * torch.tanh(pos / self.tau)

    def extra_repr(self) -> str:
        return f"tau={self.tau}, inplace={self.inplace}"


class SoftClampSiLU(nn.Module):
    """
    SiLU (Swish) với cơ chế kẹp mềm đạo hàm ở các giá trị cực lớn:
        f(x) = silu(x) / (1.0 + |x| / tau)
    Không có tham số học được (has_params=False).
    """
    def __init__(self, tau: float = 20.0):
        super().__init__()
        self.tau = float(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_val = F.silu(x)
        scale = 1.0 + torch.abs(x) / self.tau
        return silu_val / scale

    def extra_repr(self) -> str:
        return f"tau={self.tau}"


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x)^2 + b
    Biến thể phi tuyến bậc hai.
    Nếu learnable=True, bổ sung 2 tham số học được (scale, bias) -> has_params=True.
    """
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0, learnable: bool = False):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.scale = nn.Parameter(torch.tensor(scale_value, dtype=torch.float32))
            self.bias = nn.Parameter(torch.tensor(bias_value, dtype=torch.float32))
        else:
            self.register_buffer("scale", torch.tensor(scale_value, dtype=torch.float32))
            self.register_buffer("bias", torch.tensor(bias_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (F.relu(x) ** 2) + self.bias


# =============================================================================
# 2. BẢNG ĐĂNG KÝ ACTIVATION (REGISTRY)
# =============================================================================
@dataclass
class ActivationSpec:
    factory: Callable[..., nn.Module]
    has_params: bool
    doc: str


ACTIVATIONS: Dict[str, ActivationSpec] = {
    "relu": ActivationSpec(
        factory=lambda inplace=True, **kw: nn.ReLU(inplace=inplace),
        has_params=False,
        doc="Chuẩn Rectified Linear Unit (mặc định của unet_base)",
    ),
    "leaky_relu": ActivationSpec(
        factory=lambda negative_slope=0.01, inplace=True, **kw: nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace),
        has_params=False,
        doc="LeakyReLU với negative_slope tùy chỉnh",
    ),
    "gelu": ActivationSpec(
        factory=lambda approximate="none", **kw: nn.GELU(approximate=approximate),
        has_params=False,
        doc="Gaussian Error Linear Unit",
    ),
    "silu": ActivationSpec(
        factory=lambda inplace=False, **kw: nn.SiLU(inplace=inplace),
        has_params=False,
        doc="Sigmoid Linear Unit (Swish)",
    ),
    "mish": ActivationSpec(
        factory=lambda inplace=False, **kw: nn.Mish(inplace=inplace),
        has_params=False,
        doc="Mish: x * tanh(softplus(x))",
    ),
    "soft_clamp_relu": ActivationSpec(
        factory=lambda tau=10.0, inplace=False, **kw: SoftClampReLU(tau=tau, inplace=inplace),
        has_params=False,
        doc="ReLU có hàm kẹp mềm ở ngưỡng trên tau",
    ),
    "soft_clamp_silu": ActivationSpec(
        factory=lambda tau=20.0, **kw: SoftClampSiLU(tau=tau),
        has_params=False,
        doc="SiLU có hàm kẹp mềm đạo hàm ở giá trị cực lớn",
    ),
    "star_relu": ActivationSpec(
        factory=lambda scale_value=1.0, bias_value=0.0, learnable=True, **kw: StarReLU(
            scale_value=scale_value, bias_value=bias_value, learnable=learnable
        ),
        has_params=True,
        doc="StarReLU (s * relu(x)^2 + b, thêm khóa scale và bias vào state_dict)",
    ),
    "identity": ActivationSpec(
        factory=lambda **kw: nn.Identity(),
        has_params=False,
        doc="Tuyến tính (Identity / Không kích hoạt)",
    ),
}

DEFAULT_ACTIVATION: str = "relu"


def get_activation(name: Optional[str] = None, **kwargs) -> Callable[[], nn.Module]:
    """
    Trả về một FACTORY không đối số: mỗi lần gọi sinh ra một nn.Module mới.

    Args:
        name: Tên hàm kích hoạt (mặc định: DEFAULT_ACTIVATION = 'relu').
        **kwargs: Tham số bổ sung truyền cho factory.

    Returns:
        Callable[[], nn.Module]
    """
    if name is None:
        name = DEFAULT_ACTIVATION

    key = name.lower().strip().replace("-", "_")
    if key in ("rectified_linear_unit",):
        key = "relu"
    elif key in ("lrelu",):
        key = "leaky_relu"
    elif key in ("swish",):
        key = "silu"
    elif key in ("softclamprelu",):
        key = "soft_clamp_relu"
    elif key in ("softclampsilu",):
        key = "soft_clamp_silu"
    elif key in ("starrelu",):
        key = "star_relu"
    elif key in ("none", "linear"):
        key = "identity"

    if key not in ACTIVATIONS:
        valid_keys = list(ACTIVATIONS.keys())
        raise ValueError(
            f"❌ Không hỗ trợ activation '{name}' trong unet_base!\n"
            f"  Các lựa chọn hợp lệ: {valid_keys}"
        )

    spec = ACTIVATIONS[key]
    # Trả về factory không đối số
    return lambda: spec.factory(**kwargs)


def list_activations() -> List[str]:
    """Liệt kê danh sách tên các hàm kích hoạt hợp lệ trong unet_base."""
    return list(ACTIVATIONS.keys())


def _selftest():
    print("🧪 Bắt đầu selftest unet_base.activations...")
    x = torch.linspace(-5, 15, 100)

    for act_name in list_activations():
        factory = get_activation(act_name)
        mod = factory()
        out = mod(x)
        assert out.shape == x.shape, f"Lỗi shape tại {act_name}: {out.shape} != {x.shape}"
        spec = ACTIVATIONS[act_name]
        param_count = sum(p.numel() for p in mod.parameters())
        if spec.has_params:
            assert param_count > 0, f"Activation {act_name} khai báo has_params=True nhưng không có tham số!"
        else:
            assert param_count == 0, f"Activation {act_name} khai báo has_params=False nhưng có {param_count} tham số!"

    # Kiểm tra tính độc lập của các module sinh ra từ factory
    f = get_activation("relu")
    m1 = f()
    m2 = f()
    assert m1 is not m2, "Factory phải sinh ra các instance riêng biệt!"
    print(f"✅ Đã kiểm thử thành công {len(ACTIVATIONS)} hàm kích hoạt trong unet_base.activations!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Quản lý và kiểm thử Activation Functions của unet_base")
    parser.add_argument("--list", action="store_true", help="In danh sách các activation hợp lệ")
    args = parser.parse_args()

    if args.list:
        print("\n".join(list_activations()))
    else:
        _selftest()
