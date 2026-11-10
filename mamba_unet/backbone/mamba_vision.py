# -*- coding: utf-8 -*-
"""
mamba_unet/backbone/mamba_vision.py - Backbone MambaVision phục vụ phân đoạn khối u 2.5D.

Quy tắc bất biến:
  - Trích xuất 4 feature maps đa tỷ lệ [H/4, H/8, H/16, H/32] làm Skip Connections cho UNetDecoder.
  - Sử dụng các lớp dự phòng và selective_scan_fn từ mamba_unet.fallbacks.
  - Tuyệt đối KHÔNG import từ unet_base hay common (R1, R2).
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mamba_unet.fallbacks import (
    DropPath,
    LayerNorm2d,
    rearrange,
    repeat,
    selective_scan_fn,
    trunc_normal_,
)


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    """Chia feature map thành các window không chồng lấp."""
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows, Hp, Wp


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, C: int) -> torch.Tensor:
    """Ghép các window trở lại thành feature map kích thước đầy đủ."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
    return x


class Downsample(nn.Module):
    """Khối giảm chiều không gian (giảm nửa H và W, tăng gấp đôi số kênh)."""
    def __init__(self, dim: int, keep_dim: bool = False):
        super().__init__()
        out_dim = dim if keep_dim else 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.norm = LayerNorm2d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.reduction(x))


class PatchEmbed(nn.Module):
    """Khối biến đổi ảnh đầu vào ban đầu: giảm kích thước không gian xuống H/4, W/4."""
    def __init__(self, in_chans: int = 3, in_dim: int = 32, dim: int = 80):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_down(x)


class ConvBlock(nn.Module):
    """Khối tích chập sâu (Depthwise Separable Conv) cho Stage 1 & 2."""
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale: Optional[float] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm = LayerNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim, 1, 1)) if layer_scale is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        if self.gamma is not None:
            x = self.gamma * x
        return input_x + self.drop_path(x)


class MambaVisionMixer(nn.Module):
    """Khối trộn đặc trưng không gian dựa trên 1D Selective Scan State Space Model."""
    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, l, d = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = self.act(x)
        x = x.transpose(1, 2)

        x_for_proj = x.reshape(b * l, -1)
        x_dbl = self.x_proj(x_for_proj)
        dt, B_proj, C_proj = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt).view(b, l, -1).transpose(1, 2)
        B_proj = B_proj.view(b, l, -1).transpose(1, 2)
        C_proj = C_proj.view(b, l, -1).transpose(1, 2)
        x_in = x.transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        y = selective_scan_fn(
            x_in,
            dt,
            A,
            B_proj,
            C_proj,
            self.D.float(),
            z=z.transpose(1, 2),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y = y.transpose(1, 2)
        return self.out_proj(y)


class Attention(nn.Module):
    """Multi-Head Self-Attention với Relative Position Bias."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Khối kết hợp MambaVisionMixer + Attention + MLP."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: Optional[float] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = MambaVisionMixer(d_model=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.attn(self.norm2(x)))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class MambaVisionLayer(nn.Module):
    """Một tầng (Stage) của MambaVision."""
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        downsample: bool = True,
        layer_scale: Optional[float] = None,
        is_transformer_block: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.transformer_block = is_transformer_block

        if not self.transformer_block:
            self.blocks = nn.ModuleList([
                ConvBlock(
                    dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ])

        self.downsample = Downsample(dim=dim) if downsample else None

    def forward(self, x: torch.Tensor, return_feature_before_downsample: bool = True):
        B, C, H, W = x.shape

        if self.transformer_block:
            windows, Hp, Wp = window_partition(x, self.window_size)
            for blk in self.blocks:
                windows = blk(windows)
            x = window_reverse(windows, self.window_size, Hp, Wp, C)
            if Hp > H or Wp > W:
                x = x[:, :, :H, :W].contiguous()
        else:
            for blk in self.blocks:
                x = blk(x)

        feat_before_down = x
        if self.downsample is not None:
            x_next = self.downsample(x)
        else:
            x_next = x

        if return_feature_before_downsample:
            return x_next, feat_before_down
        return x_next


class MambaVision(nn.Module):
    """
    MambaVision Backbone cho phân đoạn ảnh y tế:
      Forward trả về danh sách 4 multi-scale feature maps: [H/4, H/8, H/16, H/32].
    """
    def __init__(
        self,
        dim: int = 80,
        in_dim: int = 32,
        depths: List[int] = [1, 3, 8, 4],
        window_size: List[int] = [8, 8, 14, 7],
        mlp_ratio: float = 4.0,
        num_heads: List[int] = [2, 4, 8, 16],
        drop_path_rate: float = 0.2,
        in_chans: int = 3,
        qkv_bias: bool = True,
        layer_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_levels = len(depths)
        self.dim = dim
        self.in_dim = in_dim
        self.in_chans = in_chans

        self.out_channels = [dim, dim * 2, dim * 4, dim * 8]
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            stage_dim = int(dim * 2**i)
            is_transformer = i >= 2
            level = MambaVisionLayer(
                dim=stage_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                is_transformer_block=is_transformer,
            )
            self.levels.append(level)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        features = []
        for level in self.levels:
            x, feat = level(x, return_feature_before_downsample=True)
            features.append(feat)
        return features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.forward_features(x)


# =============================================================================
# BIẾN THỂ VÀ FACTORY
# =============================================================================
def mamba_vision_T(pretrained: bool = False, in_chans: int = 3, **kwargs) -> MambaVision:
    """Biến thể MambaVision-Tiny: dim=80, depths=[1, 3, 8, 4], out_channels=[80, 160, 320, 640]."""
    model = MambaVision(
        dim=80,
        in_dim=32,
        depths=[1, 3, 8, 4],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        in_chans=in_chans,
        **kwargs,
    )
    if pretrained:
        from .pretrained import load_pretrained_mambavision
        load_pretrained_mambavision(model, "mamba_vision_t")
    return model


def mamba_vision_T2(pretrained: bool = False, in_chans: int = 3, **kwargs) -> MambaVision:
    """Biến thể MambaVision-Tiny2: dim=80, depths=[1, 3, 11, 4]."""
    model = MambaVision(
        dim=80,
        in_dim=32,
        depths=[1, 3, 11, 4],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        in_chans=in_chans,
        **kwargs,
    )
    if pretrained:
        from .pretrained import load_pretrained_mambavision
        load_pretrained_mambavision(model, "mamba_vision_t2")
    return model


def mamba_vision_S(pretrained: bool = False, in_chans: int = 3, **kwargs) -> MambaVision:
    """Biến thể MambaVision-Small: dim=96, depths=[3, 3, 10, 5], out_channels=[96, 192, 384, 768]."""
    model = MambaVision(
        dim=96,
        in_dim=64,
        depths=[3, 3, 10, 5],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        in_chans=in_chans,
        **kwargs,
    )
    if pretrained:
        from .pretrained import load_pretrained_mambavision
        load_pretrained_mambavision(model, "mamba_vision_s")
    return model


def mamba_vision_B(pretrained: bool = False, in_chans: int = 3, **kwargs) -> MambaVision:
    """Biến thể MambaVision-Base: dim=128, depths=[3, 3, 12, 5], out_channels=[128, 256, 512, 1024]."""
    model = MambaVision(
        dim=128,
        in_dim=64,
        depths=[3, 3, 12, 5],
        window_size=[8, 8, 14, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        in_chans=in_chans,
        **kwargs,
    )
    if pretrained:
        from .pretrained import load_pretrained_mambavision
        load_pretrained_mambavision(model, "mamba_vision_b")
    return model


def build_mambavision_backbone(
    name: str = "mamba_vision_T",
    pretrained: bool = False,
    in_chans: int = 3,
    **kwargs,
) -> MambaVision:
    """Factory function tạo MambaVision backbone theo tên biến thể."""
    registry = {
        "mamba_vision_t": mamba_vision_T,
        "mamba_vision_t2": mamba_vision_T2,
        "mamba_vision_s": mamba_vision_S,
        "mamba_vision_b": mamba_vision_B,
    }
    key = name.lower().strip()
    if key not in registry:
        raise ValueError(f"Không hỗ trợ backbone '{name}'. Các lựa chọn: {list(registry.keys())}")
    return registry[key](pretrained=pretrained, in_chans=in_chans, **kwargs)


def _selftest():
    print("🧪 Bắt đầu selftest mamba_unet.backbone.mamba_vision...")
    model = mamba_vision_T(pretrained=False, in_chans=3)
    model.eval()

    dummy = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feats = model(dummy)

    assert len(feats) == 4, f"Kỳ vọng 4 feature maps, nhận được {len(feats)}"
    assert feats[0].shape == (2, 80, 16, 16), f"Lỗi Stage 1: {feats[0].shape}"
    assert feats[1].shape == (2, 160, 8, 8), f"Lỗi Stage 2: {feats[1].shape}"
    assert feats[2].shape == (2, 320, 4, 4), f"Lỗi Stage 3: {feats[2].shape}"
    assert feats[3].shape == (2, 640, 2, 2), f"Lỗi Stage 4: {feats[3].shape}"
    print("✅ Selftest mamba_unet.backbone.mamba_vision hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
