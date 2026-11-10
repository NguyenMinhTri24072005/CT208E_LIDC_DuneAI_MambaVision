# -*- coding: utf-8 -*-
"""
mamba_unet/backbone/mamba_vision.py - Backbone MambaVision chuẩn NVIDIA (NVlabs/MambaVision - CVPR 2024).

Quy tắc bất biến:
  - Đồng bộ 1-1 kiến trúc gốc từ NVlabs/MambaVision:
    * Stem: PatchEmbed stride 4.
    * Stage 1 & 2: ConvBlock (Depthwise Separable Conv).
    * Stage 3 & 4: Block hybrid chứa 1 Mixer (MambaVisionMixer HOẶC WindowAttention) + Mlp(fc1, fc2).
    * MambaVisionMixer: Cấu trúc nhánh đối xứng chuẩn NVIDIA với conv1d_x, conv1d_z, Selective Scan và out_proj.
    * Tỷ lệ phân chia Stage 3: 4 Mamba + 4 Attention (công thức depths[i] // 2).
    * Tỷ lệ phân chia Stage 4: 2 Mamba + 2 Attention.
    * Downsample: nn.Conv2d stride 2 thuần túy (không norm bên trong).
  - Trích xuất 4 feature maps đa tỷ lệ [H/4, H/8, H/16, H/32] làm Skip Connections cho UNetDecoder.
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


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Chia feature map (B, C, H, W) thành các windows không chồng lấp.
    Đầu vào H, W đã được đệm bảo đảm chia hết cho window_size.
    Đầu ra: (B * num_windows, window_size * window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Ghép các window (B * num_windows, window_size * window_size, C) trở lại thành (B, C, H, W).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x


class Downsample(nn.Module):
    """Khối giảm chiều không gian chuẩn NVIDIA: chỉ dùng conv strided 2, không norm bên trong."""
    def __init__(self, dim: int, keep_dim: bool = False):
        super().__init__()
        out_dim = dim if keep_dim else 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduction(x)


class PatchEmbed(nn.Module):
    """Khối biến đổi ảnh đầu vào ban đầu chuẩn NVIDIA: giảm kích thước không gian xuống H/4, W/4."""
    def __init__(self, in_chans: int = 3, in_dim: int = 32, dim: int = 80):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_down(x)


class ConvBlock(nn.Module):
    """Khối tích chập ResNet-style chuẩn NVIDIA (Conv3x3 + BN + GELU + Conv3x3 + BN)."""
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale: Optional[float] = None, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate="tanh")
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim, 1, 1)) if self.layer_scale else 1.0
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = input_x + self.drop_path(self.gamma * x)
        return x


class Mlp(nn.Module):
    """Khối Mlp chuẩn NVIDIA (tương thích timm), sử dụng fc1 và fc2."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Any = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MambaVisionMixer(nn.Module):
    """Khối trộn đặc trưng không gian dựa trên nhánh đối xứng conv1d_x/conv1d_z và 1D Selective Scan chuẩn NVIDIA."""
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
        conv_bias: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Trong MambaVision: in_proj chiếu từ d_model sang d_inner (sau đó chunk 2 thành x và z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)

        # 2 nhánh tích chập 1D độc lập cho x và z
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            padding=d_conv - 1,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner // 2) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, l, d = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = xz.transpose(1, 2)
        x, z = xz.chunk(2, dim=1)

        x = self.conv1d_x(x)[:, :, :l]
        x = self.act(x)

        z = self.conv1d_z(z)[:, :, :l]
        z = self.act(z)

        x_trans = x.transpose(1, 2)
        x_for_proj = x_trans.reshape(b * l, -1)
        x_dbl = self.x_proj(x_for_proj)
        dt, B_proj, C_proj = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt).view(b, l, -1).transpose(1, 2)
        B_proj = B_proj.view(b, l, -1).transpose(1, 2)
        C_proj = C_proj.view(b, l, -1).transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        y = selective_scan_fn(
            x,
            dt,
            A,
            B_proj,
            C_proj,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        # Ghép nối nhánh SSM (y) và nhánh tích chập điều chế (z) theo chuẩn NVIDIA MambaVision
        y = torch.cat([y, z], dim=1)
        y = y.transpose(1, 2)
        return self.out_proj(y)


class Attention(nn.Module):
    """Multi-Head Self-Attention chuẩn NVIDIA MambaVision."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) phải chia hết cho num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_norm = qk_scale
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Khối hybrid chuẩn NVIDIA: Mỗi block chỉ chứa 1 Mixer (Mamba HOẶC Attention) + 1 MLP."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        counter: int,
        transformer_blocks: List[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: Optional[float] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                d_state=8,
                d_conv=3,
                expand=1,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        use_layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        self.g_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1.0
        self.g_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.g_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.g_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """Một Stage trong MambaVision chuẩn NVIDIA (hỗ trợ cả ConvStage và HybridStage)."""
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        conv: bool = False,
        downsample: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        layer_scale: Optional[float] = None,
        transformer_blocks: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.conv = conv
        transformer_blocks = transformer_blocks or []

        if self.conv:
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
                    counter=j,
                    transformer_blocks=transformer_blocks,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale,
                )
                for j in range(depth)
            ])

        self.downsample = Downsample(dim=dim) if downsample else None

    def forward(self, x: torch.Tensor, return_feature_before_downsample: bool = True):
        B, C, H, W = x.shape
        if not self.conv:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape

            windows = window_partition(x, self.window_size)
            for blk in self.blocks:
                windows = blk(windows)
            x = window_reverse(windows, self.window_size, Hp, Wp)

            if pad_r > 0 or pad_b > 0:
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
    MambaVision Backbone chuẩn NVIDIA phục vụ phân đoạn ảnh y tế:
      Forward trích xuất danh sách 4 multi-scale feature maps: [H/4, H/8, H/16, H/32].
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

        self.out_channels = [int(dim * 2**i) for i in range(self.num_levels)]
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            stage_dim = int(dim * 2**i)
            conv = True if (i == 0 or i == 1) else False

            # Công thức phân chia chuẩn của NVIDIA:
            # Stage 3 (depth=8) -> t_blocks = [4, 5, 6, 7] (4 Mamba + 4 Attention)
            # Stage 4 (depth=4) -> t_blocks = [2, 3]       (2 Mamba + 2 Attention)
            t_blocks = (
                list(range(depths[i] // 2 + 1, depths[i]))
                if depths[i] % 2 != 0
                else list(range(depths[i] // 2, depths[i]))
            ) if not conv else []

            level = MambaVisionLayer(
                dim=stage_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                conv=conv,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                transformer_blocks=t_blocks,
            )
            self.levels.append(level)

        # Lớp BatchNorm2d chuẩn của NVIDIA ở cuối Stage 4 (khớp norm.weight/bias trong checkpoint)
        self.num_features = int(dim * 2 ** (self.num_levels - 1))
        self.norm = nn.BatchNorm2d(self.num_features)

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

    # 1. Test khối PatchEmbed
    pe = PatchEmbed(in_chans=3, in_dim=32, dim=80)
    x = torch.randn(2, 3, 64, 64)
    pe_out = pe(x)
    assert pe_out.shape == (2, 80, 16, 16), f"Lỗi PatchEmbed: {pe_out.shape}"
    print("  ✓ PatchEmbed PASS")

    # 2. Test khối ConvBlock
    cb = ConvBlock(dim=80)
    cb_out = cb(pe_out)
    assert cb_out.shape == pe_out.shape, f"Lỗi ConvBlock: {cb_out.shape}"
    print("  ✓ ConvBlock PASS")

    # 3. Test khối Downsample
    ds = Downsample(dim=80)
    ds_out = ds(cb_out)
    assert ds_out.shape == (2, 160, 8, 8), f"Lỗi Downsample: {ds_out.shape}"
    print("  ✓ Downsample PASS")

    # 4. Test khối Mlp
    mlp = Mlp(in_features=80, hidden_features=320)
    tokens = torch.randn(2, 16, 80)
    mlp_out = mlp(tokens)
    assert mlp_out.shape == tokens.shape, f"Lỗi Mlp: {mlp_out.shape}"
    print("  ✓ Mlp PASS")

    # 5. Test khối Attention
    attn = Attention(dim=80, num_heads=2)
    attn_out = attn(tokens)
    assert attn_out.shape == tokens.shape, f"Lỗi Attention: {attn_out.shape}"
    print("  ✓ Attention PASS")

    # 6. Test khối MambaVisionMixer
    mixer = MambaVisionMixer(d_model=80)
    mixer_out = mixer(tokens)
    assert mixer_out.shape == tokens.shape, f"Lỗi MambaVisionMixer: {mixer_out.shape}"
    print("  ✓ MambaVisionMixer PASS")

    # 7. Test khối Block (Mamba Mixer và Attention Mixer)
    blk_mamba = Block(dim=80, num_heads=2, counter=0, transformer_blocks=[2, 3])
    assert isinstance(blk_mamba.mixer, MambaVisionMixer)
    assert blk_mamba(tokens).shape == tokens.shape

    blk_attn = Block(dim=80, num_heads=2, counter=2, transformer_blocks=[2, 3])
    assert isinstance(blk_attn.mixer, Attention)
    assert blk_attn(tokens).shape == tokens.shape
    print("  ✓ Block (Mamba & Attention) PASS")

    # 8. Test toàn bộ mô hình MambaVision-Tiny trên ảnh kích thước 64x64 và 128x128
    for sz in [64, 128]:
        model = mamba_vision_T(pretrained=False, in_chans=3)
        model.eval()
        dummy = torch.randn(2, 3, sz, sz)
        with torch.no_grad():
            feats = model(dummy)
        assert len(feats) == 4, f"Kỳ vọng 4 feature maps, nhận {len(feats)}"
        assert feats[0].shape == (2, 80, sz // 4, sz // 4), f"Lỗi Stage 1: {feats[0].shape}"
        assert feats[1].shape == (2, 160, sz // 8, sz // 8), f"Lỗi Stage 2: {feats[1].shape}"
        assert feats[2].shape == (2, 320, sz // 16, sz // 16), f"Lỗi Stage 3: {feats[2].shape}"
        assert feats[3].shape == (2, 640, sz // 32, sz // 32), f"Lỗi Stage 4: {feats[3].shape}"
        print(f"  ✓ MambaVision-Tiny forward (sz={sz}) PASS")

    # 9. Test khởi tạo các biến thể khác
    for variant_fn, name in [
        (mamba_vision_T2, "Tiny2"),
        (mamba_vision_S, "Small"),
        (mamba_vision_B, "Base"),
    ]:
        m = variant_fn(pretrained=False, in_chans=3)
        print(f"  ✓ Khởi tạo {name} PASS (out_channels: {m.out_channels})")

    print("✅ Selftest mamba_unet.backbone.mamba_vision hoàn tất thành công!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _selftest()
