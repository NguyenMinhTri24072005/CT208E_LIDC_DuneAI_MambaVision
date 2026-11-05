# -*- coding: utf-8 -*-
"""
custom_backbone.py - Bộ mã hóa (Encoder) tùy chỉnh tích hợp MambaVision cho DuneAI.

Đặc điểm thiết kế:
  1. Lazy Import (Nạp trễ): Chỉ nạp thư viện `transformers` / `mambavision` khi thực sự
     bật MambaVisionEncoder. Khi chạy Baseline DuneUNet (`backbone="none"`), hệ thống
     hoàn toàn không yêu cầu cài đặt các thư viện ngoài này.
  2. Tự động nhận diện số kênh (Dynamic Channel Probing): Tự động dò kích thước 4 tầng
     đặc trưng (Stage 1..4) của mọi biến thể MambaVision (T, T2, S, B, L).
  3. Cơ chế Đóng / Mở khóa trọng số (Freeze / Unfreeze Backbone):
     - `freeze()`: Khóa toàn bộ tham số backbone (`requires_grad=False`) và khóa các lớp
       BatchNorm/LayerNorm ở chế độ `.eval()` để bảo toàn tri thức Pre-trained.
     - `unfreeze()`: Mở khóa toàn bộ tham số (`requires_grad=True`) để Fine-tune toàn mạng.
  4. Hỗ trợ Mock Backbone thuần PyTorch (`use_mock=True`) để chạy `selftest` offline
     mà không cần internet hay nhân CUDA `mamba-ssm`.

Dùng:
  python DuneAI_Remake/src/models/custom_backbone.py selftest
"""
import argparse
import torch
import torch.nn as nn


# Bảng ánh xạ tên gọi ngắn gọn sang kho lưu trữ Hugging Face chính thức của NVIDIA
MAMBAVISION_REGISTRY = {
    "mambavision_t": "nvidia/MambaVision-T-1K",
    "mambavision_t2": "nvidia/MambaVision-T2-1K",
    "mambavision_s": "nvidia/MambaVision-S-1K",
    "mambavision_b": "nvidia/MambaVision-B-1K",
    "mambavision_l": "nvidia/MambaVision-L-1K",
    "mambavision_l2": "nvidia/MambaVision-L2-1K",
}


# Kênh cơ sở `dim` của từng biến thể (theo mã nguồn NVlabs/MambaVision); 4 tầng có số kênh dim * 2^i.
# MambaVision cần nhân CUDA `mamba-ssm` nên KHÔNG chạy được trên CPU -> không thể "dò" bằng forward ở đó.
MAMBAVISION_BASE_DIM = {
    "mambavision_t": 80,
    "mambavision_t2": 80,
    "mambavision_s": 96,
    "mambavision_b": 128,
    "mambavision_l": 196,
    "mambavision_l2": 196,
}


def is_mambavision_backbone(backbone_name) -> bool:
    """Kiểm tra xem tên backbone có kích hoạt MambaVision hay không (False = dùng U-Net thuần CNN)."""
    if backbone_name is None:
        return False
    name = str(backbone_name).strip().lower()
    return name not in ("", "none", "null", "false", "unet", "cnn", "baseline")


class _MockMambaVisionBackbone(nn.Module):
    """
    Mô hình giả lập (Mock) mô phỏng chính xác cấu trúc 4 tầng phân cấp của MambaVision-T:
      - Stage 1: (B,  80, H/4,  W/4)
      - Stage 2: (B, 160, H/8,  W/8)
      - Stage 3: (B, 320, H/16, W/16)
      - Stage 4: (B, 640, H/32, W/32)
    Phục vụ kiểm thử tự động (`selftest`) trên mọi môi trường CPU/GPU offline.
    """
    def __init__(self, dims=(80, 160, 320, 640)):
        super().__init__()
        self.dims = list(dims)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[2]),
            nn.GELU(),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dims[3]),
            nn.GELU(),
        )

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        out_avg_pool = f4.mean(dim=(-2, -1))
        return out_avg_pool, [f1, f2, f3, f4]


class MambaVisionEncoder(nn.Module):
    """
    Bộ chuyển đổi (Adapter) trích xuất 4 tầng đặc trưng đa tỷ lệ từ MambaVision (NVIDIA)
    để truyền qua các nhánh Skip Connections của mạng U-Net.
    """
    def __init__(
        self,
        model_name="mambavision_t",
        in_channels=3,
        pretrained=True,
        freeze_backbone=False,
        normalize_imagenet=True,
        use_mock=False,
    ):
        super().__init__()
        self.model_name = str(model_name)
        self.in_channels = int(in_channels)
        self.pretrained = bool(pretrained)
        self.normalize_imagenet = bool(normalize_imagenet)
        self.use_mock = bool(use_mock)
        self._is_frozen = False

        # Nếu số lát cắt đầu vào khác 3 (VD: 1 hoặc 5 kênh), dùng Conv 1x1 ánh xạ về 3 kênh chuẩn
        if self.in_channels != 3:
            self.input_adapter = nn.Conv2d(self.in_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # Chuẩn hóa ảnh CT [0, 1] theo phân phối ImageNet mà MambaVision đã học
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        key = self.model_name.strip().lower()
        self.hf_repo = MAMBAVISION_REGISTRY.get(key, self.model_name)

        self._key = key
        self._is_mock = bool(self.use_mock or key == "mambavision_mock")
        if self._is_mock:
            self.backbone = _MockMambaVisionBackbone()
        else:
            self.backbone = self._load_real_mambavision(self.hf_repo, pretrained=self.pretrained)

        # Tự động dò số kênh của 4 stages [C1, C2, C3, C4]
        self.out_channels = self._probe_out_channels()
        self.stem_stride = 4

        if freeze_backbone:
            self.freeze()

    @staticmethod
    def _load_real_mambavision(hf_repo, pretrained=True):
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as e:
            raise ImportError(
                "Chưa cài đặt thư viện `transformers` để nạp MambaVision. Bản thật còn cần timm, einops và "
                "`mamba-ssm` (nhân CUDA, cần Linux + GPU NVIDIA): "
                "`pip install transformers timm einops mamba-ssm`. Muốn thử trên CPU hãy dùng use_mock=True."
            ) from e

        if not torch.cuda.is_available():
            print("[MambaVisionEncoder] ⚠️ Không có CUDA: MambaVision thật dùng nhân CUDA `selective_scan_fn` "
                  "nên sẽ KHÔNG chạy được forward trên CPU.")

        if pretrained:
            print(f"[MambaVisionEncoder] Đang tải Pre-trained Backbone từ: {hf_repo} ...")
            return AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
        else:
            print(f"[MambaVisionEncoder] Khởi tạo kiến trúc MambaVision từ đầu (scratch): {hf_repo} ...")
            cfg = AutoConfig.from_pretrained(hf_repo, trust_remote_code=True)
            return AutoModel.from_config(cfg, trust_remote_code=True)

    def _probe_out_channels(self):
        if not self._is_mock and self._key in MAMBAVISION_BASE_DIM:
            d = MAMBAVISION_BASE_DIM[self._key]
            return [d * 2 ** i for i in range(4)]

        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            param = next(self.backbone.parameters(), None)
            device = param.device if param is not None else torch.device("cpu")
            if not self._is_mock and device.type != "cuda":
                raise RuntimeError(
                    f"Không biết số kênh của '{self.model_name}' và không thể dò bằng forward ngoài CUDA. "
                    f"Hãy dùng một tên trong {sorted(MAMBAVISION_BASE_DIM)} hoặc tạo encoder trên GPU."
                )
            dummy = torch.zeros(1, 3, 128, 128, device=device)
            out = self.backbone(dummy)
            features = self._extract_feature_list(out)
            if len(features) != 4:
                raise RuntimeError(
                    f"MambaVisionEncoder yêu cầu đúng 4 tầng đặc trưng, nhưng nhận được {len(features)}"
                )
            channels = [int(f.shape[1]) for f in features]
        self.backbone.train(was_training)
        return channels

    @staticmethod
    def _extract_feature_list(backbone_output):
        """Trích xuất danh sách 4 feature maps từ đầu ra của MambaVision."""
        if isinstance(backbone_output, (tuple, list)) and len(backbone_output) == 2:
            _, features = backbone_output
            if isinstance(features, (tuple, list)):
                return list(features)
        if isinstance(backbone_output, dict) and "features" in backbone_output:
            return list(backbone_output["features"])
        if hasattr(backbone_output, "features"):
            return list(backbone_output.features)
        hs = getattr(backbone_output, "hidden_states", None)
        if hs is None and isinstance(backbone_output, dict):
            hs = backbone_output.get("hidden_states")
        if hs is not None:
            return list(hs)
        raise TypeError(
            f"Định dạng đầu ra của MambaVision không hợp lệ: {type(backbone_output)}. "
            f"Cần (pooled, [4 feature maps]) hoặc đối tượng có `features`/`hidden_states`."
        )

    # =========================================================================
    # CƠ CHẾ ĐÓNG / MỞ KHÓA TRỌNG SỐ (FREEZE / UNFREEZE BACKBONE)
    # =========================================================================
    def freeze(self):
        """
        Đóng băng trọng số lõi MambaVision (`requires_grad = False`) và khóa BatchNorm/LayerNorm
        ở chế độ `.eval()`. `input_adapter` (nếu in_channels != 3) vẫn được học; khi đó forward giữ đồ thị
        autograd xuyên qua backbone đóng băng nên tốn thêm bộ nhớ kích hoạt.
        """
        self._is_frozen = True
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        return self

    def unfreeze(self):
        """Mở khóa toàn bộ trọng số MambaVision (`requires_grad = True`) để Fine-tune toàn mạng."""
        self._is_frozen = False
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train(self.training)
        return self

    def set_trainable(self, trainable: bool):
        """Công tắc đóng/mở trọng số: True = Mở khóa (Unfreeze), False = Đóng băng (Freeze)."""
        return self.unfreeze() if trainable else self.freeze()

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    def train(self, mode: bool = True):
        """
        Đảm bảo khi `trainer.py` gọi `model.train()` ở đầu mỗi Epoch, nếu backbone đang bị
        đóng băng (`_is_frozen=True`), các tầng BatchNorm bên trong `self.backbone` vẫn giữ
        nguyên trạng thái `.eval()`, tránh làm hỏng thống kê Pre-trained.
        """
        super().train(mode)
        if self._is_frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        x = self.input_adapter(x)
        if self.normalize_imagenet:
            x = (x - self.img_mean) / self.img_std

        # Nếu `input_adapter` có tham số học được thì gradient PHẢI chảy ngược qua backbone (dù backbone
        # đóng băng), nên không được bọc no_grad. Chỉ khi phía trước không có gì để học mới tiết kiệm bộ nhớ.
        adapter_trainable = any(p.requires_grad for p in self.input_adapter.parameters())
        if self._is_frozen and not adapter_trainable:
            with torch.no_grad():
                out = self.backbone(x)
        else:
            out = self.backbone(x)

        return self._extract_feature_list(out)


def _cmd_selftest(_):
    print("Đang chạy kiểm thử tự động custom_backbone.py...")
    enc = MambaVisionEncoder(
        model_name="mambavision_t",
        in_channels=3,
        pretrained=False,
        freeze_backbone=False,
        use_mock=True,
    )
    assert enc.out_channels == [80, 160, 320, 640], f"Sai số kênh: {enc.out_channels}"

    x = torch.randn(2, 3, 128, 128)
    feats = enc(x)
    expected_shapes = [
        (2, 80, 32, 32),
        (2, 160, 16, 16),
        (2, 320, 8, 8),
        (2, 640, 4, 4),
    ]
    for i, (f, exp) in enumerate(zip(feats, expected_shapes)):
        assert tuple(f.shape) == exp, f"Stage {i+1} sai shape: {tuple(f.shape)} != {exp}"

    # Kiểm tra cơ chế Đóng băng (Freeze)
    enc.freeze()
    assert enc.is_frozen is True
    enc.train(True)  # Giả lập trainer gọi model.train()
    assert enc.backbone.training is False, "Khi freeze, backbone bắt buộc phải ở chế độ eval()"
    assert all(not p.requires_grad for p in enc.backbone.parameters())

    # Kiểm tra cơ chế Mở khóa (Unfreeze)
    enc.unfreeze()
    assert enc.is_frozen is False
    assert enc.backbone.training is True
    assert all(p.requires_grad for p in enc.backbone.parameters())

    # --- Các ca hồi quy: lỗi gradient adapter, BN khi freeze, unfreeze khi eval, kích thước lẻ ---
    head = nn.Conv2d(80, 1, 1)
    e5 = MambaVisionEncoder(in_channels=5, pretrained=False, freeze_backbone=True, use_mock=True)
    head(e5(torch.randn(2, 5, 64, 64))[0]).mean().backward()
    g = e5.input_adapter.weight.grad
    assert g is not None and g.abs().sum() > 0, "Freeze + in_channels!=3: input_adapter phải nhận gradient"
    assert all(p.grad is None for p in e5.backbone.parameters()), "Backbone đóng băng không được có gradient"

    ef = MambaVisionEncoder(pretrained=False, freeze_backbone=True, use_mock=True).train()
    before = [m.running_mean.clone() for m in ef.modules() if isinstance(m, nn.BatchNorm2d)]
    ef(torch.randn(2, 3, 64, 64))
    after = [m.running_mean for m in ef.modules() if isinstance(m, nn.BatchNorm2d)]
    assert all(torch.equal(a, b) for a, b in zip(before, after)), "Freeze phải giữ nguyên thống kê BatchNorm"

    ef.eval().unfreeze()
    assert ef.backbone.training is False, "unfreeze() khi module đang eval không được ép backbone về train"

    for size in (96, 100, 512):
        out = MambaVisionEncoder(pretrained=False, use_mock=True)(torch.randn(1, 3, size, size))
        assert len(out) == 4 and out[0].shape[-1] == size // 4

    class _HFLike:  # mô phỏng ModelOutput của Hugging Face
        hidden_states = [torch.zeros(1, c, 2, 2) for c in (80, 160, 320, 640)]
    assert len(MambaVisionEncoder._extract_feature_list(_HFLike())) == 4
    assert [MAMBAVISION_BASE_DIM["mambavision_t"] * 2 ** i for i in range(4)] == enc.out_channels

    print("✅ selftest custom_backbone.py: OK!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest", help="Kiểm thử tự động MambaVisionEncoder và cơ chế Freeze/Unfreeze")
    s.set_defaults(fn=_cmd_selftest)
    args = ap.parse_args()
    args.fn(args)