"""
predict.py
----------
Inference (suy luận) trên ảnh mới (hỗ trợ cả 1 file đơn lẻ hoặc cả thư mục ảnh):
- Áp dụng đúng tiền xử lý như validation/test
- Đưa mask dự đoán về lại độ phân giải gốc của bức ảnh đầu vào
- Lưu file binary mask (.png) và ảnh overlay màu (.png)
"""

import os
import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import yaml
import cv2
import numpy as np
import torch
from tqdm import tqdm

from models import build_model
from utils.transforms import get_val_transforms

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dự đoán mask phân đoạn trên ảnh mới")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn đến 1 file ảnh hoặc 1 thư mục chứa ảnh",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions",
        help="Thư mục lưu mask và ảnh overlay dự đoán",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pth",
        help="Đường dẫn file trọng số .pth",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Đường dẫn file cấu hình yaml",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng phân loại nhị phân (mặc định 0.5)",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Lọc ảnh theo mã bệnh nhân (ví dụ: 'LIDC-IDRI-0002' hoặc '0002')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số lượng ảnh cần dự đoán (ví dụ: 5 hoặc 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ngẫu nhiên khi giới hạn số lượng ảnh (mặc định 42)",
    )
    return parser.parse_args()


def get_image_files(input_path: Path, patient: str = None, limit: int = None, seed: int = 42) -> List[Path]:
    """Lấy danh sách các file ảnh hợp lệ, có thể lọc theo mã bệnh nhân và lấy mẫu ngẫu nhiên trải qua nhiều bệnh nhân."""
    if input_path.is_file():
        if input_path.suffix.lower() in VALID_EXTS:
            files = [input_path]
        else:
            raise ValueError(f"Định dạng file không được hỗ trợ: {input_path}")
    elif input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.suffix.lower() in VALID_EXTS]
        if not files:
            raise FileNotFoundError(f"Không tìm thấy file ảnh hợp lệ nào trong: {input_path}")
        files = sorted(files)
    else:
        raise FileNotFoundError(f"Đường dẫn đầu vào không tồn tại: {input_path}")

    # Lọc theo bệnh nhân nếu có yêu cầu
    if patient:
        patient_lower = patient.strip().lower()
        filtered = [f for f in files if patient_lower in f.stem.lower()]
        if not filtered:
            available_samples = [f.stem.split('_')[0] for f in files[:10]]
            raise ValueError(
                f"Không tìm thấy ảnh nào khớp với mã bệnh nhân '{patient}' trong thư mục {input_path}.\n"
                f"Một số mã bệnh nhân có sẵn: {list(dict.fromkeys(available_samples))}"
            )
        files = filtered

    # Giới hạn số lượng mẫu: Lấy mẫu ngẫu nhiên có seed, trải qua nhiều bệnh nhân
    if limit is not None and limit > 0 and len(files) > limit:
        rng = random.Random(seed)

        # Gom nhóm ảnh theo từng bệnh nhân (ví dụ: LIDC-IDRI-0001_slice085 -> LIDC-IDRI-0001)
        patient_groups: Dict[str, List[Path]] = defaultdict(list)
        for f in files:
            pid = f.stem.rsplit('_', 1)[0] if '_' in f.stem else f.stem
            patient_groups[pid].append(f)

        # Xáo trộn ảnh bên trong từng bệnh nhân
        for pid in patient_groups:
            rng.shuffle(patient_groups[pid])

        # Xáo trộn thứ tự các bệnh nhân để phân bố đều
        patients = sorted(list(patient_groups.keys()))
        rng.shuffle(patients)

        # Lấy mẫu round-robin lần lượt qua từng bệnh nhân cho đến khi đủ limit
        selected: List[Path] = []
        idx = 0
        while len(selected) < limit:
            added_any = False
            for pid in patients:
                if idx < len(patient_groups[pid]):
                    selected.append(patient_groups[pid][idx])
                    added_any = True
                    if len(selected) == limit:
                        break
            if not added_any:
                break
            idx += 1

        files = sorted(selected)

    return files


@torch.no_grad()
def main() -> None:
    args = parse_args()

    # 1. Đọc config & chuẩn bị thiết bị
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Khởi tạo mô hình
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {ckpt_path}")

    model = build_model(cfg).to(device)
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Lấy danh sách ảnh cần dự đoán
    input_path = Path(args.input)
    image_files = get_image_files(input_path, patient=args.patient, limit=args.limit, seed=args.seed)
    filter_info = []
    if args.patient:
        filter_info.append(f"bệnh nhân '{args.patient}'")
    if args.limit:
        filter_info.append(f"lấy mẫu ngẫu nhiên {args.limit} ảnh trải qua nhiều bệnh nhân (seed={args.seed})")
    filter_str = f" ({', '.join(filter_info)})" if filter_info else ""
    print(f"🔍 Tìm thấy {len(image_files)} ảnh cần dự đoán{filter_str}.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # 4. Pipeline tiền xử lý
    image_size = cfg["data"].get("image_size", 256)
    n_channels = cfg["model"].get("n_channels", 3)
    mean = cfg["data"].get("mean", [0.5, 0.5, 0.5])
    std = cfg["data"].get("std", [0.5, 0.5, 0.5])

    val_transform = get_val_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        max_pixel_value=255.0,
        resize=not cfg["data"].get("native", False),
    )

    print("🚀 Đang tiến hành suy luận...")
    for img_file in tqdm(image_files, desc="[Predict]"):
        # Đọc ảnh gốc
        if img_file.suffix.lower() == ".npy":
            raw_arr = np.load(str(img_file))
            if raw_arr.ndim == 2:
                raw_arr = raw_arr[:, :, np.newaxis]
            if n_channels == 3 and raw_arr.shape[-1] == 1:
                raw_arr = np.repeat(raw_arr, 3, axis=-1)
            orig_img = raw_arr.astype(np.float32)
            if orig_img.max() <= 1.0:
                orig_img = (orig_img * 255.0).astype(np.uint8)
            else:
                orig_img = orig_img.astype(np.uint8)
        else:
            if n_channels == 1:
                orig_img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if orig_img is None:
                    continue
                orig_img = orig_img[:, :, np.newaxis]
            else:
                orig_img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
                if orig_img is None:
                    continue
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = orig_img.shape[:2]

        # Tiền xử lý với Albumentations
        augmented = val_transform(image=orig_img.astype(np.float32))
        inp = augmented["image"]

        if inp.ndim == 2:
            inp_tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float()
        else:
            inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()

        inp_tensor = inp_tensor.to(device)

        # Suy luận
        logits = model(inp_tensor)

        if cfg["model"].get("n_classes", 1) == 1:
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_mask = (probs > args.threshold).astype(np.uint8)
        else:
            pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        # Đưa kích thước mask về lại độ phân giải gốc của ảnh ban đầu
        pred_mask_resized = cv2.resize(
            pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )

        # Lưu file mask nhị phân (0 và 255)
        mask_save_path = mask_dir / f"{img_file.stem}_mask.png"
        cv2.imwrite(str(mask_save_path), pred_mask_resized * 255)

        # Tạo và lưu ảnh overlay
        disp_orig = orig_img.copy()
        if disp_orig.shape[-1] == 1:
            disp_orig = cv2.cvtColor(disp_orig, cv2.COLOR_GRAY2BGR)
        else:
            disp_orig = cv2.cvtColor(disp_orig, cv2.COLOR_RGB2BGR)

        overlay = disp_orig.copy()
        overlay[pred_mask_resized > 0] = [0, 0, 255]  # Màu đỏ cho vùng u dự đoán
        blended = cv2.addWeighted(disp_orig, 0.7, overlay, 0.3, 0)

        overlay_save_path = overlay_dir / f"{img_file.stem}_overlay.png"
        cv2.imwrite(str(overlay_save_path), blended)

    print(f"\n🎉 Dự đoán hoàn tất! Kết quả được lưu tại: {output_dir}")
    print(f"  - Thư mục Masks    : {mask_dir}")
    print(f"  - Thư mục Overlays : {overlay_dir}")


if __name__ == "__main__":
    main()
