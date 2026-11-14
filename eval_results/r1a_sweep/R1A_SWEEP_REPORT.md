# Báo Cáo Quét Ngưỡng 3D [R1A]
- Thời gian: 2026-11-14 16:23:50
- Checkpoint: `/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/mamba_unet/runs/mamba_colab_500item_25epoch_extra_test_R1a_Img448_Area5/best_model.pth`
- Số ca Val: 75 (Ca không u: 11)

## 1. Các mốc nổi bật trong lưới
- Đỉnh Dice 3D (Th=0.8, min_area=10.0 mm², min_vol=20.0 mm³): Dice=0.3691 | Prec=0.3952 | Rec=0.4014 | FP/ca=4.61 | Ca sạch=1/11 | Nốt=122/203 (60.1%)
- Mốc eval3d mặc định (Th=0.5, A>=5, V>=0): Dice=0.3644 | Prec=0.3826 | Rec=0.4188 | FP/ca=8.15 | Ca sạch=1/11 | Nốt=137/203 (67.5%)
- Mốc chuẩn R0b (Th=0.6, A>=6, V>=10): Dice=0.3661 | Prec=0.3943 | Rec=0.4146 | FP/ca=6.56 | Ca sạch=1/11 | Nốt=131/203 (64.5%)

## 2. Độ nhạy theo dải đường kính tại Đỉnh Dice (Th=0.8, A>=10.0, V>=20.0)
- `[0-5) mm`: 12/59 (20.3%)
- `[5-10) mm`: 71/99 (71.7%)
- `[10-20) mm`: 31/37 (83.8%)
- `>=20 mm`: 8/8 (100.0%)

## 3. Độ nhạy theo dải đường kính tại Mốc chuẩn R0b (Th=0.6, A>=6, V>=10)
- `[0-5) mm`: 15/59 (25.4%)
- `[5-10) mm`: 77/99 (77.8%)
- `[10-20) mm`: 31/37 (83.8%)
- `>=20 mm`: 8/8 (100.0%)
