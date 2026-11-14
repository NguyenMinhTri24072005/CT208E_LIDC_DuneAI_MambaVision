# Báo Cáo Quét Ngưỡng 3D [R1B]
- Thời gian: 2026-11-14 16:04:57
- Checkpoint: `/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/mamba_unet/runs/mamba_colab_500item_50epoch_R1b_NegDynamic/best_model.pth`
- Số ca Val: 75 (Ca không u: 11)

## 1. Các mốc nổi bật trong lưới
- Đỉnh Dice 3D (Th=0.8, min_area=5.0 mm², min_vol=20.0 mm³): Dice=0.3313 | Prec=0.3380 | Rec=0.3977 | FP/ca=5.68 | Ca sạch=0/11 | Nốt=118/203 (58.1%)
- Mốc eval3d mặc định (Th=0.5, A>=5, V>=0): Dice=0.3271 | Prec=0.3224 | Rec=0.4165 | FP/ca=9.75 | Ca sạch=0/11 | Nốt=129/203 (63.5%)
- Mốc chuẩn R0b (Th=0.6, A>=6, V>=10): Dice=0.3278 | Prec=0.3261 | Rec=0.4115 | FP/ca=7.83 | Ca sạch=0/11 | Nốt=125/203 (61.6%)

## 2. Độ nhạy theo dải đường kính tại Đỉnh Dice (Th=0.8, A>=5.0, V>=20.0)
- `[0-5) mm`: 6/59 (10.2%)
- `[5-10) mm`: 73/99 (73.7%)
- `[10-20) mm`: 31/37 (83.8%)
- `>=20 mm`: 8/8 (100.0%)

## 3. Độ nhạy theo dải đường kính tại Mốc chuẩn R0b (Th=0.6, A>=6, V>=10)
- `[0-5) mm`: 11/59 (18.6%)
- `[5-10) mm`: 75/99 (75.8%)
- `[10-20) mm`: 31/37 (83.8%)
- `>=20 mm`: 8/8 (100.0%)
