# Báo Cáo Đối Chứng Run 0b (Baseline Bộ Lọc Mới)
- Thời gian: 2026-11-13 17:58:47
- Checkpoint: `/content/drive/MyDrive/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/SOURCES/mamba_unet/runs/mamba_colab_500item_25epoch_extra_test/best_model.pth`
- Số ca Val: 75

## 1. Cấu hình đỉnh Dice 3D trong lưới
- Th=0.6, min_area=20.0 mm², min_vol=20.0 mm³ → Dice=0.3774, Precision=0.4148, Recall=0.4021, FP/ca=2.83, Nốt=97/203 (47.8%)

## 2. Mốc đối chứng chuẩn cho Run 1
- Bộ lọc cũ (Th=0.6, A>=20, V>=0): Dice=0.3771, Prec=0.4143, Rec=0.4021, FP/ca=2.91, Nốt=97/203 (47.8%)
- Bộ lọc mới (Th=0.6, A>=6, V>=10): Dice=0.3479, Prec=0.3863, Rec=0.3850, FP/ca=5.15, Nốt=116/203 (57.1%)

### Độ nhạy theo dải đường kính tại (Th=0.6, A>=6, V>=10)
- `[0-5) mm`: 6/59 (10.2%)
- `[5-10) mm`: 71/99 (71.7%)
- `[10-20) mm`: 31/37 (83.8%)
- `>=20 mm`: 8/8 (100.0%)
