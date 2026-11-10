import shutil, pandas as pd
from pathlib import Path

P = Path("processed")
df = pd.read_csv(P / "slices.csv")
OUT = Path("kaggle_subset")
(OUT).mkdir(exist_ok=True)

# Số bệnh nhân mỗi split (tổng 200)
quota = {"train": 140, "val": 30, "test": 30}
pids = []
for sp, n in quota.items():
    ids = df[df["split"] == sp]["patient_id"].drop_duplicates()
    pids += list(ids.sample(min(n, len(ids)), random_state=0))

# Copy 2 file mỗi bệnh nhân, để .npy nằm thẳng ở gốc (không có thư mục con)
for i, p in enumerate(pids, 1):
    for suf in ("img", "votes"):
        dst = OUT / f"{p}_{suf}.npy"
        if not dst.exists():
            shutil.copy(P / "volumes" / f"{p}_{suf}.npy", dst)
    if i % 20 == 0:
        print(f"{i}/{len(pids)}")

# slices.csv chỉ gồm 200 bệnh nhân này, để riêng (không nằm trong dataset ảnh)
Path("kaggle_meta").mkdir(exist_ok=True)
df[df["patient_id"].isin(pids)].to_csv("kaggle_meta/slices.csv", index=False)

total = sum(f.stat().st_size for f in OUT.glob("*.npy")) / 1e9
print(f"Xong: {len(pids)} bệnh nhân, {total:.1f} GB")