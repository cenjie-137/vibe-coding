# -*- coding: utf-8 -*-
"""DRIVE -> nnU-Net v2 格式转换（本地，无损）。

从 medseg_data.zip 提取 DRIVE 眼底血管数据：
  training/images/ 21~40_training.tif  -> imagesTr/<case>_0000.png (灰度)
  training/1st_manual/ 21~40_manual1.gif -> labelsTr/<case>.png (二值 0/1，255->1)
  test/images/ 01~20_test.tif          -> imagesTs/<case>_0000.png (预测用，无标签)
生成 Dataset071_DRIVE。
"""
import io
import json
import zipfile
from pathlib import Path

from PIL import Image
import numpy as np

ZIP = Path(r"c:\Users\32201\Desktop\T\vibe coding\DRIVE\medseg_data.zip")
OUT = Path(r"c:\Users\32201\Desktop\T\vibe coding\DRIVE\data\Dataset071_DRIVE")

for d in ("imagesTr", "labelsTr", "imagesTs"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

z = zipfile.ZipFile(ZIP)
name_to_data = {n: z.read(n) for n in z.namelist()}

train_ids = list(range(21, 41))   # 21..40
test_ids = list(range(1, 21))     # 1..20


def read_img(name):
    return Image.open(io.BytesIO(name_to_data[name]))


sizes = {}
for i in train_ids:
    case = f"DRIVE_{i:02d}"
    img = read_img(f"DRIVE/training/images/{i}_training.tif").convert("L")
    img.save(OUT / "imagesTr" / f"{case}_0000.png")
    sizes[img.size] = sizes.get(img.size, 0) + 1

    msk = np.array(read_img(f"DRIVE/training/1st_manual/{i}_manual1.gif"))
    Image.fromarray((msk > 0).astype(np.uint8)).save(OUT / "labelsTr" / f"{case}.png")

for i in test_ids:
    case = f"DRIVE_test_{i:02d}"
    img = read_img(f"DRIVE/test/images/{i:02d}_test.tif").convert("L")
    img.save(OUT / "imagesTs" / f"{case}_0000.png")

dataset = {
    "channel_names": {"0": "GRAYSCALE"},
    "labels": {"background": 0, "vessel": 1},
    "numTraining": len(train_ids),
    "file_ending": ".png",
}
(OUT / "dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")

print(f"train={len(train_ids)}  test={len(test_ids)}")
print("train 尺寸分布 (W,H -> 数量):")
for s, n in sorted(sizes.items(), key=lambda kv: kv[0][0] * kv[0][1]):
    print(f"  {s[0]}x{s[1]} -> {n}")
print("完成:", OUT)