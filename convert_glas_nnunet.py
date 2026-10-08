# -*- coding: utf-8 -*-
"""GLaS -> nnU-Net v2 格式转换（本地，无损）。

生成 Dataset070_GLAS：
  imagesTr/<case>_0000.png   85 张灰度图（train）
  labelsTr/<case>.png        85 张二值掩码 0/1
  imagesTs/<case>_0000.png   testA(60)+testB(20) 用于预测
  dataset.json
"""
import json
from pathlib import Path

from PIL import Image
import numpy as np

SRC = Path(r"c:\Users\32201\Desktop\T\vibe coding\DRIVE\data\glas")
OUT = Path(r"c:\Users\32201\Desktop\T\vibe coding\DRIVE\data\Dataset070_GLAS")

splits = json.loads((SRC / "splits.json").read_text(encoding="utf-8"))
train = splits["train"]
testA = splits["testA"]
testB = splits["testB"]

for d in ("imagesTr", "labelsTr", "imagesTs"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

for c in train:
    Image.open(SRC / "images" / f"{c}.png").convert("L").save(OUT / "imagesTr" / f"{c}_0000.png")
    msk = np.array(Image.open(SRC / "masks" / f"{c}.png"))
    Image.fromarray((msk > 0).astype(np.uint8)).save(OUT / "labelsTr" / f"{c}.png")

for c in testA + testB:
    Image.open(SRC / "images" / f"{c}.png").convert("L").save(OUT / "imagesTs" / f"{c}_0000.png")

dataset = {
    "channel_names": {"0": "GRAYSCALE"},
    "labels": {"background": 0, "gland": 1},
    "numTraining": len(train),
    "file_ending": ".png",
}
(OUT / "dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")

# 自检：尺寸分布
sizes = {}
for c in train:
    s = Image.open(SRC / "images" / f"{c}.png").size
    sizes[s] = sizes.get(s, 0) + 1
print(f"train={len(train)}  testA={len(testA)}  testB={len(testB)}")
print("train 图像尺寸分布 (W,H -> 数量):")
for s, n in sorted(sizes.items(), key=lambda kv: kv[0][0] * kv[0][1]):
    print(f"  {s[0]}x{s[1]} -> {n}")
print("完成:", OUT)