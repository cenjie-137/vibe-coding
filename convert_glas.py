"""convert_glas.py — GLaS 原始数据 → 统一格式 (Step 1)

原始结构 (Warwick_QU_Dataset/):
    train_N.bmp        train_N_anno.bmp      (85)
    testA_N.bmp        testA_N_anno.bmp      (60)
    testB_N.bmp        testB_N_anno.bmp      (20)
    Grade.csv          name, patient ID, grade(GlaS) benign/malignant, grade(Sirinukunwattana)

输出 (统一格式):
    images/            RGB PNG
    masks/             二值掩膜 0/255 (255=腺体)
    metadata.csv       name, split, grade_glas, grade_sirinuk
    splits.json        固定划分存档

用法:
    python convert_glas.py --src path/to/Warwick_QU_Dataset --dst data/glas
    python convert_glas.py --src path/to/archive.zip --dst data/glas
"""
import argparse
import csv
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "testA", "testB")
IMAGE_RE = re.compile(r"^(train|testA|testB)_(\d+)$")
ANNO_RE = re.compile(r"^(train|testA|testB)_(\d+)_anno$")


def locate_dataset(src: Path) -> Path:
    """返回包含图像与 Grade.csv 的目录（支持直接传 zip）。"""
    if src.is_file() and src.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="glas_raw_"))
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
        # zip 内通常有一层 Warwick_QU_Dataset/
        candidates = [tmp] + [p for p in tmp.rglob("Warwick_QU_Dataset") if p.is_dir()]
        for c in candidates:
            if (c / "Grade.csv").exists():
                return c
        raise FileNotFoundError(f"zip 内未找到 Grade.csv: {src}")
    if src.is_dir():
        if (src / "Grade.csv").exists():
            return src
        nested = src / "Warwick_QU_Dataset"
        if nested.is_dir() and (nested / "Grade.csv").exists():
            return nested
    raise FileNotFoundError(f"未找到数据集目录或 Grade.csv: {src}")


def load_grade(csv_path: Path) -> dict:
    """name -> dict(grade_glas, grade_sirinuk)"""
    grades = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            name = row["name"].strip()
            grades[name] = {
                "grade_glas": row["grade (GlaS)"].strip(),
                "grade_sirinuk": row["grade (Sirinukunwattana et al. 2015)"].strip(),
            }
    return grades


def convert(src: Path, dst: Path) -> None:
    dataset_dir = locate_dataset(src)
    grades = load_grade(dataset_dir / "Grade.csv")

    img_dir = dst / "images"
    mask_dir = dst / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    splits = {s: [] for s in SPLITS}
    splits["benign"] = []
    splits["malignant"] = []
    sizes_w, sizes_h = [], []
    fg_fracs = []

    # 遍历每个 split 的图像（排除 _anno 文件）
    for split in SPLITS:
        images = sorted(
            p for p in dataset_dir.glob(f"{split}_*.bmp")
            if not p.name.endswith("_anno.bmp")
        )
        if not images:
            print(f"[warn] split '{split}' 未找到图像", file=sys.stderr)
            continue
        for img_path in images:
            m = IMAGE_RE.match(img_path.stem)
            assert m, f"无法解析图像名: {img_path.name}"
            name = img_path.stem
            anno_path = dataset_dir / f"{name}_anno.bmp"
            if not anno_path.exists():
                print(f"[warn] 缺少标注，跳过: {name}", file=sys.stderr)
                continue

            img = Image.open(img_path).convert("RGB")
            anno = Image.open(anno_path).convert("L")
            mask = (np.array(anno) > 0).astype(np.uint8) * 255

            w, h = img.size
            sizes_w.append(w)
            sizes_h.append(h)
            fg_fracs.append(float((mask > 0).mean()))

            img.save(img_dir / f"{name}.png")
            Image.fromarray(mask, mode="L").save(mask_dir / f"{name}.png")

            g = grades.get(name, {"grade_glas": "", "grade_sirinuk": ""})
            metadata.append({
                "name": name,
                "split": split,
                "grade_glas": g["grade_glas"],
                "grade_sirinuk": g["grade_sirinuk"],
            })
            splits[split].append(name)
            if g["grade_glas"] == "benign":
                splits["benign"].append(name)
            elif g["grade_glas"] == "malignant":
                splits["malignant"].append(name)

    # 写 metadata.csv
    with (dst / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "split", "grade_glas", "grade_sirinuk"]
        )
        writer.writeheader()
        writer.writerows(metadata)

    # 写 splits.json
    with (dst / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    # 打印统计
    sizes_w = np.array(sizes_w)
    sizes_h = np.array(sizes_h)
    fg_fracs = np.array(fg_fracs)
    print("=" * 60)
    print(f"转换完成，共 {len(metadata)} 张图像 → {dst}")
    print(f"  train : {len(splits['train'])}    (benign {sum(1 for n in splits['train'] if n in splits['benign'])}, malignant {sum(1 for n in splits['train'] if n in splits['malignant'])})")
    print(f"  testA : {len(splits['testA'])}    (benign {sum(1 for n in splits['testA'] if n in splits['benign'])}, malignant {sum(1 for n in splits['testA'] if n in splits['malignant'])})")
    print(f"  testB : {len(splits['testB'])}    (benign {sum(1 for n in splits['testB'] if n in splits['benign'])}, malignant {sum(1 for n in splits['testB'] if n in splits['malignant'])})")
    print(f"  图像尺寸 W: {sizes_w.min()}~{sizes_w.max()} (mean {sizes_w.mean():.0f})")
    print(f"  图像尺寸 H: {sizes_h.min()}~{sizes_h.max()} (mean {sizes_h.mean():.0f})")
    print(f"  前景(腺体)占比: {fg_fracs.min()*100:.1f}% ~ {fg_fracs.max()*100:.1f}% (mean {fg_fracs.mean()*100:.1f}%)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="GLaS 数据转换 (Step 1)")
    parser.add_argument("--src", required=True, help="Warwick_QU_Dataset 目录或 archive.zip")
    parser.add_argument("--dst", required=True, help="输出目录，如 data/glas")
    args = parser.parse_args()
    convert(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()