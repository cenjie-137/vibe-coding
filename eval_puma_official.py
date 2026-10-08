"""eval_puma_official.py — PUMA 官方 micro Dice 评估（严格按官方定义实现）

来源：PUMA 数据集/基线论文
  · La Barbera et al., "A novel dataset for nuclei and tissue segmentation in
    melanoma with baseline nuclei segmentation and tissue segmentation benchmarks"
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11837757/
    （预印本：https://www.medrxiv.org/content/10.1101/2024.10.07.24315039v1.full）

官方原文（micro Dice 的定义，逐字）：
  "For evaluation of segmentation, the Dice score was computed for each class per
   sample and averaged across all samples (referred to as average Dice).
   Additionally, a Dice score was calculated per class on a concatenated sample.
   To create this sample, all images were combined along 1 axis, resulting in a
   single large image with the width of 1 image and a length equal to the number
   of images × the height of 1 image. This is referred to as the micro Dice."

即：**把所有图沿一个轴拼成一张大图（数据集级池化），在这张拼接图上逐类算 Dice，
再对类别取均值** —— 这就是官方 micro Dice。判定用的就是它（datasets.yaml: PUMA.metric=micro_dice）。

一个官方明确规定、必须照做的边界情形：
  当某类的 GT 与预测**都为空**时，Dice 数学上无定义，官方约定记为 **1.0**
  （见同批 PUMA 论文/评估说明）。漏掉这条会把只在少数图里出现的类别分数算错。

类别编号（与 datasets_puma.py / 官方 GeoJSON 一致，掩码像素值 = 类别号）：
  0 background · 1 stroma · 2 blood_vessel · 3 tumor · 4 epidermis · 5 necrosis
micro Dice 在 1..5 这 5 个前景类上计算，**不含背景**（与官方及本项目训练期 evaluate_puma 一致）。

用法：
    python eval_puma_official.py --gt_dir <GT 掩码目录> --pred_dir <预测掩码目录>
    python eval_puma_official.py --selftest        # 合成用例，验证算法与边界情形
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

NUM_CLASSES = 6
CLASS_NAMES = ["background", "stroma", "blood_vessel", "tumor", "epidermis", "necrosis"]
FOREGROUND = range(1, NUM_CLASSES)   # 1..5，micro Dice 只看前景类


# ── 读掩码 ────────────────────────────────────────────────────────
def _read_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def find_mask(directory: Path, stem: str):
    for ext in (".png", ".tif", ".tiff", ".bmp"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ── 官方 micro Dice 的纯函数实现（可被 selftest 直接调用）──────────
def class_dice(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    """单类 Dice：union==0（两边都空）→ 官方记 1.0；否则 2|P∩G|/(|P|+|G|)。"""
    p = pred == c
    g = gt == c
    inter = int(np.logical_and(p, g).sum())
    union = int(p.sum()) + int(g.sum())
    if union == 0:
        return 1.0
    if inter == 0:
        return 0.0
    return 2.0 * inter / union


def evaluate_masks(gt_masks: list, pred_masks: list) -> dict:
    """数据集级池化：先把所有图的逐类交/并累加（= 拼接样本），再逐类算 Dice。

    返回 {micro_dice, macro_dice, per_class:{name:dice}, n}。
    """
    assert len(gt_masks) == len(pred_masks) and gt_masks, "GT 与预测数量必须一致且非空"
    inter = np.zeros(NUM_CLASSES, dtype=np.float64)
    union = np.zeros(NUM_CLASSES, dtype=np.float64)
    per_sample = []          # 每张图自己的 micro（供 macro 用）
    for pred, gt in zip(pred_masks, gt_masks):
        if pred.shape != gt.shape:
            raise ValueError(f"预测与 GT 尺寸不一致：{pred.shape} vs {gt.shape}")
        s_inter, s_union = np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)
        for c in FOREGROUND:
            p, g = (pred == c), (gt == c)
            s_inter[c] = int(np.logical_and(p, g).sum())
            s_union[c] = int(p.sum()) + int(g.sum())
            inter[c] += s_inter[c]
            union[c] += s_union[c]
        # 逐图 micro（仅统计用；官方判定指标是池化后的 micro）
        d = [1.0 if s_union[c] == 0 else (0.0 if s_inter[c] == 0 else 2.0 * s_inter[c] / s_union[c])
             for c in FOREGROUND]
        per_sample.append(float(np.mean(d)))

    micro = [1.0 if union[c] == 0 else (0.0 if inter[c] == 0 else 2.0 * inter[c] / union[c])
             for c in FOREGROUND]
    per_class = {CLASS_NAMES[c]: float(micro[i]) for i, c in enumerate(FOREGROUND)}
    return {
        "micro_dice": float(np.mean(micro)),
        "macro_dice": float(np.mean(per_sample)),
        "per_class": per_class,
        "n": len(gt_masks),
    }


# ── 目录级评估 ────────────────────────────────────────────────────
def run_dirs(gt_dir: Path, pred_dir: Path, split: str) -> dict:
    names = sorted(p.stem for p in gt_dir.glob("*.png"))
    if not names:
        raise FileNotFoundError(f"{gt_dir} 下没找到任何 *.png GT 掩码")
    gts, preds, used = [], [], []
    for n in names:
        pp = find_mask(pred_dir, n)
        if pp is None:
            continue
        gts.append(_read_mask(gt_dir / f"{n}.png"))
        preds.append(_read_mask(pp))
        used.append(n)
    if not used:
        raise FileNotFoundError(f"{pred_dir} 下没有任何与 GT 同名的预测掩码")
    res = evaluate_masks(gts, preds)
    res["split"] = split
    res["used"] = used

    print("=" * 62)
    print("PUMA 官方协议（micro Dice = 拼接样本上逐类 Dice，再对 5 个前景类取均值）")
    print("来源：https://pmc.ncbi.nlm.nih.gov/articles/PMC11837757/")
    print("=" * 62)
    print(f"  {split:<10s} n={res['n']:3d}  microDice={res['micro_dice']:.4f}  "
          f"macroDice={res['macro_dice']:.4f}")
    for c in FOREGROUND:
        print(f"    {CLASS_NAMES[c]:<14s} dice={res['per_class'][CLASS_NAMES[c]]:.4f}")
    print("=" * 62)
    return res


# ── 自检 ──────────────────────────────────────────────────────────
def selftest() -> int:
    """合成用例：精确可手算，覆盖「部分重叠」「完全不重叠」「两边都空」三种边界。"""
    ok = 0
    cases = []

    # 用例：2 张 4×4 图
    gt1 = np.array([[0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [2, 2, 0, 0],
                    [2, 2, 0, 0]])
    # 预测的 class1 与 GT 完全一致（dice=1）；class2 整体平移、与 GT 零重叠（dice=0）
    pd1 = np.array([[0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 2, 2],
                    [0, 0, 2, 2]])
    gt2 = np.array([[3, 3, 0, 0],
                    [3, 3, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    pd2 = gt2.copy()   # class3 完全命中（dice=1）
    # class4/5 两图皆空 → 官方记 1.0
    # micro = mean(class1=1, class2=0, class3=1, class4=1, class5=1) = 0.8
    cases.append(("手算 micro=0.8（含空类=1.0 与零重叠=0.0）",
                  evaluate_masks([gt1, gt2], [pd1, pd2])["micro_dice"], 0.8))

    # 全对：预测 == GT → micro=1.0
    cases.append(("全对 micro=1.0",
                  evaluate_masks([gt1, gt2], [gt1, gt2])["micro_dice"], 1.0))

    # 整张漏检（预测全 0，GT 有 class1/class2）→ 两类 dice=0，其余三类空→1.0 → micro=3/5=0.6
    zeros = np.zeros_like(gt1)
    cases.append(("整张漏检 micro=0.6",
                  evaluate_masks([gt1], [zeros])["micro_dice"], 0.6))

    # 两边都空（GT 全 0、预测全 0）→ 5 个前景类全空 → micro=1.0
    cases.append(("全空图 micro=1.0（官方约定空对空记 1）",
                  evaluate_masks([zeros], [zeros])["micro_dice"], 1.0))

    for name, got, want in cases:
        passed = abs(got - want) < 1e-9
        ok += passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}：got={got:.6f} want={want:.6f}")

    print(f"\neval_puma_official selftest: {ok}/{len(cases)} passed")
    return 0 if ok == len(cases) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir")
    ap.add_argument("--pred_dir")
    ap.add_argument("--split", default="val", help="子集名（PUMA 判定子集为 val）")
    ap.add_argument("--out_json", default=None, help="可选：把结果写成 JSON")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not (args.gt_dir and args.pred_dir):
        ap.error("需要 --gt_dir 与 --pred_dir（或 --selftest）")

    res = run_dirs(Path(args.gt_dir), Path(args.pred_dir), args.split)
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
