# -*- coding: utf-8 -*-
"""audit_glas_metric_protocol.py — GLaS 对象级 Dice 的「口径对照表」。

同一批预测，在不同评估约定下会得到不同的 ObjDice。这里把四个维度的组合全部枚举，
量化"口径"对分数的影响，用于校准：U-Node 论文报告值属于哪种约定？我们的
eval_glas_official.py 又属于哪种？两者差多少？

四个维度：
  连通性  connectivity ∈ {4, 8}        对象连通性（scipy.ndimage.label 结构元）
  匹配    match        ∈ {argmax, mode}
          argmax: 每个分割对象匹配"重叠面积最大"的 GT 对象
          mode  : 每个分割对象匹配"内部出现次数最多"的 GT 标签（官方 MATLAB / U-Node 口径）
  加权    weight       ∈ {weighted, unweighted}
          weighted  : 按对象面积加权（ω_i = |S_i| / Σ|S_j|，官方定义）
          unweighted: 每个对象等权
  聚合    agg          ∈ {pooled, per_image}
          pooled    : 数据集级池化（分子分母在整个 split 内累加）
          per_image : 逐图算完再对图取平均

约定锚点：
  (4, argmax, weighted,   pooled)    ← 我们的 eval_glas_official.py
  (8, mode,   weighted,   per_image) ← U-Node 官方实现（metrics.py + notebook）

用法：
    python audit_glas_metric_protocol.py --gt_dir data/glas/masks --pred_dir glas_unode_pred
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, stats

S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
S8 = np.ones((3, 3), dtype=int)


def load_binary(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L")) > 0


def _match_forward(si, gl, ng, match):
    """分割对象 si 匹配到的 GT 对象掩码；无重叠返回 None。"""
    ov = np.array([int((si & (gl == j)).sum()) for j in range(1, ng + 1)])
    if ov.max() == 0:
        return None
    if match == "argmax":
        j = int(np.argmax(ov))
    else:  # mode：取 si 内部出现次数最多的 GT 标签
        labels = gl[si]
        labels = labels[labels > 0]
        if labels.size == 0:
            return None
        j = int(stats.mode(labels, keepdims=False).mode) - 1
    return gl == (j + 1)


def obj_dice_image(pred, gt, conn, match, pred_labeled=False):
    """返回 (每分割对象 Dice 列表, 每 GT 对象 Dice 列表, 分割面积数组, GT 面积数组)。

    pred_labeled=True 时，pred 直接当作已标注的标签图（U-Node postprocess 的输出形态，
    相邻对象虽在二值图上粘连但标签不同），不再重新做连通域标注。
    """
    struct = S4 if conn == 4 else S8
    gl, ngs = ndimage.label(gt, structure=struct)

    if pred_labeled:
        pl = pred.astype(np.int64)
        uniq = np.unique(pl)
        uniq = uniq[uniq > 0]
        pmasks = [pl == l for l in uniq]
        label_to_idx = {int(l): i for i, l in enumerate(uniq)}
    else:
        pl, _ = ndimage.label(pred, structure=struct)
        uniq = np.arange(1, int(pl.max()) + 1)
        pmasks = [pl == i for i in uniq]
        label_to_idx = {int(i): i - 1 for i in uniq}

    nps = len(pmasks)
    pa = np.array([int(m.sum()) for m in pmasks], dtype=float)
    ga = np.array([int((gl == j).sum()) for j in range(1, ngs + 1)], dtype=float)

    if nps == 0 and ngs == 0:
        return np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])
    if nps == 0 or ngs == 0:
        return np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([1.0])

    f, b = np.zeros(nps), np.zeros(ngs)

    # 正向：每个分割对象 → 一个 GT 对象
    for i, si in enumerate(pmasks):
        gsel = _match_forward(si, gl, ngs, match)
        if gsel is not None:
            den = int(si.sum()) + int(gsel.sum())
            f[i] = 2 * int((si & gsel).sum()) / den if den > 0 else 0.0

    # 反向：每个 GT 对象 → 一个分割对象
    for j in range(1, ngs + 1):
        gj = gl == j
        ov = np.array([int((gj & m).sum()) for m in pmasks])
        if ov.max() == 0:
            continue
        if match == "argmax":
            i = int(np.argmax(ov))
        else:
            vals = pl[gj]
            vals = vals[vals > 0]
            if vals.size == 0:
                continue
            i = label_to_idx[int(stats.mode(vals, keepdims=False).mode)]
        ssel = pmasks[i]
        den = int(gj.sum()) + int(ssel.sum())
        b[j - 1] = 2 * int((gj & ssel).sum()) / den if den > 0 else 0.0

    return f, b, pa, ga


def audit(gt_dir: Path, pred_dir: Path, labeled_pred: bool = False):
    names_all = sorted(p.stem for p in gt_dir.glob("*.png"))
    combos = [(c, m, w, a)
              for c in (4, 8) for m in ("argmax", "mode")
              for w in ("weighted", "unweighted") for a in ("pooled", "per_image")]

    def load_pred(p):
        if labeled_pred:
            return np.array(Image.open(p)).astype(np.int64)
        return load_binary(p)

    results = {}
    for split in ("testA", "testB"):
        names = [n for n in names_all if n.startswith(f"{split}_")]
        for conn, match, weight, agg in combos:
            if agg == "pooled":
                num_f = den_f = num_b = den_b = 0.0
                for n in names:
                    p = load_pred(pred_dir / f"{n}.png")
                    g = load_binary(gt_dir / f"{n}.png")
                    f, b, pa, ga = obj_dice_image(p, g, conn, match, labeled_pred)
                    num_f += float((f * pa).sum()); den_f += float(pa.sum())
                    num_b += float((b * ga).sum()); den_b += float(ga.sum())
                val = 0.5 * (num_f / den_f + num_b / den_b) if (den_f > 0 and den_b > 0) else 0.0
            else:
                per_img = []
                for n in names:
                    p = load_pred(pred_dir / f"{n}.png")
                    g = load_binary(gt_dir / f"{n}.png")
                    f, b, pa, ga = obj_dice_image(p, g, conn, match, labeled_pred)
                    if weight == "weighted":
                        fv = float((f * pa).sum() / pa.sum()) if pa.sum() > 0 else 0.0
                        bv = float((b * ga).sum() / ga.sum()) if ga.sum() > 0 else 0.0
                    else:
                        fv, bv = float(f.mean()), float(b.mean())
                    per_img.append(0.5 * (fv + bv))
                val = float(np.mean(per_img)) if per_img else 0.0
            results[(split, conn, match, weight, agg)] = val

    print("=" * 92)
    print(f"GLaS 对象级 Dice —— 评估协议对照表（testA n=60, testB n=20；"
          f"预测来源={'带标签' if labeled_pred else '二值'}）")
    print("=" * 92)
    print(f"{'连通':<5}{'匹配':<8}{'加权':<12}{'聚合':<11}{'testA':>10}{'testB':>10}   备注")
    print("-" * 92)
    for conn, match, weight, agg in combos:
        a = results[("testA", conn, match, weight, agg)]
        b = results[("testB", conn, match, weight, agg)]
        tag = ""
        if (conn, match, weight, agg) == (4, "argmax", "weighted", "pooled"):
            tag = "← 我们的 eval_glas_official.py"
        if (conn, match, weight, agg) == (8, "mode", "weighted", "per_image"):
            tag = "← U-Node 官方口径"
        print(f"{conn:<5}{match:<8}{weight:<12}{agg:<11}{a:>10.4f}{b:>10.4f}   {tag}")
    print("=" * 92)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--labeled_pred", action="store_true",
                    help="预测为带标签图（uint16 PNG，U-Node postprocess 输出形态）")
    args = ap.parse_args()
    audit(Path(args.gt_dir), Path(args.pred_dir), args.labeled_pred)


if __name__ == "__main__":
    main()
