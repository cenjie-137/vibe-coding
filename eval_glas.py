"""eval_glas.py — GLaS object-level Dice / F1 评估 (官方协议)

指标定义（https://glas.grand-challenge.org/Evaluation/）：

Object-level F1（检测）：
  对每个分割连通域 s，取其"最大重叠"的 GT 对象 g*；
  若 |s ∩ g*| ≥ 0.5·|g*| 记 True Positive，否则 False Positive。
  GT 对象 g 若无对应分割对象、或 <50% 面积被覆盖，记 False Negative。
  F1 = 2·P·R / (P+R)，P = TP/(TP+FP)，R = TP/(TP+FN)。

Object-level Dice（分割）：
  D_obj = 0.5 · [ (1/n_S)·Σ_i Dice(s_i, g_i*) + (1/n_G)·Σ_j Dice(g_j, s_j*) ]
  g_i* = 与 s_i 最大 Dice 的 GT；s_j* = 与 g_j 最大 Dice 的分割。
  Dice(a,b) = 2|a∩b|/(|a|+|b|)。逐图像计算后求平均。

Object-level Hausdorff（形状相似度）：
  用与 object Dice 相同的最大-Dice 失配方式，对每个分割对象与其最优 GT、
  每个 GT 与其最优分割对象分别求对称 Hausdorff 距离，双向平均；越低越接近。

用法:
    python eval_glas.py --gt_dir data/glas/masks --pred_dir preds \
        --metadata data/glas/metadata.csv --split testB
    python eval_glas.py --gt_dir data/glas/masks --pred_dir preds \
        --metadata data/glas/metadata.csv --split all
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.distance import cdist


def load_mask_binary(path: Path) -> np.ndarray:
    """加载掩膜，返回 bool 数组，True=前景。"""
    return np.array(Image.open(path).convert("L")) > 0


def connected_components(mask: np.ndarray):
    """返回 (labeled, n)，外围连通域 id 从 1 到 n。"""
    labeled, n = ndimage.label(mask)
    return labeled, n


def object_f1(pred: np.ndarray, gt: np.ndarray):
    """返回 (f1, precision, recall, tp, fp, fn)。"""
    pred_lab, n_s = connected_components(pred)
    gt_lab, n_g = connected_components(gt)

    if n_s == 0 and n_g == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0

    # 每个 GT 对象的面积与像素坐标集
    gt_masks = [(gt_lab == i) for i in range(1, n_g + 1)]
    gt_areas = np.array([m.sum() for m in gt_masks])

    tp = 0
    gt_detected = np.zeros(n_g, dtype=bool)
    for i in range(1, n_s + 1):
        s = pred_lab == i
        if n_g == 0:
            continue
        # 与 s 相交的 GT 里，取覆盖率 |s∩g|/|g| 最大者
        inter = np.array([int((s & g).sum()) for g in gt_masks])
        nonzero = inter > 0
        if not nonzero.any():
            continue
        frac = np.where(gt_areas > 0, inter / gt_areas, 0.0)
        frac[~nonzero] = -1.0
        best_idx = int(np.argmax(frac))
        if frac[best_idx] >= 0.5:
            tp += 1
            gt_detected[best_idx] = True

    fp = n_s - tp
    fn = int((~gt_detected).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall, tp, fp, fn


def object_dice(pred: np.ndarray, gt: np.ndarray):
    """按官方公式的 object-level Dice。"""
    pred_lab, n_s = connected_components(pred)
    gt_lab, n_g = connected_components(gt)

    if n_s == 0 and n_g == 0:
        return 1.0
    if n_s == 0 or n_g == 0:
        return 0.0

    pred_masks = [(pred_lab == i) for i in range(1, n_s + 1)]
    gt_masks = [(gt_lab == i) for i in range(1, n_g + 1)]
    pred_areas = np.array([m.sum() for m in pred_masks])
    gt_areas = np.array([m.sum() for m in gt_masks])

    def dice(a, a_area, b, b_area):
        inter = int((a & b).sum())
        denom = a_area + b_area
        return 2 * inter / denom if denom > 0 else 0.0

    sum_s = 0.0
    for s, sa in zip(pred_masks, pred_areas):
        best = max((dice(s, sa, g, ga) for g, ga in zip(gt_masks, gt_areas)), default=0.0)
        sum_s += best
    sum_g = 0.0
    for g, ga in zip(gt_masks, gt_areas):
        best = max((dice(s, sa, g, ga) for s, sa in zip(pred_masks, pred_areas)), default=0.0)
        sum_g += best

    return 0.5 * (sum_s / n_s + sum_g / n_g)


def _boundary_points(mask: np.ndarray):
    """返回物体边界的像素坐标 (N,2)，用于 Hausdorff 计算。"""
    mask = mask.astype(bool)
    if not mask.any():
        return np.empty((0, 2), dtype=int)
    eroded = ndimage.binary_erosion(mask)
    bd = mask & ~eroded
    ys, xs = np.nonzero(bd)
    if len(ys) == 0:  # 物体过小被腐蚀空，退化用全部像素
        ys, xs = np.nonzero(mask)
    return np.stack([ys, xs], axis=1)


def hausdorff_distance(pa: np.ndarray, pb: np.ndarray):
    """两点集间的对称 Hausdorff 距离；任一点集为空返回 inf。"""
    if len(pa) == 0 or len(pb) == 0:
        return float("inf")
    d = cdist(pa.astype(np.float64), pb.astype(np.float64))
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def object_hausdorff(pred: np.ndarray, gt: np.ndarray):
    """官方 object-level Hausdorff：与 object_dice 相同的失配（最大 Dice）方式，
    对每个分割对象与其最优 GT、每个 GT 与其最优分割对象分别求 Hausdorff，双向平均。"""
    pred_lab, n_s = connected_components(pred)
    gt_lab, n_g = connected_components(gt)

    if n_s == 0 and n_g == 0:
        return 0.0
    if n_s == 0 or n_g == 0:
        return float("inf")

    pred_masks = [(pred_lab == i) for i in range(1, n_s + 1)]
    gt_masks = [(gt_lab == i) for i in range(1, n_g + 1)]
    pred_areas = np.array([m.sum() for m in pred_masks])
    gt_areas = np.array([m.sum() for m in gt_masks])
    pred_points = [_boundary_points(m) for m in pred_masks]
    gt_points = [_boundary_points(m) for m in gt_masks]

    def dice(a, a_area, b, b_area):
        inter = int((a & b).sum())
        denom = a_area + b_area
        return 2 * inter / denom if denom > 0 else 0.0

    sum_s = 0.0
    for i, (s, sa, ps) in enumerate(zip(pred_masks, pred_areas, pred_points)):
        best_d = -1.0
        best_j = -1
        for j, (g, ga) in enumerate(zip(gt_masks, gt_areas)):
            d = dice(s, sa, g, ga)
            if d > best_d:
                best_d = d
                best_j = j
        sum_s += hausdorff_distance(ps, gt_points[best_j]) if best_j >= 0 else float("inf")

    sum_g = 0.0
    for j, (g, ga, gp) in enumerate(zip(gt_masks, gt_areas, gt_points)):
        best_d = -1.0
        best_i = -1
        for i, (s, sa) in enumerate(zip(pred_masks, pred_areas)):
            d = dice(s, sa, g, ga)
            if d > best_d:
                best_d = d
                best_i = i
        sum_g += hausdorff_distance(gp, pred_points[best_i]) if best_i >= 0 else float("inf")

    return 0.5 * (sum_s / n_s + sum_g / n_g)


def load_metadata(metadata_path: Path):
    """name -> grade_glas。name 为不带扩展名的文件名。"""
    import csv
    meta = {}
    if metadata_path is None or not metadata_path.exists():
        return meta
    with metadata_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            meta[row["name"].strip()] = row["grade_glas"].strip()
    return meta


def evaluate_split(gt_dir: Path, pred_dir: Path, names, meta):
    rows = []
    for name in names:
        gt_path = gt_dir / f"{name}.png"
        pred_path = pred_dir / f"{name}.png"
        if not pred_path.exists():
            print(f"[warn] 缺少预测 {pred_path.name}，跳过", file=None)
            continue
        gt = load_mask_binary(gt_path)
        pred = load_mask_binary(pred_path)
        d = object_dice(pred, gt)
        f1, p, r, tp, fp, fn = object_f1(pred, gt)
        hd = object_hausdorff(pred, gt)
        rows.append({
            "name": name,
            "grade": meta.get(name, "?"),
            "obj_dice": d,
            "obj_f1": f1,
            "obj_hd": hd,
            "tp": tp, "fp": fp, "fn": fn,
        })
    return rows


def report(rows, label: str = ""):
    if not rows:
        print(f"[{label}] 无可评估样本")
        return
    dice = np.array([r["obj_dice"] for r in rows])
    f1 = np.array([r["obj_f1"] for r in rows])
    hd = np.array([r["obj_hd"] for r in rows])
    hd_finite = hd[np.isfinite(hd)]
    hd_str = f"  ObjHD={hd_finite.mean():.2f}" if hd_finite.size else "  ObjHD=inf"
    print(f"{label:24s} n={len(rows):3d}  ObjDice={dice.mean():.4f}  ObjF1={f1.mean():.4f}{hd_str}")


def main():
    parser = argparse.ArgumentParser(description="GLaS object-level 评估")
    parser.add_argument("--gt_dir", required=True, help="GT 掩膜目录 (data/glas/masks)")
    parser.add_argument("--pred_dir", required=True, help="预测掩膜目录（与 GT 同名 PNG，0/255）")
    parser.add_argument("--metadata", default=None, help="metadata.csv 路径（用于 benign/malignant 分组）")
    parser.add_argument("--split", default="all",
                        choices=["all", "train", "testA", "testB"],
                        help="评估哪个 split")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    meta = load_metadata(Path(args.metadata)) if args.metadata else {}

    # 收集可用名称：以 GT 目录为准
    names = sorted(p.stem for p in gt_dir.glob("*.png"))

    if args.split != "all":
        names = [n for n in names if n.startswith(f"{args.split}_")]

    rows = evaluate_split(gt_dir, pred_dir, names, meta)

    print("=" * 70)
    if args.split == "all":
        for s in ("train", "testA", "testB"):
            sub = [r for r in rows if r["name"].startswith(f"{s}_")]
            report(sub, s)
    else:
        report(rows, args.split)

    # 按 benign / malignant 细分（仅对有 grade 信息的样本）
    if meta:
        benign = [r for r in rows if r["grade"] == "benign"]
        malignant = [r for r in rows if r["grade"] == "malignant"]
        print("-" * 70)
        report(benign, "benign")
        report(malignant, "malignant")
    print("=" * 70)


if __name__ == "__main__":
    main()