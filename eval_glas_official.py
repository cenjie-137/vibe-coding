"""eval_glas_official.py — GLaS 官方对象级评估（严格按官方定义实现）

来源：GlaS 官方评估页
https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/evaluation/
（对应论文 Sirinukunwattana et al., Med Image Anal 35:489-502, 2017, arXiv:1603.00275）

与常见社区实现的三个关键差异（本脚本严格遵循官方页）：

1) **面积加权**：ω_i = |S_i| / Σ_j |S_j|，ω̃_i = |G̃_i| / Σ_j |G̃_j|。
2) **数据集级池化**：n_S / n_G 与权重分母都是整个测试集（testA 或 testB）
   内所有对象的总和，而不是"逐图算完再平均"。
3) **匹配规则**：
   - Dice：G_i = 与 S_i 重叠面积最大的 GT 对象；G̃_i 同理。
   - Hausdorff：同上；**若完全无重叠**，官方规定退化为"与 S_i 距离最小的
     GT 对象"（避免 inf），反向同理。

三个指标：
   Detection F1      = 2PR/(P+R)，TP 判定为 |S∩G| ≥ 0.5·|G|
   Object-level Dice = 1/2 [ Σ ω_i·Dice(G_i,S_i) + Σ ω̃_i·Dice(G̃_i,S̃_i) ]
   Object-level HD   = 1/2 [ Σ ω_i·H(G_i,S_i)   + Σ ω̃_i·H(G̃_i,S̃_i) ]

**输入形态（2026-10-08 口径裁定，见 RESEARCH_LOG §6.11）**：

GLaS 官方提交是**实例标签图**（每个腺体一个标签，相邻腺体即便像素粘连、标签仍不同），
不是二值前景图。因此本脚本支持两种输入：

    # 二值预测（默认）：pred 按连通域重新标注成对象
    python eval_glas_official.py --gt_dir data/glas/masks --pred_dir glas_test_pred

    # 实例标签预测：pred 直接当标签图，粘连的相邻腺体不会被并成一个对象
    python eval_glas_official.py --gt_dir data/glas/masks \
        --pred_dir glas_unode_pred_label --pred_kind label --metadata data/glas/metadata.csv

⚠ **输入形态必须与预测实际形态一致（红线 R1）**：把实例标签预测二值化后再评估，
连通域标注会把"相邻但标签不同"的腺体合并，系统性低估对象级 Dice
（实测 U-Node 同一批预测：label 0.8440 vs binary 0.8215，差 ≈0.02）。
脚本对 label 模式做形态自检：图像若只有 {0,255} 两个取值即报错 —— 防止把二值图当标签图喂进来。
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.distance import cdist


def load_mask_binary(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def load_label_objects(path: Path) -> list:
    """实例标签图 → 对象布尔掩码列表。

    每个非零标签算一个独立对象 —— 这正是与二值连通域标注的关键差别：
    相邻腺体在像素上粘连，但只要标签不同就是两个对象，不会被合并。
    """
    lab = np.array(Image.open(path))
    if lab.ndim == 3:
        lab = lab[..., 0]
    ids = np.unique(lab)
    ids = ids[ids > 0]
    return [lab == i for i in ids]


def check_label_dir(pred_dir: Path, names, sample: int = 20):
    """目录级形态自检：label 模式却拿到一堆二值图 → 直接拦下。

    单张图无法区分"单对象标签图"和"二值图"（都是 {0,1}），所以按目录统计：
    若抽样中过半图像只有 ≤2 个取值，判定为二值图目录并报错。
    这是防"把二值图当标签图喂进来、静默算出无意义数字"的最后一道闸（红线 R1）。
    """
    n_binary = n_seen = 0
    for n in names[:sample]:
        p = pred_dir / f"{n}.png"
        if not p.exists():
            continue
        vals = np.unique(np.array(Image.open(p)))
        n_seen += 1
        if vals.size <= 2:
            n_binary += 1
    if n_seen and n_binary > n_seen / 2:
        raise ValueError(
            f"--pred_kind label 但 {pred_dir} 里 {n_binary}/{n_seen} 张图只有 ≤2 个取值 —— "
            f"这看起来是二值图目录。二值图请用默认模式（--pred_kind binary）。"
        )


def _boundary_points(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.empty((0, 2), dtype=int)
    eroded = ndimage.binary_erosion(mask)
    bd = mask & ~eroded
    ys, xs = np.nonzero(bd)
    if len(ys) == 0:
        ys, xs = np.nonzero(mask)
    return np.stack([ys, xs], axis=1)


def _hausdorff(pa: np.ndarray, pb: np.ndarray) -> float:
    if len(pa) == 0 or len(pb) == 0:
        return float("inf")
    d = cdist(pa.astype(np.float64), pb.astype(np.float64))
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _objects(mask: np.ndarray):
    lab, n = ndimage.label(mask)
    return [lab == i for i in range(1, n + 1)]


class Acc:
    """数据集级累加器：累加 Σ|S_i|·metric、Σ|G_i|·metric 与总面积。"""

    def __init__(self):
        self.s_area = 0.0
        self.g_area = 0.0
        self.s_num = 0.0     # Σ |S_i| · Dice(S_i, G_i)
        self.g_num = 0.0     # Σ |G_i| · Dice(G_i, S_i)
        self.s_hd = 0.0      # Σ |S_i| · H(S_i, G_i)
        self.g_hd = 0.0      # Σ |G_i| · H(G_i, S_i)
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def dice(self):
        if self.s_area == 0 or self.g_area == 0:
            return 0.0
        return 0.5 * (self.s_num / self.s_area + self.g_num / self.g_area)

    def hd(self):
        if self.s_area == 0 or self.g_area == 0:
            return float("inf")
        return 0.5 * (self.s_hd / self.s_area + self.g_hd / self.g_area)

    def f1(self):
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _overlap_matrix(src_masks, tgt_masks):
    return np.array([[int((s & t).sum()) for t in tgt_masks] for s in src_masks],
                    dtype=float)


def process_image(acc: Acc, pm, gm):
    """pm / gm：对象布尔掩码列表。

    pred 侧对象怎么来的由调用方决定 —— 二值图走连通域标注，实例标签图直接按标签切分。
    """
    if not pm and not gm:
        return
    pa = np.array([m.sum() for m in pm], dtype=float)
    ga = np.array([m.sum() for m in gm], dtype=float)
    acc.s_area += pa.sum()
    acc.g_area += ga.sum()

    ov = _overlap_matrix(pm, gm) if (pm and gm) else np.zeros((len(pm), len(gm)))
    pp = [_boundary_points(m) for m in pm]
    gp = [_boundary_points(m) for m in gm]

    # ---- 检测 F1（官方：分割对象与"最大重叠 GT"的交集 ≥ 50% 该 GT 面积 → TP）
    claimed = np.zeros(len(gm), dtype=bool)
    for i in range(len(pm)):
        if not len(gm):
            acc.fp += 1
            continue
        j = int(np.argmax(ov[i]))
        if ov[i, j] >= 0.5 * ga[j]:
            acc.tp += 1
            claimed[j] = True
        else:
            acc.fp += 1
    acc.fn += int((~claimed).sum())

    # ---- 正向：每个分割对象 → 最佳 GT
    for i, s in enumerate(pm):
        if len(gm):
            j = int(np.argmax(ov[i]))
            if ov[i, j] > 0:
                den = pa[i] + ga[j]
                acc.s_num += pa[i] * (2 * ov[i, j] / den if den > 0 else 0.0)
                acc.s_hd += pa[i] * _hausdorff(pp[i], gp[j])
                continue
            # 无重叠：Dice 记 0；Hausdorff 退化为最近 GT（官方规定）
            hds = [_hausdorff(pp[i], g) for g in gp]
            acc.s_hd += pa[i] * (min(hds) if hds else float("inf"))
        else:
            acc.s_hd += pa[i] * float("inf")

    # ---- 反向：每个 GT 对象 → 最佳分割对象
    for j, g in enumerate(gm):
        if len(pm):
            i = int(np.argmax(ov[:, j]))
            if ov[i, j] > 0:
                den = pa[i] + ga[j]
                acc.g_num += ga[j] * (2 * ov[i, j] / den if den > 0 else 0.0)
                acc.g_hd += ga[j] * _hausdorff(gp[j], pp[i])
                continue
            hds = [_hausdorff(gp[j], p) for p in pp]
            acc.g_hd += ga[j] * (min(hds) if hds else float("inf"))
        else:
            acc.g_hd += ga[j] * float("inf")


def load_metadata(path: Path):
    meta = {}
    if path is None or not path.exists():
        return meta
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            meta[row["name"].strip()] = row["grade_glas"].strip()
    return meta


def run_split(gt_dir: Path, pred_dir: Path, names, meta, label, pred_kind="binary"):
    acc_all, acc_by_grade = Acc(), {}
    n_used = 0
    for n in names:
        pp = pred_dir / f"{n}.png"
        if not pp.exists():
            continue
        gm = _objects(load_mask_binary(gt_dir / f"{n}.png"))
        pm = (load_label_objects(pp) if pred_kind == "label"
              else _objects(load_mask_binary(pp)))
        process_image(acc_all, pm, gm)
        grade = meta.get(n, "?")
        if grade in ("benign", "malignant"):
            process_image(acc_by_grade.setdefault(grade, Acc()), pm, gm)
        n_used += 1
    print(f"  {label:10s} n={n_used:3d}  ObjDice={acc_all.dice():.4f}  "
          f"ObjF1={acc_all.f1():.4f}  ObjHD={acc_all.hd():7.2f}")
    return acc_by_grade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--pred_kind", choices=("binary", "label"), default="binary",
                    help="预测输入形态：binary=二值前景图（按连通域标注对象，默认）；"
                         "label=实例标签图（官方提交形态，每个腺体一个标签）")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    meta = load_metadata(Path(args.metadata)) if args.metadata else {}
    names = sorted(p.stem for p in gt_dir.glob("*.png"))
    if args.pred_kind == "label":
        check_label_dir(pred_dir, names)

    print("=" * 62)
    print("GLaS 官方协议（面积加权 + 数据集级池化）")
    print(f"预测输入形态：{args.pred_kind}"
          + ("（实例标签图）" if args.pred_kind == "label" else "（二值图 → 连通域标注）"))
    print("=" * 62)
    grades = {}
    for s in ("testA", "testB"):
        sub = [n for n in names if n.startswith(f"{s}_")]
        g = run_split(gt_dir, pred_dir, sub, meta, s, args.pred_kind)
        for k, v in g.items():
            grades.setdefault(k, []).append(v)
    print("-" * 62)
    for k, accs in grades.items():
        # 分组合并：同一累加器结构，直接相加各 split 的累加量
        merged = Acc()
        for a in accs:
            merged.s_area += a.s_area
            merged.g_area += a.g_area
            merged.s_num += a.s_num
            merged.g_num += a.g_num
            merged.s_hd += a.s_hd
            merged.g_hd += a.g_hd
            merged.tp += a.tp
            merged.fp += a.fp
            merged.fn += a.fn
        print(f"  {k:10s} (testA+testB)  ObjDice={merged.dice():.4f}  "
              f"ObjF1={merged.f1():.4f}  ObjHD={merged.hd():7.2f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
