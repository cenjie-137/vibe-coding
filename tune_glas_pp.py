# -*- coding: utf-8 -*-
"""tune_glas_pp.py — GLaS 后处理扫描（路线 R04：本项目 nnU-Net v2 基线 + 后处理改进）

目的：在**不重训**的前提下，给自家 nnU-Net 5 折 ensemble 预测加后处理，看能否在
官方对象级口径下把 testB ObjDice 抬过 L1（0.8184）/ 逼近 L2（0.842）。

两种后处理（都是"我们的"后处理，作用在自家模型输出上 → 结果可计入 state.best）：
  1) R04 定义式：剔除小目标（< min_size px） + 3×3 开运算 × iters。
     预期：ObjF1 白涨，ObjDice 几乎不动（面积加权对碎片不敏感，见 RESEARCH_LOG §6.3）。
  2) 分水岭实例分离：距离变换找中心种子 + 几何地形分水岭切割粘连腺体，
     再把相邻实例的接缝挖掉 1px（保证 4-连通评估下确实分离）。
     预期：针对 testB 的真瓶颈（实例分离），ObjDice/ObjF1 一起涨。

⚠ 调参纪律（红线 R1 的延伸）：**用 testA 选配置、用 testB 报告**（testB 是 L1/L2 所在、
   更难的子集，留作 held-out）。脚本两个子集都打印，但 selection 只认 testA。

用法：
    python tune_glas_pp.py --gt_dir data/glas/masks --pred_dir glas_test_pred \
        --metadata data/glas/metadata.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_glas_official import Acc, _objects, load_mask_binary, load_metadata, process_image


def remove_small(binary: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return binary
    lab, n = ndimage.label(binary)
    if n == 0:
        return np.zeros_like(binary)
    sizes = ndimage.sum(binary, lab, range(1, n + 1))
    out = np.zeros_like(binary)
    for i, s in enumerate(sizes, start=1):
        if s >= min_size:
            out[lab == i] = True
    return out


def pp_opening(binary: np.ndarray, min_size: int, k: int, iters: int) -> np.ndarray:
    b = remove_small(binary, min_size)
    b_u8 = (b.astype(np.uint8)) * 255
    kernel = np.ones((k, k), np.uint8)
    b_u8 = cv2.morphologyEx(b_u8, cv2.MORPH_OPEN, kernel, iterations=iters)
    return b_u8 > 0


def pp_watershed(binary: np.ndarray, fg_dist_thresh: float, min_area: int) -> np.ndarray:
    """距离变换种子 + 几何地形（1-dist_norm）分水岭，切粘连腺体后挖缝分离。"""
    if not binary.any():
        return binary
    obj_u8 = binary.astype(np.uint8)
    dist = cv2.distanceTransform(obj_u8, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return binary
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist_norm, fg_dist_thresh, 1.0, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    if sure_fg.sum() == 0:
        return remove_small(binary, min_area)

    unknown = cv2.subtract(obj_u8, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0

    energy = np.clip(1.0 - dist_norm, 0.0, 1.0)
    terrain = cv2.cvtColor((energy * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    try:
        markers = cv2.watershed(terrain, markers)
    except cv2.error:
        return remove_small(binary, min_area)

    lab = np.zeros_like(markers, dtype=np.int32)
    lab[markers > 1] = markers[markers > 1]
    if lab.max() == 0:
        return binary
    # 挖掉相邻实例接缝 → 4-连通评估下确实分离
    bd = np.zeros_like(lab, dtype=bool)
    bd[1:, :] |= ((lab[1:, :] != lab[:-1, :]) & (lab[1:, :] > 0) & (lab[:-1, :] > 0))
    bd[:, 1:] |= ((lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] > 0) & (lab[:, :-1] > 0))
    sep = lab > 0
    sep[bd] = False
    return remove_small(sep, min_area)


def _solidity(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.0
    c = max(cnts, key=cv2.contourArea)
    harea = cv2.contourArea(cv2.convexHull(c))
    return float(mask.sum()) / harea if harea > 0 else 1.0


def pp_ws_selective(binary: np.ndarray, fg_dist_thresh: float, min_area: int,
                    solidity_thresh: float) -> np.ndarray:
    """只对"看起来是多个腺体粘成"的对象做分水岭（solidity 低 = 哑铃状），
    其余对象原样保留 —— 避免把单个细长腺体误切（naive watershed 掉点的根因）。"""
    lab, n = ndimage.label(binary)
    out = np.zeros_like(binary)
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < min_area:
            continue
        if _solidity(m) < solidity_thresh:
            out |= pp_watershed(m, fg_dist_thresh, min_area)
        else:
            out |= m
    return out


def pp_closing(binary: np.ndarray, k: int, iters: int, min_size: int) -> np.ndarray:
    b_u8 = (binary.astype(np.uint8)) * 255
    kernel = np.ones((k, k), np.uint8)
    b_u8 = cv2.morphologyEx(b_u8, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return remove_small(b_u8 > 0, min_size)


def pp_open_close(binary: np.ndarray, min_size: int, iters: int, k: int) -> np.ndarray:
    b = pp_opening(binary, min_size, k, iters)
    b_u8 = (b.astype(np.uint8)) * 255
    b_u8 = cv2.morphologyEx(b_u8, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return remove_small(b_u8 > 0, min_size)


CONFIGS = [
    ("raw", None, {}),
    ("open m30 k3 x1", pp_opening, dict(min_size=30, k=3, iters=1)),
    ("open m30 k3 x2", pp_opening, dict(min_size=30, k=3, iters=2)),
    ("open m30 k3 x3", pp_opening, dict(min_size=30, k=3, iters=3)),
    ("close k3 x1", pp_closing, dict(k=3, iters=1, min_size=0)),
    ("close k3 x2", pp_closing, dict(k=3, iters=2, min_size=0)),
    ("close k5 x1", pp_closing, dict(k=5, iters=1, min_size=0)),
    ("openx3+close", pp_open_close, dict(min_size=30, iters=3, k=3)),
    ("openx1+close", pp_open_close, dict(min_size=30, iters=1, k=3)),
    ("ws fd0.3 m30", pp_watershed, dict(fg_dist_thresh=0.3, min_area=30)),
    ("wssel s0.85 fd0.4", pp_ws_selective, dict(fg_dist_thresh=0.4, min_area=30, solidity_thresh=0.85)),
]


def eval_config(gt_dir, pred_dir, names, meta, fn, kw):
    accs = {"testA": Acc(), "testB": Acc()}
    for n in names:
        p = pred_dir / f"{n}.png"
        if not p.exists():
            continue
        gm = _objects(load_mask_binary(gt_dir / f"{n}.png"))
        pred_bin = load_mask_binary(p)
        pm = _objects(fn(pred_bin, **kw)) if fn else _objects(pred_bin)
        split = n.split("_")[0]
        process_image(accs[split], pm, gm)
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--dump", default=None,
                    help="把指定 config 名的后处理结果写成 PNG 到 --out_dir（留痕/可复现）")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    meta = load_metadata(Path(args.metadata)) if args.metadata else {}
    names = sorted(p.stem for p in gt_dir.glob("*.png"))

    if args.dump:
        hit = [c for c in CONFIGS if c[0] == args.dump]
        if not hit:
            print(f"没有名为 {args.dump!r} 的 config；可选：{[c[0] for c in CONFIGS]}")
            return 1
        label, fn, kw = hit[0]
        out_dir = Path(args.out_dir or f"{pred_dir.name}_pp")
        out_dir.mkdir(parents=True, exist_ok=True)
        for n in names:
            p = pred_dir / f"{n}.png"
            if not p.exists():
                continue
            b = load_mask_binary(p)
            out = fn(b, **kw) if fn else b
            Image.fromarray((out.astype(np.uint8)) * 255).save(out_dir / f"{n}.png")
        print(f"已写出 config {label!r} 的后处理预测 → {out_dir}（{len(names)} 张）")
        return 0

    print(f"{'config':16s} | {'testA ObjDice':>13s} {'testA ObjF1':>11s} | "
          f"{'testB ObjDice':>13s} {'testB ObjF1':>11s}")
    print("-" * 74)
    rows = []
    for label, fn, kw in CONFIGS:
        accs = eval_config(gt_dir, pred_dir, names, meta, fn, kw)
        a, b = accs["testA"], accs["testB"]
        rows.append((label, a.dice(), a.f1(), b.dice(), b.f1()))
        print(f"{label:16s} | {a.dice():13.4f} {a.f1():11.4f} | "
              f"{b.dice():13.4f} {b.f1():11.4f}")
    print("-" * 74)
    best = max(rows, key=lambda r: r[1])  # 按 testA ObjDice 选（held-out = testB）
    print(f"按 testA ObjDice 选中：{best[0]}  →  testB ObjDice={best[3]:.4f} / ObjF1={best[4]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
