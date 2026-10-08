"""tune_glas.py — GLaS 后处理阈值扫描（不重训，只调推理阈值）

对 testB/testA 逐张做 5-fold 集成 + TTA，得到 seg_prob / cont_prob 概率图，
然后按 --method 扫描参数组合，按对象级 Dice 选出最优，并保存最终二值 mask。
  contour  : 扫 (seg_thresh, cont_thresh, min_area)
  watershed: 扫 (seg_thresh, fg_dist, min_area)

用法:
    python tune_glas.py --data_dir data/glas --weights_dir output_glas \
        --output_dir output_glas/preds_tuned --folds 5 --split testB \
        --method watershed
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image

from inference_glas import (load_models, predict_image, postprocess,
                            postprocess_watershed, postprocess_hover, unletterbox)
from dataset_glas import letterbox_resize
from eval_glas import object_dice, object_f1, load_mask_binary


def main():
    parser = argparse.ArgumentParser(description="GLaS 阈值扫描")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--weights_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--target_size', type=int, default=512)
    parser.add_argument('--split', default='testB', choices=['testB', 'testA'])
    parser.add_argument('--method', choices=['contour', 'watershed', 'hover'], default='watershed')
    parser.add_argument('--model', choices=['cnndcan', 'glas_unet'], default='cnndcan')
    parser.add_argument('--seg_ths', nargs='+', type=float,
                        default=[0.3, 0.4, 0.5, 0.6])
    parser.add_argument('--cont_ths', nargs='+', type=float,
                        default=[0.3, 0.4, 0.5, 0.6])
    parser.add_argument('--fg_dists', nargs='+', type=float,
                        default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--seed_quantiles', nargs='+', type=float,
                        default=[0.3, 0.4, 0.5, 0.6])
    parser.add_argument('--min_areas', nargs='+', type=int,
                        default=[5, 10, 20, 30])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = load_models(args.weights_dir, args.folds, device, model=args.model)
    if not models:
        raise FileNotFoundError(f"未在 {args.weights_dir} 找到 best_fold_*.pth")

    image_dir = Path(args.data_dir) / 'images'
    mask_dir = Path(args.data_dir) / 'masks'
    names = sorted(p.stem for p in image_dir.glob(f"{args.split}_*.png"))
    print(f"split={args.split}, 共 {len(names)} 张")

    print("计算概率图 (5-fold 集成 + TTA) ...")
    probs = {}
    with torch.no_grad():
        for name in names:
            img = Image.open(image_dir / f"{name}.png").convert('RGB')
            W, H = img.size
            arr = np.array(img, dtype=np.float32) / 255.0
            arr, geom = letterbox_resize(arr, args.target_size, cv2.INTER_LINEAR)
            t = torch.from_numpy(np.ascontiguousarray(
                np.transpose(arr, (2, 0, 1)))).unsqueeze(0).float()
            seg_prob, cont_prob, dist_prob = predict_image(models, t, device)
            probs[name] = (seg_prob, cont_prob, dist_prob, geom, W, H)

    gts = {name: load_mask_binary(mask_dir / f"{name}.png") for name in names}

    print("扫描阈值组合 ...")
    results = []
    if args.method == 'watershed':
        for st, fd, ma in itertools.product(args.seg_ths, args.fg_dists, args.min_areas):
            dices, f1s = [], []
            for name in names:
                seg_prob, cont_prob, dist_prob, geom, W, H = probs[name]
                mask_small = postprocess_watershed(seg_prob, cont_prob, st, fd, ma)
                pred = unletterbox(mask_small, geom, W, H) > 0
                gt = gts[name]
                dices.append(object_dice(pred, gt))
                f1s.append(object_f1(pred, gt)[0])
            results.append((float(np.mean(dices)), float(np.mean(f1s)), st, fd, ma))
    elif args.method == 'hover':
        for st, sq, ma in itertools.product(args.seg_ths, args.seed_quantiles, args.min_areas):
            dices, f1s = [], []
            for name in names:
                seg_prob, cont_prob, dist_prob, geom, W, H = probs[name]
                mask_small = postprocess_hover(seg_prob, dist_prob, st, ma, sq)
                pred = unletterbox(mask_small, geom, W, H) > 0
                gt = gts[name]
                dices.append(object_dice(pred, gt))
                f1s.append(object_f1(pred, gt)[0])
            results.append((float(np.mean(dices)), float(np.mean(f1s)), st, sq, ma))
    else:
        for st, ct, ma in itertools.product(args.seg_ths, args.cont_ths, args.min_areas):
            dices, f1s = [], []
            for name in names:
                seg_prob, cont_prob, dist_prob, geom, W, H = probs[name]
                mask_small = postprocess(seg_prob, cont_prob, st, ct, ma)
                pred = unletterbox(mask_small, geom, W, H) > 0
                gt = gts[name]
                dices.append(object_dice(pred, gt))
                f1s.append(object_f1(pred, gt)[0])
            results.append((float(np.mean(dices)), float(np.mean(f1s)), st, ct, ma))

    results.sort(key=lambda r: r[0], reverse=True)
    if args.method == 'watershed':
        print("\n按 ObjDice 降序（前 10 名）:")
        print(f"{'seg_thr':>8} {'fg_dist':>8} {'min_area':>8} {'ObjDice':>9} {'ObjF1':>9}")
        for d, f, st, fd, ma in results[:10]:
            print(f"{st:>8.2f} {fd:>8.2f} {ma:>8d} {d:>9.4f} {f:>9.4f}")
    elif args.method == 'hover':
        print("\n按 ObjDice 降序（前 10 名）:")
        print(f"{'seg_thr':>8} {'seed_q':>8} {'min_area':>8} {'ObjDice':>9} {'ObjF1':>9}")
        for d, f, st, sq, ma in results[:10]:
            print(f"{st:>8.2f} {sq:>8.2f} {ma:>8d} {d:>9.4f} {f:>9.4f}")
    else:
        print("\n按 ObjDice 降序（前 10 名）:")
        print(f"{'seg_thr':>8} {'cont_thr':>9} {'min_area':>8} {'ObjDice':>9} {'ObjF1':>9}")
        for d, f, st, ct, ma in results[:10]:
            print(f"{st:>8.2f} {ct:>9.2f} {ma:>8d} {d:>9.4f} {f:>9.4f}")

    best = results[0]
    if args.method == 'watershed':
        best_d, best_f, bst, bfd, bma = best
        print("=" * 60)
        print(f"最优: seg_thresh={bst} fg_dist={bfd} min_area={bma}  "
              f"ObjDice={best_d:.4f}  ObjF1={best_f:.4f}")
    elif args.method == 'hover':
        best_d, best_f, bst, bsq, bma = best
        print("=" * 60)
        print(f"最优: seg_thresh={bst} seed_quantile={bsq} min_area={bma}  "
              f"ObjDice={best_d:.4f}  ObjF1={best_f:.4f}")
    else:
        best_d, best_f, bst, bct, bma = best
        print("=" * 60)
        print(f"最优: seg_thresh={bst} cont_thresh={bct} min_area={bma}  "
              f"ObjDice={best_d:.4f}  ObjF1={best_f:.4f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        seg_prob, cont_prob, dist_prob, geom, W, H = probs[name]
        if args.method == 'watershed':
            mask_small = postprocess_watershed(seg_prob, cont_prob, bst, bfd, bma)
        elif args.method == 'hover':
            mask_small = postprocess_hover(seg_prob, dist_prob, bst, bma, bsq)
        else:
            mask_small = postprocess(seg_prob, cont_prob, bst, bct, bma)
        Image.fromarray(unletterbox(mask_small, geom, W, H)).save(out / f"{name}.png")
    print(f"已用最优阈值保存 mask → {out}")


if __name__ == '__main__':
    main()