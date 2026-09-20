"""inference_glas.py — 5-fold 集成 + TTA + 后处理，输出对象级二值掩膜

后处理（--method 选择）：
  contour  : 前景减轮廓 + 连通域去小碎片（DCAN 式，旧方案）
  watershed: 距离变换找腺体中心种子 + 轮廓概率地形分水岭（切割粘连腺体）

用法:
    python inference_glas.py --data_dir data/glas --weights_dir output_glas \
        --output_dir output_glas/preds --method watershed
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
import cv2

from models_glas import GlandUNet


def load_models(weights_dir, num_folds, device):
    models = []
    for k in range(num_folds):
        path = os.path.join(weights_dir, f"best_fold_{k}.pth")
        if not os.path.exists(path):
            print(f"[warn] 缺少 {path}，跳过")
            continue
        model = GlandUNet(model_source="UNI", freeze_encoder=False)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model'])
        model.to(device).eval()
        models.append(model)
    return models


@torch.no_grad()
def predict_image(models, img_tensor, device):
    """多模型 + 4 方向 TTA，返回 (seg_prob, contour_prob) 各为 (H, W)。"""
    seg_sum = None
    cont_sum = None
    for model in models:
        x = img_tensor.to(device)
        batch = torch.cat([x,
                           torch.flip(x, [3]),
                           torch.flip(x, [2]),
                           torch.flip(x, [2, 3])], dim=0)
        seg_logit, cont_logit = model(batch)
        seg_p = torch.sigmoid(seg_logit)[:, 0]
        cont_p = torch.sigmoid(cont_logit)[:, 0]

        seg_p = torch.stack([seg_p[0],
                             torch.flip(seg_p[1], [1]),
                             torch.flip(seg_p[2], [0]),
                             torch.flip(seg_p[3], [0, 1])], dim=0)
        cont_p = torch.stack([cont_p[0],
                              torch.flip(cont_p[1], [1]),
                              torch.flip(cont_p[2], [0]),
                              torch.flip(cont_p[3], [0, 1])], dim=0)

        seg_mean = seg_p.mean(dim=0).cpu().numpy()
        cont_mean = cont_p.mean(dim=0).cpu().numpy()
        seg_sum = seg_mean if seg_sum is None else seg_sum + seg_mean
        cont_sum = cont_mean if cont_sum is None else cont_sum + cont_mean

    n = max(1, len(models))
    return seg_sum / n, cont_sum / n


def postprocess(seg_prob, cont_prob, seg_thresh=0.5, cont_thresh=0.5, min_area=10):
    obj = seg_prob > seg_thresh
    cont = cont_prob > cont_thresh
    separated = obj & ~cont
    labeled, n = ndimage.label(separated)
    if n == 0:
        return np.zeros_like(separated, dtype=np.uint8)
    sizes = ndimage.sum(separated, labeled, range(1, n + 1))
    mask = np.zeros_like(labeled, dtype=np.uint8)
    for i, s in enumerate(sizes, start=1):
        if s >= min_area:
            mask[labeled == i] = 255
    return mask


def postprocess_watershed(seg_prob, cont_prob, seg_thresh=0.5, fg_dist_thresh=0.4,
                          min_area=10):
    """分水岭后处理：距离变换找腺体中心种子 + 轮廓概率地形，切割粘连腺体。

    Args:
        seg_prob: (H,W) 前景概率 [0,1]
        cont_prob: (H,W) 轮廓概率 [0,1]（山脊=预测边界，引导分水岭切割）
        seg_thresh: 二值化前景阈值
        fg_dist_thresh: 距离变换相对 max 的种子阈值 [0,1]，越大种子越靠中心
        min_area: 保留的最小连通域面积
    Returns:
        mask: uint8 (H,W)，0/255
    """
    obj = seg_prob > seg_thresh
    if not obj.any():
        return np.zeros(seg_prob.shape, dtype=np.uint8)

    obj_u8 = obj.astype(np.uint8)

    # 1. 距离变换：腺体中心最亮，用于定位中心种子
    dist = cv2.distanceTransform(obj_u8, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return (obj_u8 * 255).astype(np.uint8)

    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # 2. sure foreground = 距离变换的强值区（中心种子）
    _, sure_fg = cv2.threshold(dist_norm, fg_dist_thresh, 1.0, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # 退化为二值 + 连通域的条件：无可靠中心
    def _fallback(binary):
        labeled, n = ndimage.label(binary)
        out = np.zeros(binary.shape, dtype=np.uint8)
        if n == 0:
            return out
        sizes = ndimage.sum(binary, labeled, range(1, n + 1))
        for i, s in enumerate(sizes, start=1):
            if s >= min_area:
                out[labeled == i] = 255
        return out

    if sure_fg.sum() == 0:
        return _fallback(obj)

    # 3. unknown（前景里的非中心区）与 markers
    unknown = cv2.subtract(obj_u8, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0

    # 4. 分水岭：地形用轮廓概率（边界处高=山脊，切割线落在预测边界上）
    #    注：若轮廓图噪声大，可改用 -dist_norm 作纯几何地形。
    land = np.clip(cont_prob, 0.0, 1.0)
    land_u8 = (land * 255).astype(np.uint8)
    land_bgr = cv2.cvtColor(land_u8, cv2.COLOR_GRAY2BGR)
    try:
        markers = cv2.watershed(land_bgr, markers)
    except cv2.error:
        return _fallback(obj)

    # 5. 前景实例标记（-1=分水岭线，1=背景，>=2=实例）
    lab = np.zeros_like(markers, dtype=np.int32)
    lab[markers > 1] = markers[markers > 1]
    if lab.max() == 0:
        return (obj_u8 * 255).astype(np.uint8)

    # 6. 挖掉两个不同实例相邻处，保证 4-连通分离（评估用 ndimage.label 默认 4-连通）
    bd = np.zeros_like(lab, dtype=bool)
    bd[1:, :] |= ((lab[1:, :] != lab[:-1, :]) & (lab[1:, :] > 0) & (lab[:-1, :] > 0))
    bd[:, 1:] |= ((lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] > 0) & (lab[:, :-1] > 0))
    sep = lab > 0
    sep[bd] = False

    # 7. 去小碎片
    return _fallback(sep)


def main():
    parser = argparse.ArgumentParser(description="GLaS 推理")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--weights_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--target_size', type=int, default=512)
    parser.add_argument('--min_area', type=int, default=10)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seg_thresh', type=float, default=0.5)
    parser.add_argument('--cont_thresh', type=float, default=0.5)
    parser.add_argument('--method', choices=['contour', 'watershed'], default='watershed')
    parser.add_argument('--fg_dist_thresh', type=float, default=0.4)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = load_models(args.weights_dir, args.folds, device)
    if not models:
        raise FileNotFoundError(f"未在 {args.weights_dir} 找到 best_fold_*.pth")

    image_dir = Path(args.data_dir) / 'images'
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_names = sorted(
        p.stem for p in image_dir.glob('*.png')
        if p.stem.startswith('testA_') or p.stem.startswith('testB_')
    )
    print(f"待推理 {len(test_names)} 张 (testA + testB)")

    for name in test_names:
        img = Image.open(image_dir / f"{name}.png").convert('RGB')
        W, H = img.size
        img_small = img.resize((args.target_size, args.target_size), Image.BILINEAR)
        arr = np.array(img_small, dtype=np.float32) / 255.0
        t = torch.from_numpy(np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))).unsqueeze(0).float()

        seg_prob, cont_prob = predict_image(models, t, device)
        if args.method == 'watershed':
            mask_small = postprocess_watershed(seg_prob, cont_prob,
                                               seg_thresh=args.seg_thresh,
                                               fg_dist_thresh=args.fg_dist_thresh,
                                               min_area=args.min_area)
        else:
            mask_small = postprocess(seg_prob, cont_prob,
                                     seg_thresh=args.seg_thresh,
                                     cont_thresh=args.cont_thresh,
                                     min_area=args.min_area)
        mask_full = Image.fromarray(mask_small).resize((W, H), Image.NEAREST)
        mask_full.save(out_dir / f"{name}.png")

    print(f"推理完成，共 {len(test_names)} 张 → {out_dir}")


if __name__ == '__main__':
    main()