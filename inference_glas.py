"""inference_glas.py — 5-fold 集成 + TTA + 后处理，输出对象级二值掩膜

后处理（--method 选择）：
  contour  : 前景减轮廓 + 连通域去小碎片（DCAN 式，旧方案）
  watershed: 距离变换找腺体中心种子 + 轮廓概率地形分水岭（切割粘连腺体）
  hover    : 距离变换找腺体中心种子 + HoVer 距离图方向场梯度能量地形分水岭

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

from models_glas import GlandUNet, GlandCNNDCAN
from dataset_glas import letterbox_resize


def unletterbox(mask, geom, orig_w, orig_h):
    """从 letterbox 后的正方形 mask 裁回原图尺寸。"""
    cropped = mask[geom['pad_top']:geom['pad_top'] + geom['new_h'],
                   geom['pad_left']:geom['pad_left'] + geom['new_w']]
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def load_models(weights_dir, num_folds, device, model='glas_unet'):
    models = []
    for k in range(num_folds):
        path = os.path.join(weights_dir, f"best_fold_{k}.pth")
        if not os.path.exists(path):
            print(f"[warn] 缺少 {path}，跳过")
            continue
        if model == 'cnndcan':
            mdl = GlandCNNDCAN()
        else:
            mdl = GlandUNet(model_source="UNI", freeze_encoder=False)
        ckpt = torch.load(path, map_location=device)
        mdl.load_state_dict(ckpt['model'])
        mdl.to(device).eval()
        models.append(mdl)
    return models


@torch.no_grad()
def predict_image(models, img_tensor, device):
    """多模型 + 4 方向 TTA，返回 (seg_prob, contour_prob, dist_prob)。

    seg/contour 做 TTA 平均；dist（距离图）向量翻转会改方向符号，只取原始方向并跨模型平均。
    """
    seg_sum = None
    cont_sum = None
    dist_sum = None
    n_dist = 0
    for model in models:
        x = img_tensor.to(device)
        batch = torch.cat([x,
                           torch.flip(x, [3]),
                           torch.flip(x, [2]),
                           torch.flip(x, [2, 3])], dim=0)
        out = model(batch)
        if len(out) == 3:
            seg_logit, cont_logit, dist_out = out
        else:
            seg_logit, cont_logit = out
            dist_out = None
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

        if dist_out is not None:  # 距离图，取第0个变体（原始方向）
            d = dist_out[0].cpu().numpy()  # (2,H,W)
            dist_sum = d if dist_sum is None else dist_sum + d
            n_dist += 1

    n = max(1, len(models))
    dist_prob = (dist_sum / max(1, n_dist)) if dist_sum is not None else None
    return seg_sum / n, cont_sum / n, dist_prob


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


def postprocess_hover(seg_prob, dist, seg_thresh=0.5, min_area=10, fg_dist_thresh=0.4):
    """HoVer 式实例分割后处理：方向场梯度能量做分水岭地形，切割粘连腺体。

    与 postprocess_watershed 结构一致，唯一区别：地形由 contour 概率换成 HoVer 距离图
    的方向场梯度能量（相邻实例交界处方向突变 => 高能量 = 分割山脊）；种子仍用
    距离变换的几何中心（稳健），不依赖方向场的汇聚/发散约定。

    Args:
        seg_prob: (H,W) 前景概率
        dist: (2,H,W) 距离图（通道0=dy、1=dx，全局归一化），可为 None 退化为几何地形
        seg_thresh: 前景二值化阈值
        min_area: 保留最小连通域面积
        fg_dist_thresh: 距离变换相对 max 的中心种子阈值 [0,1]
    Returns:
        mask: uint8 (H,W)，0/255
    """
    H, W = seg_prob.shape
    obj = seg_prob > seg_thresh

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

    if not obj.any():
        return np.zeros((H, W), dtype=np.uint8)

    obj_u8 = obj.astype(np.uint8)

    dist_t = cv2.distanceTransform(obj_u8, cv2.DIST_L2, 5)
    if dist_t.max() <= 0:
        return (obj_u8 * 255).astype(np.uint8)
    dist_norm = cv2.normalize(dist_t, None, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_norm, fg_dist_thresh, 1.0, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    if sure_fg.sum() == 0:
        return _fallback(obj)

    # 地形 = 方向场梯度能量（HoVer 贡献点）；退化时用几何地形
    energy = None
    if dist is not None:
        dy = dist[0].astype(np.float32)
        dx = dist[1].astype(np.float32)
        g_xx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=3)
        g_xy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=3)
        g_yx = cv2.Sobel(dy, cv2.CV_32F, 1, 0, ksize=3)
        g_yy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3)
        energy = np.sqrt(g_xx ** 2 + g_xy ** 2 + g_yx ** 2 + g_yy ** 2)
        energy[~obj] = 0.0
        if energy.max() <= 1e-6:
            energy = None
    if energy is None:
        energy = 1.0 - dist_norm

    unknown = cv2.subtract(obj_u8, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0

    terrain = cv2.cvtColor(((energy / energy.max()) * 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)
    try:
        markers = cv2.watershed(terrain, markers)
    except cv2.error:
        return _fallback(obj)

    lab = np.zeros_like(markers, dtype=np.int32)
    lab[markers > 1] = markers[markers > 1]
    if lab.max() == 0:
        return (obj_u8 * 255).astype(np.uint8)

    bd = np.zeros_like(lab, dtype=bool)
    bd[1:, :] |= ((lab[1:, :] != lab[:-1, :]) & (lab[1:, :] > 0) & (lab[:-1, :] > 0))
    bd[:, 1:] |= ((lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] > 0) & (lab[:, :-1] > 0))
    sep = lab > 0
    sep[bd] = False
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
    parser.add_argument('--method', choices=['contour', 'watershed', 'hover'], default='watershed')
    parser.add_argument('--fg_dist_thresh', type=float, default=0.4,
                        help='watershed/hover 后处理：距离变换相对 max 的中心种子阈值')
    parser.add_argument('--model', choices=['cnndcan', 'glas_unet'], default='cnndcan')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = load_models(args.weights_dir, args.folds, device, model=args.model)
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
        arr = np.array(img, dtype=np.float32) / 255.0
        arr, geom = letterbox_resize(arr, args.target_size, cv2.INTER_LINEAR)
        t = torch.from_numpy(np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))).unsqueeze(0).float()

        seg_prob, cont_prob, dist_prob = predict_image(models, t, device)
        if args.method == 'watershed':
            mask_small = postprocess_watershed(seg_prob, cont_prob,
                                               seg_thresh=args.seg_thresh,
                                               fg_dist_thresh=args.fg_dist_thresh,
                                               min_area=args.min_area)
        elif args.method == 'hover':
            mask_small = postprocess_hover(seg_prob, dist_prob,
                                           seg_thresh=args.seg_thresh,
                                           min_area=args.min_area,
                                           fg_dist_thresh=args.fg_dist_thresh)
        else:
            mask_small = postprocess(seg_prob, cont_prob,
                                     seg_thresh=args.seg_thresh,
                                     cont_thresh=args.cont_thresh,
                                     min_area=args.min_area)
        mask_full = unletterbox(mask_small, geom, W, H)
        Image.fromarray(mask_full).save(out_dir / f"{name}.png")

    print(f"推理完成，共 {len(test_names)} 张 → {out_dir}")


if __name__ == '__main__':
    main()