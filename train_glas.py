"""train_glas.py — GLaS 5-fold 训练（DCAN 式双头）

默认模型 cnndcan：ResNet34(ImageNet) 编码器 + skip 解码 + seg/contour 双头 + 多级深监督。
可用 --model glas_unet 切回旧的 UNI ViT 版本做对照。
损失：seg = BCE + 2×Dice；contour = BCE(pos_weight) + Dice；total = seg + 0.5×contour（+ 深监督副头）
选择最优模型：基于验证集 object-level Dice（TTA 后）。
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from eval_glas import object_dice, object_f1
from dataset_glas import get_glas_dataloaders
from models_glas import GlandUNet, GlandCNNDCAN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_loss(logits, target, smooth=1.0):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum(dim=(2, 3)) + smooth
    den = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
    return (1 - num / den).mean()


def seg_contour_loss(seg_logit, seg, cont_logit, cont):
    seg = seg.unsqueeze(1)
    cont = cont.unsqueeze(1)
    seg_bce = F.binary_cross_entropy_with_logits(seg_logit, seg)
    seg_dice = dice_loss(seg_logit, seg)
    seg_loss = seg_bce + 2.0 * seg_dice

    pos = cont.mean()
    pos_weight = ((1.0 - pos) / (pos + 1e-6)).clamp(max=50.0)
    cont_bce = F.binary_cross_entropy_with_logits(cont_logit, cont, pos_weight=pos_weight)
    cont_dice = dice_loss(cont_logit, cont)
    cont_loss = cont_bce + cont_dice

    return seg_loss + 0.5 * cont_loss, seg_loss, cont_loss


def hov_loss(dist_pred, dist_gt, seg, msge_weight=1.0):
    """HoVer 距离图回归损失 = 前景 MSE + λ·MSGE（梯度一致性）。

    dist_pred/dist_gt: (B,2,H,W) 全局归一化像素距离；seg: (B,H,W) {0,1}。
    MSGE 约束预测距离场的空间梯度与 GT 一致（等价于强制单位梯度），
    使方向场梯度能量在粘连边界处形成一致切割山脊（HoVer-Net 关键项）。
    """
    dist_pred = dist_pred.float()
    dist_gt = dist_gt.float()
    fg = (seg > 0.5).float().unsqueeze(1)  # (B,1,H,W)
    n = fg.sum() * 2.0 + 1e-6  # 每个前景像素 2 个通道
    mse = ((dist_pred - dist_gt) ** 2 * fg).sum() / n

    gy, gx = torch.gradient(dist_pred, dim=(2, 3))
    tgy, tgx = torch.gradient(dist_gt, dim=(2, 3))
    ge = (gy - tgy) ** 2 + (gx - tgx) ** 2
    msge = (ge * fg).sum() / n
    return mse + msge_weight * msge


@torch.no_grad()
def predict_prob(model, image, device, tta=True):
    model.eval()
    variants = [image]
    if tta:
        variants += [torch.flip(image, dims=[3]),
                     torch.flip(image, dims=[2]),
                     torch.flip(image, dims=[2, 3])]
    seg_sum = None
    cont_sum = None
    dist_out = None
    for i, x in enumerate(variants):
        out = model(x.to(device))
        if len(out) == 3:
            seg_logit, cont_logit, dist_t = out
        else:
            seg_logit, cont_logit = out
            dist_t = None
        seg_p = torch.sigmoid(seg_logit)
        cont_p = torch.sigmoid(cont_logit)
        if i == 1:
            seg_p, cont_p = torch.flip(seg_p, [3]), torch.flip(cont_p, [3])
        elif i == 2:
            seg_p, cont_p = torch.flip(seg_p, [2]), torch.flip(cont_p, [2])
        elif i == 3:
            seg_p, cont_p = torch.flip(seg_p, [2, 3]), torch.flip(cont_p, [2, 3])
        seg_sum = seg_p if seg_sum is None else seg_sum + seg_p
        cont_sum = cont_p if cont_sum is None else cont_sum + cont_p
        # 距离图向量翻转会改方向符号，只取原始方向（不做 TTA 平均）
        if i == 0 and dist_t is not None:
            dist_out = dist_t[0].cpu().numpy()  # (2,H,W)
    seg_prob = (seg_sum / len(variants))[0, 0].cpu().numpy()
    cont_prob = (cont_sum / len(variants))[0, 0].cpu().numpy()
    return seg_prob, cont_prob, dist_out


def evaluate(model, val_loader, device):
    from inference_glas import postprocess, postprocess_hover
    model.eval()
    pix_dices, obj_dices, obj_f1s = [], [], []
    for batch in val_loader:
        images = batch['image'].to(device)
        segs = batch['seg'].numpy()
        for b in range(images.shape[0]):
            seg_prob, cont_prob, dist_prob = predict_prob(model, images[b:b + 1], device, tta=True)
            if dist_prob is not None:
                pred = postprocess_hover(seg_prob, dist_prob, 0.5, 10).astype(bool)
            else:
                pred = postprocess(seg_prob, cont_prob, 0.5, 0.5, 10).astype(bool)
            gt = segs[b] > 0.5
            inter = (pred & gt).sum()
            pix_dices.append(2 * inter / (pred.sum() + gt.sum() + 1e-6))
            obj_dices.append(object_dice(pred, gt))
            obj_f1s.append(object_f1(pred, gt)[0])
    return (float(np.mean(pix_dices)), float(np.mean(obj_dices)), float(np.mean(obj_f1s)))


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train_fold(args, fold, device):
    train_loader, val_loader, train_files, val_files = get_glas_dataloaders(
        args.data_dir, fold=fold, total_folds=args.total_folds,
        batch_size=args.batch_size, target_size=args.target_size,
        num_workers=args.num_workers,
    )

    if args.model == 'cnndcan':
        model = GlandCNNDCAN()
    else:
        model = GlandUNet(model_source=args.model_source, freeze_encoder=args.freeze_encoder)
    model.to(device)

    encoder_params = [p for n, p in model.named_parameters()
                      if n.startswith('encoder.') and p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith('encoder.') and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': decoder_params, 'lr': args.lr},
        {'params': encoder_params, 'lr': args.lr * args.encoder_lr_factor},
    ], weight_decay=args.weight_decay)

    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    best_obj_dice = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            segs = batch['seg'].to(device)
            contours = batch['contour'].to(device)
            dists = batch['dist'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                out = model(images)
                if args.model == 'cnndcan':
                    seg_logit, cont_logit, aux = out
                    loss, seg_loss, cont_loss = seg_contour_loss(seg_logit, segs, cont_logit, contours)
                    if args.deep_sup:
                        for a_seg, a_cont in aux:
                            loss = loss + args.deep_sup_weight * seg_contour_loss(
                                a_seg, segs, a_cont, contours)[0]
                else:
                    seg_logit, cont_logit, dist = out
                    loss = seg_contour_loss(seg_logit, segs, cont_logit, contours)[0]
                    loss = loss + args.hov_weight * hov_loss(dist, dists, segs, args.hov_msge_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        pix_dice, obj_dice, obj_f1 = evaluate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"[Fold {fold} E{epoch + 1}/{args.epochs}] loss={running_loss / len(train_loader):.4f} "
              f"pixD={pix_dice:.4f} objD={obj_dice:.4f} objF1={obj_f1:.4f} lr={lr_now:.2e}")

        if obj_dice > best_obj_dice:
            best_obj_dice = obj_dice
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'obj_dice': obj_dice,
                'obj_f1': obj_f1,
                'pix_dice': pix_dice,
                'fold': fold,
                'train_files': train_files,
                'val_files': val_files,
            }, os.path.join(args.output_dir, f"best_fold_{fold}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    print(f"Fold {fold} done: best objD={best_obj_dice:.4f} @ epoch {best_epoch + 1}")
    return best_obj_dice


def main():
    parser = argparse.ArgumentParser(description="GLaS 5-fold training")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--total_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--target_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--encoder_lr_factor', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--model_source', default='UNI')
    parser.add_argument('--model', choices=['cnndcan', 'glas_unet'], default='cnndcan')
    parser.add_argument('--deep_sup', type=int, default=1, choices=[0, 1],
                        help='cnndcan 是否启用多级深监督（消融用：0=关）')
    parser.add_argument('--deep_sup_weight', type=float, default=0.4,
                        help='深监督辅助损失权重')
    parser.add_argument('--hov_weight', type=float, default=1.0,
                        help='HoVer 距离图回归(MSE)损失权重（glas_unet 三头时生效）')
    parser.add_argument('--hov_msge_weight', type=float, default=1.0,
                        help='HoVer 梯度一致性(MSGE)损失权重（glas_unet 三头时生效）')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}, folds={args.folds}")

    results = {}
    for fold in args.folds:
        obj_dice = train_fold(args, fold, device)
        results[str(fold)] = obj_dice

    with open(os.path.join(args.output_dir, 'folds_result.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("=" * 60)
    print("Fold objDice 汇总:")
    for k, v in results.items():
        print(f"  fold {k}: {v:.4f}")
    print(f"  mean: {np.mean(list(results.values())):.4f}")


if __name__ == '__main__':
    main()