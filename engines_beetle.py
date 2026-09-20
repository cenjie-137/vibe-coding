import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets_beetle import CLASSES


# ===================== 多分类损失函数 =====================

class MultiClassDiceLoss(nn.Module):
    """多分类 Dice 损失：对每个类别算 soft Dice，然后平均"""

    def __init__(self, smooth=1e-5, weight=None):
        super().__init__()
        self.smooth = smooth
        self.weight = weight  # 每个类别的权重

    def forward(self, logits, target):
        """
        logits: (B, C, H, W)
        target: (B, H, W) 类别的整数索引
        """
        probs = F.softmax(logits, dim=1)
        C = logits.size(1)
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        loss = 0.0
        for c in range(C):
            p = probs[:, c]
            t = target_one_hot[:, c]
            intersection = (p * t).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            class_loss = 1 - dice.mean()
            if self.weight is not None:
                class_loss = class_loss * self.weight[c]
            loss = loss + class_loss

        if self.weight is not None:
            loss = loss / sum(self.weight)
        else:
            loss = loss / C
        return loss


class MultiClassFocalLoss(nn.Module):
    """多分类 Focal 损失"""

    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # (C,) 类别权重
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        C = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        mask = (target != self.ignore_index).long()
        target_oh = F.one_hot(target.clamp_min(0), num_classes=C).permute(0, 3, 1, 2).float()
        target_oh = target_oh * mask.unsqueeze(1)

        pt = (probs * target_oh).sum(dim=1)
        log_pt = (log_probs * target_oh).sum(dim=1)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = (self.alpha.view(1, C, 1, 1) * target_oh).sum(dim=1)
            loss = -alpha_t * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        valid = mask.sum().clamp_min(1)
        return loss.sum() / valid


class MultiClassCombinedLoss(nn.Module):
    """组合损失：CrossEntropy + Dice×2 + Focal"""

    def __init__(self, dice_weight=2.0, focal_weight=1.0, ce_weight=1.0,
                 class_weights=None, focal_alpha=None, focal_gamma=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight

        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        self.dice = MultiClassDiceLoss(weight=class_weights)
        self.focal = MultiClassFocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits, target):
        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        loss_focal = self.focal(logits, target)
        loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice + self.focal_weight * loss_focal
        return loss, {
            'ce': loss_ce.item(),
            'dice': loss_dice.item(),
            'focal': loss_focal.item(),
        }


# ===================== 多分类评估指标 =====================

def compute_multiclass_metrics(preds_probs, target, num_classes=4):
    """
    preds_probs: (B, C, H, W) softmax 概率
    target: (B, H, W) 类别索引
    """
    pred_labels = preds_probs.argmax(dim=1).cpu().numpy()
    target_np = target.cpu().numpy()
    C = num_classes

    per_class_dice = np.zeros(C)
    per_class_iou = np.zeros(C)
    valid_classes = 0

    for c in range(C):
        pred_c = (pred_labels == c)
        targ_c = (target_np == c)

        if targ_c.sum() == 0 and pred_c.sum() == 0:
            per_class_dice[c] = 1.0
            per_class_iou[c] = 1.0
        elif targ_c.sum() == 0 and pred_c.sum() > 0:
            per_class_dice[c] = 0.0
            per_class_iou[c] = 0.0
        else:
            intersection = (pred_c & targ_c).sum()
            union = pred_c.sum() + targ_c.sum()
            iou = intersection / (union - intersection + 1e-7) if (union - intersection) > 0 else 1.0
            dice = 2 * intersection / (union + 1e-7) if union > 0 else 1.0
            per_class_dice[c] = dice
            per_class_iou[c] = iou

        valid_classes += 1

    overall_dice = per_class_dice.mean()
    overall_iou = per_class_iou.mean()
    accuracy = (pred_labels == target_np).mean()

    return {
        'accuracy': accuracy,
        'overall_dice': overall_dice,
        'overall_iou': overall_iou,
        'per_class_dice': per_class_dice.tolist(),
        'per_class_iou': per_class_iou.tolist(),
    }


# ===================== 训练/验证循环 =====================

def train_one_epoch_beetle(model, loader, loss_fn, optimizer, device, scaler, ds_weights=None):
    model.train()
    total_loss = 0.0
    ce_loss_sum = dice_loss_sum = focal_loss_sum = 0.0
    all_probs, all_targets = [], []
    n_batches = len(loader)

    pbar = tqdm(loader, desc="Train", disable=False)
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['mask'].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            if ds_weights is not None:
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):
                    loss_total = 0.0
                    loss_dict = None
                    for idx, out in enumerate(outputs):
                        w = ds_weights[idx]
                        loss, ld = loss_fn(out, targets)
                        loss_total = loss_total + w * loss
                        if loss_dict is None:
                            loss_dict = {k: v for k, v in ld.items()}
                    loss = loss_total / sum(ds_weights)
                else:
                    loss, loss_dict = loss_fn(outputs, targets)
                probs = F.softmax(outputs if not isinstance(outputs, (list, tuple)) else outputs[0], dim=1)
            else:
                outputs = model(images)
                loss, loss_dict = loss_fn(outputs, targets)
                probs = F.softmax(outputs, dim=1)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        ce_loss_sum += loss_dict.get('ce', 0)
        dice_loss_sum += loss_dict.get('dice', 0)
        focal_loss_sum += loss_dict.get('focal', 0)

        all_probs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_multiclass_metrics(all_probs, all_targets, num_classes=4)

    metrics['loss'] = total_loss / n_batches
    metrics['ce_loss'] = ce_loss_sum / n_batches
    metrics['dice_loss'] = dice_loss_sum / n_batches
    metrics['focal_loss'] = focal_loss_sum / n_batches
    return metrics


def validate_one_epoch_beetle(model, loader, loss_fn, device, ds_weights=None):
    model.eval()
    total_loss = 0.0
    ce_loss_sum = dice_loss_sum = focal_loss_sum = 0.0
    all_probs, all_targets = [], []
    n_batches = len(loader)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Valid", disable=False):
            images = batch['image'].to(device)
            targets = batch['mask'].to(device)

            with torch.cuda.amp.autocast():
                if ds_weights is not None:
                    outputs = model(images)
                    if isinstance(outputs, (list, tuple)):
                        loss_total = 0.0
                        loss_dict = None
                        for idx, out in enumerate(outputs):
                            w = ds_weights[idx]
                            loss, ld = loss_fn(out, targets)
                            loss_total = loss_total + w * loss
                            if loss_dict is None:
                                loss_dict = {k: v for k, v in ld.items()}
                        loss = loss_total / sum(ds_weights)
                    else:
                        loss, loss_dict = loss_fn(outputs, targets)
                    probs = F.softmax(outputs if not isinstance(outputs, (list, tuple)) else outputs[0], dim=1)
                else:
                    outputs = model(images)
                    loss, loss_dict = loss_fn(outputs, targets)
                    probs = F.softmax(outputs, dim=1)

            total_loss += loss.item()
            ce_loss_sum += loss_dict.get('ce', 0)
            dice_loss_sum += loss_dict.get('dice', 0)
            focal_loss_sum += loss_dict.get('focal', 0)

            all_probs.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_multiclass_metrics(all_probs, all_targets, num_classes=4)

    metrics['loss'] = total_loss / n_batches
    metrics['ce_loss'] = ce_loss_sum / n_batches
    metrics['dice_loss'] = dice_loss_sum / n_batches
    metrics['focal_loss'] = focal_loss_sum / n_batches
    return metrics


if __name__ == '__main__':
    from datasets_beetle import get_dataloader
    from models import MultiResUNet

    data_dir = r"C:\Users\32201\Desktop\T\vibe coding\BEETLE\data\BEETLE"
    if not os.path.exists(data_dir):
        data_dir = '/root/autodl-tmp/medseg_project/data/BEETLE'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_dataloader(data_dir, split='train', batch_size=2, augment=False)
    model = MultiResUNet(n_channels=3, n_classes=4, base_ch=64, attention='se').to(device)
    loss_fn = MultiClassCombinedLoss(
        dice_weight=2.0, focal_weight=1.0, ce_weight=1.0,
        class_weights=torch.tensor([1.0, 1.5, 2.0, 0.5], dtype=torch.float32).to(device),
        focal_alpha=torch.tensor([1.0, 1.5, 2.0, 0.5], dtype=torch.float32).to(device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    for batch in loader:
        break
    metrics = train_one_epoch_beetle(model, loader, loss_fn, optimizer, device, scaler)
    print("Train metrics:", {k: (v[:4] if isinstance(v, list) else f"{v:.4f}") for k, v in metrics.items()})
