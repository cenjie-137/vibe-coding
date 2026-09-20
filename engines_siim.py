import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score


class DiceLoss(nn.Module):
    """Soft Dice Loss，只在有气胸样本上计算"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        intersection = (input * target).sum(dim=(2, 3))
        union = input.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # 只对有气胸的样本计算
        has_pos = (target.sum(dim=(2, 3)) > 0).float().squeeze(1)
        if has_pos.sum() > 0:
            return 1.0 - (dice.squeeze(1) * has_pos).sum() / has_pos.sum()
        return torch.tensor(0.0, device=input.device, requires_grad=True)


class TverskyLoss(nn.Module):
    """Tversky Loss，只在有气胸样本上计算"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        tp = (input * target).sum(dim=(2, 3))
        fp = (input * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - input) * target).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        has_pos = (target.sum(dim=(2, 3)) > 0).float().squeeze(1)
        if has_pos.sum() > 0:
            return 1.0 - (tversky.squeeze(1) * has_pos).sum() / has_pos.sum()
        return torch.tensor(0.0, device=input.device, requires_grad=True)


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        bce = nn.functional.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice 组合损失
    BCE 对所有样本（包括空白）计算，Dice 只在有气胸样本上计算"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        bce_loss = self.bce(input, target)
        input_sig = torch.sigmoid(input)
        intersection = (input_sig * target).sum(dim=(2, 3))
        union = input_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        has_pos = (target.sum(dim=(2, 3)) > 0).float().squeeze(1)
        if has_pos.sum() > 0:
            dice_loss = 1.0 - (dice.squeeze(1) * has_pos).sum() / has_pos.sum()
        else:
            dice_loss = torch.tensor(0.0, device=input.device, requires_grad=True)
        return bce_loss + dice_loss


def compute_metrics_siim(pred, target):
    """计算单样本指标
    pred: H×W 的概率图（0~1）
    target: H×W 的二值掩码（0/1）
    返回: (metrics_dict, has_pneumo)  -- has_pneumo 表示该样本是否有气胸
    """
    pred_prob = pred.flatten()
    target_flat = target.flatten()
    pred_bin = (pred_prob > 0.5).astype(np.float32)

    has_pneumo = target_flat.sum() > 0

    tp = np.sum(pred_bin * target_flat)
    tn = np.sum((1 - pred_bin) * (1 - target_flat))
    fp = np.sum(pred_bin * (1 - target_flat))
    fn = np.sum((1 - pred_bin) * target_flat)

    # Dice
    intersection = np.sum(pred_bin * target_flat)
    union = np.sum(pred_bin) + np.sum(target_flat)
    dice = (2.0 * intersection) / (union + 1e-6) if union > 0 else 0.0

    # IoU
    iou = intersection / (union - intersection + 1e-6) if union > 0 else 0.0

    # Sensitivity（Recall）
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC
    if len(np.unique(target_flat)) < 2:
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(target_flat, pred_prob)
        except ValueError:
            auc = 0.5

    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc
    }, has_pneumo


def _aggregate_metrics(all_metrics, all_has_pneumo):
    """汇总指标：区分有/无气胸样本，Dice/IoU/Sens 只在有气胸样本上算"""
    pos_metrics = {k: [v[i] for i, hp in enumerate(all_has_pneumo) if hp]
                   for k, v in all_metrics.items()}
    neg_metrics = {k: [v[i] for i, hp in enumerate(all_has_pneumo) if not hp]
                   for k, v in all_metrics.items()}

    result = {}
    # 竞赛核心指标：只在有气胸样本上算
    for k in ['dice', 'iou', 'sensitivity']:
        vals = pos_metrics[k]
        result[k] = np.mean(vals) if len(vals) > 0 else 0.0

    # AUC：全部样本都算
    result['auc'] = np.mean(all_metrics['auc']) if len(all_metrics['auc']) > 0 else 0.0
    # Specificity：全部样本都算
    result['specificity'] = np.mean(all_metrics['specificity']) if len(all_metrics['specificity']) > 0 else 0.0

    # 附加信息
    result['pos_count'] = len(pos_metrics['dice'])
    result['neg_count'] = len(neg_metrics['dice'])
    result['pos_dice'] = result['dice']
    result['neg_dice'] = 0.0  # 空白样本 Dice 无意义

    return result


def train_one_epoch_siim(model, dataloader, criterion, optimizer, device, scaler=None, ds_weights=None):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    all_metrics = {'dice': [], 'iou': [], 'sensitivity': [], 'specificity': [], 'auc': []}
    all_has_pneumo = []
    n_samples = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                if ds_weights is not None and isinstance(outputs, (tuple, list)):
                    main_out = outputs[0]
                    loss = criterion(main_out, labels)
                    for w, aux_out in zip(ds_weights, outputs[1:]):
                        loss += w * criterion(aux_out, labels)
                else:
                    main_out = outputs
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if ds_weights is not None and isinstance(outputs, (tuple, list)):
                main_out = outputs[0]
                loss = criterion(main_out, labels)
                for w, aux_out in zip(ds_weights, outputs[1:]):
                    loss += w * criterion(aux_out, labels)
            else:
                main_out = outputs
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

        preds = torch.sigmoid(main_out).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        for i in range(images.size(0)):
            metrics, has_pneumo = compute_metrics_siim(preds[i, 0], labels_np[i, 0])
            for k, v in metrics.items():
                all_metrics[k].append(v)
            all_has_pneumo.append(has_pneumo)

    avg_loss = total_loss / n_samples
    avg_metrics = _aggregate_metrics(all_metrics, all_has_pneumo)
    return avg_loss, avg_metrics


def evaluate_siim(model, dataloader, device, use_tta=False):
    """评估 - 核心指标（Dice/IoU/Sens）只在有气胸样本上计算"""
    model.eval()
    all_metrics = {'dice': [], 'iou': [], 'sensitivity': [], 'specificity': [], 'auc': []}
    all_has_pneumo = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)

            if use_tta:
                outputs_h = model(torch.flip(images, dims=[3]))
                outputs_h = torch.flip(outputs_h, dims=[3])
                outputs_v = model(torch.flip(images, dims=[2]))
                outputs_v = torch.flip(outputs_v, dims=[2])
                outputs = (outputs + outputs_h + outputs_v) / 3.0

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            for i in range(images.size(0)):
                metrics, has_pneumo = compute_metrics_siim(preds[i, 0], labels_np[i, 0])
                for k, v in metrics.items():
                    all_metrics[k].append(v)
                all_has_pneumo.append(has_pneumo)

    avg_metrics = _aggregate_metrics(all_metrics, all_has_pneumo)
    return avg_metrics
