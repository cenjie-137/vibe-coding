import torch
import torch.nn as nn
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        intersection = (input * target).sum()
        union = input.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.6, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        tp = (input * target).sum()
        fp = (input * (1 - target)).sum()
        fn = ((1 - input) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        bce = nn.functional.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()

def post_process(pred_2d, min_size=30):
    """去除小连通域噪点，保留血管。输入必须是二维。"""
    pred_uint8 = pred_2d.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_uint8, connectivity=8)
    cleaned = np.zeros_like(pred_2d)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 1
    return cleaned

def compute_metrics(pred, target, mask=None, use_postprocess=False):
    if use_postprocess:
        # 后处理在二维图像上进行（必须先做，因为 flatten 后无法做连通域分析）
        pred_bin_2d = (pred > 0.5).astype(np.float32)
        pred_bin_2d = post_process(pred_bin_2d, min_size=30)
        pred_prob = pred_bin_2d.flatten()
    else:
        pred_prob = pred.flatten()

    target = target.flatten()

    if mask is not None:
        mask = mask.flatten()
        pred_prob = pred_prob[mask > 0]
        target = target[mask > 0]

    pred_bin = (pred_prob > 0.5).astype(np.float32)

    tp = np.sum(pred_bin * target)
    tn = np.sum((1 - pred_bin) * (1 - target))
    fp = np.sum(pred_bin * (1 - target))
    fn = np.sum((1 - pred_bin) * target)

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if len(np.unique(target)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(target, pred_prob)

    intersection = np.sum(pred_bin * target)
    union = np.sum(pred_bin) + np.sum(target)
    dice = (2.0 * intersection) / (union + 1e-6) if union > 0 else 0.0

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc,
        'dice': dice
    }

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, ds_weights=None):
    """训练一个 epoch。
    ds_weights: Deep Supervision 权重，如 [0.2, 0.3, 0.4] 对应 aux2/aux3/aux4。
                为 None 时禁用深监督。"""
    model.train()
    total_loss = 0.0
    all_metrics = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'auc': [], 'dice': []}

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device)

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

        preds = torch.sigmoid(main_out).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()

        for i in range(images.size(0)):
            metrics = compute_metrics(preds[i, 0], labels_np[i, 0], masks_np[i, 0])
            for k, v in metrics.items():
                all_metrics[k].append(v)

    avg_loss = total_loss / len(dataloader.dataset)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}

    return avg_loss, avg_metrics

def evaluate(model, dataloader, device, use_tta=False, use_postprocess=False):
    model.eval()
    all_metrics = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'auc': [], 'dice': []}

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)

            if use_tta:
                # TTA: 水平翻转 + 垂直翻转
                images_h = torch.flip(images, dims=[3])
                outputs_h = model(images_h)
                outputs_h = torch.flip(outputs_h, dims=[3])

                images_v = torch.flip(images, dims=[2])
                outputs_v = model(images_v)
                outputs_v = torch.flip(outputs_v, dims=[2])

                outputs = (outputs + outputs_h + outputs_v) / 3.0

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()

            for i in range(images.size(0)):
                metrics = compute_metrics(preds[i, 0], labels_np[i, 0], masks_np[i, 0], use_postprocess=use_postprocess)
                for k, v in metrics.items():
                    all_metrics[k].append(v)

    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    return avg_metrics
