"""
PUMA engines: multi-class loss functions, metrics, and training/evaluation loops
for 5-class tissue semantic segmentation (6 classes including background).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from datasets_puma import NUM_CLASSES, CLASS_NAMES


# ==================== Loss Functions ====================

class MultiClassDiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation (excludes background).

    Supports per-class weighting to boost rare classes.
    """

    def __init__(self, smooth=1.0, ignore_background=True, class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.class_weights = class_weights

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        n_classes = logits.size(1)
        target_onehot = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * target_onehot).sum(dim=(2, 3))  # BxC
        union = probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))  # BxC
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # BxC

        if self.ignore_background:
            dice = dice[:, 1:]
            if self.class_weights is not None:
                w = self.class_weights[1:].to(dice.device)
                w = w / w.sum()
                return 1.0 - (dice * w.unsqueeze(0)).sum(dim=1).mean()

        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Multi-class Focal loss with per-class alpha."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha.clone().detach() if torch.is_tensor(alpha) else torch.tensor(alpha))
        else:
            self.alpha = None

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[target]
            focal = alpha_t * focal
        return focal.mean()


class CEDiceLoss(nn.Module):
    """Combined CE/Focal + Dice + optional Lovász loss with class weighting."""

    def __init__(self, dice_weight=1.0, ce_weight=1.0, smooth=1.0,
                 class_weights=None, use_focal=False, focal_gamma=2.0,
                 lovasz_weight=0.0, label_smoothing=0.0):
        super().__init__()
        self.dice_loss = MultiClassDiceLoss(smooth=smooth, class_weights=class_weights)
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        if use_focal:
            self.ce_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        if lovasz_weight > 0:
            self.lovasz_loss = LovaszSoftmaxLoss()
        else:
            self.lovasz_loss = None

    def forward(self, logits, target):
        loss = self.ce_weight * self.ce_loss(logits, target) + \
               self.dice_weight * self.dice_loss(logits, target)
        if self.lovasz_loss is not None:
            loss = loss + self.lovasz_weight * self.lovasz_loss(logits, target)
        return loss


class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax loss for multi-class segmentation.

    Directly optimizes the mean of IoU across classes.
    Based on Berman et al., "Wildcat: Multi-pooling for semantic segmentation" (2018).
    """

    def __init__(self, classes='present', ignore_index=None):
        super().__init__()
        self.classes = classes
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        return self._lovasz_softmax(probs, target, self.classes, self.ignore_index)

    def _lovasz_softmax(self, probs, labels, classes='present', ignore=None):
        if probs.dim() == 2:
            probs = probs.unsqueeze(0)
        total_loss = 0.0
        batch_size = probs.size(0)

        for b in range(batch_size):
            prob = probs[b]
            lab = labels[b]
            if ignore is not None:
                mask = lab != ignore
                lab = lab[mask]
                prob = prob.permute(1, 2, 0)[mask]
            else:
                prob = prob.permute(1, 2, 0).contiguous().view(-1, prob.size(0))
            lab = lab.contiguous().view(-1)

            if classes == 'present':
                class_set = lab.unique()
            else:
                class_set = torch.arange(prob.size(1))

            losses = 0.0
            n_classes = 0
            for c in class_set:
                if c.item() == 0 and classes == 'present':
                    continue
                fg = (lab == c).float()
                if fg.sum() == 0:
                    continue
                class_prob = prob[:, c.item()]
                losses += self._lovasz_flat(class_prob, fg)
                n_classes += 1

            if n_classes > 0:
                total_loss += losses / n_classes

        return total_loss / batch_size

    @staticmethod
    def _lovasz_flat(probs, labels):
        """Binary Lovász loss (flat inputs)."""
        if probs.numel() == 0:
            return 0.0
        if labels.sum() == 0:
            return 0.0
        signs = 2.0 * labels.float() - 1.0
        errors = (1.0 - probs * signs).clamp(min=0)
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = labels[perm]
        grad = LovaszSoftmaxLoss._lovasz_grad(gt_sorted)
        return torch.dot(grad, errors_sorted)

    @staticmethod
    def _lovasz_grad(gt_sorted):
        """Compute gradient of the Lovász extension."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard


# ==================== Metrics ====================

def compute_metrics_puma(pred, target, num_classes=NUM_CLASSES):
    """Compute per-sample multi-class metrics.

    Args:
        pred: HxW predicted class indices
        target: HxW ground truth class indices

    Returns:
        dict with per-class Dice, mean Dice, and pixel accuracy
    """
    metrics = {}
    dice_scores = []

    for c in range(1, num_classes):  # Skip background
        pred_c = (pred == c)
        target_c = (target == c)

        intersection = (pred_c & target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union == 0:
            dice = 1.0  # Both empty -> perfect
        elif intersection == 0:
            dice = 0.0
        else:
            dice = 2.0 * intersection / (union + 1e-6)

        metrics['dice_' + CLASS_NAMES[c]] = float(dice)
        dice_scores.append(dice)

    metrics['mean_dice'] = float(np.mean(dice_scores)) if dice_scores else 0.0
    metrics['pixel_acc'] = float((pred == target).astype(np.float32).mean())

    return metrics


# ==================== Training Loop ====================

def train_one_epoch_puma(model, dataloader, criterion, optimizer, device, scaler=None,
                         grad_accum_steps=1, use_amp=True):
    """Train one epoch for PUMA multi-class segmentation.

    Args:
        grad_accum_steps: Number of batches to accumulate gradients before optimizer step.
        use_amp: Whether to use automatic mixed precision.
    """
    model.train()
    total_loss = 0.0
    all_metrics = {}
    n_samples = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device).float()
        labels = batch['label'].to(device).long()  # BxHxW class indices

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)  # BxCxHxW logits
                loss = criterion(outputs, labels)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / grad_accum_steps
            loss.backward()

        # Step optimizer after accumulating enough gradients
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * images.size(0)
        n_samples += images.size(0)

        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        for i in range(images.size(0)):
            metrics = compute_metrics_puma(preds[i], labels_np[i])
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)

    avg_loss = total_loss / n_samples
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


# ==================== Evaluation ====================

def evaluate_puma(model, dataloader, device, use_tta=False):
    """Evaluate PUMA model.

    Computes both macro Dice (per-sample average) and micro Dice (official metric:
    pool all predictions, then compute per-class Dice).
    """
    model.eval()

    # Accumulate for micro Dice (official metric)
    class_intersection = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_union = np.zeros(NUM_CLASSES, dtype=np.float64)
    pixel_correct = 0
    pixel_total = 0

    # Per-sample metrics (macro)
    all_metrics = {}

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device).float()
            labels = batch['label'].to(device).long()

            outputs = model(images)

            if use_tta:
                outputs_h = model(torch.flip(images, dims=[3]))
                outputs_h = torch.flip(outputs_h, dims=[3])
                outputs_v = model(torch.flip(images, dims=[2]))
                outputs_v = torch.flip(outputs_v, dims=[2])
                outputs = (outputs + outputs_h + outputs_v) / 3.0

            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            for i in range(images.size(0)):
                pred = preds[i]
                target = labels_np[i]

                # Per-sample metrics (macro)
                metrics = compute_metrics_puma(pred, target)
                for k, v in metrics.items():
                    all_metrics.setdefault(k, []).append(v)

                # Accumulate for micro Dice
                for c in range(1, NUM_CLASSES):
                    pred_c = (pred == c)
                    target_c = (target == c)
                    class_intersection[c] += (pred_c & target_c).sum()
                    class_union[c] += pred_c.sum() + target_c.sum()

                pixel_correct += (pred == target).sum()
                pixel_total += pred.size

    # Compute micro Dice (official PUMA metric)
    micro_dices = []
    for c in range(1, NUM_CLASSES):
        if class_union[c] == 0:
            micro_dices.append(1.0)
        elif class_intersection[c] == 0:
            micro_dices.append(0.0)
        else:
            micro_dices.append(2.0 * class_intersection[c] / (class_union[c] + 1e-6))

    result = {k: np.mean(v) for k, v in all_metrics.items()}
    result['micro_dice'] = float(np.mean(micro_dices))
    result['pixel_acc'] = float(pixel_correct / pixel_total)

    for i, c in enumerate(range(1, NUM_CLASSES)):
        result['micro_dice_' + CLASS_NAMES[c]] = float(micro_dices[i])

    return result
