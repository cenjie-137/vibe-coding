"""
PUMA training script: 5-fold cross-validation with early stopping for
5-class tissue semantic segmentation.

Usage:
    # Train all 5 folds
    python train_puma.py --folds 0 1 2 3 4 --base_ch 48 --batch_size 4 --target_size 512

    # Train single fold with Focal loss
    python train_puma.py --folds 0 --epochs 50 --use_focal

    # Run mask preprocessing first, then train
    python train_puma.py --preprocess --folds 0
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets_puma import (
    get_puma_dataloaders, preprocess_masks, compute_class_weights,
    NUM_CLASSES, CLASS_NAMES
)
from models import MultiResUNet, FoundationUNet
from engines_puma import CEDiceLoss, train_one_epoch_puma, evaluate_puma
import torch.nn as nn
import torch.nn.functional as F

scaler = torch.cuda.amp.GradScaler(enabled=True)


class SegFormerWrapper(nn.Module):
    """SegFormer-B2 wrapper for PUMA tissue segmentation.

    Uses pretrained ADE20K weights, replaces classification head for 6 classes.
    Upsamples output to input resolution for compatibility with training loop.
    """

    def __init__(self, num_classes=6, model_name='nvidia/segformer-b2-finetuned-ade-512-512'):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        config = SegformerConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        out = self.model(x)
        logits = out.logits
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits


def get_model(model_name, n_classes):
    """Build model by name."""
    if model_name == 'unet_resnet34':
        import segmentation_models_pytorch as smp
        return smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            classes=n_classes,
            activation=None
        )
    elif model_name == 'unet_convnextv2_base':
        import segmentation_models_pytorch as smp
        try:
            model = smp.Unet(
                encoder_name='convnextv2_base',
                encoder_weights='imagenet',
                classes=n_classes,
                activation=None,
                decoder_channels=[256, 128, 64, 32, 16],
            )
        except (KeyError, Exception):
            print("convnextv2_base not supported, falling back to timm-efficientnet-b3", flush=True)
            model = smp.Unet(
                encoder_name='timm-efficientnet-b3',
                encoder_weights='imagenet',
                classes=n_classes,
                activation=None,
                decoder_channels=[256, 128, 64, 32, 16],
            )
        return model
    elif model_name == 'unet_efficientnet_b3':
        import segmentation_models_pytorch as smp
        return smp.Unet(
            encoder_name='timm-efficientnet-b3',
            encoder_weights='imagenet',
            classes=n_classes,
            activation=None,
            decoder_channels=[256, 128, 64, 32, 16],
        )
    elif model_name == 'segformer_b2':
        return SegFormerWrapper(num_classes=n_classes)
    elif model_name == 'foundation_uni':
        return FoundationUNet(num_classes=n_classes, model_source="UNI")
    elif model_name == 'multiresunet':
        return MultiResUNet(n_channels=3, n_classes=n_classes, base_ch=48, attention='se')
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_puma_paths():
    """Get data/output/log paths (server or local)."""
    autodl_root = '/root/autodl-tmp'
    if os.path.exists(autodl_root):
        project_dir = os.path.join(autodl_root, 'medseg_project')
        data_dir = os.path.join(project_dir, 'data', 'PUMA')
        output_dir = os.path.join(project_dir, 'output_puma')
        log_dir = os.path.join(project_dir, 'logs')
    else:
        local_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(local_root), 'medseg_project', 'data', 'PUMA')
        output_dir = os.path.join(os.path.dirname(local_root), 'medseg_project', 'output_puma')
        log_dir = os.path.join(os.path.dirname(local_root), 'medseg_project', 'logs')
    return data_dir, output_dir, log_dir


def train_single_fold(data_dir, output_dir, log_dir, fold, total_folds, device,
                      base_ch=48, attention='se', batch_size=4, target_size=512,
                      num_epochs=50, lr=1e-4, use_focal=True, use_class_weights=True,
                      dice_weight=1.0, ce_weight=1.0, early_stop_patience=50,
                      use_copy_paste=False, model_type='unet_convnextv2_base',
                      lovasz_weight=0.0, label_smoothing=0.1,
                      warmup_epochs=10, use_amp=True, grad_accum_steps=1,
                      strong_augment=True, use_weighted_sampling=True):
    """Train a single fold.

    Default config tuned for ConvNeXt V2 baseline-level performance:
    - ConvNeXt V2 Base encoder (ImageNet pretrained)
    - Focal + Dice loss with class weights
    - Label smoothing 0.1
    - 10 epoch warmup + cosine decay
    - Strong augmentation (elastic, scale, shear, rotation, etc.)
    - Weighted random sampling
    - AMP mixed precision + gradient accumulation
    """
    fold_output_dir = os.path.join(output_dir, 'fold_' + str(fold))

    # Remove old checkpoint (different architecture)
    old_model = os.path.join(fold_output_dir, 'best_model.pth')
    if os.path.exists(old_model):
        os.remove(old_model)
        print("[Fold " + str(fold+1) + "] Removed old checkpoint", flush=True)

    os.makedirs(fold_output_dir, exist_ok=True)

    # Get dataloaders
    train_loader, val_loader, train_files, val_files = get_puma_dataloaders(
        data_dir, fold=fold, total_folds=total_folds,
        batch_size=batch_size, target_size=target_size, num_workers=4,
        use_copy_paste=use_copy_paste, strong_augment=strong_augment,
        use_weighted_sampling=use_weighted_sampling,
    )

    # Compute class weights
    mask_dir = os.path.join(data_dir, 'masks')
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(mask_dir, train_files, NUM_CLASSES).to(device)
        print("[Fold " + str(fold+1) + "] Class weights: " + str(class_weights.cpu().numpy()), flush=True)

    print("\n" + "=" * 80, flush=True)
    print("[Fold " + str(fold+1) + "/" + str(total_folds) + "] Starting training", flush=True)
    if model_type == 'segformer_b2':
        print("Model: SegFormer-B2 (ADE20K pretrained) n_classes=" + str(NUM_CLASSES), flush=True)
    elif model_type == 'unet_resnet34':
        print("Model: U-Net + ResNet34 (ImageNet pretrained) n_classes=" + str(NUM_CLASSES), flush=True)
    elif model_type == 'unet_convnextv2_base':
        print("Model: U-Net + ConvNeXt V2 Base (ImageNet pretrained) n_classes=" + str(NUM_CLASSES), flush=True)
    elif model_type == 'unet_efficientnet_b3':
        print("Model: U-Net + EfficientNet-B3 (ImageNet pretrained) n_classes=" + str(NUM_CLASSES), flush=True)
    elif model_type == 'foundation_uni':
        print("Model: FoundationUNet + UNI (ViT-L/16, pathology pretrained) n_classes=" + str(NUM_CLASSES), flush=True)
    else:
        print("Model: MultiResUNet base_ch=" + str(base_ch) + " attn=" + attention +
              " n_classes=" + str(NUM_CLASSES), flush=True)
    print("Image size: " + str(target_size) + "x" + str(target_size) +
          ", Batch size: " + str(batch_size) + ", LR: " + str(lr), flush=True)
    loss_name = "CE+Focal" if use_focal else "CE"
    lovasz_str = "+Lovasz(w=" + str(lovasz_weight) + ")" if lovasz_weight > 0 else ""
    print("Loss: " + loss_name + "+Dice" + lovasz_str +
          " (ce_w=" + str(ce_weight) + ", dice_w=" + str(dice_weight) + ")", flush=True)
    print("=" * 80, flush=True)

    # Create model
    model = get_model(model_type, NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.1f}M", flush=True)

    # Loss function
    criterion = CEDiceLoss(
        dice_weight=dice_weight, ce_weight=ce_weight,
        class_weights=class_weights, use_focal=use_focal,
        lovasz_weight=lovasz_weight, label_smoothing=label_smoothing
    )

    # Optimizer: differential LR for pretrained encoder
    if model_type == 'foundation_uni' and hasattr(model, 'encoder'):
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        decoder_params = [p for n, p in model.named_parameters()
                         if not n.startswith('encoder.')]
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': lr * 0.1},
            {'params': decoder_params, 'lr': lr},
        ], weight_decay=1e-4)
        enc_lr = lr * 0.1 if encoder_params else 0
        print("Differential LR: encoder=" + str(enc_lr) + " (UNI), decoder=" + str(lr), flush=True)
    elif 'unet_' in model_type and hasattr(model, 'encoder'):
        encoder_params = list(model.encoder.parameters())
        decoder_params = [p for n, p in model.named_parameters()
                         if not n.startswith('encoder.')]
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': lr * 0.1},
            {'params': decoder_params, 'lr': lr},
        ], weight_decay=1e-4)
        print("Differential LR: encoder=" + str(lr * 0.1) + ", decoder=" + str(lr), flush=True)
    elif model_type == 'segformer_b2':
        encoder_params = list(model.model.segformer.encoder.parameters())
        decoder_params = [p for n, p in model.named_parameters()
                         if not n.startswith('model.segformer.encoder')]
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': lr * 0.1},
            {'params': decoder_params, 'lr': lr},
        ], weight_decay=1e-4)
        print("Differential LR: encoder=" + str(lr * 0.1) + ", decoder=" + str(lr), flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Scheduler: warmup + cosine annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # AMP scaler
    train_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training loop
    best_dice = 0.0
    best_micro_dice = 0.0
    best_epoch = 0
    early_stop_counter = 0

    fold_log_file = os.path.join(
        log_dir, "puma_log_fold" + str(fold) + "_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"
    )
    with open(fold_log_file, 'w') as f:
        header = "Epoch\tLoss\tTrain_mD\tTrain_Acc\tLR\t"
        header += "\t".join(["Train_D_" + name for name in CLASS_NAMES[1:]])
        header += "\tVal_mD\tVal_microD\tVal_Acc\t"
        header += "\t".join(["Val_mD_" + name for name in CLASS_NAMES[1:]])
        header += "\n"
        f.write(header)

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_metrics = train_one_epoch_puma(
            model, train_loader, criterion, optimizer, device, train_scaler,
            grad_accum_steps=grad_accum_steps, use_amp=use_amp
        )
        val_metrics = evaluate_puma(model, val_loader, device)
        scheduler.step()

        # Use micro_dice for model selection (official metric)
        is_best = val_metrics['micro_dice'] > best_micro_dice
        if is_best:
            best_dice = val_metrics['mean_dice']
            best_micro_dice = val_metrics['micro_dice']
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(fold_output_dir, 'best_model.pth'))
        else:
            early_stop_counter += 1

        # Log
        with open(fold_log_file, 'a') as f:
            row = (str(epoch) + "\t" + "%.4f" % train_loss + "\t" +
                   "%.4f" % train_metrics['mean_dice'] + "\t" +
                   "%.4f" % train_metrics['pixel_acc'] + "\t" +
                   "%.6f" % current_lr + "\t")
            row += "\t".join(["%.4f" % train_metrics.get('dice_' + name, 0)
                              for name in CLASS_NAMES[1:]])
            row += ("\t" + "%.4f" % val_metrics['mean_dice'] + "\t" +
                    "%.4f" % val_metrics['micro_dice'] + "\t" +
                    "%.4f" % val_metrics['pixel_acc'] + "\t")
            row += "\t".join(["%.4f" % val_metrics.get('micro_dice_' + name, 0)
                              for name in CLASS_NAMES[1:]])
            row += "\n"
            f.write(row)

        if epoch % 5 == 0 or is_best or epoch <= 5:
            print("[Fold " + str(fold+1) + "] Epoch " + str(epoch) + "/" + str(num_epochs) + " | "
                  "Loss: " + "%.4f" % train_loss + " | "
                  "Val mDice: " + "%.4f" % val_metrics['mean_dice'] + " | "
                  "Val microD: " + "%.4f" % val_metrics['micro_dice'] + " | "
                  "LR: " + "%.6f" % current_lr + " | "
                  "Best: " + "%.4f" % best_micro_dice + " @E" + str(best_epoch), flush=True)

        if early_stop_counter >= early_stop_patience:
            print("[Fold " + str(fold+1) + "] Early stopping at epoch " + str(epoch), flush=True)
            break

    # Final TTA evaluation
    model.load_state_dict(torch.load(os.path.join(fold_output_dir, 'best_model.pth')))
    tta_metrics = evaluate_puma(model, val_loader, device, use_tta=True)

    print("\n[Fold " + str(fold+1) + "] Final Results:", flush=True)
    print("  Best macro Dice: " + "%.4f" % best_dice + " @E" + str(best_epoch), flush=True)
    print("  Best micro Dice: " + "%.4f" % best_micro_dice, flush=True)
    print("  TTA macro Dice:  " + "%.4f" % tta_metrics['mean_dice'], flush=True)
    print("  TTA micro Dice:  " + "%.4f" % tta_metrics['micro_dice'], flush=True)

    for name in CLASS_NAMES[1:]:
        d = tta_metrics.get('micro_dice_' + name, 0)
        print("    " + name + ": " + "%.4f" % d, flush=True)

    return {
        'fold': fold,
        'best_dice': best_dice,
        'best_micro_dice': best_micro_dice,
        'best_epoch': best_epoch,
        'tta_dice': tta_metrics['mean_dice'],
        'tta_micro_dice': tta_metrics['micro_dice'],
        'model_path': os.path.join(fold_output_dir, 'best_model.pth')
    }


def main():
    parser = argparse.ArgumentParser(description='PUMA Tissue Segmentation Training')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--total_folds', type=int, default=5)
    parser.add_argument('--base_ch', type=int, default=48)
    parser.add_argument('--attention', type=str, default='se', choices=['se', 'cbam', 'none'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--target_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_focal', action='store_true', default=True,
                        help='Use Focal loss instead of CE')
    parser.add_argument('--no_class_weights', action='store_false', dest='use_class_weights')
    parser.add_argument('--dice_weight', type=float, default=1.0)
    parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default 50 for long training)')
    parser.add_argument('--preprocess', action='store_true', help='Run mask preprocessing first')
    parser.add_argument('--copy_paste', action='store_true',
                        help='Enable copy-paste augmentation for rare classes')
    parser.add_argument('--model', type=str, default='unet_convnextv2_base',
                        choices=['multiresunet', 'unet_resnet34', 'unet_convnextv2_base',
                                 'unet_efficientnet_b3', 'segformer_b2', 'foundation_uni'],
                        help='Model architecture')
    parser.add_argument('--lovasz_weight', type=float, default=0.0,
                        help='Weight for Lovasz-Softmax loss (0=disabled)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (default 0.1)')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP mixed precision training')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = bs * accum)')
    parser.add_argument('--no_strong_aug', action='store_true',
                        help='Disable strong augmentation (use light aug instead)')
    parser.add_argument('--no_weighted_sampling', action='store_true',
                        help='Disable weighted random sampling')
    args = parser.parse_args()

    data_dir, output_dir, log_dir = get_puma_paths()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Check if masks exist, if not run preprocessing
    mask_dir = os.path.join(data_dir, 'masks')
    if not os.path.exists(mask_dir) or len(os.listdir(mask_dir)) == 0 or args.preprocess:
        print("Running mask preprocessing...", flush=True)
        preprocess_masks(data_dir, size=1024)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device), flush=True)
    print("Data dir: " + data_dir, flush=True)
    print("Output dir: " + output_dir, flush=True)

    fold_results = []
    for fold in args.folds:
        result = train_single_fold(
            data_dir, output_dir, log_dir, fold, args.total_folds, device,
            base_ch=args.base_ch, attention=args.attention,
            batch_size=args.batch_size, target_size=args.target_size,
            num_epochs=args.epochs, lr=args.lr,
            use_focal=args.use_focal, use_class_weights=args.use_class_weights,
            dice_weight=args.dice_weight, ce_weight=args.ce_weight,
            early_stop_patience=args.patience,
            use_copy_paste=args.copy_paste,
            model_type=args.model,
            lovasz_weight=args.lovasz_weight,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            use_amp=not args.no_amp,
            grad_accum_steps=args.grad_accum,
            strong_augment=not args.no_strong_aug,
            use_weighted_sampling=not args.no_weighted_sampling,
        )
        fold_results.append(result)

    # Summary
    print("\n" + "=" * 80, flush=True)
    print("PUMA 5-Fold Cross Validation Summary", flush=True)
    print("=" * 80, flush=True)

    for r in fold_results:
        print("Fold " + str(r['fold']+1) + ": macroD=" + "%.4f" % r['best_dice'] +
              " microD=" + "%.4f" % r['best_micro_dice'] +
              " (TTA: " + "%.4f" % r['tta_dice'] + "/" + "%.4f" % r['tta_micro_dice'] + ")", flush=True)

    if len(fold_results) > 1:
        mean_dice = np.mean([r['best_dice'] for r in fold_results])
        mean_micro = np.mean([r['best_micro_dice'] for r in fold_results])
        tta_mean = np.mean([r['tta_dice'] for r in fold_results])
        tta_micro = np.mean([r['tta_micro_dice'] for r in fold_results])
        print("\nAverage:", flush=True)
        print("  macro Dice: " + "%.4f" % mean_dice + " +/- " +
              "%.4f" % np.std([r['best_dice'] for r in fold_results]), flush=True)
        print("  micro Dice: " + "%.4f" % mean_micro + " +/- " +
              "%.4f" % np.std([r['best_micro_dice'] for r in fold_results]), flush=True)
        print("  TTA macro:  " + "%.4f" % tta_mean, flush=True)
        print("  TTA micro:  " + "%.4f" % tta_micro, flush=True)

    # Save summary
    summary_file = os.path.join(
        log_dir, "puma_kfold_summary_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"
    )
    with open(summary_file, 'w') as f:
        f.write("PUMA 5-Fold Cross Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        for r in fold_results:
            f.write("Fold " + str(r['fold']+1) + ":\n")
            f.write("  Best macro Dice: " + "%.4f" % r['best_dice'] + " @ epoch " + str(r['best_epoch']) + "\n")
            f.write("  Best micro Dice: " + "%.4f" % r['best_micro_dice'] + "\n")
            f.write("  TTA macro Dice:  " + "%.4f" % r['tta_dice'] + "\n")
            f.write("  TTA micro Dice:  " + "%.4f" % r['tta_micro_dice'] + "\n\n")

    print("\nSummary saved to: " + summary_file, flush=True)
    print("\nPUMA 5-Fold CV complete!", flush=True)


if __name__ == '__main__':
    main()
