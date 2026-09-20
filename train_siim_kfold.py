import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets_siim import get_siim_dataloaders
from models import MultiResUNet, MultiResUNetDS
from engines_siim import (
    DiceLoss, TverskyLoss, FocalLoss, BCEDiceLoss,
    train_one_epoch_siim, evaluate_siim
)

scaler = torch.cuda.amp.GradScaler(enabled=True)


def get_siim_paths():
    autodl_root = '/root/autodl-tmp'
    if os.path.exists(autodl_root):
        project_dir = os.path.join(autodl_root, 'medseg_project')
        data_dir = os.path.join(project_dir, 'data', 'SIIM', 'siim-acr-pneumothorax')
        output_dir = os.path.join(project_dir, 'output_siim')
        log_dir = os.path.join(project_dir, 'logs')
        train_csv = os.path.join(project_dir, 'data', 'SIIM', 'stage_1_train_images.csv')
        test_csv = os.path.join(project_dir, 'data', 'SIIM', 'stage_1_test_images.csv')
    else:
        local_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(local_root), 'medseg_project', 'data', 'SIIM', 'siim-acr-pneumothorax')
        output_dir = os.path.join(os.path.dirname(local_root), 'medseg_project', 'output_siim')
        log_dir = os.path.join(os.path.dirname(local_root), 'medseg_project', 'logs')
        train_csv = r"C:\Users\32201\Downloads\stage_1_train_images.csv"
        test_csv = r"C:\Users\32201\Downloads\stage_1_test_images.csv"
    return data_dir, output_dir, log_dir, train_csv, test_csv


def train_single_fold(data_dir, output_dir, log_dir, train_csv, fold, total_folds, device,
                      base_ch=32, attention='se', batch_size=8, target_size=256,
                      num_epochs=30, lr=3e-4, use_ds=False, preprocessed=True):
    """训练单个 fold"""
    fold_output_dir = os.path.join(output_dir, f'fold_{fold}')
    old_model = os.path.join(fold_output_dir, 'best_model.pth')
    if os.path.exists(old_model):
        os.remove(old_model)
        print(f"[Fold {fold+1}] Removed old checkpoint: {old_model}", flush=True)
    os.makedirs(fold_output_dir, exist_ok=True)

    train_loader, val_loader, train_files, val_files = get_siim_dataloaders(
        data_dir, train_csv, fold=fold, total_folds=total_folds,
        batch_size=batch_size, target_size=target_size, num_workers=4,
        preprocessed=preprocessed
    )

    # 统计正负样本
    mask_dir = os.path.join(data_dir, f'png_masks_{target_size}' if preprocessed else 'png_masks')
    train_has_pneumo = []
    for f in train_files:
        mask_path = os.path.join(mask_dir, f)
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('L'))
            train_has_pneumo.append((mask > 0).any())
        else:
            train_has_pneumo.append(False)
    n_pos = sum(train_has_pneumo)
    n_neg = len(train_files) - n_pos

    val_has_pneumo = []
    for f in val_files:
        mask_path = os.path.join(mask_dir, f)
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('L'))
            val_has_pneumo.append((mask > 0).any())
        else:
            val_has_pneumo.append(False)
    n_val_pos = sum(val_has_pneumo)
    n_val_neg = len(val_files) - n_val_pos

    print(f"\n{'='*80}", flush=True)
    print(f"[Fold {fold+1}/{total_folds}] Starting training", flush=True)
    print(f"Train: {len(train_files)} total ({n_pos} pos + {n_neg} neg)", flush=True)
    print(f"Val:   {len(val_files)} total ({n_val_pos} pos + {n_val_neg} neg)", flush=True)
    print(f"Model: MultiResUNet{'DS' if use_ds else ''} base_ch={base_ch} attn={attention}", flush=True)
    print(f"Image size: {target_size}x{target_size}, Batch size: {batch_size}, LR: {lr}", flush=True)
    print(f"Preprocessed: {preprocessed}", flush=True)
    print(f"{'='*80}", flush=True)

    model_cls = MultiResUNetDS if use_ds else MultiResUNet
    model = model_cls(n_channels=3, n_classes=1, base_ch=base_ch, attention=attention).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_dice = 0.0
    best_auc = 0.0
    best_epoch = 0
    early_stop_patience = 8
    early_stop_counter = 0

    ds_weights = [0.2, 0.3, 0.4] if use_ds else None

    fold_log_file = os.path.join(log_dir, f"siim_log_fold{fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(fold_log_file, 'w') as f:
        f.write("Epoch\tLoss\tDice(pos)\tIoU(pos)\tSens(pos)\tSpec\tAUC\t"
                "Val Dice(pos)\tVal IoU(pos)\tVal Sens(pos)\tVal Spec\tVal AUC\n")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics = train_one_epoch_siim(
            model, train_loader, criterion, optimizer, device, scaler, ds_weights
        )
        val_metrics = evaluate_siim(model, val_loader, device)
        scheduler.step()

        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            best_auc = val_metrics['auc']
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(fold_output_dir, 'best_model.pth'))
        else:
            early_stop_counter += 1

        with open(fold_log_file, 'a') as f:
            f.write(f"{epoch}\t{train_loss:.4f}\t"
                    f"{train_metrics['dice']:.4f}\t{train_metrics['iou']:.4f}\t"
                    f"{train_metrics['sensitivity']:.4f}\t{train_metrics['specificity']:.4f}\t"
                    f"{train_metrics['auc']:.4f}\t"
                    f"{val_metrics['dice']:.4f}\t{val_metrics['iou']:.4f}\t"
                    f"{val_metrics['sensitivity']:.4f}\t{val_metrics['specificity']:.4f}\t"
                    f"{val_metrics['auc']:.4f}\n")

        if epoch % 2 == 0 or is_best:
            print(f"[Fold {fold+1}] Epoch {epoch:3d} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Dice(pos): {val_metrics['dice']:.4f} | "
                  f"IoU(pos): {val_metrics['iou']:.4f} | "
                  f"Sens(pos): {val_metrics['sensitivity']:.4f} | "
                  f"AUC: {val_metrics['auc']:.4f} | "
                  f"Best: {best_dice:.4f} @E{best_epoch}", flush=True)

        if early_stop_counter >= early_stop_patience:
            print(f"[Fold {fold+1}] Early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(torch.load(os.path.join(fold_output_dir, 'best_model.pth')))
    tta_metrics = evaluate_siim(model, val_loader, device, use_tta=True)

    print(f"\n[Fold {fold+1}] Final Results:", flush=True)
    print(f"  Best Dice(pos): {best_dice:.4f} @E{best_epoch}", flush=True)
    print(f"  Best AUC: {best_auc:.4f}", flush=True)
    print(f"  TTA Dice(pos): {tta_metrics['dice']:.4f}", flush=True)
    print(f"  TTA IoU(pos): {tta_metrics['iou']:.4f}", flush=True)
    print(f"  TTA AUC: {tta_metrics['auc']:.4f}", flush=True)

    return {
        'fold': fold,
        'best_dice': best_dice,
        'best_auc': best_auc,
        'best_epoch': best_epoch,
        'tta_dice': tta_metrics['dice'],
        'tta_auc': tta_metrics['auc'],
        'tta_iou': tta_metrics['iou'],
        'model_path': os.path.join(fold_output_dir, 'best_model.pth')
    }


def main():
    parser = argparse.ArgumentParser(description='SIIM-ACR Pneumothorax Segmentation')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--total_folds', type=int, default=5)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--attention', type=str, default='se', choices=['se', 'cbam', 'none'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--use_ds', action='store_true')
    parser.add_argument('--preprocessed', action='store_true', default=True,
                        help='使用预处理数据（需先运行 preprocess_siim.py）')
    parser.add_argument('--no_preprocessed', action='store_false', dest='preprocessed')
    args = parser.parse_args()

    data_dir, output_dir, log_dir, train_csv, test_csv = get_siim_paths()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 检查预处理数据是否存在
    if args.preprocessed:
        img_dir = os.path.join(data_dir, f'png_images_{args.target_size}')
        if not os.path.exists(img_dir):
            print(f"Error: Preprocessed data not found at {img_dir}", flush=True)
            print("Run: python preprocess_siim.py", flush=True)
            return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    print(f"Data dir: {data_dir}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Train CSV: {train_csv}", flush=True)
    print(f"Preprocessed: {args.preprocessed}", flush=True)

    fold_results = []
    for fold in args.folds:
        result = train_single_fold(
            data_dir, output_dir, log_dir, train_csv,
            fold, args.total_folds, device,
            base_ch=args.base_ch, attention=args.attention,
            batch_size=args.batch_size, target_size=args.target_size,
            num_epochs=args.epochs, lr=args.lr, use_ds=args.use_ds,
            preprocessed=args.preprocessed
        )
        fold_results.append(result)

    print(f"\n{'='*80}", flush=True)
    print("5-Fold Cross Validation Summary", flush=True)
    print(f"{'='*80}", flush=True)

    for r in fold_results:
        print(f"Fold {r['fold']+1}: Dice(pos)={r['best_dice']:.4f} (TTA={r['tta_dice']:.4f}), "
              f"AUC={r['best_auc']:.4f} (TTA={r['tta_auc']:.4f})", flush=True)

    if len(fold_results) > 1:
        mean_dice = np.mean([r['best_dice'] for r in fold_results])
        mean_auc = np.mean([r['best_auc'] for r in fold_results])
        tta_mean_dice = np.mean([r['tta_dice'] for r in fold_results])
        tta_mean_auc = np.mean([r['tta_auc'] for r in fold_results])

        print(f"\nAverage:", flush=True)
        print(f"  Best Dice(pos): {mean_dice:.4f} ± {np.std([r['best_dice'] for r in fold_results]):.4f}", flush=True)
        print(f"  Best AUC: {mean_auc:.4f} ± {np.std([r['best_auc'] for r in fold_results]):.4f}", flush=True)
        print(f"  TTA Dice(pos): {tta_mean_dice:.4f}", flush=True)
        print(f"  TTA AUC: {tta_mean_auc:.4f}", flush=True)

    summary_file = os.path.join(log_dir, f"siim_kfold_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, 'w') as f:
        f.write("SIIM-ACR 5-Fold Cross Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        for r in fold_results:
            f.write(f"Fold {r['fold']+1}:\n")
            f.write(f"  Best Dice(pos): {r['best_dice']:.4f} @ epoch {r['best_epoch']}\n")
            f.write(f"  Best AUC: {r['best_auc']:.4f}\n")
            f.write(f"  TTA Dice(pos): {r['tta_dice']:.4f}\n")
            f.write(f"  TTA IoU(pos): {r['tta_iou']:.4f}\n")
            f.write(f"  TTA AUC: {r['tta_auc']:.4f}\n\n")

    print(f"\nSummary saved to: {summary_file}", flush=True)
    print("\n5-Fold CV complete!", flush=True)


if __name__ == '__main__':
    main()
