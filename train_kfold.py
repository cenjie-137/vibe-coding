import os
import sys
import torch
import numpy as np
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataloader
from models import MultiResUNet
from engines import DiceLoss, TverskyLoss, FocalLoss, train_one_epoch, evaluate

scaler = torch.cuda.amp.GradScaler(enabled=True)

def get_autodl_paths():
    autodl_root = '/root/autodl-tmp'
    local_root = os.path.dirname(os.path.abspath(__file__))
    
    if os.path.exists(autodl_root):
        project_dir = os.path.join(autodl_root, 'medseg_project')
        data_dir = os.path.join(project_dir, 'data', 'DRIVE')
        output_dir = os.path.join(project_dir, 'output')
        log_dir = os.path.join(project_dir, 'logs')
    else:
        project_dir = os.path.join(os.path.dirname(local_root), 'medseg_project')
        data_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\data\DRIVE"
        output_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\output"
        log_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\logs"
    
    return data_dir, output_dir, log_dir


def train_single_fold(data_dir, output_dir, log_dir, fold, total_folds, device):
    fold_output_dir = os.path.join(output_dir, f'fold_{fold}')
    os.makedirs(fold_output_dir, exist_ok=True)
    
    train_loader = get_dataloader(data_dir, split='train', batch_size=2, num_workers=0, augment=True, fold=fold, total_folds=total_folds)
    val_loader = get_dataloader(data_dir, split='val', batch_size=2, num_workers=0, augment=False, fold=fold, total_folds=total_folds)
    
    model = MultiResUNet(n_channels=3, n_classes=1, base_ch=64, attention='se').to(device)
    
    dice_loss = DiceLoss()
    tversky_loss = TverskyLoss(alpha=0.4, beta=0.6)
    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
    
    num_epochs = 150
    best_dice = 0.0
    best_auc = 0.0
    best_epoch = 0
    early_stop_patience = 30
    early_stop_counter = 0
    
    def combined_loss(outputs, labels):
        return 2 * dice_loss(outputs, labels) + tversky_loss(outputs, labels) + focal_loss(outputs, labels)
    
    fold_log_file = os.path.join(log_dir, f"training_log_fold{fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(fold_log_file, 'w') as f:
        f.write("Epoch\tTrain Loss\tTrain Acc\tTrain Sens\tTrain Spec\tTrain AUC\tTrain Dice\t"
                "Val Acc\tVal Sens\tVal Spec\tVal AUC\tVal Dice\n")
    
    print(f"\n{'='*80}", flush=True)
    print(f"[Fold {fold+1}/{total_folds}] Starting training", flush=True)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}", flush=True)
    print(f"{'='*80}", flush=True)
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, combined_loss, optimizer, device, scaler)
        val_metrics = evaluate(model, val_loader, device)
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
            f.write(f"{epoch}\t{train_loss:.4f}\t{train_metrics['accuracy']:.4f}\t"
                    f"{train_metrics['sensitivity']:.4f}\t{train_metrics['specificity']:.4f}\t"
                    f"{train_metrics['auc']:.4f}\t{train_metrics['dice']:.4f}\t"
                    f"{val_metrics['accuracy']:.4f}\t{val_metrics['sensitivity']:.4f}\t"
                    f"{val_metrics['specificity']:.4f}\t{val_metrics['auc']:.4f}\t"
                    f"{val_metrics['dice']:.4f}\n")
        
        if epoch % 10 == 0 or is_best:
            print(f"[Fold {fold+1}] Epoch {epoch:3d} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val Dice: {val_metrics['dice']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Best: {best_dice:.4f} @E{best_epoch}", flush=True)
        
        if early_stop_counter >= early_stop_patience:
            print(f"[Fold {fold+1}] Early stopping at epoch {epoch}", flush=True)
            break
    
    model.load_state_dict(torch.load(os.path.join(fold_output_dir, 'best_model.pth')))
    tta_metrics = evaluate(model, val_loader, device, use_tta=True)
    tta_pp_metrics = evaluate(model, val_loader, device, use_tta=True, use_postprocess=True)
    
    print(f"\n[Fold {fold+1}] Final Results:", flush=True)
    print(f"  Best Dice: {best_dice:.4f} @E{best_epoch}", flush=True)
    print(f"  Best AUC: {best_auc:.4f}", flush=True)
    print(f"  TTA Dice: {tta_metrics['dice']:.4f}", flush=True)
    print(f"  TTA AUC: {tta_metrics['auc']:.4f}", flush=True)
    print(f"  TTA+PP Dice: {tta_pp_metrics['dice']:.4f}", flush=True)
    print(f"  TTA+PP AUC: {tta_pp_metrics['auc']:.4f}", flush=True)
    
    return {
        'fold': fold,
        'best_dice': best_dice,
        'best_auc': best_auc,
        'best_epoch': best_epoch,
        'tta_dice': tta_metrics['dice'],
        'tta_auc': tta_metrics['auc'],
        'tta_pp_dice': tta_pp_metrics['dice'],
        'tta_pp_auc': tta_pp_metrics['auc'],
        'model_path': os.path.join(fold_output_dir, 'best_model.pth')
    }


def kfold_inference(data_dir, output_dir, device, total_folds=5):
    test_loader = get_dataloader(data_dir, split='test', batch_size=1, num_workers=0, augment=False)
    
    models = []
    for fold in range(total_folds):
        model = MultiResUNet(n_channels=3, n_classes=1, base_ch=64, attention='se').to(device)
        model_path = os.path.join(output_dir, f'fold_{fold}', 'best_model.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models.append(model)
    
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            ensemble_output = 0.0
            for model in models:
                outputs = model(images)
                outputs_h = model(torch.flip(images, dims=[3]))
                outputs_h = torch.flip(outputs_h, dims=[3])
                outputs_v = model(torch.flip(images, dims=[2]))
                outputs_v = torch.flip(outputs_v, dims=[2])
                ensemble_output += (outputs + outputs_h + outputs_v) / 3.0
            
            ensemble_output /= len(models)
            preds = torch.sigmoid(ensemble_output).detach().cpu().numpy()
            all_preds.append((batch['file_id'][0], preds[0, 0], masks[0, 0].detach().cpu().numpy()))
    
    return all_preds


def main():
    data_dir, output_dir, log_dir = get_autodl_paths()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    total_folds = 5
    fold_results = []
    
    for fold in range(total_folds):
        result = train_single_fold(data_dir, output_dir, log_dir, fold, total_folds, device)
        fold_results.append(result)
    
    print(f"\n{'='*80}", flush=True)
    print("5-Fold Cross Validation Summary", flush=True)
    print(f"{'='*80}", flush=True)
    
    for r in fold_results:
        print(f"Fold {r['fold']+1}: Dice={r['best_dice']:.4f} (TTA={r['tta_dice']:.4f}, TTA+PP={r['tta_pp_dice']:.4f}), "
              f"AUC={r['best_auc']:.4f} (TTA={r['tta_auc']:.4f})", flush=True)
    
    mean_dice = np.mean([r['best_dice'] for r in fold_results])
    mean_auc = np.mean([r['best_auc'] for r in fold_results])
    tta_mean_dice = np.mean([r['tta_dice'] for r in fold_results])
    tta_mean_auc = np.mean([r['tta_auc'] for r in fold_results])
    tta_pp_mean_dice = np.mean([r['tta_pp_dice'] for r in fold_results])
    tta_pp_mean_auc = np.mean([r['tta_pp_auc'] for r in fold_results])
    
    print(f"\nAverage:", flush=True)
    print(f"  Best Dice: {mean_dice:.4f} ± {np.std([r['best_dice'] for r in fold_results]):.4f}", flush=True)
    print(f"  Best AUC: {mean_auc:.4f} ± {np.std([r['best_auc'] for r in fold_results]):.4f}", flush=True)
    print(f"  TTA Dice: {tta_mean_dice:.4f}", flush=True)
    print(f"  TTA AUC: {tta_mean_auc:.4f}", flush=True)
    print(f"  TTA+PP Dice: {tta_pp_mean_dice:.4f}", flush=True)
    print(f"  TTA+PP AUC: {tta_pp_mean_auc:.4f}", flush=True)
    
    summary_file = os.path.join(log_dir, f"kfold_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, 'w') as f:
        f.write("5-Fold Cross Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        for r in fold_results:
            f.write(f"Fold {r['fold']+1}:\n")
            f.write(f"  Best Dice: {r['best_dice']:.4f} @ epoch {r['best_epoch']}\n")
            f.write(f"  Best AUC: {r['best_auc']:.4f}\n")
            f.write(f"  TTA Dice: {r['tta_dice']:.4f}\n")
            f.write(f"  TTA AUC: {r['tta_auc']:.4f}\n")
            f.write(f"  TTA+PP Dice: {r['tta_pp_dice']:.4f}\n")
            f.write(f"  TTA+PP AUC: {r['tta_pp_auc']:.4f}\n\n")
        f.write("=" * 50 + "\n")
        f.write(f"Average Best Dice: {mean_dice:.4f} ± {np.std([r['best_dice'] for r in fold_results]):.4f}\n")
        f.write(f"Average Best AUC: {mean_auc:.4f} ± {np.std([r['best_auc'] for r in fold_results]):.4f}\n")
        f.write(f"Average TTA Dice: {tta_mean_dice:.4f}\n")
        f.write(f"Average TTA AUC: {tta_mean_auc:.4f}\n")
        f.write(f"Average TTA+PP Dice: {tta_pp_mean_dice:.4f}\n")
        f.write(f"Average TTA+PP AUC: {tta_pp_mean_auc:.4f}\n")
    
    print(f"\nSummary saved to: {summary_file}", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print("Running ensemble inference on test set...", flush=True)
    print(f"{'='*80}", flush=True)
    
    kfold_inference(data_dir, output_dir, device, total_folds)
    
    print("\n5-Fold CV complete!", flush=True)


if __name__ == '__main__':
    main()