import os
import sys
import torch
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataloader
from models import MultiResUNet
from engines import DiceLoss, TverskyLoss, FocalLoss, train_one_epoch, evaluate

scaler = torch.cuda.amp.GradScaler(enabled=True)

def plot_training_curves(history, output_dir, start_epoch=1):
    """绘制训练曲线并保存，支持增量更新，标注最佳值"""
    epochs = list(range(start_epoch, start_epoch + len(history['train_loss'])))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ('train_loss', 'Training Loss', 'Loss', 'min'),
        ('train_acc', 'Accuracy', 'Accuracy', 'max'),
        ('train_sens', 'Sensitivity', 'Sensitivity', 'max'),
        ('train_spec', 'Specificity', 'Specificity', 'max'),
        ('train_auc', 'AUC-ROC', 'AUC', 'max'),
        ('train_dice', 'Dice Coefficient', 'Dice', 'max'),
    ]

    for idx, (train_key, title, ylabel, mode) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        val_key = train_key.replace('train_', 'val_')
        ax.plot(epochs, history[train_key], 'b-', label='Train', alpha=0.8)
        if val_key in history and len(history[val_key]) > 0:
            ax.plot(epochs, history[val_key], 'orange', label='Val', alpha=0.8)
            # 标注 Val 最佳值
            val_data = history[val_key]
            if mode == 'min':
                best_idx = val_data.index(min(val_data))
                best_val = min(val_data)
            else:
                best_idx = val_data.index(max(val_data))
                best_val = max(val_data)
            best_epoch = epochs[best_idx]
            ax.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.scatter([best_epoch], [best_val], color='r', s=50, zorder=5)
            ax.text(best_epoch, best_val, f'  Best: {best_val:.4f}\n  @E{best_epoch}',
                    fontsize=8, color='red', verticalalignment='bottom' if mode == 'max' else 'top')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Training curve updated to epoch {epochs[-1]}", flush=True)


def save_checkpoint(state, output_dir):
    """保存训练检查点"""
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
    torch.save(state, checkpoint_path)


def load_checkpoint(output_dir, model, optimizer, scheduler):
    """加载训练检查点，返回恢复后的状态"""
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        return None

    print(f"Found checkpoint at {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Resumed from epoch {checkpoint['epoch']}, best AUC: {checkpoint['best_auc']:.4f}", flush=True)
    return checkpoint


def get_autodl_paths():
    """AutoDL 环境路径自动检测"""
    autodl_root = '/root/autodl-tmp'
    local_root = os.path.dirname(os.path.abspath(__file__))
    
    if os.path.exists(autodl_root):
        print(f"[AutoDL] Using path: {autodl_root}", flush=True)
        project_dir = os.path.join(autodl_root, 'medseg_project')
        data_dir = os.path.join(project_dir, 'data', 'DRIVE')
        output_dir = os.path.join(project_dir, 'output')
        log_dir = os.path.join(project_dir, 'logs')
    else:
        print(f"[Local] Using path: {local_root}", flush=True)
        project_dir = os.path.join(os.path.dirname(local_root), 'medseg_project')
        data_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\data\DRIVE"
        output_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\output"
        log_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\logs"
    
    return data_dir, output_dir, log_dir


def main():
    data_dir, output_dir, log_dir = get_autodl_paths()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    train_loader = get_dataloader(data_dir, split='train', batch_size=2, num_workers=0, augment=True)
    val_loader = get_dataloader(data_dir, split='val', batch_size=2, num_workers=0, augment=False)

    # 使用 MultiResUNet（比基础 UNet 更强的多尺度架构）
    # attention: 'se' (default), 'cbam', or None
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
    start_epoch = 1
    early_stop_patience = 30
    early_stop_counter = 0

    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc': [], 'train_sens': [],
        'train_spec': [], 'train_auc': [], 'train_dice': [],
        'val_acc': [], 'val_sens': [], 'val_spec': [],
        'val_auc': [], 'val_dice': []
    }

    # 尝试恢复检查点
    checkpoint = load_checkpoint(output_dir, model, optimizer, scheduler)
    if checkpoint is not None:
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        best_auc = checkpoint.get('best_auc', 0.0)
        best_epoch = checkpoint['best_epoch']
        history = checkpoint.get('history', history)
        print(f"Resuming training from epoch {start_epoch}...", flush=True)

    log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # 如果从头开始，创建新的日志文件
    if start_epoch == 1:
        with open(log_file, 'w') as f:
            f.write("Epoch\tTrain Loss\tTrain Acc\tTrain Sens\tTrain Spec\tTrain AUC\tTrain Dice\t"
                    "Val Acc\tVal Sens\tVal Spec\tVal AUC\tVal Dice\n")

    print("=" * 100, flush=True)
    print(f"DRIVE Retinal Vessel Segmentation Training", flush=True)
    print(f"Model: MultiResUNet(base=64) + CLAHE + Elastic Aug + Dropout", flush=True)
    print(f"Loss: Dice×2 + Tversky(α=0.4,β=0.6) + Focal", flush=True)
    print(f"Train samples: {len(train_loader.dataset)}", flush=True)
    print(f"Val samples: {len(val_loader.dataset)}", flush=True)
    print(f"Epochs: {start_epoch}-{num_epochs}", flush=True)
    print("=" * 100, flush=True)

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        def combined_loss(outputs, labels):
            return 2 * dice_loss(outputs, labels) + tversky_loss(outputs, labels) + focal_loss(outputs, labels)

        train_loss, train_metrics = train_one_epoch(model, train_loader, combined_loss, optimizer, device, scaler)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step()

        elapsed = time.time() - epoch_start

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_sens'].append(train_metrics['sensitivity'])
        history['train_spec'].append(train_metrics['specificity'])
        history['train_auc'].append(train_metrics['auc'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_sens'].append(val_metrics['sensitivity'])
        history['val_spec'].append(val_metrics['specificity'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_dice'].append(val_metrics['dice'])

        tta_dice = val_metrics['dice']
        if epoch % 5 == 0 or epoch == num_epochs:
            tta_metrics = evaluate(model, val_loader, device, use_tta=True)
            tta_dice = tta_metrics['dice']

        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Sens: {train_metrics['sensitivity']:.4f} | "
              f"Spec: {train_metrics['specificity']:.4f} | "
              f"AUC: {train_metrics['auc']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Sens: {val_metrics['sensitivity']:.4f} | "
              f"Spec: {val_metrics['specificity']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f} | "
              f"Dice: {val_metrics['dice']:.4f} | "
              f"TTA Dice: {tta_dice:.4f} | "
              f"Time: {elapsed:.1f}s", flush=True)

        with open(log_file, 'a') as f:
            f.write(f"{epoch}\t{train_loss:.4f}\t{train_metrics['accuracy']:.4f}\t"
                    f"{train_metrics['sensitivity']:.4f}\t{train_metrics['specificity']:.4f}\t"
                    f"{train_metrics['auc']:.4f}\t{train_metrics['dice']:.4f}\t"
                    f"{val_metrics['accuracy']:.4f}\t{val_metrics['sensitivity']:.4f}\t"
                    f"{val_metrics['specificity']:.4f}\t{val_metrics['auc']:.4f}\t"
                    f"{val_metrics['dice']:.4f}\t{tta_dice:.4f}\n")

        # 保存检查点（每个 epoch 都保存，支持断点续训）
        # 用 TTA Dice 保存最佳模型（每5个epoch评估一次），普通 Dice 做早停
        is_best_tta = tta_dice > best_dice
        if is_best_tta:
            best_dice = tta_dice
            best_auc = val_metrics['auc']
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"  -> Best model saved (TTA Dice: {best_dice:.4f})", flush=True)
        else:
            early_stop_counter += 1

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': best_dice,
            'best_auc': best_auc,
            'best_epoch': best_epoch,
            'history': history,
        }, output_dir)

        # 每 5 个 epoch 更新一次训练曲线
        if epoch % 5 == 0 or epoch == num_epochs:
            plot_training_curves(history, output_dir)

        # Early Stopping
        if early_stop_counter >= early_stop_patience:
            print(f"  -> Early stopping triggered at epoch {epoch} (no improvement for {early_stop_patience} epochs)", flush=True)
            break

    # 加载最佳模型，用 TTA 重新评估
    print("=" * 100, flush=True)
    print(f"Training complete! Best Dice: {best_dice:.4f} at epoch {best_epoch}", flush=True)
    print(f"Best AUC: {best_auc:.4f}", flush=True)

    # TTA 评估
    if os.path.exists(os.path.join(output_dir, 'best_model.pth')):
        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
        tta_metrics = evaluate(model, val_loader, device, use_tta=True)
        print(f"TTA      Dice: {tta_metrics['dice']:.4f} | AUC: {tta_metrics['auc']:.4f} | "
              f"Sens: {tta_metrics['sensitivity']:.4f} | Spec: {tta_metrics['specificity']:.4f}", flush=True)
        
        tta_pp_metrics = evaluate(model, val_loader, device, use_tta=True, use_postprocess=True)
        print(f"TTA+PP   Dice: {tta_pp_metrics['dice']:.4f} | AUC: {tta_pp_metrics['auc']:.4f} | "
              f"Sens: {tta_pp_metrics['sensitivity']:.4f} | Spec: {tta_pp_metrics['specificity']:.4f}", flush=True)

    print(f"Log saved to: {log_file}", flush=True)
    print("=" * 100, flush=True)


if __name__ == '__main__':
    main()
