import os
import sys
import time
import json
import argparse
import datetime

import torch
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets_beetle import seed_everything, get_dataloader, CLASSES
from models import MultiResUNet
from engines_beetle import (
    MultiClassCombinedLoss,
    train_one_epoch_beetle,
    validate_one_epoch_beetle,
)


def get_paths():
    autodl_root = '/root/autodl-tmp'
    if os.path.exists(autodl_root):
        data_dir = os.path.join(autodl_root, 'medseg_project', 'data', 'BEETLE')
        output_dir = os.path.join(autodl_root, 'medseg_project', 'output_beetle')
        log_dir = os.path.join(autodl_root, 'medseg_project', 'logs')
    else:
        base = r"C:\Users\32201\Desktop\T\vibe coding\BEETLE"
        data_dir = os.path.join(base, 'data', 'BEETLE')
        output_dir = os.path.join(base, 'output')
        log_dir = os.path.join(base, 'logs')
    return data_dir, output_dir, log_dir


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_overall_dice': best_metric,
    }, path)
    print(f"-> Checkpoint saved: {path}", flush=True)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: strict load failed, trying partial: {e}", flush=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}", flush=True)
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}", flush=True)
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception:
            pass
    if scheduler is not None and 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except Exception:
            pass
    print(f"-> Checkpoint loaded: {path} (epoch={ckpt.get('epoch', '?')})", flush=True)
    return ckpt


def plot_curves(history, save_path):
    n_plots = 2
    fig, axes = plt.subplots(1, n_plots, figsize=(16, 5))

    axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss')
    axes[0].plot(history['epoch'], history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['epoch'], history['train_overall_dice'], label='Train Overall Dice')
    axes[1].plot(history['epoch'], history['val_overall_dice'], label='Val Overall Dice')
    best_epoch = max(history['epoch'], key=lambda e: history['val_overall_dice'][history['epoch'].index(e)])
    best_val = history['val_overall_dice'][history['epoch'].index(best_epoch)]
    axes[1].scatter([best_epoch], [best_val], color='red', s=50, zorder=5, label=f'Best @E{best_epoch}: {best_val:.4f}')
    axes[1].set_title('Overall Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def print_metrics(prefix, metrics):
    pcd = metrics['per_class_dice']
    per_class_str = ' '.join([f"{CLASSES[i][:3]}={pcd[i]:.3f}" for i in range(len(CLASSES))])
    print(f"{prefix} Loss={metrics['loss']:.4f} | Acc={metrics['accuracy']:.4f} | "
          f"Overall Dice={metrics['overall_dice']:.4f} | IoU={metrics['overall_iou']:.4f}", flush=True)
    print(f"        Per-class Dice: {per_class_str}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--attention', type=str, default='se', choices=['se', 'cbam', 'none'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--patch_per_img', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--class_balance', action='store_true',
                        help='对稀有类别（non_invasive、necrosis）加权重')
    args = parser.parse_args()

    seed_everything(args.seed)
    data_dir, output_dir, log_dir = get_paths()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    print(f"Args: {vars(args)}", flush=True)

    # 数据加载
    augment = not args.no_augment
    train_loader = get_dataloader(data_dir, split='train', batch_size=args.batch_size,
                                  patch_size=args.patch_size, patch_per_img=args.patch_per_img,
                                  augment=augment, num_workers=4)
    val_loader = get_dataloader(data_dir, split='val', batch_size=args.batch_size,
                                patch_size=args.patch_size, patch_per_img=1,
                                augment=False, num_workers=4, shuffle=False)

    # 类别权重：non_invasive_epithelium(1) 和 necrosis(2) 稀有
    if args.class_balance:
        class_w = torch.tensor([1.0, 1.8, 2.5, 0.6], dtype=torch.float32).to(device)
    else:
        class_w = None
    focal_alpha = class_w

    # 模型
    attention = None if args.attention == 'none' else args.attention
    model = MultiResUNet(n_channels=3, n_classes=4, base_ch=args.base_ch,
                         attention=attention, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.2f}M", flush=True)

    # 损失
    loss_fn = MultiClassCombinedLoss(
        dice_weight=2.0, focal_weight=1.0, ce_weight=1.0,
        class_weights=class_w, focal_alpha=focal_alpha, focal_gamma=2.0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_model_path = os.path.join(output_dir, 'best_model.pth')
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')

    # 断点续训
    best_overall_dice = 0.0
    start_epoch = 1
    ckpt = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    if ckpt is not None:
        best_overall_dice = ckpt.get('best_overall_dice', 0.0)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"  Best Overall Dice so far: {best_overall_dice:.4f}", flush=True)

    # 日志
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'training_log_beetle_{timestamp}.txt')
    history = {
        'epoch': [], 'train_loss': [], 'train_overall_dice': [],
        'val_loss': [], 'val_overall_dice': [],
        'per_class_dice_train': [], 'per_class_dice_val': [],
    }

    best_epoch = 0
    patience_counter = 0

    with open(log_path, 'w', encoding='utf-8') as log_f:
        log_f.write(f"Args: {vars(args)}\n\n")

        for epoch in range(start_epoch, args.epochs + 1):
            t_start = time.time()
            print(f"\n{'=' * 80}\nEpoch {epoch}/{args.epochs}", flush=True)

            # Train
            train_metrics = train_one_epoch_beetle(model, train_loader, loss_fn,
                                                    optimizer, device, scaler)
            print_metrics("Train", train_metrics)

            # Val
            val_metrics = validate_one_epoch_beetle(model, val_loader, loss_fn, device)
            print_metrics("Valid", val_metrics)

            # 记录
            for k in history:
                if k.startswith('per_class_'):
                    continue
                history[k].append(train_metrics[k] if k.startswith('train_') else
                                  val_metrics[k.replace('val_', '')] if k.startswith('val_')
                                  else locals().get(k, [epoch]))
            history['epoch'][-1] = epoch
            history['per_class_dice_train'].append(train_metrics['per_class_dice'])
            history['per_class_dice_val'].append(val_metrics['per_class_dice'])

            scheduler.step()

            # 保存最佳模型
            if val_metrics['overall_dice'] > best_overall_dice:
                best_overall_dice = val_metrics['overall_dice']
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
                print(f"-> New best model saved @ Overall Dice={best_overall_dice:.4f}", flush=True)
            else:
                patience_counter += 1
                print(f"-> No improvement ({patience_counter}/{args.patience})", flush=True)

            # 检查点
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_overall_dice)

            # 日志
            log_f.write(f"Epoch {epoch}\n")
            log_f.write(f"  Train: Loss={train_metrics['loss']:.4f} Dice={train_metrics['overall_dice']:.4f} "
                        f"PerClassDice={train_metrics['per_class_dice']}\n")
            log_f.write(f"  Val:   Loss={val_metrics['loss']:.4f} Dice={val_metrics['overall_dice']:.4f} "
                        f"PerClassDice={val_metrics['per_class_dice']}\n")
            log_f.flush()

            # 画曲线
            plot_path = os.path.join(output_dir, 'training_curve.png')
            plot_curves(history, plot_path)

            # Early stopping
            if patience_counter >= args.patience:
                print(f"\n-> Early stopping at epoch {epoch}", flush=True)
                break

            t_end = time.time()
            print(f"Time: {t_end - t_start:.1f}s", flush=True)

    print(f"\n{'=' * 80}\nTraining complete! Best Overall Dice: {best_overall_dice:.4f} at epoch {best_epoch}",
          flush=True)
    print(f"Log saved to: {log_path}", flush=True)


if __name__ == '__main__':
    main()
