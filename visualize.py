import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataloader
from models import MultiResUNet

from engines import compute_metrics

def create_overlay(img, mask, color=(1, 0, 0), alpha=0.8):
    """将二值 mask 以指定颜色半透明叠加到原图上"""
    overlay = img.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask > 0.5,
            img[:, :, c] * (1 - alpha) + color[c] * alpha,
            img[:, :, c]
        )
    return overlay

def visualize_results(model, dataloader, device, output_dir, num_samples=5):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_samples:
                break

            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            masks = batch['mask'].cpu().numpy()
            file_ids = batch['file_id']

            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()

            for i in range(images.size(0)):
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                label = labels[i, 0]
                pred = preds[i, 0]
                mask = masks[i, 0]

                pred_masked = pred * mask
                label_masked = label * mask
                pred_bin = (pred_masked > 0.5).astype(np.float32)

                metrics = compute_metrics(pred, label, mask)

                # 打印 GT 统计，证明不是全零
                nz = np.count_nonzero(label_masked)
                print(f"[{file_ids[i]}] GT stats: min={label_masked.min():.3f}, max={label_masked.max():.3f}, "
                      f"mean={label_masked.mean():.4f}, nonzero_pixels={nz}")

                gt_overlay = create_overlay(img, label_masked, color=(1, 0, 0), alpha=0.85)
                pred_overlay = create_overlay(img, pred_bin, color=(0, 1, 0), alpha=0.75)

                fig, axes = plt.subplots(2, 3, figsize=(18, 12))

                axes[0, 0].imshow(img)
                axes[0, 0].set_title('Input Image')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(label_masked, cmap='gray', vmin=0, vmax=1)
                axes[0, 1].set_title(f'GT (Gray)\nnonzero={nz}')
                axes[0, 1].axis('off')

                axes[0, 2].imshow(pred_masked, cmap='gray', vmin=0, vmax=1)
                axes[0, 2].set_title(f'Prediction (Gray)\nAUC={metrics["auc"]:.3f}')
                axes[0, 2].axis('off')

                axes[1, 0].imshow(gt_overlay)
                axes[1, 0].set_title('GT Overlay (Red=Vessel)')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(pred_overlay)
                axes[1, 1].set_title('Pred Overlay (Green=Vessel)')
                axes[1, 1].axis('off')

                diff = np.abs(pred_bin - label_masked)
                axes[1, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
                axes[1, 2].set_title(f'Difference\nAcc={metrics["accuracy"]:.3f}')
                axes[1, 2].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{file_ids[i]}_result.png"), dpi=150, bbox_inches='tight')
                plt.close()

                print(f"Saved visualization: {file_ids[i]}_result.png")

def plot_training_curve(log_file, output_dir):
    import pandas as pd
    
    df = pd.read_csv(log_file, sep='\t')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    axes[0, 0].plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title('Training Loss')
    
    axes[0, 1].plot(df['Epoch'], df['Train Acc'], label='Train Acc')
    axes[0, 1].plot(df['Epoch'], df['Val Acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_title('Accuracy')
    
    axes[1, 0].plot(df['Epoch'], df['Train Sens'], label='Train Sens')
    axes[1, 0].plot(df['Epoch'], df['Val Sens'], label='Val Sens')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Sensitivity')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title('Sensitivity')
    
    axes[1, 1].plot(df['Epoch'], df['Train AUC'], label='Train AUC')
    axes[1, 1].plot(df['Epoch'], df['Val AUC'], label='Val AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_title('AUC-ROC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curve: training_curve.png")

def get_autodl_paths():
    autodl_root = '/root/autodl-tmp'
    if os.path.exists(autodl_root):
        project_dir = os.path.join(autodl_root, 'medseg_project')
        data_dir = os.path.join(project_dir, 'data', 'DRIVE')
        output_dir = os.path.join(project_dir, 'output')
        log_dir = os.path.join(project_dir, 'logs')
    else:
        data_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\data\DRIVE"
        output_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\output"
        log_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\logs"
    return data_dir, output_dir, log_dir


def main():
    data_dir, output_dir, log_dir = get_autodl_paths()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MultiResUNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
    
    val_loader = get_dataloader(data_dir, split='val', batch_size=2, num_workers=0)
    
    print("Generating visualizations...")
    visualize_results(model, val_loader, device, output_dir)
    
    log_files = [f for f in os.listdir(log_dir) if f.startswith('training_log')]
    if log_files:
        latest_log = sorted(log_files)[-1]
        plot_training_curve(os.path.join(log_dir, latest_log), output_dir)
    
    print("Visualization complete!")

if __name__ == '__main__':
    main()