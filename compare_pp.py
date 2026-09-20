import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataloader
from models import MultiResUNet
from engines import post_process

def get_autodl_paths():
    autodl_root = '/root/autodl-tmp'
    if os.path.exists(autodl_root):
        data_dir = os.path.join(autodl_root, 'medseg_project', 'data', 'DRIVE')
        output_dir = os.path.join(autodl_root, 'medseg_project', 'output')
    else:
        data_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\data\DRIVE"
        output_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\output"
    return data_dir, output_dir


def main():
    data_dir, output_dir = get_autodl_paths()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = MultiResUNet(n_channels=3, n_classes=1, base_ch=64).to(device)
    model_path = os.path.join(output_dir, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    val_loader = get_dataloader(data_dir, split='val', batch_size=1, num_workers=0, augment=False)

    save_dir = os.path.join(output_dir, 'compare_pp')
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= 4:
                break

            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            masks = batch['mask'].numpy()

            # TTA 预测
            outputs = model(images)
            images_h = torch.flip(images, dims=[3])
            outputs_h = model(images_h)
            outputs_h = torch.flip(outputs_h, dims=[3])
            images_v = torch.flip(images, dims=[2])
            outputs_v = model(images_v)
            outputs_v = torch.flip(outputs_v, dims=[2])
            outputs = (outputs + outputs_h + outputs_v) / 3.0

            preds_prob = torch.sigmoid(outputs).cpu().numpy()[0, 0]
            label = labels[0, 0]
            mask = masks[0, 0]
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            # 反归一化
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            # 后处理前二值化
            pred_bin_before = (preds_prob > 0.5).astype(np.float32)
            # 后处理
            pred_bin_after = post_process(pred_bin_before, min_size=30)

            # 计算指标
            def calc_dice(pred, gt, msk):
                pp = pred[msk > 0]
                gg = gt[msk > 0]
                inter = (pp * gg).sum()
                union = pp.sum() + gg.sum()
                return (2.0 * inter) / (union + 1e-6) if union > 0 else 0.0

            dice_before = calc_dice(pred_bin_before, label, mask)
            dice_after = calc_dice(pred_bin_after, label, mask)

            # 计算连通域数量
            num_before = cv2.connectedComponentsWithStats(pred_bin_before.astype(np.uint8), connectivity=8)[0] - 1
            num_after = cv2.connectedComponentsWithStats(pred_bin_after.astype(np.uint8), connectivity=8)[0] - 1

            # 创建红绿 overlay
            def make_overlay(img, pred, gt, alpha=0.6):
                overlay = img.copy()
                # 假阳 (pred=1, gt=0) - 红色
                fp = (pred > 0.5) & (gt < 0.5)
                # 假阴 (pred=0, gt=1) - 绿色
                fn = (pred < 0.5) & (gt > 0.5)
                # 真阳 (pred=1, gt=1) - 蓝色
                tp = (pred > 0.5) & (gt > 0.5)
                for c in range(3):
                    overlay[:, :, c] = np.where(fp, overlay[:, :, c] * (1 - alpha) + np.array([1, 0, 0])[c] * alpha, overlay[:, :, c])
                    overlay[:, :, c] = np.where(fn, overlay[:, :, c] * (1 - alpha) + np.array([0, 1, 0])[c] * alpha, overlay[:, :, c])
                    overlay[:, :, c] = np.where(tp, overlay[:, :, c] * (1 - alpha) + np.array([0, 0, 1])[c] * alpha, overlay[:, :, c])
                return overlay

            overlay_before = make_overlay(img, pred_bin_before, label)
            overlay_after = make_overlay(img, pred_bin_after, label)

            # 第 1 列：原图
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title(f'Image {idx+1}', fontsize=12)
            axes[idx, 0].axis('off')

            # 第 2 列：GT
            axes[idx, 1].imshow(label, cmap='gray')
            axes[idx, 1].set_title('Ground Truth', fontsize=12)
            axes[idx, 1].axis('off')

            # 第 3 列：TTA 预测（后处理前）
            axes[idx, 2].imshow(pred_bin_before, cmap='gray')
            axes[idx, 2].set_title(f'Before PP\nDice={dice_before:.4f}\nCC={num_before}', fontsize=10)
            axes[idx, 2].axis('off')

            # 第 4 列：后处理前 overlay
            axes[idx, 3].imshow(overlay_before)
            axes[idx, 3].set_title(f'Before PP Overlay\nR=FP G=FN B=TP', fontsize=10)
            axes[idx, 3].axis('off')

            # 第 5 列：后处理后 overlay
            axes[idx, 4].imshow(overlay_after)
            axes[idx, 4].set_title(f'After PP Overlay\nDice={dice_after:.4f}\nCC={num_after}', fontsize=10)
            axes[idx, 4].axis('off')

    plt.suptitle('TTA Post-Processing Comparison (Red=FP, Green=FN, Blue=TP)', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'pp_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to: {save_path}")
    print(f"Download this file to view on local machine")


if __name__ == '__main__':
    main()
