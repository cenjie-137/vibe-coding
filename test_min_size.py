import os
import sys
import torch
import numpy as np
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


def evaluate_with_min_size(model, val_loader, device, min_size):
    all_dice = []
    all_tp, all_fp, all_fn = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            masks = batch['mask'].numpy()

            outputs = model(images)
            images_h = torch.flip(images, dims=[3])
            outputs_h = model(images_h)
            outputs_h = torch.flip(outputs_h, dims=[3])
            images_v = torch.flip(images, dims=[2])
            outputs_v = model(images_v)
            outputs_v = torch.flip(outputs_v, dims=[2])
            outputs = (outputs + outputs_h + outputs_v) / 3.0

            preds_prob = torch.sigmoid(outputs).cpu().numpy()
            for i in range(images.size(0)):
                pred_bin = (preds_prob[i, 0] > 0.5).astype(np.float32)
                if min_size > 0:
                    pred_bin = post_process(pred_bin, min_size=min_size)

                label = labels[i, 0]
                mask = masks[i, 0]
                pred_f = pred_bin.flatten()
                label_f = label.flatten()
                mask_f = mask.flatten()
                pred_f = pred_f[mask_f > 0]
                label_f = label_f[mask_f > 0]

                tp = (pred_f * label_f).sum()
                fp = (pred_f * (1 - label_f)).sum()
                fn = ((1 - pred_f) * label_f).sum()
                union = pred_f.sum() + label_f.sum()
                dice = (2.0 * tp) / (union + 1e-6) if union > 0 else 0.0

                all_dice.append(dice)
                all_tp += tp
                all_fp += fp
                all_fn += fn

    mean_dice = np.mean(all_dice)
    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall = all_tp / (all_tp + all_fn + 1e-6)
    return mean_dice, precision, recall


def main():
    data_dir, output_dir = get_autodl_paths()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiResUNet(n_channels=3, n_classes=1, base_ch=64).to(device)
    model_path = os.path.join(output_dir, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    val_loader = get_dataloader(data_dir, split='val', batch_size=1, num_workers=0, augment=False)

    print(f"{'min_size':<12} {'Dice':<10} {'Precision':<12} {'Recall':<10}")
    print("-" * 50)

    # 测试不同的 min_size 阈值
    for min_size in [0, 5, 10, 15, 20, 25, 30, 50, 80, 100, 150]:
        dice, prec, rec = evaluate_with_min_size(model, val_loader, device, min_size)
        marker = " <- Best" if min_size == 0 else ""
        print(f"{min_size:<12} {dice:<10.4f} {prec:<12.4f} {rec:<10.4f}{marker}")


if __name__ == '__main__':
    main()
