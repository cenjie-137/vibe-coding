import os
import sys
import torch
import numpy as np

sys.path.append('.')
from datasets import get_dataloader
from models import UNet
from engines import compute_metrics

data_dir = r'C:\Users\32201\Desktop\T\MOME\medseg_project\data\DRIVE'
output_dir = r'C:\Users\32201\Desktop\T\MOME\medseg_project\output'

device = torch.device('cpu')
model = UNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
model.eval()

val_loader = get_dataloader(data_dir, split='val', batch_size=2, num_workers=0, augment=False)

all_dice = []
all_acc = []
all_sens = []
all_spec = []
all_auc = []

with torch.no_grad():
    for batch in val_loader:
        images = batch['image'].to(device)
        labels = batch['label'].cpu().numpy()
        masks = batch['mask'].cpu().numpy()
        file_ids = batch['file_id']
        
        outputs = model(images)
        preds = torch.sigmoid(outputs).cpu().numpy()
        
        for i in range(images.size(0)):
            pred = preds[i, 0]
            label = labels[i, 0]
            mask = masks[i, 0]
            
            metrics = compute_metrics(pred, label, mask)
            
            print(f'[{file_ids[i]}] Dice={metrics["dice"]:.4f}, Acc={metrics["accuracy"]:.4f}, Sens={metrics["sensitivity"]:.4f}, Spec={metrics["specificity"]:.4f}, AUC={metrics["auc"]:.4f}')
            
            all_dice.append(metrics['dice'])
            all_acc.append(metrics['accuracy'])
            all_sens.append(metrics['sensitivity'])
            all_spec.append(metrics['specificity'])
            all_auc.append(metrics['auc'])

print()
print('='*70)
print(f'Val Mean Dice: {np.mean(all_dice):.4f}')
print(f'Val Mean Acc: {np.mean(all_acc):.4f}')
print(f'Val Mean Sens: {np.mean(all_sens):.4f}')
print(f'Val Mean Spec: {np.mean(all_spec):.4f}')
print(f'Val Mean AUC: {np.mean(all_auc):.4f}')
print('='*70)
