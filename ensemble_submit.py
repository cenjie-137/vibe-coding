import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import zipfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataloader
from models import MultiResUNet
from engines import compute_metrics

def get_autodl_paths():
    autodl_root = '/root/autodl-tmp'
    local_root = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(autodl_root):
        project_dir = os.path.join(autodl_root, 'medseg_project')
        data_dir = os.path.join(project_dir, 'data', 'DRIVE')
        output_dir = os.path.join(project_dir, 'output')
    else:
        project_dir = os.path.join(os.path.dirname(local_root), 'medseg_project')
        data_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\data\DRIVE"
        output_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\output"
    return data_dir, output_dir

def tta_predict(model, images):
    """单模型 TTA：原图 + 水平翻转 + 垂直翻转"""
    model.eval()
    with torch.no_grad():
        outputs = [torch.sigmoid(model(images))]
        outputs.append(torch.sigmoid(model(torch.flip(images, dims=[3]))))
        outputs.append(torch.sigmoid(model(torch.flip(images, dims=[2]))))
        outputs[1] = torch.flip(outputs[1], dims=[3])
        outputs[2] = torch.flip(outputs[2], dims=[2])
        return torch.stack(outputs).mean(dim=0)

def ensemble_predict(models, images):
    """多模型 TTA 概率平均"""
    all_probs = []
    for model in models:
        probs = tta_predict(model, images)
        all_probs.append(probs)
    return torch.stack(all_probs).mean(dim=0)

def search_threshold(models, data_dir, device):
    """在验证集上搜索最优阈值"""
    print("Searching optimal threshold on validation set...")
    val_loader = get_dataloader(data_dir, split='val', batch_size=2, num_workers=0, augment=False)

    all_preds = []
    all_labels = []
    all_masks = []

    for batch in val_loader:
        images = batch['image'].to(device)
        labels = batch['label']
        masks = batch['mask']
        probs = ensemble_predict(models, images).cpu()
        all_preds.append(probs)
        all_labels.append(labels)
        all_masks.append(masks)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_masks = torch.cat(all_masks, dim=0).numpy()

    best_dice = 0.0
    best_thresh = 0.5
    print("\nThreshold search results:")
    for thresh in np.arange(0.30, 0.71, 0.01):
        dices = []
        for i in range(all_preds.shape[0]):
            pred = (all_preds[i, 0] > thresh).astype(np.float32)
            label = all_labels[i, 0]
            mask = all_masks[i, 0]
            metrics = compute_metrics(pred, label, mask)
            dices.append(metrics['dice'])
        avg_dice = np.mean(dices)
        marker = " <-- BEST" if avg_dice > best_dice else ""
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_thresh = thresh
        print(f"  threshold={thresh:.2f}  Val Dice={avg_dice:.4f}{marker}")

    print(f"\nBest threshold: {best_thresh:.2f} with Val Dice={best_dice:.4f}")
    return best_thresh

def generate_submission(models, data_dir, output_dir, device, threshold=0.5):
    """生成竞赛提交文件"""
    print(f"\nGenerating submission with threshold={threshold}...")
    test_dir = os.path.join(data_dir, 'test', 'images')
    submit_dir = os.path.join(output_dir, f'submission_ensemble')
    os.makedirs(submit_dir, exist_ok=True)

    for i in tqdm(range(1, 21), desc="Processing"):
        img_path = os.path.join(test_dir, f'{i:02d}_test.tif')
        if not os.path.exists(img_path):
            img_path = os.path.join(test_dir, f'{i:02d}_test.png')

        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

        prob = ensemble_predict(models, img_tensor).cpu().numpy()[0, 0]
        pred = (prob > threshold).astype(np.uint8) * 255

        pred_img = Image.fromarray(pred)
        pred_img.save(os.path.join(submit_dir, f'{i}.png'))

    zip_path = os.path.join(output_dir, 'submission_ensemble.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(submit_dir):
            zf.write(os.path.join(submit_dir, f), f)

    print(f"Submission saved to: {zip_path}")
    return zip_path

def main():
    data_dir, output_dir = get_autodl_paths()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    # 加载 3 个 seed 的模型 + 当前 best_model.pth
    model_paths = [
        os.path.join(output_dir, 'best_model.pth'),
        os.path.join(output_dir, 'seed_42', 'best_model.pth'),
        os.path.join(output_dir, 'seed_123', 'best_model.pth'),
        os.path.join(output_dir, 'seed_456', 'best_model.pth'),
    ]

    models = []
    for p in model_paths:
        if not os.path.exists(p):
            print(f"SKIP: {p} not found")
            continue
        model = MultiResUNet(n_channels=3, n_classes=1, base_ch=64, attention='se').to(device)
        state = torch.load(p, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        print(f"Loaded: {p}")

    if len(models) == 0:
        print("ERROR: No models found!")
        return

    print(f"\nTotal models for ensemble: {len(models)}")

    # 阈值搜索
    best_thresh = search_threshold(models, data_dir, device)

    # 生成提交（带确认）
    user_input = input(f"\nGenerate submission with threshold={best_thresh}? [y/n]: ")
    if user_input.lower() == 'y':
        generate_submission(models, data_dir, output_dir, device, best_thresh)
    else:
        print("Submission cancelled.")

if __name__ == '__main__':
    main()
