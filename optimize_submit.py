import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import zipfile
import traceback
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataloader
from models import MultiResUNet, MultiResUNetDS


def get_autodl_paths():
    local_root = os.path.dirname(os.path.abspath(__file__))

    windows_data_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\data\DRIVE"
    windows_output_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\output"

    if os.name == 'posix' and os.path.exists('/root/autodl-tmp'):
        project_dir = '/root/autodl-tmp/medseg_project'
        data_dir = os.path.join(project_dir, 'data', 'DRIVE')
        output_dir = os.path.join(project_dir, 'output')
    elif os.path.exists(windows_data_dir):
        data_dir = windows_data_dir
        output_dir = windows_output_dir
    else:
        project_dir = os.path.join(os.path.dirname(local_root), 'medseg_project')
        data_dir = os.path.join(project_dir, 'data', 'DRIVE')
        output_dir = os.path.join(project_dir, 'output')

    return data_dir, output_dir


def load_models(output_dir, device, base_ch=64, mode='full+fold'):
    """加载模型。
    mode: 'full'      -> 只加载 best_model_full.pth
          'fold'       -> 只加载 fold_0~fold_4
          'full+fold'  -> 加载 best_model_full.pth + fold_0~fold_4（默认）
    自动尝试 SE/CBAM 两种注意力类型。"""
    models = []
    model_paths = []

    candidate_paths = []
    if mode in ['full', 'full+fold']:
        full_path = os.path.join(output_dir, 'best_model_full.pth')
        if os.path.exists(full_path):
            candidate_paths.append(full_path)
        else:
            print(f"WARNING: {full_path} not found", flush=True)

    if mode in ['fold', 'full+fold']:
        for fold in range(5):
            fold_path = os.path.join(output_dir, f'fold_{fold}', 'best_model.pth')
            if os.path.exists(fold_path):
                candidate_paths.append(fold_path)

    if not candidate_paths:
        best_path = os.path.join(output_dir, 'best_model.pth')
        if os.path.exists(best_path):
            candidate_paths.append(best_path)

    for model_path in candidate_paths:
        loaded = False
        for ch in [base_ch, 32]:
            for attention in ['se', 'cbam']:
                for model_cls in [MultiResUNetDS, MultiResUNet]:
                    try:
                        model = model_cls(n_channels=3, n_classes=1, base_ch=ch, attention=attention).to(device)
                        state = torch.load(model_path, map_location=device)
                        model.load_state_dict(state)
                        model.eval()
                        models.append(model)
                        tag = "DS" if model_cls == MultiResUNetDS else "plain"
                        model_paths.append(f"{os.path.basename(model_path)} ({tag}/{attention})")
                        loaded = True
                        print(f"Loaded {model_path} with {tag}, base_ch={ch}, attention={attention}", flush=True)
                        break
                    except RuntimeError:
                        continue
                if loaded:
                    break
            if loaded:
                break
        if not loaded:
            print(f"WARNING: Failed to load {model_path}", flush=True)

    print(f"\nLoaded {len(models)} model(s):", flush=True)
    for p in model_paths:
        print(f"  - {p}", flush=True)

    return models


def _init_missing_to_identity(model, missing_keys):
    """对旧版 checkpoint 缺失的 SE / bn_shortcut 做近似 identity 初始化，
    使行为尽量接近无 SE、无 shortcut BN 的旧版 ResPath。"""
    for name in missing_keys:
        if name.endswith('.se.fc.2.bias'):
            module = model
            for part in name.split('.')[:-1]:
                module = getattr(module, part)
            with torch.no_grad():
                module.weight.zero_()
                module.bias.fill_(10.0)  # sigmoid(10) ≈ 1.0，SE 近似 identity
        elif name.endswith('.bn_shortcut.weight'):
            module = model
            for part in name.split('.')[:-1]:
                module = getattr(module, part)
            with torch.no_grad():
                module.weight.fill_(1.0)
                module.bias.zero_()
                module.running_mean.zero_()
                module.running_var.fill_(1.0)


@torch.no_grad()
def tta_predict(model, images):
    """TTA：原图 + 水平翻转 + 垂直翻转"""
    outputs = model(images)

    outputs_h = model(torch.flip(images, dims=[3]))
    outputs_h = torch.flip(outputs_h, dims=[3])

    outputs_v = model(torch.flip(images, dims=[2]))
    outputs_v = torch.flip(outputs_v, dims=[2])

    avg = (outputs + outputs_h + outputs_v) / 3.0
    return torch.sigmoid(avg)


@torch.no_grad()
def ensemble_predict(models, images):
    """多模型 TTA 集成预测"""
    probs = [tta_predict(m, images) for m in models]
    return torch.stack(probs, dim=0).mean(dim=0)


def search_threshold(models, data_dir, device, thresholds=None):
    """在验证集上搜索最优阈值"""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.01)

    val_loader = get_dataloader(data_dir, split='val', batch_size=1, num_workers=0, augment=False)
    all_probs = []
    all_labels = []
    all_masks = []

    print("\nRunning validation inference for threshold search...", flush=True)
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask'].to(device)

            probs = ensemble_predict(models, images)
            all_probs.append(probs.cpu().numpy()[0, 0])
            all_labels.append(labels.cpu().numpy()[0, 0])
            all_masks.append(masks.cpu().numpy()[0, 0])

    best_dice = -1.0
    best_thresh = 0.5
    results = []

    print("\nThreshold search results:", flush=True)
    for thresh in thresholds:
        dices = []
        for prob, label, mask in zip(all_probs, all_labels, all_masks):
            pred_bin = (prob > thresh).astype(np.float32) * mask
            target_bin = label * mask
            inter = np.sum(pred_bin * target_bin)
            union = np.sum(pred_bin) + np.sum(target_bin)
            dice = (2.0 * inter) / (union + 1e-6) if union > 0 else 0.0
            dices.append(dice)

        mean_dice = np.mean(dices)
        results.append((thresh, mean_dice))
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_thresh = thresh

    for thresh, dice in results:
        marker = " <-- BEST" if thresh == best_thresh else ""
        print(f"  threshold={thresh:.2f}  Val Dice={dice:.4f}{marker}", flush=True)

    print(f"\nBest threshold: {best_thresh:.2f} with Val Dice={best_dice:.4f}", flush=True)
    return best_thresh, best_dice


def generate_submission(models, data_dir, output_dir, device, threshold=0.5, suffix=""):
    """生成测试集提交文件"""
    test_loader = get_dataloader(data_dir, split='test', batch_size=1, num_workers=0, augment=False)
    submit_dir = os.path.join(output_dir, f'submission{suffix}')
    os.makedirs(submit_dir, exist_ok=True)

    print(f"\nGenerating submission with threshold={threshold:.2f}...", flush=True)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            try:
                images = batch['image'].to(device)
                masks = batch['mask'].cpu().numpy()
                file_id = batch['file_id'][0] if isinstance(batch['file_id'], (list, tuple)) else batch['file_id']

                probs = ensemble_predict(models, images)
                pred_prob = probs.cpu().numpy()[0, 0]
                mask = masks[0, 0]

                pred_bin = (pred_prob > threshold).astype(np.uint8) * 255
                pred_bin = pred_bin * (mask > 0).astype(np.uint8)

                img_num = int(file_id)
                save_path = os.path.join(submit_dir, f"{img_num}.png")
                Image.fromarray(pred_bin).save(save_path)
                print(f"[{idx+1}/{len(test_loader.dataset)}] Saved: {save_path}", flush=True)

            except Exception as e:
                print(f"Error at idx {idx}: {e}", flush=True)
                traceback.print_exc()

    zip_path = os.path.join(output_dir, f'submission{suffix}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 21):
            png_path = os.path.join(submit_dir, f"{i}.png")
            if os.path.exists(png_path):
                zf.write(png_path, f"{i}.png")

    print(f"\nSubmission ready: {zip_path}", flush=True)
    return zip_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full+fold',
                        choices=['full', 'fold', 'full+fold'],
                        help='full: only best_model_full.pth, fold: only 5 folds, full+fold: all 6 models (default)')
    args = parser.parse_args()

    data_dir, output_dir = get_autodl_paths()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"Mode: {args.mode}", flush=True)

    models = load_models(output_dir, device, base_ch=64, mode=args.mode)
    if not models:
        print("ERROR: No model found. Please train or copy models to output_dir.", flush=True)
        return

    best_thresh, best_dice = search_threshold(models, data_dir, device)

    print("\n" + "="*70, flush=True)
    print("Optimization summary:", flush=True)
    print(f"  Models used: {len(models)}", flush=True)
    print(f"  Best threshold: {best_thresh:.2f}", flush=True)
    print(f"  Validation Dice at best threshold: {best_dice:.4f}", flush=True)
    print("="*70, flush=True)

    print("\nDo you want to generate submission? (y/N)", flush=True)
    try:
        import sys
        if sys.version_info >= (3, 0):
            choice = input().strip().lower()
        else:
            choice = raw_input().strip().lower()
    except:
        choice = 'n'

    if choice in ['y', 'yes']:
        zip_path = generate_submission(models, data_dir, output_dir, device, threshold=best_thresh, suffix=f"_thresh{best_thresh:.2f}")
        print(f"\nSubmission generated: {zip_path}", flush=True)
    else:
        print("\nSkipping submission generation. Use --force-submit to bypass confirmation.", flush=True)


if __name__ == '__main__':
    main()
