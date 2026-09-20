"""
PUMA submission generator: ensemble inference + Docker container creation.

Usage:
    # Run validation inference (evaluate per-fold metrics)
    python generate_puma_submit.py --folds 0 1 2 3 4 --base_ch 48 --target_size 512

    # Create Docker submission package
    python generate_puma_submit.py --folds 0 1 2 3 4 --create_docker --base_ch 48

    # Create Docker with specific export directory
    python generate_puma_submit.py --create_docker --export_dir /path/to/export
"""

import os
import sys
import json
import shutil
import argparse
import torch
import numpy as np
from PIL import Image
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets_puma import (
    get_puma_dataloaders, get_puma_test_dataloader,
    NUM_CLASSES, CLASS_NAMES, find_image
)
from models import MultiResUNet, FoundationUNet
from engines_puma import evaluate_puma


def get_puma_paths():
    """Get data/output paths (server or local)."""
    autodl_root = '/root/autodl-tmp'
    if os.path.exists(autodl_root):
        project_dir = os.path.join(autodl_root, 'medseg_project')
        data_dir = os.path.join(project_dir, 'data', 'PUMA')
        output_dir = os.path.join(project_dir, 'output_puma')
    else:
        local_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(local_root), 'medseg_project', 'data', 'PUMA')
        output_dir = os.path.join(os.path.dirname(local_root), 'medseg_project', 'output_puma')
    return data_dir, output_dir


def load_models(output_dir, fold_list, device, base_ch=48, attention='se', model_type='multiresunet'):
    """Load trained models for ensemble."""
    models = []
    for fold in fold_list:
        if model_type == 'segformer_b2':
            import torch.nn as nn
            import torch.nn.functional as F
            from transformers import SegformerForSemanticSegmentation, SegformerConfig

            class SegFormerWrapper(nn.Module):
                def __init__(self, num_classes=6):
                    super().__init__()
                    config = SegformerConfig.from_pretrained(
                        'nvidia/segformer-b2-finetuned-ade-512-512')
                    config.num_labels = num_classes
                    self.model = SegformerForSemanticSegmentation.from_pretrained(
                        'nvidia/segformer-b2-finetuned-ade-512-512',
                        config=config, ignore_mismatched_sizes=True
                    )
                def forward(self, x):
                    out = self.model(x)
                    logits = out.logits
                    return F.interpolate(logits, size=x.shape[2:],
                                          mode='bilinear', align_corners=False)

            model = SegFormerWrapper(num_classes=NUM_CLASSES).to(device)
        elif model_type == 'unet_resnet34':
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name='resnet34', encoder_weights=None,
                classes=NUM_CLASSES, activation=None
            ).to(device)
        elif model_type == 'unet_convnextv2_base':
            import segmentation_models_pytorch as smp
            try:
                model = smp.Unet(
                    encoder_name='convnextv2_base', encoder_weights=None,
                    classes=NUM_CLASSES, activation=None,
                    decoder_channels=[256, 128, 64, 32, 16],
                ).to(device)
            except (KeyError, Exception):
                model = smp.Unet(
                    encoder_name='timm-efficientnet-b3', encoder_weights=None,
                    classes=NUM_CLASSES, activation=None,
                    decoder_channels=[256, 128, 64, 32, 16],
                ).to(device)
        elif model_type == 'unet_efficientnet_b3':
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name='timm-efficientnet-b3', encoder_weights=None,
                classes=NUM_CLASSES, activation=None,
                decoder_channels=[256, 128, 64, 32, 16],
            ).to(device)
        elif model_type == 'foundation_uni':
            os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
            model = FoundationUNet(
                num_classes=NUM_CLASSES, model_source="UNI"
            ).to(device)
        else:
            model = MultiResUNet(
                n_channels=3, n_classes=NUM_CLASSES, base_ch=base_ch, attention=attention
            ).to(device)
        model_path = os.path.join(output_dir, 'fold_' + str(fold), 'best_model.pth')
        if not os.path.exists(model_path):
            print("Warning: " + model_path + " not found, skipping fold " + str(fold), flush=True)
            continue
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
        print("Loaded fold " + str(fold) + " model from " + model_path, flush=True)

    if len(models) == 0:
        raise FileNotFoundError("No models found in " + output_dir + "/fold_*/best_model.pth")

    return models


def ensemble_predict(models, images, use_tta=True):
    """Multi-class ensemble prediction (average logits, then argmax)."""
    outputs_sum = 0.0
    for model in models:
        out = model(images)
        if use_tta:
            out_h = model(torch.flip(images, dims=[3]))
            out_h = torch.flip(out_h, dims=[3])
            out_v = model(torch.flip(images, dims=[2]))
            out_v = torch.flip(out_v, dims=[2])
            out = (out + out_h + out_v) / 3.0
        outputs_sum += out
    outputs_avg = outputs_sum / len(models)
    return torch.argmax(outputs_avg, dim=1)  # BxHxW


def evaluate_validation(data_dir, output_dir, fold_list, device,
                        base_ch=48, attention='se', target_size=512, use_tta=True,
                        model_type='multiresunet'):
    """Run per-fold evaluation and report metrics."""
    all_results = []

    for fold in fold_list:
        # Load single fold model
        models = load_models(output_dir, [fold], device, base_ch, attention, model_type)
        model = models[0]

        _, val_loader, _, _ = get_puma_dataloaders(
            data_dir, fold=fold, total_folds=5,
            batch_size=1, target_size=target_size, num_workers=4
        )

        metrics = evaluate_puma(model, val_loader, device, use_tta=use_tta)

        print("\nFold " + str(fold+1) + " Validation:", flush=True)
        print("  Macro Dice: " + "%.4f" % metrics['mean_dice'], flush=True)
        print("  Micro Dice: " + "%.4f" % metrics['micro_dice'], flush=True)
        print("  Pixel Acc:  " + "%.4f" % metrics['pixel_acc'], flush=True)
        for name in CLASS_NAMES[1:]:
            d = metrics.get('micro_dice_' + name, 0)
            print("    " + name + ": " + "%.4f" % d, flush=True)

        all_results.append(metrics)

    # Average
    mean_micro = np.mean([r['micro_dice'] for r in all_results])
    mean_macro = np.mean([r['mean_dice'] for r in all_results])
    print("\n" + "=" * 80, flush=True)
    print("Average across " + str(len(all_results)) + " folds:", flush=True)
    print("  Macro Dice: " + "%.4f" % mean_macro, flush=True)
    print("  Micro Dice: " + "%.4f" % mean_micro, flush=True)

    return all_results


def evaluate_ensemble(data_dir, output_dir, fold_list, device,
                       base_ch=48, attention='se', target_size=512,
                       use_tta=True, model_type='multiresunet'):
    """Evaluate 5-fold ensemble: all models predict on each fold's validation set."""
    print("\n" + "=" * 80, flush=True)
    print("5-Fold Ensemble Evaluation", flush=True)
    print("=" * 80, flush=True)

    models = load_models(output_dir, fold_list, device, base_ch, attention, model_type)
    print("Loaded " + str(len(models)) + " models for ensemble", flush=True)

    from engines_puma import compute_metrics_puma
    all_preds = []
    all_masks = []

    for fold in fold_list:
        _, val_loader, _, _ = get_puma_dataloaders(
            data_dir, fold=fold, total_folds=5,
            batch_size=1, target_size=target_size, num_workers=4
        )
        for batch in val_loader:
            imgs, masks = batch['image'], batch['label']
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                pred = ensemble_predict(models, imgs, use_tta=use_tta)
            all_preds.append(pred.cpu())
            all_masks.append(masks.cpu())
        print("  Processed fold " + str(fold+1) + " validation set", flush=True)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_masks = torch.cat(all_masks, dim=0).numpy()
    metrics = compute_metrics_puma(all_preds, all_masks, NUM_CLASSES)

    # Compute micro Dice (official PUMA metric: pool all pixels, then per-class Dice)
    micro_dices = []
    for c in range(1, NUM_CLASSES):
        pred_c = (all_preds == c)
        target_c = (all_masks == c)
        intersection = (pred_c & target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0:
            micro_dices.append(1.0)
        elif intersection == 0:
            micro_dices.append(0.0)
        else:
            micro_dices.append(2.0 * intersection / (union + 1e-6))
        metrics['micro_dice_' + CLASS_NAMES[c]] = float(micro_dices[-1])
    metrics['micro_dice'] = float(np.mean(micro_dices))

    print("\nEnsemble Results:", flush=True)
    print("  Macro Dice: " + "%.4f" % metrics['mean_dice'], flush=True)
    print("  Micro Dice: " + "%.4f" % metrics['micro_dice'], flush=True)
    print("  Pixel Acc:  " + "%.4f" % metrics['pixel_acc'], flush=True)
    for name in CLASS_NAMES[1:]:
        d = metrics.get('micro_dice_' + name, 0)
        print("    " + name + ": " + "%.4f" % d, flush=True)

    return metrics


# ==================== Docker Submission ====================

INFERENCE_SH = '''#!/bin/bash
set -e

echo "Starting PUMA tissue segmentation inference..."

cd /opt/app/src
python inference.py

echo "Inference complete!"
'''

DOCKERFILE = '''FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install dependencies
RUN pip install --no-cache-dir \\
    pillow \\
    opencv-python-headless \\
    scikit-learn \\
    numpy \\
    tifffile \\
    segmentation_models_pytorch \\
    transformers \\
    timm \\
    huggingface_hub

# Pre-download UNI model weights during build
ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "import timm; m = timm.create_model('hf-hub:MahmoodLab/UNI', pretrained=True, init_values=1e-5, num_classes=0, dynamic_img_size=True); print('UNI cached')"

# Create app directory
WORKDIR /opt/app
COPY src/ /opt/app/src/
COPY weights/ /opt/app/weights/
COPY config.json /opt/app/config.json
COPY inference.sh /opt/app/inference.sh
RUN chmod +x /opt/app/inference.sh

# Set offline mode for inference (no internet in GC evaluation)
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Set entrypoint
ENTRYPOINT ["/opt/app/inference.sh"]
'''


def generate_docker_submission(data_dir, output_dir, fold_list, device,
                               base_ch=48, attention='se', target_size=512,
                               use_tta=True, export_dir=None, model_type='multiresunet'):
    """Create Docker submission package for PUMA grand-challenge.

    Structure:
        export/
        ├── Dockerfile
        ├── inference.sh
        ├── config.json           # Model configuration
        ├── src/
        │   ├── inference.py      # Main inference script
        │   ├── models.py         # Model definition
        │   └── datasets_puma.py  # Dataset utilities
        └── weights/
            ├── fold_0.pth
            ├── fold_1.pth
            └── ...
    """
    if export_dir is None:
        export_dir = os.path.join(output_dir, 'docker_submission')

    # Clean up old export
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    src_dir = os.path.join(export_dir, 'src')
    weights_dir = os.path.join(export_dir, 'weights')
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Copy model weights
    print("Copying model weights...", flush=True)
    for fold in fold_list:
        src = os.path.join(output_dir, 'fold_' + str(fold), 'best_model.pth')
        if os.path.exists(src):
            dst = os.path.join(weights_dir, 'fold_' + str(fold) + '.pth')
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            print("  fold_" + str(fold) + ".pth (" + "%.1f" % size_mb + " MB)", flush=True)

    # Copy source code
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in ['models.py', 'datasets_puma.py']:
        src = os.path.join(current_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(src_dir, fname))
            print("  Copied " + fname, flush=True)

    # Copy inference_docker.py as inference.py (self-contained)
    inference_src = os.path.join(current_dir, 'inference_docker.py')
    if os.path.exists(inference_src):
        shutil.copy2(inference_src, os.path.join(src_dir, 'inference.py'))
        print("  Copied inference_docker.py -> inference.py", flush=True)
    else:
        raise FileNotFoundError(
            "inference_docker.py not found at " + inference_src
        )

    # Write config.json
    config = {
        'model_type': model_type,
        'base_ch': base_ch,
        'attention': attention,
        'target_size': target_size,
        'use_tta': use_tta,
        'fold_list': fold_list,
    }
    with open(os.path.join(export_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print("  Written config.json: " + str(config), flush=True)

    # Write inference.sh
    with open(os.path.join(export_dir, 'inference.sh'), 'w') as f:
        f.write(INFERENCE_SH)
    os.chmod(os.path.join(export_dir, 'inference.sh'), 0o755)

    # Write Dockerfile
    with open(os.path.join(export_dir, 'Dockerfile'), 'w') as f:
        f.write(DOCKERFILE)

    # Print summary
    print("\n" + "=" * 80, flush=True)
    print("Docker submission package created at: " + export_dir, flush=True)
    print("=" * 80, flush=True)

    total_size = 0
    print("\nFiles:", flush=True)
    for root, dirs, files in os.walk(export_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, export_dir)
            size = os.path.getsize(fpath)
            total_size += size
            print("  " + rel + " (" + "%.0f" % (size / 1024) + " KB)", flush=True)

    print("\nTotal size: " + "%.1f" % (total_size / (1024 * 1024)) + " MB", flush=True)

    print("\nTo build and test locally:", flush=True)
    print("  cd " + export_dir, flush=True)
    print("  docker build -t puma-submission .", flush=True)
    print("  docker run --gpus all -v /path/to/test/input:/input -v /path/to/output:/output puma-submission", flush=True)

    print("\nTo export for grand-challenge submission:", flush=True)
    print("  docker save puma-submission | gzip > puma-submission.tar.gz", flush=True)

    print("\nNote: Upload puma-submission.tar.gz to https://puma.grand-challenge.org/", flush=True)

    return export_dir


def main():
    parser = argparse.ArgumentParser(description='PUMA Submission Generator')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--base_ch', type=int, default=48)
    parser.add_argument('--attention', type=str, default='se')
    parser.add_argument('--target_size', type=int, default=512)
    parser.add_argument('--no_tta', action='store_true')
    parser.add_argument('--create_docker', action='store_true', help='Create Docker submission package')
    parser.add_argument('--export_dir', type=str, default=None)
    parser.add_argument('--model', type=str, default='foundation_uni',
                        choices=['multiresunet', 'unet_resnet34', 'unet_convnextv2_base',
                                 'unet_efficientnet_b3', 'segformer_b2', 'foundation_uni'],
                        help='Model architecture')
    args = parser.parse_args()

    data_dir, output_dir = get_puma_paths()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device), flush=True)

    if args.create_docker:
        generate_docker_submission(
            data_dir, output_dir, args.folds, device,
            base_ch=args.base_ch, attention=args.attention,
            target_size=args.target_size, use_tta=not args.no_tta,
            export_dir=args.export_dir, model_type=args.model
        )
    else:
        evaluate_validation(
            data_dir, output_dir, args.folds, device,
            base_ch=args.base_ch, attention=args.attention,
            target_size=args.target_size, use_tta=not args.no_tta,
            model_type=args.model
        )
        if len(args.folds) > 1:
            evaluate_ensemble(
                data_dir, output_dir, args.folds, device,
                base_ch=args.base_ch, attention=args.attention,
                target_size=args.target_size, use_tta=not args.no_tta,
                model_type=args.model
            )


if __name__ == '__main__':
    main()
