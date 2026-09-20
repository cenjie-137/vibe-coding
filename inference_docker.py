"""
PUMA grand-challenge Docker inference script.

Reads input TIF from /input/, runs 5-model ensemble inference with sliding window,
writes tissue segmentation mask to /output/ and empty nuclei JSON.

Docker paths (grand-challenge format):
  Input:  /input/images/melanoma-whole-slide-image/<uuid>.tif
  Output: /output/images/melanoma-tissue-mask-segmentation/<uuid>.tif
  Nuclei: /output/melanoma-3-class-nuclei-segmentation.json

Output TIFF format (from baseline output_rename.py):
  dtype: int8, resolution: (300, 300) DPI
  MinSampleValue=1, MaxSampleValue=max pixel value

Usage (Docker):   python inference.py
Usage (local):    python inference_docker.py --input /path/to/input --output /path/to/output --weights /path/to/weights
"""

import os
import sys
import glob
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets_puma import NUM_CLASSES, CLASS_NAMES


# ==================== Configuration ====================

def load_config():
    """Load configuration from config.json (Docker) or environment variables."""
    # Try config.json in parent directory (Docker: /opt/app/config.json)
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config.json'
    )
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    # Fallback to environment variables with defaults
    fold_str = os.environ.get('FOLD_LIST', '0 1 2 3 4')
    return {
        'model_type': os.environ.get('MODEL_TYPE', 'foundation_uni'),
        'base_ch': int(os.environ.get('BASE_CH', '48')),
        'attention': os.environ.get('ATTENTION', 'se'),
        'target_size': int(os.environ.get('TARGET_SIZE', '512')),
        'use_tta': os.environ.get('USE_TTA', '1') == '1',
        'fold_list': [int(f) for f in fold_str.split()],
    }


CONFIG = load_config()
MODEL_TYPE = CONFIG.get('model_type', 'foundation_uni')
BASE_CH = CONFIG.get('base_ch', 48)
ATTENTION = CONFIG.get('attention', 'se')
TARGET_SIZE = CONFIG.get('target_size', 512)
USE_TTA = CONFIG.get('use_tta', True)
FOLD_LIST = CONFIG.get('fold_list', [0, 1, 2, 3, 4])

# grand-challenge Docker paths
INPUT_DIR = '/input/images/melanoma-whole-slide-image'
OUTPUT_DIR = '/output/images/melanoma-tissue-mask-segmentation'
NUCLEI_OUTPUT = '/output/melanoma-3-class-nuclei-segmentation.json'
WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'weights'
)


# ==================== Model Definitions ====================

class SegFormerWrapper(nn.Module):
    """SegFormer-B2 wrapper for PUMA tissue segmentation."""

    def __init__(self, num_classes=6):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        config = SegformerConfig.from_pretrained(
            'nvidia/segformer-b2-finetuned-ade-512-512')
        config.num_labels = num_classes
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            'nvidia/segformer-b2-finetuned-ade-512-512',
            config=config, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        out = self.model(x)
        return F.interpolate(out.logits, size=x.shape[2:],
                             mode='bilinear', align_corners=False)


def load_models(device, weights_dir=None):
    """Load trained models for ensemble inference."""
    wdir = weights_dir or WEIGHTS_DIR
    models = []

    for fold in FOLD_LIST:
        if MODEL_TYPE == 'unet_resnet34':
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name='resnet34', encoder_weights=None,
                classes=NUM_CLASSES, activation=None
            ).to(device)
        elif MODEL_TYPE == 'unet_convnextv2_base':
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
        elif MODEL_TYPE == 'unet_efficientnet_b3':
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name='timm-efficientnet-b3', encoder_weights=None,
                classes=NUM_CLASSES, activation=None,
                decoder_channels=[256, 128, 64, 32, 16],
            ).to(device)
        elif MODEL_TYPE == 'segformer_b2':
            model = SegFormerWrapper(num_classes=NUM_CLASSES).to(device)
        elif MODEL_TYPE == 'foundation_uni':
            from models import FoundationUNet
            os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            model = FoundationUNet(
                num_classes=NUM_CLASSES, model_source="UNI"
            ).to(device)
        else:
            from models import MultiResUNet
            model = MultiResUNet(
                n_channels=3, n_classes=NUM_CLASSES,
                base_ch=BASE_CH, attention=ATTENTION
            ).to(device)

        weight_path = os.path.join(wdir, f'fold_{fold}.pth')
        if not os.path.exists(weight_path):
            print(f'Warning: {weight_path} not found, skipping fold {fold}')
            continue

        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        print(f'Loaded fold {fold} from {weight_path}')

    if len(models) == 0:
        raise FileNotFoundError(f'No model weights found in {wdir}!')

    print(f'Total models loaded: {len(models)}')
    return models


# ==================== Ensemble Inference ====================

def ensemble_predict_logits(models, img_tensor, use_tta=True):
    """Get averaged logits from ensemble with optional TTA."""
    outputs_sum = 0.0
    for m in models:
        out = m(img_tensor)
        if use_tta:
            out_h = m(torch.flip(img_tensor, dims=[3]))
            out_h = torch.flip(out_h, dims=[3])
            out_v = m(torch.flip(img_tensor, dims=[2]))
            out_v = torch.flip(out_v, dims=[2])
            out = (out + out_h + out_v) / 3.0
        outputs_sum += out
    return outputs_sum / len(models)


def sliding_window_inference(models, image_np, target_size=512, overlap=64,
                              device='cpu', use_tta=True, batch_size=4):
    """Process images of any size using sliding window with probability blending.

    For small images (<= target_size), processes directly with padding.
    For large images, divides into overlapping patches and blends softmax
    probabilities in overlap regions for seamless stitching.

    Args:
        image_np: HxWx3 float32 array (0-1 range)
        target_size: patch size for inference
        overlap: overlap between adjacent patches (pixels)
        device: torch device
        use_tta: test-time augmentation (flip H/V)
        batch_size: patches per forward pass

    Returns:
        pred: HxW uint8 array with class indices 0-5
    """
    h, w = image_np.shape[:2]

    # Small image: process directly
    if h <= target_size and w <= target_size:
        pad_h = target_size - h
        pad_w = target_size - w
        padded = np.pad(image_np,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode='reflect')
        img_tensor = torch.from_numpy(
            padded.transpose(2, 0, 1)
        ).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = ensemble_predict_logits(models, img_tensor, use_tta)
            probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(probs[0, :, :h, :w], dim=0)
        return pred.cpu().numpy().astype(np.uint8)

    # Large image: sliding window
    step = target_size - overlap
    h_steps = max(1, int(np.ceil((h - overlap) / step)))
    w_steps = max(1, int(np.ceil((w - overlap) / step)))

    # Probability accumulators
    pred_sum = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    # Collect patches and coordinates
    patches = []
    coords = []
    for hi in range(h_steps):
        for wi in range(w_steps):
            y1 = min(hi * step, max(0, h - target_size))
            y2 = min(y1 + target_size, h)
            x1 = min(wi * step, max(0, w - target_size))
            x2 = min(x1 + target_size, w)

            patch = image_np[y1:y2, x1:x2]
            ph = target_size - (y2 - y1)
            pw = target_size - (x2 - x1)
            if ph > 0 or pw > 0:
                patch = np.pad(patch,
                               ((0, ph), (0, pw), (0, 0)),
                               mode='reflect')

            patches.append(patch)
            coords.append((y1, y2, x1, x2))

    print(f'  Sliding window: {len(patches)} patches '
          f'({h_steps}x{w_steps}), step={step}, overlap={overlap}')

    # Process in batches
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i + batch_size]
        batch_coords = coords[i:i + batch_size]

        batch_tensor = torch.from_numpy(
            np.array([p.transpose(2, 0, 1) for p in batch_patches])
        ).float().to(device)

        with torch.no_grad():
            logits = ensemble_predict_logits(models, batch_tensor, use_tta)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        for j, (y1, y2, x1, x2) in enumerate(batch_coords):
            ph = y2 - y1
            pw = x2 - x1
            pred_sum[y1:y2, x1:x2] += probs[j, :, :ph, :pw].transpose(1, 2, 0)
            count[y1:y2, x1:x2] += 1.0

    # Average overlapping regions and take argmax
    count = np.maximum(count, 1.0)
    pred = np.argmax(pred_sum / count[..., np.newaxis], axis=2)
    return pred.astype(np.uint8)


# ==================== Output ====================

def save_tissue_mask(pred, output_path):
    """Save tissue mask as TIFF with PUMA-required metadata.

    Format (from baseline output_rename.py):
    - dtype: int8
    - resolution: (300, 300) DPI
    - MinSampleValue tag: 1
    - MaxSampleValue tag: max pixel value
    """
    try:
        import tifffile
        max_val = int(pred.max())
        pred_int8 = pred.astype(np.int8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with tifffile.TiffWriter(output_path) as tif:
            tif.write(
                pred_int8,
                resolution=(300, 300),
                extratags=[
                    ('MinSampleValue', 'I', 1, int(1)),
                    ('MaxSampleValue', 'I', 1, max_val),
                ]
            )
        print(f'  Saved (tifffile): {output_path} '
              f'shape={pred.shape} max={max_val}')
    except ImportError:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(pred.astype(np.uint8)).save(
            output_path, format='TIFF')
        print(f'  Saved (PIL fallback): {output_path} '
              f'shape={pred.shape}')


def save_empty_nuclei(nuclei_path=None):
    """Save empty nuclei detection JSON (tissue-only submission)."""
    path = nuclei_path or NUCLEI_OUTPUT
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'type': 'Multiple polygons',
            'polygons': [],
            'version': {'major': 1, 'minor': 0}
        }, f)
    print(f'Saved nuclei output: {path} (empty - tissue only submission)')


# ==================== Main ====================

def process_image(input_path, output_path, models, device):
    """Process a single input image: read -> sliding window inference -> save."""
    basename = os.path.splitext(os.path.basename(input_path))[0]
    print(f'\nProcessing: {basename}')

    img = Image.open(input_path).convert('RGB')
    orig_w, orig_h = img.size
    print(f'  Original size: {orig_w}x{orig_h}')

    img_np = np.array(img, dtype=np.float32) / 255.0

    pred = sliding_window_inference(
        models, img_np,
        target_size=TARGET_SIZE,
        overlap=64,
        device=device,
        use_tta=USE_TTA,
        batch_size=4
    )

    save_tissue_mask(pred, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='PUMA Docker Inference')
    parser.add_argument('--input', type=str, default=None,
                        help='Override input directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--weights', type=str, default=None,
                        help='Override weights directory')
    parser.add_argument('--nuclei_output', type=str, default=None,
                        help='Override nuclei JSON path')
    args = parser.parse_args()

    input_dir = args.input or INPUT_DIR
    output_dir = args.output or OUTPUT_DIR
    weights_dir = args.weights or WEIGHTS_DIR
    nuclei_path = args.nuclei_output or NUCLEI_OUTPUT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Input: {input_dir}')
    print(f'Output: {output_dir}')
    print(f'Weights: {weights_dir}')
    print(f'Model: {MODEL_TYPE}, folds={FOLD_LIST}, '
          f'TTA={USE_TTA}, target_size={TARGET_SIZE}')

    models = load_models(device, weights_dir)

    # Find input files (recursive, sorted for determinism)
    input_files = []
    for ext in ['*.tif', '*.tiff']:
        input_files.extend(
            glob.glob(os.path.join(input_dir, '**', ext), recursive=True)
        )

    if len(input_files) == 0:
        print(f'ERROR: No input files found in {input_dir}!')
        sys.exit(1)

    print(f'Found {len(input_files)} input images')

    for input_path in sorted(input_files):
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, basename + '.tif')
        process_image(input_path, output_path, models, device)

    save_empty_nuclei(nuclei_path)

    print('\nInference complete!')


if __name__ == '__main__':
    main()
