import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets_siim import get_test_dataloader, apply_clahe, mask2rle
from models import MultiResUNet, MultiResUNetDS


def get_siim_paths():
    autodl_root = '/root/autodl-tmp'
    if os.path.exists(autodl_root):
        project_dir = os.path.join(autodl_root, 'medseg_project')
        data_dir = os.path.join(project_dir, 'data', 'SIIM', 'siim-acr-pneumothorax')
        output_dir = os.path.join(project_dir, 'output_siim')
        test_csv = os.path.join(project_dir, 'data', 'SIIM', 'stage_1_test_images.csv')
    else:
        local_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(local_root), 'medseg_project', 'data', 'SIIM', 'siim-acr-pneumothorax')
        output_dir = os.path.join(os.path.dirname(local_root), 'medseg_project', 'output_siim')
        test_csv = r"C:\Users\32201\Downloads\stage_1_test_images.csv"
    return data_dir, output_dir, test_csv


def ensemble_predict(models, images, use_tta=True):
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
    return torch.sigmoid(outputs_avg)


def post_process_mask(pred_prob, threshold=0.5, min_size=100):
    import cv2
    pred_bin = (pred_prob > threshold).astype(np.uint8)
    if pred_bin.sum() == 0:
        return pred_bin
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
    cleaned = np.zeros_like(pred_bin)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 1
    return cleaned


def generate_submission(data_dir, output_dir, test_csv, device,
                       fold_list=None, target_size=256, threshold=0.5,
                       use_tta=True, use_postprocess=True, min_size=100,
                       use_ds=False, base_ch=32, attention='se',
                       preprocessed=True):
    if fold_list is None:
        fold_list = [0, 1, 2, 3, 4]

    models = []
    model_cls = MultiResUNetDS if use_ds else MultiResUNet
    for fold in fold_list:
        model = model_cls(n_channels=3, n_classes=1, base_ch=base_ch, attention=attention).to(device)
        model_path = os.path.join(output_dir, f'fold_{fold}', 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping fold {fold}", flush=True)
            continue
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
        print(f"Loaded fold {fold} model from {model_path}", flush=True)

    if len(models) == 0:
        raise FileNotFoundError(f"No models found in {output_dir}/fold_*/best_model.pth")

    test_loader, image_ids, file_names = get_test_dataloader(
        data_dir, test_csv, batch_size=8, target_size=target_size,
        num_workers=4, preprocessed=preprocessed
    )
    print(f"Test set size: {len(file_names)}", flush=True)
    print(f"Using {len(models)} models, TTA={use_tta}, "
          f"threshold={threshold}, postprocess={use_postprocess}", flush=True)

    results = []
    original_size = 1024
    global_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            batch_size = images.size(0)

            preds = ensemble_predict(models, images, use_tta=use_tta)
            preds = preds.detach().cpu().numpy()

            for j in range(batch_size):
                pred_prob = preds[j, 0]

                if use_postprocess:
                    pred_mask = post_process_mask(pred_prob, threshold=threshold, min_size=min_size)
                else:
                    pred_mask = (pred_prob > threshold).astype(np.uint8)

                # 上采样回 1024×1024
                pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
                pred_mask_orig = np.array(
                    Image.fromarray(pred_mask_uint8).resize(
                        (original_size, original_size), Image.NEAREST
                    )
                )
                pred_mask_orig = (pred_mask_orig > 127).astype(np.uint8)

                rle = mask2rle(pred_mask_orig)
                img_id = image_ids[global_idx + j]
                results.append({'ImageId': img_id, 'EncodedPixels': rle})

            global_idx += batch_size

    submit_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submit_file = os.path.join(output_dir, f'submission_{timestamp}.csv')
    submit_df.to_csv(submit_file, index=False)
    print(f"\nSubmission saved to: {submit_file}", flush=True)
    print(f"Total predictions: {len(submit_df)}", flush=True)
    print(f"Predicted positive: {(submit_df['EncodedPixels'] != '-1').sum()}", flush=True)
    print(f"Predicted negative: {(submit_df['EncodedPixels'] == '-1').sum()}", flush=True)
    return submit_file


def main():
    parser = argparse.ArgumentParser(description='SIIM-ACR Submission Generator')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--no_tta', action='store_true')
    parser.add_argument('--no_postprocess', action='store_true')
    parser.add_argument('--min_size', type=int, default=100)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--attention', type=str, default='se')
    parser.add_argument('--use_ds', action='store_true')
    parser.add_argument('--preprocessed', action='store_true', default=True)
    parser.add_argument('--no_preprocessed', action='store_false', dest='preprocessed')
    args = parser.parse_args()

    data_dir, output_dir, test_csv = get_siim_paths()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    generate_submission(
        data_dir, output_dir, test_csv, device,
        fold_list=args.folds, target_size=args.target_size,
        threshold=args.threshold, use_tta=not args.no_tta,
        use_postprocess=not args.no_postprocess, min_size=args.min_size,
        use_ds=args.use_ds, base_ch=args.base_ch, attention=args.attention,
        preprocessed=args.preprocessed
    )


if __name__ == '__main__':
    main()
