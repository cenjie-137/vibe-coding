"""
Convert PUMA dataset to nnU-Net v2 / CamylaNet format.

PUMA: .tif (1024x1024 RGB) + GeoJSON annotations -> .nii.gz + dataset.json

Output structure:
  camylanet_raw/DatasetXXX_PUMA/
    ├── imagesTr/
    │   ├── PUMA_0000_0000.nii.gz  (R channel)
    │   ├── PUMA_0000_0001.nii.gz  (G channel)
    │   ├── PUMA_0000_0002.nii.gz  (B channel)
    │   └── ...
    ├── labelsTr/
    │   ├── PUMA_0000.nii.gz       (6-class label, uint8)
    │   └── ...
    └── dataset.json

Usage:
  python convert_puma_to_nnunet.py \\
      --data_dir /root/autodl-tmp/medseg_project/data/PUMA \\
      --output_dir /root/autodl-tmp/camylanet_raw \\
      --dataset_id 101

After conversion, run CamylaNet preprocessing:
  python -c "import camylanet; camylanet.plan_and_preprocess(101, configurations=['2d'])"
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

try:
    import SimpleITK as sitk
except ImportError:
    print("SimpleITK not found. Install with: pip install SimpleITK")
    sys.exit(1)

from PIL import Image

# Import mask generation from existing code
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets_puma import (
    geojson_to_mask, find_geojson, find_image,
    TISSUE_CLASSES, CLASS_NAMES, NUM_CLASSES
)


def save_channel_nii_gz(channel_arr, path):
    """Save a single 2D channel as .nii.gz (3D with z=1 for nnU-Net 2D)."""
    arr_3d = channel_arr[:, :, np.newaxis]
    img = sitk.GetImageFromArray(arr_3d.astype(np.float32))
    sitk.WriteImage(img, path)


def save_label_nii_gz(label_arr, path):
    """Save a 2D label array as .nii.gz (uint8, 3D with z=1)."""
    arr_3d = label_arr[:, :, np.newaxis]
    img = sitk.GetImageFromArray(arr_3d.astype(np.uint8))
    sitk.WriteImage(img, path)


def create_dataset_json(output_dir, dataset_name="PUMA", num_cases=0):
    """Create nnU-Net v2 dataset.json for PUMA."""
    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "file_ending": ".nii.gz",
        "labels": {
            "background": 0,
            "stroma": 1,
            "blood_vessel": 2,
            "tumor": 3,
            "epidermis": 4,
            "necrosis": 5
        },
        "numTraining": num_cases,
        "name": dataset_name,
        "description": "PUMA Challenge - 5-class tissue segmentation in H&E melanoma ROIs",
        "reference": "https://puma.grand-challenge.org/",
        "licence": "CC-BY-NC 4.0",
        "release": "1.0 06/2024",
        "date": datetime.now().strftime("%Y-%m-%d"),
    }

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    print(f"  dataset.json written to {json_path}")
    return json_path


def create_splits(all_basenames, preprocessed_dir, dataset_folder_name,
                  total_folds=5, random_state=42):
    """Create 5-fold stratified split file for nnU-Net v2.

    Stratification: by primary/metastatic (inferred from filename),
    matching the logic in datasets_puma.py.
    """
    labels = np.array([1 if 'metastatic' in f.lower() else 0 for f in all_basenames])
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=random_state)

    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_basenames, labels)):
        train_files = [f"PUMA_{i:04d}" for i in sorted(train_idx)]
        val_files = [f"PUMA_{i:04d}" for i in sorted(val_idx)]
        splits.append({"train": train_files, "val": val_files})

        n_train_pri = sum(1 for i in train_idx if labels[i] == 0)
        n_train_meta = sum(1 for i in train_idx if labels[i] == 1)
        n_val_pri = sum(1 for i in val_idx if labels[i] == 0)
        n_val_meta = sum(1 for i in val_idx if labels[i] == 1)
        print(f"  Fold {fold_idx+1}: Train {len(train_files)} "
              f"({n_train_pri} primary + {n_train_meta} metastatic), "
              f"Val {len(val_files)} "
              f"({n_val_pri} primary + {n_val_meta} metastatic)")

    splits_path = os.path.join(preprocessed_dir, dataset_folder_name,
                               "splits_final.json")
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"  splits_final.json written to {splits_path}")
    return splits_path


def convert_puma(data_dir, output_dir, dataset_id=101):
    """Main conversion function.

    Args:
        data_dir: PUMA data root (containing tif_ROIs/ and tissue_geojson/)
        output_dir: CamylaNet raw data root (e.g. /root/autodl-tmp/camylanet_raw)
        dataset_id: nnU-Net dataset ID (e.g. 101)
    """
    image_dir = os.path.join(data_dir, 'tif_ROIs')
    geojson_dir = os.path.join(data_dir, 'tissue_geojson')
    mask_dir = os.path.join(data_dir, 'masks')

    # Find all TIF images
    tif_files = []
    for ext in ['*.tif', '*.tiff']:
        tif_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    tif_files = sorted(tif_files)

    if len(tif_files) == 0:
        print(f"ERROR: No TIF files found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(tif_files)} TIF images")
    print(f"Image dir: {image_dir}")
    print(f"GeoJSON dir: {geojson_dir}")

    # Prepare output directory
    dataset_name = f"Dataset{dataset_id:03d}_PUMA"
    dataset_dir = os.path.join(output_dir, dataset_name)
    images_tr_dir = os.path.join(dataset_dir, "imagesTr")
    labels_tr_dir = os.path.join(dataset_dir, "labelsTr")
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)

    # Collect basenames for split generation
    all_basenames = [os.path.splitext(os.path.basename(p))[0] for p in tif_files]

    success = 0
    failed = 0
    failed_cases = []

    for idx, tif_path in enumerate(tif_files):
        basename = all_basenames[idx]
        case_id = f"PUMA_{idx:04d}"

        r_path = os.path.join(images_tr_dir, f"{case_id}_0000.nii.gz")
        label_path = os.path.join(labels_tr_dir, f"{case_id}.nii.gz")
        if os.path.exists(r_path) and os.path.exists(label_path):
            success += 1
            continue

        try:
            img_pil = Image.open(tif_path).convert('RGB')
            img = np.array(img_pil)

            mask_png_path = os.path.join(mask_dir, basename + '.png')
            if os.path.exists(mask_png_path):
                mask = np.array(Image.open(mask_png_path), dtype=np.uint8)
            else:
                geojson_path = find_geojson(geojson_dir, basename)
                if geojson_path is None:
                    print(f"  [{idx+1}/{len(tif_files)}] No GeoJSON for {basename}")
                    failed += 1
                    failed_cases.append(basename)
                    continue
                mask = geojson_to_mask(geojson_path, size=img.shape[0])

            for ch in range(3):
                channel_path = os.path.join(images_tr_dir,
                                            f"{case_id}_{ch:04d}.nii.gz")
                save_channel_nii_gz(img[:, :, ch], channel_path)

            save_label_nii_gz(mask, label_path)

            success += 1
            if (idx + 1) % 20 == 0 or idx == 0:
                unique, counts = np.unique(mask, return_counts=True)
                dist = {CLASS_NAMES[c]: cnt for c, cnt in zip(unique, counts)
                        if c < len(CLASS_NAMES)}
                print(f"  [{idx+1}/{len(tif_files)}] {basename} -> {case_id} "
                      f"(classes: {list(dist.keys())})")

        except Exception as e:
            print(f"  [{idx+1}/{len(tif_files)}] ERROR {basename}: {e}")
            failed += 1
            failed_cases.append(basename)

    print(f"\nConversion complete: {success} success, {failed} failed")
    if failed_cases:
        print(f"Failed cases: {failed_cases[:10]}"
              f"{'...' if len(failed_cases)>10 else ''}")

    create_dataset_json(dataset_dir, "PUMA", num_cases=success)

    preprocessed_dir = output_dir.replace('camylanet_raw', 'camylanet_preprocessed')
    create_splits(all_basenames, preprocessed_dir, dataset_name)

    print(f"\nNext steps:")
    print(f"  1. Set environment variables:")
    print(f"     export camylanet_raw={output_dir}")
    print(f"     export camylanet_preprocessed={preprocessed_dir}")
    print(f"  2. Run preprocessing:")
    print(f'     python -c "import camylanet; '
          f'camylanet.plan_and_preprocess({dataset_id}, configurations=[\'2d\'])"')
    print(f"  3. Run experiment_runner.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert PUMA dataset to nnU-Net/CamylaNet format')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='PUMA data root directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='CamylaNet raw data root')
    parser.add_argument('--dataset_id', type=int, default=101,
                        help='nnU-Net dataset ID (default: 101)')
    args = parser.parse_args()

    convert_puma(args.data_dir, args.output_dir, args.dataset_id)
