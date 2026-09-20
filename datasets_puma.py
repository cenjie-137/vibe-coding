"""
PUMA dataset: GeoJSON annotation parsing, mask generation, and data loading
for 5-class tissue semantic segmentation (6 classes including background).

Class mapping (from official evaluation code):
  0: background  (tissue_white_background)
  1: stroma     (tissue_stroma)
  2: blood_vessel (tissue_blood_vessel)
  3: tumor      (tissue_tumor)
  4: epidermis  (tissue_epidermis)
  5: necrosis   (tissue_necrosis)
"""

import os
import json
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold


# ==================== Class Definitions ====================

TISSUE_CLASSES = {
    'tissue_white_background': 0,
    'tissue_stroma': 1,
    'tissue_blood_vessel': 2,
    'tissue_tumor': 3,
    'tissue_epidermis': 4,
    'tissue_necrosis': 5,
}

NUM_CLASSES = 6  # 5 tissue types + background

CLASS_NAMES = ['background', 'stroma', 'blood_vessel', 'tumor', 'epidermis', 'necrosis']


# ==================== File Finding ====================

def find_geojson(geojson_dir, basename):
    """Find a GeoJSON file matching the given basename (recursive search).

    PUMA naming: TIF is 'name.tif', GeoJSON is 'name_tissue.geojson'.
    Also tries exact match as fallback.
    """
    candidates = [basename + '_tissue', basename]
    for candidate in candidates:
        for ext in ['.geojson', '.json']:
            path = os.path.join(geojson_dir, candidate + ext)
            if os.path.exists(path):
                return path
            matches = glob.glob(os.path.join(geojson_dir, '**', candidate + ext), recursive=True)
            if matches:
                return matches[0]
    return None


def find_image(image_dir, basename):
    """Find a TIF image matching the given basename (recursive search)."""
    for ext in ['.tif', '.tiff']:
        path = os.path.join(image_dir, basename + ext)
        if os.path.exists(path):
            return path
        matches = glob.glob(os.path.join(image_dir, '**', basename + ext), recursive=True)
        if matches:
            return matches[0]
    return None


# ==================== GeoJSON -> Mask ====================

def geojson_to_mask(geojson_path, size=1024):
    """Convert tissue GeoJSON annotations to a multi-class mask.

    Args:
        geojson_path: Path to tissue GeoJSON file
        size: Output mask size (default 1024x1024, matching ROI resolution)

    Returns:
        mask: HxW uint8 array with class indices 0-5
    """
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    mask = np.zeros((size, size), dtype=np.uint8)

    for feature in data.get('features', []):
        props = feature.get('properties', {})
        classification = props.get('classification', {})
        class_name = classification.get('name', '')

        if class_name not in TISSUE_CLASSES:
            continue

        class_id = TISSUE_CLASSES[class_name]
        geometry = feature.get('geometry', {})
        geom_type = geometry.get('type', '')
        coords = geometry.get('coordinates', [])

        if geom_type == 'Polygon':
            polygons = [coords]
        elif geom_type == 'MultiPolygon':
            polygons = coords
        else:
            continue

        for poly in polygons:
            if len(poly) > 0 and len(poly[0]) >= 3:
                polygon = np.array(poly[0], dtype=np.int32)
                cv2.fillPoly(mask, [polygon], class_id)

    return mask


def preprocess_masks(data_dir, size=1024):
    """Pre-convert all tissue GeoJSON annotations to PNG masks.

    Run this once before training:
        python datasets_puma.py --data_dir /path/to/PUMA
    """
    image_dir = os.path.join(data_dir, 'tif_ROIs')
    geojson_dir = os.path.join(data_dir, 'tissue_geojson')
    mask_dir = os.path.join(data_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)

    # Find all TIF images
    tif_files = []
    for ext in ['*.tif', '*.tiff']:
        tif_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    tif_files = sorted(tif_files)

    print(f"Found {len(tif_files)} TIF images in {image_dir}")
    print(f"Looking for GeoJSON files in {geojson_dir}")

    success = 0
    failed = 0

    for tif_path in tif_files:
        basename = os.path.splitext(os.path.basename(tif_path))[0]
        mask_path = os.path.join(mask_dir, basename + '.png')

        if os.path.exists(mask_path):
            success += 1
            continue

        geojson_path = find_geojson(geojson_dir, basename)
        if geojson_path is None:
            print(f"  Warning: No GeoJSON found for {basename}")
            failed += 1
            continue

        try:
            mask = geojson_to_mask(geojson_path, size)
            Image.fromarray(mask).save(mask_path)
            success += 1

            unique, counts = np.unique(mask, return_counts=True)
            dist = {CLASS_NAMES[c]: cnt for c, cnt in zip(unique, counts) if c < len(CLASS_NAMES)}
            print(f"  {basename}: {dist}")
        except Exception as e:
            print(f"  Error processing {basename}: {e}")
            failed += 1

    print(f"\nDone! Success: {success}, Failed: {failed}")
    print(f"Masks saved to: {mask_dir}")
    return success, failed


def compute_class_weights(mask_dir, file_list, num_classes=NUM_CLASSES):
    """Compute sqrt inverse-frequency class weights from mask distribution.

    Uses sqrt to avoid extreme weights for very rare classes (necrosis, epidermis).
    """
    class_counts = np.zeros(num_classes, dtype=np.float64)

    for fname in file_list:
        mask_path = os.path.join(mask_dir, fname + '.png')
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            for c in range(num_classes):
                class_counts[c] += (mask == c).sum()

    total = class_counts.sum()
    freq = total / (num_classes * (class_counts + 1))
    class_weights = np.sqrt(freq)
    class_weights = class_weights / class_weights.mean()
    return torch.FloatTensor(class_weights)


# ==================== Augmentation ====================

_STAIN_MATRIX = np.array([
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [0.27, 0.57, 0.78]
])
_STAIN_MATRIX_INV = np.linalg.pinv(_STAIN_MATRIX)


def _rgb_to_hed(rgb_uint8):
    """Convert RGB to HED color space (Ruifrok-Johnston method).

    Returns raw concentration values (range ~0-2.5, not scaled by 255).
    """
    rgb = rgb_uint8.astype(np.float64) / 255.0
    rgb = np.clip(rgb, 1e-6, 1.0)
    od = -np.log(rgb)
    hed = od @ _STAIN_MATRIX_INV
    return hed


def _hed_to_rgb(hed):
    """Convert HED concentrations back to RGB uint8."""
    od = hed @ _STAIN_MATRIX
    rgb = np.exp(-od)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def apply_stain_augmentation(img):
    """HED stain augmentation for H&E histopathology images.

    Perturbs hematoxylin and eosin stain concentrations independently,
    simulating natural staining variation across slides.
    """
    img_uint8 = (img * 255).astype(np.uint8)
    hed = _rgb_to_hed(img_uint8)

    alpha_he = np.random.uniform(0.9, 1.1, 2)
    beta_he = np.random.uniform(-0.05, 0.05, 2)
    hed[:, :, 0] = np.clip(hed[:, :, 0] * alpha_he[0] + beta_he[0], 0, None)
    hed[:, :, 1] = np.clip(hed[:, :, 1] * alpha_he[1] + beta_he[1], 0, None)

    if np.random.random() < 0.3:
        hed[:, :, 2] = np.clip(hed[:, :, 2] * np.random.uniform(0.95, 1.05), 0, None)

    img_aug = _hed_to_rgb(hed)
    return img_aug.astype(np.float32) / 255.0


def augment_image_mask(img, mask, strong=False):
    """Apply geometric + color augmentation.

    Args:
        img: HxWx3 float32 image (0-1 range)
        mask: HxW int64 mask
        strong: if True, enable heavy augmentation (elastic, scale, shear, etc.)
    """
    h, w = img.shape[:2]

    # Horizontal flip
    if np.random.random() < 0.5:
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()

    # Vertical flip
    if np.random.random() < 0.5:
        img = np.flipud(img).copy()
        mask = np.flipud(mask).copy()

    if strong:
        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        # Random scale / zoom (0.8 ~ 1.2)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if scale > 1.0:
                # Crop center
                y0 = (new_h - h) // 2
                x0 = (new_w - w) // 2
                img = img_resized[y0:y0+h, x0:x0+w]
                mask = mask_resized[y0:y0+h, x0:x0+w]
            else:
                # Pad with reflect
                pad_top = (h - new_h) // 2
                pad_bottom = h - new_h - pad_top
                pad_left = (w - new_w) // 2
                pad_right = w - new_w - pad_left
                img = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right,
                                         cv2.BORDER_REFLECT_101)
                mask = cv2.copyMakeBorder(mask_resized, pad_top, pad_bottom, pad_left, pad_right,
                                          cv2.BORDER_REFLECT_101)

        # Random rotation (-179 ~ 179 degrees)
        if np.random.random() < 0.75:
            angle = np.random.uniform(-179, 179)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT_101)

        # Random shear
        if np.random.random() < 0.5:
            shear = np.random.uniform(-0.1, 0.1)
            M = np.float32([[1, shear, 0], [0, 1, 0]])
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT_101)

        # Elastic deformation
        if np.random.random() < 0.7:
            alpha = np.random.uniform(80, 160)
            sigma = 8
            dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
            dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.remap(mask.astype(np.float32), map_x, map_y,
                             interpolation=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_REFLECT_101).astype(np.int64)

        # Random translate
        if np.random.random() < 0.5:
            tx = int(np.random.uniform(-0.05, 0.05) * w)
            ty = int(np.random.uniform(-0.05, 0.05) * h)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT_101)

        # Random brightness / contrast (color_scale = 0.4)
        if np.random.random() < 0.7:
            brightness = np.random.uniform(-0.1, 0.1)
            contrast = np.random.uniform(0.8, 1.2)
            img = np.clip((img - 0.5) * contrast + 0.5 + brightness, 0, 1)

    else:
        # Light rotation
        if np.random.random() < 0.3:
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT_101)

    # HED stain augmentation (always applied when augmenting)
    img = apply_stain_augmentation(img)
    return img, mask


# ==================== Datasets ====================

class CopyPasteAugmenter:
    """Copy-paste augmentation for rare classes.

    Pre-loads rare class regions from training images.
    During training, randomly pastes rare class patches into the current image.
    """

    def __init__(self, image_dir, mask_dir, file_list, target_size=512,
                 rare_classes=None, p=0.3):
        self.target_size = target_size
        self.rare_classes = rare_classes or [5, 4, 2]  # necrosis, epidermis, blood_vessel
        self.p = p

        self.regions = []
        for fname in file_list:
            mask_path = os.path.join(mask_dir, fname + '.png')
            if not os.path.exists(mask_path):
                continue
            mask = np.array(Image.open(mask_path), dtype=np.uint8)
            img_path = find_image(image_dir, fname)
            if img_path is None:
                continue
            img = Image.open(img_path).convert('RGB')
            img = img.resize((target_size, target_size), Image.BILINEAR)
            img = np.array(img, dtype=np.float32) / 255.0
            mask = np.array(Image.fromarray(mask).resize(
                (target_size, target_size), Image.NEAREST), dtype=np.uint8)

            for c in self.rare_classes:
                if (mask == c).sum() > 0:
                    ys, xs = np.where(mask == c)
                    y1, y2 = max(0, ys.min() - 5), min(target_size, ys.max() + 6)
                    x1, x2 = max(0, xs.min() - 5), min(target_size, xs.max() + 6)
                    self.regions.append({
                        'img': img[y1:y2, x1:x2].copy(),
                        'mask': mask[y1:y2, x1:x2].copy()
                    })

        print(f"  Copy-paste: {len(self.regions)} rare class regions loaded")

    def __call__(self, img, mask):
        if np.random.random() > self.p or len(self.regions) == 0:
            return img, mask

        region = self.regions[np.random.randint(len(self.regions))]
        src_img = region['img']
        src_mask = region['mask']

        h, w = img.shape[:2]
        sh, sw = src_img.shape[:2]

        if sh > h or sw > w:
            scale = min(h / sh, w / sw, 0.5)
            new_sh, new_sw = max(1, int(sh * scale)), max(1, int(sw * scale))
            src_img = cv2.resize(src_img, (new_sw, new_sh))
            src_mask = cv2.resize(src_mask, (new_sw, new_sh), interpolation=cv2.INTER_NEAREST)
            sh, sw = new_sh, new_sw

        y = np.random.randint(0, max(1, h - sh))
        x = np.random.randint(0, max(1, w - sw))

        img[y:y+sh, x:x+sw] = src_img
        mask[y:y+sh, x:x+sw] = src_mask

        return img, mask


class PUMADataset(Dataset):
    """PUMA tissue segmentation dataset (multi-class)."""

    def __init__(self, image_dir, mask_dir, file_list, target_size=512,
                 augment=False, copy_paste_augmenter=None, strong_augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.target_size = target_size
        self.augment = augment
        self.copy_paste = copy_paste_augmenter
        self.strong_augment = strong_augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]

        # Load image
        img_path = find_image(self.image_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0

        # Load mask
        mask_path = os.path.join(self.mask_dir, fname + '.png')
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = np.array(Image.fromarray(mask).resize(
            (self.target_size, self.target_size), Image.NEAREST), dtype=np.int64)

        if self.copy_paste is not None:
            img, mask = self.copy_paste(img, mask)
            mask = mask.astype(np.int64)

        if self.augment:
            img, mask = augment_image_mask(img, mask, strong=self.strong_augment)
            mask = mask.astype(np.int64)

        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return {
            'image': torch.from_numpy(img),
            'label': torch.from_numpy(mask.astype(np.int64)),
            'file_id': fname
        }


class PUMATestDataset(Dataset):
    """PUMA test dataset (no masks, for inference)."""

    def __init__(self, image_dir, file_list, target_size=512):
        self.image_dir = image_dir
        self.file_list = file_list
        self.target_size = target_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img_path = find_image(self.image_dir, fname)
        img = Image.open(img_path).convert('RGB')
        orig_size = img.size  # (W, H)
        img_resized = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        img_arr = np.transpose(img_arr, (2, 0, 1))
        return {
            'image': torch.from_numpy(img_arr),
            'file_id': fname,
            'orig_size': orig_size
        }


# ==================== DataLoaders ====================

def compute_sample_weights(mask_dir, file_list, num_classes=NUM_CLASSES):
    """Compute per-image weights for WeightedRandomSampler.

    Images containing rare classes (blood_vessel, epidermis, necrosis)
    get higher weight to ensure the model sees enough rare class pixels.
    """
    rare_classes = [2, 4, 5]  # blood_vessel, epidermis, necrosis

    sample_weights = []
    n_rare_images = 0
    for fname in file_list:
        mask_path = os.path.join(mask_dir, fname + '.png')
        if not os.path.exists(mask_path):
            sample_weights.append(1.0)
            continue

        mask = np.array(Image.open(mask_path))
        weight = 1.0
        has_rare = False
        for c in rare_classes:
            frac = (mask == c).sum() / mask.size
            if frac > 0:
                weight += 3.0
                has_rare = True
        if has_rare:
            n_rare_images += 1
        sample_weights.append(weight)

    print(f"  Weighted sampling: {n_rare_images}/{len(file_list)} images contain rare classes")
    return torch.DoubleTensor(sample_weights)


def get_puma_dataloaders(data_dir, fold=0, total_folds=5, batch_size=4,
                         target_size=512, num_workers=4, use_weighted_sampling=False,
                         use_copy_paste=False, strong_augment=False):
    """Build 5-fold stratified DataLoader for PUMA tissue segmentation.

    Stratification: by primary/metastatic (inferred from filename).
    Optional: WeightedRandomSampler and copy-paste augmentation for rare classes.

    Returns:
        train_loader, val_loader, train_files, val_files
    """
    image_dir = os.path.join(data_dir, 'tif_ROIs')
    mask_dir = os.path.join(data_dir, 'masks')

    # Get all files with both image and mask
    mask_files = set(os.path.splitext(f)[0]
                    for f in os.listdir(mask_dir) if f.endswith('.png'))

    tif_basenames = set()
    for ext in ['*.tif', '*.tiff']:
        for path in glob.glob(os.path.join(image_dir, '**', ext), recursive=True):
            tif_basenames.add(os.path.splitext(os.path.basename(path))[0])

    all_files = sorted(tif_basenames & mask_files)
    print(f"Total samples (image + mask): {len(all_files)}")

    # Stratify by primary/metastatic
    labels = np.array([1 if 'metastatic' in f.lower() else 0 for f in all_files])

    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
    splits = list(skf.split(all_files, labels))

    train_idx, val_idx = splits[fold]
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]

    n_primary_train = sum(1 for f in train_files if 'metastatic' not in f.lower())
    n_meta_train = len(train_files) - n_primary_train
    n_primary_val = sum(1 for f in val_files if 'metastatic' not in f.lower())
    n_meta_val = len(val_files) - n_primary_val

    print(f"Fold {fold+1}/{total_folds}:")
    print(f"  Train: {len(train_files)} ({n_primary_train} primary + {n_meta_train} metastatic)")
    print(f"  Val:   {len(val_files)} ({n_primary_val} primary + {n_meta_val} metastatic)")

    copy_paste_aug = None
    if use_copy_paste:
        copy_paste_aug = CopyPasteAugmenter(
            image_dir, mask_dir, train_files, target_size=target_size, p=0.3
        )

    train_dataset = PUMADataset(image_dir, mask_dir, train_files,
                                target_size=target_size, augment=True,
                                copy_paste_augmenter=copy_paste_aug,
                                strong_augment=strong_augment)
    val_dataset = PUMADataset(image_dir, mask_dir, val_files,
                             target_size=target_size, augment=False)

    if use_weighted_sampling:
        sample_weights = compute_sample_weights(mask_dir, train_files)
        sampler = WeightedRandomSampler(sample_weights, len(train_files), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_files, val_files


def get_puma_test_dataloader(image_dir, batch_size=1, target_size=512, num_workers=4):
    """Load test images for inference (no masks)."""
    tif_basenames = []
    for ext in ['*.tif', '*.tiff']:
        for path in sorted(glob.glob(os.path.join(image_dir, '**', ext), recursive=True)):
            tif_basenames.append(os.path.splitext(os.path.basename(path))[0])

    dataset = PUMATestDataset(image_dir, tif_basenames, target_size=target_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
    return loader, tif_basenames


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PUMA mask preprocessing')
    parser.add_argument('--data_dir', type=str, required=True, help='PUMA data directory')
    parser.add_argument('--size', type=int, default=1024, help='Mask size (default 1024)')
    args = parser.parse_args()
    preprocess_masks(args.data_dir, args.size)
