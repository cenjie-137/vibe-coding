"""dataset_glas.py — GLaS 数据加载：图像 + 二值掩膜 + 轮廓 GT + H&E 增强

每个样本返回:
  image   : (3, H, W) float32 [0,1]
  seg     : (H, W) float32 {0,1}  腺体前景
  contour : (H, W) float32 {0,1}  腺体边界（morphological gradient）
"""
import csv
import os

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from datasets_puma import apply_stain_augmentation


def mask_to_contour(mask, ksize=3):
    mask_u8 = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(mask_u8, kernel)
    eroded = cv2.erode(mask_u8, kernel)
    return (dilated - eroded) > 0


def augment_glas(img, seg, contour):
    # 水平/垂直翻转
    if np.random.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        seg = np.ascontiguousarray(seg[:, ::-1])
        contour = np.ascontiguousarray(contour[:, ::-1])
    if np.random.random() < 0.5:
        img = np.ascontiguousarray(img[::-1])
        seg = np.ascontiguousarray(seg[::-1])
        contour = np.ascontiguousarray(contour[::-1])
    # 90/180/270 旋转
    if np.random.random() < 0.5:
        k = np.random.randint(1, 4)
        img = np.ascontiguousarray(np.rot90(img, k))
        seg = np.ascontiguousarray(np.rot90(seg, k))
        contour = np.ascontiguousarray(np.rot90(contour, k))
    # 连续旋转 ±15° + 缩放 0.85~1.15
    H, W = img.shape[:2]
    angle = np.random.uniform(-15, 15)
    scale = np.random.uniform(0.85, 1.15)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    seg = cv2.warpAffine(seg, M, (W, H), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REFLECT_101)
    contour = cv2.warpAffine(contour, M, (W, H), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_REFLECT_101)
    # 亮度/对比度抖动
    if np.random.random() < 0.5:
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-0.1, 0.1)
        img = np.clip(alpha * img + beta, 0.0, 1.0).astype(np.float32)
    # H&E 染色增强
    img = apply_stain_augmentation(img)
    return img, seg, contour


class GLASDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, target_size=768, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img = Image.open(os.path.join(self.image_dir, fname + '.png')).convert('RGB')
        mask = np.array(Image.open(os.path.join(self.mask_dir, fname + '.png')).convert('L'))
        seg = (mask > 0)
        contour = mask_to_contour(mask)

        img = np.array(
            img.resize((self.target_size, self.target_size), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        seg = cv2.resize(
            seg.astype(np.uint8), (self.target_size, self.target_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        contour = cv2.resize(
            contour.astype(np.uint8), (self.target_size, self.target_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

        if self.augment:
            img, seg, contour = augment_glas(img, seg, contour)

        img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)).astype(np.float32))
        return {
            'image': torch.from_numpy(img),
            'seg': torch.from_numpy(np.ascontiguousarray(seg)),
            'contour': torch.from_numpy(np.ascontiguousarray(contour)),
            'file_id': fname,
        }


def load_grades(metadata_path):
    grades = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                grades[row['name'].strip()] = row['grade_glas'].strip()
    return grades


def get_glas_dataloaders(data_dir, fold=0, total_folds=5, batch_size=4,
                         target_size=768, num_workers=4):
    image_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    grades = load_grades(os.path.join(data_dir, 'metadata.csv'))

    train_files = sorted(
        f[:-4] for f in os.listdir(mask_dir)
        if f.startswith('train_') and f.endswith('.png')
    )
    # 按良性/恶性分层划分
    y = np.array([0 if grades.get(f) == 'benign' else 1 for f in train_files])

    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
    splits = list(skf.split(train_files, y))
    train_idx, val_idx = splits[fold]
    train_list = [train_files[i] for i in train_idx]
    val_list = [train_files[i] for i in val_idx]

    n_benign = sum(1 for f in train_list if grades.get(f) == 'benign')
    n_malig = len(train_list) - n_benign
    print(f"Fold {fold}: train={len(train_list)} (benign {n_benign} / malignant {n_malig}), "
          f"val={len(val_list)}")

    train_ds = GLASDataset(image_dir, mask_dir, train_list, target_size, augment=True)
    val_ds = GLASDataset(image_dir, mask_dir, val_list, target_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_list, val_list