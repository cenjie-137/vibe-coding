import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold


def apply_clahe(img_uint8):
    """CLAHE 对比度增强"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_uint8)


def rle2mask(rle, width, height):
    """RLE 解码为二值掩码（Kaggle SIIM-ACR 格式：列优先）"""
    if rle == '-1' or pd.isna(rle) or rle == '':
        return np.zeros((height, width), dtype=np.uint8)
    s = rle.split()
    starts, lengths = [np.array(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape((width, height)).T


def mask2rle(mask):
    """二值掩码编码为 RLE（Kaggle SIIM-ACR 提交格式：列优先）"""
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return '-1'
    return ' '.join(str(x) for x in runs)


class SIIMDataset(Dataset):
    """SIIM-ACR 气胸分割数据集
    支持在线模式（原图+CLAHE+缩放）和预处理模式（直接加载预处理好的图片）
    """
    def __init__(self, image_dir, mask_dir, file_list, target_size=512,
                 augment=False, preprocessed=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.target_size = target_size
        self.augment = augment
        self.preprocessed = preprocessed

    def __len__(self):
        return len(self.file_list)

    def _load_image_mask(self, fname):
        img = Image.open(os.path.join(self.image_dir, fname)).convert('L')
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert('L')

        if not self.preprocessed:
            # 在线模式：降采样 + CLAHE
            img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
            mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
            img = np.array(img, dtype=np.uint8)
            mask = np.array(mask, dtype=np.uint8)
            mask = (mask > 127).astype(np.float32)
            img = apply_clahe(img)
        else:
            # 预处理模式：直接加载，图片已是 target_size
            img = np.array(img, dtype=np.uint8)
            mask = np.array(mask, dtype=np.uint8)
            mask = (mask > 127).astype(np.float32)

        # 灰度图复制 3 通道
        img = np.stack([img, img, img], axis=-1)
        img = img.astype(np.float32) / 255.0
        return img, mask

    def _augment(self, img, mask):
        if np.random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if np.random.random() < 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.85, 1.15)
            img = np.clip(img * factor, 0, 1)
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.85, 1.15)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 1)
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)
        if np.random.random() < 0.3:
            angle = np.random.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        return img, mask

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img, mask = self._load_image_mask(fname)
        if self.augment:
            img, mask = self._augment(img, mask)
        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        return {
            'image': torch.from_numpy(img),
            'label': torch.from_numpy(mask),
            'file_id': fname.replace('.png', '')
        }


def _get_dirs(data_dir, img_size, preprocessed):
    """根据是否预处理返回图片和掩码目录"""
    if preprocessed:
        img_dir = os.path.join(data_dir, f'png_images_{img_size}')
        mask_dir = os.path.join(data_dir, f'png_masks_{img_size}')
    else:
        img_dir = os.path.join(data_dir, 'png_images')
        mask_dir = os.path.join(data_dir, 'png_masks')
    return img_dir, mask_dir


def get_siim_dataloaders(data_dir, train_csv, fold=0, total_folds=5,
                         batch_size=8, target_size=512, num_workers=4,
                         preprocessed=False):
    """构建 5-Fold 分层交叉验证的 DataLoader"""
    image_dir, mask_dir = _get_dirs(data_dir, target_size, preprocessed)

    df = pd.read_csv(train_csv)
    file_names = df['new_filename'].values
    has_pneumo = df['has_pneumo'].values

    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
    splits = list(skf.split(file_names, has_pneumo))

    train_idx, val_idx = splits[fold]
    train_files = file_names[train_idx].tolist()
    val_files = file_names[val_idx].tolist()

    train_files = [f for f in train_files if os.path.exists(os.path.join(image_dir, f))]
    val_files = [f for f in val_files if os.path.exists(os.path.join(image_dir, f))]

    train_dataset = SIIMDataset(image_dir, mask_dir, train_files,
                                target_size=target_size, augment=True,
                                preprocessed=preprocessed)
    val_dataset = SIIMDataset(image_dir, mask_dir, val_files,
                             target_size=target_size, augment=False,
                             preprocessed=preprocessed)

    train_has_pneumo = df.set_index('new_filename').loc[train_files, 'has_pneumo'].values
    neg_count = (train_has_pneumo == 0).sum()
    pos_count = (train_has_pneumo == 1).sum()
    weights = np.where(train_has_pneumo == 1, neg_count / max(pos_count, 1), 1.0)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_files), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_files, val_files


def get_test_dataloader(data_dir, test_csv, batch_size=8, target_size=512,
                        num_workers=4, preprocessed=False):
    """加载测试集"""
    image_dir, mask_dir = _get_dirs(data_dir, target_size, preprocessed)

    df = pd.read_csv(test_csv)
    file_names = df['new_filename'].tolist()
    image_ids = df['ImageId'].tolist()

    valid = [(f, iid) for f, iid in zip(file_names, image_ids)
             if os.path.exists(os.path.join(image_dir, f))]
    file_names = [x[0] for x in valid]
    image_ids = [x[1] for x in valid]

    dataset = SIIMDataset(image_dir, mask_dir, file_names,
                         target_size=target_size, augment=False,
                         preprocessed=preprocessed)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)

    return loader, image_ids, file_names
