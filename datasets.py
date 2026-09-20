import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def apply_clahe(img):
    """CLAHE 对比度增强，提高血管可见度"""
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced.astype(np.float32) / 255.0


def elastic_transform(image, label, mask, alpha=1000, sigma=30, random_state=None):
    """弹性形变增强，模拟眼底图像形变"""
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape[:2]
    dx = random_state.rand(*shape) * 2 - 1
    dy = random_state.rand(*shape) * 2 - 1
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
    label = cv2.remap(label, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
    mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
    return image, label, mask


class RandomAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label, mask):
        if np.random.random() < self.p:
            img = np.fliplr(img).copy()
            label = np.fliplr(label).copy()
            mask = np.fliplr(mask).copy()

        if np.random.random() < self.p:
            img = np.flipud(img).copy()
            label = np.flipud(label).copy()
            mask = np.flipud(mask).copy()

        if np.random.random() < self.p:
            img, label, mask = elastic_transform(img, label, mask)

        if np.random.random() < self.p:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)

        if np.random.random() < self.p:
            factor = np.random.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 1)

        if np.random.random() < self.p:
            noise = np.random.normal(0, 0.01, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        return img, label, mask

class DRIVEDataset(Dataset):
    def __init__(self, data_dir, split='train', val_split=0.2, augment=True, fold=None, total_folds=5, use_all=False):
        self.data_dir = data_dir
        self.split = split
        self.val_split = val_split
        self.augment = augment and (split == 'train')
        self.aug = RandomAugmentation(p=0.5)
        
        all_file_ids = [f"{i:02d}" for i in range(21, 41)]
        
        if fold is not None:
            fold_size = len(all_file_ids) // total_folds
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size
            val_ids = all_file_ids[val_start:val_end]
            train_ids = [id for id in all_file_ids if id not in val_ids]
            
            if split == 'train':
                self.file_ids = train_ids
            elif split == 'val':
                self.file_ids = val_ids
        elif use_all and split == 'train':
            self.file_ids = all_file_ids
        else:
            if split == 'train':
                split_idx = int(len(all_file_ids) * (1 - val_split))
                self.file_ids = all_file_ids[:split_idx]
            elif split == 'val':
                split_idx = int(len(all_file_ids) * (1 - val_split))
                self.file_ids = all_file_ids[split_idx:]
        
        if split in ['train', 'val']:
            self.image_dir = os.path.join(data_dir, 'training', 'images')
            self.label_dir = os.path.join(data_dir, 'training', '1st_manual')
            self.mask_dir = os.path.join(data_dir, 'training', 'mask')
        else:
            self.image_dir = os.path.join(data_dir, 'test', 'images')
            self.label_dir = None
            self.mask_dir = os.path.join(data_dir, 'test', 'mask')
            self.file_ids = [f"{i:02d}" for i in range(1, 21)]
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        if self.split in ['train', 'val']:
            img_path = os.path.join(self.image_dir, f"{file_id}_training.tif")
            mask_path = os.path.join(self.mask_dir, f"{file_id}_training_mask.gif")
            label_path = os.path.join(self.label_dir, f"{file_id}_manual1.gif")
        else:
            img_path = os.path.join(self.image_dir, f"{file_id}_test.tif")
            mask_path = os.path.join(self.mask_dir, f"{file_id}_test_mask.gif")
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # CLAHE 对比度增强（训练集和验证集都应用，属于预处理）
        img = apply_clahe(img)
        
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 0).astype(np.float32)
        
        if self.split in ['train', 'val']:
            label = np.array(Image.open(label_path).convert('L'))
            label = (label > 0).astype(np.float32)
        else:
            label = np.zeros_like(mask)
        
        if self.augment:
            img, label, mask = self.aug(img, label, mask)
        
        img = np.transpose(img, (2, 0, 1))
        label = np.expand_dims(label, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return {
            'image': torch.from_numpy(img),
            'label': torch.from_numpy(label),
            'mask': torch.from_numpy(mask),
            'file_id': file_id
        }

def get_dataloader(data_dir, split='train', batch_size=4, num_workers=0, augment=True, fold=None, total_folds=5, use_all=False):
    dataset = DRIVEDataset(data_dir, split=split, augment=augment, fold=fold, total_folds=total_folds, use_all=use_all)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), 
                      num_workers=num_workers, pin_memory=True)