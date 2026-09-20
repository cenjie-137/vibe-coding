import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


CLASSES = ['invasive_epithelium', 'non_invasive_epithelium', 'necrosis', 'other']
CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BEETLEDataset(Dataset):
    """
    BEETLE 数据集加载

    数据目录结构：
    data/BEETLE/
    ├── train/
    │   ├── images/          # xxx.png
    │   └── masks/           # xxx.png (像素值: 0,1,2,3)
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/                # 公开的 170 张 ROI（用于生成提交）
        └── images/

    类别映射：
    0: invasive_epithelium
    1: non_invasive_epithelium
    2: necrosis
    3: other
    """

    def __init__(self, data_dir, split='train', patch_size=512, patch_per_img=4, augment=False):
        self.split = split
        self.patch_size = patch_size
        self.patch_per_img = patch_per_img
        self.augment = augment

        if split == 'test':
            img_dir = os.path.join(data_dir, 'test', 'images')
            self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.tif'))])
            self.mask_paths = None
        else:
            split_dir = 'train' if split == 'train' else 'val'
            img_dir = os.path.join(data_dir, split_dir, 'images')
            mask_dir = os.path.join(data_dir, split_dir, 'masks')

            self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.tif'))])
            self.mask_paths = []
            for p in self.image_paths:
                base = os.path.splitext(os.path.basename(p))[0]
                # 尝试多种命名方式
                for ext in ['.png', '.tif', '.jpg']:
                    mask_path = os.path.join(mask_dir, base + ext)
                    if os.path.exists(mask_path):
                        self.mask_paths.append(mask_path)
                        break
                else:
                    self.mask_paths.append(os.path.join(mask_dir, base + '.png'))

        print(f"[{split}] Loaded {len(self.image_paths)} images", flush=True)

    def __len__(self):
        if self.split == 'train':
            return len(self.image_paths) * self.patch_per_img
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.split == 'train':
            img_idx = idx // self.patch_per_img
        else:
            img_idx = idx

        image = Image.open(self.image_paths[img_idx]).convert('RGB')

        if self.split != 'test':
            mask = Image.open(self.mask_paths[img_idx]).convert('L')
        else:
            mask = None

        # 预处理：归一化
        img_tensor = TF.to_tensor(image)
        img_tensor = TF.normalize(img_tensor,
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

        if mask is not None:
            mask_np = np.array(mask, dtype=np.int64)
            # 确保类别值在 0-3 范围内
            mask_np = np.clip(mask_np, 0, 3)
            mask_tensor = torch.from_numpy(mask_np).long()
        else:
            mask_tensor = torch.zeros(image.height, image.width, dtype=torch.long)

        if self.split == 'train' and self.augment:
            # 随机 crop patch
            _, h, w = img_tensor.shape
            if h > self.patch_size and w > self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                img_tensor = img_tensor[:, top:top + self.patch_size, left:left + self.patch_size]
                mask_tensor = mask_tensor[top:top + self.patch_size, left:left + self.patch_size]

            # 随机翻转
            if random.random() < 0.5:
                img_tensor = TF.hflip(img_tensor)
                mask_tensor = TF.hflip(mask_tensor)
            if random.random() < 0.5:
                img_tensor = TF.vflip(img_tensor)
                mask_tensor = TF.vflip(mask_tensor)

            # 颜色增强（组织病理 H&E 染色的颜色变化）
            if random.random() < 0.3:
                img_tensor = TF.adjust_brightness(img_tensor, brightness_factor=random.uniform(0.8, 1.2))
            if random.random() < 0.3:
                img_tensor = TF.adjust_contrast(img_tensor, contrast_factor=random.uniform(0.8, 1.2))
            if random.random() < 0.3:
                img_tensor = TF.adjust_saturation(img_tensor, saturation_factor=random.uniform(0.8, 1.2))

        elif self.split == 'train' and not self.augment:
            # 无增强时也裁剪 patch（保证尺寸一致）
            _, h, w = img_tensor.shape
            if h > self.patch_size and w > self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                img_tensor = img_tensor[:, top:top + self.patch_size, left:left + self.patch_size]
                mask_tensor = mask_tensor[top:top + self.patch_size, left:left + self.patch_size]

        file_id = os.path.splitext(os.path.basename(self.image_paths[img_idx]))[0]
        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'file_id': file_id,
        }


def get_dataloader(data_dir, split='train', batch_size=4, patch_size=512,
                   patch_per_img=4, num_workers=0, augment=True, shuffle=None):
    dataset = BEETLEDataset(data_dir, split=split, patch_size=patch_size,
                            patch_per_img=patch_per_img, augment=augment)
    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == '__main__':
    data_dir = r"C:\Users\32201\Desktop\T\vibe coding\BEETLE\data\BEETLE"
    if not os.path.exists(data_dir):
        data_dir = '/root/autodl-tmp/medseg_project/data/BEETLE'

    loader = get_dataloader(data_dir, split='train', batch_size=2, augment=True)
    for batch in loader:
        print("Image shape:", batch['image'].shape)
        print("Mask shape:", batch['mask'].shape)
        print("Mask unique values:", torch.unique(batch['mask']))
        print("File id:", batch['file_id'])
        break
