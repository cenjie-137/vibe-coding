"""SIIM-ACR 离线预处理：CLAHE + 256x256 缩放，预处理后大幅加速训练"""
import os
import cv2
import numpy as np
from PIL import Image
from multiprocessing import Pool
import time


def process_single_image(args):
    input_path, output_path, target_size = args
    try:
        img = Image.open(input_path).convert('L')
        img = img.resize((target_size, target_size), Image.BILINEAR)
        img_uint8 = np.array(img, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_uint8)
        Image.fromarray(img_clahe).save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def process_single_mask(args):
    input_path, output_path, target_size = args
    try:
        img = Image.open(input_path).convert('L')
        img = img.resize((target_size, target_size), Image.NEAREST)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    target_size = 256  # 降采样到 256x256（64x 像素减少，64x 加速）
    n_workers = 8

    autodl_root = '/root/autodl-tmp'
    if os.path.exists(autodl_root):
        data_dir = os.path.join(autodl_root, 'medseg_project', 'data', 'SIIM', 'siim-acr-pneumothorax')
    else:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'medseg_project', 'data', 'SIIM', 'siim-acr-pneumothorax')

    input_img_dir = os.path.join(data_dir, 'png_images')
    input_mask_dir = os.path.join(data_dir, 'png_masks')
    output_img_dir = os.path.join(data_dir, f'png_images_{target_size}')
    output_mask_dir = os.path.join(data_dir, f'png_masks_{target_size}')

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_img_dir) if f.endswith('.png')]
    print(f"Processing {len(image_files)} images to {target_size}x{target_size}...")

    # 准备参数列表
    img_args = [(os.path.join(input_img_dir, f),
                 os.path.join(output_img_dir, f),
                 target_size) for f in image_files]
    mask_args = [(os.path.join(input_mask_dir, f),
                  os.path.join(output_mask_dir, f),
                  target_size) for f in image_files]

    start = time.time()

    # 并行处理图片（CLAHE + resize）
    with Pool(n_workers) as pool:
        results = pool.map(process_single_image, img_args)
    n_done = sum(results)
    print(f"Images: {n_done}/{len(image_files)} done in {time.time()-start:.1f}s")

    # 并行处理掩码（仅 resize）
    with Pool(n_workers) as pool:
        results = pool.map(process_single_mask, mask_args)
    n_done = sum(results)
    print(f"Masks: {n_done}/{len(image_files)} done")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output directories:")
    print(f"  Images: {output_img_dir}")
    print(f"  Masks:  {output_mask_dir}")


if __name__ == '__main__':
    main()
