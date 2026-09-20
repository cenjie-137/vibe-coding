import os
import sys
import torch
import numpy as np
from PIL import Image
import zipfile
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataloader
from models import MultiResUNet

def get_autodl_paths():
    autodl_root = '/root/autodl-tmp'
    if os.path.exists(autodl_root):
        data_dir = os.path.join(autodl_root, 'medseg_project', 'data', 'DRIVE')
        output_dir = os.path.join(autodl_root, 'medseg_project', 'output')
    else:
        data_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\data\DRIVE"
        output_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\output"
    return data_dir, output_dir


def main():
    data_dir, output_dir = get_autodl_paths()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # 加载模型
    model = MultiResUNet(n_channels=3, n_classes=1, base_ch=64).to(device)
    model_path = os.path.join(output_dir, 'fold_0', 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(output_dir, 'best_model.pth')
    print(f"Loading model: {model_path}", flush=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully", flush=True)

    test_loader = get_dataloader(data_dir, split='test', batch_size=1, num_workers=0, augment=False)
    print(f"Test samples: {len(test_loader.dataset)}", flush=True)

    # 创建提交目录
    submit_dir = os.path.join(output_dir, 'submission')
    os.makedirs(submit_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            try:
                images = batch['image'].to(device)
                masks = batch['mask'].cpu().numpy()
                file_id = batch['file_id'][0] if isinstance(batch['file_id'], (list, tuple)) else batch['file_id']

                # TTA 预测
                outputs = model(images)
                images_h = torch.flip(images, dims=[3])
                outputs_h = model(images_h)
                outputs_h = torch.flip(outputs_h, dims=[3])
                images_v = torch.flip(images, dims=[2])
                outputs_v = model(images_v)
                outputs_v = torch.flip(outputs_v, dims=[2])
                outputs = (outputs + outputs_h + outputs_v) / 3.0

                pred_prob = torch.sigmoid(outputs).cpu().numpy()[0, 0]
                mask = masks[0, 0]

                # 二值化
                pred_bin = (pred_prob > 0.5).astype(np.uint8) * 255

                # 只保留 mask 内区域
                pred_bin = pred_bin * (mask > 0).astype(np.uint8)

                # 保存
                img_num = int(file_id)
                save_path = os.path.join(submit_dir, f"{img_num}.png")
                Image.fromarray(pred_bin).save(save_path)
                print(f"[{idx+1}/20] Saved: {save_path}", flush=True)

            except Exception as e:
                print(f"Error at idx {idx}: {e}", flush=True)
                traceback.print_exc()

    # 打包为 zip
    zip_path = os.path.join(output_dir, 'submission.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 21):
            png_path = os.path.join(submit_dir, f"{i}.png")
            zf.write(png_path, f"{i}.png")

    print(f"\nSubmission ready: {zip_path}", flush=True)


if __name__ == '__main__':
    main()
