import os
import sys
import torch
import numpy as np
import cv2
import zipfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import DRIVEDataset, apply_clahe
from models import MultiResUNet


def get_autodl_paths():
    autodl_root = '/root/autodl-tmp'
    local_root = os.path.dirname(os.path.abspath(__file__))
    
    if os.path.exists(autodl_root):
        project_dir = os.path.join(autodl_root, 'medseg_project')
        data_dir = os.path.join(project_dir, 'data', 'DRIVE')
        output_dir = os.path.join(project_dir, 'output')
    else:
        project_dir = os.path.join(os.path.dirname(local_root), 'medseg_project')
        data_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\data\DRIVE"
        output_dir = r"C:\Users\32201\Desktop\T\MOME\medseg_project\output"
    
    return data_dir, output_dir


def main():
    data_dir, output_dir = get_autodl_paths()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    model = MultiResUNet(n_channels=3, n_classes=1, base_ch=64).to(device)
    
    model_path = os.path.join(output_dir, 'best_model_full.pth')
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}", flush=True)
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}", flush=True)
    
    test_image_dir = os.path.join(data_dir, 'test', 'images')
    test_mask_dir = os.path.join(data_dir, 'test', 'mask')
    test_file_ids = [f"{i:02d}" for i in range(1, 21)]
    
    submission_dir = os.path.join(output_dir, 'submission_full')
    os.makedirs(submission_dir, exist_ok=True)
    
    print(f"Generating predictions for {len(test_file_ids)} test images...", flush=True)
    
    for file_id in test_file_ids:
        img_path = os.path.join(test_image_dir, f"{file_id}_test.tif")
        mask_path = os.path.join(test_mask_dir, f"{file_id}_test_mask.gif")
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        img = apply_clahe(img)
        
        from PIL import Image
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 0).astype(np.float32)
        
        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            
            outputs_h = model(torch.flip(img_tensor, dims=[3]))
            outputs_h = torch.flip(outputs_h, dims=[3])
            
            outputs_v = model(torch.flip(img_tensor, dims=[2]))
            outputs_v = torch.flip(outputs_v, dims=[2])
            
            outputs = (outputs + outputs_h + outputs_v) / 3.0
            
            pred_prob = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
        pred_bin = (pred_prob > 0.5).astype(np.uint8) * 255
        pred_bin = pred_bin * (mask > 0).astype(np.uint8)
        
        save_path = os.path.join(submission_dir, f"{file_id}.png")
        cv2.imwrite(save_path, pred_bin)
        print(f"  Saved: {save_path}", flush=True)
    
    zip_path = os.path.join(output_dir, 'submission_full.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for i in range(1, 21):
            png_path = os.path.join(submission_dir, f"{i:02d}.png")
            zipf.write(png_path, f"{i}.png")
    
    print(f"\nSubmission zip saved to: {zip_path}", flush=True)
    print(f"Total files: {len(test_file_ids)}")
    print("Ready to submit to DRIVE challenge!")


if __name__ == '__main__':
    main()
