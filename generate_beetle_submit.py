import os
import sys
import argparse
import traceback
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import zipfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets_beetle import get_dataloader, CLASSES, CLASS_IDS
from models import MultiResUNet


def get_paths():
    autodl_root = '/root/autodl-tmp'
    if os.path.exists(autodl_root):
        data_dir = os.path.join(autodl_root, 'medseg_project', 'data', 'BEETLE')
        output_dir = os.path.join(autodl_root, 'medseg_project', 'output_beetle')
    else:
        base = r"C:\Users\32201\Desktop\T\vibe coding\BEETLE"
        data_dir = os.path.join(base, 'data', 'BEETLE')
        output_dir = os.path.join(base, 'output')
    return data_dir, output_dir


def load_model(model_path, device, base_ch=64, attention='se'):
    attn = None if attention == 'none' else attention
    model = MultiResUNet(n_channels=3, n_classes=4, base_ch=base_ch,
                         attention=attn, dropout=0.1).to(device)
    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except Exception:
        # 可能缺少 bn_shortcut / 有不同 attention，宽松加载
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:5]}", flush=True)
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}", flush=True)
    model.eval()
    print(f"Model loaded: {model_path}", flush=True)
    return model


def sliding_window_inference(model, image_np, patch_size=512, overlap=0.25, device='cuda'):
    """
    滑动窗口推理，处理整张大图 WSI ROI
    image_np: (3, H, W) 归一化后的 tensor
    返回: (4, H, W) softmax 概率
    """
    C, H, W = image_np.shape
    stride = int(patch_size * (1 - overlap))
    pad_h = (stride - (H - patch_size) % stride) % stride
    pad_w = (stride - (W - patch_size) % stride) % stride
    image_padded = F.pad(image_np.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
    _, pH, pW = image_padded.shape

    count_map = torch.zeros(4, pH, pW, dtype=torch.float32, device=device)
    prob_sum = torch.zeros(4, pH, pW, dtype=torch.float32, device=device)

    ys = list(range(0, pH - patch_size + 1, stride))
    xs = list(range(0, pW - patch_size + 1, stride))

    with torch.no_grad(), torch.cuda.amp.autocast():
        for y in ys:
            for x in xs:
                patch = image_padded[:, y:y + patch_size, x:x + patch_size].unsqueeze(0).to(device)
                out = model(patch)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                prob = F.softmax(out, dim=1).squeeze(0)
                prob_sum[:, y:y + patch_size, x:x + patch_size] += prob
                count_map[:, y:y + patch_size, x:x + patch_size] += 1

    # 防止除零
    count_map = count_map.clamp_min(1)
    avg_prob = prob_sum / count_map
    avg_prob = avg_prob[:, :H, :W]
    return avg_prob


def tta_inference(model, image_tensor, patch_size=512, overlap=0.25, device='cuda'):
    """简单 TTA：原图 + 水平翻转 + 垂直翻转"""
    prob = sliding_window_inference(model, image_tensor, patch_size, overlap, device)

    image_h = torch.flip(image_tensor, dims=[2])
    prob_h = sliding_window_inference(model, image_h, patch_size, overlap, device)
    prob_h = torch.flip(prob_h, dims=[2])

    image_v = torch.flip(image_tensor, dims=[1])
    prob_v = sliding_window_inference(model, image_v, patch_size, overlap, device)
    prob_v = torch.flip(prob_v, dims=[1])

    prob_avg = (prob + prob_h + prob_v) / 3.0
    return prob_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--attention', type=str, default='se', choices=['se', 'cbam', 'none'])
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--overlap', type=float, default=0.25)
    parser.add_argument('--no_tta', action='store_true')
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()

    data_dir, output_dir = get_paths()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    model_path = os.path.join(output_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}", flush=True)
        return

    model = load_model(model_path, device, base_ch=args.base_ch, attention=args.attention)

    test_dir = os.path.join(data_dir, 'test', 'images')
    if not os.path.exists(test_dir):
        print(f"ERROR: Test images not found at {test_dir}", flush=True)
        return

    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.tif'))])
    print(f"Test samples: {len(test_files)}", flush=True)

    submit_dir = os.path.join(output_dir, f'submission{suffix}')
    os.makedirs(submit_dir, exist_ok=True)

    per_class_counts = np.zeros(4)

    import torchvision.transforms.functional as TF

    for idx, fname in enumerate(tqdm(test_files, desc="Inference")):
        try:
            img_path = os.path.join(test_dir, fname)
            image = Image.open(img_path).convert('RGB')
            img_tensor = TF.to_tensor(image)
            img_tensor = TF.normalize(img_tensor,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

            if args.no_tta:
                probs = sliding_window_inference(model, img_tensor, args.patch_size, args.overlap, device)
            else:
                probs = tta_inference(model, img_tensor, args.patch_size, args.overlap, device)

            pred_labels = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)

            # 统计类别分布
            for c in range(4):
                per_class_counts[c] += (pred_labels == c).sum()

            # 保存 PNG（像素值 0-3）
            file_id = os.path.splitext(fname)[0]
            save_path = os.path.join(submit_dir, f"{file_id}.png")
            Image.fromarray(pred_labels).save(save_path)

        except Exception as e:
            print(f"Error at {fname}: {e}", flush=True)
            traceback.print_exc()

    # 统计
    total = per_class_counts.sum()
    ratio = per_class_counts / total
    print(f"\nPrediction class distribution (total={total}):", flush=True)
    for i, name in enumerate(CLASSES):
        print(f"  {name}: {ratio[i] * 100:.2f}%", flush=True)

    # 打包
    zip_path = os.path.join(output_dir, f'submission{suffix}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in test_files:
            file_id = os.path.splitext(fname)[0]
            png_path = os.path.join(submit_dir, f"{file_id}.png")
            zf.write(png_path, os.path.basename(png_path))

    print(f"\nSubmission ready: {zip_path}", flush=True)


if __name__ == '__main__':
    main()
