"""models_glas.py — GlandUNet：UNI 病理编码器 + U-Net 解码器 + 双头（分割 + 轮廓）

DCAN 式轮廓感知网络（GLaS 2015 官方冠军策略）：
  seg head      -> 腺体前景（二值）
  contour head  -> 腺体边界（用于拆分粘连腺体，对象级评估关键）

注：解码头为 v1 简单双卷积块（Conv-BN-ReLU ×2），与 v1 权重
    output_glas/best_fold_*.pth 一一对应，可 strict=True 加载。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import FoundationViTEncoder


def _conv_block(in_ch, out_ch):
    """v1 简单解码块：Conv-BN-ReLU ×2（与 FoundationUNet 同构）。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class GlandUNet(nn.Module):
    def __init__(self, model_source="UNI", freeze_encoder=False):
        super().__init__()
        self.encoder = FoundationViTEncoder(
            model_source=model_source, freeze_encoder=freeze_encoder
        )
        D = self.encoder.embed_dim

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = _conv_block(D, 512)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = _conv_block(512, 256)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = _conv_block(256, 128)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = _conv_block(128, 64)

        self.seg_head = nn.Conv2d(64, 1, 1)
        self.contour_head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        x = self.encoder.forward_features(x)
        x = self.dec1(self.up1(x))
        x = self.dec2(self.up2(x))
        x = self.dec3(self.up3(x))
        x = self.dec4(self.up4(x))
        if x.shape[2:] != size:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        seg = self.seg_head(x)
        contour = self.contour_head(x)
        return seg, contour