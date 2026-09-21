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


class ResNet34Encoder(nn.Module):
    """ImageNet 预训练 ResNet34 编码器（DCAN 冠军用 VGG16，这里用同族的 CNN 骨干）。

    输出多尺度特征：x0(/2), s1(/4), s2(/8), s3(/16), s4(/32)，供 U-Net skip 连接。
    """

    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision
        try:
            weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            rn = torchvision.models.resnet34(weights=weights)
        except Exception as e:
            print(f"[WARN] ResNet34 ImageNet 权重加载失败，改为随机初始化：{e}", flush=True)
            rn = torchvision.models.resnet34(weights=None)

        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))   # /2, 64
        xp = self.maxpool(x0)                      # /4, 64
        s1 = self.layer1(xp)                       # /4, 64
        s2 = self.layer2(s1)                       # /8, 128
        s3 = self.layer3(s2)                       # /16, 256
        s4 = self.layer4(s3)                       # /32, 512
        return x0, s1, s2, s3, s4


class GlandCNNDCAN(nn.Module):
    """DCAN 式轮廓感知网络（GLaS 冠军配方，CNN 版）：

    ResNet34(ImageNet) 编码器 + U-Net skip 解码器 + seg/contour 双头 + 多级深监督。
    训练时额外返回 2 组低分辨率辅助输出（seg,contour），推理只返回主双头。
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained=pretrained)

        self.dec4 = _conv_block(512, 256)
        self.dec3 = _conv_block(256 + 256, 128)
        self.dec2 = _conv_block(128 + 128, 64)
        self.dec1 = _conv_block(64 + 64, 64)
        self.dec0 = _conv_block(64 + 64, 32)

        self.seg_head = nn.Conv2d(32, 1, 1)
        self.contour_head = nn.Conv2d(32, 1, 1)

        # DCAN 多级辅助监督：在 dec1(/4)、dec2(/8) 加 seg+contour 副头
        self.aux_seg1 = nn.Conv2d(64, 1, 1)
        self.aux_cont1 = nn.Conv2d(64, 1, 1)
        self.aux_seg2 = nn.Conv2d(64, 1, 1)
        self.aux_cont2 = nn.Conv2d(64, 1, 1)

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, x):
        size = x.shape[2:]
        x0, s1, s2, s3, s4 = self.encoder(x)

        d4 = self.dec4(s4)
        d3 = self.dec3(torch.cat([self._up(d4, s3.shape[2:]), s3], 1))
        d2 = self.dec2(torch.cat([self._up(d3, s2.shape[2:]), s2], 1))
        d1 = self.dec1(torch.cat([self._up(d2, s1.shape[2:]), s1], 1))
        d0 = self.dec0(torch.cat([self._up(d1, x0.shape[2:]), x0], 1))
        out = self._up(d0, size)

        seg = self.seg_head(out)
        contour = self.contour_head(out)

        if self.training:
            a1_seg = self.aux_seg1(self._up(d1, size))
            a1_cont = self.aux_cont1(self._up(d1, size))
            a2_seg = self.aux_seg2(self._up(d2, size))
            a2_cont = self.aux_cont2(self._up(d2, size))
            return seg, contour, [(a1_seg, a1_cont), (a2_seg, a2_cont)]
        return seg, contour