import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== MultiResUNet ====================
# Ibtehaz, N. & Rahman, M.S. (2020). MultiResUNet: Rethinking the U-Net
# Architecture for Multimodal Biomedical Image Segmentation. Neural Networks.

class MultiResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, alpha=1.67):
        super().__init__()
        w = int(out_ch * alpha)
        self.conv_3x3 = nn.Conv2d(in_ch, w, 3, padding=1)
        self.bn_3x3 = nn.BatchNorm2d(w)
        self.conv_5x5 = nn.Conv2d(w, w, 3, padding=1)
        self.bn_5x5 = nn.BatchNorm2d(w)
        self.conv_7x7 = nn.Conv2d(w, w, 3, padding=1)
        self.bn_7x7 = nn.BatchNorm2d(w)
        self.conv_fuse = nn.Conv2d(w * 3, out_ch, 1)
        self.bn_fuse = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        self.bn_shortcut = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.bn_3x3(self.conv_3x3(x)))
        x2 = self.relu(self.bn_5x5(self.conv_5x5(x1)))
        x3 = self.relu(self.bn_7x7(self.conv_7x7(x2)))
        multi = torch.cat([x1, x2, x3], dim=1)
        multi = self.bn_fuse(self.conv_fuse(multi))
        shortcut = self.bn_shortcut(self.shortcut(x))
        out = self.relu(multi + shortcut)
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        ca = self.channel_attn(x).view(b, c, 1, 1)
        x = x * ca.expand_as(x)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        x = x * sa.expand_as(x)

        return x


class ResPath(nn.Module):
    """替代 skip connection，减少语义差距"""
    def __init__(self, in_ch, out_ch, length=4, attention='se'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(length):
            self.convs.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            self.bns.append(nn.BatchNorm2d(out_ch))
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        self.bn_shortcut = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if attention == 'cbam':
            self.attn = CBAMBlock(out_ch)
        elif attention == 'se':
            self.attn = SEBlock(out_ch)
        else:
            self.attn = None

    def forward(self, x):
        shortcut = self.bn_shortcut(self.shortcut(x))
        out = x
        for conv, bn in zip(self.convs, self.bns):
            out = self.relu(bn(conv(out)))
        out = out + shortcut
        if self.attn is not None:
            out = self.attn(out)
        return out


class MultiResUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_ch=64, attention='se'):
        super().__init__()
        self.enc1 = MultiResBlock(n_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = MultiResBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = MultiResBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = MultiResBlock(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.center = MultiResBlock(base_ch * 8, base_ch * 16)
        self.dropout = nn.Dropout2d(0.1)

        self.respath1 = ResPath(base_ch, base_ch, length=4, attention=attention)
        self.respath2 = ResPath(base_ch * 2, base_ch * 2, length=3, attention=attention)
        self.respath3 = ResPath(base_ch * 4, base_ch * 4, length=2, attention=attention)
        self.respath4 = ResPath(base_ch * 8, base_ch * 8, length=1, attention=attention)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = MultiResBlock(base_ch * 16 + base_ch * 8, base_ch * 8)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = MultiResBlock(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = MultiResBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = MultiResBlock(base_ch * 2 + base_ch, base_ch)

        self.outc = nn.Conv2d(base_ch, n_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.dropout(self.center(self.pool4(x4)))

        x = self.up4(x5)
        x = self._align_and_concat(x, self.respath4(x4))
        x = self.dec4(x)

        x = self.up3(x)
        x = self._align_and_concat(x, self.respath3(x3))
        x = self.dec3(x)

        x = self.up2(x)
        x = self._align_and_concat(x, self.respath2(x2))
        x = self.dec2(x)

        x = self.up1(x)
        x = self._align_and_concat(x, self.respath1(x1))
        x = self.dec1(x)

        return self.outc(x)

    def _align_and_concat(self, x, skip):
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return torch.cat([skip, x], dim=1)


# ==================== Foundation Model UNet (UNI) ====================

class FoundationViTEncoder(nn.Module):
    """ViT encoder wrapper for pathology foundation models (UNI/CONCH).

    Loads pretrained ViT via timm and extracts spatial patch features.
    """

    def __init__(self, model_source="UNI", freeze_encoder=False):
        super().__init__()
        self.model_source = model_source
        self.freeze_encoder = freeze_encoder

        os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

        import timm
        source_map = {
            "UNI": "hf-hub:MahmoodLab/UNI",
            "CONCH": "hf-hub:MahmoodLab/conch",
            "imagenet": "vit_large_patch16_224",
        }
        model_name = source_map.get(model_source, source_map["imagenet"])

        try:
            if model_source in ("UNI", "CONCH"):
                self.encoder = timm.create_model(
                    model_name, pretrained=True, init_values=1e-5, num_classes=0,
                    dynamic_img_size=True
                )
            else:
                self.encoder = timm.create_model(
                    model_name, pretrained=True, num_classes=0,
                    dynamic_img_size=True
                )
            print(f"Foundation encoder loaded: {model_source}", flush=True)
        except Exception as e:
            print(f"Failed to load {model_source}: {e}", flush=True)
            print("Falling back to ImageNet ViT-L/16", flush=True)
            self.encoder = timm.create_model(
                "vit_large_patch16_224", pretrained=True, num_classes=0,
                dynamic_img_size=True
            )
            self.model_source = "imagenet"

        self.embed_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_embed.patch_size[0]

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print(f"Encoder frozen ({model_source})", flush=True)


    def forward_features(self, x):
        """Extract spatial patch features: (B, C, H/ps, W/ps)."""
        B, C, H, W = x.shape
        ps = self.patch_size

        if H % ps != 0 or W % ps != 0:
            pad_h = (ps - H % ps) % ps
            pad_w = (ps - W % ps) % ps
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H, W = x.shape[2:]

        x = self.encoder.forward_features(x)

        if x.ndim == 3:
            nh, nw = H // ps, W // ps
            if x.shape[1] == nh * nw + 1:
                x = x[:, 1:]
            x = x.transpose(1, 2).reshape(B, self.embed_dim, nh, nw)
        return x


class FoundationUNet(nn.Module):
    """ViT foundation encoder + progressive U-Net decoder.

    ViT outputs at H/16 x W/16; decoder upsamples 4x (16x total)
    to recover the input resolution.
    """

    def __init__(self, num_input_channels=3, num_classes=6,
                 model_source="UNI", freeze_encoder=False):
        super().__init__()
        self.encoder = FoundationViTEncoder(
            model_source=model_source, freeze_encoder=freeze_encoder
        )
        D = self.encoder.embed_dim

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = conv_block(D, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = conv_block(512, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = conv_block(256, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = conv_block(128, 64)

        self.outc = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        size = x.shape[2:]
        x = self.encoder.forward_features(x)
        x = self.dec1(self.up1(x))
        x = self.dec2(self.up2(x))
        x = self.dec3(self.up3(x))
        x = self.dec4(self.up4(x))
        if x.shape[2:] != size:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return self.outc(x)


class MultiResUNetDS(MultiResUNet):
    """MultiResUNet + Deep Supervision（深监督）
    训练时从 decoder 中间层输出辅助分割图，帮助梯度传播。
    推理时只返回主输出，不增加计算成本。"""
    def __init__(self, n_channels=3, n_classes=1, base_ch=64, attention='se'):
        super().__init__(n_channels, n_classes, base_ch, attention)
        self.aux4 = nn.Conv2d(base_ch * 8, n_classes, 1)
        self.aux3 = nn.Conv2d(base_ch * 4, n_classes, 1)
        self.aux2 = nn.Conv2d(base_ch * 2, n_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.dropout(self.center(self.pool4(x4)))

        d4 = self.up4(x5)
        d4 = self._align_and_concat(d4, self.respath4(x4))
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._align_and_concat(d3, self.respath3(x3))
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._align_and_concat(d2, self.respath2(x2))
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._align_and_concat(d1, self.respath1(x1))
        d1 = self.dec1(d1)

        out = self.outc(d1)

        if self.training:
            aux4 = F.interpolate(self.aux4(d4), size=x.shape[2:], mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.aux3(d3), size=x.shape[2:], mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux2(d2), size=x.shape[2:], mode='bilinear', align_corners=False)
            return out, aux2, aux3, aux4
        return out


# ==================== 基础 UNet ====================

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        self.inc = conv_block(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = conv_block(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = conv_block(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = conv_block(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = conv_block(512, 512)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv1 = conv_block(1024, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv2 = conv_block(512, 128)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv3 = conv_block(256, 64)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv4 = conv_block(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, 1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))
        x5 = self.conv4(self.down4(x4))
        
        x = self.up1(x5)
        x = self._align_and_concat(x, x4)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = self._align_and_concat(x, x3)
        x = self.up_conv2(x)
        
        x = self.up3(x)
        x = self._align_and_concat(x, x2)
        x = self.up_conv3(x)
        
        x = self.up4(x)
        x = self._align_and_concat(x, x1)
        x = self.up_conv4(x)
        
        logits = self.outc(x)
        return logits
    
    def _align_and_concat(self, x, skip):
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return torch.cat([skip, x], dim=1)