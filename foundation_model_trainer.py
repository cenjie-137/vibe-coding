"""
Foundation Model Trainer for CamylaNet.

Integrates pathology foundation models (UNI, UNI2-h, CONCH) as encoders
into the CamylaNet/nnU-Net v2 training framework.

Architecture: ViT encoder (frozen or fine-tuned) + U-Net decoder

Models:
  - UNI: ViT-L/16, trained on 100M+ patches (HuggingFace: mahmoodlab/UNI)
  - UNI2-h: ViT-H/14, newer and larger (HuggingFace: mahmoodlab/UNI2-h)
  - CONCH: ViT-B/16, vision-language model (HuggingFace: mahmoodlab/conch)
  - Fallback: ImageNet ViT-L/16 if HF access not yet granted

Usage in experiment_runner:
  from foundation_model_trainer import FoundationModelTrainer
  camylanet.training_network_1epoch(
      dataset_id=101, configuration='2d',
      trainer_class=FoundationModelTrainer,
      plans_identifier=plans,
      exp_name='smoke_UNI'
  )

Standalone test:
  python foundation_model_trainer.py --test
"""

import os
import sys
import argparse
from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# CamylaNet base trainer
from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision
)


# ==================== Foundation Model Encoder ====================

class FoundationViTEncoder(nn.Module):
    """ViT encoder wrapper for pathology foundation models.

    Loads a pretrained ViT from timm/HuggingFace and extracts
    patch-level features for segmentation decoding.
    """

    def __init__(self, model_source="UNI", freeze_encoder=False):
        super().__init__()
        self.model_source = model_source
        self.freeze_encoder = freeze_encoder

        self.encoder = self._load_encoder(model_source)

        self.embed_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.num_features = self.encoder.num_features

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"  Encoder frozen ({model_source})")
        else:
            print(f"  Encoder trainable ({model_source})")

    def _load_encoder(self, model_source):
        """Load foundation model weights via timm."""
        import timm

        source_config = {
            "UNI": {
                "model_name": "hf-hub:MahmoodLab/UNI",
                "init_values": 1e-5,
            },
            "UNI2-h": {
                "model_name": "hf-hub:MahmoodLab/UNI2-h",
                "init_values": 1e-5,
            },
            "CONCH": {
                "model_name": "hf-hub:MahmoodLab/conch",
                "init_values": 1e-5,
            },
            "imagenet_vitl16": {
                "model_name": "vit_large_patch16_224",
                "init_values": None,
            },
        }

        cfg = source_config.get(model_source, source_config["imagenet_vitl16"])

        try:
            if cfg["init_values"] is not None:
                model = timm.create_model(
                    cfg["model_name"],
                    pretrained=True,
                    init_values=cfg["init_values"],
                    num_classes=0,
                )
            else:
                model = timm.create_model(
                    cfg["model_name"],
                    pretrained=True,
                    num_classes=0,
                )
            print(f"  Loaded {model_source} from HuggingFace")
        except Exception as e:
            print(f"  Failed to load {model_source}: {e}")
            print(f"  Falling back to ImageNet ViT-L/16")
            model = timm.create_model(
                "vit_large_patch16_224",
                pretrained=True,
                num_classes=0,
            )
            self.model_source = "imagenet_vitl16"
            self.embed_dim = model.embed_dim

        return model

    def forward_features(self, x):
        """Extract spatial patch features.

        Args:
            x: B, C, H, W (H, W must be divisible by patch_size)

        Returns:
            feat: B, embed_dim, H/patch_size, W/patch_size
        """
        B, C, H, W = x.shape
        ps = self.patch_size

        if H % ps != 0 or W % ps != 0:
            pad_h = (ps - H % ps) % ps
            pad_w = (ps - W % ps) % ps
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H, W = x.shape[2:]

        x = self.encoder.forward_features(x)

        if x.ndim == 3:
            n_patches_h = H // ps
            n_patches_w = W // ps
            if x.shape[1] == n_patches_h * n_patches_w + 1:
                x = x[:, 1:]
            x = x.transpose(1, 2).reshape(
                B, self.embed_dim, n_patches_h, n_patches_w
            )
        elif x.ndim == 4:
            pass

        return x


# ==================== U-Net Decoder ====================

class DecoderBlock(nn.Module):
    """U-Net decoder block: upsample + double conv."""

    def __init__(self, in_ch, out_ch, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class FoundationUNet(nn.Module):
    """ViT encoder + progressive U-Net decoder for segmentation.

    The decoder progressively upsamples the ViT bottleneck features
    to the original input resolution.
    """

    def __init__(self, num_input_channels=3, num_classes=6,
                 model_source="UNI", freeze_encoder=False):
        super().__init__()
        self.encoder = FoundationViTEncoder(
            model_source=model_source,
            freeze_encoder=freeze_encoder
        )

        embed_dim = self.encoder.embed_dim

        self.decoder = nn.ModuleList([
            DecoderBlock(embed_dim, 512),       # bottleneck -> 2x up
            DecoderBlock(512, 256),             # -> 4x up
            DecoderBlock(256, 128),             # -> 8x up
            DecoderBlock(128, 64),              # -> 16x up (to original res)
        ])

        self.final_conv = nn.Conv2d(64, num_classes, 1)

        patch_size = self.encoder.patch_size
        self.upsample_factor = patch_size  # 16 for ViT-L/16

    def forward(self, x):
        input_size = x.shape[2:]

        feat = self.encoder.forward_features(x)

        for decoder_block in self.decoder:
            feat = decoder_block(feat)

        if feat.shape[2:] != input_size:
            feat = F.interpolate(
                feat, size=input_size,
                mode='bilinear', align_corners=False
            )

        return self.final_conv(feat)


# ==================== CamylaNet Trainer ====================

class FoundationModelTrainer(nnUNetTrainerNoDeepSupervision):
    """CamylaNet trainer using pathology foundation model encoder.

    Inherits from nnUNetTrainerNoDeepSupervision (NOT nnUNetTrainer).
    Only overrides build_network_architecture.

    Default: UNI encoder + U-Net decoder, encoder trainable.
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        model_source = arch_init_kwargs.get('model_source', 'UNI')
        freeze_encoder = arch_init_kwargs.get('freeze_encoder', False)

        network = FoundationUNet(
            num_input_channels=num_input_channels,
            num_classes=num_output_channels,
            model_source=model_source,
            freeze_encoder=freeze_encoder,
        )

        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(
            p.numel() for p in network.parameters() if p.requires_grad
        )
        print(f"FoundationModelTrainer:")
        print(f"  Model source: {model_source}")
        print(f"  Encoder: {'frozen' if freeze_encoder else 'trainable'}")
        print(f"  Total params: {total_params/1e6:.1f}M")
        print(f"  Trainable params: {trainable_params/1e6:.1f}M")

        return network


class FoundationModelFrozenTrainer(nnUNetTrainerNoDeepSupervision):
    """Variant with frozen encoder (only train decoder).

    Use for quick experiments where encoder fine-tuning is not needed.
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        network = FoundationUNet(
            num_input_channels=num_input_channels,
            num_classes=num_output_channels,
            model_source=arch_init_kwargs.get('model_source', 'UNI'),
            freeze_encoder=True,
        )

        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(
            p.numel() for p in network.parameters() if p.requires_grad
        )
        print(f"FoundationModelFrozenTrainer:")
        print(f"  Total params: {total_params/1e6:.1f}M")
        print(f"  Trainable (decoder only): {trainable_params/1e6:.1f}M")

        return network


# ==================== Standalone Test ====================

def test_model():
    """Quick test: load model, forward pass on dummy input."""
    print("=" * 50)
    print("Testing FoundationUNet")
    print("=" * 50)

    model = FoundationUNet(
        num_input_channels=3,
        num_classes=6,
        model_source="UNI",
        freeze_encoder=False,
    )

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == (1, 6, 512, 512), \
        f"Expected (1, 6, 512, 512), got {out.shape}"
    print("  PASSED: output shape matches input resolution")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total/1e6:.1f}M")
    print(f"  Trainable: {trainable/1e6:.1f}M")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                        help='Run standalone model test')
    parser.add_argument('--model_source', type=str, default='UNI',
                        choices=['UNI', 'UNI2-h', 'CONCH', 'imagenet_vitl16'],
                        help='Foundation model to use')
    args = parser.parse_args()

    if args.test:
        test_model()
