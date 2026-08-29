"""
Model B — Classification.

Grades a cropped, storm-centred satellite image on IMD's 7-tier cyclone
intensity scale (see src/config.py for the exact category boundaries).

This is a *hybrid fusion* model, which is the main technical differentiator
called out in the idea pitch: a plain image classifier only sees cloud
shape, but real intensity is also driven by physical quantities (sea-surface
temperature, wind shear, humidity) that ERA5 reanalysis provides directly.
We concatenate a small CNN's image embedding with those ERA5 scalar
features before the final classification head, so the model can learn to
use both cues — this is what gives it an edge on rare, high-impact
categories where cloud shape alone is ambiguous.

Uses a plain (non-pretrained-by-default) CNN backbone here so this file
runs anywhere with zero network access. When you train for real on
Colab/Kaggle (which has full internet), flip `pretrained=True` to swap in
ImageNet-pretrained ResNet18/EfficientNet-B0 weights via torchvision —
see the `backbone` argument below.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class SimpleCNNBackbone(nn.Module):
    """A small from-scratch CNN image encoder — no internet/pretrained weights needed."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 128 -> 64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 64 -> 32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                            # 32 -> 16
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                    # -> (B, 128, 1, 1)
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        feats = self.features(x).flatten(1)   # (B, 128)
        return self.proj(feats)                # (B, embed_dim)


def build_pretrained_backbone(name: str = "resnet18", embed_dim: int = 128, in_channels: int = 1):
    """
    Optional: swap in an ImageNet-pretrained torchvision backbone.
    Only call this in an environment with normal internet access
    (Colab/Kaggle/local) — it downloads weights from torchvision's hub.
    """
    import torchvision.models as tv_models

    if name == "resnet18":
        net = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Linear(net.fc.in_features, embed_dim)
        return net
    elif name == "efficientnet_b0":
        net = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        if in_channels != 3:
            net.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, embed_dim)
        return net
    else:
        raise ValueError(f"Unknown backbone: {name}")


class CycloneClassifier(nn.Module):
    """
    Hybrid image + physical-feature fusion classifier.

    forward(image, era5_features) -> logits over NUM_CATEGORIES (IMD scale).

    Args:
        num_categories: number of IMD intensity tiers (default from config).
        num_era5_features: how many scalar ERA5 features are fused in
            (sea-surface temp, wind shear, humidity, MSLP by default — see
            config.ERA5_FEATURES).
        image_backbone: nn.Module producing a (B, embed_dim) image embedding.
            Defaults to the from-scratch SimpleCNNBackbone; pass the result
            of build_pretrained_backbone(...) to use pretrained weights.
    """

    def __init__(
        self,
        num_categories: int = 8,
        num_era5_features: int = 4,
        embed_dim: int = 128,
        image_backbone: nn.Module | None = None,
        in_channels: int = 1,
    ):
        super().__init__()
        self.backbone = image_backbone or SimpleCNNBackbone(in_channels=in_channels, embed_dim=embed_dim)

        self.era5_encoder = nn.Sequential(
            nn.Linear(num_era5_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        fused_dim = embed_dim + 32
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_categories),
        )

    def forward(self, image: torch.Tensor, era5_features: torch.Tensor) -> torch.Tensor:
        img_embed = self.backbone(image)              # (B, embed_dim)
        era5_embed = self.era5_encoder(era5_features)  # (B, 32)
        fused = torch.cat([img_embed, era5_embed], dim=1)
        return self.head(fused)                         # (B, num_categories) logits


if __name__ == "__main__":
    from src.config import NUM_CATEGORIES, ERA5_FEATURES, CATEGORY_NAMES

    model = CycloneClassifier(
        num_categories=NUM_CATEGORIES,
        num_era5_features=len(ERA5_FEATURES),
    )
    dummy_image = torch.randn(4, 1, 128, 128)
    dummy_era5 = torch.randn(4, len(ERA5_FEATURES))

    logits = model(dummy_image, dummy_era5)
    pred_idx = logits.argmax(dim=1)

    print("CycloneClassifier output shape:", logits.shape)
    print("Predicted category indices:", pred_idx.tolist())
    print("Predicted category names:", [CATEGORY_NAMES[i] for i in pred_idx.tolist()])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
