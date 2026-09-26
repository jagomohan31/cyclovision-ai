"""
Model A â€” Detection / Identification.

A compact U-Net that takes a full (or tiled) INSAT infrared/water-vapour
frame and outputs a pixel-wise probability mask of "this pixel belongs to
the cyclone's cloud system", plus the storm's eye can be located as the
argmax / centroid of the mask. This is the first stage of the pipeline:
it answers "is there a cyclone here, and where exactly."

Kept intentionally small (4 encoder/decoder levels, few channels) so it
trains fast on a single Colab/Kaggle GPU. Widen `base_channels` once you
have a full dataset and want more capacity.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x2 â€” the basic U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CycloneUNet(nn.Module):
    """
    U-Net for cyclone cloud-system segmentation.

    Args:
        in_channels: number of input satellite channels (e.g. 1 for IR only,
            3 for IR + water-vapour + visible stacked).
        base_channels: channel width of the first encoder level; doubles at
            each downsampling step.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(c * 8, c * 16)

        # Decoder (transpose conv upsampling + skip connections)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        # 1x1 conv to a single-channel cyclone-probability mask
        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W), H and W must be divisible by 16
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        mask_logits = self.out_conv(d1)  # (B, 1, H, W) â€” apply sigmoid outside for BCEWithLogitsLoss
        return mask_logits


def locate_eye_from_mask(mask_prob: torch.Tensor) -> torch.Tensor:
    """
    Given a (B, 1, H, W) probability mask (post-sigmoid), return the
    intensity-weighted centroid (row, col) per batch item as a (B, 2) tensor.
    This is the simple, dependable way to turn a segmentation mask into a
    single storm-center coordinate for the classification/prediction stages.
    """
    b, _, h, w = mask_prob.shape
    device = mask_prob.device
    rows = torch.arange(h, device=device).view(1, h, 1).float()
    cols = torch.arange(w, device=device).view(1, 1, w).float()

    weights = mask_prob.squeeze(1)  # (B, H, W)
    total = weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    centroid_row = (weights * rows).sum(dim=(1, 2), keepdim=True) / total
    centroid_col = (weights * cols).sum(dim=(1, 2), keepdim=True) / total

    return torch.cat([centroid_row.view(b, 1), centroid_col.view(b, 1)], dim=1)


if __name__ == "__main__":
    # Quick self-test: confirm shapes flow correctly end-to-end.
    model = CycloneUNet(in_channels=1, base_channels=16)
    dummy = torch.randn(2, 1, 128, 128)
    logits = model(dummy)
    probs = torch.sigmoid(logits)
    eye = locate_eye_from_mask(probs)
    print("CycloneUNet output shape:", logits.shape)
    print("Located eye coordinates (row, col):", eye.shape, eye)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

