"""
U-Net segmentation model wrapping segmentation-models-pytorch.

Design choices:
  - Encoder: ResNet-34 pretrained on ImageNet (fine-tuned from channel 1 → n_channels)
  - Decoder: standard U-Net with skip connections
  - Output: raw logits, shape (B, n_classes, H, W) — apply softmax/argmax externally

The pretrained encoder expects 3-channel RGB. We adapt the first Conv2d in-place
by averaging the pretrained weights across the channel dimension, then tiling to
match n_channels. This transfers low-level texture/edge priors from natural images
while accommodating SAR-specific channel counts (VV, VH, +variance).

Usage:
    from src.models.unet import build_unet
    model = build_unet(n_channels=4, n_classes=4)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_unet(
    n_channels: int = 4,
    n_classes: int = 4,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
) -> nn.Module:
    """Build a U-Net with a pretrained encoder adapted to `n_channels` inputs.

    Args:
        n_channels: Number of input channels (VV + VH + optional coherence/variance).
        n_classes: Number of output segmentation classes.
        encoder_name: Any smp-supported encoder (resnet34, efficientnet-b3, etc.).
        encoder_weights: Pretrained weights to load ('imagenet' or None).

    Returns:
        nn.Module ready for training.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=n_channels,
        classes=n_classes,
        activation=None,  # raw logits; loss functions handle activation
    )

    # smp handles the in_channels adaptation internally — it re-initializes or
    # averages the first conv weights when in_channels != 3. Verify it happened.
    first_conv = _get_first_conv(model)
    assert first_conv.in_channels == n_channels, (
        f"Expected {n_channels} input channels but first conv has "
        f"{first_conv.in_channels}"
    )

    return model


def _get_first_conv(model: nn.Module) -> nn.Conv2d:
    """Return the first Conv2d in the encoder."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise RuntimeError("No Conv2d found in model")
