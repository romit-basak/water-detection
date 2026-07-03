"""
Augmentation pipelines for SAR flood detection.

SAR-specific considerations:
  - No color jitter (SAR is not optical; intensity is physically meaningful)
  - Horizontal/vertical flips are safe (backscatter is symmetric)
  - Rotations should be multiples of 90° (preserves grid alignment)
  - Brightness/contrast shifts are acceptable in dB space within a small range
  - Speckle noise augmentation is realistic for SAR (optional, not yet implemented)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transforms(image_size: int = 512) -> A.Compose:
    """Standard training augmentations for SAR chips."""
    return A.Compose([
        A.RandomCrop(height=image_size, width=image_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Small brightness shift in normalized space mimics sensor calibration variation
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),
        # Coarse dropout simulates no-data/shadow regions
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 64),
            hole_width_range=(16, 64),
            fill=0.0,
            p=0.2,
        ),
    ])


def val_transforms(image_size: int = 512) -> A.Compose:
    """Validation transforms — deterministic center crop only."""
    return A.Compose([
        A.CenterCrop(height=image_size, width=image_size, p=1.0),
    ])
