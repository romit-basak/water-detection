"""
Quick diagnostic: check UrbanSARFloods chips for NaN/Inf values
and verify the local variance computation doesn't produce NaN.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from src.data.dataset import UrbanSARFloodsDataset, make_splits
from src.data.transforms import train_transforms

root = 'data/urban_sar_floods'
train_idx, _ = make_splits(root, seed=42)

ds = UrbanSARFloodsDataset(
    root=root,
    split_indices=train_idx[:50],
    transform=train_transforms(512),
    use_change_features=False,
    use_local_variance=True,
    binary=True,
)

nan_chips = []
for i in range(len(ds)):
    s = ds[i]
    img = s['image']
    msk = s['mask']
    if torch.isnan(img).any():
        nan_chips.append((i, s['chip_path'], 'image NaN'))
    if torch.isinf(img).any():
        nan_chips.append((i, s['chip_path'], 'image Inf'))
    if torch.isnan(msk.float()).any():
        nan_chips.append((i, s['chip_path'], 'mask NaN'))
    # Check variance channels specifically
    var_channels = img[2:]
    if (var_channels < 0).any():
        nan_chips.append((i, s['chip_path'], f'negative variance: min={var_channels.min():.4f}'))

if nan_chips:
    print(f'Found {len(nan_chips)} problematic chips:')
    for idx, path, issue in nan_chips[:20]:
        print(f'  [{idx}] {issue}: {path}')
else:
    print('No NaN/Inf in first 50 chips — checking loss...')

    # Simulate a forward + loss pass
    import segmentation_models_pytorch as smp
    from src.models.unet import build_unet
    model = build_unet(n_channels=4, n_classes=3, encoder_name='resnet34',
                       encoder_weights=None)
    model.eval()

    batch = torch.stack([ds[i]['image'] for i in range(4)])
    masks = torch.stack([ds[i]['mask'] for i in range(4)])
    print(f'batch: {batch.shape} min={batch.min():.4f} max={batch.max():.4f}')
    print(f'masks unique: {masks.unique().tolist()}')

    with torch.no_grad():
        logits = model(batch)
    print(f'logits: {logits.shape} nan={torch.isnan(logits).any()} inf={torch.isinf(logits).any()}')

    dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=0)
    ce   = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=0)
    print(f'dice loss: {dice(logits, masks).item():.4f}')
    print(f'ce loss:   {ce(logits, masks).item():.4f}')
