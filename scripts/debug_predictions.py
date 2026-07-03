"""
Check what the model actually predicts on test chips.
Compares training-time F1 with per-chip evaluation.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import numpy as np
import torch
from src.data.fast_dataset import FastUrbanSARFloods, make_urban_event_split
from src.data.transforms import val_transforms
from src.models.unet import build_unet
from src.utils.metrics import SegmentationMetrics

FINETUNE_TEST_EVENTS  = ['Hagibis','Sydney','Coraki','Niger','Hebei','Beledweyne','PortMacquarie']
FINETUNE_TRAIN_EVENTS = ['Houston','Beira','Japan','Canada','Iran','Lumberton','Somalia']

device = (torch.device('mps') if torch.backends.mps.is_available()
          else torch.device('cpu'))

ckpt  = torch.load('runs/three_stage_v5/stage3/best.pt', map_location='cpu')
model = build_unet(n_channels=4, n_classes=4, encoder_name='resnet34',
                   encoder_weights=None).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

_, test_idx = make_urban_event_split(
    'data/urban_sar_floods_preprocessed',
    train_events=FINETUNE_TRAIN_EVENTS,
    test_events=FINETUNE_TEST_EVENTS,
)
ds = FastUrbanSARFloods('data/urban_sar_floods_preprocessed', test_idx,
                        transform=val_transforms(512), binary=False)

# Replicate training-time evaluation exactly
from torch.utils.data import DataLoader
loader  = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
metrics = SegmentationMetrics(n_classes=4, ignore_index=0)

pred_class_counts = np.zeros(4, dtype=np.int64)

with torch.no_grad():
    for i, batch in enumerate(loader):
        images = batch['image'].to(device)
        masks  = batch['mask'].to(device)
        logits = model(images)
        preds  = logits.argmax(dim=1)
        metrics.update(preds, masks)

        # Count what the model actually predicts
        for c in range(4):
            pred_class_counts[c] += (preds == c).sum().item()

        if i == 0:
            # Show first batch detail
            print(f'First batch:')
            print(f'  logits range: {logits.min():.3f} to {logits.max():.3f}')
            print(f'  pred unique: {preds.unique().cpu().tolist()}')
            print(f'  mask unique: {masks.unique().cpu().tolist()}')
            print(f'  pred class dist: {[(preds==c).sum().item() for c in range(4)]}')

results = metrics.compute()
total_pred = pred_class_counts.sum()
print(f'\nTraining-time macro F1: {results["macro_f1"]:.4f}')
print(f'Per-class F1: {results["per_class_f1"]}')
print(f'\nModel prediction distribution across all test chips:')
for c in range(4):
    print(f'  class {c}: {pred_class_counts[c]:>10,}  ({100*pred_class_counts[c]/total_pred:.1f}%)')
print('\nClass meanings: 0=ignore/bg, 1=open_flood, 2=urban_flood, 3=non_flood')
