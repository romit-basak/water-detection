"""Check per-class F1 breakdown for the best aux model."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import torch
from torch.utils.data import DataLoader
from src.data.fast_dataset import FastUrbanSARFloods, make_urban_event_split
from src.data.transforms import val_transforms
from src.models.unet import build_unet
from src.utils.metrics import SegmentationMetrics
import segmentation_models_pytorch as smp
import torch.nn as nn

FINETUNE_TEST_EVENTS  = ['Hagibis','Sydney','Coraki','Niger','Hebei','Beledweyne','PortMacquarie']
FINETUNE_TRAIN_EVENTS = ['Houston','Beira','Japan','Canada','Iran','Lumberton','Somalia']

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

ckpt  = torch.load('runs/three_stage_aux/stage3/best.pt', map_location='cpu')
model = build_unet(n_channels=7, n_classes=4, encoder_name='resnet34',
                   encoder_weights=None).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Checkpoint: epoch {ckpt['epoch']}  stored_F1={ckpt['metric']:.4f}")

_, test_idx = make_urban_event_split(
    'data/urban_sar_floods_aux',
    train_events=FINETUNE_TRAIN_EVENTS,
    test_events=FINETUNE_TEST_EVENTS,
)
ds = FastUrbanSARFloods('data/urban_sar_floods_aux', test_idx,
                        transform=val_transforms(512), binary=False)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

# Evaluate with 4-class metrics (what the aux model was trained with)
metrics4 = SegmentationMetrics(n_classes=4, ignore_index=0)
pred_dist = {0:0, 1:0, 2:0, 3:0}

with torch.no_grad():
    for batch in loader:
        img  = batch['image'].to(device)
        mask = batch['mask'].to(device)
        pred = model(img).argmax(dim=1)
        metrics4.update(pred, mask)
        for c in range(4):
            pred_dist[c] += (pred == c).sum().item()

r = metrics4.compute()
total_pred = sum(pred_dist.values())
print(f'\n4-class evaluation (model was trained with 4 classes):')
print(f'  macro F1:  {r["macro_f1"]:.4f}')
print(f'  per-class F1: {[f"{x:.3f}" for x in r["per_class_f1"]]}')
print(f'  (classes: 0=ignore, 1=open_flood, 2=urban_flood, 3=non_flood)')
print(f'\nPrediction distribution:')
for c, n in pred_dist.items():
    print(f'  class {c}: {n:>10,}  ({100*n/total_pred:.1f}%)')
