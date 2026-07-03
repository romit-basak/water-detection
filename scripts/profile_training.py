"""
Profile exactly where time is being spent per batch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from src.data.dataset import UrbanSARFloodsDataset, make_splits
from src.data.transforms import train_transforms
from src.models.unet import build_unet

device = torch.device('mps')
root   = 'data/urban_sar_floods'
train_idx, _ = make_splits(root, seed=42)

ds = UrbanSARFloodsDataset(root, train_idx,
    transform=train_transforms(512),
    use_change_features=False, use_local_variance=True, binary=False)

loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

model = build_unet(n_channels=4, n_classes=4, encoder_name='resnet34',
                   encoder_weights=None).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=0)
ce   = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=0)

# Warmup
batch = next(iter(loader))
images = batch['image'].to(device)
masks  = batch['mask'].to(device)
logits = model(images)
loss   = dice(logits, masks) + ce(logits, masks)
loss.backward()
optimizer.zero_grad()

# Profile 10 batches
N = 10
times = {'load': [], 'to_device': [], 'forward': [], 'loss': [], 'backward': [], 'step': []}

it = iter(loader)
for _ in range(N):
    t0 = time.perf_counter()
    batch = next(it)
    times['load'].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    images = batch['image'].to(device)
    masks  = batch['mask'].to(device)
    torch.mps.synchronize()
    times['to_device'].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    logits = model(images)
    torch.mps.synchronize()
    times['forward'].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    loss = dice(logits, masks) + ce(logits, masks)
    torch.mps.synchronize()
    times['loss'].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    optimizer.zero_grad()
    loss.backward()
    torch.mps.synchronize()
    times['backward'].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    optimizer.step()
    torch.mps.synchronize()
    times['step'].append(time.perf_counter() - t0)

print(f'\nPer-batch timing breakdown (avg over {N} batches, batch_size=16):')
total = 0
for k, v in times.items():
    avg = sum(v) / len(v)
    total += avg
    print(f'  {k:<12}: {avg*1000:6.1f} ms')
print(f'  {"TOTAL":<12}: {total*1000:6.1f} ms')
print(f'\nProjected epoch time for 3261 chips @ batch=16:')
n_batches = 3261 // 16
print(f'  {n_batches} batches × {total:.3f}s = {n_batches*total/60:.1f} minutes')
