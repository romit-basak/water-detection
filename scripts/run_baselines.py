"""
scripts/run_baselines.py

Runs the two baseline comparisons for the paper:

Baseline A: Random init → fine-tune on UrbanSARFloods (no pretraining at all)
Baseline B: ImageNet init → fine-tune on UrbanSARFloods (standard transfer learning)

Both use identical hyperparameters and the same event-level split as the main model.
Results are saved to runs/baseline_A/ and runs/baseline_B/.

Compare against:
  Our model (v7): Sen1Floods11 pretrain → UrbanSARFloods fine-tune, F1=0.356

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \\
    uv run python scripts/run_baselines.py
"""

import sys
import csv
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from src.data.fast_dataset import FastUrbanSARFloods, make_urban_event_split
from src.data.transforms import train_transforms, val_transforms
from src.models.unet import build_unet
from src.utils.metrics import SegmentationMetrics

URBAN_SAR_DIR = 'data/urban_sar_floods_preprocessed'
N_CLASSES     = 4
EPOCHS        = 10
LR            = 1e-4
BATCH         = 16

FINETUNE_TRAIN_EVENTS = ['Houston','Beira','Japan','Canada','Iran','Lumberton','Somalia']
FINETUNE_TEST_EVENTS  = ['Hagibis','Sydney','Coraki','Niger','Hebei','Beledweyne','PortMacquarie']


def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def build_loss(device):
    dice = smp.losses.DiceLoss(mode='multiclass', ignore_index=0)
    ce   = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=0)
    class C(nn.Module):
        def forward(self, l, t): return dice(l, t) + ce(l, t)
    return C().to(device)


def run_epoch(model, loader, loss_fn, optimizer, metrics, device, is_train):
    model.train(is_train)
    metrics.reset()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            img  = batch['image'].to(device)
            mask = batch['mask'].to(device)
            logits = model(img)
            loss   = loss_fn(logits, mask)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            metrics.update(logits.argmax(dim=1), mask)
            total += loss.item(); n += 1
    r = metrics.compute()
    r['loss'] = total / max(n, 1)
    return r


def run_baseline(name: str, encoder_weights, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f'\n{"="*60}')
    print(f'BASELINE: {name}')
    print(f'encoder_weights={encoder_weights}')
    print(f'{"="*60}')

    train_idx, test_idx = make_urban_event_split(
        URBAN_SAR_DIR,
        train_events=FINETUNE_TRAIN_EVENTS,
        test_events=FINETUNE_TEST_EVENTS,
    )

    train_ds = FastUrbanSARFloods(URBAN_SAR_DIR, train_idx,
        transform=train_transforms(512), binary=False)
    test_ds  = FastUrbanSARFloods(URBAN_SAR_DIR, test_idx,
        transform=val_transforms(512),  binary=False)

    print(f'Train chips: {len(train_ds)}  Test chips: {len(test_ds)}')

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_unet(n_channels=4, n_classes=N_CLASSES,
                       encoder_name='resnet34',
                       encoder_weights=encoder_weights).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)

    # Cosine schedule with warmup
    def lr_lambda(epoch):
        warmup = 3
        if epoch < warmup: return (epoch + 1) / warmup
        p = (epoch - warmup) / max(1, EPOCHS - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = build_loss(device)
    metrics = SegmentationMetrics(n_classes=N_CLASSES, ignore_index=0)

    fields  = ['epoch', 'train_loss', 'test_loss', 'test_f1', 'test_iou']
    with open(out / 'log.csv', 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0      = time.time()
        train_r = run_epoch(model, train_loader, loss_fn, optimizer,
                            metrics, device, is_train=True)
        test_r  = run_epoch(model, test_loader,  loss_fn, None,
                            metrics, device, is_train=False)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f'Ep {epoch:2d}/{EPOCHS} | loss={train_r["loss"]:.4f} | '
              f'test_F1={test_r["macro_f1"]:.4f} | '
              f'test_IoU={test_r["macro_iou"]:.4f} | '
              f'lr={lr:.2e} | {time.time()-t0:.0f}s')

        with open(out / 'log.csv', 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writerow({
                'epoch': epoch,
                'train_loss': round(train_r['loss'], 5),
                'test_loss':  round(test_r['loss'], 5),
                'test_f1':    round(test_r['macro_f1'], 5),
                'test_iou':   round(test_r['macro_iou'], 5),
            })

        if test_r['macro_f1'] > best_f1:
            best_f1 = test_r['macro_f1']
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'metric': best_f1}, out / 'best.pt')
            print(f'  → New best F1={best_f1:.4f}')

    print(f'\n{name} best F1={best_f1:.4f}')
    with open(out / 'result.txt', 'w') as f:
        f.write(f'best_f1: {best_f1:.5f}\n')
        f.write(f'encoder_weights: {encoder_weights}\n')
        f.write(f'epochs: {EPOCHS}\n')
    return best_f1


if __name__ == '__main__':
    f1_a = run_baseline('Baseline A — Random init',
                        encoder_weights=None,
                        out_dir='runs/baseline_A')

    f1_b = run_baseline('Baseline B — ImageNet init',
                        encoder_weights='imagenet',
                        out_dir='runs/baseline_B')

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'Baseline A (random init):   F1={f1_a:.4f}')
    print(f'Baseline B (ImageNet init): F1={f1_b:.4f}')
    print(f'Our model  (Sen1Fl pretrain): F1=0.3558')
    print(f'Zero-shot  (no UrbanSAR):   F1=0.252')
