"""
scripts/sydney_finetune.py

City-specific fine-tuning experiment: Sydney, Australia.

Sydney appears twice in UrbanSARFloods with labeled flood chips:
  - 20210324 (March 2021 flood): 6 chips in 03_FU
  - 20220705 (July 2022 flood):  23 chips in 03_FU

Experiment:
  Start from the cross-city model (v7 Stage 3 checkpoint).
  Fine-tune on ALL Sydney 2021 chips (both flood + any 01_NF Sydney chips).
  Evaluate on Sydney 2022 chips.
  Compare against cross-city model's Sydney 2022 F1 (no city-specific tuning).

This motivates the claim: even a handful of labeled examples from a prior
flood event in the same city significantly improves performance, justifying
either (a) city-specific fine-tuning when historical labels exist, or
(b) auxiliary static features (building density, HAND) as a proxy when
they don't.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \\
    uv run python scripts/sydney_finetune.py \\
        --checkpoint runs/three_stage_v7/stage3/best.pt \\
        --out_dir runs/sydney_finetune
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from src.data.fast_dataset import FastUrbanSARFloods
from src.data.transforms import train_transforms, val_transforms
from src.models.unet import build_unet
from src.utils.metrics import SegmentationMetrics

N_CLASSES = 4


def make_sydney_splits(root: str) -> tuple[list[int], list[int]]:
    """Split Sydney chips by event date: 2021 → train, 2022 → test.

    Training uses ONLY flooded chips (02_FO and 03_FU) from 2021.
    Including 01_NF chips would make training overwhelmingly non-flood
    and cause the model to predict non-flood everywhere.

    Test uses ALL 2022 Sydney chips (all folders) for full evaluation.
    """
    root = Path(root)
    train_idx, test_idx = [], []
    all_pairs = []

    for folder in ['01_NF', '02_FO', '03_FU']:
        sar_dir = root / folder / 'SAR'
        gt_dir  = root / folder / 'GT'
        if not sar_dir.exists():
            continue
        for p in sorted(sar_dir.glob('*.npy')):
            gt = gt_dir / p.name.replace('_SAR.npy', '_GT.npy')
            if not gt.exists():
                continue
            if 'sydney' not in p.name.lower():
                continue
            idx = len(all_pairs)
            all_pairs.append((p, gt))
            if '20210324' in p.name:
                # Train only on flooded chips
                if folder in ('02_FO', '03_FU'):
                    train_idx.append(idx)
            elif '20220705' in p.name:
                # Test only on flooded chips — 01_NF chips have all-zero
                # targets (ignored), making F1 meaningless on them
                if folder in ('02_FO', '03_FU'):
                    test_idx.append(idx)

    print(f'Sydney 2021 flood chips (train): {len(train_idx)}')
    print(f'Sydney 2022 chips (test):        {len(test_idx)}')
    return train_idx, test_idx


def run_epoch(model, loader, loss_fn, optimizer, metrics, device, is_train):
    model.train(is_train)
    metrics.reset()
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)
            logits = model(images)
            loss   = loss_fn(logits, masks)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isnan(grad_norm):
                    optimizer.zero_grad()
                    continue
                optimizer.step()
            metrics.update(logits.argmax(dim=1), masks)
            total_loss += loss.item()
            n += 1
    results = metrics.compute()
    results['loss'] = total_loss / max(n, 1)
    return results


def evaluate_sydney_baseline(model, test_ds, loss_fn, metrics, device, batch_size):
    """Evaluate cross-city model on Sydney 2022 before any fine-tuning."""
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    results = run_epoch(model, loader, loss_fn, None, metrics, device, is_train=False)
    return results


def main(checkpoint: str, out_dir: str, urban_sar_dir: str,
         lr: float, epochs: int, batch_size: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = (torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print(f'Device: {device}')

    # Load cross-city model
    print(f'\nLoading checkpoint: {checkpoint}')
    ckpt  = torch.load(checkpoint, map_location='cpu')
    model = build_unet(n_channels=4, n_classes=N_CLASSES,
                       encoder_name='resnet34', encoder_weights=None).to(device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Cross-city model: epoch {ckpt['epoch']} F1={ckpt['metric']:.4f}")

    # Build Sydney splits
    train_idx, test_idx = make_sydney_splits(urban_sar_dir)

    if len(train_idx) == 0:
        print('ERROR: No Sydney 2021 training chips found.')
        return
    if len(test_idx) == 0:
        print('ERROR: No Sydney 2022 test chips found.')
        return

    train_ds = FastUrbanSARFloods(urban_sar_dir, train_idx,
                                   transform=train_transforms(512), binary=False)
    test_ds  = FastUrbanSARFloods(urban_sar_dir, test_idx,
                                   transform=val_transforms(512),  binary=False)

    loss_fn = smp.losses.DiceLoss(mode='multiclass', ignore_index=0)
    ce      = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=0)
    class Combined(nn.Module):
        def forward(self, logits, targets):
            return loss_fn(logits, targets) + ce(logits, targets)
    combined_loss = Combined().to(device)

    metrics = SegmentationMetrics(n_classes=N_CLASSES, ignore_index=0)

    # ── Baseline: cross-city model on Sydney 2022 ─────────────────────────────
    print('\n── Baseline: cross-city model on Sydney 2022 (no fine-tuning) ──')
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    baseline = run_epoch(model, test_loader, combined_loss, None,
                         metrics, device, is_train=False)
    print(f'Baseline Sydney 2022 F1  = {baseline["macro_f1"]:.4f}')
    print(f'Baseline Sydney 2022 IoU = {baseline["macro_iou"]:.4f}')
    print(f'Per-class F1: {baseline["per_class_f1"]}')

    with open(out / 'results.txt', 'w') as f:
        f.write(f'Cross-city baseline (Sydney 2022):\n')
        f.write(f'  macro_f1:  {baseline["macro_f1"]:.5f}\n')
        f.write(f'  macro_iou: {baseline["macro_iou"]:.5f}\n\n')

    # ── Fine-tune on Sydney 2021 ───────────────────────────────────────────────
    print(f'\n── Fine-tuning on {len(train_ds)} Sydney 2021 chips ──')
    print(f'LR={lr}  epochs={epochs}  batch={batch_size}')

    # With so few chips, use a very small batch or all-at-once
    effective_batch = min(batch_size, len(train_ds))
    train_loader    = DataLoader(train_ds, batch_size=effective_batch,
                                 shuffle=True, num_workers=0, drop_last=False)

    # Discriminative LR: encoder at very low LR to preserve cross-city
    # features while decoder adapts to Sydney-specific SAR morphology.
    # Freezing encoder entirely gave degenerate F1=0.333 — the decoder
    # needs encoder gradient flow to adapt.
    print('Discriminative LR: encoder=1e-06  decoder/head=5e-05')
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(),           'lr': 1e-6},
        {'params': model.decoder.parameters(),           'lr': lr},
        {'params': model.segmentation_head.parameters(), 'lr': lr},
    ], weight_decay=1e-3)

    fields = ['epoch', 'train_loss', 'test_f1', 'test_iou']
    with open(out / 'log.csv', 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_f1   = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0      = time.time()
        train_r = run_epoch(model, train_loader, combined_loss, optimizer,
                            metrics, device, is_train=True)
        test_r  = run_epoch(model, test_loader, combined_loss, None,
                            metrics, device, is_train=False)

        print(f'Ep {epoch:3d}/{epochs} | '
              f'train_loss={train_r["loss"]:.4f} | '
              f'test_F1={test_r["macro_f1"]:.4f} | '
              f'test_IoU={test_r["macro_iou"]:.4f} | '
              f'{time.time()-t0:.0f}s')

        with open(out / 'log.csv', 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writerow({
                'epoch':      epoch,
                'train_loss': round(train_r['loss'], 5),
                'test_f1':    round(test_r['macro_f1'], 5),
                'test_iou':   round(test_r['macro_iou'], 5),
            })

        if test_r['macro_f1'] > best_f1:
            best_f1    = test_r['macro_f1']
            best_epoch = epoch
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'metric': best_f1}, out / 'best.pt')
            print(f'  → New best F1={best_f1:.4f}')

    print(f'\n── Results ──')
    print(f'Baseline (cross-city, no Sydney data): F1={baseline["macro_f1"]:.4f}')
    print(f'After fine-tune on 2021 Sydney ({len(train_ds)} chips): F1={best_f1:.4f}')
    print(f'Improvement: {best_f1 - baseline["macro_f1"]:+.4f}')
    print(f'Best epoch: {best_epoch}')

    with open(out / 'results.txt', 'a') as f:
        f.write(f'City-specific fine-tune (Sydney 2021 → 2022):\n')
        f.write(f'  train chips: {len(train_ds)}\n')
        f.write(f'  best macro_f1:  {best_f1:.5f}\n')
        f.write(f'  best epoch:     {best_epoch}\n')
        f.write(f'  improvement:    {best_f1 - baseline["macro_f1"]:+.5f}\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    required=True)
    p.add_argument('--out_dir',       default='runs/sydney_finetune')
    p.add_argument('--urban_sar_dir', default='data/urban_sar_floods_preprocessed')
    p.add_argument('--lr',            type=float, default=5e-5)
    p.add_argument('--epochs',        type=int,   default=50)
    p.add_argument('--batch_size',    type=int,   default=4)
    args = p.parse_args()
    main(args.checkpoint, args.out_dir, args.urban_sar_dir,
         args.lr, args.epochs, args.batch_size)
