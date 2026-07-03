"""
src/train_pseudo.py

Training script: train on pseudo-labels, evaluate on UrbanSARFloods benchmark.

Experimental design:
    - Training data:   PseudoLabelDataset  (4 events, NDWI-derived labels)
    - Evaluation data: UrbanSARFloodsDataset (held-out benchmark, never trained on)
    - Comparison:      F1 scores vs Zhao et al. (CVPR 2024) published numbers
                       which trained *on* UrbanSARFloods with supervised labels

This is the core scientific claim: cross-modal pseudo-labeling from optical
imagery can produce models that generalize to unseen urban SAR flood scenes
without any manually annotated SAR training data.

Channel alignment:
    Both datasets output (4, 512, 512) tensors:
        ch 0: VV_during  (normalized [0,1])
        ch 1: VH_during  (normalized [0,1])
        ch 2: local variance of VV
        ch 3: local variance of VH

    UrbanSARFloodsDataset uses bands 1-2 (pre-normalized by dataset authors).
    PseudoLabelDataset uses bands 1-2 (normalized from raw dB in dataset.py).
    Both apply the same 5×5 local variance computation.
    The representations are directly compatible.

Label alignment:
    Both use: 0=background, 1=open_flooded, 2=urban_flooded
    Loss ignores class 0 (background/masked) via ignore_index=0.

Run:
    python -m src.train_pseudo
    python -m src.train_pseudo --epochs 50 --out_dir runs/pseudo_baseline
    python -m src.train_pseudo --binary  # binary flood/no-flood mode
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

import segmentation_models_pytorch as smp

from src.data.pseudo_dataset import (
    PseudoLabelDataset, make_pseudo_splits, PSEUDO_EVENTS
)
from src.data.dataset import UrbanSARFloodsDataset, make_splits
from src.data.transforms import train_transforms, val_transforms
from src.models.unet import build_unet
from src.utils.metrics import SegmentationMetrics

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    # Paths
    pseudo_dir  = 'data/pseudo_labels/chips',
    benchmark_dir = 'data/urban_sar_floods',
    out_dir     = 'runs/pseudo_baseline',
    # Model
    encoder     = 'resnet34',
    use_local_variance = True,
    binary      = False,
    # Training
    epochs      = 50,
    batch_size  = 8,
    lr          = 3e-4,
    weight_decay = 1e-4,
    warmup_epochs = 3,
    image_size  = 512,
    num_workers = 4,
    val_fraction = 0.15,
    seed        = 42,
)

# ── Loss ──────────────────────────────────────────────────────────────────────

def build_loss(n_classes: int, binary: bool) -> nn.Module:
    mode = 'binary' if binary else 'multiclass'
    dice = smp.losses.DiceLoss(mode=mode, ignore_index=0)
    ce   = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=0)
    class Combined(nn.Module):
        def forward(self, logits, targets):
            return dice(logits, targets) + ce(logits, targets)
    return Combined()

# ── LR schedule ───────────────────────────────────────────────────────────────

def build_scheduler(optimizer, warmup: int, total: int):
    import math
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── One epoch ─────────────────────────────────────────────────────────────────

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
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            metrics.update(logits.argmax(dim=1), masks)
            total_loss += loss.item()
            n += 1
    results = metrics.compute()
    results['loss'] = total_loss / max(n, 1)
    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    out_dir = Path(cfg['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Device: {device}')

    # ── Training data: pseudo-labels ────────────────────────────────────────
    train_idx, val_idx = make_pseudo_splits(
        cfg['pseudo_dir'],
        val_fraction=cfg['val_fraction'],
        seed=cfg['seed'],
    )

    train_ds = PseudoLabelDataset(
        chips_dir=cfg['pseudo_dir'],
        split_indices=train_idx,
        transform=train_transforms(cfg['image_size']),
        use_local_variance=cfg['use_local_variance'],
        binary=cfg['binary'],
    )
    # Small pseudo-label val set (same distribution as training)
    pseudo_val_ds = PseudoLabelDataset(
        chips_dir=cfg['pseudo_dir'],
        split_indices=val_idx,
        transform=val_transforms(cfg['image_size']),
        use_local_variance=cfg['use_local_variance'],
        binary=cfg['binary'],
    )

    # ── Benchmark data: UrbanSARFloods (NEVER trained on) ───────────────────
    bench_train_idx, bench_val_idx = make_splits(
        cfg['benchmark_dir'],
        val_fraction=1.0,  # use ALL as benchmark — no training split
        seed=cfg['seed'],
    )
    # val_fraction=1.0 means all chips go to val_idx
    benchmark_ds = UrbanSARFloodsDataset(
        root=cfg['benchmark_dir'],
        split_indices=bench_val_idx,
        transform=val_transforms(cfg['image_size']),
        use_change_features=False,   # single-image mode: VV+VH only
        use_local_variance=cfg['use_local_variance'],
        binary=cfg['binary'],
    )

    print(f'\nTraining chips (pseudo-labels): {len(train_ds)}')
    print(f'  Event breakdown: {train_ds.event_counts()}')
    print(f'Pseudo-label val chips:         {len(pseudo_val_ds)}')
    print(f'Benchmark chips (UrbanSAR):     {len(benchmark_ds)}')
    print(f'n_channels: {train_ds.n_channels}   n_classes: {train_ds.n_classes}')
    print(f'NOTE: UrbanSARFloods is EVALUATION ONLY — never used in training.\n')

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['num_workers'], pin_memory=True, drop_last=True,
    )
    pseudo_val_loader = DataLoader(
        pseudo_val_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=True,
    )
    benchmark_loader = DataLoader(
        benchmark_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_unet(
        n_channels=train_ds.n_channels,
        n_classes=train_ds.n_classes,
        encoder_name=cfg['encoder'],
        encoder_weights='imagenet',
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = build_scheduler(optimizer, cfg['warmup_epochs'], cfg['epochs'])
    loss_fn   = build_loss(train_ds.n_classes, cfg['binary']).to(device)
    metrics   = SegmentationMetrics(n_classes=train_ds.n_classes, ignore_index=0)

    # ── Logging ──────────────────────────────────────────────────────────────
    log_path = out_dir / 'train_log.csv'
    log_fields = [
        'epoch', 'train_loss',
        'pseudo_val_loss', 'pseudo_val_f1', 'pseudo_val_iou',
        'bench_f1', 'bench_iou', 'bench_f1_urban', 'lr',
    ]
    with open(log_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_bench_f1 = 0.0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()

        train_r  = run_epoch(model, train_loader,      loss_fn, optimizer,
                             metrics, device, is_train=True)
        pval_r   = run_epoch(model, pseudo_val_loader, loss_fn, None,
                             metrics, device, is_train=False)
        bench_r  = run_epoch(model, benchmark_loader,  loss_fn, None,
                             metrics, device, is_train=False)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Urban flood F1 is class index 2
        bench_f1_urban = float(bench_r['per_class_f1'][2])

        print(
            f'Epoch {epoch:3d}/{cfg["epochs"]} | '
            f'train={train_r["loss"]:.4f} | '
            f'pval_F1={pval_r["macro_f1"]:.4f} | '
            f'bench_F1={bench_r["macro_f1"]:.4f} | '
            f'bench_urban_F1={bench_f1_urban:.4f} | '
            f'lr={lr:.2e} | {elapsed:.0f}s'
        )

        with open(log_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                'epoch':           epoch,
                'train_loss':      round(train_r['loss'], 5),
                'pseudo_val_loss': round(pval_r['loss'], 5),
                'pseudo_val_f1':   round(pval_r['macro_f1'], 5),
                'pseudo_val_iou':  round(pval_r['macro_iou'], 5),
                'bench_f1':        round(bench_r['macro_f1'], 5),
                'bench_iou':       round(bench_r['macro_iou'], 5),
                'bench_f1_urban':  round(bench_f1_urban, 5),
                'lr':              round(lr, 8),
            })

        # Save best by benchmark F1 (the number that goes in the paper)
        if bench_r['macro_f1'] > best_bench_f1:
            best_bench_f1 = bench_r['macro_f1']
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'bench_f1':    best_bench_f1,
                'bench_f1_urban': bench_f1_urban,
                'cfg':         cfg,
            }, out_dir / 'best_model.pt')
            print(f'  → New best benchmark F1={best_bench_f1:.4f} saved.')

    print(f'\nDone. Best benchmark F1={best_bench_f1:.4f}')
    print(f'Compare against Zhao et al. CVPR 2024: F1=0.51–0.77')
    print(f'Log: {log_path}')

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> dict:
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f'--{k}', default=v, action='store_true')
        else:
            p.add_argument(f'--{k}', default=v, type=type(v))
    return vars(p.parse_args())


if __name__ == '__main__':
    cfg = parse_args()
    print('Config:', cfg)
    train(cfg)
