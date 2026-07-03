"""
scripts/train_with_pseudo.py

Four-stage training pipeline using pseudo-labeled new-event chips.

Stage 0 — Pretrain on pseudo-labeled chips (new events, confidence-weighted)
Stage 1 — Zero-shot eval on UrbanSARFloods held-out test split
Stage 2 — Fine-tune on UrbanSARFloods training split (event-level)
Stage 3 — Final eval on UrbanSARFloods held-out test split

This extends train_three_stage.py by adding Stage 0 pseudo-pretraining.
The hypothesis: pseudo-labeled chips from ~12 new events provide enough
diverse SAR flood signatures to escape the data-volume ceiling at 7 cities.

Event splits (same as train_three_stage.py — DO NOT change):
  UrbanSARFloods train: Houston, Beira, Japan, Canada, Iran, Lumberton, Somalia
  UrbanSARFloods test:  Hagibis, Sydney, Coraki, Niger, Hebei, Beledweyne, PortMacquarie

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \\
    uv run python scripts/train_with_pseudo.py

    # Skip Stage 0 (resume from existing pseudo-pretrain checkpoint):
    uv run python scripts/train_with_pseudo.py \\
        --skip_pseudo --pseudo_ckpt runs/pseudo/stage0/best.pt

    # Ablation: fine-tune only (ImageNet init, no pseudo pretraining) as baseline:
    uv run python scripts/train_with_pseudo.py --finetune_only

Comparison target: v9_aux baseline
    Peak flood F1 = 0.1504 (8-channel, fw=3, ImageNet init, 7 training cities)
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fast_dataset import (
    FastUrbanSARFloods,
    make_urban_event_split,
)
from src.data.pseudo_chip_dataset import PseudoChipDataset
from src.data.transforms import train_transforms, val_transforms
from src.models.unet import build_unet
from src.utils.metrics import SegmentationMetrics

# ── Constants ─────────────────────────────────────────────────────────────────
FINETUNE_TRAIN_EVENTS = [
    'Houston', 'Beira', 'Japan', 'Canada', 'Iran', 'Lumberton', 'Somalia',
]
FINETUNE_TEST_EVENTS = [
    'Hagibis', 'Sydney', 'Coraki', 'Niger', 'Hebei', 'Beledweyne', 'PortMacquarie',
]

N_CLASSES    = 2
IGNORE_INDEX = -1

DEFAULTS = dict(
    pseudo_chips_dir   = 'data/pseudo_chips',
    urban_sar_dir      = 'data/urban_sar_floods_aux',  # 8-channel aux chips
    out_dir            = 'runs/pseudo',
    encoder            = 'resnet34',
    # Stage 0 (pseudo pretrain)
    pseudo_epochs      = 20,
    pseudo_lr          = 1e-4,
    pseudo_batch       = 16,
    # Stage 2 (UrbanSARFloods fine-tune)
    finetune_epochs    = 40,
    finetune_lr        = 1e-4,
    finetune_batch     = 16,
    flood_weight       = 3.0,
    num_workers        = 0,
    seed               = 42,
    image_size         = 512,
    # Control flags
    skip_pseudo        = False,   # skip Stage 0, load from pseudo_ckpt
    pseudo_ckpt        = '',      # path to Stage 0 checkpoint
    finetune_only      = False,   # ImageNet init baseline, no pseudo pretraining
)


# =============================================================================
# Utilities (mirrors train_three_stage.py)
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    pin    = torch.cuda.is_available()
    mp_ctx = 'spawn' if (num_workers > 0 and not torch.cuda.is_available()) else None
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context=mp_ctx,
        drop_last=shuffle,
    )


def build_loss(device: torch.device, flood_weight: float = 3.0) -> nn.Module:
    gamma = 2.0
    alpha = torch.tensor(
        [1.0 / (1.0 + flood_weight), flood_weight / (1.0 + flood_weight)],
        device=device
    )

    class FocalLoss(nn.Module):
        def forward(self, logits, targets, weights=None):
            log_p  = torch.nn.functional.log_softmax(logits, dim=1)
            p      = log_p.exp()
            valid  = targets != IGNORE_INDEX
            t_safe = targets.clone()
            t_safe[~valid] = 0
            t_exp  = t_safe.unsqueeze(1)
            log_pt = log_p.gather(1, t_exp).squeeze(1)
            pt     = p.gather(1, t_exp).squeeze(1)
            a_t    = alpha[t_safe]
            loss   = -a_t * (1 - pt) ** gamma * log_pt
            loss   = loss * valid.float()
            if weights is not None:
                loss = loss * weights
            return loss.sum() / valid.float().sum().clamp(min=1)

    return FocalLoss().to(device)


def build_scheduler(optimizer, warmup: int, total: int):
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        p = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, loss_fn, optimizer, metrics, device, is_train: bool):
    model.train(is_train)
    metrics.reset()
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            images  = batch['image'].to(device)
            masks   = batch['mask'].to(device)
            weights = batch.get('weight')
            if weights is not None:
                weights = weights.to(device)
            logits = model(images)
            loss   = loss_fn(logits, masks, weights)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isnan(grad_norm):
                    print('  WARNING: NaN gradient, skipping batch')
                    optimizer.zero_grad()
                    continue
                optimizer.step()
            metrics.update(logits.argmax(dim=1), masks)
            total_loss += loss.item()
            n += 1
    results          = metrics.compute()
    results['loss']  = total_loss / max(n, 1)
    return results


def save_ckpt(path, model, optimizer, epoch, metric, cfg):
    torch.save({
        'epoch': epoch, 'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metric': metric, 'cfg': cfg,
    }, path)


def write_log(path, fields, row):
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


# =============================================================================
# Urban SAR split helpers
# =============================================================================

def make_test_loader(cfg: dict) -> DataLoader:
    """UrbanSARFloods held-out test set (never touched during training)."""
    _, test_idx = make_urban_event_split(
        cfg['urban_sar_dir'],
        train_events=FINETUNE_TRAIN_EVENTS,
        test_events=FINETUNE_TEST_EVENTS,
    )
    ds = FastUrbanSARFloods(
        root=cfg['urban_sar_dir'],
        split_indices=test_idx,
        transform=val_transforms(cfg['image_size']),
    )
    print(f'Test set: {len(ds)} chips ({len(test_idx)} indices)')
    return make_loader(ds, cfg['finetune_batch'], False, cfg['num_workers'])


def make_train_loader(cfg: dict, augment: bool = True) -> DataLoader:
    """UrbanSARFloods training split."""
    train_idx, _ = make_urban_event_split(
        cfg['urban_sar_dir'],
        train_events=FINETUNE_TRAIN_EVENTS,
        test_events=FINETUNE_TEST_EVENTS,
    )
    ds = FastUrbanSARFloods(
        root=cfg['urban_sar_dir'],
        split_indices=train_idx,
        transform=(train_transforms(cfg['image_size']) if augment
                   else val_transforms(cfg['image_size'])),
    )
    print(f'Fine-tune train set: {len(ds)} chips')
    return make_loader(ds, cfg['finetune_batch'], True, cfg['num_workers'])


# =============================================================================
# Stage 0: Pseudo-label pretraining
# =============================================================================

def stage0_pseudo_pretrain(cfg: dict, device: torch.device) -> nn.Module:
    out = Path(cfg['out_dir']) / 'stage0'
    out.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*60)
    print('STAGE 0 -- Pseudo-label pretraining on new-event chips')
    print('='*60)

    pseudo_ds = PseudoChipDataset(
        root=cfg['pseudo_chips_dir'],
        transform=train_transforms(cfg['image_size']),
    )

    if len(pseudo_ds) == 0:
        raise RuntimeError(
            f'No pseudo chips found in {cfg["pseudo_chips_dir"]}. '
            'Run scripts/precompute_pseudo_chips.py first.'
        )

    n_ch = pseudo_ds.n_channels
    print(f'Pseudo chips: {len(pseudo_ds)}, n_channels={n_ch}')

    # Validation: use UrbanSARFloods TEST set for zero-shot monitoring.
    # This is read-only -- these chips never enter the loss.
    val_loader = make_test_loader(cfg)

    train_loader = make_loader(
        pseudo_ds, cfg['pseudo_batch'], shuffle=True,
        num_workers=cfg['num_workers']
    )

    model = build_unet(
        n_channels=n_ch,
        n_classes=N_CLASSES,
        encoder_name=cfg['encoder'],
        encoder_weights='imagenet',  # ImageNet init for encoder
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['pseudo_lr'], weight_decay=1e-4
    )
    scheduler = build_scheduler(optimizer, warmup=3, total=cfg['pseudo_epochs'])
    loss_fn   = build_loss(device, flood_weight=cfg['flood_weight'])
    metrics   = SegmentationMetrics(n_classes=N_CLASSES, ignore_index=IGNORE_INDEX)

    fields = ['epoch', 'train_loss', 'val_flood_f1', 'val_macro_f1']
    with open(out / 'log.csv', 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_f1    = 0.0
    patience   = 6
    no_improve = 0

    for epoch in range(1, cfg['pseudo_epochs'] + 1):
        t0      = time.time()
        train_r = run_epoch(model, train_loader, loss_fn, optimizer,
                            metrics, device, is_train=True)
        # Monitor on UrbanSARFloods test set (zero-shot -- not used for model selection,
        # just to watch whether pseudo pretraining is helping transfer)
        val_r   = run_epoch(model, val_loader, loss_fn, None,
                            metrics, device, is_train=False)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f'S0 {epoch:3d}/{cfg["pseudo_epochs"]} | '
              f'train_loss={train_r["loss"]:.4f} | '
              f'UrbanSAR_flood_F1={val_r["flood_f1"]:.4f} | '
              f'lr={lr:.2e} | {time.time()-t0:.0f}s')

        write_log(out / 'log.csv', fields, {
            'epoch':        epoch,
            'train_loss':   round(train_r['loss'], 5),
            'val_flood_f1': round(val_r['flood_f1'], 5),
            'val_macro_f1': round(val_r['macro_f1'], 5),
        })

        # Save checkpoint every epoch (cheap, enables inspection)
        save_ckpt(out / f'epoch_{epoch:02d}.pt', model, optimizer, epoch,
                  val_r['flood_f1'], cfg)

        if val_r['flood_f1'] > best_f1:
            best_f1    = val_r['flood_f1']
            no_improve = 0
            save_ckpt(out / 'best.pt', model, optimizer, epoch, best_f1, cfg)
            print(f'  -> New best zero-shot F1={best_f1:.4f}')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping: no zero-shot improvement for {patience} epochs')
                break

    print(f'\nStage 0 done. Best zero-shot flood F1={best_f1:.4f}')
    print(f'Baseline (ImageNet init, no pseudo): ~0.08-0.12')
    return model


# =============================================================================
# Stage 1: Zero-shot eval on UrbanSARFloods (after pseudo pretrain, before FT)
# =============================================================================

def stage1_zeroshot(model: nn.Module, cfg: dict, device: torch.device) -> dict:
    print('\n' + '='*60)
    print('STAGE 1 -- Zero-shot eval on UrbanSARFloods held-out test set')
    print('(Pseudo-pretrained weights, no UrbanSARFloods fine-tuning yet)')
    print('='*60)

    val_loader = make_test_loader(cfg)
    loss_fn    = build_loss(device, flood_weight=cfg['flood_weight'])
    metrics    = SegmentationMetrics(n_classes=N_CLASSES, ignore_index=IGNORE_INDEX)
    results    = run_epoch(model, val_loader, loss_fn, None,
                           metrics, device, is_train=False)

    print(f'Zero-shot flood F1  = {results["flood_f1"]:.4f}')
    print(f'Zero-shot macro F1  = {results["macro_f1"]:.4f}')
    print(f'v9_aux baseline     = 0.1504 (ImageNet init, 7 cities, no pseudo)')

    out = Path(cfg['out_dir']) / 'stage1'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'zeroshot_results.txt', 'w') as f:
        for k, v in results.items():
            if isinstance(v, (float, int)):
                f.write(f'{k}: {v:.5f}\n')

    return results


# =============================================================================
# Stage 2: Fine-tune on UrbanSARFloods training split
# =============================================================================

def stage2_finetune(model: nn.Module, cfg: dict, device: torch.device) -> nn.Module:
    out = Path(cfg['out_dir']) / 'stage2'
    out.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*60)
    print('STAGE 2 -- Fine-tune on UrbanSARFloods (event-level split)')
    print(f'Train: {FINETUNE_TRAIN_EVENTS}')
    print(f'Test:  {FINETUNE_TEST_EVENTS}')
    print('='*60)

    train_loader = make_train_loader(cfg, augment=True)
    test_loader  = make_test_loader(cfg)

    # Discriminative LR: encoder 5x lower than decoder
    # Preserves pseudo-pretrained encoder features while decoder adapts quickly
    enc_lr = cfg['finetune_lr'] / 5
    dec_lr = cfg['finetune_lr']
    print(f'Discriminative LR: encoder={enc_lr:.2e}, decoder={dec_lr:.2e}')

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(),           'lr': enc_lr},
        {'params': model.decoder.parameters(),           'lr': dec_lr},
        {'params': model.segmentation_head.parameters(), 'lr': dec_lr},
    ], weight_decay=5e-4)
    scheduler = build_scheduler(optimizer, warmup=3, total=cfg['finetune_epochs'])
    loss_fn   = build_loss(device, flood_weight=cfg['flood_weight'])
    metrics   = SegmentationMetrics(n_classes=N_CLASSES, ignore_index=IGNORE_INDEX)

    fields = ['epoch', 'train_loss', 'test_flood_f1', 'test_macro_f1',
              'test_precision', 'test_recall']
    with open(out / 'log.csv', 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_f1    = 0.0
    patience   = 8
    no_improve = 0

    for epoch in range(1, cfg['finetune_epochs'] + 1):
        t0      = time.time()
        train_r = run_epoch(model, train_loader, loss_fn, optimizer,
                            metrics, device, is_train=True)
        test_r  = run_epoch(model, test_loader,  loss_fn, None,
                            metrics, device, is_train=False)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        # Extract precision/recall if available in metrics
        prec = test_r.get('flood_precision', float('nan'))
        rec  = test_r.get('flood_recall',    float('nan'))

        print(f'S2 {epoch:3d}/{cfg["finetune_epochs"]} | '
              f'loss={train_r["loss"]:.4f} | '
              f'flood_F1={test_r["flood_f1"]:.4f} | '
              f'P={prec:.3f} R={rec:.3f} | '
              f'lr={lr:.2e} | {time.time()-t0:.0f}s')

        write_log(out / 'log.csv', fields, {
            'epoch':          epoch,
            'train_loss':     round(train_r['loss'], 5),
            'test_flood_f1':  round(test_r['flood_f1'], 5),
            'test_macro_f1':  round(test_r['macro_f1'], 5),
            'test_precision': round(prec, 5) if not np.isnan(prec) else '',
            'test_recall':    round(rec, 5) if not np.isnan(rec) else '',
        })

        if test_r['flood_f1'] > best_f1:
            best_f1    = test_r['flood_f1']
            no_improve = 0
            save_ckpt(out / 'best.pt', model, optimizer, epoch, best_f1, cfg)
            print(f'  -> New best flood F1={best_f1:.4f}')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping at epoch {epoch}')
                break

    print(f'\nStage 2 done. Best flood F1={best_f1:.4f}')
    print(f'v9_aux baseline (no pseudo): 0.1504')
    return model


# =============================================================================
# Stage 3: Final evaluation + per-city breakdown
# =============================================================================

def stage3_eval(model: nn.Module, cfg: dict, device: torch.device):
    print('\n' + '='*60)
    print('STAGE 3 -- Final evaluation (best Stage 2 checkpoint)')
    print('='*60)

    ckpt_path = Path(cfg['out_dir']) / 'stage2' / 'best.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        print(f'Loaded best checkpoint: epoch {ckpt["epoch"]}, '
              f'F1={ckpt["metric"]:.4f}')
    else:
        print('No best.pt found -- using current model weights')

    # Per-city eval using existing per_city_eval.py
    print('\nRun per-city breakdown with:')
    print(f'  uv run python scripts/per_city_eval.py '
          f'--ckpt {ckpt_path} '
          f'--urban_sar_dir {cfg["urban_sar_dir"]}')


# =============================================================================
# Main
# =============================================================================

def main(cfg: dict):
    Path(cfg['out_dir']).mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f'Device: {device}')
    print(f'Config: out_dir={cfg["out_dir"]}, flood_weight={cfg["flood_weight"]}, '
          f'pseudo_epochs={cfg["pseudo_epochs"]}, finetune_epochs={cfg["finetune_epochs"]}')

    if cfg['finetune_only']:
        # Ablation baseline: ImageNet init, no pseudo pretraining
        print('\nFINETUNE-ONLY mode (ImageNet init, no pseudo pretraining)')
        print('This replicates the v9_aux baseline for direct comparison.')
        _, test_idx = make_urban_event_split(
            cfg['urban_sar_dir'],
            train_events=FINETUNE_TRAIN_EVENTS,
            test_events=FINETUNE_TEST_EVENTS,
        )
        probe_ds = FastUrbanSARFloods(cfg['urban_sar_dir'], test_idx[:1])
        n_ch     = probe_ds.n_channels
        model    = build_unet(
            n_channels=n_ch, n_classes=N_CLASSES,
            encoder_name=cfg['encoder'], encoder_weights='imagenet',
        ).to(device)

    elif cfg['skip_pseudo'] and cfg['pseudo_ckpt']:
        # Resume from existing pseudo-pretrain checkpoint
        print(f'\nLoading pseudo-pretrain checkpoint: {cfg["pseudo_ckpt"]}')
        ckpt  = torch.load(cfg['pseudo_ckpt'], map_location='cpu')
        _, test_idx = make_urban_event_split(
            cfg['urban_sar_dir'],
            train_events=FINETUNE_TRAIN_EVENTS,
            test_events=FINETUNE_TEST_EVENTS,
        )
        probe_ds = FastUrbanSARFloods(cfg['urban_sar_dir'], test_idx[:1])
        n_ch     = probe_ds.n_channels
        model    = build_unet(
            n_channels=n_ch, n_classes=N_CLASSES,
            encoder_name=cfg['encoder'], encoder_weights=None,
        ).to(device)
        model.load_state_dict(ckpt['model_state'])
        print(f'Loaded: epoch {ckpt["epoch"]}, zero-shot F1={ckpt["metric"]:.4f}')

    else:
        # Full pipeline: Stage 0 pseudo pretrain
        model = stage0_pseudo_pretrain(cfg, device)

    # Stage 1: zero-shot eval (pseudo-pretrained weights, no UrbanSARFloods FT)
    if not cfg['finetune_only']:
        stage1_zeroshot(model, cfg, device)

    # Stage 2: fine-tune on UrbanSARFloods
    model = stage2_finetune(model, cfg, device)

    # Stage 3: final eval
    stage3_eval(model, cfg, device)

    print('\n' + '='*60)
    print(f'PIPELINE COMPLETE -- outputs in {cfg["out_dir"]}')
    print('='*60)


def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f'--{k}', default=v, action='store_true')
        else:
            p.add_argument(f'--{k}', default=v,
                           type=type(v) if v != '' else str)
    return vars(p.parse_args())


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
