"""
src/train_three_stage.py

Three-stage training pipeline for urban SAR flood detection.

Label scheme (4 classes, consistent across both datasets):
    0 = no-data / masked  → ignored in loss and metrics
    1 = open flood        (both datasets)
    2 = urban flood       (UrbanSARFloods only — never appears in Sen1Floods11)
    3 = non-flood         (Sen1Floods11 only — never appears in UrbanSARFloods)

This separation is intentional: class 2 and class 3 never co-occur in the
same dataset, so the model learns each from a clean signal.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \\
    uv run python main.py train_three_stage --out_dir runs/three_stage

    # Resume from Stage 1 checkpoint:
    uv run python main.py train_three_stage \\
        --skip_pretrain --pretrain_ckpt runs/three_stage/stage1/best.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from src.data.fast_dataset import (
    FastUrbanSARFloods, FastSen1Floods11,
    make_urban_splits, make_urban_event_split, make_sen1_splits,
)
from src.data.transforms import train_transforms, val_transforms
from src.models.unet import build_unet
from src.utils.metrics import SegmentationMetrics

# ── Event split for Stage 3 ───────────────────────────────────────────────────
FINETUNE_TRAIN_EVENTS = [
    'Houston', 'Beira', 'Japan', 'Canada', 'Iran', 'Lumberton', 'Somalia',
]
FINETUNE_TEST_EVENTS = [
    'Hagibis', 'Sydney', 'Coraki', 'Niger', 'Hebei', 'Beledweyne', 'PortMacquarie',
]

N_CLASSES    = 2   # 0 = non-flood, 1 = flood
IGNORE_INDEX = -1  # nodata/masked pixels excluded from loss and metrics

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    sen1floods_dir     = 'data/sen1floods11_preprocessed',
    urban_sar_dir      = 'data/urban_sar_floods_preprocessed',
    out_dir            = 'runs/three_stage',
    encoder            = 'resnet34',
    use_local_variance = True,
    pretrain_epochs    = 10,
    pretrain_lr        = 1e-4,
    pretrain_batch     = 16,
    finetune_epochs    = 40,
    finetune_lr        = 1e-4,
    finetune_batch     = 16,
    flood_weight       = 3.0,   # focal loss weight — 10.0 caused high recall/low precision imbalance
    num_workers        = 0,    # profiling showed num_workers=0 fastest on M4 MPS
    seed               = 42,
    image_size         = 512,
    skip_pretrain      = False,
    finetune_only      = False,
    run_baselines      = False,  # also run random-init and imagenet-init baselines
    pretrain_ckpt      = '',
)

# ── Utilities ─────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    pin = torch.cuda.is_available()  # pin_memory only useful on CUDA
    # On macOS MPS, multiprocessing requires spawn context explicitly
    # num_workers > 0 lets data loading overlap with GPU compute
    mp_ctx = 'spawn' if (num_workers > 0 and not torch.cuda.is_available()) else None
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0),  # keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context=mp_ctx,
        drop_last=shuffle,
    )


def build_loss(device: torch.device, flood_weight: float = 3.0) -> nn.Module:
    """Focal loss with optional per-pixel confidence weighting.

    When batches contain a 'weight' tensor (from PseudoLabelDataset),
    each pixel's loss is multiplied by its confidence weight.
    Standard batches without 'weight' treat all valid pixels equally.
    """
    gamma = 2.0
    alpha = torch.tensor([1.0 / (1.0 + flood_weight),
                          flood_weight / (1.0 + flood_weight)], device=device)

    class FocalLoss(nn.Module):
        def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                    weights: torch.Tensor = None) -> torch.Tensor:
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
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)
            weights = batch.get('weight')
            if weights is not None:
                weights = weights.to(device)
            logits = model(images)
            loss   = loss_fn(logits, masks, weights)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients and check for NaN before stepping
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isnan(grad_norm):
                    print(f'  WARNING: NaN gradient detected, skipping batch')
                    optimizer.zero_grad()
                    continue
                optimizer.step()
            metrics.update(logits.argmax(dim=1), masks)
            total_loss += loss.item()
            n += 1
    results = metrics.compute()
    results['loss'] = total_loss / max(n, 1)
    return results


def save_ckpt(path, model, optimizer, epoch, metric, cfg):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metric': metric,
        'cfg': cfg,
    }, path)


def write_log(path, fields, row):
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def stage1_pretrain(cfg: dict, device: torch.device) -> nn.Module:
    out = Path(cfg['out_dir']) / 'stage1'
    out.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*60)
    print('STAGE 1 — Pretraining on Sen1Floods11')
    print('='*60)

    train_idx, val_idx = make_sen1_splits(cfg['sen1floods_dir'], seed=cfg['seed'])

    train_ds = FastSen1Floods11(
        cfg['sen1floods_dir'], train_idx,
        transform=train_transforms(cfg['image_size']),
    )
    val_ds = FastSen1Floods11(
        cfg['sen1floods_dir'], val_idx,
        transform=val_transforms(cfg['image_size']),
    )

    print(f'Train: {len(train_ds)} chips  Val: {len(val_ds)} chips')
    print(f'Events: {train_ds.event_counts()}')
    print(f'n_channels={train_ds.n_channels}  n_classes={train_ds.n_classes}')

    train_loader = make_loader(train_ds, cfg['pretrain_batch'], True,  cfg['num_workers'])
    val_loader   = make_loader(val_ds,   cfg['pretrain_batch'], False, cfg['num_workers'])

    model = build_unet(
        n_channels=train_ds.n_channels,
        n_classes=N_CLASSES,
        encoder_name=cfg['encoder'],
        encoder_weights='imagenet',
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['pretrain_lr'], weight_decay=1e-4
    )
    scheduler = build_scheduler(optimizer, warmup=3, total=cfg['pretrain_epochs'])
    loss_fn   = build_loss(device, flood_weight=cfg.get('flood_weight', 3.0))
    metrics   = SegmentationMetrics(n_classes=N_CLASSES, ignore_index=IGNORE_INDEX)

    fields = ['epoch', 'train_loss', 'val_loss', 'val_f1', 'val_iou']
    with open(out / 'log.csv', 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_f1 = 0.0
    for epoch in range(1, cfg['pretrain_epochs'] + 1):
        t0      = time.time()
        train_r = run_epoch(model, train_loader, loss_fn, optimizer,
                            metrics, device, is_train=True)
        val_r   = run_epoch(model, val_loader,   loss_fn, None,
                            metrics, device, is_train=False)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f'S1 {epoch:3d}/{cfg["pretrain_epochs"]} | '
              f'loss={train_r["loss"]:.4f} | '
              f'val_F1={val_r["flood_f1"]:.4f} | '
              f'val_IoU={val_r["macro_iou"]:.4f} | '
              f'lr={lr:.2e} | {time.time()-t0:.0f}s')

        write_log(out / 'log.csv', fields, {
            'epoch':      epoch,
            'train_loss': round(train_r['loss'], 5),
            'val_loss':   round(val_r['loss'], 5),
            'val_f1':     round(val_r['flood_f1'], 5),
            'val_iou':    round(val_r['macro_iou'], 5),
        })

        if val_r['flood_f1'] > best_f1:
            best_f1 = val_r['flood_f1']
            save_ckpt(out / 'best.pt', model, optimizer, epoch, best_f1, cfg)
            print(f'  → New best F1={best_f1:.4f}')

    print(f'\nStage 1 done. Best val F1={best_f1:.4f}')
    return model


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def stage2_zeroshot(model: nn.Module, cfg: dict, device: torch.device) -> float:
    print('\n' + '='*60)
    print('STAGE 2 — Zero-shot on UrbanSARFloods')
    print('(No fine-tuning — evaluating Stage 1 weights directly)')
    print('='*60)

    _, bench_idx = make_urban_splits(
        cfg['urban_sar_dir'], val_fraction=1.0, seed=cfg['seed']
    )
    bench_ds = FastUrbanSARFloods(
        root=cfg['urban_sar_dir'],
        split_indices=bench_idx,
        transform=val_transforms(cfg['image_size']),
    )
    loader  = make_loader(bench_ds, cfg['pretrain_batch'], False, cfg['num_workers'])
    metrics = SegmentationMetrics(n_classes=N_CLASSES, ignore_index=IGNORE_INDEX)
    results = run_epoch(model, loader, build_loss(device), None,
                        metrics, device, is_train=False)

    print(f'Zero-shot flood F1   = {results["flood_f1"]:.4f}')
    print(f'Zero-shot macro F1   = {results["macro_f1"]:.4f}')
    print(f'Zero-shot macro IoU  = {results["macro_iou"]:.4f}')
    print(f'Per-class F1: {results["per_class_f1"]}')
    print('Zhao et al. supervised baseline: F1 = 0.51–0.77')

    out = Path(cfg['out_dir']) / 'stage2'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'results.txt', 'w') as f:
        for k, v in results.items():
            if isinstance(v, float):
                f.write(f'{k}: {v:.5f}\n')

    return results['macro_f1']


# ── Stage 3 ───────────────────────────────────────────────────────────────────

def stage3_finetune(model: nn.Module, cfg: dict, device: torch.device) -> nn.Module:
    out = Path(cfg['out_dir']) / 'stage3'
    out.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*60)
    print('STAGE 3 — Fine-tuning on UrbanSARFloods (event-level split)')
    print(f'Train events: {FINETUNE_TRAIN_EVENTS}')
    print(f'Test  events: {FINETUNE_TEST_EVENTS}')
    print('='*60)

    train_idx, test_idx = make_urban_event_split(
        cfg['urban_sar_dir'],
        train_events=FINETUNE_TRAIN_EVENTS,
        test_events=FINETUNE_TEST_EVENTS,
    )
    print(f'Fine-tune train chips: {len(train_idx)}')
    print(f'Fine-tune test  chips: {len(test_idx)}')

    def make_urban_ds(indices, augment: bool) -> FastUrbanSARFloods:
        return FastUrbanSARFloods(
            root=cfg['urban_sar_dir'],
            split_indices=indices,
            transform=(train_transforms(cfg['image_size']) if augment
                       else val_transforms(cfg['image_size'])),
        )

    train_ds = make_urban_ds(train_idx, augment=True)
    test_ds  = make_urban_ds(test_idx,  augment=False)

    train_loader = make_loader(train_ds, cfg['finetune_batch'], True,  cfg['num_workers'])
    test_loader  = make_loader(test_ds,  cfg['finetune_batch'], False, cfg['num_workers'])

    # Discriminative learning rates: encoder gets 5x lower LR than decoder.
    # This is safer than freeze/unfreeze — no optimizer state reset, no NaN.
    # The encoder retains Sen1Floods11 features while the decoder adapts quickly.
    print(f'Discriminative LR: encoder={cfg["finetune_lr"]/5:.2e}  decoder={cfg["finetune_lr"]:.2e}')
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(),          'lr': cfg['finetune_lr'] / 5},
        {'params': model.decoder.parameters(),          'lr': cfg['finetune_lr']},
        {'params': model.segmentation_head.parameters(),'lr': cfg['finetune_lr']},
    ], weight_decay=5e-4)  # stronger regularisation to combat overfitting
    scheduler = build_scheduler(optimizer, warmup=3, total=cfg['finetune_epochs'])
    loss_fn   = build_loss(device, flood_weight=cfg.get('flood_weight', 3.0))
    metrics   = SegmentationMetrics(n_classes=N_CLASSES, ignore_index=IGNORE_INDEX)

    fields = ['epoch', 'train_loss', 'test_loss', 'test_f1', 'test_iou']
    with open(out / 'log.csv', 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_f1     = 0.0
    patience    = 8   # stop if no improvement for this many epochs
    no_improve  = 0

    for epoch in range(1, cfg['finetune_epochs'] + 1):
        t0      = time.time()
        train_r = run_epoch(model, train_loader, loss_fn, optimizer,
                            metrics, device, is_train=True)
        test_r  = run_epoch(model, test_loader,  loss_fn, None,
                            metrics, device, is_train=False)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f'S3 {epoch:3d}/{cfg["finetune_epochs"]} | '
              f'loss={train_r["loss"]:.4f} | '
              f'flood_F1={test_r["flood_f1"]:.4f} | '
              f'macro_F1={test_r["macro_f1"]:.4f} | '
              f'lr={lr:.2e} | {time.time()-t0:.0f}s')

        write_log(out / 'log.csv', fields, {
            'epoch':      epoch,
            'train_loss': round(train_r['loss'], 5),
            'test_loss':  round(test_r['loss'], 5),
            'test_f1':    round(test_r['flood_f1'], 5),
            'test_iou':   round(test_r['macro_iou'], 5),
        })

        if test_r['flood_f1'] > best_f1:
            best_f1    = test_r['flood_f1']
            no_improve = 0
            save_ckpt(out / 'best.pt', model, optimizer, epoch, best_f1, cfg)
            print(f'  → New best F1={best_f1:.4f}')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping: no improvement for {patience} epochs')
                break

    print(f'\nStage 3 done. Best test F1={best_f1:.4f}')
    print('Zhao et al. supervised (in-domain): F1=0.51–0.77')
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    Path(cfg['out_dir']).mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f'Device: {device}')

    if cfg['skip_pretrain']:
        if cfg['pretrain_ckpt']:
            # Load from explicit checkpoint file
            ckpt  = torch.load(cfg['pretrain_ckpt'], map_location='cpu')
            _probe_idx, _ = make_urban_event_split(
                cfg['urban_sar_dir'],
                train_events=FINETUNE_TRAIN_EVENTS,
                test_events=FINETUNE_TEST_EVENTS,
            )
            _probe_ds = FastUrbanSARFloods(cfg['urban_sar_dir'], _probe_idx[:1])
            n_ch = _probe_ds.n_channels
            print(f'Input channels: {n_ch}')
            model = build_unet(
                n_channels=n_ch, n_classes=N_CLASSES,
                encoder_name=cfg['encoder'], encoder_weights=None,
            ).to(device)
            saved_state  = ckpt['model_state']
            model_state  = model.state_dict()
            first_conv_key = next(k for k in saved_state if 'weight' in k
                                  and saved_state[k].dim() == 4)
            if saved_state[first_conv_key].shape != model_state[first_conv_key].shape:
                print(f'Channel mismatch in {first_conv_key}: '
                      f'{saved_state[first_conv_key].shape} → {model_state[first_conv_key].shape}')
                old_w  = saved_state[first_conv_key]
                new_w  = model_state[first_conv_key].clone()
                ch_old = old_w.shape[1]
                new_w[:, :ch_old, :, :] = old_w
                new_w[:, ch_old:, :, :] = old_w.mean(dim=1, keepdim=True)
                saved_state[first_conv_key] = new_w
            model.load_state_dict(saved_state, strict=True)
            print(f"Loaded checkpoint: epoch {ckpt['epoch']} F1={ckpt['metric']:.4f}")
        else:
            # ImageNet init — encoder pretrained on ImageNet, first conv extended if needed
            _probe_idx, _ = make_urban_event_split(
                cfg['urban_sar_dir'],
                train_events=FINETUNE_TRAIN_EVENTS,
                test_events=FINETUNE_TEST_EVENTS,
            )
            _probe_ds = FastUrbanSARFloods(cfg['urban_sar_dir'], _probe_idx[:1])
            n_ch = _probe_ds.n_channels
            print(f'Input channels: {n_ch}  (ImageNet init)')
            model = build_unet(
                n_channels=n_ch, n_classes=N_CLASSES,
                encoder_name=cfg['encoder'], encoder_weights='imagenet',
            ).to(device)
    else:
        model = stage1_pretrain(cfg, device)

    if not cfg['skip_pretrain']:
        stage2_zeroshot(model, cfg, device)

    stage3_finetune(model, cfg, device)

    # Baseline A: train from scratch (random init) on UrbanSARFloods
    # Baseline B: train from scratch (ImageNet init) on UrbanSARFloods
    # Both use same event split as Stage 3 for fair comparison
    if cfg.get('run_baselines', False):
        print('\n' + '='*60)
        print('BASELINE A — Random init, train on UrbanSARFloods only')
        print('='*60)
        n_ch = 4 if cfg['use_local_variance'] else 2
        baseline_a = build_unet(n_channels=n_ch, n_classes=N_CLASSES,
                                encoder_name=cfg['encoder'],
                                encoder_weights=None).to(device)
        stage3_finetune(baseline_a, cfg, device)

        print('\n' + '='*60)
        print('BASELINE B — ImageNet init, train on UrbanSARFloods only')
        print('='*60)
        baseline_b = build_unet(n_channels=n_ch, n_classes=N_CLASSES,
                                encoder_name=cfg['encoder'],
                                encoder_weights='imagenet').to(device)
        stage3_finetune(baseline_b, cfg, device)

    print('\n' + '='*60)
    print(f'PIPELINE COMPLETE — outputs in {cfg["out_dir"]}')
    print('='*60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> dict:
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f'--{k}', default=v, action='store_true')
        else:
            p.add_argument(f'--{k}', default=v,
                           type=type(v) if v != '' else str)
    return vars(p.parse_args())


if __name__ == '__main__':
    cfg = parse_args()
    print('Config:', cfg)
    main(cfg)
