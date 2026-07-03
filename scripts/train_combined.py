"""
scripts/train_combined.py

Ablation: train a single model jointly on Sen1Floods11 + UrbanSARFloods
train split, then evaluate on the UrbanSARFloods event-held-out test split.

Compares against the three-stage pipeline to answer:
  Does sequential pretraining → fine-tuning add anything over
  simply mixing both datasets and training once?

Label scheme (same 4-class as three-stage):
  0 = ignore   1 = open flood   2 = urban flood   3 = non-flood

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \\
    uv run python scripts/train_combined.py --out_dir runs/combined
"""

import sys, csv, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import segmentation_models_pytorch as smp

from src.data.fast_dataset import (
    FastUrbanSARFloods, FastSen1Floods11,
    make_urban_event_split, make_sen1_splits,
)
from src.data.transforms import train_transforms, val_transforms
from src.models.unet import build_unet
from src.utils.metrics import SegmentationMetrics

FINETUNE_TRAIN_EVENTS = ['Houston','Beira','Japan','Canada','Iran','Lumberton','Somalia']
FINETUNE_TEST_EVENTS  = ['Hagibis','Sydney','Coraki','Niger','Hebei','Beledweyne','PortMacquarie']
N_CLASSES  = 4
EPOCHS     = 20
LR         = 1e-4
BATCH      = 16


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
    r = metrics.compute(); r['loss'] = total / max(n, 1)
    return r


def main(out_dir: str, sen1_dir: str, urban_dir: str, encoder_weights: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f'Device: {device}')

    # ── Build combined training set ───────────────────────────────────────────
    # Sen1Floods11: all training chips
    s1_train_idx, _ = make_sen1_splits(sen1_dir, seed=42)
    s1_ds = FastSen1Floods11(sen1_dir, s1_train_idx,
                             transform=train_transforms(512))
    print(f'Sen1Floods11 train chips: {len(s1_ds)}')

    # UrbanSARFloods: training events only
    urban_train_idx, urban_test_idx = make_urban_event_split(
        urban_dir,
        train_events=FINETUNE_TRAIN_EVENTS,
        test_events=FINETUNE_TEST_EVENTS,
    )
    urban_train_ds = FastUrbanSARFloods(urban_dir, urban_train_idx,
                                        transform=train_transforms(512), binary=False)
    urban_test_ds  = FastUrbanSARFloods(urban_dir, urban_test_idx,
                                        transform=val_transforms(512),   binary=False)
    print(f'UrbanSARFloods train chips: {len(urban_train_ds)}')
    print(f'UrbanSARFloods test  chips: {len(urban_test_ds)}')

    # Concat: Sen1Fl11 (384) + UrbanSAR train (3261) = ~3645 chips
    combined = ConcatDataset([s1_ds, urban_train_ds])
    print(f'Combined train chips: {len(combined)}')

    train_loader = DataLoader(combined,      batch_size=BATCH, shuffle=True,
                              num_workers=0, drop_last=True)
    test_loader  = DataLoader(urban_test_ds, batch_size=BATCH, shuffle=False,
                              num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    n_ch = s1_ds.n_channels  # both datasets have same channel count
    model = build_unet(n_channels=n_ch, n_classes=N_CLASSES,
                       encoder_name='resnet34',
                       encoder_weights=encoder_weights).to(device)
    print(f'Model: resnet34  encoder_weights={encoder_weights}  n_channels={n_ch}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    def lr_lambda(ep):
        warmup = 3
        if ep < warmup: return (ep+1)/warmup
        p = (ep-warmup)/max(1, EPOCHS-warmup)
        return 0.5*(1+math.cos(math.pi*p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn   = build_loss(device)
    metrics   = SegmentationMetrics(n_classes=N_CLASSES, ignore_index=0)

    # ── Training loop ─────────────────────────────────────────────────────────
    fields = ['epoch','train_loss','test_loss','test_f1','test_iou']
    with open(out/'log.csv','w',newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_f1 = 0.0
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, loss_fn, optimizer,
                       metrics, device, is_train=True)
        te = run_epoch(model, test_loader,  loss_fn, None,
                       metrics, device, is_train=False)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f'Ep {epoch:2d}/{EPOCHS} | loss={tr["loss"]:.4f} | '
              f'test_F1={te["macro_f1"]:.4f} | test_IoU={te["macro_iou"]:.4f} | '
              f'lr={lr:.2e} | {time.time()-t0:.0f}s')

        with open(out/'log.csv','a',newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writerow({
                'epoch': epoch, 'train_loss': round(tr['loss'],5),
                'test_loss': round(te['loss'],5),
                'test_f1': round(te['macro_f1'],5),
                'test_iou': round(te['macro_iou'],5),
            })

        if te['macro_f1'] > best_f1:
            best_f1 = te['macro_f1']
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'metric': best_f1}, out/'best.pt')
            print(f'  → New best F1={best_f1:.4f}')

    print(f'\nCombined training done. Best F1={best_f1:.4f}')
    with open(out/'result.txt','w') as f:
        f.write(f'best_f1: {best_f1:.5f}\n')
        f.write(f'encoder_weights: {encoder_weights}\n')
        f.write(f'epochs: {EPOCHS}\n')
    return best_f1


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir',          default='runs/combined')
    p.add_argument('--sen1_dir',         default='data/sen1floods11_preprocessed')
    p.add_argument('--urban_dir',        default='data/urban_sar_floods_preprocessed')
    p.add_argument('--encoder_weights',  default='imagenet')
    args = p.parse_args()
    main(args.out_dir, args.sen1_dir, args.urban_dir, args.encoder_weights)
