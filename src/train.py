"""
Training script for SAR urban flood segmentation.

Run from project root:
    python -m src.train

Key design decisions:
  - Loss: Dice + CrossEntropy combined. Dice handles class imbalance (flood pixels
    are rare); CE stabilizes training and provides per-pixel calibration.
  - Optimizer: AdamW with cosine LR schedule + linear warmup.
  - Checkpointing: saves best model by validation macro F1 (not loss).
  - Logging: plain stdout + a CSV log file for easy plotting.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from src.data.dataset import UrbanSARFloodsDataset, make_splits
from src.data.transforms import train_transforms, val_transforms
from src.models.unet import build_unet
from src.utils.metrics import SegmentationMetrics


# ---------------------------------------------------------------------------
# Config defaults (override via CLI flags)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    data_root="data/urban_sar_floods",
    out_dir="runs/baseline",
    # Model
    encoder="resnet34",
    use_change_features=True,
    use_local_variance=True,
    binary=False,
    # Training
    epochs=50,
    batch_size=8,
    lr=3e-4,
    weight_decay=1e-4,
    warmup_epochs=3,
    image_size=512,
    num_workers=4,
    val_fraction=0.15,
    seed=42,
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def build_loss(n_classes: int, binary: bool, device: torch.device) -> nn.Module:
    """Dice + CrossEntropy combined loss. Ignores class 0 (background/no-data)."""
    # smp losses handle ignore_index and multi-class cleanly
    mode = "binary" if binary else "multiclass"
    dice = smp.losses.DiceLoss(mode=mode, ignore_index=0)
    ce   = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=0)

    class CombinedLoss(nn.Module):
        def forward(self, logits, targets):
            return dice(logits, targets) + ce(logits, targets)

    return CombinedLoss().to(device)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        import math
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    metrics: SegmentationMetrics,
    device: torch.device,
    is_train: bool,
) -> dict:
    model.train(is_train)
    metrics.reset()
    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            images = batch["image"].to(device)   # (B, C, H, W)
            masks = batch["mask"].to(device)     # (B, H, W)

            logits = model(images)               # (B, n_classes, H, W)
            loss = loss_fn(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            preds = logits.argmax(dim=1)
            metrics.update(preds, masks)
            total_loss += loss.item()
            n_batches += 1

    results = metrics.compute()
    results["loss"] = total_loss / max(n_batches, 1)
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    train_idx, val_idx = make_splits(
        cfg["data_root"],
        val_fraction=cfg["val_fraction"],
        seed=cfg["seed"],
    )

    train_ds = UrbanSARFloodsDataset(
        root=cfg["data_root"],
        split_indices=train_idx,
        transform=train_transforms(cfg["image_size"]),
        use_change_features=cfg["use_change_features"],
        use_local_variance=cfg["use_local_variance"],
        binary=cfg["binary"],
    )
    val_ds = UrbanSARFloodsDataset(
        root=cfg["data_root"],
        split_indices=val_idx,
        transform=val_transforms(cfg["image_size"]),
        use_change_features=cfg["use_change_features"],
        use_local_variance=cfg["use_local_variance"],
        binary=cfg["binary"],
    )

    print(f"Train: {len(train_ds)} chips  |  Val: {len(val_ds)} chips")
    print(f"Input channels: {train_ds.n_channels}  |  Classes: {train_ds.n_classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # --- Model ---
    model = build_unet(
        n_channels=train_ds.n_channels,
        n_classes=train_ds.n_classes,
        encoder_name=cfg["encoder"],
        encoder_weights="imagenet",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    # --- Training setup ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg["warmup_epochs"], cfg["epochs"])
    loss_fn = build_loss(train_ds.n_classes, cfg["binary"], device)
    metrics = SegmentationMetrics(n_classes=train_ds.n_classes, ignore_index=0)

    # --- CSV log ---
    log_path = out_dir / "train_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_macro_iou", "lr"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_f1 = 0.0
    best_epoch = 0

    # --- Epoch loop ---
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_results = run_epoch(
            model, train_loader, loss_fn, optimizer, metrics, device, is_train=True
        )
        val_results = run_epoch(
            model, val_loader, loss_fn, None, metrics, device, is_train=False
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"train_loss={train_results['loss']:.4f} | "
            f"val_loss={val_results['loss']:.4f} | "
            f"val_F1={val_results['macro_f1']:.4f} | "
            f"val_IoU={val_results['macro_iou']:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.0f}s"
        )

        # Log CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow({
                "epoch": epoch,
                "train_loss": round(train_results["loss"], 5),
                "val_loss": round(val_results["loss"], 5),
                "val_macro_f1": round(val_results["macro_f1"], 5),
                "val_macro_iou": round(val_results["macro_iou"], 5),
                "lr": round(current_lr, 8),
            })

        # Checkpoint
        val_f1 = val_results["macro_f1"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "cfg": cfg,
                },
                out_dir / "best_model.pt",
            )
            print(f"  → New best F1={best_f1:.4f} saved.")

    print(f"\nTraining complete. Best val F1={best_f1:.4f} at epoch {best_epoch}.")
    print(f"Logs: {log_path}  |  Checkpoint: {out_dir / 'best_model.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> dict:
    p = argparse.ArgumentParser(description="Train SAR flood segmentation model")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", default=v, action="store_true")
        else:
            p.add_argument(f"--{k}", default=v, type=type(v))
    args = vars(p.parse_args())
    return args


if __name__ == "__main__":
    cfg = parse_args()
    print("Config:", cfg)
    train(cfg)
