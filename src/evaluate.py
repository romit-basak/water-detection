"""
Evaluation script — loads a saved checkpoint and reports full metrics.

Run from project root:
    python -m src.evaluate --checkpoint runs/baseline/best_model.pt

Outputs:
  - Per-class F1, IoU, precision, recall table (stdout)
  - Confusion matrix saved as PNG
  - Sample prediction grid saved as PNG (random chips from val set)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import UrbanSARFloodsDataset, make_splits, CLASS_NAMES
from src.data.transforms import val_transforms
from src.models.unet import build_unet
from src.utils.metrics import SegmentationMetrics


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(conf: np.ndarray, class_names: dict, save_path: Path) -> None:
    n = conf.shape[0]
    labels = [class_names.get(i, str(i)) for i in range(n)]
    norm = conf.astype(float) / (conf.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def plot_predictions(
    model: torch.nn.Module,
    dataset: UrbanSARFloodsDataset,
    device: torch.device,
    n_samples: int = 6,
    save_path: Path = Path("runs/predictions.png"),
) -> None:
    """Plot a grid of: VV intensity | ground truth | prediction."""
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"].numpy()

            logits = model(image)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

            vv = sample["image"][0].numpy()  # VV channel (first)

            axes[row, 0].imshow(vv, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title("VV intensity")
            axes[row, 1].imshow(mask, cmap="tab10", vmin=0, vmax=3)
            axes[row, 1].set_title("Ground truth")
            axes[row, 2].imshow(pred, cmap="tab10", vmin=0, vmax=3)
            axes[row, 2].set_title("Prediction")

            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Prediction grid saved: {save_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path: str, data_root: str, n_preview: int = 6) -> None:
    ckpt_path = Path(checkpoint_path)
    out_dir = ckpt_path.parent

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val F1={ckpt['val_f1']:.4f})")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Recreate val split with same seed as training
    _, val_idx = make_splits(data_root, val_fraction=cfg["val_fraction"], seed=cfg["seed"])

    val_ds = UrbanSARFloodsDataset(
        root=data_root,
        split_indices=val_idx,
        transform=val_transforms(cfg["image_size"]),
        use_change_features=cfg["use_change_features"],
        use_local_variance=cfg["use_local_variance"],
        binary=cfg["binary"],
    )

    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model = build_unet(
        n_channels=val_ds.n_channels,
        n_classes=val_ds.n_classes,
        encoder_name=cfg["encoder"],
        encoder_weights=None,  # loading from checkpoint
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    metrics = SegmentationMetrics(n_classes=val_ds.n_classes, ignore_index=0)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            preds = model(images).argmax(dim=1)
            metrics.update(preds, masks)

    results = metrics.compute()

    # --- Print table ---
    print("\n── Per-class metrics ─────────────────────────────────────────")
    print(f"{'Class':<20} {'F1':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8}")
    print("-" * 56)
    eval_classes = [c for c in range(val_ds.n_classes) if c != 0]
    for c in eval_classes:
        name = CLASS_NAMES.get(c, str(c))
        print(
            f"{name:<20} "
            f"{results['per_class_f1'][c]:>8.4f} "
            f"{results['per_class_iou'][c]:>8.4f}"
        )
    print("-" * 56)
    print(f"{'MACRO':>20} {results['macro_f1']:>8.4f} {results['macro_iou']:>8.4f} "
          f"{results['macro_precision']:>8.4f} {results['macro_recall']:>8.4f}")

    # --- Plots ---
    plot_confusion_matrix(results["confusion_matrix"], CLASS_NAMES, out_dir / "confusion_matrix.png")
    plot_predictions(model, val_ds, device, n_samples=n_preview, save_path=out_dir / "predictions.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--data_root", default="data/urban_sar_floods")
    p.add_argument("--n_preview", type=int, default=6)
    args = p.parse_args()
    evaluate(args.checkpoint, args.data_root, args.n_preview)
