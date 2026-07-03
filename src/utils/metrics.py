"""
Segmentation metrics: F1 (per-class and macro), IoU, precision, recall.

All functions operate on flat numpy arrays or torch tensors of integer labels.
They are designed for batch accumulation: call update() per batch, then compute()
at epoch end.

Usage:
    metrics = SegmentationMetrics(n_classes=4, ignore_index=0)
    for batch in loader:
        preds = model(batch["image"]).argmax(dim=1)
        metrics.update(preds, batch["mask"])
    results = metrics.compute()
    metrics.reset()
"""

from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates confusion matrix and computes per-class F1, IoU, precision, recall.

    Args:
        n_classes: Total number of classes (including background).
        ignore_index: Class index to exclude from all metrics (typically 0 = background).
    """

    def __init__(self, n_classes: int, ignore_index: int = 0):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self._conf = np.zeros((n_classes, n_classes), dtype=np.int64)

    def reset(self) -> None:
        self._conf[:] = 0

    def update(
        self,
        preds: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> None:
        """Accumulate predictions and targets into confusion matrix.

        Args:
            preds: Predicted class indices, shape (B, H, W) or (H, W).
            targets: Ground-truth class indices, same shape as preds.

        NOTE on ignore_index:
            ignore_index pixels are excluded from the confusion matrix
            entirely — they contribute neither TP/FP/FN to any class.
            In UrbanSARFloods, class 0 is no-data/background (ignore).
            In Sen1Floods11 binary mode, class 0 is non-flood (DO NOT ignore).
            Set ignore_index=None if class 0 should be evaluated.
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        preds   = preds.ravel().astype(np.int64)
        targets = targets.ravel().astype(np.int64)

        # Remove ignored pixels (no-data, cloud, etc.)
        if self.ignore_index is not None:
            valid   = targets != self.ignore_index
            preds   = preds[valid]
            targets = targets[valid]

        # Clip to valid range — prevents negative indices wrapping in np.add.at
        preds   = np.clip(preds,   0, self.n_classes - 1)
        targets = np.clip(targets, 0, self.n_classes - 1)

        np.add.at(self._conf, (targets, preds), 1)

    def compute(self) -> dict[str, float | np.ndarray]:
        """Return dict of metrics computed from accumulated confusion matrix."""
        conf = self._conf
        # Indices of classes to evaluate (skip ignore_index and background=0)
        eval_classes = [c for c in range(self.n_classes) if c != self.ignore_index]

        tp = np.diag(conf)
        fp = conf.sum(axis=0) - tp
        fn = conf.sum(axis=1) - tp

        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)

        # Macro averages over evaluated classes only
        macro_f1 = f1[eval_classes].mean()
        macro_iou = iou[eval_classes].mean()
        macro_precision = precision[eval_classes].mean()
        macro_recall = recall[eval_classes].mean()

        # Flood-specific F1 (class 1 only) — the metric that matters operationally.
        # Macro F1 can be misleading when a dominant non-flood class inflates the average.
        flood_f1 = float(f1[1]) if self.n_classes > 1 else float(macro_f1)
        # Flood-specific precision/recall (class 1 only). These are the correct
        # components of flood_f1 — NOT the macro averages. (macro_* mix in the
        # non-flood class and are the wrong thing to log as "flood precision".)
        flood_precision = float(precision[1]) if self.n_classes > 1 else float(macro_precision)
        flood_recall    = float(recall[1])    if self.n_classes > 1 else float(macro_recall)

        return {
            "macro_f1": float(macro_f1),
            "flood_f1": flood_f1,           # class 1 F1 only — use this for model selection
            "flood_precision": flood_precision,  # class 1 precision only
            "flood_recall": flood_recall,        # class 1 recall only
            "macro_iou": float(macro_iou),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "per_class_f1": f1,
            "per_class_iou": iou,
            "confusion_matrix": conf,
        }
