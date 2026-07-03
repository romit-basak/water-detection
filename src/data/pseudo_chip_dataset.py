"""
src/data/pseudo_chip_dataset.py

Dataset for new-event pseudo-labeled chips produced by
scripts/precompute_pseudo_chips.py.

Unlike PseudoLabelDataset (which overlays confidence maps on existing
UrbanSARFloods chips), this class is self-contained: the .npy GT files
ARE the confidence maps (float32, -1 to 1), written by the chipping script.

Label conversion at __getitem__ time:
  conf >= FLOOD_THRESH    -> label 1 (flood),     weight = conf
  conf <= NONFLOOD_THRESH -> label 0 (non-flood), weight = 1 - conf
  otherwise               -> label -1 (ignore)
  conf < 0                -> label -1 (ignore, shadow/cloud/nodata)

This deferred conversion allows threshold experimentation without re-chipping.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

FLOOD_THRESH    = 0.65
NONFLOOD_THRESH = 0.35


class PseudoChipDataset(Dataset):
    """
    Loads pseudo-labeled chips from data/pseudo_chips/{event}/SAR/*.npy.

    Args:
        root:       Path to pseudo_chips root (contains one dir per event).
        events:     List of event names to include. None = all available.
        transform:  Albumentations transform (same interface as FastUrbanSARFloods).
        flood_thresh:    Confidence threshold above which pixel is flood.
        nonflood_thresh: Confidence threshold below which pixel is non-flood.
        seed:       Random seed for reproducible per-epoch behaviour.
    """

    def __init__(
        self,
        root: str | Path,
        events: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        flood_thresh: float = FLOOD_THRESH,
        nonflood_thresh: float = NONFLOOD_THRESH,
    ):
        self.root        = Path(root)
        self.transform   = transform
        self.flood_thresh = flood_thresh
        self.nf_thresh   = nonflood_thresh
        self.pairs       = self._discover(events)

        # Stats for sanity check
        print(f'PseudoChipDataset: {len(self.pairs)} chips from '
              f'{len(set(p[0].parts[-3] for p in self.pairs))} events')

    def _discover(self, events: Optional[list[str]]) -> list[tuple[Path, Path]]:
        pairs = []
        event_dirs = sorted(self.root.iterdir()) if self.root.exists() else []
        for event_dir in event_dirs:
            if not event_dir.is_dir():
                continue
            if events is not None and event_dir.name not in events:
                continue
            sar_dir = event_dir / 'SAR'
            gt_dir  = event_dir / 'GT'
            if not sar_dir.exists():
                continue
            for sar_path in sorted(sar_dir.glob('*_SAR.npy')):
                gt_path = gt_dir / sar_path.name.replace('_SAR.npy', '_GT.npy')
                if gt_path.exists():
                    pairs.append((sar_path, gt_path))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        sar_path, gt_path = self.pairs[idx]

        image = np.load(sar_path).astype(np.float32)   # (8, 512, 512)
        conf  = np.load(gt_path).astype(np.float32)    # (512, 512) raw confidence

        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        # Convert confidence to binary labels + per-pixel weights
        mask   = np.full((512, 512), -1, dtype=np.int64)  # default: ignore
        mask   = np.where(conf >= self.flood_thresh,  1, mask)
        mask   = np.where(conf <= self.nf_thresh,     0, mask)
        mask   = np.where(conf < 0,                  -1, mask)  # shadow/cloud → ignore

        weight = np.where(
            conf >= self.flood_thresh,
            conf.astype(np.float32),
            np.where(
                conf <= self.nf_thresh,
                (1.0 - conf).astype(np.float32),
                np.zeros_like(conf),
            )
        ).astype(np.float32)

        if self.transform is not None:
            hwc   = np.transpose(image, (1, 2, 0))
            out   = self.transform(image=hwc, mask=mask.astype(np.float32))
            image = np.transpose(out['image'], (2, 0, 1))
            mask  = out['mask'].astype(np.int64)

        return {
            'image':     torch.from_numpy(image),
            'mask':      torch.from_numpy(mask),
            'weight':    torch.from_numpy(weight),
            'chip_path': str(sar_path),
            'has_conf':  True,
        }

    @property
    def n_channels(self) -> int:
        return int(np.load(self.pairs[0][0]).shape[0])

    def flood_chip_count(self) -> int:
        """Count chips that have at least one flood pixel. Slow — for diagnostics only."""
        n = 0
        for _, gt_path in self.pairs:
            conf = np.load(gt_path)
            if (conf >= self.flood_thresh).any():
                n += 1
        return n
