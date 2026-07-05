"""
src/data/fast_dataset.py

Fast dataset classes that load from precomputed .npy chips.
Variance is pre-baked — no per-epoch recomputation.
np.load on float16 .npy is ~10x faster than rasterio + variance computation.

Use after running scripts/precompute_chips.py.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

FOLDER_CLASS = {'01_NF': 0, '02_FO': 1, '03_FU': 2}


class FastUrbanSARFloods(Dataset):
    """Loads pre-baked 4-channel float16 chips from .npy files."""

    def __init__(
        self,
        root: str | Path,
        split_indices: list[int],
        transform: Optional[Callable] = None,
    ):
        self.root      = Path(root)
        self.transform = transform
        self._all      = self._discover()
        self.pairs     = [self._all[i] for i in split_indices]

    def _discover(self) -> list[tuple[Path, Path]]:
        pairs = []
        for folder in sorted(FOLDER_CLASS.keys()):
            sar_dir = self.root / folder / 'SAR'
            gt_dir  = self.root / folder / 'GT'
            if not sar_dir.exists():
                continue
            for p in sorted(sar_dir.glob('*.npy')):
                gt = gt_dir / p.name.replace('_SAR.npy', '_GT.npy')
                if gt.exists():
                    pairs.append((p, gt))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        sar_path, gt_path = self.pairs[idx]
        image = np.load(sar_path).astype(np.float32)   # (C, H, W)
        mask  = np.load(gt_path).astype(np.int64)       # (H, W)

        # Replace any NaN/Inf
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        # Binary labels: 0 = non-flood, 1 = flood, -1 = ignore.
        # Stored values on disk (verified):
        #   01_NF chips: all 0s (confirmed non-flood background)
        #   02_FO/03_FU: 0=unlabeled bg, 1=open flood, 2=urban flood
        folder = sar_path.parts[-3]
        if folder == '01_NF':
            # All zeros on disk = confirmed non-flood. Keep as 0.
            pass
        else:
            # 0=unlabeled background → ignore (-1)
            # 1=open flood → flood (1), 2=urban flood → flood (1)
            mask = np.where(mask == 0, -1,
                   np.where(mask == 2,  1, mask)).astype(np.int64)

        if self.transform is not None:
            hwc   = np.transpose(image, (1, 2, 0))
            out   = self.transform(image=hwc, mask=mask.astype(np.float32))
            image = np.transpose(out['image'], (2, 0, 1))
            mask  = out['mask'].astype(np.int64)

        return {
            'image':     torch.from_numpy(image),
            'mask':      torch.from_numpy(mask),
            'chip_path': str(sar_path),
        }

    @property
    def n_channels(self) -> int:
        # Infer from first chip — handles 4-channel (SAR) and 7-channel (SAR+aux)
        return int(np.load(self.pairs[0][0]).shape[0])


class FastSen1Floods11(Dataset):
    """Loads pre-baked 4-channel float16 chips from .npy files."""

    def __init__(
        self,
        root: str | Path,
        split_indices: list[int],
        transform: Optional[Callable] = None,
        event_filter: Optional[list[str]] = None,
    ):
        self.root        = Path(root)
        self.transform   = transform
        self._all        = self._discover(event_filter)
        self.pairs       = [self._all[i] for i in split_indices]

    def _discover(self, event_filter):
        s1_dir  = self.root / 'S1Hand'
        lbl_dir = self.root / 'LabelHand'
        pairs   = []
        for p in sorted(s1_dir.glob('*_S1Hand.npy')):
            lbl = lbl_dir / p.name.replace('_S1Hand.npy', '_LabelHand.npy')
            if not lbl.exists():
                continue
            if event_filter:
                event = p.name.split('_')[0]
                if event not in event_filter:
                    continue
            pairs.append((p, lbl))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        s1_path, lbl_path = self.pairs[idx]
        image = np.load(s1_path).astype(np.float32)    # (C, H, W)
        mask  = np.load(lbl_path).astype(np.int64)     # (H, W)

        # Replace any NaN/Inf that survived preprocessing
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        # Binary labels: 0 = non-flood, 1 = flood, -1 = ignore.
        # Stored values on disk (verified): 0=nodata, 1=flood, 3=non-flood
        # 0 (nodata/ignore) → -1
        # 1 (flood)         →  1
        # 3 (non-flood)     →  0
        mask = np.where(mask == 0, -1,
               np.where(mask == 3,  0, mask)).astype(np.int64)

        if self.transform is not None:
            hwc   = np.transpose(image, (1, 2, 0))
            out   = self.transform(image=hwc, mask=mask.astype(np.float32))
            image = np.transpose(out['image'], (2, 0, 1))
            mask  = out['mask'].astype(np.int64)

        return {
            'image':     torch.from_numpy(image),
            'mask':      torch.from_numpy(mask),
            'chip_path': str(s1_path),
            'event':     s1_path.name.split('_')[0],
        }

    @property
    def n_channels(self) -> int:
        # Infer from first chip rather than hardcoding
        # Handles both 4-channel (SAR only) and 7-channel (SAR + aux) chips
        p = self.pairs[0][0]
        return int(np.load(p).shape[0])

    @property
    def n_classes(self) -> int:
        return 2  # 0=non-flood, 1=flood (binary)

    def event_counts(self):
        counts: dict[str, int] = {}
        for p, _ in self.pairs:
            e = p.name.split('_')[0]
            counts[e] = counts.get(e, 0) + 1
        return counts


def make_urban_splits(
    root: str | Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    root = Path(root)
    rng  = random.Random(seed)
    all_pairs: list[tuple[Path, Path]] = []
    groups: dict[str, list[int]] = {}

    for folder in sorted(FOLDER_CLASS.keys()):
        sar_dir = root / folder / 'SAR'
        gt_dir  = root / folder / 'GT'
        if not sar_dir.exists():
            continue
        idxs = []
        for p in sorted(sar_dir.glob('*.npy')):
            gt = gt_dir / p.name.replace('_SAR.npy', '_GT.npy')
            if gt.exists():
                idxs.append(len(all_pairs))
                all_pairs.append((p, gt))
        groups[folder] = idxs

    train, val = [], []
    for folder, idxs in groups.items():
        sh = idxs[:]
        rng.shuffle(sh)
        if val_fraction >= 1.0:
            val.extend(sh)
        else:
            n = max(1, int(len(sh) * val_fraction))
            val.extend(sh[:n])
            train.extend(sh[n:])
    return train, val


def make_train_val_split(
    train_idx: list[int],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Chip-level, seeded split of an already-chosen training index list into
    (train_sub, val) — used for LEAKAGE-FREE checkpoint selection.

    Splits the 7 event-held-out training cities' chips randomly so all 7 events
    remain represented in training (preserves the "7 cities" framing). NOTE: the
    resulting val set is IN-DISTRIBUTION (same events as train), so it selects
    for in-distribution fit and only loosely tracks cross-city test F1 — that is
    exactly why the study also reports mean-of-final-epochs and the peak-over-test
    protocol alongside this one. Deterministic given `seed`.
    """
    rng = random.Random(seed)
    sh  = list(train_idx)
    rng.shuffle(sh)
    n_val = max(1, int(len(sh) * val_frac))
    val   = sorted(sh[:n_val])
    train = sorted(sh[n_val:])
    return train, val


def make_urban_event_split(
    root: str | Path,
    train_events: list[str],
    test_events: list[str],
) -> tuple[list[int], list[int]]:
    root = Path(root)
    all_pairs: list[tuple[Path, Path]] = []
    train_idx, test_idx = [], []
    unassigned = []

    for folder in sorted(FOLDER_CLASS.keys()):
        sar_dir = root / folder / 'SAR'
        gt_dir  = root / folder / 'GT'
        if not sar_dir.exists():
            continue
        for p in sorted(sar_dir.glob('*.npy')):
            gt = gt_dir / p.name.replace('_SAR.npy', '_GT.npy')
            if not gt.exists():
                continue
            idx = len(all_pairs)
            all_pairs.append((p, gt))
            parts      = p.stem.split('_')
            event_name = parts[1] if len(parts) > 1 else 'unknown'
            if any(e.lower() in event_name.lower() for e in train_events):
                train_idx.append(idx)
            elif any(e.lower() in event_name.lower() for e in test_events):
                test_idx.append(idx)
            else:
                unassigned.append(event_name)

    if unassigned:
        print(f'Warning: {len(unassigned)} chips unassigned: {sorted(set(unassigned))}')
    return train_idx, test_idx


def make_sen1_splits(
    root: str | Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    root = Path(root)
    rng  = random.Random(seed)
    s1_dir  = root / 'S1Hand'
    lbl_dir = root / 'LabelHand'
    all_pairs, groups = [], {}

    for p in sorted(s1_dir.glob('*_S1Hand.npy')):
        lbl = lbl_dir / p.name.replace('_S1Hand.npy', '_LabelHand.npy')
        if not lbl.exists():
            continue
        event = p.name.split('_')[0]
        idx   = len(all_pairs)
        all_pairs.append((p, lbl))
        groups.setdefault(event, []).append(idx)

    train, val = [], []
    for event, idxs in groups.items():
        sh = idxs[:]
        rng.shuffle(sh)
        n = max(1, int(len(sh) * val_fraction))
        val.extend(sh[:n])
        train.extend(sh[n:])
    return train, val
