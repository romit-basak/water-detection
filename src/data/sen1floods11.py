"""
src/data/sen1floods11.py

PyTorch Dataset for Sen1Floods11 hand-labeled chips.

Directory layout (after gsutil download):
    data/sen1floods11/
        S1Hand/         ← Sentinel-1 chips: {event}_{chipid}_S1Hand.tif
        LabelHand/      ← Labels:           {event}_{chipid}_LabelHand.tif

S1 chips: 2-band float32 [VV, VH] in LINEAR scale (not dB, not normalized).
    Raw Sentinel-1 GRD values, range roughly [0, 0.5] for water, higher for land.
    We convert to dB (10 * log10) then normalize to [0, 1] using the same
    range as UrbanSARFloods bands 5-8 (-60 to +7 dB) for cross-dataset
    compatibility.

Label encoding:
    -1 = no data / cloud (treat as 0 — ignore in loss)
     0 = no water / non-flood
     1 = water / flood

    We remap to match UrbanSARFloods:
     0 = background/masked  (original -1 and 0)
     1 = flood              (original 1)

    NOTE: Sen1Floods11 is binary (flood/no-flood). UrbanSARFloods has
    class 2 (urban flood). We train with n_classes=2 (binary mode) when
    mixing datasets, or map Sen1Floods11 class 1 → UrbanSARFloods class 1
    (open flood) when training in 3-class mode.

Events in Sen1Floods11 hand-labeled subset (446 chips, 11 events):
    Bolivia, Canada, Denmark, Finland, Ghana, India,
    Mekong, Nigeria, Pakistan, Paraguay, USA (Nebraska)

Usage:
    from src.data.sen1floods11 import Sen1Floods11Dataset
    ds = Sen1Floods11Dataset('data/sen1floods11')
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# Sen1Floods11 S1 is in linear scale — convert to dB for consistency
# with UrbanSARFloods. Clip to valid range before log to avoid -inf.
LINEAR_CLIP_MIN = 1e-6   # avoid log(0)
LINEAR_CLIP_MAX = 1.0    # values above 1 are rare noise

# After dB conversion, normalize using same range as UrbanSARFloods
SAR_DB_MIN = -60.0
SAR_DB_MAX =   7.0


class Sen1Floods11Dataset(Dataset):
    """PyTorch Dataset for Sen1Floods11 hand-labeled Sentinel-1 chips.

    Args:
        root: Path to sen1floods11/ directory containing S1Hand/ and LabelHand/.
        split_indices: Indices into the full chip list. Use make_splits() below.
        transform: Albumentations transform (joint image + mask).
        use_local_variance: Append 5×5 local variance of VV and VH.
            Must match the setting used in UrbanSARFloodsDataset for
            cross-dataset compatibility.
        event_filter: If provided, only load chips from these event names.
    """

    def __init__(
        self,
        root: str | Path,
        split_indices: list[int] | None = None,
        transform: Optional[Callable] = None,
        use_local_variance: bool = True,
        event_filter: Optional[list[str]] = None,
    ):
        self.root              = Path(root)
        self.transform         = transform
        self.use_local_variance = use_local_variance

        self._all_pairs = self._discover_pairs(event_filter)
        if split_indices is None:
            self.pairs = self._all_pairs
        else:
            self.pairs = [self._all_pairs[i] for i in split_indices]

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_pairs(
        self,
        event_filter: Optional[list[str]] = None,
    ) -> list[tuple[Path, Path]]:
        s1_dir    = self.root / 'S1Hand'
        label_dir = self.root / 'LabelHand'

        pairs = []
        for s1_path in sorted(s1_dir.glob('*_S1Hand.tif')):
            # Derive label filename: replace _S1Hand.tif → _LabelHand.tif
            label_name = s1_path.name.replace('_S1Hand.tif', '_LabelHand.tif')
            label_path = label_dir / label_name
            if not label_path.exists():
                continue
            if event_filter is not None:
                event = s1_path.name.split('_')[0]
                if event not in event_filter:
                    continue
            pairs.append((s1_path, label_path))
        return pairs

    # ------------------------------------------------------------------
    # Reading + normalisation
    # ------------------------------------------------------------------

    def _read_s1(self, path: Path) -> np.ndarray:
        """Read S1 chip, convert linear → dB, normalize to [0, 1]."""
        with rasterio.open(path) as src:
            data = src.read([1, 2]).astype(np.float32)  # VV, VH

        # Replace NaN / negative values
        data = np.where(np.isfinite(data) & (data > 0), data, LINEAR_CLIP_MIN)
        data = np.clip(data, LINEAR_CLIP_MIN, LINEAR_CLIP_MAX)

        # Linear → dB
        data = 10.0 * np.log10(data)

        # Normalize to [0, 1] using UrbanSARFloods range
        data = np.clip(data, SAR_DB_MIN, SAR_DB_MAX)
        data = (data - SAR_DB_MIN) / (SAR_DB_MAX - SAR_DB_MIN)
        return data

    def _read_label(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            raw = src.read(1).astype(np.int64)

        # Unified 3-class scheme:
        #   0 = ignore (cloud/nodata: original -1)
        #   1 = flood  (original 1)
        #   2 = non-flood (original 0)
        label = np.where(raw == 1, 1, np.where(raw == 0, 2, 0)).astype(np.int64)
        return label

    def _local_variance(self, data: np.ndarray, kernel: int = 5) -> np.ndarray:
        """5x5 local variance per channel using PyTorch for speed."""
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(data).unsqueeze(0)  # (1, C, H, W)
        pad = kernel // 2
        t_pad = F.pad(t, (pad, pad, pad, pad), mode='reflect')
        mean    = F.avg_pool2d(t_pad, kernel, stride=1, padding=0)
        mean_sq = F.avg_pool2d(t_pad ** 2, kernel, stride=1, padding=0)
        var     = (mean_sq - mean ** 2).clamp(min=0)
        return var.squeeze(0).numpy()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        s1_path, label_path = self.pairs[idx]

        image = self._read_s1(s1_path)       # (2, H, W)
        mask  = self._read_label(label_path)  # (H, W)

        if self.use_local_variance:
            var   = self._local_variance(image, kernel=5)
            image = np.concatenate([image, var], axis=0)  # (4, H, W)

        if self.transform is not None:
            hwc   = np.transpose(image, (1, 2, 0))
            out   = self.transform(image=hwc, mask=mask.astype(np.float32))
            image = np.transpose(out['image'], (2, 0, 1))
            mask  = out['mask'].astype(np.int64)

        event = s1_path.name.split('_')[0]

        return {
            'image':     torch.from_numpy(image),
            'mask':      torch.from_numpy(mask),
            'chip_path': str(s1_path),
            'event':     event,
        }

    @property
    def n_channels(self) -> int:
        return 4 if self.use_local_variance else 2

    @property
    def n_classes(self) -> int:
        return 4  # 0=ignore, 1=open_flood, 2=urban_flood(unused), 3=non_flood

    def event_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s1_path, _ in self.pairs:
            event = s1_path.name.split('_')[0]
            counts[event] = counts.get(event, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def make_splits(
    root: str | Path,
    val_fraction: float = 0.15,
    seed: int = 42,
    event_filter: Optional[list[str]] = None,
) -> tuple[list[int], list[int]]:
    """Stratified train/val split by event name."""
    import random
    root = Path(root)
    rng  = random.Random(seed)

    s1_dir    = root / 'S1Hand'
    label_dir = root / 'LabelHand'

    all_pairs: list[tuple[Path, Path]] = []
    event_groups: dict[str, list[int]] = {}

    for s1_path in sorted(s1_dir.glob('*_S1Hand.tif')):
        label_path = label_dir / s1_path.name.replace('_S1Hand.tif',
                                                       '_LabelHand.tif')
        if not label_path.exists():
            continue
        event = s1_path.name.split('_')[0]
        if event_filter and event not in event_filter:
            continue
        idx = len(all_pairs)
        all_pairs.append((s1_path, label_path))
        event_groups.setdefault(event, []).append(idx)

    train_indices, val_indices = [], []
    for event, indices in event_groups.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val_indices.extend(shuffled[:n_val])
        train_indices.extend(shuffled[n_val:])

    return train_indices, val_indices
