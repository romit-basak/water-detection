"""
UrbanSARFloods dataset loader.

Directory layout (after extracting urban_sar_floods.tar):
    data/urban_sar_floods/
        01_NF/          <- Non-Flooded (negative examples only)
            SAR/        <- 8-band GeoTIFF chips (*_SAR.tif)
            GT/         <- label masks (*_GT.tif)
        02_FO/          <- Flooded Open
            SAR/
            GT/
        03_FU/          <- Flooded Urban
            SAR/
            GT/

Band layout (confirmed from data inspection):
    Band 1 (idx 0): VV during-event,  pre-normalized [0, 1]
    Band 2 (idx 1): VH during-event,  pre-normalized [0, 1]
    Band 3 (idx 2): VV pre-event,     pre-normalized [0, 1]
    Band 4 (idx 3): VH pre-event,     pre-normalized [0, 1]
    Band 5 (idx 4): VV during-event,  raw dB  (duplicate of band 1)
    Band 6 (idx 5): VH during-event,  raw dB  (duplicate of band 2)
    Band 7 (idx 6): VV pre-event,     raw dB  (duplicate of band 3)
    Band 8 (idx 7): VH pre-event,     raw dB  (duplicate of band 4)

We use bands 1-4 (pre-normalized). No manual dB normalization needed.

Label encoding (confirmed from data inspection):
    0: background / no-data  (ignore in loss)
    1: flooded open area
    2: flooded urban area

    01_NF chips: masks contain only 0 (pure negative examples —
                 no positive flood label, but the SAR texture teaches
                 the model what non-flooded urban/land looks like)
    02_FO chips: masks contain 0 and 1
    03_FU chips: masks contain 0, 1, and 2

Input modes (controlled by constructor flags):
    use_change_features=True  (default): use all 4 normalized bands
                               [VV_during, VH_during, VV_pre, VH_pre]
    use_change_features=False: use only during-event bands [VV, VH]
                               — pure single-image inference mode,
                               harder problem, more novel contribution
    use_local_variance=True   (default): append 5x5 local variance of
                               VV_during and VH_during as extra channels

Usage:
    from src.data.dataset import UrbanSARFloodsDataset, make_splits
    train_idx, val_idx = make_splits("data/urban_sar_floods")
    ds = UrbanSARFloodsDataset("data/urban_sar_floods", train_idx)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    0: "background",
    1: "open_flooded",
    2: "urban_flooded",
}

# Number of semantic classes including background
N_CLASSES = 3

FOLDER_CLASS = {
    "01_NF": 0,   # no positive label — pure negative chips
    "02_FO": 1,   # open flooded
    "03_FU": 2,   # urban flooded
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UrbanSARFloodsDataset(Dataset):
    """PyTorch Dataset for UrbanSARFloods GeoTIFF chips.

    Args:
        root: Path to extracted urban_sar_floods/ directory.
        split_indices: List of integer indices into the full chip list.
            Use make_splits() to get train/val index lists.
        transform: Albumentations transform (applied to image + mask jointly).
        use_change_features: If True (default), use all 4 pre-normalized bands
            [VV_during, VH_during, VV_pre, VH_pre]. If False, use only the
            2 during-event bands — single-image inference mode.
        use_local_variance: If True (default), append 5x5 local variance of
            VV_during and VH_during as extra channels. This is the core
            shadow-disambiguation feature: shadow pixels have near-zero
            variance; water pixels have non-zero speckle variance.
        binary: If True, collapse to binary flood / no-flood.
            Classes 1 and 2 become 1; class 0 stays 0.
    """

    def __init__(
        self,
        root: str | Path,
        split_indices: list[int],
        transform: Optional[Callable] = None,
        use_change_features: bool = True,
        use_local_variance: bool = True,
        binary: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform
        self.use_change_features = use_change_features
        self.use_local_variance = use_local_variance
        self.binary = binary

        self._all_pairs = self._discover_pairs()
        self.pairs = [self._all_pairs[i] for i in split_indices]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_pairs(self) -> list[tuple[Path, Path]]:
        pairs = []
        for folder_name in sorted(FOLDER_CLASS.keys()):
            sar_folder = self.root / folder_name / "SAR"
            gt_folder  = self.root / folder_name / "GT"
            if not sar_folder.exists():
                continue
            for chip_path in sorted(sar_folder.glob("*_SAR.tif")):
                mask_path = gt_folder / chip_path.name.replace("_SAR.tif", "_GT.tif")
                if mask_path.exists():
                    pairs.append((chip_path, mask_path))
        return pairs

    def _read_chip(self, chip_path: Path) -> np.ndarray:
        """Read SAR chip. Returns float32 array (C, H, W), values in [0, 1]."""
        with rasterio.open(chip_path) as src:
            # Bands 1-4 are pre-normalized [0,1]. Read as 0-indexed: 0-3.
            if self.use_change_features:
                bands = [1, 2, 3, 4]   # VV_during, VH_during, VV_pre, VH_pre
            else:
                bands = [1, 2]         # VV_during, VH_during only
            data = src.read(bands).astype(np.float32)  # (C, H, W)
        # Replace NaN/Inf with 0.0 — NaN propagates through the network
        # and corrupts gradients for the entire batch.
        # NaN in UrbanSARFloods chips occurs at SAR swath edges.
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(data, 0.0, 1.0)

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

    def _read_mask(self, mask_path: Path) -> np.ndarray:
        with rasterio.open(mask_path) as src:
            return src.read(1).astype(np.int64)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        chip_path, mask_path = self.pairs[idx]

        image = self._read_chip(chip_path)    # (C, H, W) float32
        mask  = self._read_mask(mask_path)    # (H, W) int64

        # Append local variance of during-event channels only (first 2)
        if self.use_local_variance:
            var = self._local_variance(image[:2], kernel=5)
            image = np.concatenate([image, var], axis=0)

        # Binary: any flood (1 or 2) → 1, background → 0
        if self.binary:
            mask = (mask >= 1).astype(np.int64)

        # Unified 3-class scheme (matches Sen1Floods11 remapping):
        #   0 = ignore/no-data  (background stays 0)
        #   1 = flood           (classes 1+2 → 1)
        #   2 = non-flood       (01_NF chips: background pixels that ARE
        #                        non-flood get remapped to 2 using folder info)
        # NOTE: UrbanSARFloods 01_NF masks are all 0 (no positive label).
        # We can't reliably generate class 2 from them without the folder tag.
        # For now, binary=True (flood vs ignore) is the safe cross-dataset option.
        # The metrics ignore class 0, so 01_NF chips still penalise false positives.

        if self.transform is not None:
            image_hwc = np.transpose(image, (1, 2, 0))
            out = self.transform(image=image_hwc, mask=mask.astype(np.float32))
            image = np.transpose(out["image"], (2, 0, 1))
            mask  = out["mask"].astype(np.int64)

        return {
            "image":     torch.from_numpy(image),   # (C, H, W) float32
            "mask":      torch.from_numpy(mask),    # (H, W) int64
            "chip_path": str(chip_path),
        }

    @property
    def n_channels(self) -> int:
        base = 4 if self.use_change_features else 2
        if self.use_local_variance:
            base += 2   # variance of VV_during and VH_during
        return base

    @property
    def n_classes(self) -> int:
        return 2 if self.binary else N_CLASSES


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def make_splits(
    root: str | Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Stratified train/val split by folder.

    Returns (train_indices, val_indices) as flat index lists into the
    full chip list discovered by _discover_pairs().
    """
    root = Path(root)
    rng  = random.Random(seed)

    all_pairs: list[tuple[Path, Path]] = []
    class_groups: dict[str, list[int]] = {}

    for folder_name in sorted(FOLDER_CLASS.keys()):
        sar_folder = root / folder_name / "SAR"
        gt_folder  = root / folder_name / "GT"
        if not sar_folder.exists():
            continue
        indices = []
        for chip_path in sorted(sar_folder.glob("*_SAR.tif")):
            mask_path = gt_folder / chip_path.name.replace("_SAR.tif", "_GT.tif")
            if mask_path.exists():
                indices.append(len(all_pairs))
                all_pairs.append((chip_path, mask_path))
        class_groups[folder_name] = indices

    train_indices, val_indices = [], []
    for folder_name, indices in class_groups.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        if val_fraction >= 1.0:
            # Use entire dataset as val (benchmark-only mode)
            val_indices.extend(shuffled)
        else:
            n_val = max(1, int(len(shuffled) * val_fraction))
            val_indices.extend(shuffled[:n_val])
            train_indices.extend(shuffled[n_val:])

    return train_indices, val_indices


def make_event_split(
    root: str | Path,
    train_events: list[str],
    test_events: list[str],
) -> tuple[list[int], list[int]]:
    """Split UrbanSARFloods by flood event name rather than randomly.

    This is the correct split for fine-tuning evaluation — the model
    must generalize to cities it has never seen, not just chips from
    the same city as training.

    Event names are embedded in chip filenames:
        20170830_Houston_ID_27_24_SAR.tif  → event = 'Houston'
        20190320_Beira_ID_0_25_SAR.tif    → event = 'Beira'

    Args:
        root: Path to urban_sar_floods/ directory.
        train_events: Event names to include in training split.
        test_events:  Event names to include in test/val split.

    Returns:
        (train_indices, test_indices)

    Example — fine-tune on Houston+Beira+Japan, test on Hagibis+Sydney:
        train_idx, test_idx = make_event_split(
            'data/urban_sar_floods',
            train_events=['Houston', 'Beira', 'Japan'],
            test_events=['Hagibis', 'Sydney', 'Hebei'],
        )
    """
    root = Path(root)

    all_pairs: list[tuple[Path, Path]] = []
    train_indices: list[int] = []
    test_indices:  list[int] = []
    unassigned: list[str]    = []

    for folder_name in sorted(FOLDER_CLASS.keys()):
        sar_folder = root / folder_name / 'SAR'
        gt_folder  = root / folder_name / 'GT'
        if not sar_folder.exists():
            continue
        for chip_path in sorted(sar_folder.glob('*_SAR.tif')):
            mask_path = gt_folder / chip_path.name.replace('_SAR.tif', '_GT.tif')
            if not mask_path.exists():
                continue
            idx = len(all_pairs)
            all_pairs.append((chip_path, mask_path))

            # Extract event name from filename
            # Format: YYYYMMDD_{EventName}_ID_row_col_SAR.tif
            parts = chip_path.stem.split('_')  # stem drops _SAR.tif
            # parts[0] = date, parts[1] = event (city/country), rest = ID
            event_name = parts[1] if len(parts) > 1 else 'unknown'

            if any(e.lower() in event_name.lower() for e in train_events):
                train_indices.append(idx)
            elif any(e.lower() in event_name.lower() for e in test_events):
                test_indices.append(idx)
            else:
                unassigned.append(event_name)

    if unassigned:
        unique_unassigned = sorted(set(unassigned))
        print(f'Warning: {len(unassigned)} chips from unassigned events: '
              f'{unique_unassigned}')
        print('These chips are excluded from both train and test splits.')

    return train_indices, test_indices


# Convenience: known event names in UrbanSARFloods
# Use these when calling make_event_split()
URBAN_SAR_FLOODS_EVENTS = [
    'Houston',      # Harvey 2017 (03_FU) + pre-Harvey 2016 (01_NF)
    'Beira',        # Cyclone Idai 2019
    'Japan',        # Western Japan 2018
    'Hagibis',      # Typhoon Hagibis 2019 (Tokyo)
    'Sydney',       # Sydney 2021 + 2022
    'Coraki',       # Coraki Australia 2022
    'Niger',        # Lokoja Nigeria 2022
    'Hebei',        # Hebei China 2023
    'Beledweyne',   # Beledweyne Somalia 2023
    'Canada',       # Canada 2019
    'Iran',         # Iran 2019
    'Lumberton',    # Lumberton NC 2016
    'Somalia',      # Somalia 2018
    'PortMacquarie',# Port Macquarie Australia 2021
]
