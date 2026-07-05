"""
src/data/pseudo_dataset.py

Dataset class for AWEI_sh confidence-weighted pseudo-labeled chips.

Loads SAR chips from UrbanSARFloods together with AWEI_sh confidence
maps exported from GEE. Confidence maps provide per-pixel label
reliability rather than hard binary labels.

Confidence encoding (from GEE export):
  -1.0 = ignore (shadow / cloud / nodata)
   0.0 = confidently non-water (AWEI_sh <= -0.3)
   0.5 = uncertain (AWEI_sh near 0 threshold)
   1.0 = confidently water (AWEI_sh >= +0.3)

Label generation:
  confidence >= FLOOD_THRESH  → label = 1 (flood), loss weight = confidence
  confidence <= NONFLOOD_THRESH → label = 0 (non-flood), loss weight = 1 - confidence
  NONFLOOD_THRESH < confidence < FLOOD_THRESH → label = -1 (ignore)
  confidence = -1 → label = -1 (ignore)

This allows uncertain pixels to contribute proportionally less gradient
than confident pixels, rather than either including them with full weight
(corrupting labels) or excluding them entirely (wasting signal).
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
from rasterio.windows import from_bounds
import torch
from torch.utils.data import Dataset

# Confidence thresholds for label assignment
FLOOD_THRESH    = 0.65   # confidence >= this → flood label
NONFLOOD_THRESH = 0.35   # confidence <= this → non-flood label
# Between NONFLOOD_THRESH and FLOOD_THRESH: ignore (too uncertain)


def load_confidence_at_bounds(conf_path: Path, bounds) -> Optional[np.ndarray]:
    """Sample confidence raster at chip geographic bounds → (512, 512) float32."""
    if not conf_path.exists():
        return None
    try:
        with rasterio.open(conf_path) as src:
            rb = src.bounds
            cb = bounds
            if (cb.left > rb.right or cb.right < rb.left or
                    cb.bottom > rb.top or cb.top < rb.bottom):
                return None
            window = from_bounds(
                max(cb.left, rb.left), max(cb.bottom, rb.bottom),
                min(cb.right, rb.right), min(cb.top, rb.top),
                src.transform
            )
            if window.width < 1 or window.height < 1:
                return None
            data = src.read(1, window=window,
                            out_shape=(512, 512),
                            resampling=rasterio.enums.Resampling.bilinear)
            return data.astype(np.float32)
    except Exception as e:
        print(f'  Warning: could not read {conf_path.name}: {e}')
        return None


class PseudoLabelDataset(Dataset):
    """UrbanSARFloods chips with AWEI_sh confidence-weighted pseudo-labels.

    Wraps an existing FastUrbanSARFloods dataset by replacing its binary
    GT labels with confidence-derived labels where confidence maps exist.
    Falls back to original GT labels when no confidence map is available.

    Args:
        sar_root: Path to urban_sar_floods_aux directory (8-channel chips).
        conf_dir: Path to directory containing {Event}_awei_confidence.tif files.
        orig_root: Path to original urban_sar_floods directory (for bounds).
        split_indices: List of chip indices to include.
        transform: Albumentations transform.
        flood_thresh: Confidence threshold above which pixel is labeled flood.
        nonflood_thresh: Confidence threshold below which pixel is labeled non-flood.
    """

    def __init__(
        self,
        sar_root: str | Path,
        conf_dir: str | Path,
        orig_root: str | Path,
        split_indices: list[int],
        transform: Optional[Callable] = None,
        flood_thresh: float = FLOOD_THRESH,
        nonflood_thresh: float = NONFLOOD_THRESH,
    ):
        self.sar_root      = Path(sar_root)
        self.conf_dir      = Path(conf_dir)
        self.orig_root     = Path(orig_root)
        self.transform     = transform
        self.flood_thresh  = flood_thresh
        self.nf_thresh     = nonflood_thresh

        self._all  = self._discover()
        self.pairs = [self._all[i] for i in split_indices]

        # Count how many chips have confidence maps available
        n_with_conf = sum(1 for p, _ in self.pairs
                         if self._conf_path(p).exists())
        print(f'PseudoLabelDataset: {len(self.pairs)} chips, '
              f'{n_with_conf} with confidence maps '
              f'({100*n_with_conf/max(len(self.pairs),1):.0f}%)')

    def _discover(self):
        pairs = []
        for folder in ['01_NF', '02_FO', '03_FU']:
            sar_dir = self.sar_root / folder / 'SAR'
            gt_dir  = self.sar_root / folder / 'GT'
            if not sar_dir.exists():
                continue
            for p in sorted(sar_dir.glob('*.npy')):
                gt = gt_dir / p.name.replace('_SAR.npy', '_GT.npy')
                if gt.exists():
                    pairs.append((p, gt))
        return pairs

    def _conf_path(self, sar_path: Path) -> Path:
        """Get the confidence raster path for a chip."""
        event = sar_path.stem.split('_')[1]
        return self.conf_dir / f'{event}_awei_confidence.tif'

    def _orig_bounds(self, sar_path: Path, folder: str):
        """Get geographic bounds from the original SAR GeoTIFF."""
        orig = self.orig_root / folder / 'SAR' / \
               sar_path.name.replace('.npy', '.tif')
        if not orig.exists():
            return None
        with rasterio.open(orig) as src:
            return src.bounds

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        sar_path, gt_path = self.pairs[idx]
        image = np.load(sar_path).astype(np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        folder = sar_path.parts[-3]

        # ── Try to load confidence map ────────────────────────────────────
        conf_path = self._conf_path(sar_path)
        bounds    = self._orig_bounds(sar_path, folder)
        conf      = None
        if bounds is not None:
            conf = load_confidence_at_bounds(conf_path, bounds)

        if conf is not None:
            # Confidence-derived labels
            mask = np.full((512, 512), -1, dtype=np.int64)  # default: ignore
            # Shadow / cloud / nodata: keep as -1
            # Flood: high confidence water pixels
            mask = np.where(conf >= self.flood_thresh,  1, mask)
            # Non-flood: low confidence (confidently non-water)
            mask = np.where(conf <= self.nf_thresh,     0, mask)
            # Keep -1 in shadow regions (conf == -1.0)
            mask = np.where(conf < 0,                  -1, mask)

            # Per-pixel loss weight: how confident is this label?
            weight = np.where(
                conf >= self.flood_thresh,
                conf.astype(np.float32),                  # flood weight = confidence
                np.where(
                    conf <= self.nf_thresh,
                    (1.0 - conf).astype(np.float32),      # non-flood weight = 1 - confidence
                    np.zeros_like(conf)                   # uncertain: weight 0 (ignored anyway)
                )
            ).astype(np.float32)
        else:
            # Fall back to original GT labels (same as FastUrbanSARFloods)
            raw  = np.load(gt_path).astype(np.int64)
            if folder == '01_NF':
                mask = np.zeros_like(raw)                 # all non-flood
            else:
                mask = np.where(raw == 0, -1,
                       np.where(raw == 2,  1, raw)).astype(np.int64)
            weight = np.ones((512, 512), dtype=np.float32)

        if self.transform is not None:
            hwc   = np.transpose(image, (1, 2, 0))
            out   = self.transform(image=hwc,
                                   mask=mask.astype(np.float32))
            image = np.transpose(out['image'], (2, 0, 1))
            mask  = out['mask'].astype(np.int64)

        return {
            'image':      torch.from_numpy(image),
            'mask':       torch.from_numpy(mask),
            'weight':     torch.from_numpy(weight),
            'chip_path':  str(sar_path),
            'has_conf':   conf is not None,
        }

    @property
    def n_channels(self) -> int:
        return int(np.load(self.pairs[0][0]).shape[0])
