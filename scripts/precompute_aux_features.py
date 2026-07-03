"""
scripts/precompute_aux_features.py

Bakes auxiliary context features into new .npy files alongside
the existing SAR chips. Each chip gets 4 extra channels:
  - Channel 4: Google Buildings count density (normalized)
  - Channel 5: WorldCover built-up fraction  [0, 1]
  - Channel 6: HAND elevation, clipped to [0, 30m] then normalized to [0, 1]
  - Channel 7: Incidence angle, normalized [20, 46] degrees -> [0, 1]

The augmented chips are saved to:
  data/urban_sar_floods_aux/
    01_NF/SAR/*.npy   (8-channel: VV, VH, var_VV, var_VH, buildings, builtup, hand, angle)
    01_NF/GT/*.npy
    ...

Run after:
  1. scripts/precompute_chips.py (SAR preprocessing)
  2. GEE exports downloaded to:
       data/building_density/{EventName}_building_density.tif
       data/aux_features/{EventName}_worldcover.tif
       data/aux_features/{EventName}_hand.tif
       data/aux_features/{EventName}_incidence_angle.tif

Usage:
    uv run python scripts/precompute_aux_features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from tqdm import tqdm

SAR_ROOT  = Path('data/urban_sar_floods_preprocessed')
AUX_ROOT  = Path('data/urban_sar_floods_aux')
ORIG_ROOT = Path('data/urban_sar_floods')

BUILDING_DIR   = Path('data/building_density')
WORLDCOVER_DIR = Path('data/aux_features')
HAND_DIR       = Path('data/aux_features')
ANGLE_DIR      = Path('data/aux_features')

HAND_MAX = 30.0  # clip and normalise HAND to [0, 1] over [0, 30m]


def event_from_chip(chip_path: Path) -> str:
    parts = chip_path.stem.split('_')
    return parts[1] if len(parts) > 1 else 'unknown'


def load_raster_at_bounds(raster_path: Path, bounds) -> np.ndarray | None:
    """Sample a raster at a given geographic bounding box -> (512, 512) float32."""
    if not raster_path.exists():
        return None
    try:
        with rasterio.open(raster_path) as src:
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
            data = data.astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = 0.0
            return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f'  Warning: could not read {raster_path.name}: {e}')
        return None


def get_chip_bounds(chip_path: Path, folder: str):
    orig = ORIG_ROOT / folder / 'SAR' / chip_path.name.replace('.npy', '.tif')
    if not orig.exists():
        return None
    with rasterio.open(orig) as src:
        return src.bounds


def process_folder(folder: str):
    sar_src = SAR_ROOT / folder / 'SAR'
    gt_src  = SAR_ROOT / folder / 'GT'
    sar_dst = AUX_ROOT / folder / 'SAR'
    gt_dst  = AUX_ROOT / folder / 'GT'

    if not sar_src.exists():
        return

    sar_dst.mkdir(parents=True, exist_ok=True)
    gt_dst.mkdir(parents=True, exist_ok=True)

    chips = sorted(sar_src.glob('*.npy'))
    print(f'\n{folder}: {len(chips)} chips')

    missing = {'building': 0, 'worldcover': 0, 'hand': 0, 'incidence_angle': 0}

    for chip_path in tqdm(chips, desc=folder):
        out_sar = sar_dst / chip_path.name
        out_gt  = gt_dst  / chip_path.name.replace('_SAR.npy', '_GT.npy')

        if out_sar.exists() and out_gt.exists():
            continue

        sar    = np.load(chip_path).astype(np.float32)  # (4, 512, 512)
        bounds = get_chip_bounds(chip_path, folder)
        event  = event_from_chip(chip_path)

        # ── Channel 4: Google Buildings density ───────────────────────────
        b_data = load_raster_at_bounds(
            BUILDING_DIR / f'{event}_building_density.tif', bounds)
        if b_data is None:
            b_data = np.zeros((512, 512), dtype=np.float32)
            missing['building'] += 1
        else:
            p99 = np.percentile(b_data[b_data > 0], 99) if (b_data > 0).any() else 1.0
            b_data = np.clip(b_data / max(p99, 1.0), 0.0, 1.0)

        # ── Channel 5: WorldCover built-up fraction ────────────────────────
        w_data = load_raster_at_bounds(
            WORLDCOVER_DIR / f'{event}_worldcover.tif', bounds)
        if w_data is None:
            w_data = np.zeros((512, 512), dtype=np.float32)
            missing['worldcover'] += 1
        else:
            w_data = np.clip(w_data, 0.0, 1.0)

        # ── Channel 6: HAND elevation ──────────────────────────────────────
        h_data = load_raster_at_bounds(
            HAND_DIR / f'{event}_hand.tif', bounds)
        if h_data is None:
            h_data = np.full((512, 512), 0.5, dtype=np.float32)
            missing['hand'] += 1
        else:
            h_data = np.clip(h_data / HAND_MAX, 0.0, 1.0)

        # ── Channel 7: Incidence angle ─────────────────────────────────────
        # Already normalised [20, 46] -> [0, 1] in GEE export.
        # 0.5 fallback = 33 degrees (midrange), a safe neutral prior.
        ia_data = load_raster_at_bounds(
            ANGLE_DIR / f'{event}_incidence_angle.tif', bounds)
        if ia_data is None:
            ia_data = np.full((512, 512), 0.5, dtype=np.float32)
            missing['incidence_angle'] += 1
        else:
            ia_data = np.clip(ia_data, 0.0, 1.0)

        # Stack: (4 SAR + 4 aux) = (8, 512, 512)
        aux   = np.stack([b_data, w_data, h_data, ia_data], axis=0)
        data8 = np.concatenate([sar, aux], axis=0).astype(np.float16)
        np.save(out_sar, data8)

        gt_src_path = gt_src / chip_path.name.replace('_SAR.npy', '_GT.npy')
        if gt_src_path.exists():
            np.save(out_gt, np.load(gt_src_path))

    if any(missing.values()):
        print(f'  Missing rasters: {missing}')


if __name__ == '__main__':
    print('Baking 8-channel aux chips (SAR + buildings + worldcover + HAND + incidence)...')
    for folder in ['01_NF', '02_FO', '03_FU']:
        process_folder(folder)
    print('\nDone. Chips saved to:', AUX_ROOT)
    print('Channels: VV VH varVV varVH buildings worldcover hand incidence_angle')
