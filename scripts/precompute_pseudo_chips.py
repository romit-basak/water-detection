"""
scripts/precompute_pseudo_chips.py

Chips new-event SAR GeoTIFFs + AWEI_sh confidence maps into .npy format
matching urban_sar_floods_aux (8-channel float16, 512x512 chips).

These are entirely NEW flood events not in UrbanSARFloods. The chips produced
here are consumed by src/data/pseudo_dataset_v2.py for Stage 0 pretraining,
which runs BEFORE fine-tuning on UrbanSARFloods.

Input layout expected (after GEE exports + SAR downloads):

  data/pseudolabels/
    confidence_maps/
      Brisbane_2022_awei_confidence.tif
      Valencia_Spain_2024_awei_confidence.tif
      ... (one per event, downloaded from GEE)
    sar/
      Brisbane_2022_VV.tif         ← Sentinel-1 GRD, VV polarisation
      Brisbane_2022_VH.tif         ← Sentinel-1 GRD, VH polarisation
      ... (download from GEE using scripts/export_sar_pseudo.js)
    aux/
      Brisbane_2022_buildings.tif  ← Google Buildings density (same GEE exports
      Brisbane_2022_worldcover.tif    as existing events, just new AOIs)
      Brisbane_2022_hand.tif
      Brisbane_2022_angle.tif

Output layout:

  data/pseudo_chips/
    Brisbane_2022/
      SAR/
        pseudo_Brisbane_2022_0000_SAR.npy   ← (8, 512, 512) float16
        pseudo_Brisbane_2022_0001_SAR.npy
        ...
      GT/
        pseudo_Brisbane_2022_0000_GT.npy    ← (512, 512) float32 confidence
        pseudo_Brisbane_2022_0001_GT.npy    (confidence values, NOT binary labels)
        ...

The GT files store RAW CONFIDENCE VALUES (float32, -1 to 1), not binary labels.
Binary label conversion (conf >= 0.65 → flood, conf <= 0.35 → non-flood) is
done at training time in PseudoChipDataset, so we can experiment with different
thresholds without re-chipping.

Chipping strategy:
- Slide a 512x512 window (at 10m/pixel = 5.12km x 5.12km) with 50% overlap
  over the SAR footprint, matching UrbanSARFloods chip size exactly.
- Chips with > 90% nodata in VV are skipped (ocean/missing data).
- Chips with > 95% ignore pixels in the confidence map (all shadow/cloud)
  are skipped (no useful label signal).
- Chips with any flood signal (conf >= 0.65 in > 0.1% of pixels) are
  kept preferentially; non-flood-only chips are subsampled to 3:1
  non-flood:flood ratio to avoid swamping the dataset with dry chips.

SAR preprocessing (matches precompute_chips.py exactly):
- VV, VH: clip to [-30, 0] dB, normalise to [0, 1]
- varVV, varVH: local variance in 5x5 window, normalise by 99th percentile
- Channels 4-7: buildings, worldcover, HAND, incidence angle (same as aux pipeline)

Usage:
    uv run python scripts/precompute_pseudo_chips.py

    # Dry run to check chip counts without writing:
    uv run python scripts/precompute_pseudo_chips.py --dry_run

    # Single event:
    uv run python scripts/precompute_pseudo_chips.py --event Brisbane_2022
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import uniform_filter, uniform_filter1d
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PSEUDO_ROOT   = Path('data/pseudo_labels')
CONF_DIR      = PSEUDO_ROOT / 'confidence_maps'
SAR_DIR       = PSEUDO_ROOT / 'sar'
AUX_DIR       = PSEUDO_ROOT / 'aux'
OUT_ROOT      = Path('data/pseudo_chips')

# ── Chip geometry ─────────────────────────────────────────────────────────────
CHIP_PX       = 512       # pixels per side, matching UrbanSARFloods
STRIDE_PX     = 256       # 50% overlap — same density as UrbanSARFloods chipping
SAR_DB_MIN    = -30.0     # dB clamp lower bound
SAR_DB_MAX    =   0.0     # dB clamp upper bound
VAR_WINDOW    =   5       # local variance window (pixels)
HAND_MAX      =  30.0     # HAND normalisation ceiling (metres)

# ── Chip filtering thresholds ─────────────────────────────────────────────────
NODATA_SKIP_FRAC   = 0.90   # skip chip if > 90% VV pixels are nodata/zero
IGNORE_SKIP_FRAC   = 0.95   # skip chip if > 95% confidence pixels are -1
FLOOD_SIGNAL_FRAC  = 0.001  # chip "has flood" if this fraction of pixels conf >= 0.65
FLOOD_THRESH       = 0.65   # confidence threshold for flood label
NF_SUBSAMPLE_RATIO = 3      # keep at most this many non-flood chips per flood chip

# ── Events (name must match GEE export filename prefix) ───────────────────────
# fmt: off
EVENTS = [
    'Brisbane_2022',
    'Sydney_2022',
    'Valencia_Spain_2024',
    'Liege_Belgium_2021_A',
    'Liege_Belgium_2021_B',
    'Cologne_Germany_2021',
    'Zhengzhou_China_2021',
    'Derna_Libya_2023_A',
    'Derna_Libya_2023_B',
    'Emilia_Romagna_2023',
    'Rio_Grande_Sul_2024',
]
# fmt: on


# =============================================================================
# SAR preprocessing helpers (mirrors precompute_chips.py exactly)
# =============================================================================

def db_to_norm(arr: np.ndarray) -> np.ndarray:
    """Convert linear SAR amplitude to dB, clip to [SAR_DB_MIN, 0], normalise to [0,1]."""
    arr = arr.astype(np.float32)
    # GEE exports Sentinel-1 GRD as linear power (float); convert to dB
    # Guard against log(0): replace zeros/negatives with a small floor
    arr = np.where(arr <= 0, 1e-10, arr)
    db  = 10.0 * np.log10(arr)
    db  = np.clip(db, SAR_DB_MIN, SAR_DB_MAX)
    return (db - SAR_DB_MIN) / (SAR_DB_MAX - SAR_DB_MIN)  # [0, 1]


def local_variance(arr: np.ndarray, window: int = VAR_WINDOW) -> np.ndarray:
    """Per-pixel local variance over a square window via E[X^2] - E[X]^2."""
    arr   = arr.astype(np.float64)
    mean  = uniform_filter(arr, size=window, mode='reflect')
    mean2 = uniform_filter(arr ** 2, size=window, mode='reflect')
    var   = np.maximum(mean2 - mean ** 2, 0.0).astype(np.float32)
    return var


def normalise_variance(var: np.ndarray) -> np.ndarray:
    """Normalise variance by its 99th percentile (matching precompute_chips.py)."""
    p99 = np.percentile(var[var > 0], 99) if (var > 0).any() else 1.0
    return np.clip(var / max(p99, 1e-8), 0.0, 1.0).astype(np.float32)


# =============================================================================
# Raster loading helpers
# =============================================================================

def read_band(path: Path, nodata_fill: float = 0.0) -> tuple[np.ndarray, object]:
    """Read first band of a raster, return (array, transform)."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        t    = src.transform
        nd   = src.nodata
    if nd is not None:
        data[data == nd] = nodata_fill
    data = np.nan_to_num(data, nan=nodata_fill, posinf=nodata_fill, neginf=nodata_fill)
    return data, t


def read_band_meta(path: Path):
    """Return (height, width, transform, crs) without loading data."""
    with rasterio.open(path) as src:
        return src.height, src.width, src.transform, src.crs


def resample_to_shape(path: Path, target_h: int, target_w: int,
                      nodata_fill: float = 0.0) -> np.ndarray:
    """Read and resample raster to (target_h, target_w)."""
    if not path.exists():
        return np.full((target_h, target_w), nodata_fill, dtype=np.float32)
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(target_h, target_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        nd = src.nodata
    if nd is not None:
        data[data == nd] = nodata_fill
    return np.nan_to_num(data, nan=nodata_fill, posinf=nodata_fill, neginf=nodata_fill)


def chip_slice(arr2d: np.ndarray, row: int, col: int) -> np.ndarray:
    """Extract CHIP_PX x CHIP_PX window, zero-padding if near edge."""
    h, w    = arr2d.shape
    r0, c0  = row, col
    r1, c1  = min(row + CHIP_PX, h), min(col + CHIP_PX, w)
    out     = np.zeros((CHIP_PX, CHIP_PX), dtype=arr2d.dtype)
    out[:r1-r0, :c1-c0] = arr2d[r0:r1, c0:c1]
    return out


# =============================================================================
# Per-event chipping
# =============================================================================

def chip_event(event: str, dry_run: bool = False) -> dict:
    """
    Chip one event into .npy files.
    Returns stats dict: {'total', 'skipped_nodata', 'skipped_ignore',
                         'flood_chips', 'nonflood_chips', 'written'}.
    """
    vv_path   = SAR_DIR  / f'{event}_VV.tif'
    vh_path   = SAR_DIR  / f'{event}_VH.tif'
    conf_path = CONF_DIR / f'{event}_awei_confidence.tif'

    # ── Verify required inputs ────────────────────────────────────────────
    missing = [p for p in [vv_path, vh_path, conf_path] if not p.exists()]
    if missing:
        print(f'  SKIP {event}: missing files: {[p.name for p in missing]}')
        return {'event': event, 'written': 0, 'error': 'missing_inputs'}

    # ── Load SAR bands ────────────────────────────────────────────────────
    print(f'\n  Loading SAR: {event}')
    vv_raw, vv_transform = read_band(vv_path, nodata_fill=0.0)
    vh_raw, _            = read_band(vh_path, nodata_fill=0.0)
    H, W                 = vv_raw.shape

    # Preprocess: dB normalise + local variance
    vv_norm  = db_to_norm(vv_raw)
    vh_norm  = db_to_norm(vh_raw)
    vv_var   = normalise_variance(local_variance(vv_norm))
    vh_var   = normalise_variance(local_variance(vh_norm))

    # ── Load confidence map (resampled to SAR grid) ───────────────────────
    print(f'  Loading confidence map: {conf_path.name}')
    conf = resample_to_shape(conf_path, H, W, nodata_fill=-1.0)

    # ── Load auxiliary channels (optional; fall back to safe defaults) ─────
    def load_aux(suffix: str, fill: float, clip_lo: float = 0.0,
                 clip_hi: float = 1.0, scale: float = 1.0) -> np.ndarray:
        p = AUX_DIR / f'{event}_{suffix}.tif'
        if not p.exists():
            print(f'    Warning: {p.name} not found, using fill={fill}')
            return np.full((H, W), fill, dtype=np.float32)
        data = resample_to_shape(p, H, W, nodata_fill=fill)
        return np.clip(data * scale, clip_lo, clip_hi).astype(np.float32)

    # Channel 4: Google Buildings density — normalise by 99th percentile
    bld_raw = load_aux('buildings', fill=0.0)
    p99     = np.percentile(bld_raw[bld_raw > 0], 99) if (bld_raw > 0).any() else 1.0
    bld     = np.clip(bld_raw / max(p99, 1.0), 0.0, 1.0).astype(np.float32)

    # Channel 5: WorldCover built-up fraction [0, 1]
    wc = load_aux('worldcover', fill=0.0, clip_lo=0.0, clip_hi=1.0)

    # Channel 6: HAND elevation clipped to [0, HAND_MAX] → [0, 1]
    hand_raw = load_aux('hand', fill=0.5)  # 0.5 = 15m, safe neutral prior
    hand     = np.clip(hand_raw / HAND_MAX, 0.0, 1.0).astype(np.float32)

    # Channel 7: Incidence angle, already normalised [20,46]°→[0,1] in GEE
    angle = load_aux('angle', fill=0.5, clip_lo=0.0, clip_hi=1.0)

    # ── Identify nodata mask from VV ──────────────────────────────────────
    # GEE exports use 0 as nodata for SAR (no valid returns); any chip that is
    # overwhelmingly zero in VV is outside the SAR swath.
    vv_nodata = (vv_raw == 0.0)

    # ── Slide chipping window ─────────────────────────────────────────────
    out_dir = OUT_ROOT / event
    sar_out = out_dir / 'SAR'
    gt_out  = out_dir / 'GT'
    if not dry_run:
        sar_out.mkdir(parents=True, exist_ok=True)
        gt_out.mkdir(parents=True, exist_ok=True)

    stats = {
        'event': event, 'total': 0, 'skipped_nodata': 0,
        'skipped_ignore': 0, 'flood_chips': [], 'nonflood_chips': [],
        'written': 0,
    }

    rows = range(0, H - CHIP_PX // 2, STRIDE_PX)  # stop when less than half a chip remains
    cols = range(0, W - CHIP_PX // 2, STRIDE_PX)

    for row in tqdm(rows, desc=f'  {event}', leave=False):
        for col in cols:
            stats['total'] += 1

            # ── Nodata check ──────────────────────────────────────────────
            nd_chip = chip_slice(vv_nodata.astype(np.float32), row, col)
            if nd_chip.mean() > NODATA_SKIP_FRAC:
                stats['skipped_nodata'] += 1
                continue

            # ── Confidence chip ───────────────────────────────────────────
            conf_chip = chip_slice(conf, row, col)
            ignore_frac = (conf_chip < 0).mean()
            if ignore_frac > IGNORE_SKIP_FRAC:
                stats['skipped_ignore'] += 1
                continue

            # ── Classify chip as flood or non-flood ───────────────────────
            flood_frac = (conf_chip >= FLOOD_THRESH).mean()
            has_flood  = flood_frac > FLOOD_SIGNAL_FRAC

            chip_id = f'pseudo_{event}_{stats["total"]:05d}'
            entry   = (row, col, chip_id)
            if has_flood:
                stats['flood_chips'].append(entry)
            else:
                stats['nonflood_chips'].append(entry)

    # ── Subsample non-flood chips to NF_SUBSAMPLE_RATIO : 1 ──────────────
    n_flood = len(stats['flood_chips'])
    n_nf_keep = max(n_flood * NF_SUBSAMPLE_RATIO, 50)  # always keep at least 50
    rng = np.random.default_rng(seed=42)
    if len(stats['nonflood_chips']) > n_nf_keep:
        keep_idx = rng.choice(len(stats['nonflood_chips']), size=n_nf_keep, replace=False)
        stats['nonflood_chips'] = [stats['nonflood_chips'][i] for i in sorted(keep_idx)]

    all_chips = stats['flood_chips'] + stats['nonflood_chips']

    if dry_run:
        print(f'  DRY RUN {event}: {len(all_chips)} chips '
              f'({n_flood} flood, {len(stats["nonflood_chips"])} non-flood, '
              f'{stats["skipped_nodata"]} skipped nodata, '
              f'{stats["skipped_ignore"]} skipped ignore)')
        stats['written'] = len(all_chips)
        return stats

    # ── Write chips ───────────────────────────────────────────────────────
    for (row, col, chip_id) in tqdm(all_chips, desc=f'  Writing {event}', leave=False):
        sar_path  = sar_out / f'{chip_id}_SAR.npy'
        gt_path   = gt_out  / f'{chip_id}_GT.npy'

        if sar_path.exists() and gt_path.exists():
            continue  # already written (resumable)

        # SAR channels
        vv_c  = chip_slice(vv_norm,  row, col)
        vh_c  = chip_slice(vh_norm,  row, col)
        vvv_c = chip_slice(vv_var,   row, col)
        vhv_c = chip_slice(vh_var,   row, col)

        # Aux channels
        bld_c   = chip_slice(bld,   row, col)
        wc_c    = chip_slice(wc,    row, col)
        hand_c  = chip_slice(hand,  row, col)
        angle_c = chip_slice(angle, row, col)

        # Stack: (8, 512, 512) float16
        chip_arr = np.stack(
            [vv_c, vh_c, vvv_c, vhv_c, bld_c, wc_c, hand_c, angle_c],
            axis=0
        ).astype(np.float16)

        # GT: raw confidence values (float32) — label conversion at training time
        conf_c = chip_slice(conf, row, col).astype(np.float32)

        np.save(sar_path, chip_arr)
        np.save(gt_path,  conf_c)
        stats['written'] += 1

    print(f'  {event}: wrote {stats["written"]} chips '
          f'({n_flood} flood + {len(stats["nonflood_chips"])} non-flood, '
          f'{stats["skipped_nodata"]} nodata-skipped, '
          f'{stats["skipped_ignore"]} ignore-skipped)')
    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry_run', action='store_true',
                   help='Count chips without writing any files')
    p.add_argument('--event', type=str, default='',
                   help='Process only this event (default: all)')
    args = p.parse_args()

    events = [args.event] if args.event else EVENTS
    mode   = 'DRY RUN' if args.dry_run else 'WRITING'

    print(f'=== Pseudo-chip precomputation ({mode}) ===')
    print(f'Events: {events}')
    print(f'Output: {OUT_ROOT}')
    print(f'Chip size: {CHIP_PX}x{CHIP_PX}px, stride: {STRIDE_PX}px (50% overlap)')
    print()

    all_stats = []
    for event in events:
        print(f'[{event}]')
        s = chip_event(event, dry_run=args.dry_run)
        all_stats.append(s)

    # ── Summary ───────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    total_written = 0
    for s in all_stats:
        if 'error' in s:
            print(f'  {s["event"]:<30}  ERROR: {s["error"]}')
        else:
            n_flood = len(s["flood_chips"]) if isinstance(s["flood_chips"], list) else 0
            n_nf    = len(s["nonflood_chips"]) if isinstance(s["nonflood_chips"], list) else 0
            print(f'  {s["event"]:<30}  {s["written"]:4d} chips  '
                  f'({n_flood} flood + {n_nf} non-flood)')
            total_written += s['written']

    print(f'\nTotal chips: {total_written}')
    print(f'UrbanSARFloods training chips (7 events): ~3261')
    print(f'Pseudo chips as % of fine-tune training: ~{100*total_written/(total_written+3261):.0f}%')
    print()
    print('Next steps:')
    print('  1. Verify a few chips visually:')
    print('     python -c "import numpy as np; import matplotlib.pyplot as plt')
    print('     c = np.load(\'data/pseudo_chips/Brisbane_2022/SAR/pseudo_Brisbane_2022_00001_SAR.npy\')')
    print('     print(c.shape, c.dtype, c.min(), c.max())"')
    print('  2. Run Stage 0 pretraining:')
    print('     uv run python scripts/train_with_pseudo.py')


if __name__ == '__main__':
    main()
