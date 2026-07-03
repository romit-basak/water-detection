"""
scripts/04_chip_and_validate.py

Chips full-AOI GeoTIFFs exported from GEE into 512×512 training pairs.

Input  (data/pseudo_labels/raw/):
    {event}_SAR.tif    — 2-band float32 [VV_dB, VH_dB] at 10m
    {event}_LABEL.tif  — uint8 pseudo-label (0=masked, 1=flood_open, 2=flood_urban)

Output (data/pseudo_labels/chips/):
    {event}_{row:04d}_{col:04d}_SAR.tif
    {event}_{row:04d}_{col:04d}_LABEL.tif

Quality filters:
    - SAR no-data fraction < MAX_NODATA_FRAC   (rejects edge/ocean chips)
    - Valid label fraction ≥ MIN_VALID_FRAC     (rejects fully masked chips)
    - Flood pixel fraction ≥ MIN_FLOOD_FRAC     (rejects chips with no flood signal)
      Flood fraction = (class 1 + class 2) / valid pixels

Run:
    uv run python scripts/04_chip_and_validate.py
    uv run python scripts/04_chip_and_validate.py --event UK_Yorkshire_2015
    uv run python scripts/04_chip_and_validate.py --dry_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

CHIP_SIZE        = 512
MIN_FLOOD_FRAC   = 0.02   # ≥ 2% flood pixels (class 1 or 2) among valid pixels
MIN_VALID_FRAC   = 0.30   # ≥ 30% non-masked pixels (label != 0)
MIN_URBAN_FRAC   = 0.10   # ≥ 10% of SAR pixels must be non-NaN urban
                           # (NaN in SAR = outside urban zone, masked at export)
MAX_NODATA_FRAC  = 0.10   # < 10% NaN in SAR bands

RAW_DIR   = Path('data/pseudo_labels/raw')
CHIPS_DIR = Path('data/pseudo_labels/chips')

EVENTS = [
    'UK_Yorkshire_2015',
    'Zhengzhou_2021',
    'Western_Europe_2021',
    'Typhoon_Vamco_2020',
]


# ── Core chipping function ────────────────────────────────────────────────────

def chip_event(
    event: str,
    raw_dir: Path,
    chips_dir: Path,
    dry_run: bool = False,
) -> dict:
    sar_path   = raw_dir   / f'{event}_SAR.tif'
    label_path = raw_dir   / f'{event}_LABEL.tif'

    if not sar_path.exists():
        print(f'  SKIP {event} — SAR file not found: {sar_path}')
        return {}
    if not label_path.exists():
        print(f'  SKIP {event} — label file not found: {label_path}')
        return {}

    chips_dir.mkdir(parents=True, exist_ok=True)

    stats = dict(
        event=event, total=0, accepted=0,
        rej_nodata=0, rej_valid=0, rej_flood=0,
        flood_open_px=0, flood_urban_px=0,
    )

    with rasterio.open(sar_path) as sar_src, \
         rasterio.open(label_path) as lbl_src:

        # Spatial alignment check
        assert sar_src.crs    == lbl_src.crs,    f'CRS mismatch for {event}'
        assert sar_src.width  == lbl_src.width,  f'Width mismatch for {event}'
        assert sar_src.height == lbl_src.height, f'Height mismatch for {event}'

        h, w   = sar_src.height, sar_src.width
        n_rows = h // CHIP_SIZE
        n_cols = w // CHIP_SIZE

        print(f'\n{event}: {w}×{h}px → {n_rows}×{n_cols} = '
              f'{n_rows * n_cols} candidate chips')

        for row in tqdm(range(n_rows), desc=f'  {event}'):
            for col in range(n_cols):
                stats['total'] += 1

                win = Window(
                    col_off=col * CHIP_SIZE,
                    row_off=row * CHIP_SIZE,
                    width=CHIP_SIZE,
                    height=CHIP_SIZE,
                )

                sar_chip = sar_src.read(window=win).astype(np.float32)  # (2,H,W)
                lbl_chip = lbl_src.read(1, window=win).astype(np.uint8) # (H,W)

                # ── Quality filters ─────────────────────────────────────

                # 1. SAR no-data
                nodata_frac = np.isnan(sar_chip).mean()
                if nodata_frac > MAX_NODATA_FRAC:
                    stats['rej_nodata'] += 1
                    continue

                # 2. Urban coverage: NaN in SAR = rural (masked at export)
                #    Non-NaN fraction = urban zone fraction
                urban_frac = 1.0 - nodata_frac
                if urban_frac < MIN_URBAN_FRAC:
                    stats['rej_nodata'] += 1  # counted as nodata rejection
                    continue

                total_px = CHIP_SIZE * CHIP_SIZE
                valid_px = int((lbl_chip != 0).sum())
                flood_px = int((lbl_chip >= 1).sum())  # class 1 + class 2

                # 2. Valid pixel fraction
                if valid_px / total_px < MIN_VALID_FRAC:
                    stats['rej_valid'] += 1
                    continue

                # 3. Flood fraction among valid pixels
                if flood_px / max(valid_px, 1) < MIN_FLOOD_FRAC:
                    stats['rej_flood'] += 1
                    continue

                stats['accepted']       += 1
                stats['flood_open_px']  += int((lbl_chip == 1).sum())
                stats['flood_urban_px'] += int((lbl_chip == 2).sum())

                if dry_run:
                    continue

                # ── Write chip pair ─────────────────────────────────────
                chip_stem = f'{event}_{row:04d}_{col:04d}'
                transform = sar_src.window_transform(win)

                _write_tif(
                    chips_dir / f'{chip_stem}_SAR.tif',
                    sar_chip, count=2, dtype=np.float32,
                    crs=sar_src.crs, transform=transform,
                )
                _write_tif(
                    chips_dir / f'{chip_stem}_LABEL.tif',
                    lbl_chip[np.newaxis], count=1, dtype=np.uint8,
                    crs=lbl_src.crs, transform=transform,
                )

    return stats


def _write_tif(
    path: Path,
    data: np.ndarray,
    count: int,
    dtype,
    crs,
    transform,
) -> None:
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=CHIP_SIZE, width=CHIP_SIZE,
        count=count, dtype=dtype,
        crs=crs, transform=transform,
    ) as dst:
        dst.write(data)


# ── Reporting ────────────────────────────────────────────────────────────────

def print_stats(s: dict) -> None:
    if not s:
        return
    total    = s['total']
    accepted = s['accepted']
    pct      = 100 * accepted / max(total, 1)
    urban_px = s['flood_urban_px']
    open_px  = s['flood_open_px']
    print(f'\n── {s["event"]} ────────────────────────────────')
    print(f'  Candidate chips : {total:6d}')
    print(f'  Accepted        : {accepted:6d}  ({pct:.1f}%)')
    print(f'  Rej. (no-data)  : {s["rej_nodata"]:6d}')
    print(f'  Rej. (too masked): {s["rej_valid"]:6d}')
    print(f'  Rej. (no flood) : {s["rej_flood"]:6d}')
    print(f'  Flood open  px  : {open_px:10,}')
    print(f'  Flood urban px  : {urban_px:10,}')
    if accepted > 0:
        urban_frac = 100 * urban_px / max(open_px + urban_px, 1)
        print(f'  Urban flood %   : {urban_frac:.1f}%')


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Chip pseudo-label GeoTIFFs')
    parser.add_argument('--event',     default=None,
                        help='Process single event (default: all)')
    parser.add_argument('--dry_run',   action='store_true',
                        help='Stats only, no files written')
    parser.add_argument('--raw_dir',   default=str(RAW_DIR))
    parser.add_argument('--chips_dir', default=str(CHIPS_DIR))
    args = parser.parse_args()

    raw_dir   = Path(args.raw_dir)
    chips_dir = Path(args.chips_dir)
    events    = [args.event] if args.event else EVENTS

    if args.dry_run:
        print('DRY RUN — no files will be written.')

    all_stats = []
    for event in events:
        s = chip_event(event, raw_dir, chips_dir, dry_run=args.dry_run)
        print_stats(s)
        if s:
            all_stats.append(s)

    if all_stats:
        total_accepted    = sum(s['accepted']       for s in all_stats)
        total_chips       = sum(s['total']           for s in all_stats)
        total_flood_urban = sum(s['flood_urban_px']  for s in all_stats)
        total_flood_open  = sum(s['flood_open_px']   for s in all_stats)
        total_flood       = total_flood_urban + total_flood_open
        print(f'\n{"═"*56}')
        print(f'TOTAL: {total_accepted}/{total_chips} chips accepted '
              f'({100*total_accepted/max(total_chips,1):.1f}%)')
        print(f'Total flood pixels : {total_flood:,}')
        print(f'  Urban flood      : {total_flood_urban:,} '
              f'({100*total_flood_urban/max(total_flood,1):.1f}%)')
        print(f'  Open flood       : {total_flood_open:,} '
              f'({100*total_flood_open/max(total_flood,1):.1f}%)')
        print(f'Output             : {chips_dir.resolve()}')


if __name__ == '__main__':
    main()
