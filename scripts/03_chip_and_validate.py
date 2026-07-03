"""
03_chip_and_validate.py

Takes the full-AOI GeoTIFFs exported from GEE (02_pseudo_label_export.js)
and produces 512×512 chip pairs ready for training.

Input (from Google Drive, placed in data/pseudo_labels/raw/):
    {event}_SAR.tif     — 2-band float32 [VV_dB, VH_dB] at 10m
    {event}_LABEL.tif   — uint8 pseudo-label mask

Output (written to data/pseudo_labels/chips/):
    {event}_{row}_{col}_SAR.tif    — 512×512 SAR chip
    {event}_{row}_{col}_LABEL.tif  — 512×512 label chip

Quality filters applied:
    - Chip must have ≥ MIN_FLOOD_FRACTION flood pixels (label == 1)
    - Chip must have ≥ MIN_VALID_FRACTION non-masked pixels (label != 0)
    - Chip must not be mostly no-data (SAR NaN fraction < 0.1)

Run from project root:
    uv run python scripts/03_chip_and_validate.py
    uv run python scripts/03_chip_and_validate.py --event Houston_Harvey_2017
    uv run python scripts/03_chip_and_validate.py --dry_run  # stats only, no files written
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────

CHIP_SIZE      = 512       # pixels
MIN_FLOOD_FRAC = 0.02      # chip needs ≥ 2% flood pixels
MIN_VALID_FRAC = 0.30      # chip needs ≥ 30% non-masked pixels
MAX_NODATA_FRAC = 0.10     # chip rejected if > 10% SAR is NaN/nodata

RAW_DIR   = Path("data/pseudo_labels/raw")    # GeoTIFFs from GEE
CHIPS_DIR = Path("data/pseudo_labels/chips")  # output chips

EVENTS = [
    "Houston_Harvey_2017",
    "Valencia_Spain_2024",
    "Brisbane_Australia_2022",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def chip_image(
    sar_path: Path,
    label_path: Path,
    event: str,
    out_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Chip a SAR + label GeoTIFF pair into 512×512 tiles.
    Returns statistics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "event": event,
        "total_chips": 0,
        "accepted": 0,
        "rejected_flood": 0,
        "rejected_valid": 0,
        "rejected_nodata": 0,
    }

    with rasterio.open(sar_path) as sar_src, \
         rasterio.open(label_path) as lbl_src:

        # Verify spatial alignment
        assert sar_src.crs == lbl_src.crs, \
            f"CRS mismatch: SAR={sar_src.crs}, label={lbl_src.crs}"
        assert sar_src.width == lbl_src.width and sar_src.height == lbl_src.height, \
            f"Shape mismatch: SAR={sar_src.shape}, label={lbl_src.shape}"

        h, w = sar_src.height, sar_src.width
        n_rows = h // CHIP_SIZE
        n_cols = w // CHIP_SIZE

        print(f"\n{event}: {w}×{h}px → {n_rows}×{n_cols} grid = "
              f"{n_rows * n_cols} chips")

        for row in tqdm(range(n_rows), desc=f"  Rows ({event})"):
            for col in range(n_cols):
                stats["total_chips"] += 1

                win = Window(
                    col_off=col * CHIP_SIZE,
                    row_off=row * CHIP_SIZE,
                    width=CHIP_SIZE,
                    height=CHIP_SIZE,
                )

                # Read SAR chip (2 bands: VV, VH)
                sar_chip = sar_src.read(window=win).astype(np.float32)
                # Read label chip
                lbl_chip = lbl_src.read(1, window=win).astype(np.uint8)

                # ── Quality filters ───────────────────────────────────────

                # 1. SAR nodata check (GEE exports NaN for missing pixels)
                nodata_frac = np.isnan(sar_chip).mean()
                if nodata_frac > MAX_NODATA_FRAC:
                    stats["rejected_nodata"] += 1
                    continue

                total_px  = CHIP_SIZE * CHIP_SIZE
                valid_px  = (lbl_chip != 0).sum()
                flood_px  = (lbl_chip == 1).sum()

                # 2. Valid pixel fraction
                if valid_px / total_px < MIN_VALID_FRAC:
                    stats["rejected_valid"] += 1
                    continue

                # 3. Flood pixel fraction (among valid pixels only)
                flood_frac = flood_px / max(valid_px, 1)
                if flood_frac < MIN_FLOOD_FRAC:
                    stats["rejected_flood"] += 1
                    continue

                stats["accepted"] += 1

                if dry_run:
                    continue

                # ── Write chip pair ───────────────────────────────────────

                chip_name = f"{event}_{row:04d}_{col:04d}"

                # Compute chip transform (georeferenced)
                chip_transform = sar_src.window_transform(win)

                # Write SAR chip
                sar_out = out_dir / f"{chip_name}_SAR.tif"
                with rasterio.open(
                    sar_out, "w",
                    driver="GTiff",
                    height=CHIP_SIZE, width=CHIP_SIZE,
                    count=2,
                    dtype=np.float32,
                    crs=sar_src.crs,
                    transform=chip_transform,
                ) as dst:
                    dst.write(sar_chip)

                # Write label chip
                lbl_out = out_dir / f"{chip_name}_LABEL.tif"
                with rasterio.open(
                    lbl_out, "w",
                    driver="GTiff",
                    height=CHIP_SIZE, width=CHIP_SIZE,
                    count=1,
                    dtype=np.uint8,
                    crs=lbl_src.crs,
                    transform=chip_transform,
                ) as dst:
                    dst.write(lbl_chip[np.newaxis])

    return stats


def print_stats(stats: dict) -> None:
    total    = stats["total_chips"]
    accepted = stats["accepted"]
    print(f"\n── {stats['event']} ─────────────────────────")
    print(f"  Total chips:          {total:5d}")
    print(f"  Accepted:             {accepted:5d}  ({100*accepted/max(total,1):.1f}%)")
    print(f"  Rejected (nodata):    {stats['rejected_nodata']:5d}")
    print(f"  Rejected (too masked):{stats['rejected_valid']:5d}")
    print(f"  Rejected (no flood):  {stats['rejected_flood']:5d}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default=None,
                        help="Process a single event by name (default: all)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print stats only, do not write chips")
    parser.add_argument("--raw_dir",  default=str(RAW_DIR))
    parser.add_argument("--chips_dir", default=str(CHIPS_DIR))
    args = parser.parse_args()

    raw_dir   = Path(args.raw_dir)
    chips_dir = Path(args.chips_dir)
    events    = [args.event] if args.event else EVENTS

    if args.dry_run:
        print("DRY RUN — no files will be written.")

    all_stats = []
    for event in events:
        sar_path   = raw_dir / f"{event}_SAR.tif"
        label_path = raw_dir / f"{event}_LABEL.tif"

        if not sar_path.exists():
            print(f"\nSkipping {event} — SAR file not found: {sar_path}")
            continue
        if not label_path.exists():
            print(f"\nSkipping {event} — label file not found: {label_path}")
            continue

        stats = chip_image(
            sar_path, label_path, event,
            out_dir=chips_dir,
            dry_run=args.dry_run,
        )
        print_stats(stats)
        all_stats.append(stats)

    if all_stats:
        total_accepted = sum(s["accepted"] for s in all_stats)
        total_chips    = sum(s["total_chips"] for s in all_stats)
        print(f"\n══ TOTAL: {total_accepted}/{total_chips} chips accepted "
              f"({100*total_accepted/max(total_chips,1):.1f}%)")
        print(f"Output: {chips_dir.resolve()}")


if __name__ == "__main__":
    main()
