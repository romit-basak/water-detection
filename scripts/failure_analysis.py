"""
scripts/failure_analysis.py

Per-chip failure analysis: correlates model F1 with building density.

For each chip in the UrbanSARFloods test split, computes:
  - Per-chip F1 score (flood classes 1+2 vs background 0)
  - Building density from ESA WorldCover (fraction of 50=built-up pixels)
  - Dominant flood class (open vs urban)
  - Event name (city)

Outputs:
  - failure_analysis.csv: per-chip stats
  - failure_analysis.png: F1 vs building density scatter + per-event breakdown

Usage:
    uv run python scripts/failure_analysis.py \
        --checkpoint runs/three_stage_v5/stage3/best.pt \
        --out_dir runs/three_stage_v5/analysis

Building density source:
    Uses ESA WorldCover v200 (2021) class 50 = built-up, fetched per-chip
    from the chip's GeoTIFF bounding box via rasterio + a local WorldCover tile.
    Fallback: estimates from the SAR local variance (high variance in non-flood
    areas correlates with building density due to double-bounce).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from src.data.fast_dataset import FastUrbanSARFloods, make_urban_event_split
from src.data.transforms import val_transforms
from src.models.unet import build_unet

# ── Config ────────────────────────────────────────────────────────────────────

FINETUNE_TRAIN_EVENTS = [
    'Houston', 'Beira', 'Japan', 'Canada', 'Iran', 'Lumberton', 'Somalia',
]
FINETUNE_TEST_EVENTS = [
    'Hagibis', 'Sydney', 'Coraki', 'Niger', 'Hebei', 'Beledweyne', 'PortMacquarie',
]

N_CLASSES    = 4
URBAN_SAR_DIR = 'data/urban_sar_floods_preprocessed'
ORIG_SAR_DIR  = 'data/urban_sar_floods'   # for reading geo-coordinates


# ── Per-chip F1 ───────────────────────────────────────────────────────────────

def chip_f1(pred: np.ndarray, target: np.ndarray) -> float:
    """Per-chip F1 using ALL non-background pixels as the evaluation space.

    flood_target  = pixels labeled 1 or 2 (open flood, urban flood)
    nflood_target = pixels labeled 0 that the model predicts as flood
                    (false positives on background)

    We evaluate over ALL pixels including background:
      - background pixels (0) should be predicted as 0
      - flood pixels (1,2) should be predicted as 1 or 2
    This way the model is penalised for predicting flood on background pixels.
    """
    flood_pred   = (pred >= 1) & (pred <= 2)
    flood_target = (target >= 1) & (target <= 2)

    # Require at least some flood pixels to compute meaningful F1
    if flood_target.sum() == 0:
        return float('nan')

    # Evaluate over ALL pixels (including background)
    tp = int((flood_pred &  flood_target).sum())
    fp = int((flood_pred & ~flood_target).sum())
    fn = int((~flood_pred & flood_target).sum())

    if tp + fp + fn == 0:
        return float('nan')
    return float(2 * tp / (2 * tp + fp + fn + 1e-8))


def chip_flood_fraction(target: np.ndarray) -> float:
    valid = target != 0
    if not valid.any():
        return 0.0
    return float((target[valid] >= 1).mean())


# ── Building density from SAR variance ───────────────────────────────────────
# WorldCover GeoTIFF is 100GB+ globally and not practical to query locally.
# Instead we use the SAR local variance channels (ch 2+3 in preprocessed chips)
# as a proxy for building density:
#   - Shadow regions (behind buildings): near-zero variance
#   - Building surfaces (double-bounce): high variance
#   - Open water: low but non-zero variance (speckle)
# Mean variance in non-flooded areas correlates strongly with building density.

BUILDING_DENSITY_DIR = None  # set to 'data/building_density' after GEE export
                              # if None, falls back to SAR variance proxy


def get_building_density(
    image: np.ndarray,
    target: np.ndarray,
    chip_path: str,
    density_dir: str | None,
) -> float:
    """Get building density for a chip.

    If density_dir is set and a matching raster exists, samples the
    Google Buildings count raster at the chip's bounding box.
    Otherwise falls back to SAR variance proxy.
    """
    if density_dir is not None:
        event = get_event_from_path(chip_path)
        raster = Path(density_dir) / f'{event}_building_density.tif'
        if raster.exists():
            # Read the chip's geographic bounds from the original SAR file
            orig_path = Path(ORIG_SAR_DIR) / Path(chip_path).parent.parent.name \
                        / 'SAR' / Path(chip_path).name.replace('.npy', '.tif')
            if orig_path.exists():
                with rasterio.open(orig_path) as src:
                    bounds = src.bounds
                with rasterio.open(raster) as bd_src:
                    from rasterio.windows import from_bounds
                    win = from_bounds(*bounds, bd_src.transform)
                    try:
                        data = bd_src.read(1, window=win)
                        if data.size > 0:
                            return float(data.mean())
                    except Exception:
                        pass
    # Fallback: SAR variance proxy
    return estimate_building_density_from_variance(image, target)


def estimate_building_density_from_variance(
    image: np.ndarray,
    target: np.ndarray,
) -> float:
    """Proxy for building density using SAR variance in non-flood areas."""
    var_vv = image[2]
    var_vh = image[3]
    mean_var = (var_vv + var_vh) / 2.0
    non_flood = (target == 0) | (target == 3)
    if non_flood.sum() < 100:
        non_flood = np.ones_like(target, dtype=bool)
    return float(mean_var[non_flood].mean())


def get_event_from_path(chip_path: str) -> str:
    """Extract event name from chip filename."""
    name = Path(chip_path).stem
    parts = name.split('_')
    return parts[1] if len(parts) > 1 else 'unknown'


def get_folder_from_path(chip_path: str) -> str:
    """Extract folder (01_NF, 02_FO, 03_FU) from path."""
    p = Path(chip_path)
    for part in p.parts:
        if part in ('01_NF', '02_FO', '03_FU'):
            return part
    return 'unknown'


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis(checkpoint: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
    device = (torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))

    # Load model
    print(f'Loading checkpoint: {checkpoint}')
    ckpt  = torch.load(checkpoint, map_location='cpu')
    model = build_unet(n_channels=4, n_classes=N_CLASSES,
                       encoder_name='resnet34', encoder_weights=None)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()
    print(f'Loaded epoch {ckpt["epoch"]} F1={ckpt["metric"]:.4f}')

    # Test dataset
    _, test_idx = make_urban_event_split(
        URBAN_SAR_DIR,
        train_events=FINETUNE_TRAIN_EVENTS,
        test_events=FINETUNE_TEST_EVENTS,
    )
    ds = FastUrbanSARFloods(
        root=URBAN_SAR_DIR,
        split_indices=test_idx,
        transform=val_transforms(512),
        binary=False,
    )
    print(f'Test chips: {len(ds)}')

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Per-chip analysis
    rows = []
    print('Running per-chip analysis...')

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 200 == 0:
                print(f'  {i}/{len(ds)}')

            image   = batch['image'].to(device)  # (1, 4, H, W)
            target  = batch['mask'][0].numpy()    # (H, W)
            path    = batch['chip_path'][0]

            logits  = model(image)
            pred    = logits[0].argmax(dim=0).cpu().numpy()  # (H, W)

            f1      = chip_f1(pred, target)
            ff      = chip_flood_fraction(target)
            bd      = get_building_density(
                          image[0].cpu().numpy(), target,
                          path, BUILDING_DENSITY_DIR)
            event   = get_event_from_path(path)
            folder  = get_folder_from_path(path)

            # Dominant flood type
            urban_px = int((target == 2).sum())
            open_px  = int((target == 1).sum())
            flood_type = 'urban' if urban_px > open_px else 'open'

            rows.append({
                'chip_path':        path,
                'event':            event,
                'folder':           folder,
                'flood_type':       flood_type,
                'f1':               round(f1, 4) if not np.isnan(f1) else -1,
                'flood_fraction':   round(ff, 4),
                'building_density': round(bd, 6),
                'urban_flood_px':   urban_px,
                'open_flood_px':    open_px,
            })

    # Filter to chips with valid F1 (has some flood pixels)
    valid_rows = [r for r in rows if r['f1'] >= 0]
    print(f'\nChips with flood pixels (valid F1): {len(valid_rows)}/{len(rows)}')

    # Write CSV
    csv_path = out / 'failure_analysis.csv'
    fields = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'Saved: {csv_path}')

    # ── Plots ─────────────────────────────────────────────────────────────────

    f1s = np.array([r['f1'] for r in valid_rows])
    bds = np.array([r['building_density'] for r in valid_rows])
    ffs = np.array([r['flood_fraction'] for r in valid_rows])
    events = [r['event'] for r in valid_rows]
    ftypes = [r['flood_type'] for r in valid_rows]

    unique_events = sorted(set(events))
    event_colors  = plt.cm.tab10(np.linspace(0, 1, len(unique_events)))
    event_color_map = dict(zip(unique_events, event_colors))

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    # 1. F1 vs building density scatter
    ax1 = fig.add_subplot(gs[0, :2])
    for evt in unique_events:
        mask = np.array([e == evt for e in events])
        ax1.scatter(bds[mask], f1s[mask], alpha=0.4, s=15,
                    color=event_color_map[evt], label=evt)
    # Trend line
    z = np.polyfit(bds, f1s, 1)
    x_line = np.linspace(bds.min(), bds.max(), 100)
    ax1.plot(x_line, np.polyval(z, x_line), 'k--', linewidth=1.5,
             label=f'trend (slope={z[0]:.1f})')
    ax1.set_xlabel('Building density proxy (SAR variance in non-flood areas)')
    ax1.set_ylabel('Per-chip F1 (flood classes 1+2)')
    ax1.set_title('F1 vs building density — test set chips')
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_ylim(-0.05, 1.05)

    # 2. Per-event F1 box plot
    ax2 = fig.add_subplot(gs[0, 2])
    event_f1s = [[r['f1'] for r in valid_rows if r['event'] == e]
                 for e in unique_events]
    bp = ax2.boxplot(event_f1s, labels=unique_events, patch_artist=True,
                     medianprops={'color': 'black', 'linewidth': 2})
    for patch, evt in zip(bp['boxes'], unique_events):
        patch.set_facecolor(event_color_map[evt])
        patch.set_alpha(0.7)
    ax2.set_ylabel('Per-chip F1')
    ax2.set_title('F1 by event (test cities)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(-0.05, 1.05)

    # 3. F1 vs flood fraction
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(ffs, f1s, c=bds, cmap='YlOrRd', alpha=0.5, s=15)
    plt.colorbar(scatter, ax=ax3, label='Building density')
    ax3.set_xlabel('Flood fraction (fraction of valid pixels that are flood)')
    ax3.set_ylabel('Per-chip F1')
    ax3.set_title('F1 vs flood fraction\n(colour = building density)')

    # 4. F1 distribution by flood type (urban vs open)
    ax4 = fig.add_subplot(gs[1, 1])
    urban_f1s = [r['f1'] for r in valid_rows if r['flood_type'] == 'urban']
    open_f1s  = [r['f1'] for r in valid_rows if r['flood_type'] == 'open']
    ax4.hist(urban_f1s, bins=30, alpha=0.6, label=f'Urban flood (n={len(urban_f1s)})',
             color='#D85A30', density=True)
    ax4.hist(open_f1s,  bins=30, alpha=0.6, label=f'Open flood (n={len(open_f1s)})',
             color='#1D9E75', density=True)
    ax4.axvline(np.mean(urban_f1s), color='#D85A30', linestyle='--', linewidth=2)
    ax4.axvline(np.mean(open_f1s),  color='#1D9E75', linestyle='--', linewidth=2)
    ax4.set_xlabel('Per-chip F1')
    ax4.set_ylabel('Density')
    ax4.set_title('F1 distribution: urban vs open flood chips')
    ax4.legend(fontsize=8)

    # 5. Summary statistics table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    summary = [
        ['Metric', 'Value'],
        ['Total chips (test)', str(len(rows))],
        ['Chips with flood', str(len(valid_rows))],
        ['Mean F1 (all)', f'{f1s.mean():.3f}'],
        ['Median F1', f'{np.median(f1s):.3f}'],
        ['F1 std', f'{f1s.std():.3f}'],
        ['Mean F1 (urban chips)', f'{np.mean(urban_f1s):.3f}' if urban_f1s else 'N/A'],
        ['Mean F1 (open chips)',  f'{np.mean(open_f1s):.3f}'  if open_f1s  else 'N/A'],
        ['Corr(F1, bld density)', f'{np.corrcoef(f1s, bds)[0,1]:.3f}'],
        ['Corr(F1, flood frac)',  f'{np.corrcoef(f1s, ffs)[0,1]:.3f}'],
    ]
    tbl = ax5.table(cellText=summary[1:], colLabels=summary[0],
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    ax5.set_title('Summary statistics', pad=20)

    plt.suptitle('Failure mode analysis — UrbanSARFloods test split',
                 fontsize=13, y=1.01)
    plot_path = out / 'failure_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {plot_path}')

    # Print key findings
    print('\n── Key findings ──')
    print(f'Mean F1:          {f1s.mean():.3f}')
    print(f'Median F1:        {np.median(f1s):.3f}')
    if urban_f1s and open_f1s:
        print(f'Mean F1 urban:    {np.mean(urban_f1s):.3f}')
        print(f'Mean F1 open:     {np.mean(open_f1s):.3f}')
        print(f'Urban/open gap:   {np.mean(open_f1s)-np.mean(urban_f1s):.3f}')
    print(f'Corr(F1, density):{np.corrcoef(f1s, bds)[0,1]:.3f}')
    print(f'  (negative = higher density → lower F1, confirms urban challenge)')
    print(f'\nPer-event mean F1:')
    for evt in unique_events:
        ef1 = [r['f1'] for r in valid_rows if r['event'] == evt]
        if ef1:
            print(f'  {evt:<20} {np.mean(ef1):.3f}  (n={len(ef1)})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True,
                   help='Path to Stage 3 best.pt checkpoint')
    p.add_argument('--out_dir', default='runs/analysis')
    args = p.parse_args()
    run_analysis(args.checkpoint, args.out_dir)
