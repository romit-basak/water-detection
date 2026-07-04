"""M3: sim-vs-real contrast statistics — the corpus swarm's acceptance gate.

Real side: per-class Sentinel-1 dB statistics from raw UrbanSARFloods tifs
of the two urban test events (Hebei dense-urban, Sydney fragmented-urban).
Raw bands (verified empirically): 1-4 normalized, 5=VH_pre, 6=VV_pre,
7=VH_post, 8=VV_post (dB). Classes: GT 2=urban flood, 1=open flood;
confirmed non-flood comes from the same city's 01_NF chips (GT==0 in FU
chips is UNLABELED, not non-flood — never use it as a class).

Synthetic side: sarsim pilot wet/dry backprojections (from
runs/pilot_3way/sarsim_bp.npz), multilooked to 10 m — the synthetic
wet-dry change at flooded-street pixels is the analogue of the real
post-pre change at urban-flood pixels. Pilot scale (40 m -> 4x4 px) makes
this a spot check; the binding gate re-runs on A1's corpus-scale
calibration scenes against the targets JSON this script writes.

Run with the ROOT env (needs rasterio): uv run python scripts/m3_sim_vs_real.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/urban_sar_floods'
CITIES = ('Hebei', 'Sydney')
BANDS = {'vh_pre': 5, 'vv_pre': 6, 'vh_post': 7, 'vv_post': 8}


def collect_class_pixels(city: str, nf_every: int = 5):
    """Per-class dicts of {band: 1D array} for one city."""
    out = {'urban_flood': {b: [] for b in BANDS},
           'open_flood': {b: [] for b in BANDS},
           'non_flood': {b: [] for b in BANDS}}

    for p in sorted(glob.glob(str(RAW / f'03_FU/SAR/*{city}*_SAR.tif'))):
        with rasterio.open(p.replace('/SAR/', '/GT/')
                           .replace('_SAR.tif', '_GT.tif')) as f:
            g = f.read(1)
        masks = {'urban_flood': g == 2, 'open_flood': g == 1}
        if not any(m.any() for m in masks.values()):
            continue
        with rasterio.open(p) as f:
            a = f.read(list(BANDS.values())).astype(np.float32)
        finite = np.isfinite(a).all(axis=0)
        for cls, m in masks.items():
            mm = m & finite
            if mm.any():
                for i, b in enumerate(BANDS):
                    out[cls][b].append(a[i][mm])

    nf = sorted(glob.glob(str(RAW / f'01_NF/SAR/*{city}*_SAR.tif')))[::nf_every]
    rng = np.random.default_rng(0)
    for p in nf:
        with rasterio.open(p) as f:
            a = f.read(list(BANDS.values())).astype(np.float32)
        finite = np.isfinite(a).all(axis=0)
        idx = np.flatnonzero(finite.ravel())
        take = rng.choice(idx, size=min(4000, len(idx)), replace=False)
        for i, b in enumerate(BANDS):
            out['non_flood'][b].append(a[i].ravel()[take])

    return {cls: {b: np.concatenate(v) if v else np.array([])
                  for b, v in d.items()} for cls, d in out.items()}


def stats(v: np.ndarray) -> dict:
    if not len(v):
        return {}
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return {'n': int(len(v)), 'median': float(q50),
            'iqr': [float(q25), float(q75)]}


def synthetic_10m():
    """Multilook the sarsim pilot to 10 m; per-pixel wet-dry change (dB)."""
    z = np.load(ROOT / 'runs/pilot_3way/sarsim_bp.npz')
    dry, wet, axis = z['dry'], z['wet'], z['axis']
    pix = float(axis[1] - axis[0])
    block = max(int(round(10.0 / pix)), 1)
    n = (dry.shape[0] // block) * block

    def look(img):
        b = img[:n, :n].reshape(n // block, block, n // block, block)
        return b.mean(axis=(1, 3))          # incoherent multilook of power

    d10, w10 = look(dry), look(wet)
    y10 = axis[:n].reshape(-1, block).mean(1)
    delta = 10 * np.log10(np.maximum(w10, 1e-30)
                          / np.maximum(d10, 1e-30))
    # flooded-street rows: corridor y in [-14, 2] (scene truth)
    street = (y10 >= -14) & (y10 <= 2)
    return {'delta_flooded_street_px': [float(v)
                                        for v in delta[street].ravel()],
            'delta_all_px': [float(v) for v in delta.ravel()],
            'block_px': block}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(ROOT / 'runs/m3_sim_vs_real'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {'bands': BANDS, 'cities': {}}
    data = {}
    for city in CITIES:
        data[city] = collect_class_pixels(city)
        cs = {}
        for cls, d in data[city].items():
            e = {b: stats(v) for b, v in d.items()}
            if len(d['vv_post']) and len(d['vv_pre']):
                e['dvv'] = stats(d['vv_post'] - d['vv_pre'])
                e['dvh'] = stats(d['vh_post'] - d['vh_pre'])
            cs[cls] = e
        report['cities'][city] = cs
        for cls in ('urban_flood', 'open_flood', 'non_flood'):
            s = cs[cls]
            if s.get('vv_post'):
                print(f'{city:8s} {cls:12s} n={s["vv_post"]["n"]:>9,} '
                      f'VVpost {s["vv_post"]["median"]:+6.1f} dB  '
                      f'dVV {s["dvv"]["median"]:+5.1f} '
                      f'[{s["dvv"]["iqr"][0]:+5.1f},{s["dvv"]["iqr"][1]:+5.1f}]')

    # acceptance targets for the corpus (A1 gate 2): per-city class
    # contrasts vs non-flood and change statistics
    targets = {}
    for city in CITIES:
        cs = report['cities'][city]
        nf_vv = cs['non_flood']['vv_post']['median']
        targets[city] = {
            'contrast_urban_flood_vs_nf_db':
                cs['urban_flood']['vv_post']['median'] - nf_vv,
            'contrast_open_flood_vs_nf_db':
                cs['open_flood']['vv_post']['median'] - nf_vv,
            'dvv_urban_flood': cs['urban_flood']['dvv'],
            'dvv_open_flood': cs['open_flood']['dvv'],
            'dvv_non_flood': cs['non_flood']['dvv'],
        }
    report['targets'] = targets

    syn = synthetic_10m()
    report['synthetic_pilot_10m'] = {
        'delta_flooded_street_px': syn['delta_flooded_street_px'],
        'note': 'sarsim pilot wet-dry at 10 m; analogue of real dVV at '
                'urban-flood pixels. 40 m scene -> spot check only; the '
                'binding comparison uses corpus-scale calibration scenes.'}

    (out / 'targets.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    colors = {'urban_flood': 'tab:red', 'open_flood': 'tab:blue',
              'non_flood': 'tab:gray'}
    for ax, city in zip(axes[:2], CITIES):
        for cls, col in colors.items():
            v = data[city][cls]['vv_post']
            if len(v):
                ax.hist(v, bins=80, range=(-35, 5), density=True, alpha=0.5,
                        color=col, label=f'{cls} (n={len(v):,})')
        ax.set_title(f'{city}: post-event VV (dB)')
        ax.set_xlabel('VV backscatter (dB)')
        ax.legend(fontsize=8)
    ax = axes[2]
    for city, ls in zip(CITIES, ('-', '--')):
        for cls, col in colors.items():
            v = data[city][cls]['vv_post'] - data[city][cls]['vv_pre']
            if len(v):
                ax.hist(v, bins=80, range=(-15, 15), density=True,
                        histtype='step', ls=ls, color=col,
                        label=f'{city} {cls}')
    sd = np.array(syn['delta_flooded_street_px'])
    for v in sd:
        ax.axvline(v, color='tab:green', alpha=0.5, lw=1)
    ax.axvline(np.median(sd), color='tab:green', lw=2.5,
               label='sarsim pilot 10 m flooded-street px (green)')
    ax.set_title('post−pre change ΔVV (dB); synthetic wet−dry overlaid')
    ax.set_xlabel('ΔVV (dB)')
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / 'm3_sim_vs_real.png', dpi=110, bbox_inches='tight')
    print(out / 'm3_sim_vs_real.png')


if __name__ == '__main__':
    main()
