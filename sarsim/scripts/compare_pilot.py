"""M2: three-way comparison artifact on the shared flood-pilot scene.

One figure + table + JSON comparing sarsim (phase history + backprojection),
RaySAR (ray-contribution histogram layers), and Benji's SAR_RayTracer
(radar-viewpoint stylized intensity) on identical geometry ± street water.

Every simulator's output is reduced to the same scene coordinate: y (m),
the street/range axis, radar looking from -y at 37.5 deg incidence.
Scene truth: wall base (corner line) y=2, water corridor y in [-14, 2],
buildings y in [2, 10], wall layover reaches y >= -5.8, building shadow
y in [10, 17.8]. Metric windows follow sarsim's Gate 2 exactly:
wall strip y in [1, 2], open water y in [-12, -8], |x| <= 16.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from sarsim import raysar, recon
from sarsim.config import SarSystem, SimConfig, Trajectory

ROOT = Path(__file__).resolve().parents[2]        # water-detection/
BENJI = Path('/Users/romitbasak/Projects/SAR_RayTracer')
POVRAY = Path('/Users/romitbasak/Projects/RaySAR/POV-Ray/'
              'POV-Ray_3.7_Source_Code_for_RaySAR/unix/povray')

W_STRIP = (1.0, 2.0)      # wall-water corner strip (y)
W_OPEN = (-12.0, -8.0)    # open water beyond layover (y)
W_SHADOW = (11.0, 15.0)   # behind buildings (y)
X_HALF = 16.0


BX = np.linspace(-12, 12, 5)      # building centers (scene truth)


def db(p_num: float, p_den: float) -> float:
    """Power ratio in dB, floored at -40 (RaySAR shadow bins are exact 0)."""
    return float(10 * np.log10(max(p_num, p_den * 1e-4) / p_den))


def band_power(img2: np.ndarray, y: np.ndarray, x: np.ndarray,
               ywin: tuple[float, float],
               footprints_only: bool = False) -> float:
    """Mean POWER of img2 (already |.|^2) in a y-band, |x| <= X_HALF.

    footprints_only: restrict x to the five building footprints (shadow
    metric — the gaps between buildings are lit ground and dilute it).
    """
    xm = np.abs(x) <= X_HALF
    if footprints_only:
        xm &= np.min(np.abs(x[None, :] - BX[:, None]), axis=0) <= 2.0
    m = ((y[:, None] >= ywin[0]) & (y[:, None] <= ywin[1]) & xm[None, :])
    return float(img2[m].mean())


def localization_width(img2_wet, img2_dry, y, x) -> float:
    """Width (m) of the wet/dry gain region >= half its peak dB, y in [-5,5].

    Line-like double bounce -> ~1-2 m; corridor-wide brightening -> >8 m.
    """
    xm = np.abs(x) <= X_HALF
    prof = 10 * np.log10(np.maximum(img2_wet[:, xm].mean(1), 1e-30)
                         / np.maximum(img2_dry[:, xm].mean(1), 1e-30))
    sel = (y >= -5) & (y <= 5)
    ps, ys = prof[sel], y[sel]
    peak = ps.max()
    if peak <= 0:
        return float('nan')
    above = ps >= peak / 2
    return float(np.abs(np.diff(ys).mean()) * above.sum())


# ---------------------------------------------------------------- sarsim
def load_sarsim():
    cfg = SimConfig(       # identical to cli.flood_pilot (Gate 2)
        sarsystem=SarSystem(carrier_ghz=5.405, bandwidth_mhz=150.0,
                            n_freq=256),
        trajectory=Trajectory(mode='circular_spotlight',
                              slant_range_m=10000.0, depression_deg=52.5,
                              az_start_deg=269.2, az_end_deg=270.8,
                              n_pulses=128),
        scene_extent_m=40.0, pixel_m=0.25, name='flood_pilot')
    out = {}
    for tag, key in [('dry', 'dry'), ('wet', 'water')]:
        S = np.load(ROOT / f'runs/sarsim_flood_pilot/S_{key}.npz')['S']
        img, axis = recon.backproject(S.astype(np.complex128), cfg,
                                      window='hann')
        out[tag] = np.abs(img) ** 2
    # rows follow axis as y, cols as x (cli.py meshgrid convention)
    return out, np.asarray(axis), np.asarray(axis)


# ----------------------------------------------------------------- benji
def read_p3(path) -> np.ndarray:
    tok = Path(path).read_text().split()
    w, h, mx = int(tok[1]), int(tok[2]), int(tok[3])
    a = np.array(tok[4:4 + w * h * 3], float).reshape(h, w, 3) / mx
    return a.mean(2)


def load_benji(retime: bool):
    wall = None
    if retime:
        t0 = time.perf_counter()
        subprocess.run(['./build/SAR_RayTracer', '10'], cwd=BENJI,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        wall = time.perf_counter() - t0
    imgs = {}
    for tag, f in [('dry', 'pilot_dry.ppm'), ('wet', 'pilot_wet.ppm')]:
        g = read_p3(BENJI / f)
        imgs[tag] = g ** 2            # undo gamma-2: PPM stores sqrt(linear)
    # row/col <-> scene map from the optical reference: ground-plane corners
    # at content rows [92, 425] = y [+20, -20], cols [42, 469] = x [-20, +20]
    n = imgs['dry'].shape[0]
    rows, cols = np.arange(n), np.arange(imgs['dry'].shape[1])
    y = 20.0 - 40.0 * (rows - 92) / (425 - 92)
    x = -20.0 + 40.0 * (cols - 42) / (469 - 42)
    return imgs, y, x, wall


# ---------------------------------------------------------------- raysar
def load_raysar(retime: bool):
    wall = None
    if retime:
        wd = ROOT / 'runs/raysar_pilot/wet'
        t0 = time.perf_counter()
        subprocess.run([str(POVRAY), '+Iscene.pov', '+W512', '+H512', '-D',
                        '+Oscene.png'], cwd=wd, capture_output=True,
                       check=True)
        wall = time.perf_counter() - t0
    z = np.load(ROOT / 'runs/raysar_pilot/raysar_pilot_layers.npz')
    a_min, a_max, r_min, r_max = z['extent']
    imgs = {t: z[f'{t}_all'] for t in ('dry', 'wet')}
    dbl = z['wet_double']
    nr, na = imgs['dry'].shape
    g = r_min + (np.arange(nr) + 0.5) * (r_max - r_min) / nr
    x = a_min + (np.arange(na) + 0.5) * (a_max - a_min) / na
    # anchor: corner line (wet double-bounce peak row) is scene y = +2;
    # ground range increases with +y (radar at -y)
    g_corner = g[dbl.sum(1).argmax()]
    y = g - g_corner + 2.0
    dblayers = {t: z[f'{t}_double'] for t in ('dry', 'wet')}
    return imgs, dblayers, y, x, wall


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(ROOT / 'runs/pilot_3way'))
    ap.add_argument('--sarsim-time', type=float, default=None,
                    help='wall seconds for one sarsim flood-pilot sim+BP')
    ap.add_argument('--no-retime', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sims = {}

    imgs, y, x = load_sarsim()[0:3]
    (imgs_s, y_s, x_s) = load_sarsim()
    sims['sarsim'] = dict(
        imgs=imgs_s, y=y_s, x=x_s, wall=args.sarsim_time,
        model='phase history (SBR) + backprojection',
        double_layer=None)

    imgs_b, y_b, x_b, wall_b = load_benji(retime=not args.no_retime)
    sims['benji'] = dict(imgs=imgs_b, y=y_b, x=x_b, wall=wall_b,
                         model='stylized intensity (colocated light)',
                         double_layer=None)

    imgs_r, dbl_r, y_r, x_r, wall_r = load_raysar(retime=not args.no_retime)
    sims['raysar'] = dict(imgs=imgs_r, y=y_r, x=x_r, wall=wall_r,
                          model='ray-contribution histogram layers',
                          double_layer=dbl_r)

    table = {}
    for name, s in sims.items():
        wet, dry = s['imgs']['wet'], s['imgs']['dry']
        yv, xv = s['y'], s['x']
        row = {
            'strip_gain_db': db(band_power(wet, yv, xv, W_STRIP),
                                band_power(dry, yv, xv, W_STRIP)),
            'open_water_db': db(band_power(wet, yv, xv, W_OPEN),
                                band_power(dry, yv, xv, W_OPEN)),
            'shadow_vs_soil_db': db(
                band_power(dry, yv, xv, W_SHADOW, footprints_only=True),
                band_power(dry, yv, xv, W_OPEN)),
            'gain_halfwidth_m': localization_width(wet, dry, yv, xv),
            'wall_s': s['wall'],
            'signal_model': s['model'],
        }
        if s['double_layer'] is not None:
            d = s['double_layer']
            row['double_energy_ratio'] = float(
                d['wet'].sum() / max(d['dry'].sum(), 1e-12))
        table[name] = row

    # power images + axis for downstream use (M3 runs in the root env
    # without mitsuba and can't backproject)
    np.savez_compressed(out / 'sarsim_bp.npz',
                        dry=sims['sarsim']['imgs']['dry'],
                        wet=sims['sarsim']['imgs']['wet'],
                        axis=sims['sarsim']['y'])

    (out / 'metrics_3way.json').write_text(json.dumps(table, indent=2))
    hdr = (f'{"sim":8s} {"strip":>7s} {"open":>7s} {"shadow":>7s} '
           f'{"width":>6s} {"time":>7s}  model')
    print(hdr)
    for name, r in table.items():
        t = f'{r["wall_s"]:.1f}s' if r['wall_s'] else '—'
        print(f'{name:8s} {r["strip_gain_db"]:+6.1f} {r["open_water_db"]:+6.1f}'
              f' {r["shadow_vs_soil_db"]:+6.1f} {r["gain_halfwidth_m"]:5.1f}m'
              f' {t:>7s}  {r["signal_model"]}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    for i, name in enumerate(sims):
        s = sims[name]
        # shared display convention: +y (away from radar) UP in every row
        yv, imgs_i = s['y'], s['imgs']
        if yv[0] < yv[-1]:
            yv = yv[::-1]
            imgs_i = {k: v[::-1] for k, v in imgs_i.items()}
        ref = max(imgs_i['wet'].max(), imgs_i['dry'].max())
        ext = [s['x'][0], s['x'][-1], yv[-1], yv[0]]
        for j, tag in enumerate(('dry', 'wet')):
            axes[i, j].imshow(
                10 * np.log10(np.maximum(imgs_i[tag] / ref, 1e-6)),
                extent=ext, vmin=-45, vmax=0, cmap='gray', aspect='auto')
            axes[i, j].set_title(f'{name}: {tag} (dB)')
        d = 10 * np.log10(np.maximum(imgs_i['wet'], 1e-30)
                          / np.maximum(imgs_i['dry'], 1e-30))
        axes[i, 2].imshow(d, extent=ext, vmin=-15, vmax=15,
                          cmap='RdBu_r', aspect='auto')
        axes[i, 2].set_title(f'{name}: wet − dry (dB)')
        for j in range(3):
            axes[i, j].set_ylim(-20, 20)
            axes[i, j].axhline(2.0, color='c', lw=0.5, ls='--')
            axes[i, j].set_ylabel('y (m)  [corner at +2]')
    fig.suptitle('Flood pilot ± street water — identical geometry, '
                 'three simulators (radar from −y, 37.5° incidence)')
    plt.tight_layout()
    plt.savefig(out / 'pilot_3way.png', dpi=110, bbox_inches='tight')
    print(out / 'pilot_3way.png')


if __name__ == '__main__':
    main()
