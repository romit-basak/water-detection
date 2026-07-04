"""M1 driver: render the flood pilot through RaySAR and form image layers.

Renders pilot_{dry,wet}.pov with the locally built RaySAR POV-Ray (each in
its own working directory — Contributions.txt is a hardcoded CWD filename),
then bins contributions into single/double/all layers and writes the pilot
artifact (npz + figure) to runs/raysar_pilot/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_pilot_pov import pov_scene, INCIDENCE_DEG  # noqa: E402

from sarsim import raysar  # noqa: E402

POVRAY = ('/Users/romitbasak/Projects/RaySAR/POV-Ray/'
          'POV-Ray_3.7_Source_Code_for_RaySAR/unix/povray')


def render(workdir: Path, water: bool, size: int, povray: str) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    pov = workdir / 'scene.pov'
    pov.write_text(pov_scene(water))
    t0 = time.perf_counter()
    # no +A: clean one-ray-per-pixel sampling for reproducible bin counts
    r = subprocess.run(
        [povray, '+Iscene.pov', f'+W{size}', f'+H{size}', '-D',
         '+Oscene.png'],
        cwd=workdir, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    contrib = workdir / 'Contributions.txt'
    if r.returncode != 0 or not contrib.exists():
        sys.stderr.write(r.stderr[-3000:])
        raise RuntimeError(f'povray failed in {workdir}')
    c = raysar.load_contributions(contrib)
    return {'c': c, 'wall_s': wall}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='runs/raysar_pilot')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--pix', type=float, default=0.25, help='bin size (m)')
    ap.add_argument('--povray', default=POVRAY)
    args = ap.parse_args()
    out = Path(args.out)

    res, layers = {}, {}
    for water, name in [(False, 'dry'), (True, 'wet')]:
        res[name] = render(out / name, water, args.size, args.povray)
        c = res[name]['c']
        n_by_level = np.bincount(c.level, minlength=6)
        print(f'{name}: {len(c.az)} contributions '
              f'(L1={n_by_level[1]}, L2={n_by_level[2]}, '
              f'L3={n_by_level[3]}), render {res[name]["wall_s"]:.2f}s')

    # shared bounds so dry/wet grids align bin-for-bin
    both = [res[n]['c'] for n in ('dry', 'wet')]
    ground = [c.ra / np.sin(np.deg2rad(INCIDENCE_DEG)) for c in both]
    bounds = (min(c.az.min() for c in both), max(c.az.max() for c in both),
              min(g.min() for g in ground), max(g.max() for g in ground))
    for name in ('dry', 'wet'):
        layers[name], extent = raysar.form_layers(
            res[name]['c'], a_pix=args.pix, r_pix=args.pix,
            incidence_deg=INCIDENCE_DEG, bounds=bounds)

    np.savez_compressed(
        out / 'raysar_pilot_layers.npz',
        extent=np.array(extent),
        **{f'{n}_{k}': v for n in layers for k, v in layers[n].items()},
        wall_s=np.array([res['dry']['wall_s'], res['wet']['wall_s']]))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, name in enumerate(('dry', 'wet')):
        for j, k in enumerate(('single', 'double', 'all')):
            db = raysar.to_db(layers[name][k], floor_db=-50)
            im = axes[i, j].imshow(db, cmap='gray', vmin=-50, vmax=0,
                                   extent=[extent[0], extent[1],
                                           extent[3], extent[2]],
                                   aspect='auto')
            axes[i, j].set_title(f'{name}: {k} (dB rel peak)')
            axes[i, j].set_xlabel('azimuth (m)')
            axes[i, j].set_ylabel('ground range (m)')
    plt.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle('RaySAR flood pilot — bounce-level layers '
                 f'({INCIDENCE_DEG} deg incidence)')
    plt.savefig(out / 'raysar_pilot.png', dpi=110, bbox_inches='tight')
    print(out / 'raysar_pilot.png')

    # M1 gate: double-bounce energy appears with water, localized in range
    d_dry = layers['dry']['double'].sum()
    d_wet = layers['wet']['double'].sum()
    print(f'double-bounce energy: dry={d_dry:.3f} wet={d_wet:.3f} '
          f'ratio={d_wet / max(d_dry, 1e-12):.1f}x')


if __name__ == '__main__':
    main()
