"""Diagnostic: is the uf_dvv gate failure a materials problem or a
classification-band problem? Renders the same 8 calibration seeds/bins
once and reports median dVV/coherence at several building-distance
cutoffs, instead of the single bdist<=20 band calibrate_corpus.py gates
on. If a narrower band (closer to the true corner-reflector footprint)
clears the gate while bdist<=20 doesn't, the fix is the classification
band, not more material tuning.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion

from sarsim.city import CityParams, generate_city
from sarsim.corpus import render_pair

BINS = ['near_peak', 'near_peak', 'mid_recession', 'near_peak',
        'mid_recession', 'receded_wet', 'dry->wet', 'near_peak']
SEEDS = list(range(16))
THRESHOLDS = [4, 8, 12, 16, 20]


def building_dist(footprints, extent_m, n_px):
    ax = -extent_m / 2 + (extent_m / n_px) * (np.arange(n_px) + 0.5)
    X, Y = np.meshgrid(ax, ax)
    d = np.full(X.shape, np.inf)
    for (bx, by, ex, ey) in footprints:
        d = np.minimum(d, np.hypot(np.maximum(np.abs(X - bx) - ex, 0),
                                   np.maximum(np.abs(Y - by) - ey, 0)))
    return d


def main():
    pooled_dvv = {t: [] for t in THRESHOLDS}
    pooled_coh = {t: [] for t in THRESHOLDS}
    pooled_n = {t: 0 for t in THRESHOLDS}
    of_dvv, of_coh = [], []

    for i, seed in enumerate(SEEDS):
        fb = BINS[i % len(BINS)]
        _, _, fps, _ = generate_city(CityParams(seed=seed, extent_m=320.0))
        item, meta, wall = render_pair(seed, extent_m=320.0, n_rays_side=384,
                                       force_bin=fb)
        n_px = item['mask_standing_t2'].shape[0]
        bdist = building_dist(fps, 320.0, n_px)
        dvv = item['vv_t2_db'].astype(np.float64) \
            - item['vv_t1_db'].astype(np.float64)
        coh = item['coh'].astype(np.float64)
        standing = item['mask_standing_t2'].astype(bool)

        of = binary_erosion(standing, iterations=2) & (bdist > 50)
        if of.any():
            of_dvv.append(dvv[of]); of_coh.append(coh[of])

        counts = []
        for t in THRESHOLDS:
            m = standing & (bdist <= t)
            counts.append(int(m.sum()))
            if m.any():
                pooled_dvv[t].append(dvv[m])
                pooled_coh[t].append(coh[m])
                pooled_n[t] += int(m.sum())
        m20 = standing & (bdist <= 20)
        seed_med = float(np.median(dvv[m20])) if m20.any() else float('nan')
        print(f'seed {seed} [{meta["pair"]["scenario"]:>24s}] {wall:.0f}s '
              f'inc={meta["incidence_deg"]:.0f} az={meta["look_az_deg"]:.0f} '
              f'counts@{THRESHOLDS}={counts} seed_uf20_dvv_med={seed_med:+.2f}',
              flush=True)

    print(f'\n{"bdist<=":8s} {"n":>7s} {"dvv_med":>8s} {"dvv_IQR":>17s}'
          f' {"coh_med":>8s} {"coh_IQR":>17s}')
    for t in THRESHOLDS:
        vs = [v for v in pooled_dvv[t] if len(v)]
        cs = [v for v in pooled_coh[t] if len(v)]
        if not vs:
            print(f'{t:8d}       0  (no pixels)')
            continue
        v = np.concatenate(vs)
        c = np.concatenate(cs)
        dq25, dq50, dq75 = np.percentile(v, [25, 50, 75])
        cq25, cq50, cq75 = np.percentile(c, [25, 50, 75])
        print(f'{t:8d} {len(v):7,d} {dq50:+8.2f} [{dq25:+7.2f},{dq75:+6.2f}]'
              f' {cq50:8.2f} [{cq25:7.2f},{cq75:6.2f}]')

    if of_dvv:
        v = np.concatenate(of_dvv)
        print(f'\n(of, bdist>50, for reference) n={len(v):,} '
              f'dvv_med={np.median(v):+.2f}')
    print('\ngate: uf_dvv median in [+0.10,+3.10], uf_coh median in '
          '[0.15,0.32]')


if __name__ == '__main__':
    main()
