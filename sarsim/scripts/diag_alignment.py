"""Decisive test: does uf_dvv go strongly positive when look_az is FORCED
to broadside-align with the axis-aligned building grid (0 deg = wall normal
direction)? If yes, the dihedral energy budget is fine (matches Gate 1) and
the calibration median failure is alignment-fraction dilution across random
look_az draws. If no even under ideal alignment, it's a deeper energy-budget
issue in tracer_ref's bounce accounting.

Duplicates render_pair's internals with look_az/incidence forced instead of
randomly drawn.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.ndimage import binary_erosion

from sarsim.city import (CityParams, GAMMA_SYS, SurfaceField, WaterAwareField,
                         draw_material, generate_city, gt_masks,
                         level_at_frac, with_water)
from sarsim.corpus import corpus_config, multilook_and_coherence
from sarsim.hydro import HydroParams, sample_pair
from sarsim.sbr import simulate_sbr
from sarsim import recon


def building_dist(footprints, extent_m, n_px):
    ax = -extent_m / 2 + (extent_m / n_px) * (np.arange(n_px) + 0.5)
    X, Y = np.meshgrid(ax, ax)
    d = np.full(X.shape, np.inf)
    for (bx, by, ex, ey) in footprints:
        d = np.minimum(d, np.hypot(np.maximum(np.abs(X - bx) - ex, 0),
                                   np.maximum(np.abs(Y - by) - ey, 0)))
    return d


def render_forced_az(seed, incidence, look_az, extent_m=320.0,
                     n_rays_side=384, force_bin='near_peak'):
    scene, meta, footprints, kinds = generate_city(
        CityParams(seed=seed, extent_m=extent_m))
    rng_h = np.random.default_rng(seed ^ 0x5EED_0F1D)
    pervious = float(meta['open_prob'] + 0.10)
    hydro = HydroParams.draw(rng_h, meta['height_mean_m'],
                             meta['fill_prob'], meta['slope_pct'], pervious)
    pair = sample_pair(rng_h, hydro)
    for _ in range(200):
        if force_bin in pair.scenario:
            break
        pair = sample_pair(rng_h, hydro)

    cfg = corpus_config(extent_m, incidence, look_az, 2.5)
    lam = cfg.sarsystem.wavelength

    jitter = (seed * 7919 + 13) % 2**31
    aux_rng = np.random.default_rng(seed ^ 0xAA77)
    water_mat = draw_material(aux_rng, 'water')
    gamma_sys = float(aux_rng.uniform(*GAMMA_SYS))

    event = pair.has_flood or pair.scenario == 'receded_wet'
    imgs = {}
    for tag, state, other, wfrac in (
            ('t1', pair.state_t1, pair.state_t2, pair.w1_frac),
            ('t2', pair.state_t2, pair.state_t1, pair.w2_frac)):
        level = level_at_frac(meta, wfrac)
        field_rng = np.random.default_rng(seed ^ 0xF1E1D)
        common = dict(meta=meta, state=state, other_state=other,
                      level_now=level, wavelength=lam,
                      acq_seed=(seed * 2 + (tag == 't2')) & 0x7FFFFFFF,
                      scene_rng=field_rng, apply_noise=(tag == 't2'),
                      sat_enabled=(tag == 't2' and event),
                      gamma_sys=gamma_sys)
        if level is not None:
            sc, kk = with_water(scene, kinds, level, water_mat)
            fld = WaterAwareField(kinds, n_base=len(kinds), **common)
        else:
            sc, fld = scene, SurfaceField(kinds, **common)
        S = simulate_sbr(cfg, sc, n_rays_side=n_rays_side, max_depth=3,
                         shadow_rays=True, seed=jitter, backend='mi_llvm',
                         accumulator='binned', bin_frac_of_lambda=1.0,
                         surface=fld, fixed_rays=True)
        img, axis = recon.backproject(S, cfg, window='hann')
        imgs[tag] = np.asarray(img)

    p1, p2, coh, n_px = multilook_and_coherence(imgs['t1'], imgs['t2'],
                                                np.asarray(axis))
    dvv = 10 * np.log10(np.maximum(p2, 1e-30)) - 10 * np.log10(np.maximum(p1, 1e-30))
    standing = gt_masks(meta, footprints, n_px, level_at_frac(meta, pair.w2_frac),
                        meta['w_peak_level'])['mask_standing_t2'].astype(bool)
    bdist = building_dist(footprints, extent_m, n_px)
    uf = standing & (bdist <= 20)
    if uf.any():
        print(f'seed={seed} inc={incidence:.0f} az={look_az:.0f} '
              f'n_uf={int(uf.sum())} uf_dvv_med={np.median(dvv[uf]):+.2f} '
              f'scenario={pair.scenario}')
    else:
        print(f'seed={seed} inc={incidence:.0f} az={look_az:.0f} n_uf=0 '
              f'scenario={pair.scenario}')


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    inc = float(sys.argv[2]) if len(sys.argv) > 2 else 37.5
    for az in (0.0, 45.0, 90.0):
        render_forced_az(seed, inc, az)
