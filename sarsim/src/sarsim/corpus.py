"""Corpus rendering v2: acquisition PAIRS -> intensity + coherence chips.

One seed = one scene + one hydrograph + one sampled acquisition pair
(sarsim.hydro). Both acquisitions share geometry, base materials, and the
per-pulse ray jitter (pair-shared seed), so unchanged surfaces correlate to
gamma ~ 1 by construction; state changes decorrelate via (a) the water
plane geometry, (b) per-material temporal-alpha range jitter applied at t2
(city.SurfaceField -> tracer hash_noise). Coherence is estimated over the
SAME 4x4 blocks as the 10 m multilook (16 looks).

Chip npz per seed: vv_t1_db, vv_t2_db, coh (10 m, f16), mask_peak,
mask_standing_t2, mask_drained_t2 (i8), meta JSON (scene + hydro + pair +
look draws + natural_weight + scenario bin). Raw dB kept so the
intensity-vs-coherence channel choice stays open downstream.
"""
from __future__ import annotations

import time

import numpy as np

from . import recon
from .city import (CityParams, SurfaceField, WaterAwareField, draw_material,
                   generate_city, gt_masks, level_at_frac, with_water)
from .config import SarSystem, SimConfig, Trajectory
from .hydro import HydroParams, sample_pair
from .sbr import simulate_sbr

C0 = 299_792_458.0


def corpus_config(extent_m: float, incidence_deg: float, look_az_deg: float,
                  rho_m: float = 2.5, slant_km: float = 10.0) -> SimConfig:
    """C-band spotlight config sized for the scene: resolution rho_m with
    range/cross-range ambiguities beyond the scene diagonal."""
    diag = extent_m * np.sqrt(2)
    bw_hz = C0 / (2 * rho_m)
    n_freq = 256
    while C0 / (2 * bw_hz / n_freq) < 1.2 * diag:      # range ambiguity
        n_freq *= 2
    lam = C0 / 5.405e9
    dtheta = lam / (2 * rho_m)                         # full aperture (rad)
    n_pulses = 64
    while lam / (2 * (dtheta / n_pulses)) < 1.2 * diag:  # x-range ambiguity
        n_pulses *= 2
    half_deg = np.rad2deg(dtheta) / 2
    return SimConfig(
        sarsystem=SarSystem(carrier_ghz=5.405, bandwidth_mhz=bw_hz / 1e6,
                            n_freq=n_freq),
        trajectory=Trajectory(mode='circular_spotlight',
                              slant_range_m=slant_km * 1e3,
                              depression_deg=90.0 - incidence_deg,
                              az_start_deg=look_az_deg - half_deg,
                              az_end_deg=look_az_deg + half_deg,
                              n_pulses=n_pulses),
        scene_extent_m=extent_m, pixel_m=rho_m, name='corpus')


def multilook_and_coherence(I1: np.ndarray, I2: np.ndarray,
                            axis: np.ndarray, out_m: float = 10.0):
    """(power_t1, power_t2, coherence) at out_m cells from two complex
    images. Intensity is multilooked at out_m. Coherence is estimated over
    2x-larger windows (2*out_m) and posted back to the out_m grid: adjacent
    BP pixels are PSF-correlated (Hann IRW ~1.44x), so a 4x4 window has
    only ~6-8 effective looks and a |gamma|-bias floor ~0.35 — the 8x8
    window restores a ~0.2 floor, matching real S1 coherence products
    (which likewise estimate over larger windows than their posting)."""
    pix = float(axis[1] - axis[0])
    block = max(int(round(out_m / pix)), 1)
    n = (I1.shape[0] // block) * block

    def blocks(x, b):
        m = (x.shape[0] // b) * b
        return x[:m, :m].reshape(m // b, b, m // b, b)

    p1 = blocks(np.abs(I1) ** 2, block).mean(axis=(1, 3))[:n // block,
                                                          :n // block]
    p2 = blocks(np.abs(I2) ** 2, block).mean(axis=(1, 3))[:n // block,
                                                          :n // block]
    # 3x window (30 m at 10 m posting): ~40-50 effective looks -> |gamma|
    # bias floor ~0.13, matching real S1 coherence products (ESA processes
    # ~70+ looks; real open-water coherence medians ~0.12 are floor-set)
    b2 = 3 * block
    cp1 = blocks(np.abs(I1) ** 2, b2).mean(axis=(1, 3))
    cp2 = blocks(np.abs(I2) ** 2, b2).mean(axis=(1, 3))
    cross = blocks(I1 * np.conj(I2), b2).mean(axis=(1, 3))
    coh_c = np.abs(cross) / np.sqrt(np.maximum(cp1 * cp2, 1e-30))
    coh = np.repeat(np.repeat(coh_c, 3, axis=0), 3, axis=1)
    npx = n // block
    coh = coh[:npx, :npx]
    if coh.shape[0] < npx:                       # tail guard
        pad = npx - coh.shape[0]
        coh = np.pad(coh, ((0, pad), (0, pad)), mode='edge')
    return p1, p2, np.clip(coh, 0.0, 1.0), npx


def render_pair(seed: int, extent_m: float = 320.0, n_rays_side: int = 384,
                rho_m: float = 2.5, backend: str = 'mi_llvm',
                tier1: bool = True, force_bin: str | None = None):
    """Full corpus item for one seed. Returns (item dict, meta dict, wall s).

    force_bin: calibration hook — rejection-sample the pair until its
    scenario contains this substring (e.g. 'near_peak'); corpus runs leave
    it None so the stratified sampler's mix stands.
    """
    t_start = time.perf_counter()
    scene, meta, footprints, kinds = generate_city(
        CityParams(seed=seed, extent_m=extent_m, tier1=tier1))

    rng_h = np.random.default_rng(seed ^ 0x5EED_0F1D)
    pervious = float(meta['open_prob'] + 0.10)
    hydro = HydroParams.draw(rng_h, meta['height_mean_m'],
                             meta['fill_prob'], meta['slope_pct'], pervious)
    pair = sample_pair(rng_h, hydro)
    if force_bin is not None:
        for _ in range(200):
            if force_bin in pair.scenario:
                break
            pair = sample_pair(rng_h, hydro)
        else:
            raise RuntimeError(f'could not draw scenario {force_bin!r}')

    look_rng = np.random.default_rng(seed + 123456789)
    incidence = float(look_rng.uniform(30.0, 46.0))
    look_az = float(look_rng.uniform(0.0, 360.0))
    cfg = corpus_config(extent_m, incidence, look_az, rho_m)
    lam = cfg.sarsystem.wavelength

    jitter = (seed * 7919 + 13) % 2**31          # PAIR-SHARED ray jitter
    aux_rng = np.random.default_rng(seed ^ 0xAA77)
    water_mat = draw_material(aux_rng, 'water')
    from .city import GAMMA_SYS
    gamma_sys = float(aux_rng.uniform(*GAMMA_SYS))

    event = pair.has_flood or pair.scenario == 'receded_wet'
    imgs = {}
    for tag, state, other, wfrac in (
            ('t1', pair.state_t1, pair.state_t2, pair.w1_frac),
            ('t2', pair.state_t2, pair.state_t1, pair.w2_frac)):
        level = level_at_frac(meta, wfrac)
        # per-material gain/alpha draws shared across the pair: same stream
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
                         shadow_rays=True, seed=jitter, backend=backend,
                         accumulator='binned', bin_frac_of_lambda=1.0,
                         surface=fld, fixed_rays=True)
        img, axis = recon.backproject(S, cfg, window='hann')
        imgs[tag] = np.asarray(img)

    p1, p2, coh, n_px = multilook_and_coherence(imgs['t1'], imgs['t2'],
                                                np.asarray(axis))
    masks = gt_masks(meta, footprints, n_px,
                     level_at_frac(meta, pair.w2_frac),
                     meta['w_peak_level'])
    if not pair.has_flood and pair.scenario != 'receded_wet':
        masks['mask_peak'][:] = 0        # non-flood pair: no event at all
        masks['mask_drained_t2'][:] = 0

    meta_all = dict(scene=meta, pair=pair.to_meta(),
                    incidence_deg=incidence, look_az_deg=look_az,
                    gamma_sys=gamma_sys,
                    n_rays_side=n_rays_side, rho_m=rho_m, backend=backend,
                    n_freq=cfg.sarsystem.n_freq,
                    n_pulses=cfg.trajectory.n_pulses)
    item = {
        'vv_t1_db': 10 * np.log10(np.maximum(p1, 1e-30)).astype(np.float16),
        'vv_t2_db': 10 * np.log10(np.maximum(p2, 1e-30)).astype(np.float16),
        'coh': coh.astype(np.float16),
        **masks,
    }
    wall = time.perf_counter() - t_start
    meta_all['wall_s'] = wall
    return item, meta_all, wall
