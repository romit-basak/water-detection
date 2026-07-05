"""SBR pulse-loop driver: trajectory → rays → tracer → (A, dR) → S(f_k, τ_n)."""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from . import geometry, phase, tracer_ref
from .config import SimConfig


def simulate_sbr(cfg: SimConfig, scene: 'tracer_ref.TriScene',
                 n_rays_side: int = 512, max_depth: int = 3,
                 shadow_rays: bool = False, seed: int | None = None,
                 backend: str = 'ref', accumulator: str = 'direct',
                 bin_frac_of_lambda: float = 1 / 16,
                 surface=None, fixed_rays: bool = False,
                 progress: bool = False) -> np.ndarray:
    """Ray-traced phase history, (K, Np) complex128.

    backend='ref' is the fp64 numpy reference tracer; 'mi_llvm'/'mi_cuda'
    dispatch to tracer_mi (Stage-1b). accumulator='binned' for dense scenes
    (ground planes: ~n_rays_side² scatterers/pulse makes 'direct' O(K·N)
    intractable).
    """
    f_k = cfg.sarsystem.f_k
    positions = cfg.trajectory.positions()
    center = np.asarray(cfg.trajectory.target_xyz, np.float64)
    rng = np.random.default_rng(seed) if seed is not None else None

    if backend.startswith('mi_'):
        from .tracer_mi import MiTriScene
        scene = MiTriScene(scene, variant=backend.removeprefix('mi_'))
    elif backend != 'ref':
        raise ValueError(f'unknown backend {backend!r}')
    advance = backend.startswith('mi_')   # fp32 safety: small traced t only

    binner = None
    if accumulator == 'binned':
        # swath: scene diagonal + slack for multi-bounce path extension.
        # bin width: only the BAND term is bin-approximated (carrier is
        # exact per scatterer), so tolerable width scales with 1/bandwidth
        # — corpus scenes pass ~1.0 (lambda bins) instead of the lambda/16
        # default, or the DFT matrix explodes with swath size
        swath_half = cfg.scene_extent_m * np.sqrt(2)
        binner = phase.BinnedAccumulator(f_k, swath_half,
                                         bin_frac_of_lambda=bin_frac_of_lambda)

    S = np.zeros((cfg.sarsystem.n_freq, cfg.trajectory.n_pulses),
                 dtype=np.complex128)

    # fixed_rays: draw the jittered ground-point ensemble ONCE and aim at
    # it from every pulse. A per-pulse re-jitter resamples the surface each
    # pulse, so any position-tied phase (surface hash noise) varies across
    # the aperture and its energy defocuses into an image-wide pedestal —
    # a fixed irregular ensemble keeps scatterer identity across pulses
    # (like a real rough surface) while still avoiding the regular-grid
    # quantization artifact (Gate-2 finding).
    fixed_pts = None
    if fixed_rays:
        a0 = positions[0]
        o0, d0, pw0 = geometry.scene_grid_rays(
            a0, center, cfg.scene_extent_m, n_rays_side, jitter_rng=rng)
        t_ground = -o0[:, 2] / np.where(np.abs(d0[:, 2]) > 1e-12,
                                        d0[:, 2], 1e-12)
        fixed_pts = o0 + t_ground[:, None] * d0
        fixed_pw = pw0

    it = enumerate(positions)
    if progress:
        it = tqdm(list(it), desc='SBR pulses')
    for n, a in it:
        if fixed_pts is not None:
            d = fixed_pts - a[None, :]
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            o = np.broadcast_to(a, d.shape).copy()
            pw = fixed_pw
        else:
            o, d, pw = geometry.scene_grid_rays(a, center,
                                                cfg.scene_extent_m,
                                                n_rays_side, jitter_rng=rng)
        if advance:
            o, _ = geometry.advanced_origins(o, d, center)
        A, dR, _ = tracer_ref.trace_pulse(scene, a, center, o, d, pw,
                                          max_depth=max_depth,
                                          shadow_rays=shadow_rays,
                                          surface=surface)
        if len(A):
            if binner is not None:
                S[:, n] = binner.column(A, dR)
            else:
                phase.accumulate_direct(f_k, A, dR, out=S[:, n])
    return S
