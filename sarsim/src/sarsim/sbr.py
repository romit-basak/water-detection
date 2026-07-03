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

    if backend != 'ref':
        from . import tracer_mi
        return tracer_mi.simulate(cfg, scene, n_rays_side, max_depth,
                                  shadow_rays, seed,
                                  variant=backend.removeprefix('mi_'),
                                  progress=progress)

    binner = None
    if accumulator == 'binned':
        # swath: scene diagonal + slack for multi-bounce path extension
        swath_half = cfg.scene_extent_m * np.sqrt(2)
        binner = phase.BinnedAccumulator(f_k, swath_half)

    S = np.zeros((cfg.sarsystem.n_freq, cfg.trajectory.n_pulses),
                 dtype=np.complex128)
    it = enumerate(positions)
    if progress:
        it = tqdm(list(it), desc='SBR pulses')
    for n, a in it:
        o, d, pw = geometry.scene_grid_rays(a, center, cfg.scene_extent_m,
                                            n_rays_side, jitter_rng=rng)
        A, dR, _ = tracer_ref.trace_pulse(scene, a, center, o, d, pw,
                                          max_depth=max_depth,
                                          shadow_rays=shadow_rays)
        if len(A):
            if binner is not None:
                S[:, n] = binner.column(A, dR)
            else:
                phase.accumulate_direct(f_k, A, dR, out=S[:, n])
    return S
