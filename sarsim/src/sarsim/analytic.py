"""Stage 0 — analytic point-scatterer phase-history generator (no ray tracing).

This is the correctness anchor that replaces the Gorham & Moore MATLAB toolbox
as ground truth: direct fp64 evaluation of paper eq. 4 over point targets.
The ray-traced SBR path (Stage 1) must reproduce these phase histories when
pointed at plate proxies placed at the same target positions.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .phase import phase_history


def simulate(cfg: SimConfig, targets_xyz: np.ndarray,
             targets_amp: np.ndarray | None = None) -> np.ndarray:
    """Returns S (K, Np) complex128 for point scatterers."""
    targets_xyz = np.atleast_2d(np.asarray(targets_xyz, dtype=np.float64))
    if targets_amp is None:
        targets_amp = np.ones(len(targets_xyz), dtype=np.float64)
    positions = cfg.trajectory.positions()
    center = np.asarray(cfg.trajectory.target_xyz, dtype=np.float64)
    return phase_history(cfg.sarsystem.f_k, positions, center,
                         targets_xyz, np.asarray(targets_amp, np.float64))
