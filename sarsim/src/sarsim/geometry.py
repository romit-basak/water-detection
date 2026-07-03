"""Antenna frames and aperture ray generation.

Validation-mode aiming: rays go from the antenna position through a regular
grid on the ground plane covering the scene (+margin). This decouples ray
coverage from antenna FNBW modeling (config override per design) and washes
out of Gate-1 comparisons via the global complex calibration.

Ray-origin advance (fp32-safety for Mitsuba): a ray traced from o' = o + t_adv·d
intersects the identical surfaces at t' = t − t_adv, but the *traced* t' stays
O(100 m) instead of O(10 km), keeping float32 ulp ≈ 1e−5 m. The true first-leg
length is recovered host-side in fp64.
"""
from __future__ import annotations

import numpy as np


def look_frame(antenna: np.ndarray, center: np.ndarray):
    """Orthonormal (fwd, right, up) with fwd from antenna toward center."""
    fwd = center - antenna
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fwd, right, up


def scene_grid_rays(antenna: np.ndarray, center: np.ndarray,
                    extent_m: float, n_side: int, margin: float = 1.1,
                    jitter_rng: np.random.Generator | None = None):
    """Rays from `antenna` through an n_side×n_side ground grid around center.

    Returns (origins (N,3), dirs (N,3) unit, per_ray_power (scalar)).
    Deterministic when jitter_rng is None (Gate-1 mode).
    """
    half = extent_m * margin / 2
    axis = np.linspace(-half, half, n_side, dtype=np.float64)
    X, Y = np.meshgrid(axis, axis)
    pts = np.stack([X.ravel() + center[0], Y.ravel() + center[1],
                    np.full(X.size, center[2])], axis=1)
    if jitter_rng is not None:
        cell = axis[1] - axis[0]
        pts[:, :2] += jitter_rng.uniform(-cell / 2, cell / 2, (len(pts), 2))
    d = pts - antenna[None, :]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    origins = np.broadcast_to(antenna, d.shape).copy()
    return origins, d, 1.0 / len(d)


def advanced_origins(origins: np.ndarray, dirs: np.ndarray,
                     center: np.ndarray, standoff_m: float = 200.0):
    """Move ray origins to `standoff_m` from scene center along each ray.

    Returns (origins_adv, t_advance (N,)) — exact line reparameterization.
    """
    to_center = center[None, :] - origins
    t_center = np.einsum('ij,ij->i', to_center, dirs)
    t_adv = np.maximum(t_center - standoff_m, 0.0)
    return origins + t_adv[:, None] * dirs, t_adv


def stable_first_leg(hit: np.ndarray, antenna: np.ndarray,
                     center: np.ndarray, r_ref: float) -> np.ndarray:
    """|hit − antenna| − R_ref via the catastrophic-cancellation-free identity

        |p−a| − R = ((p−c)·(p + c − 2a)) / (|p−a| + R),   R = |c−a|

    hit may be float32 data upcast to fp64; p−c is small so the numerator is
    accurate even when p and a are 10 km apart.
    """
    p = hit.astype(np.float64)
    num = np.einsum('ij,ij->i', p - center[None, :],
                    p + center[None, :] - 2 * antenna[None, :])
    d_pa = np.linalg.norm(p - antenna[None, :], axis=1)
    return num / (d_pa + r_ref)
