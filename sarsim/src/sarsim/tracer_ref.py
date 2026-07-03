"""Reference SBR tracer: brute-force Möller–Trumbore in numpy (fp64).

Backend contract (shared with tracer_mi): given a pulse's antenna position,
emit bounce-tagged flat streams (A, dR, depth) that phase.accumulate turns
into a phase-history column. Suitable for scenes up to ~1k triangles; this is
also the Plan-C fallback if Mitsuba misbehaves.

Backscatter model (paper §4.3.3): at every hit,
    A = E · (ρ_d·cosθ_i + ρ_s·max(0, r̂·ŝ)^p)
with θ_i incidence angle (foreshortening), r̂ the specular reflection of the
incoming direction, ŝ the unit vector hit→antenna. Specular children spawn
with E' = E·ρ_s up to max_depth or an energy floor.

Range convention: dR = (path_to_hit + |hit→antenna|)/2 − R_ref, i.e. the
one-way-equivalent differential range of the full monostatic path. For
depth-1 this equals |hit−a| − R_ref; for a dihedral double bounce it lands
the return at the corner range — the urban-flood signature under test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-9


@dataclass
class Material:
    rho_d: float = 1.0        # diffuse reflectance
    rho_s: float = 0.0        # specular reflectance
    phong_p: float = 20.0     # specular exponent


@dataclass
class TriScene:
    """Triangle soup: v0/v1/v2 (T,3) float64 + per-triangle material index."""
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    mat_idx: np.ndarray
    materials: list[Material]

    @classmethod
    def from_triangles(cls, tris: np.ndarray, mat_idx: np.ndarray,
                       materials: list[Material]) -> 'TriScene':
        t = np.asarray(tris, dtype=np.float64)
        return cls(t[:, 0], t[:, 1], t[:, 2],
                   np.asarray(mat_idx, dtype=np.int64), materials)

    @property
    def normals(self) -> np.ndarray:
        n = np.cross(self.v1 - self.v0, self.v2 - self.v0)
        return n / np.linalg.norm(n, axis=1, keepdims=True)


def _intersect(scene: TriScene, o: np.ndarray, d: np.ndarray):
    """Vectorized Möller–Trumbore, all rays × all triangles.

    Returns (t (N,), tri (N,), hit_mask (N,)). Brute force O(N·T).
    """
    e1 = scene.v1 - scene.v0                                  # (T,3)
    e2 = scene.v2 - scene.v0
    pvec = np.cross(d[:, None, :], e2[None, :, :])            # (N,T,3)
    det = np.einsum('tj,ntj->nt', e1, pvec)
    inv = np.where(np.abs(det) > EPS, 1.0 / det, 0.0)
    tvec = o[:, None, :] - scene.v0[None, :, :]
    u = np.einsum('ntj,ntj->nt', tvec, pvec) * inv
    qvec = np.cross(tvec, e1[None, :, :])
    v = np.einsum('ntj,ntj->nt', d[:, None, :], qvec) * inv
    t = np.einsum('tj,ntj->nt', e2, qvec) * inv
    valid = ((np.abs(det) > EPS) & (u >= -EPS) & (v >= -EPS)
             & (u + v <= 1 + EPS) & (t > 1e-4))
    t = np.where(valid, t, np.inf)
    tri = np.argmin(t, axis=1)
    tmin = t[np.arange(len(d)), tri]
    hit = np.isfinite(tmin)
    return tmin, tri, hit


def trace_pulse(scene: TriScene, antenna: np.ndarray, center: np.ndarray,
                rays_o: np.ndarray, rays_d: np.ndarray, ray_power: float,
                max_depth: int = 3, energy_floor: float = 1e-4,
                shadow_rays: bool = False):
    """Run the SBR bounce loop for one pulse.

    Returns (A (M,), dR (M,), depth (M,)) concatenated over bounces.
    All math fp64 (reference grade). rays_o may be advanced origins; the true
    antenna position is passed separately for return-leg / dR computation.
    """
    r_ref = float(np.linalg.norm(center - antenna))
    mats_d = np.array([m.rho_d for m in scene.materials])
    mats_s = np.array([m.rho_s for m in scene.materials])
    mats_p = np.array([m.phong_p for m in scene.materials])
    normals = scene.normals

    A_out, dR_out, dep_out = [], [], []

    o, d = rays_o.copy(), rays_d.copy()
    # path length from the TRUE antenna to current ray origin
    path = np.linalg.norm(o - antenna[None, :], axis=1)
    E = np.full(len(o), ray_power, dtype=np.float64)
    alive = np.ones(len(o), dtype=bool)

    for depth in range(1, max_depth + 1):
        if not alive.any():
            break
        t, tri, hit = _intersect(scene, o[alive], d[alive])
        idx = np.flatnonzero(alive)[hit]
        if idx.size == 0:
            break
        th, trih = t[hit], tri[hit]
        p_hit = o[idx] + th[:, None] * d[idx]
        n = normals[trih]
        # flip normals toward the incoming ray
        facing = -np.sign(np.einsum('ij,ij->i', d[idx], n))
        n = n * facing[:, None]
        cos_i = np.clip(-np.einsum('ij,ij->i', d[idx], n), 0.0, 1.0)
        m = scene.mat_idx[trih]
        # specular reflection of incoming dir + unit vector back to antenna
        r_dir = d[idx] - 2 * np.einsum('ij,ij->i', d[idx], n)[:, None] * n
        s = antenna[None, :] - p_hit
        s_len = np.linalg.norm(s, axis=1)
        s_hat = s / s_len[:, None]
        spec = np.clip(np.einsum('ij,ij->i', r_dir, s_hat), 0.0, 1.0)
        A = E[idx] * (mats_d[m] * cos_i + mats_s[m] * spec ** mats_p[m])

        if shadow_rays:
            # occlusion test on the return leg hit→antenna
            t_s, _, hit_s = _intersect(scene, p_hit + 1e-6 * s_hat, s_hat)
            A = np.where(hit_s & (t_s < s_len - 1e-3), 0.0, A)

        path_hit = path[idx] + th
        dR = 0.5 * (path_hit + s_len) - r_ref
        keep = A > 0
        A_out.append(A[keep]); dR_out.append(dR[keep])
        dep_out.append(np.full(int(keep.sum()), depth))

        # spawn specular children; floor is RELATIVE to per-ray launch power
        # (an absolute floor silently kills all children when N rays >> 1/floor)
        E_child = E[idx] * mats_s[m]
        live = E_child > energy_floor * ray_power
        new_alive = np.zeros(len(o), dtype=bool)
        if depth < max_depth and live.any():
            ci = idx[live]
            o[ci] = p_hit[live] + 1e-6 * r_dir[live]
            d[ci] = r_dir[live]
            path[ci] = path_hit[live]
            E[ci] = E_child[live]
            new_alive[ci] = True
        alive = new_alive

    if not A_out:
        z = np.zeros(0)
        return z, z, z.astype(int)
    return (np.concatenate(A_out), np.concatenate(dR_out),
            np.concatenate(dep_out))
