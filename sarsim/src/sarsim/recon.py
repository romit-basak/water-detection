"""Image formation: matched filter (paper eq. 8) and backprojection
(Gorham & Moore 2010 formulation, paper eqs. 9–11).

Conventions: phase history S(f_k, τ_n) = Σ A·exp(−j4π f_k dR/c) with
f_k = f_start + k·Δf. The IFFT of S over k therefore peaks at range bin
r = dR with residual carrier exp(−j4π f_start dR/c), which backprojection
removes per-pixel before coherent summation over pulses.
"""
from __future__ import annotations

import numpy as np

from . import C0
from .config import SimConfig


def pixel_grid(cfg: SimConfig, n_pix: int | None = None):
    """(P,3) ground-plane pixel centers + (ny, nx) shape, z = target z."""
    if n_pix is None:
        n_pix = int(round(cfg.scene_extent_m / cfg.pixel_m))
    half = cfg.scene_extent_m / 2
    axis = np.linspace(-half, half, n_pix, dtype=np.float64)
    cx, cy, cz = cfg.trajectory.target_xyz
    X, Y = np.meshgrid(axis + cx, axis + cy)
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, cz)], axis=1)
    return P, (n_pix, n_pix), axis


def _pixel_dR(positions: np.ndarray, center: np.ndarray,
              P: np.ndarray) -> np.ndarray:
    """(Np, P) differential range for every pixel/pulse, float64."""
    d_c = np.linalg.norm(positions - center[None, :], axis=1)         # (Np,)
    d_p = np.linalg.norm(positions[:, None, :] - P[None, :, :], axis=2)
    return d_p - d_c[:, None]


def matched_filter(S: np.ndarray, cfg: SimConfig, n_pix: int = 128,
                   chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """I(p) = (1/NpK) Σ_n Σ_k S(f_k,τ_n)·exp(+j4πf_k dR_p(n)/c). O(P·Np·K)."""
    f_k = cfg.sarsystem.f_k
    positions = cfg.trajectory.positions()
    center = np.asarray(cfg.trajectory.target_xyz, np.float64)
    P, shape, axis = pixel_grid(cfg, n_pix)
    dR = _pixel_dR(positions, center, P)                              # (Np, P)
    K, Np = S.shape
    I = np.zeros(P.shape[0], dtype=np.complex128)
    coef = 4j * np.pi / C0
    for lo in range(0, P.shape[0], chunk):
        hi = min(lo + chunk, P.shape[0])
        # (K, Np, chunk) phase — evaluated per chunk to bound memory
        ph = np.exp(coef * f_k[:, None, None] * dR[None, :, lo:hi])
        I[lo:hi] = np.einsum('kn,knp->p', S, ph) / (Np * K)
    return I.reshape(shape), axis


def backproject(S: np.ndarray, cfg: SimConfig, n_pix: int | None = None,
                upsample: int = 32) -> tuple[np.ndarray, np.ndarray]:
    # upsample=32: linear-interp amplitude loss is quadratic in bin size;
    # 8x left ~2% peak-amplitude spread (Gate-0 finding), 32x -> ~0.1%.
    """Gorham & Moore backprojection. O(Np·(K logK + P))."""
    f_k = cfg.sarsystem.f_k
    f_start = f_k[0]
    delta_f = cfg.sarsystem.delta_f
    positions = cfg.trajectory.positions()
    center = np.asarray(cfg.trajectory.target_xyz, np.float64)
    P, shape, axis = pixel_grid(cfg, n_pix)
    dR = _pixel_dR(positions, center, P)                              # (Np, P)

    K, Np = S.shape
    nfft = int(2 ** np.ceil(np.log2(K * upsample)))
    # range-bin axis after fftshifted IFFT: r_m = m·c/(2·nfft·Δf), m signed
    m = np.arange(-nfft // 2, nfft // 2)
    r_bins = m * C0 / (2 * nfft * delta_f)
    dr_bin = r_bins[1] - r_bins[0]

    I = np.zeros(P.shape[0], dtype=np.complex128)
    for n in range(Np):
        rp = np.fft.fftshift(np.fft.ifft(S[:, n], nfft)) * nfft
        # linear interpolation of the complex range profile at each pixel's dR
        x = (dR[n] - r_bins[0]) / dr_bin
        i0 = np.clip(np.floor(x).astype(int), 0, nfft - 2)
        w = x - i0
        samp = rp[i0] * (1 - w) + rp[i0 + 1] * w
        # remove residual carrier at the IFFT reference frequency f_start
        I += samp * np.exp(4j * np.pi * f_start * dR[n] / C0)
    return (I / (Np * K)).reshape(shape), axis
