"""Phase-history accumulation.

Dataflow contract (all backends): a scatterer stream per pulse is a pair of
flat float arrays (A, dR) — amplitude and *differential* one-way-equivalent
range dR = d_target − d_phase_center. The accumulator turns streams into the
complex phase-history column:

    S(f_k, τ_n) = Σ_i A_i · exp(−j 4π f_k dR_i / c)          (paper eq. 4)

Stage 0 uses `accumulate_direct` in float64 (reference-grade). The factored
float32 path (paper §4.3.4: f_k = f_c + kΔf with f_c/c, Δf/c precomputed in
float64) is `accumulate_factored32`, kept for the Stage-1 numerics experiment.
"""
from __future__ import annotations

import numpy as np

from . import C0


def accumulate_direct(f_k: np.ndarray, A: np.ndarray, dR: np.ndarray,
                      out: np.ndarray | None = None) -> np.ndarray:
    """Direct fp64 accumulation for one pulse. Returns S_col (K,) complex128.

    f_k: (K,) float64; A, dR: (N,) float64.
    """
    phase = (-4j * np.pi / C0) * np.outer(f_k, dR)       # (K, N)
    col = (np.exp(phase) * A[None, :]).sum(axis=1)
    if out is not None:
        out += col
        return out
    return col


def accumulate_factored32(f_c: float, delta_f: float, k_idx: np.ndarray,
                          A: np.ndarray, dR: np.ndarray) -> np.ndarray:
    """Paper §4.3.4 float32-safe factoring, one pulse → (K,) complex64.

    φ_ik = a·dR_i + b·(k_i·dR_i) with a = 4π f_c/c, b = 4π Δf/c computed in
    float64 host-side; device math in float32. k_idx are the signed indices
    k = −(K−1)/2 … +(K−1)/2 so f_k = f_c + k·Δf.
    """
    a = np.float64(4 * np.pi * f_c / C0)
    b = np.float64(4 * np.pi * delta_f / C0)
    A32 = A.astype(np.float32)
    dR32 = dR.astype(np.float32)
    # carrier term: reduce a·dR mod 2π in float64 BEFORE casting (the whole
    # point — a·dR is ~1e6 rad and float32 would keep only ~1e-1 rad precision)
    carrier = np.mod(np.float64(a) * dR.astype(np.float64), 2 * np.pi)
    carrier32 = carrier.astype(np.float32)                      # (N,)
    kdR = np.outer(k_idx.astype(np.float32), dR32)              # (K, N)
    phase = carrier32[None, :] + np.float32(b) * kdR            # (K, N)
    col = (A32[None, :] * np.exp(-1j * phase.astype(np.float32))).sum(axis=1)
    return col.astype(np.complex64)


def phase_history(f_k: np.ndarray, positions: np.ndarray,
                  phase_center: np.ndarray, targets_xyz: np.ndarray,
                  targets_amp: np.ndarray) -> np.ndarray:
    """Vectorized fp64 phase history for point scatterers, all pulses.

    positions: (Np, 3) antenna phase centers; phase_center: (3,) scene center;
    targets_xyz: (T, 3); targets_amp: (T,). Returns S (K, Np) complex128.
    """
    d_c = np.linalg.norm(positions - phase_center[None, :], axis=1)   # (Np,)
    d_t = np.linalg.norm(positions[:, None, :] - targets_xyz[None, :, :],
                         axis=2)                                      # (Np, T)
    dR = d_t - d_c[:, None]                                           # (Np, T)
    # S[k, n] = Σ_t A_t exp(−j4π f_k dR[n,t] / c)
    phase = (-4j * np.pi / C0) * f_k[:, None, None] * dR[None, :, :]
    return (np.exp(phase) * targets_amp[None, None, :]).sum(axis=2)
