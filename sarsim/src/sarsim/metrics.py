"""Image-quality metrics for validation gates: peak location, impulse
response width (IRW, −3 dB), peak sidelobe ratio (PSLR)."""
from __future__ import annotations

import numpy as np


def peak_near(img: np.ndarray, axis: np.ndarray, xy: tuple[float, float],
              radius_m: float = 0.5):
    """Highest |img| pixel within radius of expected (x, y).

    Returns (x_meas, y_meas, amp). Image convention: img[iy, ix] with
    x = axis[ix], y = axis[iy] (recon.pixel_grid meshgrid layout).
    """
    a = np.abs(img)
    X, Y = np.meshgrid(axis, axis)
    mask = (X - xy[0]) ** 2 + (Y - xy[1]) ** 2 <= radius_m ** 2
    masked = np.where(mask, a, 0.0)
    iy, ix = np.unravel_index(np.argmax(masked), a.shape)
    return float(axis[ix]), float(axis[iy]), float(a[iy, ix])


def _cut(img: np.ndarray, iy: int, ix: int, along: str) -> np.ndarray:
    return np.abs(img[iy, :]) if along == 'x' else np.abs(img[:, iy * 0 + ix])


def irw(img: np.ndarray, axis: np.ndarray, iy: int, ix: int,
        along: str = 'x', up: int = 32) -> float:
    """−3 dB width of the mainlobe through (iy, ix), meters."""
    cut = _cut(img, iy, ix, along)
    fine = np.linspace(axis[0], axis[-1], len(axis) * up)
    cutf = np.interp(fine, axis, cut)
    ipk = np.argmax(cutf)
    half = cutf[ipk] / np.sqrt(2)               # −3 dB in amplitude
    lo = ipk
    while lo > 0 and cutf[lo] > half:
        lo -= 1
    hi = ipk
    while hi < len(cutf) - 1 and cutf[hi] > half:
        hi += 1
    return float(fine[hi] - fine[lo])


def pslr_db(img: np.ndarray, axis: np.ndarray, iy: int, ix: int,
            along: str = 'x', up: int = 32) -> float:
    """Peak sidelobe ratio along a principal cut, dB (negative)."""
    cut = _cut(img, iy, ix, along)
    fine = np.linspace(axis[0], axis[-1], len(axis) * up)
    cutf = np.interp(fine, axis, cut)
    ipk = int(np.argmax(cutf))
    # walk out to the first nulls on each side
    def first_null(idx, step):
        while 0 < idx < len(cutf) - 1 and cutf[idx + step] <= cutf[idx]:
            idx += step
        return idx
    ln, rn = first_null(ipk, -1), first_null(ipk, +1)
    side = np.concatenate([cutf[:max(ln, 0)], cutf[rn + 1:]])
    if side.size == 0 or cutf[ipk] == 0:
        return float('nan')
    return float(20 * np.log10(side.max() / cutf[ipk]))
