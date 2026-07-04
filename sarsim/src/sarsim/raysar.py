"""Python port of RaySAR's MATLAB image formation (Gen_Refl_Map.m).

Contract from /Users/romitbasak/Projects/RaySAR/ASSESSMENT.md §3-4. The
MATLAB layer is a scatter-add histogram over ray contributions — no other
physics. Deliberate deviations from the reference implementation:
  - wavelength is a parameter (MATLAB hardcodes X-band 0.031);
  - dB uses 10*log10 (MATLAB's `10*log(x)` is natural log);
  - 6-column mode only (9-column bounce-chain filtering assumes row
    adjacency, which multithreaded POV-Ray rendering breaks).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEVEL_NAMES = ['single', 'double', 'triple', 'fourfold', 'fivefold']


@dataclass
class Contributions:
    az: np.ndarray        # azimuth (m, relative to image center)
    ra: np.ndarray        # slant range (m, zero ref = image plane)
    el: np.ndarray        # elevation (m, relative to image center)
    amp: np.ndarray       # intensity 0..1
    level: np.ndarray     # bounce level 1..5
    spec: np.ndarray      # specular flag 0/1


def load_contributions(path) -> Contributions:
    """Parse a 6-column Contributions.txt (SAR_Intersection 0)."""
    R = np.loadtxt(path)
    if R.ndim == 1:
        R = R[None, :]
    if R.shape[1] < 6:
        raise ValueError(f'expected >=6 columns, got {R.shape[1]}')
    return Contributions(az=R[:, 0], ra=R[:, 1], el=R[:, 2], amp=R[:, 3],
                         level=R[:, 4].astype(int),
                         spec=R[:, 5].astype(int))


def form_layers(c: Contributions, a_pix: float = 0.5, r_pix: float = 0.5,
                incidence_deg: float | None = None, coherent: bool = False,
                wavelength: float = 0.0555, max_level: int = 5,
                bounds: tuple[float, float, float, float] | None = None):
    """Bin contributions into per-bounce-level SAR image layers.

    incidence_deg: if given, project slant range to ground range
    (Ra/sin(ang), Gen_Refl_Map.m line 75). coherent: accumulate complex
    with two-way phase -4*pi/lambda*Ra (slant range drives phase even in
    ground-range images, as in the reference).

    Returns (layers dict incl. 'all', extent (a_min,a_max,r_min,r_max)).
    Row 0 = r_min: near range at the top, like the MATLAB maps.
    """
    ra_img = c.ra / np.sin(np.deg2rad(incidence_deg)) \
        if incidence_deg is not None else c.ra
    if bounds is None:
        a_min, a_max = c.az.min(), c.az.max()
        r_min, r_max = ra_img.min(), ra_img.max()
    else:
        a_min, a_max, r_min, r_max = bounds
    num_a = int(np.ceil((a_max - a_min) / a_pix))
    num_r = int(np.ceil((r_max - r_min) / r_pix))

    col = np.floor((c.az - a_min) / a_pix).astype(int)
    row = np.floor((ra_img - r_min) / r_pix).astype(int)
    ok = (col >= 0) & (col < num_a) & (row >= 0) & (row < num_r)

    if coherent:
        phi = -(4 * np.pi / wavelength) * c.ra          # two-way, slant
        sig = c.amp * np.exp(1j * phi)
        dtype = np.complex128
    else:
        sig = c.amp.astype(np.float64)
        dtype = np.float64

    layers = {}
    total = np.zeros((num_r, num_a), dtype=dtype)
    for lv in range(1, max_level + 1):
        img = np.zeros((num_r, num_a), dtype=dtype)
        m = ok & (c.level == lv)
        np.add.at(img, (row[m], col[m]), sig[m])
        layers[LEVEL_NAMES[lv - 1]] = img
        total += img
    layers['all'] = total
    if coherent:
        layers = {k: np.abs(v) for k, v in layers.items()}
    return layers, (a_min, a_max, r_min, r_max)


def to_db(img: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """10*log10 relative to the image peak, floored (NOT MATLAB's 10*ln)."""
    peak = img.max()
    if peak <= 0:
        return np.full_like(img, floor_db, dtype=np.float64)
    return np.maximum(10 * np.log10(np.maximum(img / peak, 1e-30)), floor_db)
