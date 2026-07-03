"""Programmatic scene builders (triangle soup for tracer_ref / tracer_mi)."""
from __future__ import annotations

import numpy as np

from .tracer_ref import Material, TriScene


def _quad(center: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Two triangles for the parallelogram center ± u ± v. (2,3,3)."""
    c = np.asarray(center, np.float64)
    p00, p01 = c - u - v, c - u + v
    p10, p11 = c + u - v, c + u + v
    return np.array([[p00, p10, p11], [p00, p11, p01]])


def plates_at_points(points: np.ndarray, aim_from: np.ndarray,
                     size_m: float = 0.06) -> TriScene:
    """Small diffuse plates centered at `points`, facing `aim_from` (Gate-1
    point-target proxies). Plate << resolution cell so each acts point-like."""
    tris, midx = [], []
    for p in np.atleast_2d(points):
        n = aim_from - p
        n = n / np.linalg.norm(n)
        helper = np.array([0.0, 0.0, 1.0])
        if abs(n @ helper) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        u = np.cross(n, helper); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        q = _quad(p, u * size_m / 2, v * size_m / 2)
        tris.append(q); midx += [0, 0]
    return TriScene.from_triangles(np.concatenate(tris), np.array(midx),
                                   [Material(rho_d=1.0, rho_s=0.0)])


def dihedral(corner_xy: tuple[float, float] = (0.0, 0.0), size_m: float = 2.0,
             open_toward_deg: float = 50.0) -> TriScene:
    """Vertical wall + horizontal ground plate meeting at a right-angle line
    (the double-bounce unit scene). The corner line sits on the ground at
    z = 0, oriented perpendicular to the `open_toward_deg` azimuth so the
    dihedral opens toward the radar."""
    az = np.deg2rad(open_toward_deg)
    toward = np.array([np.cos(az), np.sin(az), 0.0])   # horizontal, to radar
    along = np.array([-np.sin(az), np.cos(az), 0.0])   # corner-line direction
    c = np.array([corner_xy[0], corner_xy[1], 0.0])
    h = size_m / 2
    # ground plate extending toward the radar from the corner line
    g = _quad(c + toward * h, along * h, toward * h)
    # vertical wall rising from the corner line
    w = _quad(c + np.array([0, 0, h]), along * h, np.array([0, 0, h]))
    tris = np.concatenate([g, w])
    midx = np.array([0, 0, 0, 0])
    return TriScene.from_triangles(
        tris, midx, [Material(rho_d=0.05, rho_s=0.9, phong_p=200.0)])


def flood_pilot(water: bool, extent_m: float = 40.0) -> TriScene:
    """Stage-2 urban block: ground, 5 boxes along a street, optional smooth
    water plane at z=0.05 covering the street corridor."""
    mats = [Material(rho_d=0.15, rho_s=0.05, phong_p=10.0),    # 0 soil
            Material(rho_d=0.10, rho_s=0.50, phong_p=60.0),    # 1 wall
            Material(rho_d=0.20, rho_s=0.02, phong_p=5.0),     # 2 roof
            Material(rho_d=0.01, rho_s=0.70, phong_p=400.0)]   # 3 water
    tris, midx = [], []

    def add(quads: np.ndarray, m: int):
        tris.append(quads)
        midx.extend([m] * len(quads))

    h = extent_m / 2
    add(_quad(np.zeros(3), np.array([h, 0, 0]), np.array([0, h, 0])), 0)

    ex, ey, ez = 2.5, 4.0, 6.0                      # box half-x/half-y/height
    for i, bx in enumerate(np.linspace(-12, 12, 5)):
        c = np.array([bx, 6.0, 0.0])                # row of buildings at y=+6
        # 4 walls + roof (no floor)
        add(_quad(c + [0, -ey, ez / 2], np.array([ex, 0, 0]),
                  np.array([0, 0, ez / 2])), 1)     # front wall (faces street)
        add(_quad(c + [0, +ey, ez / 2], np.array([ex, 0, 0]),
                  np.array([0, 0, ez / 2])), 1)
        add(_quad(c + [-ex, 0, ez / 2], np.array([0, ey, 0]),
                  np.array([0, 0, ez / 2])), 1)
        add(_quad(c + [+ex, 0, ez / 2], np.array([0, ey, 0]),
                  np.array([0, 0, ez / 2])), 1)
        add(_quad(c + [0, 0, ez], np.array([ex, 0, 0]),
                  np.array([0, ey, 0])), 2)
    if water:
        # flooded corridor in front of the building row, y in [-14, 2]: wide
        # enough that its far portion (y < -6) escapes the 7.8 m wall layover
        # (6 m walls at 37.5 deg incidence) so open-water darkening is
        # measurable outside layover contamination
        add(_quad(np.array([0, -6.0, 0.05]), np.array([h, 0, 0]),
                  np.array([0, 8.0, 0])), 3)
    return TriScene.from_triangles(np.concatenate(tris), np.array(midx), mats)
