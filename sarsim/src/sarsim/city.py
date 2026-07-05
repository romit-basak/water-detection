"""Procedural city generator + OBJ/MTL export.

One geometry source feeding all three simulators: TriScene directly into
sarsim (ref / mi_llvm / mi_cuda), OBJ+MTL into Benji's SAR_RayTracer, and
(via conversion) POV-Ray for RaySAR. Parameterized to span the morphologies
UrbanSARFloods struggles with (dense high-rise vs fragmented low-rise).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .scenes import _quad
from .tracer_ref import Material, TriScene

MATERIALS = {                      # name -> (Material, MTL Kd/Ks proxy)
    'soil':  (Material(rho_d=0.15, rho_s=0.05, phong_p=10.0), (0.4, 0.35, 0.3)),
    'wall':  (Material(rho_d=0.10, rho_s=0.50, phong_p=60.0), (0.7, 0.7, 0.7)),
    'roof':  (Material(rho_d=0.20, rho_s=0.02, phong_p=5.0),  (0.5, 0.3, 0.3)),
    'water': (Material(rho_d=0.01, rho_s=0.70, phong_p=400.0), (0.1, 0.2, 0.4)),
}
_MAT_ORDER = list(MATERIALS)


def generate_city(seed: int = 0, extent_m: float = 120.0,
                  block_m: float = 30.0, street_m: float = 8.0,
                  height_mean_m: float = 8.0, height_sigma: float = 0.5,
                  fill_prob: float = 0.85, water: bool = False,
                  water_level_m: float = 0.08):
    """Parameterized rectangular-grid city. Returns (TriScene, meta dict).

    height ~ lognormal(mean=log(height_mean), sigma) captures the tall-tail
    of dense urban cores; fill_prob < 1 gives fragmented low-rise fabric.
    Water floods every street corridor (not building footprints) when set.
    """
    rng = np.random.default_rng(seed)
    tris, midx = [], []

    def add(quads: np.ndarray, mname: str):
        tris.append(quads)
        midx.extend([_MAT_ORDER.index(mname)] * len(quads))

    half = extent_m / 2
    add(_quad(np.zeros(3), np.array([half, 0, 0]), np.array([0, half, 0])),
        'soil')

    pitch = block_m + street_m
    n_blocks = int(extent_m // pitch)
    origin = -(n_blocks * pitch) / 2
    n_buildings = 0
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            if rng.uniform() > fill_prob:
                continue
            cx = origin + bi * pitch + street_m / 2 + block_m / 2
            cy = origin + bj * pitch + street_m / 2 + block_m / 2
            # building footprint jittered within its block
            ex = rng.uniform(0.30, 0.46) * block_m
            ey = rng.uniform(0.30, 0.46) * block_m
            cx += rng.uniform(-0.5, 0.5) * (block_m / 2 - ex)
            cy += rng.uniform(-0.5, 0.5) * (block_m / 2 - ey)
            h = float(rng.lognormal(np.log(height_mean_m), height_sigma))
            h = min(max(h, 3.0), 60.0)
            c = np.array([cx, cy, 0.0])
            add(_quad(c + [0, -ey, h / 2], np.array([ex, 0, 0]),
                      np.array([0, 0, h / 2])), 'wall')
            add(_quad(c + [0, +ey, h / 2], np.array([ex, 0, 0]),
                      np.array([0, 0, h / 2])), 'wall')
            add(_quad(c + [-ex, 0, h / 2], np.array([0, ey, 0]),
                      np.array([0, 0, h / 2])), 'wall')
            add(_quad(c + [+ex, 0, h / 2], np.array([0, ey, 0]),
                      np.array([0, 0, h / 2])), 'wall')
            add(_quad(c + [0, 0, h], np.array([ex, 0, 0]),
                      np.array([0, ey, 0])), 'roof')
            n_buildings += 1
    if water:
        # one plane over the whole extent just above ground: streets flood,
        # buildings pierce it (their walls start at z=0 < water level, so
        # the wall-water dihedral forms at every street-facing façade)
        add(_quad(np.array([0, 0, water_level_m]),
                  np.array([half, 0, 0]), np.array([0, half, 0])), 'water')
    scene = TriScene.from_triangles(
        np.concatenate(tris), np.array(midx),
        [MATERIALS[k][0] for k in _MAT_ORDER])
    meta = dict(seed=seed, extent_m=extent_m, block_m=block_m,
                street_m=street_m, height_mean_m=height_mean_m,
                height_sigma=height_sigma, fill_prob=fill_prob, water=water,
                n_buildings=n_buildings, n_tris=len(scene.mat_idx))
    return scene, meta


def export_obj(scene: TriScene, path: str | Path,
               mat_names: list[str] | None = None) -> Path:
    """TriScene -> OBJ + MTL (for Benji's tracer / Blender / conversion)."""
    path = Path(path)
    names = mat_names or _MAT_ORDER
    verts = np.concatenate([scene.v0, scene.v1, scene.v2])
    mtl_path = path.with_suffix('.mtl')
    with open(mtl_path, 'w') as f:
        for name in names:
            kd = MATERIALS.get(name, (None, (0.5, 0.5, 0.5)))[1]
            m = MATERIALS.get(name, (Material(),))[0]
            f.write(f'newmtl {name}\n'
                    f'Kd {kd[0]} {kd[1]} {kd[2]}\n'
                    f'Ks {m.rho_s} {m.rho_s} {m.rho_s}\n'
                    f'Ns {m.phong_p}\n\n')
    with open(path, 'w') as f:
        f.write(f'mtllib {mtl_path.name}\n')
        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        T = len(scene.v0)
        order = np.argsort(scene.mat_idx, kind='stable')
        cur = -1
        for ti in order:
            if scene.mat_idx[ti] != cur:
                cur = int(scene.mat_idx[ti])
                f.write(f'usemtl {names[cur]}\n')
            f.write(f'f {ti + 1} {ti + 1 + T} {ti + 1 + 2 * T}\n')
    return path
