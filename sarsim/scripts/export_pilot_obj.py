"""Export the Gate-2 flood pilot scene as Y-up OBJ/MTL for external tracers.

Benji's SAR_RayTracer (like most graphics tools) is Y-up; sarsim is Z-up.
(x, y, z)_zup -> (x, z, -y)_yup is a proper rotation, so triangle winding
(and thus geometric normals) survives unflipped.

Material names in the MTL must exist in SAR_RayTracer's tex_map (texture.h)
for its wavelength-dependent roughness switching to engage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sarsim.city import export_obj
from sarsim.scenes import flood_pilot
from sarsim.tracer_ref import TriScene


def to_yup(s: TriScene) -> TriScene:
    def m(v: np.ndarray) -> np.ndarray:
        return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)
    return TriScene(m(s.v0), m(s.v1), m(s.v2), s.mat_idx, s.materials)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out_dir', help='directory for pilot_{dry,wet}.obj/.mtl')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for water, name in [(False, 'pilot_dry'), (True, 'pilot_wet')]:
        scene = flood_pilot(water=water)
        p = export_obj(to_yup(scene), out / f'{name}.obj')
        print(f'{p}  ({len(scene.mat_idx)} tris, water={water})')


if __name__ == '__main__':
    main()
