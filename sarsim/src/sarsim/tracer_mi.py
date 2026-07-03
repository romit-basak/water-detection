"""Mitsuba 3 / Dr.Jit intersection backend.

Mitsuba is used ONLY as a batched ray-casting engine (BVH + hardware
traversal: Embree on the LLVM variant, OptiX/RT cores on cuda) — every piece
of SBR physics stays in tracer_ref.trace_pulse's fp64 numpy, so 'ref' and
'mi_*' backends are the SAME code with the intersection call swapped.

Material lookup: the whole TriScene is uploaded as ONE merged mi.Mesh, so
si.prim_index is the global triangle index and maps straight into
TriScene.mat_idx (no shape-pointer bookkeeping).

fp32 note: Mitsuba wheels are single precision. Scene coordinates are small
(tens of m), so hit POINTS are accurate; the long antenna→scene leg never
enters device math when ray origins are pre-advanced (geometry.advanced_origins)
— trace_pulse recomputes true path lengths host-side in fp64 from hit points.
"""
from __future__ import annotations

import numpy as np

from .tracer_ref import TriScene

_MI = {}


def _mi(variant: str):
    """Import + variant-select mitsuba once per process."""
    import sarsim  # noqa: F401  (sets DRJIT_LIBLLVM_PATH on macOS)
    import mitsuba as mi
    want = {'llvm': 'llvm_ad_rgb', 'cuda': 'cuda_ad_rgb'}[variant]
    if mi.variant() != want:
        mi.set_variant(want)
    return mi


class MiTriScene:
    """TriScene mirror backed by a Mitsuba scene. Exposes the trace_pulse
    backend interface: .intersect(o, d) -> (t, n, mat, hit), .materials."""

    def __init__(self, tri_scene: TriScene, variant: str = 'llvm'):
        mi = self._m = _mi(variant)
        self.materials = tri_scene.materials
        self.mat_idx = tri_scene.mat_idx
        V = np.concatenate([tri_scene.v0, tri_scene.v1, tri_scene.v2])
        F = (np.arange(len(tri_scene.v0))[:, None]
             + np.array([0, 1, 2])[None, :] * len(tri_scene.v0))
        mesh = mi.Mesh('soup', vertex_count=len(V), face_count=len(F),
                       has_vertex_normals=False, has_vertex_texcoords=False)
        p = mi.traverse(mesh)
        p['vertex_positions'] = V.astype(np.float32).ravel()
        p['faces'] = F.astype(np.uint32).ravel()
        p.update()
        self.scene = mi.load_dict({'type': 'scene', 'soup': mesh})

    def intersect(self, o: np.ndarray, d: np.ndarray):
        mi = self._m
        ray = mi.Ray3f(mi.Point3f(*o.astype(np.float32).T),
                       mi.Vector3f(*d.astype(np.float32).T))
        si = self.scene.ray_intersect(ray)
        hit = np.array(si.is_valid())
        t = np.array(si.t, dtype=np.float64)
        t = np.where(hit, t, np.inf)
        n = np.stack([np.array(si.n.x), np.array(si.n.y),
                      np.array(si.n.z)], axis=1).astype(np.float64)
        prim = np.array(si.prim_index, dtype=np.int64)
        prim = np.where(hit, np.clip(prim, 0, len(self.mat_idx) - 1), 0)
        return t, n, self.mat_idx[prim], hit
