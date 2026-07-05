"""Procedural city generator v3 — acquisition-pair corpus (Sprint v4).

Design principle (Auer's LOD compromise, RaySAR thesis §7.1): model only the
structures that change 10 m aggregate statistics; fold everything sub-render-
resolution into randomized *effective* material parameters.

v3 over v2: geometry and base materials are STATE-FREE — one scene per seed.
Surface state (dry / rain-wet / saturated-recently-drained) is applied at
TRACE time via `SurfaceField` (amplitude gain + temporal-decorrelation
sigma per hit), so:
- an acquisition pair shares bitwise-identical geometry and base materials;
- the saturated halo (terrain below the PEAK waterline but above the
  CURRENT one) needs no geometry subdivision — it's a positional field;
- standing water is the only geometric difference (a plane via
  `with_water`), correct occlusion for free.

The v2 realism items stand (M3-gate-driven): asphalt streets, per-building
material draws, tilted terrain, Tier-1 facade ledges / roof clutter / cars,
open blocks, edge fillers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .scenes import _quad
from .tracer_ref import Material, TriScene

# legacy v1 table (export_obj default names; flood_pilot-compatible)
MATERIALS = {
    'soil':  (Material(rho_d=0.15, rho_s=0.05, phong_p=10.0), (0.4, 0.35, 0.3)),
    'wall':  (Material(rho_d=0.10, rho_s=0.50, phong_p=60.0), (0.7, 0.7, 0.7)),
    'roof':  (Material(rho_d=0.20, rho_s=0.02, phong_p=5.0),  (0.5, 0.3, 0.3)),
    'water': (Material(rho_d=0.01, rho_s=0.70, phong_p=400.0), (0.1, 0.2, 0.4)),
}
_MAT_ORDER = list(MATERIALS)

# effective-parameter ranges (rho_d, rho_s, phong_p) as (lo, hi) draws.
# asphalt specular LOW on purpose: the flood dihedral jump is driven by
# water_rho_s - street_rho_s (wall specular scales dry AND flooded equally)
MAT_RANGES = {
    'soil':    ((0.08, 0.25), (0.02, 0.08), (5, 15)),
    'asphalt': ((0.03, 0.09), (0.06, 0.18), (40, 150)),
    # wall phong LOW = wide specular lobe: the aggregate of window
    # reveals/balconies returns dihedral energy over wide azimuths
    # (box-flat walls retro-reflect only when facing the radar, which
    # pins the MEDIAN flooded-street dVV at the water-darkening value
    # regardless of wall brightness — the iteration-2 dead end)
    'wall':    ((0.05, 0.18), (0.50, 0.90), (4, 15)),
    'roof':    ((0.10, 0.30), (0.01, 0.12), (4, 15)),
    'metal':   ((0.05, 0.15), (0.45, 0.85), (4, 20)),
    # water rho_d models the weak Bragg/ripple return: real C-band open
    # water sits ~8-15 dB below land, NOT 30+. Too-dark water lets coherent
    # PSF-sidelobe leakage dominate its cells and props up pair coherence.
    # phong LOW (iter5): near-mirror water (formerly 200-500) only connects
    # the water->wall double-bounce chain for ~5 deg of look_az per wall
    # (see diag_alignment.py). Widened for rain-disturbed flood water.
    'water':   ((0.025, 0.06), (0.70, 0.90), (20, 60)),
}
GROUND_KINDS = ('soil', 'asphalt')

# rain-wet dielectric brightening: AMPLITUDE gains (power adds 20*log10(g);
# M3 anchor: real non-flood dVV = +0.4..+0.7 dB -> g ~ 1.05-1.08 median)
WET_GAIN = {'rho_like': (1.02, 1.14)}
# saturated (recently drained): wetter than rain-wet
SAT_GAIN = (1.10, 1.30)

# temporal correlation alpha by kind (pair coherence; D2/D3). 'changed'
# applies to ground kinds whose dry/wet state differs between acquisitions.
# Values calibrated against CO-EVENT band-4 within-urban Hebei stats
# (targets.json): real 6/12-day pairs are low-coherence EVERYWHERE
# (land median 0.39, urban flood 0.23, open flood 0.12) — vegetation,
# atmosphere and thermal noise decorrelate all classes, folded here into
# the per-pair gamma_sys factor plus modest per-kind alphas.
ALPHA = {
    'wall': (0.55, 0.85), 'roof': (0.50, 0.80), 'metal': (0.60, 0.90),
    'ground_stable': (0.30, 0.60),
    'ground_changed': (0.08, 0.30),
    'saturated': (0.04, 0.15),
    'water': (0.02, 0.02),
}
GAMMA_SYS = (0.55, 0.80)     # per-pair system/temporal decorrelation floor


def draw_material(rng: np.random.Generator, kind: str) -> Material:
    (d0, d1), (s0, s1), (p0, p1) = MAT_RANGES[kind]
    return Material(rho_d=rng.uniform(d0, d1), rho_s=rng.uniform(s0, s1),
                    phong_p=rng.uniform(p0, p1))


@dataclass
class CityParams:
    """Seeded per-scene draws (all resolved values land in corpus meta)."""
    seed: int = 0
    extent_m: float = 640.0
    block_m: float | None = None           # U(24, 60)
    street_m: float | None = None          # U(5, 11): narrowed, more area near walls
    fill_prob: float | None = None         # U(0.65, 0.95): raised, denser urban fabric
    open_prob: float | None = None         # U(0.02, 0.15) park/plaza blocks
    height_mean_m: float | None = None     # U(5, 18)
    height_sigma: float | None = None      # U(0.3, 0.7)
    slope_pct: float | None = None         # U(0.1, 0.8) terrain tilt
    flood_frac: float | None = None        # U(0.15, 0.6): PEAK extent quantile
    tier1: bool = True
    resolved: dict = field(default_factory=dict)

    def resolve(self, rng: np.random.Generator) -> dict:
        r = {
            'block_m': self.block_m or rng.uniform(24, 60),
            'street_m': self.street_m or rng.uniform(5, 11),
            'fill_prob': self.fill_prob or rng.uniform(0.65, 0.95),
            'open_prob': self.open_prob if self.open_prob is not None
                         else rng.uniform(0.02, 0.15),
            'height_mean_m': self.height_mean_m or rng.uniform(5, 18),
            'height_sigma': self.height_sigma or rng.uniform(0.3, 0.7),
            'slope_pct': self.slope_pct or rng.uniform(0.1, 0.8),
            'flood_frac': self.flood_frac or rng.uniform(0.15, 0.6),
            'slope_az_deg': rng.uniform(0, 360),
        }
        self.resolved = r
        return r


class _Builder:
    def __init__(self):
        self.tris, self.midx = [], []
        self.materials, self.kinds = [], []

    def mat(self, m: Material, kind: str) -> int:
        self.materials.append(m)
        self.kinds.append(kind)
        return len(self.materials) - 1

    def add(self, quads: np.ndarray, mi: int):
        self.tris.append(quads)
        self.midx.extend([mi] * len(quads))

    def box(self, c, ex, ey, z0, h, mi_wall, mi_top):
        cz = np.array([c[0], c[1], z0 + h / 2])
        self.add(_quad(cz + [0, -ey, 0], np.array([ex, 0, 0]),
                       np.array([0, 0, h / 2])), mi_wall)
        self.add(_quad(cz + [0, +ey, 0], np.array([ex, 0, 0]),
                       np.array([0, 0, h / 2])), mi_wall)
        self.add(_quad(cz + [-ex, 0, 0], np.array([0, ey, 0]),
                       np.array([0, 0, h / 2])), mi_wall)
        self.add(_quad(cz + [+ex, 0, 0], np.array([0, ey, 0]),
                       np.array([0, 0, h / 2])), mi_wall)
        self.add(_quad([c[0], c[1], z0 + h], np.array([ex, 0, 0]),
                       np.array([0, ey, 0])), mi_top)

    def scene(self) -> TriScene:
        return TriScene.from_triangles(
            np.concatenate(self.tris), np.array(self.midx), self.materials)


def generate_city(params: CityParams):
    """Build the state-free base scene.

    Returns (TriScene, meta, footprints, kinds) — kinds is the per-material
    kind list (parallel to scene.materials) that SurfaceField consumes.
    meta carries terrain-plane coefficients + peak waterline level.
    """
    p = params
    rng = np.random.default_rng(p.seed)
    r = p.resolve(rng)
    B = _Builder()

    half = p.extent_m / 2
    az = np.deg2rad(r['slope_az_deg'])
    g = r['slope_pct'] / 100.0
    gx, gy = g * np.cos(az), g * np.sin(az)

    def terrain_z(x, y):
        return gx * x + gy * y

    pitch = r['block_m'] + r['street_m']
    n_blocks = max(int(p.extent_m // pitch), 1)
    origin = -(n_blocks * pitch) / 2

    m_asphalt = B.mat(draw_material(rng, 'asphalt'), 'asphalt')
    m_soils = [B.mat(draw_material(rng, 'soil'), 'soil') for _ in range(3)]

    def ground_quad(cx, cy, hx, hy, mi):
        c = np.array([cx, cy, terrain_z(cx, cy)])
        u = np.array([hx, 0, gx * hx])
        v = np.array([0, hy, gy * hy])
        B.add(_quad(c, u, v), mi)

    cells = []
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            cx = origin + bi * pitch + r['street_m'] + r['block_m'] / 2
            cy = origin + bj * pitch + r['street_m'] + r['block_m'] / 2
            ground_quad(cx, cy, r['block_m'] / 2, r['block_m'] / 2,
                        m_soils[int(rng.integers(0, len(m_soils)))])
            cells.append((cx, cy))
    for bi in range(n_blocks + 1):
        x = origin + bi * pitch + r['street_m'] / 2
        ground_quad(x, 0, r['street_m'] / 2, half, m_asphalt)
    for bj in range(n_blocks + 1):
        y = origin + bj * pitch + r['street_m'] / 2
        ground_quad(0, y, half, r['street_m'] / 2, m_asphalt)
    # edge fillers: the grid spans [origin, origin+n*pitch+street], NOT the
    # full extent — un-tiled margins would be radar-black geometry voids
    lo = origin
    hi = origin + n_blocks * pitch + r['street_m']
    if lo > -half:
        ground_quad((lo - half) / 2, 0, (lo + half) / 2, half, m_soils[0])
        ground_quad(0, (lo - half) / 2, half, (lo + half) / 2, m_soils[1])
    if hi < half:
        ground_quad((hi + half) / 2, 0, (half - hi) / 2, half, m_soils[0])
        ground_quad(0, (hi + half) / 2, half, (half - hi) / 2, m_soils[1])

    footprints = []
    n_buildings = 0
    for (cx, cy) in cells:
        if rng.uniform() < r['open_prob']:
            continue
        if rng.uniform() > r['fill_prob']:
            continue
        ex = rng.uniform(0.28, 0.46) * r['block_m']
        ey = rng.uniform(0.28, 0.46) * r['block_m']
        bx = cx + rng.uniform(-0.5, 0.5) * (r['block_m'] / 2 - ex)
        by = cy + rng.uniform(-0.5, 0.5) * (r['block_m'] / 2 - ey)
        h = float(np.clip(rng.lognormal(np.log(r['height_mean_m']),
                                        r['height_sigma']), 3.0, 80.0))
        z0 = terrain_z(bx, by) - 0.3
        m_wall = B.mat(draw_material(rng, 'wall'), 'wall')
        m_roof = B.mat(draw_material(rng, 'roof'), 'roof')
        B.box((bx, by), ex, ey, z0, h + 0.3, m_wall, m_roof)
        footprints.append((bx, by, ex, ey))
        n_buildings += 1

        if p.tier1:
            for k in range(1, min(int(h // 3), 20) + 1):
                z = z0 + 0.3 + 3.0 * k
                if z > z0 + h:
                    break
                for sy in (-1, +1):
                    c = np.array([bx, by + sy * (ey + 0.2), z])
                    B.add(_quad(c, np.array([ex * 0.9, 0, 0]),
                                np.array([0, 0.4, 0])), m_wall)
            if ex * ey > 40 and rng.uniform() < 0.7:
                m_hvac = B.mat(draw_material(rng, 'metal'), 'metal')
                for _ in range(int(rng.integers(1, 3))):
                    hx, hy = rng.uniform(0.8, 1.8), rng.uniform(0.8, 1.8)
                    px = bx + rng.uniform(-0.5, 0.5) * ex
                    py = by + rng.uniform(-0.5, 0.5) * ey
                    B.box((px, py), hx, hy, z0 + h + 0.3,
                          rng.uniform(0.8, 2.0), m_hvac, m_hvac)

    # PEAK waterline level (quantile of terrain at flood_frac); the
    # acquisition-time level is derived from it by the hydrograph fraction
    zs = terrain_z(rng.uniform(-half, half, 4096),
                   rng.uniform(-half, half, 4096))
    w_peak_level = float(np.quantile(zs, r['flood_frac']))
    z_min = float(-abs(gx) * half - abs(gy) * half)

    if p.tier1:
        m_car = B.mat(draw_material(rng, 'metal'), 'metal')
        n_cars = int((p.extent_m / 640.0) ** 2 * rng.integers(60, 160))
        for _ in range(n_cars):
            k = int(rng.integers(0, n_blocks + 1))
            along = rng.uniform(-half * 0.95, half * 0.95)
            off = rng.uniform(-r['street_m'] * 0.3, r['street_m'] * 0.3)
            base = origin + k * pitch + r['street_m'] / 2
            x, y = (base + off, along) if rng.uniform() < 0.5 \
                else (along, base + off)
            B.box((x, y), 2.2, 0.9, terrain_z(x, y), 1.5, m_car, m_car)

    meta = dict(seed=p.seed, extent_m=p.extent_m,
                n_buildings=n_buildings, n_tris=len(B.midx),
                terrain=dict(gx=gx, gy=gy), w_peak_level=w_peak_level,
                z_min=z_min,
                **{k: float(v) for k, v in r.items()})
    return B.scene(), meta, footprints, list(B.kinds)


def level_at_frac(meta: dict, frac: float) -> float | None:
    """Waterline level when depth is `frac` of peak (linear in depth from
    the terrain minimum up to the peak waterline)."""
    if frac <= 0:
        return None
    return meta['z_min'] + frac * (meta['w_peak_level'] - meta['z_min'])


def with_water(scene: TriScene, kinds: list[str], level: float,
               water_mat: Material) -> tuple[TriScene, list[str]]:
    """Base scene + horizontal water plane at `level` (terrain above it
    wins as the nearer hit — correct occlusion for free)."""
    half = float(max(np.abs(scene.v0[:, :2]).max(),
                     np.abs(scene.v2[:, :2]).max()))
    quads = _quad(np.array([0.0, 0.0, level]),
                  np.array([half, 0, 0]), np.array([0, half, 0]))
    tris = np.stack([np.concatenate([scene.v0, quads[:, 0]]),
                     np.concatenate([scene.v1, quads[:, 1]]),
                     np.concatenate([scene.v2, quads[:, 2]])], axis=1)
    midx = np.concatenate([scene.mat_idx,
                           np.full(2, len(scene.materials))])
    return (TriScene.from_triangles(tris, midx,
                                    scene.materials + [water_mat]),
            kinds + ['water'])


class SurfaceField:
    """Trace-time surface state for ONE acquisition (tracer_ref contract).

    __call__(p_hit, mat_idx) -> (gain, sigma_dR). Gain: rain-wet /
    saturated dielectric brightening per material instance (drawn once per
    scene, shared across the pair so wet->wet correlates in amplitude).
    Sigma: temporal-decorrelation range jitter from the pair's alpha —
    applied on t2 ONLY (apply_noise), the reference acquisition carries no
    jitter so pair coherence corresponds to alpha exactly once.
    """

    def __init__(self, kinds: list[str], meta: dict, *, state: str,
                 other_state: str, level_now: float | None,
                 wavelength: float, acq_seed: int,
                 scene_rng: np.random.Generator, apply_noise: bool,
                 sat_enabled: bool = False, gamma_sys: float = 1.0):
        self.acq_seed = acq_seed
        self.gx = meta['terrain']['gx']
        self.gy = meta['terrain']['gy']
        # the saturated halo exists only at t2 of a real flood event
        # (post-recession); at t1 or in non-flood pairs it must be off
        self.level_peak = meta['w_peak_level'] if sat_enabled else -np.inf
        self.level_now = -np.inf if level_now is None else level_now
        kinds = list(kinds)
        n = len(kinds)
        self.is_ground = np.array([k in GROUND_KINDS for k in kinds])

        # per-material draws SHARED across the pair (same scene_rng stream)
        wet_gain = scene_rng.uniform(*WET_GAIN['rho_like'], size=n)
        sat_gain = scene_rng.uniform(*SAT_GAIN, size=n)
        a_stable = np.array([scene_rng.uniform(*ALPHA.get(
            k if k not in GROUND_KINDS else 'ground_stable', (0.9, 1.0)))
            for k in kinds])
        a_changed = scene_rng.uniform(*ALPHA['ground_changed'], size=n)
        a_sat = scene_rng.uniform(*ALPHA['saturated'], size=n)

        wet = state in ('wet', 'flooded')
        self.gain_base = wet_gain if wet else np.ones(n)
        self.gain_sat = sat_gain * wet_gain      # saturated ⊂ wet halo
        changed = ((state in ('wet', 'flooded')) !=
                   (other_state in ('wet', 'flooded')))
        alpha = np.where(self.is_ground,
                         a_changed if changed else
                         np.minimum(a_stable, 0.9), a_stable)
        alpha = alpha * gamma_sys
        a_sat = a_sat * gamma_sys
        self.sigma = self._sigma(alpha, wavelength) if apply_noise \
            else np.zeros(n)
        self.sigma_sat = self._sigma(a_sat, wavelength) if apply_noise \
            else np.zeros(n)
        self.water_sigma = self._sigma(np.array([ALPHA['water'][0]]),
                                       wavelength)[0] if apply_noise else 0.0

    @staticmethod
    def _sigma(alpha: np.ndarray, wavelength: float) -> np.ndarray:
        a = np.clip(alpha, 0.02, 0.999)
        return (wavelength / (4 * np.pi)) * np.sqrt(-2.0 * np.log(a))

    def __call__(self, p_hit: np.ndarray, mat_idx: np.ndarray):
        gain = self.gain_base[mat_idx].copy()
        sigma = self.sigma[mat_idx].copy()
        ground = self.is_ground[mat_idx]
        if ground.any():
            tz = self.gx * p_hit[:, 0] + self.gy * p_hit[:, 1]
            sat = ground & (tz < self.level_peak) & (tz >= self.level_now)
            gain[sat] = self.gain_sat[mat_idx[sat]]
            sigma[sat] = self.sigma_sat[mat_idx[sat]]
        return gain, sigma


class WaterAwareField(SurfaceField):
    """SurfaceField extended for scenes that carry the appended water
    material (index == len(base kinds)): water gets gain 1, alpha~0."""

    def __init__(self, *args, n_base: int, **kw):
        super().__init__(*args, **kw)
        self.n_base = n_base
        self.gain_base = np.append(self.gain_base, 1.0)
        self.gain_sat = np.append(self.gain_sat, 1.0)
        self.sigma = np.append(self.sigma, self.water_sigma)
        self.sigma_sat = np.append(self.sigma_sat, self.water_sigma)
        self.is_ground = np.append(self.is_ground, False)


def gt_masks(meta: dict, footprints, n_px: int, level_standing: float | None,
             level_peak: float, sub: int = 4) -> dict:
    """10 m masks: peak extent, standing water at t2, drained-but-saturated
    at t2. Majority over sub-sampled cell area; buildings excluded."""
    ext = meta['extent_m']
    gx, gy = meta['terrain']['gx'], meta['terrain']['gy']
    step = ext / (n_px * sub)
    ax = -ext / 2 + step * (np.arange(n_px * sub) + 0.5)
    X, Y = np.meshgrid(ax, ax)
    tz = gx * X + gy * Y
    bld = np.zeros(X.shape, dtype=bool)
    for (bx, by, ex, ey) in footprints:
        bld |= (np.abs(X - bx) <= ex) & (np.abs(Y - by) <= ey)

    def to_cells(m):
        frac = m.reshape(n_px, sub, n_px, sub).mean(axis=(1, 3))
        return (frac > 0.5).astype(np.int8)

    peak = (tz < level_peak) & ~bld
    if level_standing is None:
        standing = np.zeros_like(peak)
    else:
        standing = (tz < level_standing) & ~bld
    return {'mask_peak': to_cells(peak),
            'mask_standing_t2': to_cells(standing),
            'mask_drained_t2': to_cells(peak & ~standing)}


def export_obj(scene: TriScene, path: str | Path,
               mat_names: list[str] | None = None) -> Path:
    """TriScene -> OBJ + MTL (Benji's tracer / Blender / conversion)."""
    path = Path(path)
    names = mat_names or (_MAT_ORDER if len(scene.materials) <= 4 else
                          [f'm{i}' for i in range(len(scene.materials))])
    verts = np.concatenate([scene.v0, scene.v1, scene.v2])
    mtl_path = path.with_suffix('.mtl')
    with open(mtl_path, 'w') as f:
        for i, name in enumerate(names):
            if name in MATERIALS:
                m, kd = MATERIALS[name]
            else:
                m, kd = scene.materials[i], (0.5, 0.5, 0.5)
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
