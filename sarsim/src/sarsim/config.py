"""Simulation configuration, mirroring the YAML schema of Willis et al. 2020
(Table 1): sarsystem / trajectory / antenna / materials / scene objects.

All derived spectral and geometric quantities are computed here in float64,
including the ambiguity guards that catch mis-sized scenes before a single
ray is traced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import C0


@dataclass
class SarSystem:
    carrier_ghz: float          # f_c
    bandwidth_mhz: float        # B  (paper specifies range resolution; B = c/2ρ_r)
    n_freq: int                 # K
    polarity: str = 'HH'        # parsed + recorded; scattering model is scalar

    @property
    def f_c(self) -> float:
        return self.carrier_ghz * 1e9

    @property
    def bandwidth(self) -> float:
        return self.bandwidth_mhz * 1e6

    @property
    def f_k(self) -> np.ndarray:
        """K frequencies spanning [f_c - B/2, f_c + B/2], float64."""
        return np.linspace(self.f_c - self.bandwidth / 2,
                           self.f_c + self.bandwidth / 2,
                           self.n_freq, dtype=np.float64)

    @property
    def delta_f(self) -> float:
        return self.bandwidth / (self.n_freq - 1)

    @property
    def wavelength(self) -> float:
        return C0 / self.f_c

    @property
    def range_resolution(self) -> float:
        """Slant-range resolution ρ_r = c / 2B."""
        return C0 / (2 * self.bandwidth)

    @property
    def range_ambiguity(self) -> float:
        """Unambiguous range window c / 2Δf."""
        return C0 / (2 * self.delta_f)


@dataclass
class Trajectory:
    mode: str                   # 'circular_spotlight' | 'linear_spotlight'
    slant_range_m: float        # ρ_s at point of closest approach
    depression_deg: float       # ψ
    az_start_deg: float
    az_end_deg: float
    n_pulses: int               # Np
    target_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def aperture_rad(self) -> float:
        return np.deg2rad(self.az_end_deg - self.az_start_deg)

    def positions(self) -> np.ndarray:
        """(Np, 3) float64 antenna phase-center positions.

        circular_spotlight: constant slant range/depression, azimuth sweep.
        linear_spotlight: straight line tangent to the circular path at the
        center azimuth (same center geometry, linear sampling along-track).
        """
        c = np.asarray(self.target_xyz, dtype=np.float64)
        psi = np.deg2rad(self.depression_deg)
        rho_g = self.slant_range_m * np.cos(psi)      # ground-plane radius
        z = self.slant_range_m * np.sin(psi)

        if self.mode == 'circular_spotlight':
            th = np.deg2rad(np.linspace(self.az_start_deg, self.az_end_deg,
                                        self.n_pulses, dtype=np.float64))
            pos = np.stack([c[0] + rho_g * np.cos(th),
                            c[1] + rho_g * np.sin(th),
                            np.full_like(th, c[2] + z)], axis=1)
            return pos
        if self.mode == 'linear_spotlight':
            th0 = np.deg2rad(0.5 * (self.az_start_deg + self.az_end_deg))
            center = c + np.array([rho_g * np.cos(th0), rho_g * np.sin(th0), z])
            # along-track direction = tangent to the azimuth circle
            tangent = np.array([-np.sin(th0), np.cos(th0), 0.0])
            half_len = rho_g * np.tan(self.aperture_rad / 2)
            s = np.linspace(-half_len, half_len, self.n_pulses,
                            dtype=np.float64)
            return center[None, :] + s[:, None] * tangent[None, :]
        raise NotImplementedError(
            f'trajectory mode {self.mode!r} (scanning/stripmap deferred)')


@dataclass
class SimConfig:
    sarsystem: SarSystem
    trajectory: Trajectory
    scene_extent_m: float = 10.0        # square scene, centered on target_xyz
    pixel_m: float = 0.02
    name: str = 'sim'
    # Stage-1+ fields (parsed, unused by Stage 0)
    materials: dict = field(default_factory=dict)
    objects: list = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'SimConfig':
        raw = yaml.safe_load(Path(path).read_text())
        cfg = cls(
            sarsystem=SarSystem(**raw['sarsystem']),
            trajectory=Trajectory(**{
                **raw['trajectory'],
                'target_xyz': tuple(raw['trajectory'].get('target_xyz',
                                                          (0, 0, 0))),
            }),
            scene_extent_m=raw.get('scene_extent_m', 10.0),
            pixel_m=raw.get('pixel_m', 0.02),
            name=raw.get('name', Path(path).stem),
            materials=raw.get('materials', {}),
            objects=raw.get('objects', []),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Ambiguity guards (Gate-0 config asserts)."""
        diag = self.scene_extent_m * np.sqrt(2)
        ra = self.sarsystem.range_ambiguity
        if ra <= diag:
            raise ValueError(
                f'range ambiguity window {ra:.1f} m <= scene diagonal '
                f'{diag:.1f} m — increase n_freq or shrink scene')
        # cross-range ambiguity: λ/(2 δθ_step) must exceed scene extent
        dth = self.trajectory.aperture_rad / max(1, self.trajectory.n_pulses - 1)
        xra = self.sarsystem.wavelength / (2 * dth) if dth > 0 else np.inf
        if xra <= self.scene_extent_m:
            raise ValueError(
                f'cross-range ambiguity {xra:.1f} m <= scene extent '
                f'{self.scene_extent_m:.1f} m — increase n_pulses')

    @property
    def cross_range_resolution(self) -> float:
        """ρ_az = λ / (2 Δθ)."""
        return self.sarsystem.wavelength / (2 * self.trajectory.aperture_rad)
