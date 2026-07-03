"""Gate-1b: Mitsuba backend parity with the reference tracer (LLVM, local)."""
import numpy as np
import pytest

from sarsim.config import SarSystem, Trajectory, SimConfig
from sarsim import scenes
from sarsim.sbr import simulate_sbr

pytest.importorskip('mitsuba')


@pytest.fixture
def cfg():
    return SimConfig(
        sarsystem=SarSystem(carrier_ghz=10.0, bandwidth_mhz=600.0, n_freq=64),
        trajectory=Trajectory(mode='circular_spotlight', slant_range_m=10000.0,
                              depression_deg=30.0, az_start_deg=48.5,
                              az_end_deg=51.5, n_pulses=32),
        scene_extent_m=10.0, pixel_m=0.05,
    )


def test_mi_llvm_matches_ref_plates(cfg):
    targets = np.array([[0.0, 0.0, 0.0], [-2.0, -2.0, 0.0], [3.0, 1.0, 0.0]])
    mid = cfg.trajectory.positions()[cfg.trajectory.n_pulses // 2]
    scene = scenes.plates_at_points(targets, aim_from=mid)
    S_ref = simulate_sbr(cfg, scene, n_rays_side=384, max_depth=1)
    S_mi = simulate_sbr(cfg, scene, n_rays_side=384, max_depth=1,
                        backend='mi_llvm')
    a, b = S_mi.ravel(), S_ref.ravel()
    corr = abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f'mi_llvm plates corr vs ref = {corr:.6f}')
    assert corr > 0.999


def test_mi_llvm_dihedral_double_bounce(cfg):
    scene = scenes.dihedral(corner_xy=(0.0, 0.0), size_m=2.0,
                            open_toward_deg=50.0)
    S2 = simulate_sbr(cfg, scene, n_rays_side=384, max_depth=2,
                      backend='mi_llvm')
    S1 = simulate_sbr(cfg, scene, n_rays_side=384, max_depth=1,
                      backend='mi_llvm')
    e2, e1 = np.linalg.norm(S2), np.linalg.norm(S1)
    drop_db = 20 * np.log10(e2 / max(e1, 1e-12))
    print(f'mi_llvm dihedral energy: depth2/depth1 = {drop_db:.1f} dB')
    assert drop_db >= 10.0
