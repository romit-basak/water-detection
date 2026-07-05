"""Gate 1 (reference tracer): SBR-vs-analytic + dihedral double-bounce."""
import numpy as np
import pytest

from sarsim.config import SarSystem, Trajectory, SimConfig
from sarsim import analytic, recon, metrics, scenes
from sarsim.sbr import simulate_sbr


@pytest.fixture
def cfg():
    return SimConfig(
        sarsystem=SarSystem(carrier_ghz=10.0, bandwidth_mhz=600.0, n_freq=128),
        trajectory=Trajectory(mode='circular_spotlight', slant_range_m=10000.0,
                              depression_deg=30.0, az_start_deg=48.5,
                              az_end_deg=51.5, n_pulses=64),
        scene_extent_m=10.0, pixel_m=0.05,
    )


def test_gate1_plates_match_analytic(cfg):
    """Plate proxies at the 3 point positions reproduce the analytic phase
    history after one global complex calibration."""
    targets = np.array([[0.0, 0.0, 0.0], [-2.0, -2.0, 0.0], [3.0, 1.0, 0.0]])
    S_an = analytic.simulate(cfg, targets)

    mid = cfg.trajectory.positions()[cfg.trajectory.n_pulses // 2]
    scene = scenes.plates_at_points(targets, aim_from=mid)
    S_rt = simulate_sbr(cfg, scene, n_rays_side=512, max_depth=1)

    a, b = S_rt.ravel(), S_an.ravel()
    s = np.vdot(b, a) / np.vdot(b, b)          # least-squares complex scale
    corr = abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    nrmse = np.linalg.norm(a - s * b) / np.linalg.norm(s * b)
    print(f'gate1: corr={corr:.6f} nrmse={nrmse:.4f} scale={abs(s):.3e}')
    assert corr > 0.999
    assert nrmse < 0.05

    # reconstructed peak positions must match per-axis to 1/2 pixel
    img, axis = recon.backproject(S_rt, cfg, n_pix=200)
    px = axis[1] - axis[0]
    for t in targets:
        xm, ym, _ = metrics.peak_near(img, axis, (t[0], t[1]))
        assert abs(xm - t[0]) <= px / 2 + 1e-9
        assert abs(ym - t[1]) <= px / 2 + 1e-9


def test_gate1_dihedral_double_bounce(cfg):
    """Corner response localizes at the corner range and collapses (≥10 dB)
    when multi-bounce is disabled."""
    scene = scenes.dihedral(corner_xy=(0.0, 0.0), size_m=2.0,
                            open_toward_deg=50.0)

    S2 = simulate_sbr(cfg, scene, n_rays_side=512, max_depth=2)
    S1 = simulate_sbr(cfg, scene, n_rays_side=512, max_depth=1)

    img2, axis = recon.backproject(S2, cfg, n_pix=200)
    img1, _ = recon.backproject(S1, cfg, n_pix=200)

    # measure energy in a small window around the corner (0,0)
    ix = np.abs(axis) < 0.5
    win2 = np.abs(img2[np.ix_(ix, ix)]).max()
    win1 = np.abs(img1[np.ix_(ix, ix)]).max()
    drop_db = 20 * np.log10(win2 / max(win1, 1e-12))
    print(f'dihedral corner peak: depth2={win2:.3e} depth1={win1:.3e} '
          f'drop={drop_db:.1f} dB')
    assert drop_db >= 10.0

    # A dihedral is a LINE target: its return localizes on the corner line
    # (along-direction position is arbitrary within the plate extent). Check
    # the peak's offset PERPENDICULAR to the corner line (range direction).
    xm, ym, _ = metrics.peak_near(img2, axis, (0.0, 0.0), radius_m=1.5)
    az = np.deg2rad(50.0)
    range_offset = abs(xm * np.cos(az) + ym * np.sin(az))
    along_offset = abs(-xm * np.sin(az) + ym * np.cos(az))
    print(f'dihedral peak at ({xm:+.2f},{ym:+.2f}): range_off='
          f'{range_offset:.3f} m, along_off={along_offset:.3f} m')
    assert range_offset < 0.15          # ON the corner line (range dir)
    assert along_offset <= 1.1          # within the 2 m plate extent
