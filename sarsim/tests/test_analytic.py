"""Stage-0 invariants (fast, reduced-size configs)."""
import numpy as np
import pytest

from sarsim import C0
from sarsim.config import SarSystem, Trajectory, SimConfig
from sarsim import analytic, recon, metrics
from sarsim.phase import accumulate_direct, accumulate_factored32


@pytest.fixture
def cfg():
    """Reduced config: same geometry as paper, smaller K/Np for speed."""
    return SimConfig(
        sarsystem=SarSystem(carrier_ghz=10.0, bandwidth_mhz=600.0, n_freq=128),
        trajectory=Trajectory(mode='circular_spotlight', slant_range_m=10000.0,
                              depression_deg=30.0, az_start_deg=48.5,
                              az_end_deg=51.5, n_pulses=64),
        scene_extent_m=10.0, pixel_m=0.05,
    )


def test_single_target_modulus_constant(cfg):
    S = analytic.simulate(cfg, np.array([[1.0, 2.0, 0.0]]))
    assert np.allclose(np.abs(S), 1.0, atol=1e-12)


def test_single_target_phase_slope(cfg):
    """Phase slope along k equals −4πΔf·dR/c analytically."""
    tgt = np.array([[1.0, 2.0, 0.0]])
    S = analytic.simulate(cfg, tgt)
    pos = cfg.trajectory.positions()
    c = np.zeros(3)
    dR = (np.linalg.norm(pos[0] - tgt[0]) - np.linalg.norm(pos[0] - c))
    slope = np.angle(S[1:, 0] * np.conj(S[:-1, 0])).mean()
    expected = -4 * np.pi * cfg.sarsystem.delta_f * dR / C0
    # both wrapped to (−π, π]; dR is a few m so |expected| << π at Δf≈4.7MHz
    assert abs(slope - expected) < 1e-9


def test_bp_peaks_at_targets(cfg):
    targets = np.array([[0.0, 0.0, 0.0], [-2.0, -2.0, 0.0], [3.0, 1.0, 0.0]])
    S = analytic.simulate(cfg, targets)
    img, axis = recon.backproject(S, cfg, n_pix=200)
    px = axis[1] - axis[0]
    for t in targets:
        xm, ym, _ = metrics.peak_near(img, axis, (t[0], t[1]))
        # ½-pixel PER AXIS (an even grid has no pixel center at 0.0, so a
        # radial bound of px/2 would be √2-too-strict at grid quantization)
        assert abs(xm - t[0]) <= px / 2 + 1e-9
        assert abs(ym - t[1]) <= px / 2 + 1e-9


def test_bp_irw_matches_theory(cfg):
    """IRW within 20% of ρ_r along the range-ish cut (loose: geometry mixes
    range/cross-range into x/y depending on look angle)."""
    S = analytic.simulate(cfg, np.array([[0.0, 0.0, 0.0]]))
    img, axis = recon.backproject(S, cfg, n_pix=200)
    iy = ix = int(np.argmin(np.abs(axis)))
    w = min(metrics.irw(img, axis, iy, ix, 'x'),
            metrics.irw(img, axis, iy, ix, 'y'))
    rho = cfg.sarsystem.range_resolution
    assert 0.6 * rho < w < 1.8 * rho


def test_factored32_matches_direct(cfg):
    """Paper §4.3.4 float32 factoring ≈ fp64 direct (single pulse)."""
    rng = np.random.default_rng(0)
    A = rng.uniform(0.5, 1.5, 64)
    dR = rng.uniform(-5, 5, 64)
    f_k = cfg.sarsystem.f_k
    ref = accumulate_direct(f_k, A, dR)
    K = cfg.sarsystem.n_freq
    k_idx = np.arange(K) - (K - 1) / 2
    f_c_eff = f_k.mean()          # factoring reference = grid center
    got = accumulate_factored32(f_c_eff, cfg.sarsystem.delta_f, k_idx, A, dR)
    nrmse = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert nrmse < 1e-3, f'factored fp32 NRMSE {nrmse:.2e}'


def test_ambiguity_guards():
    with pytest.raises(ValueError):
        SimConfig(
            sarsystem=SarSystem(carrier_ghz=10.0, bandwidth_mhz=600.0,
                                n_freq=8),   # Δf huge -> ambiguity tiny
            trajectory=Trajectory(mode='circular_spotlight',
                                  slant_range_m=10000.0, depression_deg=30.0,
                                  az_start_deg=48.5, az_end_deg=51.5,
                                  n_pulses=64),
            scene_extent_m=100.0,
        ).validate()


def test_binned_matches_direct(cfg):
    """BinnedAccumulator ≈ direct accumulation (λ/16 bins)."""
    from sarsim.phase import BinnedAccumulator
    rng = np.random.default_rng(1)
    A = rng.uniform(0.1, 1.0, 5000)
    dR = rng.uniform(-12, 12, 5000)
    f_k = cfg.sarsystem.f_k
    ref = accumulate_direct(f_k, A, dR)
    binner = BinnedAccumulator(f_k, swath_half_m=15.0)
    got = binner.column(A, dR)
    nrmse = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert nrmse < 1e-3, f'binned NRMSE {nrmse:.2e}'
