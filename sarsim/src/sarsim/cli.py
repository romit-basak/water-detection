"""sarsim CLI.

    sarsim validate-paper3 [--n-pix 128] [--mf]   Gate-0 runner
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from . import analytic, recon, metrics
from .config import SimConfig

RUNS = Path(__file__).resolve().parents[3] / 'runs'


def validate_paper3(args) -> int:
    cfg_path = Path(__file__).resolve().parents[2] / 'configs' / 'paper_3point.yaml'
    cfg = SimConfig.from_yaml(cfg_path)
    targets = np.array(yaml.safe_load(cfg_path.read_text())['targets'],
                       dtype=np.float64)
    out = RUNS / 'sarsim_gate0'
    out.mkdir(parents=True, exist_ok=True)

    print(f'Gate 0 — {cfg.name}: K={cfg.sarsystem.n_freq} Np={cfg.trajectory.n_pulses}')
    print(f'  theory: rho_r={cfg.sarsystem.range_resolution:.4f} m, '
          f'rho_az={cfg.cross_range_resolution:.4f} m')

    t0 = time.time()
    S = analytic.simulate(cfg, targets)
    print(f'  phase history {S.shape} in {time.time()-t0:.2f}s '
          f'(|S| in [{np.abs(S).min():.3f}, {np.abs(S).max():.3f}])')

    t0 = time.time()
    img_bp, axis = recon.backproject(S, cfg, n_pix=args.n_pix)
    print(f'  backprojection {img_bp.shape} in {time.time()-t0:.2f}s')

    results = {'targets': targets.tolist(), 'bp': [], 'checks': {}}
    ok = True
    px = axis[1] - axis[0]
    for t in targets:
        xm, ym, amp = metrics.peak_near(img_bp, axis, (t[0], t[1]))
        err = float(np.hypot(xm - t[0], ym - t[1]))
        iy = int(np.argmin(np.abs(axis - ym)))
        ix = int(np.argmin(np.abs(axis - xm)))
        w_x = metrics.irw(img_bp, axis, iy, ix, 'x')
        p_db = metrics.pslr_db(img_bp, axis, iy, ix, 'x')
        results['bp'].append({'expected': t.tolist(),
                              'measured': [xm, ym], 'pos_err_m': err,
                              'amp': amp, 'irw_x_m': w_x, 'pslr_db': p_db})
        print(f'  target {t[:2]} -> ({xm:+.3f},{ym:+.3f}) err={err*100:.1f}cm '
              f'IRW_x={w_x:.3f}m PSLR={p_db:.1f}dB amp={amp:.4f}')
        # ½-pixel per axis (even grids have no pixel center at 0.0)
        ok &= abs(xm - t[0]) <= px / 2 + 1e-9
        ok &= abs(ym - t[1]) <= px / 2 + 1e-9

    amps = [r['amp'] for r in results['bp']]
    amp_spread = float(np.ptp(amps) / np.mean(amps))
    results['checks']['amp_spread'] = amp_spread
    results['checks']['pos_ok'] = bool(ok)
    ok &= amp_spread < 0.01

    if args.mf:
        t0 = time.time()
        img_mf, axis_mf = recon.matched_filter(S, cfg, n_pix=args.n_pix)
        print(f'  matched filter {img_mf.shape} in {time.time()-t0:.2f}s')
        for t in targets:
            xm, ym, amp = metrics.peak_near(img_mf, axis_mf, (t[0], t[1]))
            err = float(np.hypot(xm - t[0], ym - t[1]))
            print(f'  MF target {t[:2]} -> err={err*100:.1f}cm amp={amp:.4f}')
            ok &= err <= (axis_mf[1] - axis_mf[0]) / 2 + 1e-9
        # MF-vs-BP agreement on shared grid (dB-normalized central region)
        img_bp_c, _ = recon.backproject(S, cfg, n_pix=args.n_pix)
        n1 = lambda a: np.abs(a) / np.abs(a).max()
        nrmse = float(np.sqrt(np.mean((n1(img_mf) - n1(img_bp_c)) ** 2)))
        results['checks']['mf_bp_nrmse'] = nrmse
        print(f'  MF-vs-BP |img| NRMSE = {nrmse:.4f}')

    # save artifacts
    np.savez_compressed(out / 'phase_history.npz', S=S.astype(np.complex64),
                        f_k=cfg.sarsystem.f_k, targets=targets)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        db = 20 * np.log10(np.abs(img_bp) / np.abs(img_bp).max() + 1e-12)
        plt.figure(figsize=(6, 5))
        plt.imshow(db, extent=[axis[0], axis[-1], axis[-1], axis[0]],
                   vmin=-40, vmax=0, cmap='gray')
        plt.colorbar(label='dB')
        plt.scatter(targets[:, 0], targets[:, 1], s=80, facecolors='none',
                    edgecolors='r')
        plt.title(f'Gate 0 backprojection — {cfg.name}')
        plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.savefig(out / 'gate0_bp.png', dpi=150, bbox_inches='tight')
        print(f'  wrote {out / "gate0_bp.png"}')
    except Exception as e:                      # plotting is not gate-critical
        print(f'  (plot skipped: {e})')

    results['checks']['gate0_pass'] = bool(ok)
    (out / 'metrics.json').write_text(json.dumps(results, indent=2))
    print(f'GATE 0: {"PASS" if ok else "FAIL"}  -> {out}')
    return 0 if ok else 1


def flood_pilot(args) -> int:
    """Gate 2 — urban flood pilot: C-band SBR ± water, double-bounce check."""
    from . import scenes
    from .sbr import simulate_sbr
    from .config import SarSystem, Trajectory

    cfg = SimConfig(
        sarsystem=SarSystem(carrier_ghz=5.405, bandwidth_mhz=150.0,
                            n_freq=256),
        trajectory=Trajectory(mode='circular_spotlight', slant_range_m=10000.0,
                              depression_deg=52.5,       # incidence 37.5°
                              az_start_deg=269.2, az_end_deg=270.8,  # ρ_az≈1 m
                              n_pulses=128),
        scene_extent_m=40.0, pixel_m=0.25, name='flood_pilot',
    )
    # Radar at azimuth 270° (−y): the building row's front walls (y=2, normal
    # −y) FACE the radar and the water corridor (y<2) lies in FRONT of them —
    # the wall–water dihedral opens toward the radar. (First attempt looked
    # from +y: dihedral opened away + street sat in building shadow → +0.8 dB
    # nothing. Geometry matters.)
    out = RUNS / 'sarsim_flood_pilot'
    out.mkdir(parents=True, exist_ok=True)
    print(f'Gate 2 — {cfg.name}: C-band {cfg.sarsystem.carrier_ghz} GHz, '
          f'incidence {90 - cfg.trajectory.depression_deg:.1f}°, '
          f'rho_r={cfg.sarsystem.range_resolution:.2f} m, '
          f'rho_az={cfg.cross_range_resolution:.2f} m')

    imgs = {}
    for water in (True, False):
        tag = 'water' if water else 'dry'
        scene = scenes.flood_pilot(water=water)
        t0 = time.time()
        # seed=0: Monte-Carlo jittered rays (paper §4.2.1) — a deterministic
        # grid quantizes per-pulse amplitudes and smears a bright dihedral
        # into an image-wide noise pedestal (first-run finding: flat +16 dB)
        S = simulate_sbr(cfg, scene, n_rays_side=args.n_rays, max_depth=3,
                         shadow_rays=True, accumulator='binned', seed=0,
                         progress=True)
        # hann: corner line is 30+ dB over soil; -13 dB sinc sidelobes would
        # contaminate the open-water metric window
        img, axis = recon.backproject(S, cfg, window='hann')
        imgs[tag] = img
        print(f'  {tag}: {len(scene.mat_idx)} tris, S{S.shape}, '
              f'{time.time()-t0:.0f}s')
        np.savez_compressed(out / f'S_{tag}.npz', S=S.astype(np.complex64))

    # Gate-2 metrics
    X, Y = np.meshgrid(axis, axis)
    strip = (np.abs(X) <= 16) & (Y >= 1.0) & (Y <= 2.0)   # wall-water corner
    # open water BEYOND wall-layover reach (walls lay over y in [-5.8, 2])
    openw = (np.abs(X) <= 16) & (Y >= -12.0) & (Y <= -8.0)
    p2 = lambda img, m: float(np.mean(np.abs(img[m]) ** 2))
    strip_gain_db = 10 * np.log10(p2(imgs['water'], strip)
                                  / p2(imgs['dry'], strip))
    open_drop_db = 10 * np.log10(p2(imgs['water'], openw)
                                 / p2(imgs['dry'], openw))
    ok = (strip_gain_db >= 6.0) and (open_drop_db <= -5.0)
    print(f'  wall-strip gain (water vs dry): {strip_gain_db:+.1f} dB '
          f'(gate: >= +6)')
    print(f'  open-street change:             {open_drop_db:+.1f} dB '
          f'(gate: <= -5)')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        ref = max(np.abs(imgs['water']).max(), np.abs(imgs['dry']).max())
        for ax, tag in zip(axs[:2], ('dry', 'water')):
            db = 20 * np.log10(np.abs(imgs[tag]) / ref + 1e-12)
            im = ax.imshow(db, extent=[axis[0], axis[-1], axis[-1], axis[0]],
                           vmin=-45, vmax=0, cmap='gray')
            ax.set_title(f'{tag} (dB)'); ax.set_xlabel('x (m)')
            plt.colorbar(im, ax=ax, shrink=0.8)
        d = (20 * np.log10(np.abs(imgs['water']) + 1e-12)
             - 20 * np.log10(np.abs(imgs['dry']) + 1e-12))
        im = axs[2].imshow(d, extent=[axis[0], axis[-1], axis[-1], axis[0]],
                           vmin=-15, vmax=15, cmap='RdBu_r')
        axs[2].set_title('water − dry (dB)')
        plt.colorbar(im, ax=axs[2], shrink=0.8)
        fig.suptitle('sarsim flood pilot — C-band SBR, depth 3, shadow rays')
        fig.savefig(out / 'gate2_pilot.png', dpi=150, bbox_inches='tight')
        print(f'  wrote {out / "gate2_pilot.png"}')
    except Exception as e:
        print(f'  (plot skipped: {e})')

    (out / 'metrics.json').write_text(json.dumps(
        {'strip_gain_db': strip_gain_db, 'open_drop_db': open_drop_db,
         'gate2_pass': bool(ok)}, indent=2))
    print(f'GATE 2: {"PASS" if ok else "FAIL"}  -> {out}')
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(prog='sarsim')
    sub = p.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate-paper3', help='Gate-0 3-point validation')
    v.add_argument('--n-pix', type=int, default=128)
    v.add_argument('--mf', action='store_true', help='also run matched filter')
    v.set_defaults(fn=validate_paper3)
    f = sub.add_parser('flood-pilot', help='Gate-2 urban flood pilot ± water')
    f.add_argument('--n-rays', type=int, default=512,
                   help='rays per side of the ground grid')
    f.set_defaults(fn=flood_pilot)
    args = p.parse_args()
    return args.fn(args)


if __name__ == '__main__':
    raise SystemExit(main())
