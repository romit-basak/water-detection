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


def main() -> int:
    p = argparse.ArgumentParser(prog='sarsim')
    sub = p.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate-paper3', help='Gate-0 3-point validation')
    v.add_argument('--n-pix', type=int, default=128)
    v.add_argument('--mf', action='store_true', help='also run matched filter')
    v.set_defaults(fn=validate_paper3)
    args = p.parse_args()
    return args.fn(args)


if __name__ == '__main__':
    raise SystemExit(main())
