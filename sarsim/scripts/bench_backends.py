"""G1: backend throughput benchmark — mi_llvm (CPU/Embree) vs mi_cuda
(OptiX/RT cores), same scenes, same physics code path.

Reference points: Willis/Hossain/Godwin report 1.53-4.26 Mray/s on an RTX
2070; our M4 mi_llvm city smoke ran ~6 Mray/s. Rays counted as primary
rays only (pulses x rays_side^2) — bounce children add work that varies
per scene, so primary-ray throughput is the conservative, comparable rate.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from sarsim import scenes
from sarsim.city import CityParams, generate_city
from sarsim.config import SarSystem, SimConfig, Trajectory
from sarsim.corpus import corpus_config
from sarsim.sbr import simulate_sbr


def bench_cases():
    pilot = scenes.flood_pilot(water=True)
    gate2 = SimConfig(
        sarsystem=SarSystem(carrier_ghz=5.405, bandwidth_mhz=150.0,
                            n_freq=256),
        trajectory=Trajectory(mode='circular_spotlight',
                              slant_range_m=10000.0, depression_deg=52.5,
                              az_start_deg=269.2, az_end_deg=270.8,
                              n_pulses=64),          # half Gate-2 pulses
        scene_extent_m=40.0, pixel_m=0.25, name='pilot')
    city_s, _, _, _ = generate_city(CityParams(seed=0, extent_m=320.0))
    ccfg = corpus_config(320.0, 37.5, 270.0)
    ccfg.trajectory.n_pulses = 64                     # benchmark slice
    return [
        ('pilot_54tri', pilot, gate2, 512),
        ('city320', city_s, ccfg, 384),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backends', nargs='+', default=['mi_llvm'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out = args.out or 'bench_backends.json'
    rows = []

    def flush():
        # Persist after every row so a hard crash in a later case (e.g. a
        # mismatched OptiX lib segfaulting mi_cuda) never loses rows already
        # measured.
        result = {'host': platform.node(), 'machine': platform.machine(),
                  'rows': rows}
        Path(out).write_text(json.dumps(result, indent=2))

    for backend in args.backends:
        for name, scene, cfg, rays in bench_cases():
            try:
                t0 = time.perf_counter()
                simulate_sbr(cfg, scene, n_rays_side=rays, max_depth=3,
                             shadow_rays=True, seed=0, backend=backend,
                             accumulator='binned', bin_frac_of_lambda=1.0)
                dt = time.perf_counter() - t0
            except Exception as e:      # a backend missing on this host
                print(f'{backend:8s} {name:18s} SKIP: {e}', flush=True)
                rows.append(dict(backend=backend, scene=name,
                                 error=str(e)[:200]))
                flush()
                continue
            primary = cfg.trajectory.n_pulses * rays * rays
            r = dict(backend=backend, scene=name,
                     tris=int(len(scene.mat_idx)),
                     pulses=cfg.trajectory.n_pulses, rays_side=rays,
                     wall_s=round(dt, 2),
                     mray_s=round(primary / dt / 1e6, 2))
            rows.append(r)
            flush()
            print(f'{backend:8s} {name:18s} {r["tris"]:>6d} tris '
                  f'{r["wall_s"]:>7.1f}s  {r["mray_s"]:>6.2f} Mray/s',
                  flush=True)

    print(f'-> {out}')


if __name__ == '__main__':
    main()
