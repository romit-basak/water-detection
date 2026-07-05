"""D6 swarm worker: render acquisition-pair corpus seeds -> npz chips in GCS.
Idempotent.

Each seed's object holds one acquisition PAIR: vv_t1_db, vv_t2_db, coh
(10 m, f16), three GT masks (mask_peak/mask_standing_t2/mask_drained_t2),
and full meta (scene + hydro + pair + look draws + natural_weight). See
docs/SYNTHETIC_CORPUS.md for the design. Skip-if-exists makes preemption/
restart/multi-worker overlap all safe; sharding partitions the seed range
without coordination.

Worker (VM or M4):
  python3 scripts/render_corpus.py --count 2000 --shard 0/6 \
      --gcs gs://urban-water-detection-sweep/synthetic_corpus/v1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from sarsim.corpus import render_pair


def gcs_exists(url: str) -> bool:
    return subprocess.run(['gcloud', 'storage', 'ls', url],
                          capture_output=True).returncode == 0


def gcs_put(local: Path, url: str) -> None:
    subprocess.run(['gcloud', 'storage', 'cp', str(local), url], check=True,
                   capture_output=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--shard', default='0/1', help='k/N seed partition')
    ap.add_argument('--rotate', action='store_true',
                    help='swarm mode: all seeds, rotated start (self-healing)')
    ap.add_argument('--extent', type=float, default=320.0)
    ap.add_argument('--rays', type=int, default=384)
    ap.add_argument('--rho', type=float, default=2.5)
    ap.add_argument('--backend', default='mi_llvm')
    ap.add_argument('--gcs', default=None,
                    help='gs://... prefix; omit for --local-dir only')
    ap.add_argument('--local-dir', default=None)
    args = ap.parse_args()
    k, n = map(int, args.shard.split('/'))
    local = Path(args.local_dir) if args.local_dir else None
    if local:
        local.mkdir(parents=True, exist_ok=True)
    if not args.gcs and not local:
        sys.exit('need --gcs and/or --local-dir')

    if args.rotate:
        # self-healing swarm mode: every worker walks ALL seeds, starting at
        # its own offset — skip-if-exists dedupes, and a preempted worker's
        # remainder is absorbed by the others (strict shards would strand it)
        allseeds = list(range(args.start, args.start + args.count))
        off = (k * len(allseeds)) // n
        seeds = allseeds[off:] + allseeds[:off]
    else:
        seeds = [s for s in range(args.start, args.start + args.count)
                 if s % n == k]
    print(f'shard {k}/{n}: {len(seeds)} seeds '
          f'({args.extent:.0f} m, rays {args.rays}, {args.backend})',
          flush=True)
    done = skipped = 0
    t0 = time.time()
    for seed in seeds:
        name = f'seed{seed:06d}.npz'
        url = f'{args.gcs.rstrip("/")}/{name}' if args.gcs else None
        if url and gcs_exists(url):
            skipped += 1
            continue
        if local and (local / name).exists():
            skipped += 1
            continue
        item, meta, wall = render_pair(
            seed, extent_m=args.extent, n_rays_side=args.rays,
            rho_m=args.rho, backend=args.backend)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / name
            np.savez_compressed(tmp, **item, meta=json.dumps(meta))
            if local:
                (local / name).write_bytes(tmp.read_bytes())
            if url:
                gcs_put(tmp, url)
        done += 1
        rate = (time.time() - t0) / max(done, 1)
        print(f'[{done}/{len(seeds)} +{skipped} skip] seed {seed} '
              f'{wall:.0f}s render, {rate:.0f}s/seed avg', flush=True)
    print(f'shard {k}/{n} COMPLETE: {done} rendered, {skipped} skipped',
          flush=True)


if __name__ == '__main__':
    main()
