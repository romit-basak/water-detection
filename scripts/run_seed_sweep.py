"""
scripts/run_seed_sweep.py

Multi-seed sweep driver for the "cross-city ceiling" study.

Runs the 4 headline conditions — ALL from the single train_with_pseudo.py
harness (this retires train_three_stage.py as a second codebase and removes the
report's "top rows from a different pipeline / estimated macro F1" caveat) —
across N seeds, so each condition gets a real ACROSS-SEED error bar instead of
the old within-run across-epoch SD.

Each run writes runs/seed_sweep/<config>/seed<k>/stage2/summary.json with three
point estimates (val-selected [leakage-free], peak-over-test [leaky], final5).
Aggregate with scripts/aggregate_seed_sweep.py.

RESUMABLE: a run whose summary.json already exists is skipped, so an
overnight/multi-night sweep can be stopped and restarted freely.

Usage (SMOKE — proves plumbing, meaningless numbers):
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/run_seed_sweep.py \
        --seeds 0 --finetune_epochs 2 --pseudo_epochs 2 --max_chips 24

Usage (FULL sweep — overnight/multi-night, deferred to the user):
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/run_seed_sweep.py \
        --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_with_pseudo import DEFAULTS, main as train_main

# The 4 conditions, each fully specified as overrides on train_with_pseudo DEFAULTS.
CONFIGS = {
    'sar_only_fw10': dict(finetune_only=True,  flood_weight=10.0,
                          urban_sar_dir='data/urban_sar_floods_preprocessed'),
    'aux_fw10':      dict(finetune_only=True,  flood_weight=10.0,
                          urban_sar_dir='data/urban_sar_floods_aux'),
    'aux_fw3':       dict(finetune_only=True,  flood_weight=3.0,
                          urban_sar_dir='data/urban_sar_floods_aux'),
    'pseudo_ft_fw3': dict(finetune_only=False, flood_weight=3.0,
                          urban_sar_dir='data/urban_sar_floods_aux'),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--configs', type=str, nargs='+', default=list(CONFIGS),
                   choices=list(CONFIGS))
    p.add_argument('--out_root', type=str, default='runs/seed_sweep')
    p.add_argument('--finetune_epochs', type=int, default=DEFAULTS['finetune_epochs'])
    p.add_argument('--pseudo_epochs',   type=int, default=DEFAULTS['pseudo_epochs'])
    p.add_argument('--max_chips',       type=int, default=0,
                   help='>0 caps each split (SMOKE TEST ONLY)')
    return p.parse_args()


def main():
    args     = parse_args()
    out_root = Path(args.out_root)
    planned  = [(c, s) for c in args.configs for s in args.seeds]
    print(f'Planned runs: {len(planned)} ({len(args.configs)} configs x '
          f'{len(args.seeds)} seeds)')
    if args.max_chips:
        print(f'*** SMOKE MODE: max_chips={args.max_chips}, '
              f'finetune_epochs={args.finetune_epochs}, '
              f'pseudo_epochs={args.pseudo_epochs} — numbers are meaningless ***')

    for config_name, seed in planned:
        out_dir = out_root / config_name / f'seed{seed}'
        summary = out_dir / 'stage2' / 'summary.json'
        if summary.exists():
            print(f'[skip] {config_name}/seed{seed} — summary.json exists')
            continue

        print('\n' + '#' * 70)
        print(f'# RUN: config={config_name}  seed={seed}  ->  {out_dir}')
        print('#' * 70)

        cfg = dict(DEFAULTS)
        cfg.update(CONFIGS[config_name])
        cfg.update(
            seed=seed,
            out_dir=str(out_dir),
            finetune_epochs=args.finetune_epochs,
            pseudo_epochs=args.pseudo_epochs,
            max_chips=args.max_chips,
        )
        train_main(cfg)

    print('\nSweep complete. Aggregate with:')
    print(f'  uv run python scripts/aggregate_seed_sweep.py --out_root {args.out_root}')


if __name__ == '__main__':
    main()
