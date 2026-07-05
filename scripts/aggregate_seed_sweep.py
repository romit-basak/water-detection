"""
scripts/aggregate_seed_sweep.py

Aggregate the multi-seed sweep (scripts/run_seed_sweep.py) into the numbers that
decide the "cross-city ceiling" question.

For each config, reports mean ± ACROSS-SEED SD for all three point estimates,
side by side:
    val_selected   (LEAKAGE-FREE — checkpoint chosen on val, test F1 reported)
    peak_over_test (LEAKY — matches the old report's "peak flood F1")
    final5_mean    (robustness)

Then answers, per protocol:
  1. Do any two configs differ by MORE than the pooled across-seed SD?
     (the honest, assumption-light check the writeup can lead with)
  2. Welch's t-test between each config pair (if scipy is available), so the
     claim "differences are within seed noise" has a p-value behind it.

Also prints, per config, the peak-vs-val-selected gap = the amount the old
test-set-selection protocol inflated the reported ceiling.

Usage:
    uv run python scripts/aggregate_seed_sweep.py --out_root runs/seed_sweep
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _scipy_stats
except Exception:  # scipy optional
    _scipy_stats = None

PROTOCOLS = ['val_selected', 'peak_over_test', 'final5_mean']


def _get_f1(summary: dict, protocol: str) -> float | None:
    node = summary.get(protocol)
    if not node:
        return None
    return node.get('test_flood_f1')


def collect(out_root: Path) -> dict[str, dict[str, list[float]]]:
    """-> {config_name: {protocol: [f1 across seeds]}}"""
    data: dict[str, dict[str, list[float]]] = {}
    for summary_path in sorted(out_root.glob('*/seed*/stage2/summary.json')):
        config_name = summary_path.parents[2].name
        summary     = json.loads(summary_path.read_text())
        cfg         = data.setdefault(config_name, {p: [] for p in PROTOCOLS})
        for p in PROTOCOLS:
            v = _get_f1(summary, p)
            if v is not None and not math.isnan(v):
                cfg[p].append(float(v))
    return data


def mean_sd(xs: list[float]) -> tuple[float, float, int]:
    n = len(xs)
    if n == 0:
        return float('nan'), float('nan'), 0
    m  = float(np.mean(xs))
    sd = float(np.std(xs, ddof=1)) if n > 1 else 0.0
    return m, sd, n


def print_table(data: dict) -> None:
    configs = sorted(data)
    print('\n' + '=' * 78)
    print('PER-CONFIG FLOOD F1  (mean ± across-seed SD, n seeds)')
    print('=' * 78)
    header = f'{"config":<16}' + ''.join(f'{p:>21}' for p in PROTOCOLS)
    print(header)
    print('-' * len(header))
    for c in configs:
        row = f'{c:<16}'
        for p in PROTOCOLS:
            m, sd, n = mean_sd(data[c][p])
            row += f'{f"{m:.3f}±{sd:.3f} (n={n})":>21}'
        print(row)

    print('\nInflation from test-set selection (peak_over_test - val_selected):')
    for c in configs:
        mv, _, _ = mean_sd(data[c]['val_selected'])
        mp, _, _ = mean_sd(data[c]['peak_over_test'])
        print(f'  {c:<16} +{mp - mv:.3f}  (leaky {mp:.3f} vs leakage-free {mv:.3f})')


def print_pairwise(data: dict) -> None:
    configs = sorted(data)
    for p in PROTOCOLS:
        print('\n' + '=' * 78)
        print(f'PAIRWISE COMPARISON — protocol: {p}')
        print('=' * 78)
        for a, b in combinations(configs, 2):
            xa, xb = data[a][p], data[b][p]
            ma, sda, na = mean_sd(xa)
            mb, sdb, nb = mean_sd(xb)
            if na < 1 or nb < 1:
                continue
            gap    = abs(ma - mb)
            pooled = math.sqrt((sda ** 2 + sdb ** 2) / 2) if (na > 1 and nb > 1) else float('nan')
            sep    = 'SEPARATES' if (pooled == pooled and gap > pooled) else 'within noise'
            line = (f'  {a:<15} vs {b:<15} | gap={gap:.3f} '
                    f'pooledSD={pooled:.3f} -> {sep}')
            if _scipy_stats is not None and na > 1 and nb > 1:
                t, pval = _scipy_stats.ttest_ind(xa, xb, equal_var=False)
                line += f' | Welch t={t:+.2f} p={pval:.3f}'
            print(line)
        if _scipy_stats is None:
            print('  (install scipy for Welch t-test p-values)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', type=str, default='runs/seed_sweep')
    args = ap.parse_args()

    out_root = Path(args.out_root)
    data     = collect(out_root)
    if not data:
        print(f'No summary.json files found under {out_root}/*/seed*/stage2/. '
              'Run scripts/run_seed_sweep.py first.')
        return

    print_table(data)
    print_pairwise(data)
    print('\nInterpretation: if every pair is "within noise" under the '
          'val_selected (leakage-free) protocol, the ceiling is confirmed and '
          'the bottleneck is labeled-city diversity — not features/loss/pseudo.')


if __name__ == '__main__':
    main()
