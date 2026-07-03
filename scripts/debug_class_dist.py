"""
Check class distribution in UrbanSARFloods to diagnose the F1=0.5 plateau.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from collections import Counter
from src.data.dataset import FOLDER_CLASS

root = Path('data/urban_sar_floods')
counts = Counter()
n_chips = 0

for folder in sorted(FOLDER_CLASS.keys()):
    gt_dir = root / folder / 'GT'
    if not gt_dir.exists():
        continue
    chips = list(gt_dir.glob('*_GT.tif'))
    print(f'{folder}: {len(chips)} chips')
    for p in chips[:200]:  # sample 200 per folder
        with rasterio.open(p) as src:
            data = src.read(1).ravel()
        for v in [0, 1, 2]:
            counts[f'{folder}_class{v}'] += int((data == v).sum())
        n_chips += 1

total = sum(counts.values())
print(f'\nSampled {n_chips} chips, {total:,} pixels total')
print('\nClass distribution across all folders:')
c0 = sum(v for k,v in counts.items() if 'class0' in k)
c1 = sum(v for k,v in counts.items() if 'class1' in k)
c2 = sum(v for k,v in counts.items() if 'class2' in k)
print(f'  class 0 (background): {c0:>12,}  ({100*c0/total:.1f}%)')
print(f'  class 1 (open flood): {c1:>12,}  ({100*c1/total:.1f}%)')
print(f'  class 2 (urban flood):{c2:>12,}  ({100*c2/total:.1f}%)')
print(f'\nFlood prevalence (class 1+2 among non-background): {100*(c1+c2)/(c1+c2+1e-8):.1f}%')
print(f'Flood prevalence (class 1+2 among all pixels):    {100*(c1+c2)/total:.1f}%')

# With binary=True, class 0 = background (ignored), class 1 = any flood
# A model predicting all-flood gets:
#   recall = 1.0, precision = (c1+c2)/(c1+c2+c_nonflood_noignore)
# But with ignore_index=0, ALL pixels are either flood or... nothing?
# This reveals the real issue: with binary=True and ignore_index=0,
# there are NO non-flood pixels in the loss — only flood pixels (class 1)
# The model learns to predict class 1 everywhere and gets F1=1.0 on flood
# but the macro average over classes [1,2] collapses to 0.5
print('\n--- Binary mode analysis ---')
print('With binary=True: labels are 0 (background/ignore) and 1 (flood)')
print('With ignore_index=0: class 0 pixels are EXCLUDED from loss and metrics')
print('Result: the model only sees flood pixels in the loss signal')
print('A model predicting flood everywhere gets recall=1.0, precision=1.0')
print('because ALL non-ignored pixels ARE flood pixels.')
print('This is the degenerate solution.')
