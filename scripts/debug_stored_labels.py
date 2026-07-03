"""Check what values are actually stored in precomputed label files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

print('=== Sen1Floods11 label files ===')
lbl_dir = Path('data/sen1floods11_preprocessed/LabelHand')
for p in sorted(lbl_dir.glob('*.npy'))[:5]:
    d = np.load(p)
    vals, counts = np.unique(d, return_counts=True)
    print(f'  {p.name}: {dict(zip(vals.tolist(), counts.tolist()))}')

print('\n=== UrbanSARFloods 01_NF GT ===')
for p in sorted(Path('data/urban_sar_floods_preprocessed/01_NF/GT').glob('*.npy'))[:3]:
    d = np.load(p)
    vals, counts = np.unique(d, return_counts=True)
    print(f'  {p.name}: {dict(zip(vals.tolist(), counts.tolist()))}')

print('\n=== UrbanSARFloods 03_FU GT ===')
for p in sorted(Path('data/urban_sar_floods_preprocessed/03_FU/GT').glob('*.npy'))[:3]:
    d = np.load(p)
    vals, counts = np.unique(d, return_counts=True)
    print(f'  {p.name}: {dict(zip(vals.tolist(), counts.tolist()))}')
