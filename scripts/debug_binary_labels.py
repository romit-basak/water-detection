"""Verify binary label remapping is correct for all folder types."""
import sys, numpy as np
sys.path.insert(0, '.')
from src.data.fast_dataset import FastUrbanSARFloods, make_urban_event_split
from pathlib import Path

root = Path('data/urban_sar_floods_preprocessed')

# Check one chip from each folder type directly
for folder in ['01_NF', '02_FO', '03_FU']:
    chips = sorted((root / folder / 'SAR').glob('*.npy'))[:1]
    gts   = sorted((root / folder / 'GT').glob('*.npy'))[:1]
    if not chips: continue
    
    # Simulate what FastUrbanSARFloods.__getitem__ does
    mask = np.load(gts[0]).astype('int64')
    print(f'{folder} raw stored values: {dict(zip(*[x.tolist() for x in np.unique(mask, return_counts=True)]))}')
    
    if folder == '01_NF':
        pass  # non-flood, keep as 0
    else:
        mask = np.where(mask == 0, -1,
               np.where(mask == 2,  1, mask)).astype('int64')
    
    print(f'{folder} after remap:       {dict(zip(*[x.tolist() for x in np.unique(mask, return_counts=True)]))}')
    print(f'  (-1=ignore, 0=non-flood, 1=flood)')
    print()
