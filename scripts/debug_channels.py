"""Verify channel counts for both preprocessed datasets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

# Sen1Floods11 — should be 5 channels (VV, VH, varVV, varVH, incidence_angle)
s1 = next(Path('data/sen1floods11_preprocessed/S1Hand').glob('*.npy'))
d = np.load(s1)
print(f'Sen1Floods11: {d.shape}  ({d.shape[0]} channels)')
for i, name in enumerate(['VV','VH','varVV','varVH','incidence_angle']):
    print(f'  ch{i} {name}: min={d[i].min():.3f} max={d[i].max():.3f} mean={d[i].mean():.4f}')

print()

# UrbanSARFloods aux — should be 8 channels
sar = next(Path('data/urban_sar_floods_aux/03_FU/SAR').glob('*.npy'))
d2 = np.load(sar)
print(f'UrbanSARFloods aux: {d2.shape}  ({d2.shape[0]} channels)')
for i, name in enumerate(['VV','VH','varVV','varVH','buildings','worldcover','hand','incidence_angle']):
    print(f'  ch{i} {name}: min={d2[i].min():.3f} max={d2[i].max():.3f} mean={d2[i].mean():.4f}')
