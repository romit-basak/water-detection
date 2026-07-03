"""Check what bands are in Sen1Floods11 GeoTIFF files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rasterio
import numpy as np

s1 = Path('data/sen1floods11/S1Hand')
for p in sorted(s1.glob('*.tif'))[:3]:
    with rasterio.open(p) as src:
        print(f'{p.name}:')
        print(f'  bands: {src.count}')
        print(f'  crs: {src.crs}')
        print(f'  bounds: {src.bounds}')
        for i in range(1, src.count+1):
            d = src.read(i).ravel()
            print(f'  band {i}: min={d.min():.4f} max={d.max():.4f} mean={d.mean():.4f}')
        print()
