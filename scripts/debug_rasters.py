"""Quick check on building density rasters."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rasterio
import numpy as np

bd_dir = Path('data/building_density')
for f in sorted(bd_dir.glob('*.tif')):
    try:
        with rasterio.open(f) as src:
            data = src.read(1)
            nonzero = (data > 0).sum()
            print(f'{f.name:<45} shape={data.shape}  nonzero={nonzero}  max={data.max():.1f}  crs={src.crs}')
    except Exception as e:
        print(f'{f.name:<45} ERROR: {e}')
