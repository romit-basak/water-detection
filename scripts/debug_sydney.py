"""Check Sydney chip counts by folder and date."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

root = Path('data/urban_sar_floods_preprocessed')
for folder in ['01_NF', '02_FO', '03_FU']:
    for date in ['20210324', '20220705']:
        chips = list((root / folder / 'SAR').glob(f'*{date}*Sydney*.npy'))
        if chips:
            print(f'{folder} / {date}: {len(chips)} chips')
