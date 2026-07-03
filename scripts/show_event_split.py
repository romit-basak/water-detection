"""Show chip counts per event in UrbanSARFloods."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import defaultdict

root = Path('data/urban_sar_floods_preprocessed')
counts = defaultdict(int)
for folder in ['01_NF', '02_FO', '03_FU']:
    for p in (root / folder / 'SAR').glob('*.npy'):
        event = p.stem.split('_')[1]
        counts[event] += 1

train = ['Houston','Beira','Japan','Canada','Iran','Lumberton','Somalia']
test  = ['Hagibis','Sydney','Coraki','Niger','Hebei','Beledweyne','PortMacquarie']

print(f'{"Event":<15} {"Split":<8} {"Chips":>6}')
print('-' * 32)
total_train, total_test = 0, 0
for event in sorted(counts):
    split = 'train' if event in train else 'test' if event in test else '???'
    n = counts[event]
    print(f'{event:<15} {split:<8} {n:>6}')
    if split == 'train': total_train += n
    elif split == 'test': total_test += n
print('-' * 32)
print(f'{"TRAIN TOTAL":<15} {"":8} {total_train:>6}')
print(f'{"TEST TOTAL":<15} {"":8} {total_test:>6}')
