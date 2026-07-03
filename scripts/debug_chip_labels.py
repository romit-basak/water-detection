"""Quick diagnostic: check label distribution in flooded chips."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data.fast_dataset import FastUrbanSARFloods, make_urban_event_split

FINETUNE_TEST_EVENTS = [
    'Hagibis', 'Sydney', 'Coraki', 'Niger', 'Hebei', 'Beledweyne', 'PortMacquarie',
]
FINETUNE_TRAIN_EVENTS = [
    'Houston', 'Beira', 'Japan', 'Canada', 'Iran', 'Lumberton', 'Somalia',
]

_, test_idx = make_urban_event_split(
    'data/urban_sar_floods_preprocessed',
    train_events=FINETUNE_TRAIN_EVENTS,
    test_events=FINETUNE_TEST_EVENTS,
)

ds = FastUrbanSARFloods('data/urban_sar_floods_preprocessed', test_idx, binary=False)

# Sample 20 flooded chips
flooded = [(i, ds[i]) for i in range(len(ds))
           if ds[i]['mask'].max().item() > 0][:20]

print(f'Total test chips: {len(ds)}')
print(f'Chips with any flood label: checking...')

all_flood_fracs = []
for i, s in flooded:
    mask = s['mask'].numpy()
    total = mask.size
    bg    = (mask == 0).sum()
    flood = ((mask == 1) | (mask == 2)).sum()
    frac  = flood / total
    all_flood_fracs.append(frac)
    print(f'  chip {i}: bg={bg/total:.1%}  flood={flood/total:.1%}  '
          f'(class1={( mask==1).sum()}  class2={(mask==2).sum()})')

print(f'\nMean flood fraction in flooded chips: {np.mean(all_flood_fracs):.3f}')
print(f'This means ~{np.mean(all_flood_fracs)*100:.1f}% of pixels are flood')
print('A model predicting flood everywhere gets:')
ff = np.mean(all_flood_fracs)
recall = 1.0
prec = ff  # flood pixels / all pixels
f1 = 2*prec*recall/(prec+recall)
print(f'  precision={prec:.3f}  recall={recall:.3f}  F1={f1:.3f}')
print(f'So F1=0.077 means the model is doing WORSE than predicting flood everywhere')
print(f'It is predicting very few flood pixels — essentially predicting nothing')
