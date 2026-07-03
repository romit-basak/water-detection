"""
Per-city flood F1 breakdown using direct filename parsing.
Groups chips by their second underscore-separated token.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/per_city_eval.py
"""
import sys, torch
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader, Subset
from src.data.fast_dataset import FastUrbanSARFloods, make_urban_event_split
from src.data.transforms import val_transforms
from src.models.unet import build_unet
from src.utils.metrics import SegmentationMetrics

TRAIN_EVENTS = ['Houston','Beira','Japan','Canada','Iran','Lumberton','Somalia']
TEST_EVENTS  = ['Hagibis','Sydney','Coraki','Niger','Hebei','Beledweyne','PortMacquarie']
N_CLASSES    = 2
IGNORE_INDEX = -1

RUNS = [
    ('SAR-only (fw=10)',      'runs/three_stage_v8_sar_only/stage3/best.pt',
                              'data/urban_sar_floods_preprocessed'),
    ('Aux fw=10',             'runs/three_stage_v8_aux/stage3/best.pt',
                              'data/urban_sar_floods_aux'),
    ('Aux fw=3  (best)',      'runs/three_stage_v9_aux/stage3/best.pt',
                              'data/urban_sar_floods_aux'),
]

device = torch.device('mps') if torch.backends.mps.is_available() \
         else torch.device('cpu')

def get_event_tag(chip_path: Path) -> str:
    """Extract the primary event name from a chip filename.
    e.g. 20220302_Coraki_Australia_ID_... -> Coraki
         20230805_Hebei_1_ID_...          -> Hebei
    """
    parts = chip_path.stem.split('_')
    return parts[1] if len(parts) > 1 else 'unknown'

for run_name, ckpt_path, data_dir in RUNS:
    print(f'\n{"="*62}')
    print(f'{run_name}')
    print(f'{"="*62}')

    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Load the full test split
    _, test_idx = make_urban_event_split(
        data_dir, TRAIN_EVENTS, TEST_EVENTS)

    full_ds = FastUrbanSARFloods(data_dir, test_idx,
                                  transform=val_transforms(512))

    n_ch  = full_ds.n_channels
    model = build_unet(n_channels=n_ch, n_classes=N_CLASSES,
                       encoder_name='resnet34',
                       encoder_weights=None).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Group test indices by event tag
    event_indices = defaultdict(list)
    for local_i, pair in enumerate(full_ds.pairs):
        tag = get_event_tag(pair[0])
        event_indices[tag].append(local_i)

    print(f'\n{"Event":<16} {"Chips":>6} {"Flood F1":>10} '
          f'{"Precision":>11} {"Recall":>9}')
    print('-' * 57)

    city_f1s = []
    for event in sorted(event_indices):
        idxs   = event_indices[event]
        subset = Subset(full_ds, idxs)
        loader = DataLoader(subset, batch_size=16,
                            shuffle=False, num_workers=0)
        metrics = SegmentationMetrics(n_classes=N_CLASSES,
                                      ignore_index=IGNORE_INDEX)
        with torch.no_grad():
            for batch in loader:
                img  = batch['image'].to(device)
                mask = batch['mask'].to(device)
                metrics.update(model(img).argmax(dim=1), mask)

        r  = metrics.compute()
        cm = r['confusion_matrix']
        tp = cm[1, 1]; fp = cm[0, 1]; fn = cm[1, 0]
        eps = 1e-8
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)

        city_f1s.append(f1)
        tag_label = event[:15]
        print(f'{tag_label:<16} {len(idxs):>6} {f1:>10.4f} '
              f'{prec:>11.4f} {rec:>9.4f}')

    print('-' * 57)
    print(f'{"Mean":<16} {"":>6} {sum(city_f1s)/len(city_f1s):>10.4f}')
    print(f'\nStored best checkpoint F1: {ckpt["metric"]:.4f}')
