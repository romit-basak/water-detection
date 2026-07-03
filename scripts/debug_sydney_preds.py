"""Check Sydney 2022 label distribution and model predictions."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import numpy as np
import torch
from src.data.fast_dataset import FastUrbanSARFloods
from src.models.unet import build_unet

root = Path('data/urban_sar_floods_preprocessed')
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# Load model
ckpt  = torch.load('runs/three_stage_v7/stage3/best.pt', map_location='cpu')
model = build_unet(4, 4, 'resnet34', None).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

# Find Sydney 2022 chips
all_pairs = []
for folder in ['01_NF', '02_FO', '03_FU']:
    sar_dir = root / folder / 'SAR'
    gt_dir  = root / folder / 'GT'
    if not sar_dir.exists(): continue
    for p in sorted(sar_dir.glob('*20220705*Sydney*.npy')):
        gt = gt_dir / p.name.replace('_SAR.npy', '_GT.npy')
        if gt.exists():
            all_pairs.append((folder, p, gt))

print(f'Sydney 2022 chips: {len(all_pairs)}')
print()

for folder, sar_p, gt_p in all_pairs[:10]:
    image  = torch.from_numpy(np.load(sar_p).astype(np.float32)).unsqueeze(0).to(device)
    target = np.load(gt_p).astype(np.int64)
    with torch.no_grad():
        pred = model(image).argmax(dim=1)[0].cpu().numpy()

    unique_t, counts_t = np.unique(target, return_counts=True)
    unique_p, counts_p = np.unique(pred,   return_counts=True)
    t_dist = dict(zip(unique_t.tolist(), counts_t.tolist()))
    p_dist = dict(zip(unique_p.tolist(), counts_p.tolist()))
    total  = target.size

    print(f'{folder} | {sar_p.name[:40]}')
    print(f'  target: { {k: f"{v/total:.1%}" for k,v in t_dist.items()} }')
    print(f'  pred:   { {k: f"{v/total:.1%}" for k,v in p_dist.items()} }')
    print()
