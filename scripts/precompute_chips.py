"""
scripts/precompute_chips.py

Precomputes local variance channels and saves 4-channel chips to disk.
Run once before training — dramatically reduces per-epoch load time.

Output structure mirrors the input:
    data/urban_sar_floods_preprocessed/
        01_NF/SAR/*.npy   (4-channel: VV, VH, var_VV, var_VH)
        01_NF/GT/*.npy
        02_FO/SAR/*.npy
        ...
    data/sen1floods11_preprocessed/
        S1Hand/*.npy      (4-channel: VV, VH, var_VV, var_VH)
        LabelHand/*.npy

Saves as float16 for SAR (halves disk usage with negligible precision loss)
and uint8 for labels.

Usage:
    uv run python scripts/precompute_chips.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from rasterio.windows import from_bounds
import torch
import torch.nn.functional as F
from tqdm import tqdm

KERNEL = 5
SAR_DB_MIN = -60.0
SAR_DB_MAX  =  7.0
LINEAR_CLIP_MIN = 1e-6
ANGLE_DIR = Path('data/aux_features')  # {Event}_s1_incidence_angle.tif


def local_variance(data: np.ndarray, kernel: int = 5) -> np.ndarray:
    t = torch.from_numpy(data).unsqueeze(0)
    pad = kernel // 2
    t_pad = F.pad(t, (pad, pad, pad, pad), mode='reflect')
    mean    = F.avg_pool2d(t_pad, kernel, stride=1, padding=0)
    mean_sq = F.avg_pool2d(t_pad ** 2, kernel, stride=1, padding=0)
    return (mean_sq - mean ** 2).clamp(min=0).squeeze(0).numpy()


def process_urban_sar(src_root: str, dst_root: str):
    src = Path(src_root)
    dst = Path(dst_root)

    for folder in ['01_NF', '02_FO', '03_FU']:
        sar_src = src / folder / 'SAR'
        gt_src  = src / folder / 'GT'
        sar_dst = dst / folder / 'SAR'
        gt_dst  = dst / folder / 'GT'

        if not sar_src.exists():
            continue

        sar_dst.mkdir(parents=True, exist_ok=True)
        gt_dst.mkdir(parents=True, exist_ok=True)

        chips = sorted(sar_src.glob('*_SAR.tif'))
        print(f'{folder}: {len(chips)} chips')

        for chip_path in tqdm(chips, desc=folder):
            out_sar = sar_dst / chip_path.name.replace('.tif', '.npy')
            out_gt  = gt_dst  / chip_path.name.replace('_SAR.tif', '_GT.npy')

            if out_sar.exists() and out_gt.exists():
                continue  # skip already processed

            # SAR: read bands 1+2 (pre-normalized [0,1])
            with rasterio.open(chip_path) as src_f:
                raw = src_f.read([1, 2]).astype(np.float32)
            raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
            raw = np.clip(raw, 0.0, 1.0)

            # Append variance channels
            var  = local_variance(raw, kernel=KERNEL)
            data = np.concatenate([raw, var], axis=0)  # (4, H, W)

            # Save as float16 to halve disk usage
            np.save(out_sar, data.astype(np.float16))

            # Label
            gt_path = gt_src / chip_path.name.replace('_SAR.tif', '_GT.tif')
            with rasterio.open(gt_path) as src_f:
                label = src_f.read(1).astype(np.uint8)
            np.save(out_gt, label)

    print(f'UrbanSARFloods preprocessed → {dst}')


def process_sen1floods11(src_root: str, dst_root: str):
    src = Path(src_root)
    dst = Path(dst_root)

    s1_src    = src / 'S1Hand'
    lbl_src   = src / 'LabelHand'
    s1_dst    = dst / 'S1Hand'
    lbl_dst   = dst / 'LabelHand'

    s1_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    chips = sorted(s1_src.glob('*_S1Hand.tif'))
    print(f'Sen1Floods11: {len(chips)} chips')

    for chip_path in tqdm(chips, desc='S1Hand'):
        out_s1  = s1_dst  / chip_path.name.replace('.tif', '.npy')
        lbl_path = lbl_src / chip_path.name.replace('_S1Hand.tif', '_LabelHand.tif')
        out_lbl  = lbl_dst / chip_path.name.replace('_S1Hand.tif', '_LabelHand.npy')

        if out_s1.exists() and out_lbl.exists():
            continue

        # SAR: linear scale → dB → [0,1]
        with rasterio.open(chip_path) as src_f:
            raw = src_f.read([1, 2]).astype(np.float32)
        raw = np.where(np.isfinite(raw) & (raw > 0), raw, LINEAR_CLIP_MIN)
        raw = np.clip(raw, LINEAR_CLIP_MIN, 1.0)
        raw = 10.0 * np.log10(raw)
        raw = np.clip(raw, SAR_DB_MIN, SAR_DB_MAX)
        raw = (raw - SAR_DB_MIN) / (SAR_DB_MAX - SAR_DB_MIN)

        # Incidence angle: sample from GEE-exported raster at chip bounds
        event_name = chip_path.name.split('_')[0]  # e.g. 'Bolivia'
        angle_path = ANGLE_DIR / f'{event_name}_s1_incidence_angle.tif'
        ia_data = None
        if angle_path.exists():
            try:
                with rasterio.open(angle_path) as a_src:
                    with rasterio.open(chip_path) as c_src:
                        cb = c_src.bounds
                    rb = a_src.bounds
                    if not (cb.left > rb.right or cb.right < rb.left or
                            cb.bottom > rb.top or cb.top < rb.bottom):
                        win = from_bounds(
                            max(cb.left, rb.left), max(cb.bottom, rb.bottom),
                            min(cb.right, rb.right), min(cb.top, rb.top),
                            a_src.transform
                        )
                        if win.width >= 1 and win.height >= 1:
                            ia_data = a_src.read(1, window=win,
                                out_shape=raw.shape[1:],
                                resampling=rasterio.enums.Resampling.bilinear
                            ).astype(np.float32)
            except Exception:
                pass
        if ia_data is None:
            ia_data = np.full(raw.shape[1:], 0.5, dtype=np.float32)

        # Stack: SAR (2) + variance (2) + incidence_angle (1) = 5 channels
        var  = local_variance(raw, kernel=KERNEL)
        data = np.concatenate([raw, var, ia_data[np.newaxis]], axis=0)  # (5, H, W)
        np.save(out_s1, data.astype(np.float16))

        # Label: remap -1→0, 0→3, 1→1
        with rasterio.open(lbl_path) as src_f:
            lbl_raw = src_f.read(1).astype(np.int64)
        label = np.where(lbl_raw == 1, 1, np.where(lbl_raw == 0, 3, 0)).astype(np.uint8)
        np.save(out_lbl, label)

    print(f'Sen1Floods11 preprocessed → {dst}')


if __name__ == '__main__':
    print('Preprocessing UrbanSARFloods...')
    process_urban_sar(
        src_root='data/urban_sar_floods',
        dst_root='data/urban_sar_floods_preprocessed',
    )

    print('\nPreprocessing Sen1Floods11...')
    process_sen1floods11(
        src_root='data/sen1floods11',
        dst_root='data/sen1floods11_preprocessed',
    )

    print('\nDone. Update cfg paths in train_three_stage.py:')
    print('  sen1floods_dir  = "data/sen1floods11_preprocessed"')
    print('  urban_sar_dir   = "data/urban_sar_floods_preprocessed"')
