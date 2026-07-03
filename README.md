# Cross-City Urban Flood Detection from SAR

**CS5330 Computer Vision — Spring 2026, Northeastern University**  
Romit Basak · Advisor: Prof. Bruce Maxwell

---

## Overview

This project investigates cross-city generalization for urban flood detection from Sentinel-1 SAR imagery. The core challenge: a model trained on flooded Houston cannot reliably detect flooding in Hebei or Sydney, because urban SAR backscatter signatures are city-specific (double-bounce geometry, building morphology, street orientation all vary). We evaluate whether two strategies can overcome this:

1. **Geographic context features** — WorldCover built-up fraction, Google Open Buildings density, HAND elevation, and Sentinel-1 incidence angle as additional model inputs
2. **AWEI_sh pseudo-label pretraining** — Confidence-weighted labels derived from Sentinel-2 optical imagery for 11 new flood events, used to pretrain before fine-tuning on UrbanSARFloods

**Key finding**: All experimental conditions converge to a data-volume ceiling of flood F1 ≈ 0.10–0.15 (macro F1 ≈ 0.49–0.55) regardless of feature set, loss tuning, or pseudo-labeling. 7 labeled training cities is not enough.

---

## Results Summary

| Model | Channels | Peak Flood F1 | Macro F1 |
|---|---|---|---|
| SAR-only (fw=10) | 4 | 0.120 | ~0.48 |
| Aux features (fw=10) | 8 | 0.102 | ~0.49 |
| Aux features (fw=3) | 8 | **0.150** | ~0.52 |
| Finetune-only, fw=3 | 8 | 0.143 | 0.551 |
| Pseudo+Finetune, fw=3 | 8 | 0.136 | 0.550 |

Evaluation: strict event-held-out split — no chip from any test city appears during training.  
Degenerate (all-non-flood) baseline: macro F1 = 0.33, flood F1 = 0.

---

## Architecture

- **Model**: ResNet-34 U-Net (`segmentation-models-pytorch`), ImageNet init
- **Input**: 8-channel — VV, VH, var(VV), var(VH), building density, WorldCover built-up fraction, HAND elevation, incidence angle
- **Loss**: Focal loss (γ=2), flood weight tuned (fw=3 best)
- **Training**: AdamW, discriminative LR (encoder 2e-5, decoder 1e-4), cosine decay, early stopping

---

## Dataset

**UrbanSARFloods** — 8,879 chips, 18 urban flood events  
Event-level split (zero city overlap):

| Split | Events |
|---|---|
| Train (7) | Houston, Beira, Japan, Canada, Iran, Lumberton, Somalia |
| Test (7) | Hagibis, Sydney, Coraki, Niger, Hebei, Beledweyne, PortMacquarie |

**Pseudo-labeled events** (disjoint from above) — 11 new events, 2,073 chips:  
Brisbane 2022, Sydney 2022, Valencia 2024, Liège 2021 (×2), Cologne 2021, Zhengzhou 2021, Derna 2023 (×2), Emilia-Romagna 2023, Rio Grande do Sul 2024

Label scheme: flood=1, non-flood=0, ignore=−1. Pseudo-labels derived from AWEI_sh confidence maps (Sentinel-2 SR, shadow-masked, SCL cloud-masked), with confidence-weighted focal loss.

---

## Repository Structure

```
water-detection/
├── src/
│   ├── data/
│   │   ├── fast_dataset.py          # UrbanSARFloods dataset (binary labels)
│   │   ├── pseudo_chip_dataset.py   # New-event pseudo-labeled chips
│   │   └── transforms.py            # Albumentations augmentation pipeline
│   ├── models/
│   │   └── unet.py                  # ResNet-34 U-Net (multi-channel input)
│   └── utils/
│       └── metrics.py               # Flood F1, macro F1, confusion matrix
├── scripts/
│   ├── export_sar_pseudo.js         # GEE: export S1 SAR for new events
│   ├── export_awei_confidence_pseudo.js  # GEE: AWEI_sh confidence maps
│   ├── export_aux_pseudo.js         # GEE: buildings, WorldCover, HAND, angle
│   ├── precompute_chips.py          # Chip UrbanSARFloods into .npy
│   ├── precompute_aux_features.py   # Bake 8-channel aux chips
│   ├── precompute_pseudo_chips.py   # Chip new-event SAR + confidence maps
│   ├── train_three_stage.py         # Main training: SAR-only and aux models
│   ├── train_with_pseudo.py         # Pseudo-pretrain + UrbanSARFloods finetune
│   └── per_city_eval.py             # Per-city F1 / precision / recall breakdown
├── report/
│   └── report.tex                   # IEEE two-column final report
├── data/
│   ├── urban_sar_floods/            # Raw UrbanSARFloods chips (not in repo)
│   ├── urban_sar_floods_aux/        # 8-channel preprocessed chips (not in repo)
│   ├── pseudo_labels/               # GEE exports: SAR + confidence + aux
│   ├── pseudo_chips/                # Chipped pseudo-labeled .npy files
│   └── discovery_summary.csv        # SAR/optical pair discovery results
└── runs/                            # Training logs and checkpoints
```

---

## Reproducing the Pipeline

### 1. Prerequisites

```bash
# Install dependencies (uses uv)
uv sync

# Required: Python 3.12, PyTorch with MPS (Apple Silicon) or CUDA
# Google Earth Engine account for data export scripts
```

### 2. Precompute UrbanSARFloods chips

```bash
# 4-channel SAR chips
uv run python scripts/precompute_chips.py

# 8-channel aux chips (requires GEE exports in data/aux_features/ and data/building_density/)
uv run python scripts/precompute_aux_features.py
```

### 3. Train (SAR-only or aux features)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run python scripts/train_three_stage.py \
  --urban_sar_dir data/urban_sar_floods_aux \
  --out_dir runs/my_run \
  --flood_weight 3.0
```

### 4. Pseudo-label pipeline (optional)

```bash
# a. Export from GEE (run in GEE code editor):
#    scripts/export_sar_pseudo.js         → data/pseudo_labels/sar/
#    scripts/export_awei_confidence_pseudo.js → data/pseudo_labels/confidence_maps/
#    scripts/export_aux_pseudo.js         → data/pseudo_labels/aux/

# b. Chip the new events
uv run python scripts/precompute_pseudo_chips.py --dry_run  # check first
uv run python scripts/precompute_pseudo_chips.py

# c. Train with pseudo-pretraining
PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run python scripts/train_with_pseudo.py \
  --out_dir runs/pseudo_v1
```

### 5. Per-city evaluation

```bash
uv run python scripts/per_city_eval.py \
  --ckpt runs/my_run/stage3/best.pt \
  --urban_sar_dir data/urban_sar_floods_aux
```

---

## Failure Modes Documented

Four implementation pitfalls are documented with fixes in the report:

1. **Degenerate multi-class labeling** — 4-class scheme reported macro F1=0.39; true flood F1 was 0.116. Fix: binary labels.
2. **Collapse-to-non-flood** — cross-entropy and Dice loss both collapsed by epoch 3. Fix: focal loss.
3. **Sen1Floods11 pretraining collapse** — F1 dropped 0.289→0.063 after 2 epochs. Fix: ImageNet init directly.
4. **Pseudo-label pretraining collapse** — zero-shot F1 peaked at 0.124 then collapsed to 0.001 in 9 epochs due to dataset size, label sparsity in dense urban scenes, and domain mismatch. Fix: SAR change detection (future work).

---

## References

- Zhao et al. (2024) — UrbanSARFloods — CVPR Workshop on Earth Vision
- Bonafilia et al. (2020) — Sen1Floods11 — CVPR Workshops
- Al Mehedi et al. (2025) — Urban/peri-urban SAR flood comparison — PLOS Water
- Lin et al. (2017) — Focal Loss — ICCV
- Feyisa et al. (2014) — AWEI_sh — Remote Sensing of Environment
- Yadav et al. (2024) — Kuro Siwo — NeurIPS Datasets and Benchmarks

---

## Future Work

- SAR change detection pseudo-labels (avoids optical cloud/shadow limitations)
- Geospatial foundation model encoders (Prithvi-EO-2.0)
- HAND-based flood shadow interpolation
- Scale to 20+ labeled training cities for Kolkata monsoon deployment
