# Project Timeline — Urban Flood Detection

**Deadline:** April 27, 2025 (Monday)  
**Exam constraint:** April 21 (Tuesday) — other class final exam  
**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done

---

## Week 1 · Apr 7–13 · Data pipeline (full focus)

### Apr 7 (Tue) — today
- [ ] Set up GEE export pipeline (chip size, CRS, drive paths)
- [ ] Verify S1 + S2 co-registration for Houston Harvey 2017

### Apr 8 (Wed)
- [ ] Export Houston SAR–optical pairs from GEE to Drive
- [ ] Confirm coincident acquisition pairs (< 48 hr gap)

### Apr 9 (Thu)
- [ ] Implement NDWI computation + dry-season median baseline subtraction
- [ ] Generate confidence-weighted pseudo-label masks (Houston)

### Apr 10 (Fri)
- [ ] Run pseudo-label pipeline for Valencia Spain 2024
- [ ] Run pseudo-label pipeline for Brisbane 2022
- [ ] QC: visually inspect a sample of generated masks

### Apr 11 (Sat)
- [ ] Compute local variance channel (5×5 filter, VV + VH)
- [ ] Integrate ERA5 72-hr precipitation as optional conditioning channel

### Apr 12 (Sun)
- [ ] Build PyTorch DataLoader (chip loading, normalization, augmentation)
- [ ] Verify input shapes: (B, C, 256, 256) with all channels

### Apr 13 (Mon)
- [ ] Scaffold U-Net architecture in PyTorch
- [ ] Smoke test: forward pass on dummy batch, loss computes correctly
- [ ] **Checkpoint: data pipeline complete, model runnable**

---

## Week 2 · Apr 14–20 · Model training (AM) + exam prep (PM/eve)

> Strategy: launch Colab training runs in the morning so they run while you study.

### Apr 14 (Tue)
- [ ] Launch Sen1Floods11 pretraining run on Colab (A100)
- [ ] Exam prep (evening)

### Apr 15 (Wed)
- [ ] Check pretraining results; adjust LR / batch size if needed
- [ ] Launch fine-tuning on UrbanSARFloods + pseudo-labels
- [ ] Exam prep (evening)

### Apr 16 (Thu)
- [ ] Run Otsu thresholding baseline (replicate Al Mehedi et al. setup)
- [ ] Run Random Forest baseline on raw backscatter
- [ ] Log baseline F1 scores
- [ ] Exam prep (evening)

### Apr 17 (Fri) — exam prep primary
- [ ] Exam prep (primary focus)
- [ ] Check any overnight training runs, relaunch if needed (< 30 min)

### Apr 18 (Sat) — exam prep primary
- [ ] Exam prep (primary focus)

### Apr 19 (Sun) — exam prep primary
- [ ] Exam prep (primary focus)

### Apr 20 (Mon) — final review
- [ ] Final exam review
- [ ] Confirm training runs are saved; nothing blocking Apr 22

---

## Apr 21 (Tue) — EXAM DAY

- [ ] Take exam
- [ ] Rest

---

## Sprint · Apr 22–27 · Evaluation + write-up

### Apr 22 (Wed)
- [ ] Evaluate fine-tuned U-Net on held-out city (UrbanSARFloods split)
- [ ] Compute F1, IoU, precision, recall per class
- [ ] Compare against Otsu + RF baselines

### Apr 23 (Thu)
- [ ] Error analysis: where does the model fail? (shadow confusion? boundary blur?)
- [ ] Ablation: run without local variance channel to quantify its contribution
- [ ] Note results for write-up

### Apr 24 (Fri)
- [ ] Draft report (intro, data, model, results sections)
- [ ] Generate figures: sample predictions, confusion matrix, F1 comparison table

### Apr 25 (Sat)
- [ ] Polish report; fill in discussion + limitations
- [ ] Optional: begin radar shadow channel if energy allows (building footprints + SAR geometry)
- [ ] Update README with final results

### Apr 26 (Sun) — personal soft deadline
- [ ] Final proofread of report and code
- [ ] Clean up repo: remove scratch notebooks, confirm reproducibility
- [ ] **Aim to be fully done today**

### Apr 27 (Mon) — DEADLINE
- [ ] Final submission
- [ ] Buffer for any last-minute issues

---

## Must-haves (for April 27)
- [ ] Pseudo-label pipeline producing ~27 SAR–optical pairs (Houston, Valencia, Brisbane)
- [ ] U-Net trained and evaluated with F1 reported
- [ ] Otsu + RF baselines for comparison
- [ ] Local variance + incidence angle as model input channels
- [ ] Written report submitted

## Stretch goals (post-submission / thesis)
- [ ] Radar shadow channel (building footprints + SAR geometry prior)
- [ ] HAND elevation integration for shadow flood interpolation
- [ ] Cross-city generalization to Kolkata
- [ ] Phase 1b: seasonal baseline expansion (hundreds of training pairs)
- [ ] L-band (ALOS-2) generalization
