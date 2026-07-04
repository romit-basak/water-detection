# M3: sim-vs-real contrast statistics — corpus acceptance targets

July 3, 2026. Real side: raw UrbanSARFloods dB bands (5=VH_pre, 6=VV_pre,
7=VH_post, 8=VV_post — identified empirically via open-flood darkening),
urban test events Hebei + Sydney. Classes: GT 2=urban flood, 1=open flood,
non-flood from the same city's 01_NF chips (FU GT==0 is unlabeled, never
used as a class). Numbers: `targets.json`; figure: `m3_sim_vs_real.png`.

## Real Sentinel-1 targets (median [IQR])

| city | class | n px | VV_post (dB) | ΔVV = post−pre (dB) |
|---|---|---|---|---|
| Hebei | urban flood | 162k | −4.1 | **+1.6 [−1.3, +4.6]** |
| Hebei | open flood | 1.15M | −15.7 | −8.4 [−11.2, −5.6] |
| Hebei | non-flood | 491k | −7.2 | +0.4 [−1.9, +2.7] |
| Sydney | urban flood | 2.3k | −6.2 | **+0.3 [−3.2, +3.3]** |
| Sydney | open flood | 344k | −18.1 | −7.2 [−10.0, −4.4] |
| Sydney | non-flood | 664k | −8.2 | +0.7 [−1.8, +3.2] |

Two findings stand alone:
1. **The urban-flood signature at 10 m is +1.6 dB, not +16 dB.** The
   sub-pixel wall–water dihedral physics (+16.4 dB in the high-res pilot)
   survives aggregation to a modest median shift whose IQR spans zero.
   This is the resolution-mismatch thesis with a measured number on it.
2. **Sydney urban flood is statistically invisible per-pixel** (ΔVV +0.3
   vs non-flood +0.7). Whatever signal exists is spatial/contextual, not
   radiometric — consistent with Sydney being a hardest city in the main
   results table.

## GATE RESULT: naive pilot FAILS (by design — that's the gate working)

sarsim pilot multilooked to 10 m: flooded-street pixels show
**ΔdB = +16.6** (4 px, median) vs the Hebei target +1.6 [−1.3, +4.6] —
an order of magnitude too strong. A corpus rendered with pilot-scene
settings would teach "flooded street = beacon". Root causes, in order:

1. **Wall-to-wall dihedral**: 25 m of unbroken smooth wall facing the
   radar broadside + pure smooth water — a perfect corner-reflector array.
   Real streets: broken facades, orientation diversity (most walls are NOT
   normal to the look), clutter. → A1 Tier-0 azimuth randomization +
   Tier-1 facade relief/clutter.
2. **Dry reference too dark**: dry street = rough soil (rho_s 0.05) kills
   the dry-scene dihedral; real asphalt/pavement is smooth enough that dry
   urban ALREADY has double-bounce, so real flooding adds little. → A1
   asphalt streets (the dark-confuser item is also the delta-shrinker).
3. **Pure mixtures**: 16 m pure-water corridor vs real mixed pixels.
   → larger scenes, structured flood masks, narrower streets.

## Acceptance bands for A1 calibration scenes (10 m, per-pixel)

- flooded-street ΔdB: median in **[0, +5]**, distribution overlapping
  the Hebei urban-flood IQR; NOT >+8.
- open-water ΔdB: median in **[−10, −6]** (pilot already correct: −8.3).
- non-flood ΔdB: |median| ≤ 1, IQR within ±3 (requires speckle/jitter
  realism so the dry pair isn't bit-identical).
- VV-class ordering: open flood < non-flood < urban flood in VV_post,
  with overlaps comparable to the histograms (no clean separation —
  a synthetically separable corpus is a wrong corpus).
