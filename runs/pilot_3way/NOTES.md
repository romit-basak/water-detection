# Three-way pilot comparison — notes (July 3, 2026)

Identical geometry (sarsim `flood_pilot()`: 5 buildings, ground, ± street
water), C-band, radar from −y at 37.5° incidence, one geometry source
exported to each simulator's native format. Figure: `pilot_3way.png`;
numbers: `metrics_3way.json`. Metric windows follow sarsim Gate 2
(strip y∈[1,2], open water y∈[-12,-8], shadow y∈[11,15] under building
footprints, |x|≤16).

## Headline table

| sim | wall strip | open water | shadow | gain width | time (M4) | signal model |
|---|---|---|---|---|---|---|
| sarsim | **+16.4 dB** | **−8.3 dB** | −2.1 dB | 8.1 m | 203 s† | phase history (SBR) + backprojection |
| Benji | +5.6 dB | **+5.4 dB (wrong sign)** | −0.2 dB‡ | 7.4 m | 4.0 s | stylized intensity, colocated light |
| RaySAR | +5.9 dB* | −11.9 dB | −9.2 dB | 0.2 m | 1.0 s | ray-contribution histogram layers |

† reference (brute-force numpy) backend as shipped in the Gate-2 CLI; the
Mitsuba/Embree backend is ~4× faster on the same M4, CUDA row pending (G1).
\* RaySAR's corner return is a sub-pixel line; band-averaging over the 1 m
strip dilutes it (double-bounce layer energy ratio wet/dry = 6.6×).
‡ not meaningful — see Benji limitations below.

## What distinguishes the model classes (paper material)

1. **Direction of building displacement.** sarsim and RaySAR both lay
   facades over TOWARD the radar (below the corner line) and cast radar
   shadow AWAY (beyond it) — real SAR geometry. Benji's perspective render
   does the optical opposite: buildings lean away from the viewer, and
   "radar shadow" doesn't exist (those pixels are occluded — the camera
   can't see them, so the shadow metric is confounded with roof returns).
2. **Sign of the open-water response.** Real S1 physics: smooth water is
   dark (specular away). sarsim −8.3, RaySAR −11.9 — correct. Benji +5.4 —
   his water *brightens*, because his OBJ material path passes wavelength
   (0.06) where the mixer expects a ratio, making "specular" surfaces 94%
   diffuse under a colocated light.
3. **Double-bounce localization.** RaySAR's phase-center averaging puts the
   return exactly on the geometric corner line (0.2 m — histogram, no PSF).
   sarsim spreads gain over ~8 m: the resolution-limited PSF of the corner
   response plus real wall-multipath in the layover zone (the flooded
   corridor contains the wall's mirror image). Benji's 7.4 m width is not
   localization at all — it's the whole corridor diffusely brightening.
4. **Only sarsim produces a raw signal.** Phase history → arbitrary imaging
   algorithms, speckle from coherent summation, resolution set by physics
   (bandwidth/aperture) rather than raster resolution. RaySAR gives
   geometrically-correct intensity layers; Benji gives a radar-lit picture.

## Engineering-maturity note

- Benji's tracer: 3 upstream bugs found while wiring the pilot (fixed in
  our local clone): colocated-light importance-sampling quad displaced by a
  `viewport_u`+`viewport_u` typo (SAR renders come out ~black);
  `aabb operator*` translates instead of scales (any scene aiming the
  camera via `bounding_box().get_center()` mis-frames, including two of his
  own scenes); wavelength-as-mix-ratio in the OBJ material path (see #2).
- RaySAR: builds natively on arm64 macOS with 4 small patches (committed to
  local branch `macos-arm64-build`); MATLAB layer ported to Python
  (`sarsim/src/sarsim/raysar.py`) — λ parameterized (MATLAB hardcodes
  X-band), true `10*log10` (MATLAB displays `10*ln`).

## Verdict for Thread A

sarsim is the corpus renderer: correct signs on all three phenomena, raw
signal output, and the only one whose 10 m-downsampled statistics can
inherit speckle from coherent physics. RaySAR remains valuable as a fast
geometric cross-check; Benji's tracer as the baseline showing why
"ray tracing" alone is not SAR simulation.
