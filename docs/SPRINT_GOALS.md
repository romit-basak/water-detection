# Sprint Goals — GPU window July 3–12, 2026

Written Friday evening, July 3. The binding constraint is the **$300 GCP
credit expiring Saturday, July 12** — nine calendar days, of which the
GPU-relevant window is really Jul 4–11 (leave the 12th as slack for
archival/teardown, not compute).

Second-L4 quota request (`GPUS_ALL_REGIONS` 1→2) is filed; GCP says up to
2 business days, so the answer lands **Monday Jul 6 or Tuesday Jul 7**.
The weekend plan is therefore identical in both scenarios; the fork only
opens Monday.

## Saturday July 4 update

Both W2 and much of the deferred W4 stretch goal are further ahead than
this doc's Friday plan assumed — see CLAUDE.md's "In Progress: sarsim"
section for the authoritative current state. Highlights not reflected in
the table below (written Friday, kept for history):

- M1-M3 measurement phase (RaySAR leg, 3-way comparison artifact, sim-vs-real
  gates) all DONE.
- Sprint v4 corpus pipeline (acquisition-pair design, hydrology-lite
  recession, temporal-decorrelation coherence) DONE, D1-D5 complete, all 6
  calibration gates PASS. Full design + a real physics finding (narrow
  dihedral azimuth acceptance) documented in `docs/SYNTHETIC_CORPUS.md`.
- **New quota wall hit**: `CPUS_ALL_REGIONS` capped at 32 vCPUs
  project-wide (separate from the GPU quota) — the originally-planned
  6-VM/3000-seed swarm only fits 2× n2-standard-16 VMs. Increase request
  filed (2 business days, same as the GPU one). Running 2 VMs / 1200 seeds
  now rather than block on the increase.
- **Found + fixed a real swarm-worker bug** during first launch: Debian's
  metadata-script-runner leaves `$HOME` unset, which crashes Dr.Jit's
  `.drjit` cache-dir init immediately; and Debian doesn't ship `libLLVM.so`
  by default (needs `apt-get install libllvm16` + `DRJIT_LIBLLVM_PATH`).
  Fixed in `sarsim/deploy/corpus_startup.sh` (now committed, was
  session-scratchpad-only before). `render_corpus.py` was also still
  calling the removed `render_seed` (pre-pair-corpus API) — fixed to call
  `render_pair`.

## Sunday July 5 early-AM update — G1 CUDA benchmark DONE (the hard way)

- **W0 sweep COMPLETE (20/20)**: final AGGREGATE confirms the ceiling —
  under leakage-free `val_selected`, all four configs are within noise
  (0.087–0.101, largest pairwise p=0.287). Leak inflation is systematically
  larger for the fw3 configs (+0.054/+0.058) than fw10 (+0.010/+0.015).
- **G1 `mi_cuda` on the L4**: pilot 0.72 Mray/s, city320 0.70 Mray/s —
  vs M4 CPU (mi_llvm) 1.95/1.62 and the paper's RTX-2070 1.53–4.26.
  Finding: at phase-history pulse granularity (147–262k rays/pulse,
  sync per pulse, per-kernel OptiX JIT), the workload is dispatch-bound —
  RT cores never see enough in-flight work; a laptop-class CPU via LLVM
  outruns a cloud L4. Corpus-render confirms: L4 mi_cuda ≈ 118–133 s/seed,
  parity with one n2-standard-16 LLVM worker.
- **Getting OptiX to run on GCP at all is a reproducibility finding**:
  GCP's DLVM datacenter driver (580.159.03) omits the entire graphics
  userspace — no `libnvoptix`, no `libnvidia-rtcore`, no `nvoptix.bin`,
  no `libnvidia-gpucomp`. Fix: pull the *exact-version* public datacenter
  `.run`, extract (`sh nv.run -x`), and no-clobber-copy ALL
  `libnvidia-*.so.*` + `libnvoptix.so.*` into `/usr/lib/x86_64-linux-gnu/`
  + `nvoptix.bin` → `/usr/share/nvidia/` + `ldconfig`. Copying only the
  documented trio still fails (OPTIX_ERROR_INTERNAL_COMPILER_ERROR from
  the missing GPU-compiler lib). Also hit: G2 zonal stockout blocks
  restarting an *existing stopped* VM (~5.5 h wait, 10-min retry loop).
- **L4 joined the corpus swarm** as a 5th worker (mi_cuda backend,
  `--rotate` shard 3/8 from seed 900, skip-if-exists) for the tail;
  corpus 2150/2400 as of 04:30 UTC, ETA ~06:00 UTC. Backend-numerics
  parity mi_llvm↔mi_cuda was validated at 0.999998 in the M-phase; the
  L4's log records which seeds it rendered.

---

## State as of Friday evening

| Item | Status |
|---|---|
| W0 multi-seed sweep (4 configs × 5 seeds) | RUNNING on `water-detection-l4`, 8/20 done, ETA ~Sat 5–8 AM EDT; VM self-powers-off, results land in `gs://urban-water-detection-sweep/runs/` |
| W2 sarsim (phase-history simulator) | DONE locally — Gates 0/1/2 pass, Mitsuba backend parity 0.999998, city generator + OBJ export (commits `35c26b3`..`e3df52e`) |
| W2 CUDA benchmark (`mi_cuda` on L4) | PENDING — blocked until sweep frees the GPU (Sat AM) |
| W3 Benji tracer on the pilot scene | PENDING — builds/runs on M4; needs a scene function for our OBJ (CPU-only, no GPU needed) |
| W3 RaySAR | PENDING — clone, POV-Ray build (fallback: $0.10/hr Linux CPU VM), Python port of the MATLAB analysis layer |
| W4 Thread A corpus + pretrain | STRETCH — generator exists; render → downsample → chips → Stage-0 pretrain not started |
| W5 author email (Willis/Hossain) | DRAFTED (`sarsim/AUTHORS_EMAIL_DRAFT.md`) — **Romit to send** |

Spend so far: roughly $30–40 of $300 (T4 experiments + the L4 sweep).
Money is not the constraint; days are.

---

## Deliverables, ranked

1. **Sweep aggregate table** (Sat AM): per-config mean ± across-seed SD
   under all three protocols (`val_selected` / `peak_over_test` /
   `final5_mean`), the leak-inflation delta, pairwise significance.
   This either hardens or revises the headline ceiling claim — every
   downstream decision (more labeling vs synthetic data vs MIKE+) cites
   this table.
2. **Three-way comparison artifact** (Track B paper pilot result): one
   figure + table — sarsim vs Benji's tracer vs RaySAR on the *identical*
   pilot scene ± water. Qualitative checklist (double-bounce line, shadow,
   layover) + runtime + signal-model class.
3. **CUDA benchmark**: sarsim `mi_cuda` rays/s on the L4 vs the paper's
   1.5–4.3 Mray/s (RTX 2070) and vs our M4 LLVM 6 Mray/s. One table row,
   but it's the "RTX-accelerated" claim of the reimplementation.
4. **Thread A stretch**: synthetic city corpus (N seeds × 2 morphologies ×
   ±water), downsample to 10 m, chip, Stage-0 pretrain via
   `train_with_pseudo.py`, zero-shot + finetune eval on UrbanSARFloods.
   Explicitly the stretch, not the promise — but it's the item that
   actually *needs* sustained GPU time, so it's what the second GPU is for.

---

## Common track: the weekend (identical in both scenarios)

**Saturday Jul 4**
- AM: sweep completes → pull `AGGREGATE.txt` + summaries, write the
  results table into CLAUDE.md (and a paper-notes file). Confirm VM
  powered itself off.
- `mi_cuda` benchmark: restart the L4 briefly (~1 hr, ~$1.40), run the
  Gate-1/Gate-2 scenes + a city scene on the CUDA backend, log rays/s and
  wall-time vs M4. Stop the VM after.
- Start W3 locally: scene function in Benji's tracer loading our pilot
  OBJ (C-band per its `SPECTRAL_MAP`, camera at 37.5° incidence to match
  sarsim's Gate 2), render ± water, save PPMs.

**Sunday Jul 5**
- RaySAR: clone, attempt the POV-Ray build on the M4; if arm64 fights for
  more than ~2 hrs, spin the cheap Linux CPU VM instead of debugging.
  Port the MATLAB signal-binning/analysis step to Python. Convert the
  pilot geometry to POV-Ray format, render ± water.
- If RaySAR goes smoothly: first draft of the 3-way comparison figure.

**Weekend GPU usage is minimal by design** (one ~1 hr benchmark session).
The L4 sits stopped — stopped VMs cost only disk pennies — so nothing is
wasted while the CPU-side comparators get built.

---

## Fork A — quota DENIED or still pending (single L4)

Everything serializes onto one box; the L4 time-shares between Track B
odds-and-ends and Thread A. Order matters: corpus render before pretrain,
short jobs before long ones.

- **Mon Jul 6**: finish the 3-way artifact (figure + table + writeup
  notes). Meanwhile render a *small* pilot corpus locally on the M4
  (LLVM backend, ~21 s/city — a few hundred scenes overnight is feasible
  locally) to debug the downsample-to-10m + chipping pipeline end-to-end
  before any GPU time is spent on it.
- **Tue Jul 7**: start the L4. Full corpus render on CUDA (thousands of
  scenes; benchmark will tell us the per-scene rate — if M4 does 21 s,
  L4 should be well under 10 s). Same box, same day if render finishes:
  kick off Stage-0 pretrain on the synthetic chips.
- **Wed–Thu Jul 8–9**: pretrain runs (Stage 0 ~20 epochs) → zero-shot
  eval → finetune on UrbanSARFloods with the same seed protocol as the
  sweep (at minimum seeds 0–2 to get an honest SD). This is the direct
  successor experiment to failure mode 4.
- **Fri Jul 10–Sat Jul 11**: buffer for reruns/failures; aggregate the
  synthetic-pretrain results; final CLAUDE.md + paper-notes update.
- **Sun Jul 12**: no compute. Archive + teardown only (checklist below).

Cost: ~$40–70 more. Ends comfortably under budget.

Single-GPU rule of thumb: **the GPU should either be running a job or be
stopped** — the sweep's self-poweroff pattern gets reused for every
multi-hour job (render, pretrain) so nothing idles overnight.

## Fork B — quota GRANTED Mon/Tue (two L4s in parallel)

Split by workload class, not by paper: box 1 stays the short-job /
interactive box, box 2 becomes the long-job box.

- **Immediately on approval**: create `water-detection-l4b` (same
  g2-standard-12 recipe, us-west1-a or wherever stock exists — reuse the
  zone-fallback script; stockouts hit G2 before, assume they can again).
- **Box 2 (long jobs)**: full corpus render → Stage-0 pretrain → the
  seeds-0–2 finetune matrix, chained with the self-poweroff pattern.
  Everything Fork A schedules for Tue–Thu, started 1–2 days earlier and
  without competing for the GPU.
- **Box 1 (short/interactive)**: 3-way artifact reruns, any sweep
  follow-ups the Saturday table motivates (e.g., if the leak-inflation
  delta is large, a 5-seed rerun of one config with a different val_frac
  as a robustness check), RaySAR CPU work if the Mac build failed,
  sarsim CUDA profiling. Started/stopped per task.
- **Net effect**: Thread A's stretch goal stops being a stretch — the
  extra 1–2 days of parallel GPU time is exactly the slack the pretrain
  matrix needs. If the quota lands Tuesday, Fork B still beats Fork A by
  a day; if it lands Wednesday or later, ignore it and stay on Fork A
  (spinning up a second box for <2 days of remaining window isn't worth
  the coordination overhead).

Cost: ~$70–120 more. Still under half the credit.

---

## GPU-independent work (fill-in at any point, M4 only)

- Send the author email (Romit — draft is ready).
- Loop Prof. Maxwell back in on the two-thread direction (CLAUDE.md
  flags this as not yet done).
- Paper-notes skeleton for Track B: methods section practically writes
  itself from the gate reports (Gate 0/1/2 numbers, the MC-jitter and
  layover findings are Results-section material already).
- Downsample-methodology literature check (coherent vs incoherent
  aggregation to 10 m) — flagged in CLAUDE.md as unvalidated and it
  gates the credibility of Thread A's step 4.

## Before the credit dies — archival checklist (do by Sat Jul 11)

1. `gs://urban-water-detection-sweep/runs/` → pull to local repo `runs/`
   (small) and commit the summaries/aggregate.
2. Full bucket (~51 GB: preprocessed chips, pseudo chips, code tarball)
   → copy to Google Drive first (GCP integrates more smoothly with GDrive
   than OneDrive — `rclone`/Drive API from a VM), then a second hop
   GDrive → Northeastern OneDrive (5 TB, tied to Romit's MS enrollment
   rather than a hard September cutoff — more durable, transfer mechanism
   TBD). Or accept ~$1/month and keep the bucket. Decide, don't drift.
3. Any synthetic corpus generated on GCP → Drive (could be 50–100 GB),
   same two-hop path as above.
4. Delete all VMs (not just stop — boot disks bill after credits).
5. Note in CLAUDE.md what lives where post-sprint.

## Decision points

| When | Decision | Input |
|---|---|---|
| Sat AM | Does the ceiling claim survive honest selection? | `AGGREGATE.txt` — if configs separate under `val_selected`, the paper story changes |
| Sun PM | RaySAR: local build vs CPU VM | 2-hr timebox on the arm64 build |
| Mon/Tue | Fork A vs Fork B | Quota email; if granted Wed+, stay Fork A |
| Tue | Corpus size for Thread A | CUDA per-scene render time × remaining days |
| Fri Jul 10 | Cut scope? | If pretrain matrix isn't launched by Thu night, ship 3-way artifact + single-seed pretrain result and defer the matrix |
