# Draft: code-availability request to Willis / Hossain (for Romit to send)

To: arwillis@uncc.edu (Andrew R. Willis; CC Md Sajjad Hossain if an address
can be found — GitHub: eagle034)
Subject: Code availability — "Hardware-Accelerated SAR Simulation with
NVIDIA-RTX Technology" (SPIE 2020)

Dear Prof. Willis,

I'm a graduate student at Northeastern University working on SAR-based urban
flood mapping (cross-city generalization of flood segmentation from
Sentinel-1). Your SPIE 2020 paper with Md Sajjad Hossain and Jamie Godwin,
"Hardware-Accelerated SAR Simulation with NVIDIA-RTX Technology," is directly
relevant to a study I'm running: benchmarking ray-tracing SAR simulators
(RaySAR, a custom student-built tracer, and a phase-history SBR approach) as
sources of synthetic training data for urban flood segmentation, where the
wall–water double-bounce signature is the physical effect of interest.

The paper describes the simulator as open source, but I couldn't locate a
public repository. Is the implementation available somewhere, or could you
share it (or its scene configs) for research use with attribution?

In the meantime I've reimplemented the method from the paper — SBR phase
history with the factored single-precision phase computation of §4.3.4,
validated per §5.3.2 (3-point targets, matched filter + backprojection
reconstructions; positions exact, MF-vs-BP NRMSE 0.1%) and extended to a
C-band urban flood scene where the wall–water dihedral produces a +16 dB
double-bounce line. I'd be glad to share that reproduction, compare against
your original on the point-target benchmarks, and cite the paper as the
method source in any resulting publication.

Thank you for the work — the factored phase trick and the aperture-sampling
design were both load-bearing for getting this right.

Best regards,
Romit Basak
MS student, Northeastern University
basak.r@northeastern.edu

---
(Notes for Romit, not part of the email: send from your NEU address; attach
nothing unless asked — the offer to share is the hook. If Hossain's current
email surfaces via eagle034/LinkedIn, CC him; he was the implementing
student. If they reply with code, it becomes comparator #4 / a validation
cross-check, not a replacement for ours.)
