# Matched short-BPTT screen: completed result review

All eight arms completed300 updates in 730.797 seconds (12.18 minutes).
**All four full-BPTT controls failed the frozen far-paired qualification.**
The formal comparison therefore remains BASELINE_UNQUALIFIED for both
architectures. This does not erase a positive short-window case, and it
does not establish either general truncation failure or superiority.

## Primary size32/T64 endpoint

Percentages. Far means graph distance>16; the statistic averages the
16 per-map paired fractions. BOTH fresh source alternatives must be correct.

| Architecture | Seed | K64 BA | K8 BA | K64 far paired | K8 far paired | Effect pp | K8 reach / hold predicates |
|---|---:|---:|---:|---:|---:|---:|---|
| ws_additive | 0 | 87.20 | 91.22 | 15.83 | 23.18 | +7.36 | False / False |
| ws_additive | 1 | 87.73 | 98.03 | 0.00 | 80.87 | +80.87 | True / True |
| ws_revision | 0 | 78.04 | 83.90 | 21.44 | 0.67 | -20.77 | False / False |
| ws_revision | 1 | 86.16 | 64.38 | 0.00 | 17.81 | +17.81 | False / False |

Full controls require BA>=85% and far paired>=80%; none met the latter.
The additive K8 seed1 satisfies its reach and hold predicates, but the
predeclared architecture screen also requires qualified full controls and
both seeds to pass. Additive seed0 and both revision short arms fail.

## Positive case and its limits

Additive K8 seed1, size32: BA98.03% and far paired80.87% atT64.
Far paired remains79.20% atT128 and78.24% atT256; BA256 is96.56%.
This is a descriptive example of source-dependent computation beyond the
K8 single-window dependency radius, under this trained shared rule.

However,80.87% is a per-map average. The corresponding pooled-pixel fraction
is59.53%; maps have between2 and345 far pixels. The frozen averaging stays
unchanged, and both views must remain visible. Long-distance performance
is much weaker than the aggregate headline:

| Size | T | Distance bin | Paired accuracy % | Eligible maps | Pixels |
|---|---:|---|---:|---:|---:|
| 32 | 64 | 16_32 | 90.47 | 16 | 1291 |
| 32 | 64 | 32_64 | 45.60 | 9 | 481 |
| 32 | 64 | 64_100000 | 0.00 | 1 | 65 |
| 32 | 256 | 16_32 | 86.15 | 16 | 1291 |
| 32 | 256 | 32_64 | 43.91 | 9 | 481 |
| 32 | 256 | 64_100000 | 0.00 | 1 | 65 |
| 64 | 64 | 16_32 | 48.92 | 16 | 5085 |
| 64 | 64 | 32_64 | 15.94 | 16 | 9511 |
| 64 | 64 | 64_100000 | 8.68 | 9 | 4333 |
| 64 | 256 | 16_32 | 61.54 | 16 | 5085 |
| 64 | 256 | 32_64 | 36.41 | 16 | 9511 |
| 64 | 256 | 64_100000 | 6.11 | 9 | 4333 |

The[16,32) bin includes distance16, whereas the primary far metric uses
strictly>16. The size32 distance>=64 bin has only one eligible map; it
cannot support a population conclusion. At size64, this same checkpoint's
far paired is28.40% atT64 and41.51% atT256. Robust distance/scale extrapolation
is not established. Two communication phases per step mean K8 reaches at
most16 graph edges per gradient window; distance is not the window count.

## Optimization and systems

| Architecture | Seed | K | Training seconds | Peak allocated MiB | Gradient clipping % | Last logged loss |
|---|---:|---:|---:|---:|---:|---:|
| ws_additive | 0 | 64 | 95.98 | 514.04 | 88.67 | 0.3041 |
| ws_additive | 1 | 64 | 86.77 | 514.04 | 90.67 | 0.4862 |
| ws_additive | 0 | 8 | 86.59 | 90.49 | 6.67 | 0.2651 |
| ws_additive | 1 | 8 | 86.09 | 90.49 | 6.67 | 0.1568 |
| ws_revision | 0 | 64 | 90.19 | 514.04 | 58.67 | 0.6030 |
| ws_revision | 1 | 64 | 88.36 | 514.04 | 63.67 | 0.7959 |
| ws_revision | 0 | 8 | 93.28 | 90.49 | 40.67 | 0.3290 |
| ws_revision | 1 | 8 | 86.34 | 90.49 | 53.33 | 0.4279 |

Peak allocated CUDA memory falls from514.04 to90.49 MiB
(82.4% lower). These peaks include data/optimizer/gradient
storage, not just activations. Training durations are broadly similar;
the screen does not establish a meaningful speedup or accuracy-matched
systems advantage. Timings are sequential observations on one GPU.

Additive fullK64 clips88.67%/90.67% of updates, versus6.67%/6.67% forK8.
This is a useful optimization observation, not proof that clipping caused
the full-control failure. Detaching changes the accumulated gradient and
therefore how the identical clipping rule acts. ShortK8 revision BA falls
from83.90/64.38% atT64 to59.64/46.18% atT256 across seeds0/1.

## Validity and claim boundary

- All sources match their executed snapshots. Initial and final parameter
  hashes, data banks, schedules and all four matched identities verify.
- 96 BA and 336 paired aggregates were independently recomputed
  from per-map values; all four frozen pair decisions agree with the runner.
- Forward length64, eight equally weighted losses, one optimizer step per
  trajectory and300 updates are shared. Only the graph cuts differ within
  an architecture/seed pair. Logged backward calls/cut counts agree.
- The300-update budget was selected from preflight timing before efficacy
  training. This changed objective and budget cannot borrow qualification
  from the earlier600-update reach/auxiliary experiment.
- Training uses fresh64-step trajectories; inference extends to256. The
  result does not test256-step training, state-pool handoffs, reopen or
  arbitrary delayed credit assignment, and does not establish novelty.
- No extra training, checkpoint replay, retuning or rescue run was performed
  for this inspection. Two seeds are descriptive, not significance evidence.

[All48 curve rows](curves.csv), [structured analysis](analysis.json),
[figure](overview.png), [provenance](provenance.json).
