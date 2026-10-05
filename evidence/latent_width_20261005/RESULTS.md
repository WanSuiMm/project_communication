# Native execution-state width qualification

Execution: COMPLETE

Primary verdict: NO_W8_RELIABILITY_QUALIFICATION

| Arm | Full successes | Parameters |
|---|---:|---:|
| w24 | 0/16 | 5033 |
| w8 | 0/16 | 2521 |
| w24_capacity | 0/16 | 2521 |
| w16 | 0/16 | 3777 |
| w4 | 0/16 | 1893 |
| w2_alternating | 0/16 | 1579 |

W8 vs W24: wins 0, losses 0; net 0/16; exact two-sided p=1.

Read summary.json first, then perarm.json and metrics.csv. Per-arm evaluation/*_summary.json
contains every unchanged Full component. Curves, Boolean NPZ traces and frontier CSVs are secondary.
W2 alternates axes and is not a width-only contrast. Capacity control changes F/Q widths.
Scientific qualification failure is separate from complete execution.

## Publication and timing

All 96 training curves, per-map evaluation summaries, 192 paired Boolean
trace banks, matched-frontier tables, systems measurements, 576 aggregate
metric rows and exact batch schedules are retained. Curve/plan JSON and
frontier CSV use lossless gzip; NPZ files remain byte-exact. `perarm.json`
keeps original relative references; append `.gz` for a training-curve path.
Full numeric precision is unchanged. Banks were regenerated from frozen seeds
on CPU for publication and verified against the runtime tensor hashes.

13 eager arms were imported and 83 were trained with the qualified CUDA Graph
engine. The recorded 2771.016 seconds covers only the accelerated worker;
it is not the duration of all 96 arms including prior work and qualification.
Six block00 300-update replays exactly matched final parameters and Adam states.
That finite runtime check is separate from scientific qualification.

Checkpoints, host/PID manifests, launch receipts and logs stay local. Parameter
and checkpoint hashes remain available. Publication uses saved data and CPU
bank regeneration only; no new training, model inference or optimizer updates.
Concurrent W24 also has zero Full successes. This result rejects reliability
qualification for the frozen recipe; it does not reject all continuous latents
or establish an intrinsic carrier dimension. W2 changes transport.
