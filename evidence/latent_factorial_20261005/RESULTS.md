# C8 read / R2 factorial

Execution: COMPLETE

Primary verdict: NO_D_MINUS_A_RELIABILITY_QUALIFICATION

| Arm | Full successes / completed | Parameters |
|---|---:|---:|
| native | 0/32 | 2521 |
| factorized | 0/32 | 2649 |
| native_r2 | 0/32 | 3275 |
| factorized_r2 | 0/32 | 3403 |

D-A wins=0, losses=0, net=0/32; exact two-sided p=1.

Read summary.json, then perarm.json and metrics.csv. Complete Full components are in
each arm evaluation/*_summary.json. Traces, curves and frontier CSVs are secondary.
Completed execution and scientific qualification are distinct. R2 effects combine memory,
computation and added parameters; a D win alone does not establish interaction.


## Publication and timing

This package retains all 128 CUDA Graph K8 arms from 32 paired
initialization/schedule blocks: 128 complete 300-update curves, 256 paired
source-flip Boolean trace banks covering every integer step from 0 through 256
at sizes 32 and 64, all per-map evaluation summaries, runtime and systems
records, matched-frontier CSVs, the exact schedules, and 768 aggregate metric
rows. Curves, plans and per-block frontier CSVs use lossless gzip. Boolean NPZ
files remain byte-exact, and numeric precision is unchanged. The three
publication banks were regenerated on CPU from their frozen seeds and checked
against the recorded tensor hashes.

All 128 formal arms used the CUDA Graph K8 backend. The recorded 4361.047
seconds covers the completed 128-arm formal run only; it excludes the
five-update-per-arm preflight and preparation before the run started. The
preflight checked the actual batch-8, size-32 shape with five bitwise
eager-versus-graph updates per arm. Its qualification scope is a short replay,
not full 300-update equivalence. Its statistics field is the synthetic
eight-win fixture used to test the frozen threshold calculation, not the
formal efficacy endpoint. The formal endpoint is 0 Full successes in 32
blocks for every arm, with D-A wins=0, losses=0, net=0/32 and exact two-sided
p=1. This does not qualify the tested factorized-read/R2 recipe for reliability
on the fixed cohort; it does not reject all sidecars, latents or NCA.

Checkpoints, source snapshots, machine manifests, status/PID records, launch
receipts and logs stay local. Initial/final parameter tensor hashes and final
checkpoint file hashes are retained. Publication uses saved results and CPU
bank regeneration only; it performs no model inference, training or optimizer
updates.
