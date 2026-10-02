# Persistent Roles: completed development screen

**DEVELOPMENT_NO_GO.** All 12 arms completed 300 updates; the recorded formal run took 1204.2 seconds.
Historical baseline and streaming controls reproduce all eight final parameter hashes, full evaluation payloads, and the bound training data and schedule.
Reach-and-hold counts: baseline 2/4; stream 1/4; roles 0/4.

## Frozen primary: size 32, T64, strict 16 < d < 32

| Seed | Variant | BA original / flipped (%) | Per-map mean (%) | Pooled (correct/pixels) | Reach | Hold | Reach + hold |
|---:|---|---:|---:|---:|---|---|---|
| 2 | baseline | 98.45 / 97.68 | 93.70 | 2713/2918 (92.97%) | True | True | True |
| 2 | stream | 57.44 / 55.51 | 1.08 | 43/2918 (1.47%) | False | False | False |
| 2 | roles | 78.25 / 79.28 | 0.00 | 0/2918 (0.00%) | False | True | False |
| 3 | baseline | 90.58 / 90.59 | 21.85 | 504/2918 (17.27%) | False | False | False |
| 3 | stream | 89.03 / 90.74 | 8.28 | 164/2918 (5.62%) | False | False | False |
| 3 | roles | 83.62 / 83.66 | 0.53 | 9/2918 (0.31%) | False | True | False |
| 4 | baseline | 94.86 / 94.72 | 52.40 | 1306/2918 (44.76%) | False | True | False |
| 4 | stream | 96.98 / 96.86 | 88.89 | 2507/2918 (85.92%) | True | True | True |
| 4 | roles | 50.00 / 51.56 | 0.00 | 0/2918 (0.00%) | False | True | False |
| 5 | baseline | 98.24 / 97.74 | 96.45 | 2792/2918 (95.68%) | True | True | True |
| 5 | stream | 87.44 / 88.19 | 16.05 | 380/2918 (13.02%) | False | True | False |
| 5 | roles | 82.48 / 78.68 | 1.80 | 46/2918 (1.58%) | False | True | False |

At T64, roles primary correct counts by seed 2/3/4/5 are 0/9/0/46 out of 2918 each (pooled accuracy 0.00%/0.31%/0.00%/1.58%). The T128 and T256 values are reported from their own raw metrics below; they are not inferred from T64. All four roles seeds miss reach, so their hold predicates do not qualify them as reach-and-hold successes.

## Rollout summaries

The primary rollout summary reports the size-32 strict 16<d<32 band; the far summary reports the size-64 d>32 band. Values pool counts across the four independent initialization seeds for description; per-seed values remain available in analysis.json.

| Band | T | Variant | Pooled correct / pixels | Pooled accuracy |
|---|---:|---|---:|---:|
| size 32, strict_16_32 | 64 | baseline | 7315/11672 | 62.67% |
| size 32, strict_16_32 | 64 | stream | 3094/11672 | 26.51% |
| size 32, strict_16_32 | 64 | roles | 55/11672 | 0.47% |
| size 32, strict_16_32 | 128 | baseline | 6923/11672 | 59.31% |
| size 32, strict_16_32 | 128 | stream | 3365/11672 | 28.83% |
| size 32, strict_16_32 | 128 | roles | 69/11672 | 0.59% |
| size 32, strict_16_32 | 256 | baseline | 6914/11672 | 59.24% |
| size 32, strict_16_32 | 256 | stream | 3417/11672 | 29.28% |
| size 32, strict_16_32 | 256 | roles | 78/11672 | 0.67% |
| size 64, far_gt32 | 64 | baseline | 4095/109744 | 3.73% |
| size 64, far_gt32 | 64 | stream | 3701/109744 | 3.37% |
| size 64, far_gt32 | 64 | roles | 30/109744 | 0.03% |
| size 64, far_gt32 | 128 | baseline | 10123/109744 | 9.22% |
| size 64, far_gt32 | 128 | stream | 11201/109744 | 10.21% |
| size 64, far_gt32 | 128 | roles | 54/109744 | 0.05% |
| size 64, far_gt32 | 256 | baseline | 14159/109744 | 12.90% |
| size 64, far_gt32 | 256 | stream | 17001/109744 | 15.49% |
| size 64, far_gt32 | 256 | roles | 74/109744 | 0.07% |

## Scope and interpretation

This is a development comparison on previously inspected seeds and maps, conditional on one shared training bank and schedule (n=4). The complete roles parameterization changes H/C allocation, carrier width, perception, shared local rule, and readout clock together, so this result does not isolate H causality or establish a general impossibility. Its K8 bounds are carrier radius 16 and task-readout radius 15; the frozen strict d>16 primary remains outside that window. No p-value, population reliability, or generalization claim is supported.
The learned candidate misses the frozen primary endpoint for every seed. Its passing hold predicate at seed 4 does not repair that failed endpoint. Farther bands are descriptive and do not replace reach and hold.

## Systems and verification

Timing, peak allocation, clipping, and median update time are descriptive summaries; raw arm records retain the update traces.

| Variant | Median training seconds | Median peak allocated MiB | Median clipped fraction |
|---|---:|---:|---:|
| baseline | 90.84 | 95.12 | 3.17% |
| stream | 121.97 | 97.08 | 6.83% |
| roles | 74.71 | 93.97 | 45.83% |

Verified 51 bound source hashes against the current checkout and both formal/preflight snapshots; all twelve CPU checkpoint state hashes; all eight historical control reproductions; data and schedule bindings; 144 BA means; 1008 paired metric/count aggregates; twelve gate decisions; 936 curve rows; and 624 paired contrasts.
Checkpoint tensors were read on CPU with weights_only=True to compare state hashes. No model inference, new training, sweep, or confirmation run was performed for this audit.

See [compact analysis](analysis.json), [validation](validation.json), [all curves](curves.csv), [paired effects](paired_effects.csv), and [raw arm records](raw/).
