# Local Interface: completed development screen

**DEVELOPMENT_NO_GO.** All 12 arms completed 300 updates; the recorded formal run took 1281.5 seconds.
Historical baseline and streaming controls reproduce all eight final parameter hashes, full evaluation payloads, and the bound training data and schedule.
Reach-and-hold counts: baseline 2/4; stream 1/4; interface 0/4.

## Frozen primary: size 32, T64, strict 16 < d < 32

| Seed | Variant | BA original / flipped (%) | Per-map mean (%) | Pooled (correct/pixels) | Reach | Hold | Reach + hold |
|---:|---|---:|---:|---:|---|---|---|
| 2 | baseline | 98.45 / 97.68 | 93.70 | 2713/2918 (92.97%) | True | True | True |
| 2 | stream | 57.44 / 55.51 | 1.08 | 43/2918 (1.47%) | False | False | False |
| 2 | interface | 72.22 / 67.53 | 0.00 | 0/2918 (0.00%) | False | False | False |
| 3 | baseline | 90.58 / 90.59 | 21.85 | 504/2918 (17.27%) | False | False | False |
| 3 | stream | 89.03 / 90.74 | 8.28 | 164/2918 (5.62%) | False | False | False |
| 3 | interface | 58.82 / 56.43 | 0.00 | 0/2918 (0.00%) | False | False | False |
| 4 | baseline | 94.86 / 94.72 | 52.40 | 1306/2918 (44.76%) | False | True | False |
| 4 | stream | 96.98 / 96.86 | 88.89 | 2507/2918 (85.92%) | True | True | True |
| 4 | interface | 76.55 / 71.44 | 0.00 | 0/2918 (0.00%) | False | True | False |
| 5 | baseline | 98.24 / 97.74 | 96.45 | 2792/2918 (95.68%) | True | True | True |
| 5 | stream | 87.44 / 88.19 | 16.05 | 380/2918 (13.02%) | False | True | False |
| 5 | interface | 70.60 / 65.49 | 0.00 | 0/2918 (0.00%) | False | False | False |

The interface candidate scored 0/2918 on the primary paired metric for each seed at T64, T128, and T256. At size 64, its far changed-component band d>32 scored 0/27436 for each seed at all three horizons. Interface seed 4 has hold=True while reach=False; it does not pass the endpoint gate.

## Rollout summaries

The primary rollout summary reports the size-32 strict 16<d<32 band; the far summary reports the size-64 d>32 band. Values pool counts across the four independent initialization seeds for description; per-seed values remain available in analysis.json.

| Band | T | Variant | Pooled correct / pixels | Pooled accuracy |
|---|---:|---|---:|---:|
| size 32, strict_16_32 | 64 | baseline | 7315/11672 | 62.67% |
| size 32, strict_16_32 | 64 | stream | 3094/11672 | 26.51% |
| size 32, strict_16_32 | 64 | interface | 0/11672 | 0.00% |
| size 32, strict_16_32 | 128 | baseline | 6923/11672 | 59.31% |
| size 32, strict_16_32 | 128 | stream | 3365/11672 | 28.83% |
| size 32, strict_16_32 | 128 | interface | 0/11672 | 0.00% |
| size 32, strict_16_32 | 256 | baseline | 6914/11672 | 59.24% |
| size 32, strict_16_32 | 256 | stream | 3417/11672 | 29.28% |
| size 32, strict_16_32 | 256 | interface | 0/11672 | 0.00% |
| size 64, far_gt32 | 64 | baseline | 4095/109744 | 3.73% |
| size 64, far_gt32 | 64 | stream | 3701/109744 | 3.37% |
| size 64, far_gt32 | 64 | interface | 0/109744 | 0.00% |
| size 64, far_gt32 | 128 | baseline | 10123/109744 | 9.22% |
| size 64, far_gt32 | 128 | stream | 11201/109744 | 10.21% |
| size 64, far_gt32 | 128 | interface | 0/109744 | 0.00% |
| size 64, far_gt32 | 256 | baseline | 14159/109744 | 12.90% |
| size 64, far_gt32 | 256 | stream | 17001/109744 | 15.49% |
| size 64, far_gt32 | 256 | interface | 0/109744 | 0.00% |

## Scope and interpretation

This is a development comparison on previously inspected seeds and maps, conditional on one shared training bank and schedule (n=4). The complete interface parameterization changes local state allocation, neighbor representation, and hidden widths together, so this result does not isolate a causal interface effect or establish a general impossibility. No p-value, population reliability, or generalization claim is supported.
The learned candidate misses the frozen primary endpoint for every seed. Its passing hold predicate at seed 4 does not repair that failed endpoint. Farther bands are descriptive and do not replace reach and hold.

## Systems and verification

Timing, peak allocation, clipping, and median update time are descriptive summaries; raw arm records retain the update traces.

| Variant | Median training seconds | Median peak allocated MiB | Median clipped fraction |
|---|---:|---:|---:|
| baseline | 91.89 | 95.12 | 3.17% |
| stream | 124.31 | 97.08 | 6.83% |
| interface | 84.87 | 97.31 | 8.50% |

Verified 45 bound source hashes against the current checkout and both formal/preflight snapshots; all twelve CPU checkpoint state hashes; all eight historical control reproductions; data and schedule bindings; 144 BA means; 1008 paired metric/count aggregates; twelve gate decisions; 936 curve rows; and 624 paired contrasts.
Checkpoint tensors were read on CPU with weights_only=True to compare state hashes. No model inference, new training, sweep, or confirmation run was performed for this audit.

See [compact analysis](analysis.json), [validation](validation.json), [all curves](curves.csv), [paired effects](paired_effects.csv), and [raw arm records](raw/).
