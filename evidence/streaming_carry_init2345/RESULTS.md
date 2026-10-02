# Streaming Carry: completed development screen

**DEVELOPMENT_NO_GO.** All 8 arms completed 300 updates in 828.3s (13.8 minutes).
Reach-and-hold counts under the frozen gate: baseline 2/4; stream 1/4.
All 4 baseline records reproduce the Phase II final-parameter hashes and complete saved evaluations; matched training data and schedule are also identical.

## Primary gate: size32, T64, strict16<d<32

| Seed | Variant | BA original / flipped % | Paired mean / pooled % | Reach | Hold | Reach + hold |
|---:|---|---:|---:|---|---|---|
| 2 | baseline | 98.45 / 97.68 | 93.70 / 92.97 | True | True | True |
| 2 | stream | 57.44 / 55.51 | 1.08 / 1.47 | False | False | False |
| 3 | baseline | 90.58 / 90.59 | 21.85 / 17.27 | False | False | False |
| 3 | stream | 89.03 / 90.74 | 8.28 / 5.62 | False | False | False |
| 4 | baseline | 94.86 / 94.72 | 52.40 / 44.76 | False | True | False |
| 4 | stream | 96.98 / 96.86 | 88.89 / 85.92 | True | True | True |
| 5 | baseline | 98.24 / 97.74 | 96.45 / 95.68 | True | True | True |
| 5 | stream | 87.44 / 88.19 | 16.05 / 13.02 | False | True | False |

Stream met both predicates for seed(s): 4. It did not meet both for seed(s): 2, 3, 5.
The passing stream record at seed 4 has original/flipped BA 96.98%/96.86% and primary paired mean/pooled accuracy 88.89%/85.92%; this single record remains within the overall DEVELOPMENT_NO_GO decision.
Hold passes without reach at seed(s) 5; hold alone does not satisfy the task gate.

The fixed operator is specified as a permutation of four 6-channel directional ports in W (24 channels) alongside stationary local Z (8 channels). Each open-cell port has one predecessor; blocked links reverse the lane, wall ports stay fixed, and the inverse is B T B. Thus T is bijective and preserves Euclidean norm for each payload coordinate. This property applies to T alone.
In the learned step, F receives incoming T(W), stationary Z, old L(W), old L(Z), and X before the local residual update; Q then uses W'. The macro-step has at most two graph hops. The candidate also changes F's first feature to incoming T(W), so this screen cannot isolate a causal effect of T alone. Neither T's losslessness nor norm preservation implies stability or information retention for the learned recurrence.
Reach uses both whole-grid BAs >=85% and both strict-band paired statistics >=80%. Hold requires the same BAs to drop by at most 3 percentage points and the paired statistics by at most 5 points at both T128 and T256 relative to T64.

## Size32 primary statistics across horizons

| Seed | T | Baseline BA O/F % | Baseline paired mean / pooled % | Stream BA O/F % | Stream paired mean / pooled % |
|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 98.45 / 97.68 | 93.70 / 92.97 | 57.44 / 55.51 | 1.08 / 1.47 |
| 2 | 128 | 98.27 / 97.84 | 93.48 / 92.77 | 50.48 / 48.84 | 0.11 / 0.10 |
| 2 | 256 | 98.30 / 97.86 | 93.96 / 93.21 | 48.09 / 45.54 | 0.00 / 0.00 |
| 3 | 64 | 90.58 / 90.59 | 21.85 / 17.27 | 89.03 / 90.74 | 8.28 / 5.62 |
| 3 | 128 | 89.00 / 89.23 | 11.10 / 5.72 | 65.32 / 58.27 | 1.68 / 1.88 |
| 3 | 256 | 89.07 / 88.01 | 10.54 / 7.57 | 20.38 / 18.89 | 0.00 / 0.00 |
| 4 | 64 | 94.86 / 94.72 | 52.40 / 44.76 | 96.98 / 96.86 | 88.89 / 85.92 |
| 4 | 128 | 94.79 / 94.85 | 51.07 / 43.59 | 99.08 / 99.02 | 99.64 / 99.35 |
| 4 | 256 | 94.77 / 94.93 | 50.44 / 42.56 | 99.44 / 99.56 | 100.00 / 100.00 |
| 5 | 64 | 98.24 / 97.74 | 96.45 / 95.68 | 87.44 / 88.19 | 16.05 / 13.02 |
| 5 | 128 | 98.23 / 97.56 | 95.92 / 95.17 | 87.22 / 88.09 | 16.89 / 13.98 |
| 5 | 256 | 98.00 / 97.57 | 94.75 / 93.59 | 86.06 / 87.19 | 20.03 / 17.10 |

T64 determines reach; T128 and T256 determine hold against the corresponding T64 values.

## Size64 far changed-component band (d>32)

| Seed | T | Baseline mean / pooled % | Stream mean / pooled % |
|---:|---:|---:|---:|
| 2 | 64 | 12.68 / 5.07 | 0.03 / 0.04 |
| 2 | 128 | 31.71 / 26.17 | 0.02 / 0.03 |
| 2 | 256 | 38.51 / 36.78 | 0.20 / 0.16 |
| 3 | 64 | 0.00 / 0.00 | 0.57 / 0.32 |
| 3 | 128 | 0.00 / 0.00 | 1.77 / 1.20 |
| 3 | 256 | 0.00 / 0.00 | 2.30 / 1.14 |
| 4 | 64 | 0.00 / 0.00 | 26.32 / 12.48 |
| 4 | 128 | 0.00 / 0.00 | 55.78 / 38.60 |
| 4 | 256 | 0.00 / 0.00 | 74.99 / 59.43 |
| 5 | 64 | 20.80 / 9.86 | 3.74 / 0.65 |
| 5 | 128 | 21.61 / 10.73 | 4.19 / 1.01 |
| 5 | 256 | 25.67 / 14.83 | 4.42 / 1.23 |

These farther-band values are descriptive and do not replace the primary gate. All evaluated bands, denominators, horizons, sizes and paired effects are retained in [curves](curves.csv) and [paired effects](paired_effects.csv). Empty bands remain null in JSON and blank in CSV.

## Scope and verification

- This is a development result on previously inspected seeds and maps, conditional on one training bank and schedule (n=4). It supports no significance, cross-distribution reliability or general robustness claim.
- The fixed transport is lossless; that does not imply lossless learned updates, useful information retention, or stability of the full recurrence.
- The saved-result audit does not attribute outcomes to a causal dynamics or failure mechanism.
- No new training or model inference, parameter sweep, confirmation run or monitoring was performed for this publication.

## Systems and verification

| Seed | Variant | Training seconds | Peak allocated MiB | Clipped updates % |
|---:|---|---:|---:|---:|
| 2 | baseline | 85.88 | 95.12 | 4.67 |
| 3 | baseline | 90.69 | 95.12 | 6.00 |
| 4 | baseline | 86.34 | 95.12 | 1.67 |
| 5 | baseline | 88.67 | 95.12 | 1.67 |
| 2 | stream | 113.16 | 97.08 | 7.67 |
| 3 | stream | 110.08 | 97.08 | 18.33 |
| 4 | stream | 113.00 | 97.08 | 6.00 |
| 5 | stream | 109.41 | 97.08 | 3.67 |

Verified 35 bound source files and 8 checkpoint state hashes; all initialization/data/schedule bindings; 96 BA aggregates and 672 paired aggregates/denominators; 8 arm decisions; 312 paired effects; and 936 CSV rows.
This checks saved-result arithmetic and provenance. Checkpoints were loaded on CPU only to verify saved parameter hashes; no model outputs were regenerated or rescored.

[Compact analysis](analysis.json), [validation](validation.json), [raw arm records](raw/).
