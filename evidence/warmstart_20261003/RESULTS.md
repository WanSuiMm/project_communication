# Detached warm-start development screen

Decision: **DEVELOPMENT_NOT_QUALIFIED**. The run completed 8 arms × 300 updates in 1629.375 seconds.

The independent unit is the paired initialization (four pairs, one fixed bank and batch schedule). All four baseline arms replayed their historical six-endpoint controls; the known seed-4 baseline passed the fresh full phenotype. Baseline passed 1/4 and warm-start 0/4, so the prespecified positive development criterion was not met.

## Primary gates

| Seed | Baseline replay | Baseline reach | Hold | Dynamics | Frontier | Full | Warm-start reach | Hold | Dynamics | Frontier | Full |
|---:|---|---|---|---|---|---|---|---|---|---|---|
| 2 | PASS | False | False | False | False | False | False | True | False | True | False |
| 3 | PASS | False | False | False | False | False | False | True | False | False | False |
| 4 | PASS | True | True | True | True | True | False | False | False | True | False |
| 5 | PASS | False | True | False | True | False | False | True | False | True | False |

## Partial metrics

These secondary values retain the fixed per-arm measurements even when the conjunction fails. Strict pooled coverage uses changed pixels with `16 < BFS distance < 32`. Retention, growth, and regression use all changed pixels. Rates are pooled counts; the complete map-level profiles and every frozen gate value are in the per-arm summaries.

| Arm | Strict pooled T64 | T128 | T256 | Size32 retention 64→256 | gain 64→256 | regressed / ever correct | Size64 retention 64→256 | gain 64→256 | regressed / ever correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_seed2 | 0.0068 | 0.0000 | 0.0000 | 0.0000 (0/360) | -0.0398 | 0.9990 (3824/3828) | 0.0403 (200/4967) | -0.1128 | 0.9693 (14159/14607) |
| warmstart_seed2 | 0.0487 | 0.1651 | 0.2665 | 0.8879 (2892/3257) | +0.0825 | 0.2776 (1347/4852) | 0.9611 (8506/8850) | +0.1183 | 0.1536 (2194/14283) |
| baseline_seed3 | 0.0598 | 0.0195 | 0.0000 | 0.0000 (0/3367) | -0.3774 | 1.0000 (5324/5324) | 0.2919 (2776/9511) | -0.1228 | 0.7882 (11814/14989) |
| warmstart_seed3 | 0.0055 | 0.0169 | 0.0380 | 0.7645 (1613/2110) | +0.0507 | 0.5452 (2051/3762) | 0.4043 (38/94) | -0.0008 | 0.9963 (3798/3812) |
| baseline_seed4 | 0.8512 | 0.9984 | 1.0000 | 1.0000 (6761/6761) | +0.1982 | 0.0302 (258/8529) | 0.9689 (20329/20981) | +0.2984 | 0.0681 (2296/33698) |
| warmstart_seed4 | 0.4696 | 0.4075 | 0.0149 | 0.0410 (191/4659) | -0.4883 | 0.9582 (6939/7242) | 0.0506 (724/14309) | -0.3232 | 0.9547 (29232/30620) |
| baseline_seed5 | 0.1284 | 0.1407 | 0.1648 | 0.9143 (3533/3864) | -0.0145 | 0.2610 (1218/4666) | 0.8692 (7951/9147) | +0.0112 | 0.5120 (7065/13798) |
| warmstart_seed5 | 0.0036 | 0.0068 | 0.0065 | 0.8828 (535/606) | +0.0087 | 0.9376 (2391/2550) | 0.6955 (1679/2414) | -0.0023 | 0.9680 (8140/8409) |

Partial metrics vary across paired seeds. Seed-2 warm-start improves T64 size32 open-grid BA (0.569/0.589 to 0.900/0.862), size32 retention (0.000 to 0.888), and size32 coverage gain (-0.040 to +0.083), while still missing the full gate. Seed-4 warm-start moves in the other direction: strict pooled T64 is 0.851 baseline versus 0.470 warm-start; retention is 1.000/0.969 versus 0.041/0.051 at sizes 32/64; regression is 0.030/0.068 versus 0.958/0.955. This screen does not support a uniformly worse or uniformly better claim. It tests one detached prefix-age exposure recipe on one fixed bank and schedule; it does not establish a phase transition, continuation closure, causal handoff, or general rejection of warm-start training.

## Frozen gate definitions

Reach: size32/T64 strict paired map mean and pooled coverage ≥0.80, with original and source-flipped open-grid balanced accuracy ≥0.85. Hold: size32/T128 and T256 drops from T64 ≤0.03 balanced accuracy and ≤0.05 strict mean and pooled coverage. Dynamics: at each size, all-changed T64→T256 retention ≥0.95, pooled coverage gain ≥0.05, and ever-regressed / ever-correct ≤0.15. Frontier: at each size, exact-map/time/BFS-distance matched effect ≥0.05, at least 16 eligible maps and 100 common strata. The complete gate values and individual checks are included in each arm summary.

All stage diagnostics are exploratory observations at updates 0, 100, 200, and 300 on 16 held-out size32 maps. Integer acquisition/destruction rows preserve their denominators and the identity `correct_new - correct_old = acquired - destroyed`; stage snapshots are not additional independent runs.

## Reading order

1. This report and [compact aggregate](summary.json).
2. [Frozen protocol](../../new/warmstart/PROTOCOL.md), [configuration](config.json), and [source bindings](source_bindings.json).
3. Per-arm records under `raw/`, complete gate summaries under `training/arms/`, and matched strata under `frontier/`.
4. [Stage snapshots](stage/stage_summaries.json) and [integer turnover rows](stage/turnover.csv).
5. [Provenance and excluded-artifact hashes](provenance.json), [public arithmetic validation](publication_validation.json), and [full local validation](validation/local_validation.json).

Checkpoints, NPZ traces, source snapshots, full diagnostic JSON traces, and private execution receipts remain excluded. Their available hashes are recorded in provenance.json.
