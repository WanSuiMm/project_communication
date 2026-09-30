# Inertial NCA: completed seed-0 screen

Four arms completed 800 updates each and all evaluations. No arm hit the wall-clock cap or a nonfinite status.
This is a negative exploratory result for this recipe. It is not a rejection of the architecture family.

Balanced accuracy (%), mean across 16 held-out maps at each size. One training seed is the replicate count.

| Arm | 32x32, T64 | 32x32, T256 | 64x64, T256 | 128x128, T256 |
|---|---:|---:|---:|---:|
| nca_state_matched | 85.17 | 82.85 | 73.03 | 55.54 |
| momentum_nca | 84.44 | 73.38 | 74.47 | 67.18 |
| rd_nca | 81.27 | 74.79 | 56.89 | 50.33 |
| inertial_rd | 69.94 | 53.50 | 50.00 | 50.00 |

## Paired-source correctness

Both original and flipped-source answers must be correct at each pixel of the largest component.
These percentages are pooled over changed-component pixels; they are not independent replicates.

| Arm | 32x32, T64 | 32x32, T256 | 64x64, T256 | 128x128, T256 |
|---|---:|---:|---:|---:|
| nca_state_matched | 49.31 | 38.44 | 12.75 | 4.15 |
| momentum_nca | 42.75 | 28.60 | 17.29 | 13.35 |
| rd_nca | 37.81 | 41.27 | 2.53 | 0.18 |
| inertial_rd | 17.81 | 11.30 | 0.00 | 0.00 |

## Decision and limits

- Inertial RD underperforms generic momentum already at the training size and T64. Its T256 degradation is not the only source of the negative result.
- Inertial RD reaches only 72.89% BA at T32 on 32x32, then drops to 69.94% at T64 and 53.50% at T256. At 64x64 and 128x128, T256 BA is 50% and paired-source correctness is zero.
- All sustained aggregate 95% thresholds are null. No matched-quality time-to-solution claim is available.
- Inertial RD has zero pre-damage eligible maps at every size. Its main conditional-repair outcome is unevaluable, not a measured zero repair success rate.
- Other models learn partial structure but do not solve the frozen 95% criterion. This task/recipe has not established a robust high-quality reference solution.
- Single seed, small held-out banks, fixed evaluation seeds and limited rollout checkpoints prevent general superiority or impossibility conclusions. No follow-up training was launched.

## Evidence route

1. [summary.json](summary.json) includes every accuracy horizon and paired score, coefficients, gradients, timings and repair eligibility.
2. [config.json](config.json) records the actual configuration and task/metric conventions.
3. [validation.json](validation.json) and [linear_checks.json](linear_checks.json) concern implementation validity only.
4. Read an individual arm only for full repair/revision controls or distance bins:

- [nca_state_matched](arms/nca_state_matched_seed0.json)
- [momentum_nca](arms/momentum_nca_seed0.json)
- [rd_nca](arms/rd_nca_seed0.json)
- [inertial_rd](arms/inertial_rd_seed0.json)

The per-arm JSON files are exact copies of frozen output, verified by SHA256. Machine receipts, PIDs, checkpoints and transient logs remain local. Source and evidence hashes are in [the publication manifest](../../INERTIAL_PUBLICATION_MANIFEST.json).
