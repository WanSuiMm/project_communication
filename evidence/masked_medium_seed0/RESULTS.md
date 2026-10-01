# Local inertial NCA wind tunnel

Seed 0 exploratory screen. One model seed and 16 held-out maps per size; no population-level superiority claim.

| Arm | Status | Updates | Train seconds |
|---|---|---:|---:|
| masked_rd_nca | TRAINED | 800 | 62.49 |
| masked_inertial_rd | TRAINED | 800 | 68.56 |

| Arm | Size | Last T | BA at last T | Paired correct | First sustained 95% T | Repair eligible |
|---|---:|---:|---:|---:|---:|---:|
| masked_rd_nca | 32 | 256 | 0.7668 | 0.1678 | null | 0 |
| masked_rd_nca | 64 | 256 | 0.5018 | 0.0017 | null | 0 |
| masked_rd_nca | 128 | 256 | 0.5002 | 0.0000 | null | 0 |
| masked_inertial_rd | 32 | 256 | 0.9155 | 0.4192 | null | 7 |
| masked_inertial_rd | 64 | 256 | 0.7569 | 0.1594 | null | 2 |
| masked_inertial_rd | 128 | 256 | 0.5000 | 0.0000 | null | 0 |

A missing 95% threshold remains null. Repair eligibility is measured before damage at T=64.
Per-arm JSON retains all horizons, paired distance bins, damage/revision controls, gradients and measured latency.
A completed training run is not an architecture pass. Nonfinite and budget-limited runs retain their status.

## Prespecified comparison: 32x32 / T64

| Masked arm | Status | BA delta, pp | Paired-correct delta, pp | Descriptive criterion |
|---|---|---:|---:|---|
| masked_rd_nca | TRAINED | -8.06 | -20.68 | NO_JOINT_5PP_GAIN |
| masked_inertial_rd | TRAINED | +18.85 | +19.42 | JOINT_GAIN_5PP |

One seed only. The 5pp joint criterion is descriptive, not statistical significance.
Repair with zero eligible examples is unevaluable. Masking changes connectivity, degree and spectrum.
Old and new timing runs are not a controlled speedup comparison. No extra run is scheduled.
