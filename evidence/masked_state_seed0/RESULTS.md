# Local inertial NCA wind tunnel

Seed 0 exploratory screen. One model seed and 16 held-out maps per size; no population-level superiority claim.

| Arm | Status | Updates | Train seconds |
|---|---|---:|---:|
| masked_state_nca | TRAINED | 800 | 57.72 |

| Arm | Size | Last T | BA at last T | Paired correct | First sustained 95% T | Repair eligible |
|---|---:|---:|---:|---:|---:|---:|
| masked_state_nca | 32 | 256 | 0.1138 | 0.0938 | null | 5 |
| masked_state_nca | 64 | 256 | 0.2593 | 0.0435 | null | 2 |
| masked_state_nca | 128 | 256 | 0.3859 | 0.0089 | null | 0 |

A missing 95% threshold remains null. Repair eligibility is measured before damage at T=64.
Per-arm JSON retains all horizons, paired distance bins, damage/revision controls, gradients and measured latency.
A completed training run is not an architecture pass. Nonfinite and budget-limited runs retain their status.

## Final2x2 primary comparison:32x32/T64

Descriptive outcome: `MOMENTUM_JOINT_5PP_ADVANTAGE`.

| Generic model | Whole-grid BA % | Masked BA % | Whole-grid paired % | Masked paired % |
|---|---:|---:|---:|---:|
| State-matched | 85.17 | 90.43 | 49.31 | 56.37 |
| Momentum | 84.44 | 96.45 | 42.75 | 77.25 |

| Contrast | BA difference pp | Paired difference pp |
|---|---:|---:|
| Masked Momentum minus masked state | +6.02 | +20.88 |
| State masking effect | +5.26 | +7.06 |
| Momentum masking effect | +12.01 | +34.49 |
| Difference-in-differences | +6.75 | +27.43 |

Single seed; state scalars equal, parameter counts approximate; differing H/program widths prevent isolated velocity attribution. Small gaps are not equivalence. No further experiment scheduled.
Late horizons do not replace the primary endpoint. Inspect comparison.json for all sizes/horizons.
Conditional repair cohorts can differ; timings from separate runs are not a controlled speed comparison.

## Execution provenance

Original training and all15 learning endpoints were saved, but the process did not leave a final completion marker. Original status remains unchanged. Missing size128 timings and initial-state gradient probes were recovered from the existing checkpoint without training. Two representative endpoints replayed within5e-5. See recovery.json and original_arm.json; timings are from separate sessions.
