# Local inertial NCA wind tunnel

Seed 0 exploratory screen. One model seed and 16 held-out maps per size; no population-level superiority claim.

| Arm | Status | Updates | Train seconds |
|---|---|---:|---:|
| masked_momentum_nca | TRAINED | 800 | 65.54 |

| Arm | Size | Last T | BA at last T | Paired correct | First sustained 95% T | Repair eligible |
|---|---:|---:|---:|---:|---:|---:|
| masked_momentum_nca | 32 | 256 | 0.7768 | 0.4278 | null | 12 |
| masked_momentum_nca | 64 | 256 | 0.6415 | 0.1414 | null | 4 |
| masked_momentum_nca | 128 | 256 | 0.5190 | 0.0712 | null | 1 |

A missing 95% threshold remains null. Repair eligibility is measured before damage at T=64.
Per-arm JSON retains all horizons, paired distance bins, damage/revision controls, gradients and measured latency.
A completed training run is not an architecture pass. Nonfinite and budget-limited runs retain their status.

## Same-medium primary comparison: 32x32 / T64

Descriptive outcome: `MOMENTUM_JOINT_5PP_ADVANTAGE`.

| Model | BA (%) | Paired correctness (%) | BCE |
|---|---:|---:|---:|
| masked_momentum | 96.45 | 77.25 | 0.0512 |
| masked_inertial | 88.79 | 37.22 | 0.2503 |
| unmasked_momentum | 84.44 | 42.75 | 0.3217 |

Single-seed architecture package comparison, not isolated factorization, significance, repair superiority or matched-speedup evidence.
Hidden widths and initial nonzero spatial coupling differ between generic and explicit cells.
No automatic seeds, retuning or architecture changes. All horizons are in comparison.json.
