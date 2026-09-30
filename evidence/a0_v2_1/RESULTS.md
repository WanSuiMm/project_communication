# A0 exported evidence

Run status: COMPLETED_FROZEN_SCHEDULE

Completed trials: 24/24 in the conditional schedule. The initial maximum was 40; 16 were excluded by failed calibration.

Seeds are 1729 / 2718 / 31415 / 57721. Accuracy values below are percentages; seed count is the replicate count.

| Dim | Arm | Train d16, per seed | d128, per seed | Mean seed-median latency, ms |
|---|---|---|---|---|
| 2 | attention | 100.00 / 100.00 / 100.00 / 100.00 | 100.00 / 100.00 / 100.00 / 100.00 | 22.13 |
| 2 | constant_raw | 50.78 / 46.09 / 55.47 / 43.75 | 50.78 / 45.31 / 54.69 / 46.09 | 35.90 |
| 2 | constant_normalized | 50.78 / 50.78 / 55.47 / 43.75 | 50.78 / 47.66 / 55.47 / 46.88 | 37.20 |
| 2 | learned_raw | 50.78 / 46.09 / 55.47 / 43.75 | 50.00 / 44.53 / 54.69 / 46.09 | 36.58 |
| 2 | learned_normalized | 50.78 / 45.31 / 55.47 / 44.53 | 50.78 / 45.31 / 56.25 / 45.31 | 37.33 |
| 3 | attention | 47.92 / 100.00 / 100.00 / 100.00 | 50.00 / 100.00 / 100.00 / 100.00 | 28.65 |

## Frozen decisions

```json
{
  "2": {
    "calibration": "PASS",
    "arms": {
      "constant_raw": "NOT_QUALIFIED_FIT",
      "constant_normalized": "NOT_QUALIFIED_FIT",
      "learned_raw": "NOT_QUALIFIED_FIT",
      "learned_normalized": "NOT_QUALIFIED_FIT"
    }
  },
  "3": {
    "calibration": "INCONCLUSIVE_POSITIVE_CONTROL",
    "arms": {
      "constant_raw": "NOT_RUN_CALIBRATION_NOT_PASSED",
      "constant_normalized": "NOT_RUN_CALIBRATION_NOT_PASSED",
      "learned_raw": "NOT_RUN_CALIBRATION_NOT_PASSED",
      "learned_normalized": "NOT_RUN_CALIBRATION_NOT_PASSED"
    }
  }
}
```

## Paired normalization effect

Normalized minus raw, percentage points. These are descriptive paired outcomes, not significance claims.

| Dim | Medium | train d16 mean | large d16 mean | d32 mean | d64 mean | d128 mean | d128 per seed |
|---|---|---|---|---|---|---|---|
| 2 | constant | +1.172 | -0.586 | +1.953 | -1.953 | +0.977 | +0.000 / +2.344 / +0.781 / +0.781 |
| 2 | learned | +0.000 | -0.195 | +0.977 | -0.977 | +0.586 | +0.781 / +0.781 / +1.562 / -0.781 |

## Interpretation boundaries

Oracle signal recoverability and passed numerical checks do not establish learned-model qualification. Inspect training-scale fitting before interpreting distant accuracy.
A dimension that failed attention calibration has no RT treatment estimate. Missing 3D arms are protocol skips, not failed RT runs.
A0 raw and normalized arms both emit q=c*v and use matched initial parameters/medium scale and packed solver work. A0 raw is not the historical v1 raw model.
Fixed axis splitting, narrow straight geometry and portable PyTorch kernels remain limitations. B/C and width sweeps were not run.
Read summary.json first; aggregate.json has the complete frozen per-axis metrics and histories, and checks.json has the oracle and software records.
