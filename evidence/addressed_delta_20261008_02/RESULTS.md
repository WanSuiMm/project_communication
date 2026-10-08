# Addressed delta development screen

Status: **COMPLETE**. Verdict: **NO_DEVELOPMENTAL_SIGNAL**.

The primary comparison is delta versus additive on size-32 joint readiness at u300. The four paired blocks are developmental evidence only.

Records: final 12/12; dense stages 156/156.

## u300 readiness

| Arm | Ready | Observed |
|---|---:|---:|
| current | 2 | 4 |
| additive | 0 | 4 |
| delta | 0 | 4 |

Delta-only wins=0, additive-only wins=0, ties=4, net=0; exact two-sided p=1.000 (descriptive only).

## Final-block metric means

Means below average available block values. Retention and support remain undefined when the frozen evaluator does not provide them.

| Arm | Size | R pooled T64 | R mean T64 | Retention 64→256 | Continuous survival | Support maps | Support pixels |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | 32 | 0.5292 | 0.6039 | 0.9657 | 0.9178 | 31.8 | 5049.0 |
| current | 64 | 0.6503 | 0.6471 | 0.9727 | 0.9356 | 32.0 | 16504.2 |
| additive | 32 | 0.0992 | 0.1482 | 0.0573 | 0.0419 | 30.0 | 2227.5 |
| additive | 64 | 0.1256 | 0.1239 | 0.0574 | 0.0430 | 30.8 | 5508.0 |
| delta | 32 | 0.1826 | 0.2487 | 0.1614 | 0.1515 | 32.0 | 2953.5 |
| delta | 64 | 0.1854 | 0.1864 | 0.3286 | 0.3152 | 31.8 | 6983.0 |

## Checkpoint diagnostics

| Arm | Stage records | Complete trajectories | Ever ready | Known nonready | Indeterminate without pass |
|---|---:|---:|---:|---:|---:|
| current | 52/52 | 4/4 | 2 | 2 | 0 |
| additive | 52/52 | 4/4 | 0 | 4 | 0 |
| delta | 52/52 | 4/4 | 0 | 4 | 0 |

Per-stage metrics, including both evaluation sizes and endpoint scores where available, are in `metrics.csv`; the 12 formal endpoints are in `final_metrics.csv`. No full Boolean traces or Full phenotype results are persisted.

A developmental signal is not a reliability or significance result. Even 4 delta-only wins and 0 losses give exact two-sided p=0.125.
