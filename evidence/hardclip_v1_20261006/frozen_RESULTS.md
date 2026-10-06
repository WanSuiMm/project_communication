# HardClip paired reliability

Execution status: COMPLETE
Aggregation status: COMPLETE
Overall verdict: NO_HARDCLIP_RELIABILITY_QUALIFICATION

The formal endpoint is joint readiness at u300. Checkpoint diagnostics use only u0, u25, ..., u300; no peak checkpoint is selected.

| Arm | Valid u300 blocks | Expected | Joint-ready | Wilson 95% readiness interval |
|---|---:|---:|---:|---|
| neural | 8 | 8 | 0 | [0, 0.324408] |
| hardclip | 8 | 8 | 0 | [0, 0.324408] |

## Sole primary contrast

| Contrast | Complete pairs | HardClip-only wins | Neural-only wins | Net gain | Exact two-sided p | Result |
|---|---:|---:|---:|---:|---:|---|
| hardclip − neural | 8 | 0 | 0 | 0 | 1 | NOT QUALIFIED |

HardClip qualifies only when hardclip-only minus neural-only wins are at least 6 and the exact two-sided discordant-pair binomial p value is at most 0.05. This is the sole primary contrast; no Holm adjustment.

## Old Full gate (secondary)

| Arm | Full passes | Observed | Unknown | Pass fraction |
|---|---:|---:|---:|---:|
| neural | 0 | 8 | 0 | 0 |
| hardclip | 0 | 8 | 0 | 0 |

## Predeclared checkpoint diagnostics

| Arm | Complete trajectories | Ever-ready | Ever-ready fraction | First-ready before u300 | Lost readiness later | Loss fraction |
|---|---:|---:|---:|---:|---:|---:|
| neural | 8/8 | 0/8 | 0 | 0 | 0 | — |
| hardclip | 8/8 | 0/8 | 0 | 0 | 0 | — |

Predeclared checkpoint records: 208/208; complete trajectories: 16/16. Ever-ready excludes unresolved no-pass trajectories; a zero denominator is undefined. Readiness loss counts complete trajectories that first pass before u300 and fail at a later checkpoint.

Finite evidence for this fixed K8 recipe and intervention. A negative result does not exclude all small benefits or all large-write mechanisms.

metrics.csv contains one row per checkpoint and evaluated size (416 rows when all 208 expected records are present).
