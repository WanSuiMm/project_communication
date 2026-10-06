# Hybrid writer paired reliability

Execution status: COMPLETE
Formal aggregation: COMPLETE
Overall verdict: NO_PRIMARY_RELIABILITY_QUALIFICATION

The formal endpoint is joint readiness at u300. Formation diagnostics use only the predeclared checkpoint grid u0, u25, ..., u300; no other update or peak checkpoint enters the comparisons.

| Arm | Valid formal blocks | Expected blocks | Joint-ready at u300 | Wilson 95% readiness interval |
|---|---:|---:|---:|---|
| neural | 8 | 8 | 1 | [0.0224175, 0.470888] |
| budget | 8 | 8 | 0 | [0, 0.324408] |
| hybrid | 8 | 8 | 0 | [0, 0.324408] |
| affine_hybrid | 8 | 8 | 0 | [0, 0.324408] |

## Paired contrasts

| Contrast (candidate − reference) | Complete pairs | Wins | Losses | Net gain | Exact two-sided p | Holm-adjusted p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| budget − neural | 8 | 0 | 1 | -1 | 1 | 1 | NO_RELIABILITY_QUALIFICATION |
| hybrid − budget | 8 | 0 | 0 | 0 | 1 | 1 | NO_RELIABILITY_QUALIFICATION |
| hybrid − neural | 8 | 0 | 1 | -1 | 1 | 1 | NO_RELIABILITY_QUALIFICATION |

Secondary affine-capacity contrast, affine_hybrid − hybrid: 0 wins, 0 losses, net 0; exact two-sided p=1; SECONDARY_DESCRIPTIVE_ONLY. This contrast is not part of the primary Holm family.

For each primary contrast, candidate-only minus reference-only wins must be at least 6/8 and its Holm-adjusted exact two-sided discordant binomial p value must be at most 0.05 across the three primary contrasts.

## Predeclared checkpoint diagnostics

| Arm | Complete trajectories | Ever-ready trajectories | Ever-ready fraction | First-ready trajectories before u300 | Lost readiness later | Loss fraction |
|---|---:|---:|---:|---:|---:|---:|
| neural | 8/8 | 1/8 | 0.125 | 1 | 0 | 0 |
| budget | 8/8 | 0/8 | 0 | 0 | 0 | — |
| hybrid | 8/8 | 0/8 | 0 | 0 | 0 | — |
| affine_hybrid | 8/8 | 0/8 | 0 | 0 | 0 | — |

Ever-ready counts any observed joint-ready checkpoint; incomplete trajectories with no observed pass are unresolved and excluded from its denominator. Readiness loss is the fraction of complete trajectories that first become joint-ready before u300 and fail at any later predeclared checkpoint. Missing checkpoint outcomes remain undefined.

Predeclared records present: 416/416; complete trajectories: 32/32.

The budget contrast bundles bounds and gates, so it does not isolate a mechanism. The verdict is finite evidence for this K8 recipe; it does not establish guaranteed closure, whole-NCA behavior, or 3D transport.

metrics.csv contains the compact measurements on the predeclared checkpoint grid. The runner retains the raw metric records.
