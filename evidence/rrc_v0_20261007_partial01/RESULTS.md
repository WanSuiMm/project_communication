# Publication status: PARTIAL / INTERRUPTED

Fixed snapshot: **10/24 u300 trajectories, 137/312 complete stage evaluations**. This is not the final experiment. The saved worker status was RUNNING, but progress stopped; its original tool session is unavailable and the original worker was not found. Do not infer final no-qualification from these partial counts. No run was restarted for this upload.

# RRC v0 paired reliability screen

Execution status: PARTIAL_INTERRUPTED_SNAPSHOT
Aggregation status: INCOMPLETE
Overall verdict: INCOMPLETE

The formal endpoint is joint readiness at u300. The sole primary contrast is RRC versus Factorized; Current, old Full, and intermediate checkpoint outcomes are secondary.

| Arm | Valid u300 blocks | Expected | Joint-ready | Wilson 95% readiness interval | Old Full passes | Full observed |
|---|---:|---:|---:|---|---:|---:|
| current | 4 | 8 | 1 | — (incomplete; no CI) | 0 | 4/8 |
| factorized | 3 | 8 | 0 | — (incomplete; no CI) | 0 | 3/8 |
| rrc | 3 | 8 | 0 | — (incomplete; no CI) | 0 | 3/8 |

## Sole primary contrast

| Contrast | Complete pairs | RRC-only wins | Factorized-only losses | Net gain | Exact two-sided p | Result |
|---|---:|---:|---:|---:|---:|---|
| RRC − Factorized at u300 | 3/8 | 0 | 0 | 0 | — | INCOMPLETE |

The sole primary screen is RRC-only minus Factorized-only joint-ready blocks at u300: net gain must be at least 6/8 and the exact two-sided binomial sign-test p value over discordant pairs must be at most 0.05. Current and all intermediate checkpoints are secondary.

Current is a secondary descriptive arm and does not enter the primary test.

## Paired block outcomes

| Block | Current | Factorized | RRC | Primary RRC − Factorized |
|---:|---|---|---|---|
| 0 | not ready | not ready | not ready | neither_ready |
| 1 | not ready | not ready | not ready | neither_ready |
| 2 | ready | not ready | not ready | neither_ready |
| 3 | not ready | unknown | unknown | missing_or_invalid_formal_pair |
| 4 | unknown | unknown | unknown | missing_or_invalid_formal_pair |
| 5 | unknown | unknown | unknown | missing_or_invalid_formal_pair |
| 6 | unknown | unknown | unknown | missing_or_invalid_formal_pair |
| 7 | unknown | unknown | unknown | missing_or_invalid_formal_pair |

## Final u300 reach and retention metrics

Values are arithmetic means across blocks with defined values; n reports the number of defined blocks.

| Arm | Size | R strict pooled T64 | R strict map mean T64 | S retention T64→T256 |
|---|---:|---:|---:|---:|
| current | 32 | 0.481318 (n=4/8) | 0.523092 (n=4/8) | 0.762579 (n=4/8) |
| current | 64 | 0.571516 (n=4/8) | 0.572849 (n=4/8) | 0.830805 (n=4/8) |
| factorized | 32 | 0.0128761 (n=3/8) | 0.0160238 (n=3/8) | 0.389889 (n=3/8) |
| factorized | 64 | 0.0388837 (n=3/8) | 0.0390165 (n=3/8) | 0.450144 (n=3/8) |
| rrc | 32 | 0.0747292 (n=3/8) | 0.1167 (n=3/8) | 0.650113 (n=3/8) |
| rrc | 64 | 0.099026 (n=3/8) | 0.10183 (n=3/8) | 0.418316 (n=3/8) |

## Dense checkpoint diagnostics (secondary)

| Arm | Complete trajectories | Ever-ready trajectories | Ever-ready fraction | Readiness losses after first ready |
|---|---:|---:|---:|---:|
| current | 4/8 | 1/4 | 0.25 | 1/1 |
| factorized | 3/8 | 0/3 | 0 | 0/0 |
| rrc | 3/8 | 0/3 | 0 | 0/0 |

Valid u300 rows: 10/24; paired primary blocks: 3/8. Predeclared dense records: 137/312; complete trajectories: 10/24.

Ever-ready is the fraction of trajectories with at least one observed joint-ready checkpoint; unresolved no-pass trajectories are excluded from its denominator. Intermediate checkpoints never replace the fixed u300 endpoint.

Finite evidence for this frozen RRC v0 recipe and eight paired blocks. The u300 RRC-versus-Factorized contrast is the sole reliability screen; Current, old Full, and ever-ready checkpoint counts are descriptive secondary outcomes. No peak checkpoint replaces u300.

final_metrics.csv contains u300 metrics by block, arm, and size. metrics.csv contains one row per dense checkpoint and size (624 rows when all 312 expected records are present).
