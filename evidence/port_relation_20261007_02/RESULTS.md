# Port-relation paired formation screen

Execution status: COMPLETE
Aggregation status: COMPLETE
Overall verdict: NO_CONDITIONED_RELIABILITY_QUALIFICATION

The sole primary endpoint is size-32 joint readiness at update 300. The only qualifying contrast is conditioned versus current; constant comparisons, old Full, size-64, and intermediate stages are descriptive.

| Arm | Valid u300 blocks | Expected | Joint-ready | Wilson 95% interval | Old Full passes | Full observed |
|---|---:|---:|---:|---|---:|---:|
| current | 8 | 8 | 0 | [0, 0.324408] | 0 | 8/8 |
| constant | 8 | 8 | 0 | [0, 0.324408] | 0 | 8/8 |
| conditioned | 8 | 8 | 0 | [0, 0.324408] | 0 | 8/8 |

## Sole primary contrast

| Contrast | Complete pairs | Conditioned-only wins | Current-only losses | Net gain | Exact two-sided p | Result |
|---|---:|---:|---:|---:|---:|---|
| conditioned − current at u300 | 8/8 | 0 | 0 | 0 | 1 | NOT QUALIFIED |

The sole primary screen compares conditioned with current on size-32 joint readiness at u300. Qualification requires conditioned-only minus current-only blocks >=6/8 and an exact two-sided sign-test p<=0.05 over discordant paired blocks.

A missing endpoint pair or incomplete dense grid leaves the primary verdict INCOMPLETE. A complete negative result rejects reliable formation for this candidate under this protocol only.

## Paired block outcomes

| Block | Current | Constant | Conditioned | Primary comparison |
|---:|---|---|---|---|
| 0 | not ready | not ready | not ready | neither_ready |
| 1 | not ready | not ready | not ready | neither_ready |
| 2 | not ready | not ready | not ready | neither_ready |
| 3 | not ready | not ready | not ready | neither_ready |
| 4 | not ready | not ready | not ready | neither_ready |
| 5 | not ready | not ready | not ready | neither_ready |
| 6 | not ready | not ready | not ready | neither_ready |
| 7 | not ready | not ready | not ready | neither_ready |

## Secondary paired counts

These are descriptive block counts only; they have no qualification threshold and do not change the primary conclusion.

| Contrast | Complete pairs | Candidate-only | Reference-only | Both ready | Neither ready |
|---|---:|---:|---:|---:|---:|
| conditioned − constant | 8/8 | 0 | 0 | 0 | 8 |
| constant − current | 8/8 | 0 | 0 | 0 | 8 |

## Final u300 reach and retention metrics

Values are arithmetic means across blocks with defined values; n reports the number of defined blocks.

| Arm | Size | R strict pooled T64 | R strict map mean T64 | S retention T64→T256 |
|---|---:|---:|---:|---:|
| current | 32 | 0.110424 (n=8/8) | 0.138514 (n=8/8) | 0.54894 (n=8/8) |
| current | 64 | 0.133784 (n=8/8) | 0.137794 (n=8/8) | 0.396374 (n=8/8) |
| constant | 32 | 0.13741 (n=8/8) | 0.170209 (n=8/8) | 0.42945 (n=8/8) |
| constant | 64 | 0.17447 (n=8/8) | 0.179095 (n=8/8) | 0.478002 (n=8/8) |
| conditioned | 32 | 0.0509477 (n=8/8) | 0.0821189 (n=8/8) | 0.475381 (n=8/8) |
| conditioned | 64 | 0.057166 (n=8/8) | 0.0596059 (n=8/8) | 0.366254 (n=8/8) |

## Dense stage diagnostics

Intermediate stages and ever-ready counts are formation diagnostics; they never replace the fixed u300 endpoint or select a checkpoint.

| Arm | Complete stage-record paths | Ever-ready trajectories | Ever-ready fraction | Readiness losses after first ready |
|---|---:|---:|---:|---:|
| current | 8/8 | 0/8 | 0 | 0/0 |
| constant | 8/8 | 1/8 | 0.125 | 1/1 |
| conditioned | 8/8 | 0/8 | 0 | 0/0 |

Valid u300 rows: 24/24. Predeclared stage records: 312/312; complete stage-record paths: 24/24.

This is finite evidence for this three-arm candidate and eight paired initialization/schedule blocks. A failure to form the endpoint rejects reliable formation for this candidate under this protocol; it does not close the broader port-relation route or prove a relation-algebra causal mechanism. A scheduled nonfinite evaluation is recorded as non-ready with trace_complete=false and numerical_failure=true; its attempt record is not a completed Boolean bank trace. Training or engineering errors stop the run and leave the experiment incomplete.

final_metrics.csv contains u300 rows by block, arm, and size. metrics.csv contains predeclared stage rows by block, arm, update, and size (expected 624 rows when complete).
