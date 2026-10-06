# Continuous execution-state coverage

Execution: COMPLETE

Primary: NO_CONTINUOUS_COVERAGE_RELIABILITY_QUALIFICATION

Fixed u300 is the formal endpoint; intermediate checkpoints are formation diagnostics.
K8, same batch,256 forward steps,32 losses/32 backward windows and one AdamW step per update.

| Arm | Completed trajectories | Joint readiness at u300 | Old Full at u300 |
|---|---:|---:|---:|
| reset64x4 | 8/8 | 0 | 0 |
| continuous256 | 8/8 | 0 | 0 |

Joint wins=0, losses=0, net=0/8; exact two-sided p=1.0.

Start with summary.json and metrics.csv; dense.json contains all frozen checkpoint points.
Checkpoints and Boolean packed traces are secondary. No peak checkpoint or map subset replaces u300.
Encoder-credit frequency differs as a consequence of reset; no unique visitation mechanism follows.

Dense checkpoint records completed: 208/208.
