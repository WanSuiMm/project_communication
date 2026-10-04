# Trajectory qualification: recorded data

| Stage | Status | Arms | Recorded decision |
|---|---|---:|---|
| 1: state cross | COMPLETE | 8 | selector in `stage1_state_cross/selection.json` |
| 2: fresh recipe | COMPLETE | 32 | NO_RELIABILITY_QUALIFICATION |
| 3: distance task | COMPLETE | 8 | SECOND_TASK_TRAINING_UNQUALIFIED |

## Stage 1 lookup

| Arm | Evaluation pass | Lookup |
|---|---:|---:|
| thetaH_mH_suffixH | True | True |
| thetaH_mH_suffixS | True | True |
| thetaS_mS_suffixH | True | True |
| thetaS_mS_suffixS | False | False |
| thetaH_mS_suffixH | False | False |
| thetaH_mS_suffixS | False | False |
| thetaS_mH_suffixH | False | False |
| thetaS_mH_suffixS | False | False |

## Stage 2 paired counts

16 pairs; baseline passes 1; treatment passes 1; treatment-only 0; baseline-only 0; discordant 0; exact two-sided McNemar p=1.0.

## Stage 3 per-arm gates and MAE (cells)

| Arm | Gate checks | 32 T64 | 32 T256 | 64 T64 | 64 T256 | d>32 MAE T64 | d>32 MAE T256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed41001 | 4/8 FAIL | 3.7285 | 3.7507 | 7.0569 | 7.1158 | 14.8177 | 14.8624 |
| seed41002 | 2/8 FAIL | 4.3233 | 6.5533 | 8.4330 | 10.3661 | 16.2479 | 13.9749 |
| seed41003 | 2/8 FAIL | 4.2344 | 6.0835 | 7.9019 | 9.4969 | 15.1974 | 12.7712 |
| seed41004 | 1/8 FAIL | 4.7134 | 10.0862 | 8.9464 | 16.3767 | 25.8855 | 33.0480 |
| seed41005 | 1/8 FAIL | 4.1602 | 6.2784 | 7.1361 | 8.4626 | 20.2777 | 24.0228 |
| seed41006 | 1/8 FAIL | 4.4071 | 6.8813 | 7.1414 | 8.3838 | 18.1516 | 22.4990 |
| seed41007 | 2/8 FAIL | 4.1229 | 5.7168 | 7.2157 | 7.3869 | 20.8461 | 23.9564 |
| seed41008 | 3/8 FAIL | 3.9389 | 4.1209 | 7.2641 | 7.1177 | 16.1943 | 17.1616 |

## Data reading order

1. `summary.json` and this table for arm and pair counts.
2. Each stage `summary.json`, then its `config.json` and fixed schedules.
3. Per-arm evaluation JSON and training curves.
4. `component_audit.json` for U3 component/checkpoint hashes.
5. The 96 compressed NPZ trace banks and 40 frontier CSVs are the larger raw evidence files.
