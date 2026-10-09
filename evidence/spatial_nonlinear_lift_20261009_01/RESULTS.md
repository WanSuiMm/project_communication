# Spatial nonlinear lift: cyclic 2D quadratic-write screen

Execution: **COMPLETE**. Verdict: **SHORT_CREDIT_RECOVERY_DEVELOPMENTAL**.
Completed 9/9 planned arms; predeclared control stop: False.

| Block | Arm | Heldout T64 R2 | Long128 R2 | Long256 R2 | Side16 T64 R2 |
|---|---|---:|---:|---:|---:|
| 0 | original_k64 | 1.000000 | 0.999999 | 0.999998 | 0.999999 |
| 0 | original_k8 | -0.653967 | -1.629410 | -1.405874 | -0.931875 |
| 0 | lifted_k8 | 1.000000 | 0.999999 | 0.999998 | 0.999999 |
| 1 | original_k64 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 1 | original_k8 | 0.027684 | -1.089484 | -0.913426 | -0.359911 |
| 1 | lifted_k8 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 2 | original_k64 | 0.999587 | 0.999326 | 0.999497 | 0.999762 |
| 2 | original_k8 | 0.050469 | 0.073153 | 0.089231 | 0.250447 |
| 2 | lifted_k8 | 0.999587 | 0.999326 | 0.999497 | 0.999762 |

All predictions and per-map MSE are retained. Primary model selection is fixed at u300.
Original and lifted have nine identical parameter coordinates and exact real-arithmetic forward functions.
Lifted execution uses42 rather than2 scalars per cell, plus a shared time counter.
Full-credit/lifted-short equality is predicted algebraically; empirical interest is the original K8 gap and systems cost.
Long tests use separate proportionally phased teacher-labeled episodes, not autonomous continuation or stability tests.
This is a matched finite-degree teacher screen with fixed input features and transport; no generic nonlinear-feedback claim.
