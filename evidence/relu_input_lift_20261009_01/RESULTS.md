# ReLU activation-conditioned input lift v0

Execution **COMPLETE** (3/3 new arms); verdict **NO_QUALIFIED_RELU_LIFT_BENEFIT**.

The primary endpoint is intact T64 NMSE at update 3000. Historical AU-NCA controls were reused from the pinned public package and were not retrained.

| Block | Arm | T64 NMSE | T64 alpha IoU | T128 NMSE | T256 NMSE | Damaged T256 NMSE |
|---:|---|---:|---:|---:|---:|---:|
| 0 | original_k64 | 0.000192 | 1.000000 | 0.000057 | 0.000057 | 0.102165 |
| 0 | original_k8 | 0.230793 | 0.770852 | 0.282294 | 0.329328 | 0.280363 |
| 0 | au_k8 | 0.129319 | 0.864647 | 0.100933 | 0.095394 | 0.153281 |
| 0 | relu_lift_k8 | 0.139306 | 0.855474 | 0.171467 | 0.219726 | 0.555551 |
| 1 | original_k64 | 0.000442 | 0.999786 | 0.000178 | 0.000176 | 0.015805 |
| 1 | original_k8 | 0.248437 | 0.754452 | 0.310407 | 0.444396 | 0.450296 |
| 1 | au_k8 | 0.141727 | 0.854213 | 0.244324 | 0.372152 | 0.574661 |
| 1 | relu_lift_k8 | 0.175434 | 0.858214 | 0.168919 | 0.246305 | 0.518784 |
| 2 | original_k64 | 0.004828 | 0.993619 | 0.005269 | 0.005269 | 0.266049 |
| 2 | original_k8 | 0.174778 | 0.806049 | 0.198527 | 0.258290 | 0.319260 |
| 2 | au_k8 | 0.202447 | 0.797851 | 0.230673 | 0.314675 | 0.580772 |
| 2 | relu_lift_k8 | 0.128720 | 0.854358 | 0.133375 | 0.137734 | 0.236400 |

Mean paired AU-K8 minus ReLU-lift T64 NMSE is 0.010011; positive values favor ReLU-lift. It improves on AU-K8 in 1/3 blocks, below the frozen requirement of at least two improving blocks and mean gain at least 0.03.

The independent unit is a paired initialization/schedule block (n=3); 32 evaluation episodes are repeated measurements within a block. The historical controls share the same target and packed input banks, pinned by SHA-256 in reference_bindings.json.

T128/T256 same-state continuation, damaged-state evaluation, and gradient diagnostics are secondary. The result does not establish general NCA reliability or a universal BPTT solution.

The aggregate records elapsed_seconds_this_execution=5174.0369; this is the scientific run's saved duration field.

See [historical AU-NCA results](../au_nca_20261009_01/RESULTS.md), final_metrics.csv, per_episode.csv, and validation.json.
