# AU-NCA v0: accumulated-update learned feedback

Execution **COMPLETE** (9/9 arms); verdict **AU_BENEFIT_DEVELOPMENTAL**.

| Block | Arm | T64 NMSE | T64 alpha IoU | T128 NMSE | T256 NMSE | Damaged T256 NMSE |
|---:|---|---:|---:|---:|---:|---:|
| 0 | original_k64 | 0.000192 | 1.000000 | 0.000057 | 0.000057 | 0.102165 |
| 0 | original_k8 | 0.230793 | 0.770852 | 0.282294 | 0.329328 | 0.280363 |
| 0 | au_k8 | 0.129319 | 0.864647 | 0.100933 | 0.095394 | 0.153281 |
| 1 | original_k64 | 0.000442 | 0.999786 | 0.000178 | 0.000176 | 0.015805 |
| 1 | original_k8 | 0.248437 | 0.754452 | 0.310407 | 0.444396 | 0.450296 |
| 1 | au_k8 | 0.141727 | 0.854213 | 0.244324 | 0.372152 | 0.574661 |
| 2 | original_k64 | 0.004828 | 0.993619 | 0.005269 | 0.005269 | 0.266049 |
| 2 | original_k8 | 0.174778 | 0.806049 | 0.198527 | 0.258290 | 0.319260 |
| 2 | au_k8 | 0.202447 | 0.797851 | 0.230673 | 0.314675 | 0.580772 |

All three original K64 controls meet the frozen T64 criterion (NMSE <= 0.10 and alpha IoU >= 0.80). AU-K8 beats original K8 by at least 0.05 NMSE in two of three blocks; mean paired gain is 0.060171 (positive favors AU-K8). The third block favors original K8. The stronger per-block recovery criterion is met in 0/3 blocks, so this result does not establish strong recovery.

The independent unit is a paired initialization/schedule block (n=3). The 32 mask episodes per arm are repeated measurements, not independent blocks. T128/T256 reuse the T64 state with no parameter updates; damage recovery is untrained and secondary. Long-rollout and damage outcomes vary by block.

This is a fixed-step, fixed-target developmental screen of an accumulated-update NCA adaptation, not an exact reproduction of a published Growing/Persistent NCA training recipe. It does not establish general temporal-credit recovery or reliability.

The AU state uses 145 scalars per cell versus 16 for the original cell. All arms share the same 8,336 trainable feature/projection parameters. See aggregate.json and the per-episode table for complete saved metrics.

Execution resumed at update 1176 from a committed u1175 checkpoint. The historical partial log records through u1200; its 1176..1200 tail is excluded from the canonical 27,000 update records. The reported aggregate runtime belongs to the resumed process only and excludes the earlier interrupted attempt and the gap.
