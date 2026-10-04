# Credit-locality formation audit

Status: COMPLETE. No model training or optimizer updates.
Formation screen: CREDIT_SYNCHRONY_NOT_SUPPORTED (synchrony only).

| Update | P32 | G32 | P64 | G64 | Cosine | C_parallel | C_miss | Full |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 140 | 0.9382 | 0.3768 | 0.9609 | 0.2381 | 0.7158 | 0.1674 | 0.8485 | False |
| 145 | 0.9892 | 0.4054 | 0.9989 | 0.2612 | 0.5101 | 0.1017 | 0.9145 | False |
| 175 | 0.9471 | 0.5007 | 0.9997 | 0.3529 | -0.0352 | -0.0019 | 1.0033 | False |
| 180 | 0.4958 | 0.0432 | 0.6229 | 0.0780 | 0.2243 | 0.0122 | 0.9892 | False |
| 190 | 0.9399 | 0.6177 | 0.9404 | 0.3972 | 0.8549 | 0.2489 | 0.7662 | False |
| 195 | 0.9995 | 0.6872 | 0.9855 | 0.4794 | 0.8173 | 0.2040 | 0.8089 | False |
| 200 | 0.9963 | 0.6800 | 0.9993 | 0.5303 | 0.7538 | 0.4083 | 0.6905 | False |

Read summary.json and metrics.csv first. Per-batch gradients, traces, cohort denominators,
existing Full flags and figure are retained. An absent behavior contrast is unqualified,
not a credit null. Synchrony is not proof of state-to-credit causality.
