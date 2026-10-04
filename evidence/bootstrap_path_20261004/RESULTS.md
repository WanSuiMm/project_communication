# Seed4 bootstrap-path screen: completed

23/23 arms x300 updates; 51.14 minutes. Historical H reproduces its parameters and six complete endpoints exactly. Both S controls reproduce prior results.

Data-only publication for independent analysis. One init4 trajectory family, two selected alternate schedules and reused 2D size32/64 evaluation maps. `Full` is the unchanged conjunction of reach/hold/dynamics/frontier gates, not a size32-only endpoint.

| Arm | Full | size32 strict mean T64 % | T128 % | T256 % |
|---|---|---:|---:|---:|
| H | True | 93.41 | 99.95 | 100.00 |
| S20012 | False | 35.06 | 24.36 | 23.33 |
| S20012_replace_early1 | False | 98.23 | 97.35 | 98.15 |
| S20012_preserve_early1 | False | 14.28 | 0.00 | 0.00 |
| S20012_replace_late1 | False | 83.65 | 88.39 | 87.59 |
| S20012_replace_early3 | False | 56.25 | 42.93 | 41.03 |
| S20012_preserve_early3 | False | 87.40 | 76.67 | 76.67 |
| S20012_replace_late3 | False | 81.89 | 96.14 | 97.41 |
| S20012_replace_early8 | False | 38.72 | 29.49 | 27.61 |
| S20012_preserve_early8 | False | 98.14 | 96.67 | 6.15 |
| S20012_replace_late8 | False | 61.32 | 65.11 | 64.45 |
| S20022 | False | 81.67 | 63.08 | 53.73 |
| S20022_replace_early1 | False | 42.31 | 37.60 | 36.22 |
| S20022_preserve_early1 | False | 100.00 | 100.00 | 99.82 |
| S20022_replace_late1 | False | 79.21 | 85.12 | 84.30 |
| S20022_replace_early3 | True | 99.79 | 100.00 | 100.00 |
| S20022_preserve_early3 | True | 97.99 | 96.70 | 96.67 |
| S20022_replace_late3 | False | 76.88 | 78.57 | 78.18 |
| S20022_replace_early8 | False | 88.37 | 6.84 | 0.02 |
| S20022_preserve_early8 | False | 96.67 | 85.82 | 83.40 |
| S20022_replace_late8 | False | 75.90 | 84.34 | 84.26 |
| H_swap_early8 | False | 87.77 | 7.99 | 0.00 |
| H_swap_late8 | False | 80.83 | 85.07 | 84.02 |

Read [aggregate](summary.json), [first-step measurements](first_step_summary.json), [configuration](config.json), [frozen protocol](../../new/bootstrap_path/PROTOCOL.md), [validation](validation.json) and [reproduction](REPRODUCTION.md).

Full per-arm `arms/` JSON retains denominators and gate failures. `training/` contains all300 update records. `frontier/` contains exact-distance integer counts. `traces/` contains46 compressed Boolean trajectory banks; `first_step_arrays/` contains six gradient/delta/moment/Z-write NPZ files. Large arrays are secondary. Checkpoints, PIDs, private machine manifests and logs remain local.
