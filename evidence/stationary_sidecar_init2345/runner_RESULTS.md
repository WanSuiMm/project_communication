# Stationary Sidecar screen

Execution: COMPLETE. Decision: DEVELOPMENT_NO_GO.
This is a developmental comparison on previously inspected seeds and maps.
The memory/stateless pair has identical7989 parameter capacity; only extra H carry differs.
Carry also changes activation accumulation/scale. Neither scale-independent memory causality nor FLOP equality is established.

| Seed | Variant | Original BA % | Flipped BA % | Primary mean % | Primary pooled % | Reach | Hold |
|---:|---|---:|---:|---:|---:|---|---|
| 2 | memory | 86.25 | 87.46 | 0.61 | 0.69 | False | True |
| 2 | stateless | 95.72 | 96.45 | 73.38 | 67.07 | False | True |
| 2 | stream | 57.44 | 55.51 | 1.08 | 1.47 | False | False |
| 3 | memory | 88.53 | 84.53 | 16.06 | 14.05 | False | False |
| 3 | stateless | 86.14 | 82.09 | 5.74 | 4.80 | False | True |
| 3 | stream | 89.03 | 90.74 | 8.28 | 5.62 | False | False |
| 4 | memory | 97.80 | 97.37 | 95.51 | 94.24 | True | False |
| 4 | stateless | 99.27 | 99.23 | 100.00 | 100.00 | True | False |
| 4 | stream | 96.98 | 96.86 | 88.89 | 85.92 | True | True |
| 5 | memory | 85.70 | 84.37 | 6.54 | 3.46 | False | False |
| 5 | stateless | 84.80 | 83.46 | 17.18 | 15.39 | False | False |
| 5 | stream | 87.44 | 88.19 | 16.05 | 13.02 | False | True |

Reach+hold counts: stream=1/4; memory=0/4; stateless=0/4.
Primary: size32/T64, strict16<d<32. Hold: T128 and T256 relative to T64.
Exact control reproduction requires each control seed’s final parameter SHA and full evaluation payload to match its historical record.
See curves.csv and paired_effects.csv for all seeds, sizes, horizons and distance bands.
Paired effects include memory-minus-stream and memory-minus-stateless rows.
All arms preserve the original at-most-two-hop macro clock: K8 graph radius<=16.
Empty bands are null. Farther gains cannot replace the frozen primary and hold gate.
Raw arm JSON retains per-map counts, integer denominators, clipping, timing and memory.
No reliability or population-level claim follows from four seeds and reused maps.
