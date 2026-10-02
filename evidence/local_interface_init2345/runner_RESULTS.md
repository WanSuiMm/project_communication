# Local State and Transient Interface screen

Execution: COMPLETE. Decision: DEVELOPMENT_NO_GO.
This is a developmental comparison on previously inspected seeds and maps.
The interface arm changes state allocation, neighbor representation and hidden widths together.
The result does not isolate the effect of interface factorization.

| Seed | Variant | Original BA % | Flipped BA % | Primary mean % | Primary pooled % | Reach | Hold |
|---:|---|---:|---:|---:|---:|---|---|
| 2 | baseline | 98.45 | 97.68 | 93.70 | 92.97 | True | True |
| 2 | interface | 72.22 | 67.53 | 0.00 | 0.00 | False | False |
| 2 | stream | 57.44 | 55.51 | 1.08 | 1.47 | False | False |
| 3 | baseline | 90.58 | 90.59 | 21.85 | 17.27 | False | False |
| 3 | interface | 58.82 | 56.43 | 0.00 | 0.00 | False | False |
| 3 | stream | 89.03 | 90.74 | 8.28 | 5.62 | False | False |
| 4 | baseline | 94.86 | 94.72 | 52.40 | 44.76 | False | True |
| 4 | interface | 76.55 | 71.44 | 0.00 | 0.00 | False | True |
| 4 | stream | 96.98 | 96.86 | 88.89 | 85.92 | True | True |
| 5 | baseline | 98.24 | 97.74 | 96.45 | 95.68 | True | True |
| 5 | interface | 70.60 | 65.49 | 0.00 | 0.00 | False | False |
| 5 | stream | 87.44 | 88.19 | 16.05 | 13.02 | False | True |

Reach+hold counts: baseline=2/4; stream=1/4; interface=0/4.
Primary: size32/T64, strict16<d<32. Hold: T128 and T256 relative to T64.
Exact control reproduction requires each control seed’s final parameter SHA and full evaluation payload to match its historical record.
See curves.csv and paired_effects.csv for all seeds, sizes, horizons and distance bands.
Paired effects include interface-minus-baseline and interface-minus-stream rows.
Empty bands are null. Farther gains cannot replace the frozen primary and hold gate.
Raw arm JSON retains per-map counts, integer denominators, clipping, timing and memory.
No reliability or population-level claim follows from four seeds and reused maps.
