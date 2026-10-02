# Persistent Local, Carrier and Task Roles screen

Execution: COMPLETE. Decision: DEVELOPMENT_NO_GO.
This is a developmental comparison on previously inspected seeds and maps.
The roles arm changes state allocation, carrier width, perception, shared local rule and readout clock together.
The result does not isolate H causality, match FLOPs, or establish a short-BPTT mechanism.

| Seed | Variant | Original BA % | Flipped BA % | Primary mean % | Primary pooled % | Reach | Hold |
|---:|---|---:|---:|---:|---:|---|---|
| 2 | baseline | 98.45 | 97.68 | 93.70 | 92.97 | True | True |
| 2 | roles | 78.25 | 79.28 | 0.00 | 0.00 | False | True |
| 2 | stream | 57.44 | 55.51 | 1.08 | 1.47 | False | False |
| 3 | baseline | 90.58 | 90.59 | 21.85 | 17.27 | False | False |
| 3 | roles | 83.62 | 83.66 | 0.53 | 0.31 | False | True |
| 3 | stream | 89.03 | 90.74 | 8.28 | 5.62 | False | False |
| 4 | baseline | 94.86 | 94.72 | 52.40 | 44.76 | False | True |
| 4 | roles | 50.00 | 51.56 | 0.00 | 0.00 | False | True |
| 4 | stream | 96.98 | 96.86 | 88.89 | 85.92 | True | True |
| 5 | baseline | 98.24 | 97.74 | 96.45 | 95.68 | True | True |
| 5 | roles | 82.48 | 78.68 | 1.80 | 1.58 | False | True |
| 5 | stream | 87.44 | 88.19 | 16.05 | 13.02 | False | True |

Reach+hold counts: baseline=2/4; stream=1/4; roles=0/4.
Primary: size32/T64, strict16<d<32. Hold: T128 and T256 relative to T64.
Exact control reproduction requires each control seed’s final parameter SHA and full evaluation payload to match its historical record.
See curves.csv and paired_effects.csv for all seeds, sizes, horizons and distance bands.
Paired effects include roles-minus-baseline and roles-minus-stream rows.
Candidate K8 has16 phases: carrier radius<=16, task readout radius<=15 from an initial pointwise source.
Empty bands are null. Farther gains cannot replace the frozen primary and hold gate.
Raw arm JSON retains per-map counts, integer denominators, clipping, timing and memory.
No reliability or population-level claim follows from four seeds and reused maps.
