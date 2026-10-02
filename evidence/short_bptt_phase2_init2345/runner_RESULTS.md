# Phase-II initialization replication

Execution: COMPLETE; K8 narrow reach: NOT_REPLICATED; sustained: NOT_REPLICATED.
Comparison: BASELINE_UNQUALIFIED. Fixed data/schedule; four initialization seeds.

Primary size32/T64, STRICT 16<d<32. This narrower endpoint was selected from Phase I and tested on new maps.
A narrow pass does not amend the old failed gate, prove robust farther propagation, or establish an optimal K.

| Seed | K | Original BA % | Flipped BA % | Paired mean % | Paired pooled % | Reach | Hold |
|---|---:|---:|---:|---:|---:|---|---|
| 2 | 8 | 98.45 | 97.68 | 93.70 | 92.97 | True | True |
| 2 | 16 | 97.78 | 98.32 | 99.06 | 98.83 | True | True |
| 2 | 64 | 89.12 | 90.20 | 0.16 | 0.14 | False | True |
| 3 | 8 | 90.58 | 90.59 | 21.85 | 17.27 | False | False |
| 3 | 16 | 97.36 | 97.17 | 90.19 | 87.77 | True | False |
| 3 | 64 | 95.83 | 95.36 | 76.44 | 70.56 | False | False |
| 4 | 8 | 94.86 | 94.72 | 52.40 | 44.76 | False | True |
| 4 | 16 | 93.51 | 93.73 | 57.56 | 51.75 | False | False |
| 4 | 64 | 81.37 | 76.53 | 0.00 | 0.00 | False | True |
| 5 | 8 | 98.24 | 97.74 | 96.45 | 95.68 | True | True |
| 5 | 16 | 98.32 | 99.04 | 92.90 | 90.47 | True | False |
| 5 | 64 | 92.20 | 93.45 | 12.88 | 9.49 | False | True |

Per-arm JSON retains the old d>16 endpoint, farther/normalized bands, size64, all rollout horizons,
integer correct counts, denominators, timings, clipping and memory. Do not select only successful seeds.
No state pool, optimizer-crossing persistence, RelationFirst validation or general credit-assignment claim.
