# Direct Spatial Carry development screen

Status: DEVELOPMENT_NO_GO. Complete: True.
Seeds2-5 and these maps were already inspected. This is development, not confirmation.

| Seed | Variant | BA original % | BA flipped % | Primary mean % | Primary pooled % | Reach | Hold |
|---|---|---:|---:|---:|---:|---|---|
| 2 | baseline | 98.45 | 97.68 | 93.70 | 92.97 | True | True |
| 2 | carry | 83.92 | 85.50 | 0.00 | 0.00 | False | True |
| 3 | baseline | 90.58 | 90.59 | 21.85 | 17.27 | False | False |
| 3 | carry | 90.44 | 92.05 | 2.87 | 3.12 | False | False |
| 4 | baseline | 94.86 | 94.72 | 52.40 | 44.76 | False | True |
| 4 | carry | 91.07 | 88.38 | 14.39 | 11.41 | False | False |
| 5 | baseline | 98.24 | 97.74 | 96.45 | 95.68 | True | True |
| 5 | carry | 89.85 | 86.81 | 7.59 | 4.42 | False | False |

Reach+hold counts: baseline=2/4; carry=0/4.
Primary: size32/T64, strict16<d<32. Hold: T128 AND T256 relative to T64.
Read curves.csv and paired_effects.csv for every seed, size, horizon and distance band.
Empty bands are null in JSON. Farther gains cannot rescue a failed primary.
Raw arm JSON retains per-map counts, denominators, clipping, synchronized timing and memory.
Carry is fixed lazy diffusion; a benefit alone does not identify a continuation mechanism.
No automatic confirmation, architecture rescue or monitoring is scheduled.
