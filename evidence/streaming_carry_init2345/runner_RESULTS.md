# Streaming Carry development screen

Status: DEVELOPMENT_NO_GO. Complete: True.
Seeds2-5 and these maps were already inspected. This is development, not confirmation.

| Seed | Variant | BA original % | BA flipped % | Primary mean % | Primary pooled % | Reach | Hold |
|---|---|---:|---:|---:|---:|---|---|
| 2 | baseline | 98.45 | 97.68 | 93.70 | 92.97 | True | True |
| 2 | stream | 57.44 | 55.51 | 1.08 | 1.47 | False | False |
| 3 | baseline | 90.58 | 90.59 | 21.85 | 17.27 | False | False |
| 3 | stream | 89.03 | 90.74 | 8.28 | 5.62 | False | False |
| 4 | baseline | 94.86 | 94.72 | 52.40 | 44.76 | False | True |
| 4 | stream | 96.98 | 96.86 | 88.89 | 85.92 | True | True |
| 5 | baseline | 98.24 | 97.74 | 96.45 | 95.68 | True | True |
| 5 | stream | 87.44 | 88.19 | 16.05 | 13.02 | False | True |

Reach+hold counts: baseline=2/4; stream=1/4.
Primary: size32/T64, strict16<d<32. Hold: T128 AND T256 relative to T64.
Read curves.csv and paired_effects.csv for every seed, size, horizon and distance band.
Empty bands are null in JSON. Farther gains cannot rescue a failed primary.
Raw arm JSON retains per-map counts, denominators, clipping, synchronized timing and memory.
Only the fixed transport is lossless; the full learned recurrence may erase or amplify messages.
No automatic confirmation, architecture rescue or monitoring is scheduled.
