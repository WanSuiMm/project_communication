# Phase-II result inspection

All12 arms completed300 updates in1095.047s. Frozen K8 reach and sustained replication: NOT_REPLICATED / NOT_REPLICATED.
Comparison status: BASELINE_UNQUALIFIED. Fixed train bank/schedule, four initialization seeds; conditional descriptive result.

| K | Reach count | Reach and hold count |
|---|---:|---:|
| 8 | 2/4 | 2/4 |
| 16 | 3/4 | 1/4 |
| 64 | 0/4 | 0/4 |

Primary size32/T64 STRICT16<d<32, mean and pooled>=80%, both BAs>=85%. K8 requires>=3/4 seeds.

| Seed | K | Primary mean % | Primary pooled % | Reach | Hold |
|---|---:|---:|---:|---|---|
| 2 | 8 | 93.70 | 92.97 | True | True |
| 2 | 16 | 99.06 | 98.83 | True | True |
| 2 | 64 | 0.16 | 0.14 | False | True |
| 3 | 8 | 21.85 | 17.27 | False | False |
| 3 | 16 | 90.19 | 87.77 | True | False |
| 3 | 64 | 76.44 | 70.56 | False | False |
| 4 | 8 | 52.40 | 44.76 | False | True |
| 4 | 16 | 57.56 | 51.75 | False | False |
| 4 | 64 | 0.00 | 0.00 | False | True |
| 5 | 8 | 96.45 | 95.68 | True | True |
| 5 | 16 | 92.90 | 90.47 | True | False |
| 5 | 64 | 12.88 | 9.49 | False | True |

## Interpretation

- K8 has two new positive cases (seeds2/5), both retained throughT256. The predefined>=3/4 initialization replication failed; do not call the whole screen passed.
- K16 reaches the narrow endpoint in3/4 seeds, but only1/4 meets reach+hold. This band fits within its32-edge window; it does not show cross-window composition forK16.
- All four K64 controls remain unqualified. K8 is also worse than K64 on seed3. No generally optimal truncation horizon or superiority claim.
- Low clipping is not sufficient: all K8 clip<=6%, including the two reach failures. High clipping is not invariably failure: K16 seed2 clips60% and passes reach+hold. These observations do not isolate a mechanism.
- Size32 evaluation has no changed-component pixels at distance>=64. Those entries are null, not zero. Use size64 for this distance range.

## Successful K8 cases: limits of distance/size transfer

| Seed | Size | T | Primary mean / pooled % | d>32 mean / pooled % | d>16 mean / pooled % |
|---|---:|---:|---:|---:|---:|
| 2 | 32 | 64 | 93.70 / 92.97 | 49.11 / 25.53 | 84.42 / 70.74 |
| 2 | 32 | 128 | 93.48 / 92.77 | 57.50 / 44.57 | 86.78 / 76.64 |
| 2 | 32 | 256 | 93.96 / 93.21 | 57.80 / 44.43 | 87.11 / 76.93 |
| 2 | 64 | 64 | 88.94 / 89.08 | 12.68 / 5.07 | 34.97 / 25.64 |
| 2 | 64 | 128 | 88.86 / 89.00 | 31.71 / 26.17 | 48.69 / 41.45 |
| 2 | 64 | 256 | 89.50 / 89.65 | 38.51 / 36.78 | 54.54 / 49.61 |
| 5 | 32 | 64 | 96.45 / 95.68 | 32.22 / 30.25 | 84.81 / 73.91 |
| 5 | 32 | 128 | 95.92 / 95.17 | 28.86 / 26.94 | 83.79 / 72.61 |
| 5 | 32 | 256 | 94.75 / 93.59 | 26.88 / 24.47 | 82.65 / 71.07 |
| 5 | 64 | 64 | 98.96 / 98.55 | 20.80 / 9.86 | 42.45 / 31.81 |
| 5 | 64 | 128 | 98.40 / 98.00 | 21.61 / 10.73 | 43.01 / 32.43 |
| 5 | 64 | 256 | 97.56 / 97.25 | 25.67 / 14.83 | 45.89 / 35.36 |

Both successful K8 cases retain high narrow-band scores at size64, but broad far accuracy remains much weaker. Extra rollout helps farther propagation for seed2; seed5 improves less. This is not robust global propagation.

## Systems and verification

| Seed | K | Train seconds | Peak MiB | Clipping % |
|---|---:|---:|---:|---:|
| 2 | 16 | 92.62 | 155.81 | 60.00 |
| 3 | 16 | 88.03 | 155.81 | 23.00 |
| 4 | 16 | 86.08 | 155.81 | 62.00 |
| 5 | 16 | 92.86 | 155.81 | 56.67 |
| 2 | 64 | 86.86 | 519.41 | 92.00 |
| 3 | 64 | 85.31 | 519.41 | 88.00 |
| 4 | 64 | 86.33 | 519.41 | 91.00 |
| 5 | 64 | 86.58 | 519.41 | 91.33 |
| 2 | 8 | 87.97 | 95.12 | 4.67 |
| 3 | 8 | 91.67 | 95.12 | 6.00 |
| 4 | 8 | 84.78 | 95.12 | 1.67 |
| 5 | 8 | 84.28 | 95.12 | 1.67 |

Verified18 source snapshots,12 checkpoints, shared data/schedule and initial parameters,144 BA aggregates,1008 paired aggregates with reconstructed denominators, all12 decisions and paired effects. No new training or inference.

Narrow band and architecture were selected from Phase I; new evaluation maps were frozen prospectively. The previous gate remains unchanged. n=4 is initialization replication conditional on one bank/schedule, not significance or training-distribution robustness.

[All72 curve rows](curves.csv), [structured analysis](analysis.json), [provenance](provenance.json).
