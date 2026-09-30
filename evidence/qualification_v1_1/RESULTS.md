# Reaction-Transport local qualification

Status: COMPLETED_FROZEN_SCHEDULE

Two training seeds are paired replicates; voxel counts and timing repetitions are not independent training runs.

| Dimension | Gate A | Gate B | Gate C |
|---|---|---|---|
| 2D | INCONCLUSIVE_POSITIVE_CONTROL | NOT_RUN_A_DID_NOT_PASS | NOT_RUN_B_DID_NOT_PASS |
| 3D | INCONCLUSIVE_POSITIVE_CONTROL | NOT_RUN_A_DID_NOT_PASS | NOT_RUN_B_DID_NOT_PASS |

| Gate | Dim | Variant | r | Seed | Fit d16 | d32 | d64 | d128 | ms/image |
|---|---|---|---|---|---|---|---|---|---|
| A | 2 | cnn | 32 | 1729 | 0.578 | 0.594 | 0.438 | 0.555 | 3.90 |
| A | 3 | cnn | 32 | 1729 | 0.505 | 0.505 | 0.458 | 0.521 | 5.58 |
| A | 2 | cnn | 32 | 2718 | 0.492 | 0.492 | 0.492 | 0.492 | 3.87 |
| A | 3 | cnn | 32 | 2718 | 0.490 | 0.490 | 0.490 | 0.505 | 5.10 |
| A | 2 | nca | 32 | 1729 | 0.570 | 0.562 | 0.430 | 0.531 | 3.92 |
| A | 3 | nca | 32 | 1729 | 0.510 | 0.521 | 0.443 | 0.516 | 5.19 |
| A | 2 | nca | 32 | 2718 | 0.492 | 0.492 | 0.492 | 0.492 | 3.77 |
| A | 3 | nca | 32 | 2718 | 0.484 | 0.495 | 0.500 | 0.531 | 5.28 |
| A | 2 | vit | 32 | 1729 | 1.000 | 1.000 | 1.000 | 1.000 | 9.89 |
| A | 3 | vit | 32 | 1729 | 1.000 | 1.000 | 1.000 | 1.000 | 12.22 |
| A | 2 | vit | 32 | 2718 | 0.492 | 0.492 | 0.492 | 0.492 | 10.21 |
| A | 3 | vit | 32 | 2718 | 0.484 | 0.479 | 0.484 | 0.484 | 12.05 |
| A | 2 | constant | 32 | 1729 | 0.562 | 0.594 | 0.477 | 0.578 | 34.80 |
| A | 3 | constant | 32 | 1729 | 0.510 | 0.505 | 0.448 | 0.521 | 39.08 |
| A | 2 | constant | 32 | 2718 | 0.492 | 0.492 | 0.492 | 0.492 | 33.50 |
| A | 3 | constant | 32 | 2718 | 0.500 | 0.464 | 0.526 | 0.510 | 40.27 |
| A | 2 | learned | 32 | 1729 | 0.547 | 0.523 | 0.477 | 0.523 | 35.59 |
| A | 3 | learned | 32 | 1729 | 0.510 | 0.500 | 0.458 | 0.516 | 40.08 |
| A | 2 | learned | 32 | 2718 | 0.492 | 0.492 | 0.492 | 0.492 | 35.28 |
| A | 3 | learned | 32 | 2718 | 0.479 | 0.453 | 0.505 | 0.495 | 41.71 |

Scope: thin 2D grids and narrow 3D volumes, axes rotated; this does not qualify arbitrary curved 3D topology, natural vision, or a fused-kernel performance claim.
Failure within this frozen budget is a qualification failure, not an impossibility theorem. No threshold or hyperparameter rescue is performed.
