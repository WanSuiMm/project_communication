# Direct Spatial Carry: completed negative development screen

All8 arms completed300 updates in797.797s (13.30 minutes). **DEVELOPMENT_NO_GO**.
Baseline reaches and holds in2/4; carry in0/4. All four baseline final parameter hashes and complete evaluation records exactly reproduce Phase II.

The sole change is W -> W-0.5*D_M^dagger*L_M(W) in the W skip path. This is fixed lazy diffusion over the correct masked medium.
F/Q, additive Z,5033 parameters, K8 training,300 updates, initialization and data/schedule are matched.

## Primary: size32/T64, strict16<d<32

| Seed | Baseline mean / pooled % | Carry mean / pooled % | Pooled change pp |
|---|---:|---:|---:|
| 2 | 93.70 / 92.97 | 0.00 / 0.00 | -92.97 |
| 3 | 21.85 / 17.27 | 2.87 / 3.12 | -14.15 |
| 4 | 52.40 / 44.76 | 14.39 / 11.41 | -33.34 |
| 5 | 96.45 / 95.68 | 7.59 / 4.42 | -91.26 |

Every seed loses both primary statistics. Carry does not rescue seeds3/4 and loses the previous successes2/5.
Carry seed2 has hold=True only because its already-zero primary score does not decline; it never reaches. Hold alone is not task success.
Whole-grid BA can remain relatively high while paired correctness on far changed-component pixels fails; these are different populations and requirements.

## Farther propagation on size64

| Seed | T | Baseline d>32 mean / pooled % | Carry d>32 mean / pooled % |
|---|---:|---:|---:|
| 2 | 64 | 12.68 / 5.07 | 0.00 / 0.00 |
| 2 | 128 | 31.71 / 26.17 | 0.00 / 0.00 |
| 2 | 256 | 38.51 / 36.78 | 0.00 / 0.00 |
| 3 | 64 | 0.00 / 0.00 | 0.00 / 0.00 |
| 3 | 128 | 0.00 / 0.00 | 0.00 / 0.00 |
| 3 | 256 | 0.00 / 0.00 | 0.00 / 0.00 |
| 4 | 64 | 0.00 / 0.00 | 0.00 / 0.00 |
| 4 | 128 | 0.00 / 0.00 | 0.95 / 0.70 |
| 4 | 256 | 0.00 / 0.00 | 0.00 / 0.00 |
| 5 | 64 | 20.80 / 9.86 | 0.00 / 0.00 |
| 5 | 128 | 21.61 / 10.73 | 0.00 / 0.00 |
| 5 | 256 | 25.67 / 14.83 | 0.00 / 0.00 |

All distance/size/horizon results, including isolated positive contrasts, are retained in [curves](curves.csv) and [paired effects](paired_effects.csv).
Empty bands remain null in JSON (blank in CSV); size32 has no changed-component pixels at d>=64.

## Interpretation and scope

- This rejects this rho=0.5 normalized-average W-carry recipe under the frozen K8 development protocol; do not advance it to confirmation.
- It does not reject all spatial carry, directional transport, cellular computation or short BPTT.
- The experiment replaces part of local identity retention with neighbor mixing. It does not isolate smoothing, signal dilution, cancellation-learning burden or another failure mechanism.
- Channel coordinates are preserved by the fixed operator, but task information need not be preserved. Bounded carry amplitude is not a guarantee for the whole recurrent cell.
- Seeds and evaluation maps were already inspected. n=4 is conditional on one training bank/schedule; no significance or cross-distribution claim.
- No new training, checkpoint inference, sweep, confirmation or monitoring was performed for this publication.

## Systems and verification

| Seed | Variant | Training seconds | Peak allocated MiB | Clipped updates % |
|---|---|---:|---:|---:|
| 2 | baseline | 92.53 | 95.12 | 4.67 |
| 3 | baseline | 87.86 | 95.12 | 6.00 |
| 4 | baseline | 91.78 | 95.12 | 1.67 |
| 5 | baseline | 97.75 | 95.12 | 1.67 |
| 2 | carry | 101.53 | 96.08 | 26.33 |
| 3 | carry | 103.22 | 96.08 | 17.00 |
| 4 | carry | 98.20 | 96.08 | 19.00 |
| 5 | carry | 93.36 | 96.08 | 22.67 |

Verified29 source snapshots,8 checkpoints, all initialization/data/schedule bindings,96 BA and672 paired aggregates/denominators,8 arm decisions,312 paired effects and936 CSV rows.
This checks saved-result arithmetic and provenance; it does not regenerate logits or independently rescore pixels from weights.

[Compact analysis](analysis.json), [validation](validation.json), [raw arm records](raw/).
