# Historical update-zero geometry: diagnostic

COMPLETE: four exact initial checkpoints, four reused banks per seed; zero training; 30.31 CPU seconds.

Initial parameters match the original historical hashes. Tangent/autograd checks at full8/full16/detached16 pass;
maximum absolute error 4.77e-07.
No parameter changed. Source/first-training-batch/schedule bindings pass.

All initial state Jacobians are the same transport direct-sum identity by construction.
This diagnostic measures random task features, not that state Jacobian or a learned continuation quotient.

## Primary size32: fixed definitions

| Seed | Raw lane RMS max/min | Centered max/min | Mean absolute lane cosine | Initial Q-feature effective rank | K8@64 available pair separation | K8@64 relation-kernel effective rank | K8@64 label alignment |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2.426 | 1.535 | 0.232 | 2.115 | 0.1041 | 5.324 | 0.00594 |
| 3 | 1.678 | 1.189 | 0.131 | 2.016 | 0.1172 | 5.662 | 0.01374 |
| 4 | 1.198 | 1.256 | 0.269 | 2.113 | 0.1380 | 6.014 | 0.00470 |
| 5 | 1.513 | 1.220 | 0.262 | 2.259 | 0.1184 | 5.450 | 0.00658 |

Available pair separation is the per-map mean of ||H-H_flip|| divided by the average norm of H/H_flip,
restricted to targets with cue accessibility in that K8 window. It is not accuracy or an invariant information measure.
Bias-heavy features affect this relative normalization. Initial Q-feature covariance is centered over both worlds.
Relation kernel uses paired differences, so bias terms cancel.

## Bank consistency: K8 tangent at endpoint64

| Bank | Seed | Available pair separation | Effective rank | Nonzero eigenvalue condition | Label alignment |
|---|---:|---:|---:|---:|---:|
| bank_primary32 | 2 | 0.1041 | 5.324 | 1316.31 | 0.00594 |
| bank_primary32 | 3 | 0.1172 | 5.662 | 655.90 | 0.01374 |
| bank_primary32 | 4 | 0.1380 | 6.014 | 455.76 | 0.00470 |
| bank_primary32 | 5 | 0.1184 | 5.450 | 1002.70 | 0.00658 |
| bank_primary64 | 2 | 0.0759 | 5.593 | 2279.52 | 0.00227 |
| bank_primary64 | 3 | 0.0828 | 5.578 | 1545.17 | 0.00706 |
| bank_primary64 | 4 | 0.1056 | 5.746 | 1270.02 | 0.00214 |
| bank_primary64 | 5 | 0.0825 | 5.549 | 1960.95 | 0.00266 |
| bank_confirmation32 | 2 | 0.1086 | 4.703 | 2453.61 | 0.00644 |
| bank_confirmation32 | 3 | 0.1264 | 3.928 | 1740.91 | 0.01280 |
| bank_confirmation32 | 4 | 0.1439 | 4.651 | 1285.67 | 0.00378 |
| bank_confirmation32 | 5 | 0.1215 | 4.107 | 2444.10 | 0.00565 |
| bank_confirmation64 | 2 | 0.0776 | 5.372 | 3277.67 | 0.00260 |
| bank_confirmation64 | 3 | 0.0867 | 5.704 | 2029.30 | 0.00907 |
| bank_confirmation64 | 4 | 0.1090 | 6.116 | 1074.30 | 0.00256 |
| bank_confirmation64 | 5 | 0.0902 | 5.543 | 1920.96 | 0.00414 |

## First historical batch: original K8 loss gradients

| Seed | Q-out weight gradient L2 | First singular value | Second singular value |
|---:|---:|---:|---:|
| 2 | 0.00490129 | 0.00490129 | 1.02e-08 |
| 3 | 0.00868809 | 0.00868809 | 1.16e-08 |
| 4 | 0.00437302 | 0.00437302 | 1.11e-08 |
| 5 | 0.00469370 | 0.00469370 | 9.38e-09 |

Encoder, F, Q-in and readout.weight task gradients are exactly zero at initialization.
Q-out weight gradient is rank one up to FP32 roundoff; its output direction is the readout vector.
The balanced-loss bias gradients are near-zero numerical cancellation. This is not an AdamW update analysis.

## Interpretation and boundaries

- Seed4 has the most balanced RAW lane norms on all four banks, but seed3 or seed5 has better centered balance;
  seed4 does not have the smallest cross-lane correlation. Raw balance partly describes means/biases.
- Seed4 has the largest measured available relative separation at endpoint64, on all four banks,
  while initial Q-feature effective rank is highest for seed5. Favorable spectrum rankings depend on bank/window.
- Seed3 has the highest endpoint64 K8 paired kernel-label alignment on all four banks.
  Larger separation and higher effective rank do not entail easier target learning.
- Full-history tangents and actual K8 tangents differ; both are retained at all eight windows in summary.json.
- Four reused seeds and inspected banks cannot establish statistical anomaly, a successful-initialization criterion,
  causality, or reproducibility. Same seed4 initialization already failed the full phenotype under four new schedules.
- No source-flip feature difference occurs outside the explicitly computed cue-accessibility upper bound.
  Initial inability to distinguish a remote cue before arrival is expected locality, not representation failure.

Read [summary.json](summary.json) for every time/window/bank and per-map denominator. Per-map distance-stratum rows are in [per_map_bands.csv](per_map_bands.csv); full feature NPZs remain local.
Definitions and reproduction: [protocol](../../new/initial_geometry/PROTOCOL.md),
[audit code](../../new/initial_geometry/audit.py). No new architecture gate was run.
