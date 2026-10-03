# Seed 4 dense transition audit evidence

This package describes one selected historical seed-4 training trajectory. It completed 300 training updates and 44 post-training behavior audits. The dense scan covers updates 100–200 every five updates for fixed paired map banks at sizes 32 and 64; update 300 is a separate control. The independent training unit is one trajectory. Maps, pixels, macro steps and checkpoints are not independent training replications.

## Result

The measured behavior improves gradually with substantial reversals, rather than following a smooth monotone curve. At updates 120 and 125, size-32 continuous survival64→256 is 0.9964 and 0.9970 while strict T128 coverage remains 0.1249 and 0.2409; retention is high while reach is low. Update 180 has a large, temporary regression after stronger profiles at 175. The trajectory recovers through 195–200, with a marked reduction in first-exit rate and a rise in finite-horizon first-entry survival from 195 to 200.

At update 175, size-32 pooled strict coverage is T128=0.7152, G64→128=0.2582, first-exit64→128=0.0610, and survival64→256=0.9248. At update 180 these are 0.1026, 0.0287, 0.5231, and 0.3566.

At update 195, size-32 first-entry survival through lag 64 is 0.8250 over 1326 eligible first-correct cells (17 late entries are right-censored). At update 200 it is 0.9531 over 1343 eligible cells (2 right-censored). This supports a sharper retention profile at the saved 200 checkpoint; it does not identify a latent commitment event.

Only update 200 passes the size-32 screen in the dense window. The separate update-300 control also passes; it is not adjacent to update 200. Dense checkpoints 205 and 210 were not measured, so the persistent three-checkpoint onset is NONE (persistence not established). This is not a failed candidate or failed architecture-gate verdict.

| Update | Size | Strict T64 | Strict T128 | Strict T256 | G64→128 | First exit64→128 | Survival64→256 | Net gain64→128 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 32 | 0.0007 | 0.0022 | 0.0007 | 0.0670 | 0.0333 | 0.9587 | 0.0520 |
| 115 | 32 | 0.0104 | 0.1123 | 0.2022 | 0.1630 | 0.0378 | 0.8621 | 0.0836 |
| 130 | 32 | 0.2164 | 0.3874 | 0.5331 | 0.1566 | 0.0528 | 0.8811 | 0.0489 |
| 160 | 32 | 0.3836 | 0.5933 | 0.7881 | 0.2103 | 0.0639 | 0.9109 | 0.0457 |
| 175 | 32 | 0.4773 | 0.7152 | 0.8260 | 0.2582 | 0.0610 | 0.9248 | 0.0603 |
| 180 | 32 | 0.1725 | 0.1026 | 0.0699 | 0.0287 | 0.5231 | 0.3566 | -0.0677 |
| 185 | 32 | 0.3784 | 0.5903 | 0.6461 | 0.1837 | 0.0652 | 0.9117 | 0.0476 |
| 195 | 32 | 0.5755 | 0.8565 | 0.9160 | 0.4037 | 0.0515 | 0.9453 | 0.0980 |
| 200 | 32 | 0.7063 | 0.9755 | 1.0000 | 0.5316 | 0.0003 | 0.9997 | 0.1273 |
| 300 | 32 | 0.8216 | 0.9978 | 1.0000 | 0.6143 | 0.0003 | 0.9997 | 0.1195 |

## Finite-horizon timing caveat

For size 32, the final-correct-conditioned median stable-minus-first-correct lag is already 0.0 macro steps at update 120, with 368 of 1345 primary-cohort cells final-correct. At update 180 the median remains 0.0 while only 94 of 1345 are final-correct on 1 eligible map; at update 200 it remains 0.0 with 1345 of 1345 final-correct. The conditioned median alone therefore cannot locate the update-200 change; its denominator and selection change sharply and do not establish commitment.

Map-level counts and denominators are retained in the CSV evidence so the displayed rates can be recomputed. These map and cell-step counts are descriptive; they do not support independent-cell confidence intervals.

## Evidence files

- [Canonical 44-row endpoint and transition profile table](profiles.csv)
- [Per-map strict endpoint coverage counts](strict_coverage_counts.csv)
- [Per-map all-changed transition counts](transition_counts_all_changed.csv)
- Per-map transition counts by distance: [source0](transition_counts_source0.csv), [1_16](transition_counts_1_16.csv), [17_31](transition_counts_17_31.csv), [32_63](transition_counts_32_63.csv), [64_inf](transition_counts_64_inf.csv)
- [Training loss and gradient norm by update](training_curve.csv)
- Distance and run-age loss hazard counts: [source0](age_hazard_source0.csv), [1_16](age_hazard_1_16.csv), [17_31](age_hazard_17_31.csv), [32_63](age_hazard_32_63.csv), [64_inf](age_hazard_64_inf.csv)
- Censoring-aware fixed-lag first-correct survival: [source0](fixed_lag_survival_source0.csv), [1_16](fixed_lag_survival_1_16.csv), [17_31](fixed_lag_survival_17_31.csv), [32_63](fixed_lag_survival_32_63.csv), [64_inf](fixed_lag_survival_64_inf.csv)
- Finite-horizon lag medians and conditioning counts: [source0](finite_trace_lag_medians_source0.csv), [1_16](finite_trace_lag_medians_1_16.csv), [17_31](finite_trace_lag_medians_17_31.csv), [32_63](finite_trace_lag_medians_32_63.csv), [64_inf](finite_trace_lag_medians_64_inf.csv)
- [44 compact per-checkpoint JSON summaries](checkpoints/)
- [Transition profile figure](transition_profiles.png)
- [Sanitized counter-validation result](validation/local_validation.json)
- [Reproduction and source binding notes](REPRODUCTION.md)

## Claim boundary

This is a descriptive audit of one trajectory, not a population estimate or causal intervention. Saved checkpoints locate observed behavior on this trajectory; they do not establish a physical phase transition, latent commitment, contextual type closure or generalization. Size 64 is a secondary projection. The historical gate is unchanged.
