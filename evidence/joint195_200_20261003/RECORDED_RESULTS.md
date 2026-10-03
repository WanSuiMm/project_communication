Publication note: the original generated report follows verbatim. Its `analysis.json`
reference denotes the excluded large local analysis tree; public replacements are
`factorial.json`, `case_index.json` and the compact `cases/` records. The derived
`DATA_COMPLETE` label predates the final `COMPLETE` status write. Start at `RESULTS.md`.

# Joint 195/200 Mechanism Audit

Execution status: **DATA_COMPLETE**; saved case records: **492 / 492 expected**.

This is a finite-horizon, zero-training intervention audit conditional on one selected training trajectory. The independent confirmation banks add map-set checks, not training-seed replication. Maps are the reporting units; pooled cells are descriptive.

## Saved endpoint metrics

Equal-map means are shown. See `analysis.json` for pooled counts, denominators, per-map values, all cases and uncertainty boundaries.

| Bank | Update | Strict coverage at 128 | Acquisition 64–128 | First exit 64–128 | Survival 64–256 |
|---|---:|---:|---:|---:|---:|
| confirmation32 | 195 | 0.984 | 0.859 | 0.000 | 1.000 |
| confirmation32 | 200 | 0.985 | 0.848 | 0.000 | 1.000 |
| confirmation64 | 195 | 0.962 | 0.413 | 0.008 | 0.992 |
| confirmation64 | 200 | 0.983 | 0.438 | 0.000 | 1.000 |
| primary32 | 195 | 0.908 | 0.581 | 0.060 | 0.936 |
| primary32 | 200 | 0.988 | 0.660 | 0.000 | 1.000 |
| primary64 | 195 | 0.989 | 0.438 | 0.015 | 0.985 |
| primary64 | 200 | 1.000 | 0.513 | 0.000 | 1.000 |

## Factorial block-order effects

Each block effect averages its marginal change over all 24 replacement orders. A metric is omitted from Shapley summaries unless all 16 unique coalitions are finite.

| Bank | Metric | E mean [min,max] | F mean [min,max] | Q mean [min,max] | R mean [min,max] |
|---|---|---:|---:|---:|---:|
| confirmation32 | strict128_equal_map | 0.125 [0.000, 0.250] | 0.000 [0.000, 0.000] | -0.124 [-0.249, 0.001] | 0.000 [0.000, 0.000] |
| confirmation32 | acquisition64_128_primary_equal_map | 0.184 [-0.003, 0.370] | -0.002 [-0.006, 0.000] | -0.193 [-0.381, -0.007] | -0.000 [-0.001, 0.000] |
| confirmation32 | first_exit64_128_primary_equal_map | -0.071 [-0.143, 0.000] | 0.000 [0.000, 0.000] | 0.071 [0.000, 0.143] | 0.000 [0.000, 0.000] |
| confirmation32 | survival64_256_primary_equal_map | 0.071 [0.000, 0.143] | 0.000 [0.000, 0.000] | -0.071 [-0.143, 0.000] | 0.000 [0.000, 0.000] |
| confirmation64 | strict128_equal_map | 0.173 [-0.000, 0.347] | 0.125 [-0.004, 0.256] | -0.277 [-0.580, 0.025] | 0.000 [0.000, 0.001] |
| confirmation64 | acquisition64_128_primary_equal_map | 0.063 [-0.003, 0.154] | 0.067 [0.012, 0.148] | -0.105 [-0.194, 0.013] | 0.000 [-0.001, 0.002] |
| confirmation64 | first_exit64_128_primary_equal_map | -0.187 [-0.375, 0.002] | -0.124 [-0.250, 0.003] | 0.303 [-0.010, 0.617] | -0.000 [-0.001, 0.000] |
| confirmation64 | survival64_256_primary_equal_map | 0.187 [-0.002, 0.375] | 0.123 [-0.005, 0.250] | -0.302 [-0.617, 0.012] | 0.000 [-0.000, 0.001] |
| primary32 | strict128_equal_map | 0.052 [0.001, 0.087] | 0.037 [0.002, 0.057] | -0.009 [-0.055, 0.030] | 0.000 [-0.001, 0.000] |
| primary32 | acquisition64_128_primary_equal_map | 0.057 [0.001, 0.107] | 0.037 [0.001, 0.068] | -0.015 [-0.064, 0.044] | 0.000 [-0.001, 0.002] |
| primary32 | first_exit64_128_primary_equal_map | -0.054 [-0.085, 0.000] | -0.039 [-0.060, -0.001] | 0.033 [0.000, 0.086] | 0.000 [-0.000, 0.001] |
| primary32 | survival64_256_primary_equal_map | 0.065 [0.000, 0.109] | 0.042 [-0.001, 0.064] | -0.044 [-0.110, -0.000] | 0.000 [-0.000, 0.001] |
| primary64 | strict128_equal_map | 0.174 [-0.002, 0.444] | 0.168 [-0.010, 0.441] | -0.331 [-0.731, 0.022] | 0.000 [-0.000, 0.001] |
| primary64 | acquisition64_128_primary_equal_map | 0.073 [-0.009, 0.156] | 0.076 [0.020, 0.133] | -0.075 [-0.206, 0.060] | 0.001 [-0.002, 0.004] |
| primary64 | first_exit64_128_primary_equal_map | -0.150 [-0.464, -0.000] | -0.171 [-0.512, 0.007] | 0.307 [-0.020, 0.716] | -0.000 [-0.001, -0.000] |
| primary64 | survival64_256_primary_equal_map | 0.156 [0.000, 0.467] | 0.166 [-0.013, 0.500] | -0.307 [-0.719, 0.027] | 0.000 [-0.000, 0.001] |

## Reading and claim boundary

The interpolation curves are finite-horizon comparisons in the historical shared parameter coordinates; they do not establish a bifurcation or a treatment threshold. Block Shapley effects are conditional on this checkpoint pair and selected maps. Cross-continuation preserves all producer × continuation × readout combinations. State swaps and local transplants may be off the learned trajectory; their effects support only conditional sensitivity statements.

For fixed solved/reference cohorts, `all_steps_correct` and `any_wrong_including_start` use all selected cells as the denominator, so an immediate break is retained. `first_exit` instead conditions on correctness at step 64. Localized target/downstream comparisons replay the baseline on each intervention's exact saved mask; empty cohorts remain present with null rates.

The confirmation banks test agreement across task maps only, not an independent training trajectory or treatment-threshold qualification. A correct output under a readout does not establish latent completion. No phase transition, universal mechanism, or general architectural claim is established.

`analysis.png` is a descriptive view; canonical values and all 24-order marginals are in `analysis.json`. Saved per-case JSON remains the source for full per-map denominators and intervention-specific records.
