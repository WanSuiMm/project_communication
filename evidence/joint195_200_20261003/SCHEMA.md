# Compact public evidence semantics

[case_index.json](case_index.json) lists all492 conditions, groups, banks and exact case paths. [profiles.csv](profiles.csv) repeats four decision-relevant equal-map metrics for navigation. [schema_semantics.json](schema_semantics.json) centralizes the repeated scope, population, observation/counting notes and schema identifiers removed from individual records. All retained scientific values in cases, matched baseline cohorts, updates and plans passed an exact expand/strip round trip during export; the original JSON byte hashes are retained separately.

`per_map: {column_arrays: {map_index: [...], numerator: [...], denominator: [...], rate: [...]}}` encodes the original per-map dictionaries by columns. Columns have identical lengths; row j consists of each column's j-th entry. Empty cohorts give null rates and zero eligible-map counts. Column encoding does not round floats or alter counts. The small public verifier recomputes pooled totals and equal-map means and checks the all-order factorial identities. This verifies exported arithmetic; it is separate from offline validation against the excluded raw traces.

## Domains and statistical unit

Paired correctness requires the same cell to predict correctly in both original and source-flipped cue worlds. Margins are signed by the respective target, with `paired_min` taking the smaller signed margin. Primary domain is changed, open, non-source cells with BFS distance d>0. Strict domain additionally requires16<d<32. Walls and sources are excluded. Older dense-turnover reports included sources and also emphasized pooled values; compare matching domains and averaging units rather than treating old/new rates as interchangeable.

Each task map is averaged equally when its denominator is positive. Pooled cell rates are descriptive, and rates from distinct conditional cohorts have distinct denominators. There is one selected training trajectory,16 primary maps per size and8 independent confirmation maps per size; cells, time steps, checkpoints and intervention arms are not independent training replications.

## Time, risks and fixed cohorts

The saved suffix covers macro steps64..256 inclusive. Endpoints are64/128/256. Acquisition is endpoint wrong-at-start to correct-at-end conditional on being wrong initially. Endpoint destruction uses final wrong status; first exit counts any intervening wrong step among initial-correct cells. Continuous survival is its complementary event on that initial-correct cohort. Fixed-cohort `all_steps_correct` includes the starting step and therefore records immediate intervention damage instead of conditioning it away. Turnover can count repeated transitions and has a cell-time denominator; it is not the first-exit risk.

Cross-continuation includes both own initial-correct and common-solved-at64 cohorts. State/pulse cases include `matched_baseline_case` and `matched_baseline_cohorts`: the untreated baseline evaluated on the intervention's exact saved target/ring masks. Use these for localized effects, not a baseline-wide aggregate on a different population. Plans are correctness-conditioned, paired between cue worlds and sparse; they are not label-blind deployment algorithms.

## Update probes and scope limits

E denotes encoder, F workspace residual, Q local-state residual, and R the post-hoc linear readout. The original macro step streams W, applies F using original-clock spatial features, then updates Z using the new W; it can communicate two stencil hops. R never feeds back into the recurrent dynamics. Readout-relative null projection preserves immediate logits within tolerance; it need not preserve future dynamics. The semantic telescope specifies a mediation order. Tanh prevents additive attribution of Q preactivation blocks.

The frozen protocol uses the term compact for the original local summaries; their duplicated180MB summary and420MB analysis are excluded from this publication package. Public cases are minified and column encoded. Checkpoints, raw state/trace arrays, source snapshots and private machine receipts remain local. Original source/input/evidence hashes are retained in [input_provenance.json](input_provenance.json) and the [publication manifest](../../JOINT195_PUBLICATION_MANIFEST.json). [RECORDED_RESULTS.md](RECORDED_RESULTS.md) preserves the original generated report with a publication preface; its derived DATA_COMPLETE status predates the final COMPLETE write.
