# Continuous execution-state coverage

The frozen experiment completed 16/16 trajectories and 208/208 dense checkpoint evaluations. The formal endpoint is u300; all earlier checkpoints are formation diagnostics.

Both arms had joint-readiness 0/8 and old Full 0/8. The frozen paired reliability verdict is **NO_CONTINUOUS_COVERAGE_RELIABILITY_QUALIFICATION** (joint-success wins 0, losses 0, exact two-sided p=1.0). The independent unit is the paired training block (n=8).

At u300, continuous-minus-reset mean pooled strict reach R at T64 was -0.0709758255. Mean S retention from T64 to T256 changed by +0.1171953934; S was higher in 4/8 blocks. The S gain does not meet the frozen joint reliability criterion because neither arm passed readiness in any block.

Execution took 3439.86 seconds. Publication used no model fitting, training, or inference. The all-checkpoint R/S/survival checks use saved Boolean traces and evaluation banks. Final frontier and old Full values remain the saved u300 evaluator outputs; this export did not regenerate frontier strata.

See [REPRODUCTION.md](REPRODUCTION.md), [summary.json](summary.json), [dense.json](dense.json), [perarm.json](perarm.json), [metrics.csv](metrics.csv), and [validation.json](validation.json). Packed traces, checkpoint hashes, schedules, curves and full per-checkpoint summaries are retained as secondary data.
