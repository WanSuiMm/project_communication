# Incremental data delivery: output-preserving state factorization

- Review base: `9244d97c7f47900f1b8d5ed209f47e14528206ab`.
- Evidence head: `48e70514e1cded9b465cb6bcafa8573c3945134e`.
- This later commit updates navigation only. Prior experiments and claim
  boundaries remain unchanged; no model training or additional inference ran
  to prepare this publication.

## Read first

1. [Results](evidence/state_factorization_20261004/RESULTS.md),
   [compact summary](evidence/state_factorization_20261004/summary.json),
   [validation](evidence/state_factorization_20261004/validation.json), and
   [configuration](evidence/state_factorization_20261004/config.json).
2. [Metrics](evidence/state_factorization_20261004/metrics.csv) and
   [figure](evidence/state_factorization_20261004/state_factorization.png).
3. [Frozen protocol](new/state_factorization/PROTOCOL.md),
   [source map](GPT_CONTEXT.md), and
   [reproduction](evidence/state_factorization_20261004/REPRODUCTION.md).

Large raw_summary.json and per-arm JSON are secondary; they preserve the
original records, per-map counts, cohort denominators and all gate checks.
Do not start with those files or reread prior unchanged experiments.

## Recorded outcomes

COMPLETE16/16 cells in74.172 seconds. Fixed F1 consumer, native S1/F1 T64
producers, eight W/Z/span/null combinations, fresh32-map banks at sizes32/64.
Original/flipped cues define paired correctness. Cohorts are fixed before
intervention; instantaneous output and future preservation/progress are distinct.

Native Full is S1=True and F1=False. The primary episode is
**LOCALIZATION_EPISODE_UNQUALIFIED**, confirmation **UNQUALIFIED**. All16
rescue proxies remain unqualified: size32 shared-unsolved support is6maps
(231cells); size64 shared-solved support is9maps (393cells), below the frozen
minimum16 nonempty maps for BOTH keep/prog. This is a support failure, not a
claim that all continuous behavior failed. No gate or bank was changed.

On fixed shared-solved cells, SF continuous preservation is1.000 at both
sizes. On fixed shared-unsolved cells, SF sustained/delayed progress is.5152
at32 and.6730 at64; FF is0. Immediate SF output is exactly unchanged.
Null projection checks also pass. These remain finite, selected-checkpoint
measurements; they do not establish encoded task information, a W/Z
consistency relation, or population reliability.

All scientific data are retained: **66 NPZ files**,560 metric rows, full
arm/native JSON, original raw summaries and PNG/PDF figures. Verification
recomputes cohort masks, all16 count tables/gates, actual output neutrality,
FF/native suffix equality and source/bank/state hashes without model execution.
Model checkpoints, machine manifests, launch receipts and logs remain local.

Reviewer questions: how do the output-neutral W/null interventions affect
future behavior on fixed cohorts; what remains inferable when map support is
unqualified; which further intervention would distinguish state scale/control
effects from task-specific information? No new architecture claim is proposed.

[Publication manifest](STATE_FACTORIZATION_PUBLICATION_MANIFEST.json).
From repository root (NumPy only):

    python -X utf8 -B tools/export_state_factorization.py --verify-only
