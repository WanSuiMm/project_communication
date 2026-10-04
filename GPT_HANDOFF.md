# Incremental data delivery: credit-locality formation audit

- Review base: `152b1902c3b597266f403bf780ca58754d29670b`.
- Evidence head: `605545952fec89737861bd900c3651febd43ded5`.
- This later handoff commit changes navigation only. Prior architecture claims
  remain unchanged; this release supplies code and saved data for independent
  analysis.

## Read first

1. [Results](evidence/credit_formation_20261004/RESULTS.md),
   [summary](evidence/credit_formation_20261004/summary.json),
   [configuration](evidence/credit_formation_20261004/config.json), and
   [saved-data checks](evidence/credit_formation_20261004/validation.json).
2. [Metrics](evidence/credit_formation_20261004/metrics.csv) and
   [figure](evidence/credit_formation_20261004/credit_formation.png).
   Per-update `uXXX/gradients/metadata.json`, `continuation.json` and
   `phenotype.json` preserve batch-wise results, cohorts, denominators and gates.
3. [Frozen protocol](new/credit_formation/PROTOCOL.md),
   [source map](GPT_CONTEXT.md), and
   [reproduction notes](evidence/credit_formation_20261004/REPRODUCTION.md).

## Recorded outcomes and scope

COMPLETE: 7/7 checkpoints (140,145,175,180,190,195,200), 138.5 seconds.
No model training or optimizer updates. Frozen verdict:
**CREDIT_SYNCHRONY_NOT_SUPPORTED**. The size32 behavior episode qualifies:
preservation/progress drop from0.9471/0.5007 at175 to0.4958/0.0432 at180,
then recover. At the collapse, overall cosine instead rises
from-0.0352 to0.2243 and C_parallel from-0.0019 to0.0122.
Overall/F/Q each fail the predeclared synchrony checks.

Native Full passes are **0/7** on these reused held-out banks; update200 misses
the strict T64 pooled-reach threshold (0.76648 versus0.80). Full qualification
is secondary and separate from the formation screen. Keep this failure and the
earlier checkpoint results on their original banks distinct.

All scientific JSON/CSV, the PNG/PDF figure and **65 NPZ files** are retained:
35 gradient archives,14 trace banks,14 endpoint-logit banks and2 input banks.
Validation recomputes the28 gradient pairs, mean vectors, continuation counts
and frozen screen without model inference. Machine records and checkpoints
remain local. Prior state-transfer evidence and architecture claims are unchanged.
This is one selected trajectory on reused banks; it does not establish causal
direction, population reliability, or a general credit-locality impossibility.

Reviewer questions: does the175->180 counterpattern support the proposed
necessary relation; how should later recovery and module-wise metrics be
interpreted alongside failed synchrony and native Full qualification?

[Publication manifest](CREDIT_FORMATION_PUBLICATION_MANIFEST.json)
binds the source and public data. Verify from repository root:

    python -X utf8 -B tools/export_credit_formation.py --verify-only
