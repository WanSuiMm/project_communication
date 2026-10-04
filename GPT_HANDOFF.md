# Incremental data delivery: continuation interface audit

- Review base: `d263afd147231c45b6c644665d75958f077b9522`.
- Evidence head: `a578946989ce91f15350736651c98ee41d4db236`.
- This later handoff commit changes navigation only. Prior architecture claims
  remain unchanged; this release supplies code and saved data for independent
  analysis.

## Read first

1. [Results](evidence/continuation_interface_20261004/RESULTS.md),
   [summary](evidence/continuation_interface_20261004/summary.json),
   [configuration](evidence/continuation_interface_20261004/config.json), and
   [delivery checks](evidence/continuation_interface_20261004/validation.json).
2. [Selection](evidence/continuation_interface_20261004/selection.json),
   [alignment qualification](evidence/continuation_interface_20261004/alignment/validation.json),
   [short-observable comparisons](evidence/continuation_interface_20261004/phase0/comparisons.json),
   [transfer decomposition](evidence/continuation_interface_20261004/cross/decomposition.json), and
   [gradient metadata](evidence/continuation_interface_20261004/gradients/metadata.json).
3. [Frozen protocol](new/continuation_interface/PROTOCOL.md),
   [source map](GPT_CONTEXT.md), and
   [reproduction notes](evidence/continuation_interface_20261004/REPRODUCTION.md).

## Recorded outcomes and scope

COMPLETE: 50/50 transfer cells, 439 seconds. Common native Full qualification
is true for S1/S2; the exact gauge control passes. No model training or optimizer
steps occurred. Raw success-success off-diagonal transfers both fail. Aligned
S2→S1 passes while S1→S2 fails. Raw S1→F1 and aligned S2→F2 also pass their
transfer proxies.

Restricted off-diagonal alignment qualification is **0/20**; matched
short-observable flags are **0/6**. These are separate from transfer proxy
passes. F2/F3 are nearest short-output candidates, not guaranteed matched
observables. No common algorithm, qualified invertible interface, causal
gradient explanation, or population reliability claim follows.

All scientific JSON/CSV and **274 NPZ files**, including 90 correctness trace
banks, are retained. Large arrays are secondary; begin with the summaries.
Original checkpoints, machine manifests and launch records remain local.
Saved checks and publication hashes were inspected without new model execution.

Reviewer questions: how should asymmetric transfer proxies be read alongside
failed restricted-map qualification; and what can be identified given the
short-observable matching flags and selected-checkpoint scope?

[Publication manifest](CONTINUATION_INTERFACE_PUBLICATION_MANIFEST.json)
binds the source and public data. Verify from repository root:

    python -X utf8 -B tools/export_continuation_interface.py --verify-only
