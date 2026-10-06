# Incremental review: block7 formation/collapse continuation audit

- Review base: `ebff366c6cc04c2c332b9764f8bf167b104fac12`.
- Evidence head: `8924d884c4b7a9a3cca68a713cfbc84e73e25359`.
- This later commit changes only the handoff. The evidence head remains stable.
  No training or inference is performed during publication.

## Read first

1. [Results](evidence/block7_collapse_20261006/RESULTS.md),
   [summary](evidence/block7_collapse_20261006/summary.json),
   [saved-data validation](evidence/block7_collapse_20261006/validation.json).
2. [Matrix summaries](evidence/block7_collapse_20261006/matrix_summary.csv),
   [single-step summaries](evidence/block7_collapse_20261006/single_step_summary.csv).
3. [Protocol](new/collapse_audit/PROTOCOL.md),
   [code map](GPT_CONTEXT.md),
   [reproduction and trace layouts](evidence/block7_collapse_20261006/REPRODUCTION.md).

The [full matrix](evidence/block7_collapse_20261006/matrix.json.gz),
[single-step records](evidence/block7_collapse_20261006/single_step.json.gz),
[provenance](evidence/block7_collapse_20261006/provenance.json), packed
trajectories and numeric state arrays are secondary; read the summaries first.

## New evidence

The prior dense checkpoint grid contains joint-ready block7/reset points
u250/u275, followed by failed u300. This selected zero-training audit studies
that trajectory only: init96008, schedule97008; producer and consumer
checkpoints225/250/275/300, same32-map banks at sizes32/64. It hands off the
complete numerical(C,Z) state atT64 and continues192 steps toT256.

COMPLETE32/32 matrix and12/12 single-step units. Eight native diagonal
trajectories reproduce saved original/flipped/AND Boolean bits exactly.
Parameters are unchanged; no optimizer update or state-coordinate fit.
Three fixed observation views R_P/R_C/R275 do not feed back into dynamics.
Correct/wrong cohorts are fixed by the producer's native paired correctness
atT64. This is ONE independent selected training trajectory, not44 trials.

Fixed R275, all-changed paired coverage atT256:

|Producer atT64|Consumer rule|size32|size64|
|---:|---:|---:|---:|
|275|275|.950144|.840536|
|275|300|.970374|.933684|
|300|275|.489719|.111351|
|300|300|.425492|.059384|

G300 continuously preserves u275's T64-correct cells at.999731/.999708
and has more sustained progress on u275's initially wrong cells. Successful
u225/u250 states also continue well under G300. Single-step destruction under
fixed R275 is below.0003 on all u275 states sampled at64/128/192 and both sizes.
G275 improves u300-produced states only partly; it does not restore the
successful regime.

The tested native collapse cannot be adequately described as G300 losing the
ability to continue an already-successful state. The stronger distinction is
the state supplied by the cold0..64 prefix. That prefix contains both encoder
and recurrent updates: no unique module, state component, irreversible domain
exit, or low-complexity contract is identified.

## Unchanged claims

The prior fixed-u300 continuous-coverage reliability verdict remains
`NO_CONTINUOUS_COVERAGE_RELIABILITY_QUALIFICATION`, with0/8 joint readiness
and0/8 old Full in both arms. Intermediate checkpoints do not replace the
formal endpoint. No new Full gate, population success rate, phase-transition
theorem or Hybrid architecture result is claimed. No new training was done.

## Data and verification

All32 packed matrix trajectories retain three fixed readout views and both
source worlds at every integer time64..256. All12 single-step records and10
numeric state packages are retained losslessly. Per-map numerators and
denominators are in the raw gzip JSON; the CSV summaries retain the full
view/cohort matrix. The previous publication already contains the unchanged
evaluation banks and native trajectories. Checkpoint bytes remain local;
their hashes and input/source bindings are published. Machine/session receipts
are excluded. Saved public-data verification needs no checkpoints or GPU:

    python -X utf8 -B tools/export_block7_collapse.py --verify-only

## Reviewer questions

1. Does the producer-associated pattern remain consistent across all three
   readout views, after separating immediate readout-switch changes?
2. Given successful continuation by G300, what minimal execution-state
   construction/write constraint should Hybrid v0 test, without attributing
   this result uniquely to the encoder or encoding the task answer by hand?
