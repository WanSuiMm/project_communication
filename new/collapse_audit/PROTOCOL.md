# Block7 formation/collapse: narrow continuation audit

Protocol `block7_collapse_continuation_v1`, frozen before the audit. This is a
zero-training intervention on one selected fresh training trajectory, not a
new reliability trial. The user authorized the audit only; no Hybrid training.

## Question and fixed material

Use block07/reset64x4 from `continuous_coverage_20261006_02`, checkpoints
u225,u250,u275,u300 (initialization96008, schedule97008). Keep the unchanged
C24/Z8 StreamingCell, historical local Torch2.5.1 FP32 backend and the SAME
32-map banks at sizes32/64 (seeds102032/102064). Load saved banks, not newly
sampled tasks. Bind checkpoint bytes, parameter hashes, banks, native traces
and scientific source to the published continuous-coverage manifest. Do not
load optimizer state into an optimizer, compute gradients or fit an adapter.

## Producer x consumer, 32 units

For each size, producer P in {225,250,275,300} cold-rolls both original and
source-flipped worlds to T64. Save its complete numerical (C,Z) state. For each
consumer C in the same set, clone those states, keep the inputs fixed and run
192 more steps with C's recurrent rule, to T256. No state reset at handoff.
Do not transplant parameters, moments or individual state components.

Record every integer time64..256 under three FIXED readout views: producer R_P,
consumer R_C, and common R_275. These are observation operators only: they never
change the recurrent state. The raw/native view is R_C; R_P and R_275 distinguish
immediate decoder changes from subsequent transition changes, without claiming
to eliminate representation-convention drift. Save original, flipped and AND
Boolean correctness traces for each view. Tie rule: logits>=0 predicts1.

Cohorts are always fixed by PRODUCER native paired correctness at T64, within
the source-flipped changed component. Report all_changed and strict16<d<32.
Primary diagnostics are continuous preservation (all times64..256 correct on
the producer-correct cohort), terminal retention, terminal acquisition and
sustained progress (all times241..256 correct on the producer-wrong cohort).
Also retain ever acquisition, handoff-only destruction/acquisition, coverage
at64/128/256 and integer numerators/denominators/map records. Undefined ratios
stay null. No selected-cell subset or new threshold is added after results.

Each of the eight size/checkpoint diagonals MUST reproduce the saved native
original/flipped/AND Boolean trajectory exactly, including the producer prefix
0..64 and suffix64..256. Report mismatch counts; stop on any mismatch. Confirm
finite numerical states/logits and unchanged model parameter hashes.

## Same-state single-step, 12 units

Take states from the u275 native trajectory at t64,128,192, in both worlds and
both sizes. Apply exactly ONE step of G275 or G300 to cloned states. Observe
both before/after with the fixed u275 readout. Report paired correct-to-wrong
destruction, wrong-to-correct acquisition and net coverage change, using the
same all_changed/strict selections. Save the before/after Boolean arrays and
the u275 state snapshots. The G275 step must match its saved native t+1 bits.
No channel, weight, Jacobian, gradient, width or latent-statistic audit.

## Decision and limits

Report continuous metrics for all16 matrix cells at each size, especially the
275/300 four-cell submatrix. No population p value: the independent training
unit is ONE selected trajectory, not cells, maps, readout views or checkpoints.
No old Full gate is recalculated and the previous u300 negative verdict stays.

If G300 damages u275 states under fixed R275, that is direct evidence of a
transition change on this native state cohort, not decoder drift alone. If
G275 recovers u300 states, that is recoverability under the old successful
consumer and this192-step budget. Neither observation uniquely assigns the
underlying mechanism to production, consumption or an invariant manifold.
Cross failure may be incompatibility; absence of one-step damage does not
exclude multi-step damage. A failed rescue does not prove irreversibility.

This audit guides the minimum Hybrid constraint; it does not prove bounded
coefficients imply semantic closure or require more mechanism searches.

## Execution and artifacts

Run one small CPU metric check and a two-map CUDA smoke before formal dispatch.
Measure smoke duration for an estimate; this is not a time cap. Use a persistent
foreground tool session to avoid the known detached-child lifetime problem.
Record PID/host/GPU/command locally. No monitor, watchdog or repeated status job.
Save new artifacts only in a new run directory, including this source snapshot,
provenance, eight T64 state packages and two u275 multi-time state packages,
32 packed matrix trajectories,12 packed
single-step records, per-map metrics, raw summaries and RESULTS.md. Preserve
ERROR/partial output on any nonfinite value or binding/replay failure.

Repository-root commands:

    python -X utf8 -B new/collapse_audit/check_metrics.py
    python -X utf8 -u -B new/collapse_audit/run.py --smoke --out analyses/NEW_COLLAPSE_SMOKE
    python -X utf8 -u -B new/collapse_audit/run.py --out runs/NEW_COLLAPSE_AUDIT
