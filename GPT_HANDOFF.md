# Incremental review: RRC-v0 code and interrupted partial evidence

- Review base: `71ad6ef6c83d7b792a5227874beeccedb899cd8a`.
- Evidence head: `8c5b11afca26cf6205686f51e79acf495af4e120`.
- Publication status: `PARTIAL_INTERRUPTED`; aggregation/verdict: `INCOMPLETE`.
- Snapshot contains10/24 u300 trajectories,137/312 complete checkpoint stages.
- Only3/8 primary paired blocks are complete. No final efficacy judgment.
- This handoff changes review metadata only; evidence head stays fixed.

## Read first

1. [Partial results](evidence/rrc_v0_20261007_partial01/RESULTS.md),
   [summary](evidence/rrc_v0_20261007_partial01/summary.json), and
   [endpoint table](evidence/rrc_v0_20261007_partial01/final_metrics.csv).
2. [Protocol](new/rrc_v0/PROTOCOL.md), [cell](new/rrc_v0/cells.py),
   [runner](new/rrc_v0/run.py), and [reporter](new/rrc_v0/reporting.py).
3. [Saved-data validation](evidence/rrc_v0_20261007_partial01/validation.json),
   [reproduction](evidence/rrc_v0_20261007_partial01/REPRODUCTION.md), and
   [publication bindings](RRC_V0_PARTIAL01_PUBLICATION_MANIFEST.json).

All274 fully recorded packed trace banks,137 evaluator summaries and relation
activity records, frozen curves and numeric data banks are secondary evidence.
Compressed JSON is lossless; do not open all raw files before the summary.
The partly evaluated u175 stage is excluded, while its checkpoint stays local.

## Changed code and observed evidence

Three arms: Current5033 parameters, Factorized4983, RRC4988.
Factorized and RRC share the same F31→96→6 across four six-dimensional lanes.
RRC learns four softmax relation logits and one sigmoid coefficient, initialized
at uniform pi and rho0.1. Pre-stream L(C), C24/Z8, whole-state two-hop clock,
encoder/Q/readout, data and K8/reset64x4/u300 remain fixed within this experiment.
Common E/Q/readout and the two shared F initializations match per block.

Actual-shape eager/CUDA Graph qualification passed all3 arms, three optimizer
updates each, with zero maximum gradient difference. Relation gradients/action
are exercised. No new inference or training was performed for publication.

Observed u300 joint readiness: Current1/4, Factorized0/3, RRC0/3.
Old Full is0/4,0/3,0/3 respectively. The sole primary RRC−Factorized test has
three complete pairs and zero observed discordances; final p/qualification
are unset. These incomplete counts cannot reject or establish the architecture.

Saved progress stopped advancing at13:19 local. The original tool session
is unavailable and original worker was not found. Its saved RUNNING field is
stale. The interruption cause is not established; no restart occurred as part
of this upload. Frozen evidence and local checkpoints are retained.

## Boundaries and reviewer questions

Previous HardClip/Hybrid negatives and earlier causal audits stay unchanged.
The relation map is a convex lane-permutation mixture, nonexpansive but generally
dissipative. Only pure stream is lossless. Q/Z remains a cross-lane bypass;
approximate parameter matching does not match function class or MACs.
Do not claim recurrent stability, payload-basis invariance of the whole cell,
or reliable formation from this incomplete upload.

Review whether the implementation matches the intended pre-stream intervention,
and keep incomplete endpoints separate from dense formation diagnostics.
A final result requires the remaining14 trajectories and175 checkpoint stages.

    python -X utf8 -B tools/export_rrc_v0.py --verify-only
