# StreamingCell selected-checkpoint Full reevaluation v1

Question: do the two u300 Current models selected by the addressed-delta
Joint predicate also express the historical seed4 Full phenotype?
This is a zero-training, selected-checkpoint evaluation, not a fresh success-rate
estimate or an architecture comparison. The selecting cohort is reused explicitly.

## Frozen measurement plan

Parent: `runs/addressed_delta_20261008_02`, original StreamingCell C24/Z8,
5033 parameters. Primary checkpoints are block00/current/u300 (init140001,
schedule141001) and block02/current/u300 (init140003,schedule141003).
Block00/current/u275 is a separately labelled training-stage diagnostic and
cannot replace either primary endpoint.

Each of these three checkpoints is measured on two cohorts, for six units:

1. Historical **seed4-followup Full** cohort: 32 maps at each size32 and64,
   seeds50032/50064. Recreate with the frozen generator and require tensor hashes
   `a4f1e4d2dc55bbb5bacd9e916de767ce0582f3bb325c25147c0e0a165e92e446`
   and `8225e12ab6cf5087c92cd8d3b0032c541416b2efbce79ecbaef88f5386275cec`.
   These are not the earlier streaming-development seeds40032/40064.
2. Addressed selection cohort: the existing parent NPZ banks, seeds122032/122064;
   tensor hashes must match the parent manifest. This is not independent validation.

Checkpoint file, complete marker, stage record, metadata and parameter hashes
must agree before evaluation. Use the original `StreamingCell(streaming=True)`
with strict state-dict loading. No optimizer is constructed; parameters require
no gradients. Each unit starts cold and runs original/flipped worlds separately,
with all32 maps in each world and all integer times0..256 recorded.

## Unchanged evaluation

Reuse `new/continuous_coverage/evaluation.py::evaluate(full=True)` and its
Boolean trace recorder. Reuse, without edits,
`new/seed4_followup/phenotype.py::summarize_from_traces` and `predicate`.
The old Full threshold conjunction is the only Full verdict. Persist every gate
value and failure reason, losslessly packed original/flipped/paired Boolean
trajectories, per-map summaries and all matched-frontier strata.
Outside-cone source influence and nonfinite states are errors, not a Full result.

In particular, Full includes size32 T64 reach/BA, T128/T256 hold, and at both
sizes retention>=.95, all-changed coverage gain>=.05, ever-regression<=.15,
matched-frontier effect>=.05 with>=16 maps and>=100 strata. Do not reinterpret
endpoint retention as the stricter ever-regression condition.

Known before reevaluation: saved selecting-cohort metrics already exceed the
.15 ever-regression limit at block00/u300 size32 (.16) and block02/u300 size32
(.27270) and64 (.25885). These predict Full failures on that cohort; this run
completes the missing frontier measurements and tests the historical cohort.

## Execution and interpretation

Historical Torch2.5.1 backend flags, eager FP32 inference, no training, no
optimizer updates, no architecture changes, no threshold tuning, no time cap.
One short actual-shape paired-forward and bank/checkpoint/hash check precedes
the protected independent job. No recurring monitor. Each completed unit gets
a hash-bound marker; failures preserve partial artifacts and ERROR/INCOMPLETE.

Report four primary checkpoint/cohort verdicts and two diagnostic verdicts
separately. Even a selected-checkpoint Full PASS does not estimate formation
probability, prove a common hidden interface, or establish a mechanism.

From repository root:

```powershell
python -X utf8 -u -B new/streaming_full_reeval/run.py --check --out analyses/streaming_full_reeval_check_20261008_02.json
pwsh -NoProfile -File tools/start_protected_job.ps1 -JobName streaming_full_reeval_20261008_01 -Script new/streaming_full_reeval/run.py -ScriptArguments @('--out','runs/streaming_full_reeval_20261008_01','--qualification','analyses/streaming_full_reeval_check_20261008_02.json')
```
