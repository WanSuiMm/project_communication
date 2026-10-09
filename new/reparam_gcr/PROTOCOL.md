# Exact raw-moment lift and late projection: developmental v0

Status: frozen before scientific training. One new K8 arm, one paired block.
Historical source and evidence remain read-only. No seed/threshold sweep.

## Question and intervention

Can moving the shared linear projection after fixed local moment composition
recover the previous GCR-K64 learning result with an eight-step gradient graph?
The reference is the frozen GCR evidence at commit
`58aa2720c39c57de565d4086d7fca3eeb80cb22e`, directory
`evidence/learnable_gcr_20261009_01`. Old K8/K64 held-out R2 values are
.2601032593544145/.5357503802662453. The earlier four-arm verdict stays
`POSITIVE_CONTROLS_UNQUALIFIED`; this new within-GCR test does not require the
failed Full Writer comparator to qualify. References are locked before training.

At every cell, from zero initial state and the same predecessor graph:

```math
n'=n_{pred}+1,\quad U'=U_{pred}+x,\quad
V'=V_{pred}+U_{pred}x^T.
```

At the endpoint, each step forms `S1=WU`, `S2=WVW^T`, then uses the same
log-count, workspace/count normalization, interpreter and rho=.5 EMA as old
GCR. In exact arithmetic this commutes with old projected moment composition
at every node and time, with the same full-BPTT parameter gradient. This is a
larger raw-state realization, not a bijective coordinate transformation.
Per-node state rises21->73 scalars; trainable parameters stay5921 with identical
names and initialization. Only the endpoint is projected/decoded, since other
decoder outputs never influence any future supervised state. Raw moments
are produced by local recurrence during rollout, never supplied offline.

## Frozen training and data

Reuse the exact saved banks/teacher/schedule from the reference evidence, with
SHA256 validation and byte copies into a fresh run. No regeneration or new maps.
Train512 paths on grid8, lengths16..32; held-out128 with the same lengths;
long tests128 each on grid16, lengths64/96. Teacher/data/init/schedule seeds
and parameters match the previous protocol. Init170101, schedule170201.

150 updates, batch16, terminal T64 MSE only. AdamW lr=.001, wd=.0001,
betas(.9,.999), eps1e-8; gradient norm clip1. One optimizer step per rollout.
Run steps1..56 under no_grad, detach `(n,U,V,z)`, then differentiate57..64.
The projection W and interpreter are used inside those final eight steps.
No gradient crosses the earlier temporal graph. Access to detached prefix
values by a downstream parameter is the intended architecture intervention.

No AMP, compile, CUDA Graph, long-credit training, auxiliary loss, learning-rate
change, new initialization or recurring monitor. Named checkpoints at50/100/150;
only u150 is primary. Each completed update also has an atomic recovery checkpoint.
Explicit resume requires identical source/config/data bindings; no auto-restart.

## One bounded pre-launch check

Check initial parameters and the saved old GCR-K64/u150 parameters on the same
fixed train examples, without optimizing those references. CPU float64 compares
every step's projected per-node state, endpoint output and loss (atol/rtol1e-9),
and all full64 parameter gradients (relative L2 error<=1e-8).

For the lifted model on these train lengths, compare its K8 parameter gradient
with old full64. Late raw evidence is stationary and parameter-independent, so
the expected ratio is255/256, with a tiny earlier EMA transient. Require
cosine>=.999999 and relative error against `(255/256)*old_full_grad`<=1e-7,
overall and separately for W/interpreter. The bound is conditional on these
checks and this fixed task; coefficient mass alone is not a universal norm bound.

Also check CUDA FP32 at actual batch/grid shape: forward tolerance(atol1e-5,
rtol1e-4), full-gradient relative error<=1e-4, new-short/full cosine>=.99999
and scaled relative error<=1e-4. New K8 must have exactly zero gradient to
tokens more than eight path positions from the endpoint, but nonzero W and
suffix-input gradients. One new-model optimizer smoke update measures time
and memory. Failure stops dispatch; no relaxing tolerances to rescue a result.

## Endpoints and frozen decisions

Primary: held-out T64 R2 at u150, compared with both locked references.
Secondary: held-out T128/T256, long64/96 T128/T256; all per-example predictions,
counts, MSE/R2, complete training curve, timings and memory are saved.

Let `r` be new K8 primary R2, `s`=.5357503802662453 and `b`=.2601032593544145.
The reference qualifies because s>=.5. Prespecified descriptive thresholds:

- `NEAR_FULL_CREDIT_DEVELOPMENTAL`: r>=.5 and abs(r-s)<=.02.
- `ABOVE_FULL_CREDIT_REFERENCE_DEVELOPMENTAL`: r>s+.02.
- `PARTIAL_DEVELOPMENTAL_SIGNAL`: otherwise r-b>=.10.
- Otherwise `NO_DEVELOPMENTAL_SIGNAL`.

Report continuous differences and recovered fraction `(r-b)/(s-b)` regardless
of category. The .02/.10 tolerances are design choices, not derived constants.
Numerical/execution failures are incomplete, never a qualified negative.
No intermediate checkpoint selection. No reliability estimate from one block.

Passing supports late parameter placement with larger sufficient-statistic
state on the current linear, degree-two task. It does not recover gradients to
early input occurrences, solve generic learned recurrence, establish arbitrary
nonlinear composition, or prove long autonomous NCA stability. Evaluation
reuses known cohorts; it is not an independent confirmation study.

## Reproduction

From repository root in the declared Torch2.5.1 CUDA environment:

```powershell
python -X utf8 -u -B new/reparam_gcr/run.py --check --out analyses/NEW_REPARAM_CHECK
& ./tools/start_protected_job.ps1 -JobName NEW_REPARAM_RUN -Script new/reparam_gcr/run.py -ScriptArguments @('--out','runs/NEW_REPARAM_RUN','--qualification','analyses/NEW_REPARAM_CHECK/qualification.json')
```

Use fresh names. The existing independent on-demand protected worker has no
runtime cap and temporarily prevents idle sleep; it installs no recurring
trigger. Private receipts and checkpoint contents remain local.
