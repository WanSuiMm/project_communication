# Incremental review: AU-NCA with free learned feedback

- Review base: `133526df4b3ab8ff0b8d4c0dbcb63961747d3d0e`.
- Stable code/evidence head: `fb291076adfb4fb2b01d57ec20fe1c75e9585c06`.
- This later commit changes review metadata only; the evidence head stays fixed.
- Execution: **COMPLETE**,9/9 arms,27,000 canonical optimizer updates.
- Frozen verdict: **AU_BENEFIT_DEVELOPMENTAL**; all3 K64 controls qualified.

Read [results](evidence/au_nca_20261009_01/RESULTS.md),
[45 final endpoints](evidence/au_nca_20261009_01/final_metrics.csv),
and [aggregate](evidence/au_nca_20261009_01/aggregate.json), then
[theory](new/au_nca/THEORY.md), [protocol](new/au_nca/PROTOCOL.md),
[cell](new/au_nca/cells.py) and
[qualification](evidence/au_nca_20261009_01/qualification.json).
Skip packed masks, raw predictions and training curves on the first read.

This extends the parameter-deferral experiments to an unrestricted learned
local feedback network, beyond the preceding fixed polynomial teacher family. Both realizations
have the same8,336 learned parameters and fixed-parameter forward function.
AU changes state storage16->145 scalars per cell and the K8 credit path of
the output projection W. Its upstream feature-parameter gradient still equals
ordinary K8 at matched parameters and trajectories; it does not recover
full credit for the whole system.

At fixed u3000, paired Original-K8/AU T64 NMSE values are
.230793/.129319,.248437/.141727,.174778/.202447. Mean NMSE falls from
.218003 to.157831, with gains in2/3 blocks. K64 mean NMSE is.001821.
The frozen developmental-benefit rule passes; the strong-recovery rule fails.
AU is worse in the third block and never reaches the primary.10 threshold.
The fixed target is one procedural flower, with a visible-state pool,
terminal T64 supervision and stochastic firing. Same index/firing schedules
do not imply identical learned pool contents or reset identities.

Same-state T256 mean NMSE is.344005 for Original K8 and.260740 for AU,
but both deteriorate relative to T64. Untrained damaged-T256 NMSE worsens
from.349973 to.436238. No uniform stability/regeneration or broad
short-BPTT solution follows from the primary benefit.

[Saved-data verification](evidence/au_nca_20261009_01/validation.json)
checks prediction metrics, canonical training updates, checkpoint metadata
bindings and source/input hashes without inference or training. The interrupted
prefix and one explicit u1175 recovery are preserved separately; updates
1176..1200 in the archived prefix were replayed, not double counted.
Aggregate elapsed time covers only the resumed process. Learned checkpoint
contents and private launch receipts remain local. Earlier claims/evidence
are unchanged; [reproduction](evidence/au_nca_20261009_01/REPRODUCTION.md)
provides qualification and fresh-run commands.

Reviewer questions:

1. Does the implementation preserve the original fixed-parameter forward
   function with stochastic firing, alive masks and a trainable output bias?
2. Is the W-only truncated-credit intervention correctly separated from the
   state-cost increase, divergent learned pool values and still-truncated eta?
3. Do the mixed third-block, continuation and damage results limit the
   conclusion to a developmental terminal-learning benefit on one target?

---

# Previous incremental review: spatial nonlinear lift

- Review base: `a470a02279ffb5aed827d22ee1acaed05c76fb0f`.
- Stable code/evidence head: `a764cf4c43b0d2a9c6fc92168e25afe7100a3fe4`.
- This later commit changes review metadata only; the evidence head stays fixed.
- Execution: **COMPLETE**,9/9 arms,2700 updates in137.859seconds.
- Frozen verdict: **SHORT_CREDIT_RECOVERY_DEVELOPMENTAL**.

Read [results](evidence/spatial_nonlinear_lift_20261009_01/RESULTS.md),
[36 final metrics](evidence/spatial_nonlinear_lift_20261009_01/final_metrics.csv),
and [aggregate](evidence/spatial_nonlinear_lift_20261009_01/aggregate.json), then
[proof](new/spatial_nonlinear_lift/THEORY.md),
[protocol](new/spatial_nonlinear_lift/PROTOCOL.md),
[cell](new/spatial_nonlinear_lift/cells.py), and
[qualification](evidence/spatial_nonlinear_lift_20261009_01/qualification.json).
Skip raw banks/predictions/training curves on the first read.

The gap between the supplied checks is now closed: genuine quadratic
dependence on historical state and cyclic2D transport coexist in one cell.
Original/full and lifted/short have exactly the same real-arithmetic forward
function and parameter gradients. At fixed u300, original K8 heldout R2 is
-.653966884/.027684190/.050469104; original K64 is
.999999727/.999999959/.999586778; lifted K8 matches within4.414e-10.
All three K64 controls qualify. Across all four final evaluation banks, the
maximum full/lifted R2 difference is1.476e-9.

All three arms use the same nine parameter coordinates, initialization and
minibatch schedule within each block. Lifted execution carries42 instead of2
scalars per cell; early input gradients still remain cut. The input map and
transport are fixed and the teacher is in the matched polynomial family.
Matching full/lifted is predicted by the theorem. The experimental evidence
is the large original K8 gap plus the measured state/systems cost, not a new
general feedback-learning guarantee. Three blocks are developmental, not a
population reliability estimate. Long128/256 are fresh proportionally phased
teacher-labeled episodes, not autonomous continuation-stability rollouts.

[Saved-array validation](evidence/spatial_nonlinear_lift_20261009_01/validation.json)
checks all final/intermediate predictions,2700 finite loss records, source/data
hashes and checkpoint bindings without inference or training. Exact compressed
banks/schedules and all curves remain public. Learned parameter/checkpoint
contents and private receipts remain local. Earlier evidence is unchanged.

Reviewer questions:

1. Does the quadratic lift preserve the same normalized pre-update state,
   including the distinct Q-transported M0, in both implementations?
2. Does the result isolate parameter placement while accounting for42 versus2
   state scalars, fixed features/transport, and a matched teacher?
3. What would be needed to retain a finite parameter-independent closure
   while introducing learned feedback or a broader task family?

---

# Previous incremental review: exact GCR lift and late projection

- Review base: `cd1b8efa353eb52111511bf47d94e0c897b60f60`.
- Stable code/evidence head: `bee767d5d89594bd25cb26c310b76231c32dae2c`.
- This later commit changes review metadata only; the evidence head stays fixed.
- Execution: **COMPLETE**, one new K8 arm,150 updates,20.144seconds.
- Frozen verdict: **NEAR_FULL_CREDIT_DEVELOPMENTAL**.

Read [results](evidence/reparam_gcr_20261009_01/RESULTS.md),
[seven final metric rows](evidence/reparam_gcr_20261009_01/final_metrics.csv),
and [aggregate](evidence/reparam_gcr_20261009_01/aggregate.json), then
[protocol](new/reparam_gcr/PROTOCOL.md),
[cell](new/reparam_gcr/cells.py), [runner](new/reparam_gcr/run.py),
and [qualification](evidence/reparam_gcr_20261009_01/qualification.json).
Skip raw banks, predictions and training curves on the first read.

New K8 held-out T64 R2=.535704758, versus locked old GCR K8=.260103259
and K64=.535750380. Same known banks/init/schedule,150 updates and5921
learned parameters. Fixed local raw moments are projected by live W at the
endpoint; per-cell state increases21->73. Forward/full-gradient equivalence
passes. New K8 versus old full gradient cosine is approximately1 and its
norm ratio matches the255/256 endpoint EMA factor. Early input gradients
remain zero: the change relocates shared parameter credit.

This is one developmental block for a linear degree-two task on reused
cohorts. The previous four-arm verdict remains POSITIVE_CONTROLS_UNQUALIFIED;
the original StreamingCell and historical evidence are unchanged. No fresh-seed
reliability, general nonlinear NCA, or autonomous stability claim follows.

[Saved-array verification](evidence/reparam_gcr_20261009_01/validation.json)
checks all150 losses, seven final metric rows, three checkpoint bindings and
source/data hashes without inference. The1.20MiB evidence package excludes
weights and private receipts. [Reproduction](evidence/reparam_gcr_20261009_01/REPRODUCTION.md)
explains the trained-reference checkpoint requirement and its regeneration.

Reviewer questions:

1. Does the state lift move the same learned computation into the last credit
   window, rather than add information or alter the exact forward function?
2. Which conclusion survives the21->73 state-cost increase and the limited
   linear degree-two task, and which would require fresh tasks or seeds?
3. Are the separate within-GCR result and the unchanged unqualified Full Writer
   comparison kept distinct?

---

# Previous incremental review: learnable ordered composition

- Review base: `bfa266cf97ab2118632dc5a49ae877519b7a79d0`.
- Stable code/evidence head: `58aa2720c39c57de565d4086d7fca3eeb80cb22e`.
- This later commit changes review metadata only; the evidence head stays fixed.
- Execution: **COMPLETE**,5/5 units,750 updates,94.179seconds.
- Frozen verdict: **POSITIVE_CONTROLS_UNQUALIFIED**.

Read [results](evidence/learnable_gcr_20261009_01/RESULTS.md),
[35 final metric rows](evidence/learnable_gcr_20261009_01/final_metrics.csv),
and [aggregate](evidence/learnable_gcr_20261009_01/aggregate.json), then
[protocol](new/learnable_gcr/PROTOCOL.md),
[cells](new/learnable_gcr/cells.py), [credit graph and verdict](new/learnable_gcr/run.py),
and [saved-array validation](evidence/learnable_gcr_20261009_01/validation.json).
Skip raw banks, predictions and training curves on the first read.

This update adds a new directed-path task and cell. It does not modify the
original StreamingCell, Region Identity or earlier frozen results. Endpoint
target is an ordered second-order interaction of Gaussian8D tokens divided
by path length, passed through tanh. Both primary models learn the same4D
encoder and use a20D workspace plus count. GCR accumulates literal first/second
prefix sums; Full Writer has a free vector residual head on the same graph.
Terminal T64 MSE only,150 updates, one paired initialization/schedule block;
u150 alone is primary. Train lengths16..32; long tests64/96.

| Primary held-out T64 R2 | K8 | K64 |
|---|---:|---:|
| Full Writer | -.001659 | -.001638 |
| Learnable GCR | .260103 | .535750 |

Both K64 controls had to reach.5. Full Writer did not, so the four-arm
architecture-by-credit test is unqualified. GCR K64 clears its own threshold,
but GCR's K64-minus-K8 gap is.275647, not negligible. The offline fixed raw8D
evidence MLP reaches.820608; it diagnoses a learnable readout from complete
statistics and is not a matched NCA. All saved predictions are finite through
the declared horizons. Parameter counts5921/GCR and6677/Full Writer are
reported; no capacity-matching claim is made.

The1.36MiB evidence package retains all750 losses,15 checkpoint bindings and
stage evaluations, exact banks/teacher/schedule, and all final predictions.
Saved-data verification recomputes all35 MSE/R2 rows without inference or
optimizer updates. Six executed-source hashes match the public code. Model
contents and private receipts remain local. A sandbox dispatch failure occurred
before training; the separate authorized launch completed normally, without
a training restart, and cleared idle-sleep prevention.

Reviewer questions: Which conclusions survive the Full Writer K64 control
failure? What does the within-GCR truncation gap establish on this single
task-matched block? What can the offline evidence diagnostic rule out about
readout learnability, without qualifying recurrent evidence formation? Do not
infer reliability, generic NCA expressivity or restored credit for early writes.

---

# Previous handoff: cue-once delayed credit

- Review base: `dd704eb2163b251896a8e99522d8f52edf4817a4`.
- Stable code/evidence head: `734ed9bcc8de0d0e1b710c1ddc2d031bf0740adb`.
- This later commit changes review metadata only; the evidence head stays fixed.
- Execution: **COMPLETE**,12/12 u300 trajectories,72 checkpoint bindings,
  3600 updates,1387.609seconds. Frozen verdict: **BASELINE_UNQUALIFIED**.

Read [results](evidence/delayed_credit_20261008_01/RESULTS.md) and
[endpoint metrics](evidence/delayed_credit_20261008_01/final_metrics.csv)
first, then [protocol](new/delayed_credit/PROTOCOL.md),
[credit cuts](new/delayed_credit/training.py), and
[saved-data validation](evidence/delayed_credit_20261008_01/validation.json).
Skip packed trajectories and full training curves on the first read.

The original5033-parameter StreamingCell gets cue-bearing inputs only at
initialization; all recurrent training/evaluation inputs are(mask,0,0).
Dense K8, terminal K8 and terminal K64 each have0/4 reach passes on fresh
maps. Mean size32 T64 strict pooled coverage is6.054%,0.259%,14.179%.
K64 exceeds terminal K8 in4/4 paired blocks descriptively; its best block
has45.571% pooled/53.853% mean-map coverage, below .80 qualification.
All24 trace banks are finite throughT256. Thus the K64 positive control
failed to qualify: no qualified short-credit bottleneck or unique cause
follows. B's encoder is disconnected, while F/Q train in the final8 steps.
This screen does not isolate an encoder-only effect.

The approximately25.1MiB public package includes all valid Boolean times,
768 per-map records, all training updates, exact banks/schedules and72
checkpoint bindings. Checkpoint contents and private receipts stay local.
Earlier Full results are unchanged; the deferred producer-consumer swap
was not run. No new architecture, sweep or publication-time inference.

Reviewer questions: What can the4/4 descriptive K64 advantage establish
without a qualified positive control? How does cue-once differ from the
earlier repeatedly available source cue? Keep learning, reach and long
execution stability separate; do not treat a gate failure as zero learning.

---

# Previous handoff: selected StreamingCell Full reevaluation

- Review base: `80b556a3010484707f682ffea150f48bfda91de7`.
- Stable code/evidence head: `7333c5f919a9cef89dce2750715134bddf0e59d2`.
- This later commit changes review metadata only; the evidence head stays fixed.
- Execution and saved-data verification: **COMPLETE**,6/6 units, zero training
  and optimizer updates,83.516seconds. No new architecture or threshold changes.

## Read this update first

1. [Results](evidence/streaming_full_reeval_20261008_01/RESULTS.md),
   [summary](evidence/streaming_full_reeval_20261008_01/summary.json), and
   [all144 gate values](evidence/streaming_full_reeval_20261008_01/full_gates.csv).
2. [Frozen protocol](new/streaming_full_reeval/PROTOCOL.md),
   [runner](new/streaming_full_reeval/run.py),
   [validation](evidence/streaming_full_reeval_20261008_01/validation.json), and
   [saved-data reproduction](evidence/streaming_full_reeval_20261008_01/REPRODUCTION.md).
3. Individual unit summaries only when comparing a specific failed gate.
   Skip packed NPZ traces and frontier CSV at first reading.

## New evidence and claim boundary

| Selected checkpoint | Historical Full cohort | Addressed selection cohort | Role |
|---|---|---|---|
| Current block00/u300 | PASS | FAIL | primary |
| Current block02/u300 | FAIL | FAIL | primary |
| Current block00/u275 | PASS | PASS | diagnostic |

Block00/u300's only selecting-cohort failure is size32 ever-regression .16>.15.
Block02/u300 fails that gate at both sizes in both cohorts; the historical size64
retention is also .9264164<.95. All frontier effect/support gates pass for all six
units. Diagnostic u275 cannot replace the fixed u300 endpoints.

This confirms one new selected u300 Full-success instance on the historical
seed4-followup maps (50032/50064), but neither selected u300 passes both cohorts.
The second cohort (122032/122064) is the data used to select the models by Joint;
this is not independent validation or a formation-rate estimate. The original
Current2/4 Joint result and candidate `NO_DEVELOPMENTAL_SIGNAL` remain unchanged.
This reevaluation separates Full from Joint rather than redefining either.

The approximately7.4MiB package retains all12 paired/original/flipped trace banks
at integer times0..256,384 per-map summaries,53,599 frontier strata and both exact
map cohorts. Export verification checks hashes, bank identity, Boolean conjunction,
trace-derived transitions/regression, CSV-derived frontier effects and the unchanged
Full predicate. No new model inference is used during publication verification.
Model contents and private process receipts stay local; their provenance hashes remain.

Reviewer questions: How should one new strict historical-cohort success be weighed
against the selecting-cohort regression failure? What does u275 passing both cohorts
and u300 losing one gate establish, without a causal intervention? Which conclusions
actually require cross-cohort Full versus Joint? Avoid inferring a common hidden
interface or mechanism from phenotype alone.

---

# Previous handoff: completed addressed-delta development screen

- Review base: `f080e69a79451fd6341e3b9fa3e5a23070d863b5`.
- Stable code/evidence head: `7a5ee01886c1a8d214625bb278217f4d10a4056a`.
- This later commit changes review metadata only; the evidence head stays fixed.
- Execution and aggregation: **COMPLETE**,12/12 u300 endpoints,156/156 stages.
- Frozen verdict: `NO_DEVELOPMENTAL_SIGNAL`.

## Read this update first

1. [Results](evidence/addressed_delta_20261008_02/RESULTS.md),
   [summary](evidence/addressed_delta_20261008_02/summary.json), and
   [formal metrics](evidence/addressed_delta_20261008_02/final_metrics.csv).
2. [All-stage metrics](evidence/addressed_delta_20261008_02/metrics.csv),
   [saved-data validation](evidence/addressed_delta_20261008_02/validation.json),
   [recovery](evidence/addressed_delta_20261008_02/interruption_and_recovery.json).
3. [Protocol](new/addressed_delta/PROTOCOL.md), [cells](new/addressed_delta/cells.py),
   [writer](new/addressed_delta/writer.py), [source map](ARCHITECTURE.md).
4. [Reproduction](evidence/addressed_delta_20261008_02/REPRODUCTION.md),
   [qualification](evidence/addressed_delta_20261008_02/runtime_qualification.json),
   [publication manifest](ADDRESSED_DELTA_PUBLICATION_MANIFEST.json).

Do not open the four raw block archives first. The approximately3.09MiB evidence
package retains all9,984 per-map metric rows,156 evaluations and completion
markers,12 training curves, banks and schedules, with lossless compression.
Checkpoint contents and private process metadata stay local. Full Boolean
trajectories were not persisted; old Full was not evaluated.

## New evidence and claim boundary

At fixed u300, Current is joint-ready2/4, Additive0/4, Delta0/4. Neither candidate
is ever-ready on the saved checkpoint grid. Primary Delta-minus-Additive has
0 wins,0 losses,4 ties and descriptive p=1; ties do not establish equivalence.
Size32 block-mean strict pooled T64 coverage is.5292/.0992/.1826; retention is
.9657/.0573/.1614 (Current/Additive/Delta). Four blocks are developmental only.

Candidates replace the entire carrier writer with matched initially identical
K/V heads: C'=U+.1K^T V versus C'=U+.1K^T(V-KU), K=A/sqrt(1+||A||_F^2).
They retain C24/Z8, E/Q/readout, transport, pre-stream perception and K8 training.
Current5033 and candidates5689 parameters are not capacity matched. The feedback
term includes structured damping; this is not a semantic duplicate proof or
a general failure theorem for addressed writing.

The cancelled parent retained Current u300 and Additive u200. After one failed
pre-training restart, the child inherited22 stages and resumed Additive u200
model/Adam/RNG with a JSON tuple/list compatibility fix. Numerical training source
was unchanged; helper source is separately bound. Recovered process time is
2570.515seconds. Nondeterminism prevents an uninterrupted bitwise-replay claim.
Qualification bound71 source files individually; the protocol's integrated
qualification-time ZIP was not generated. The public record states this deviation.

The saved-data exporter verifies all source/bank/archive hashes, records,
per-map count-to-aggregate agreement, gates, report tables and marker/curve
bindings, with no model inference or optimizer updates. CPU primitive/cell/report
checks pass; publication launched no new task training campaign.

Reviewer questions: What does the matched zero-success primary contrast establish,
given the positive contextual Current results? Can writer replacement be cleanly
distinguished from the feedback subtraction using these controls? Previous
port-relation and other frozen conclusions below remain unchanged; skip them
unless comparing experiments.

---

## Previous handoff: full-writer port-relation screen

- Review base: `d3a95be5230b48532bad51370c6d1d021e4e9775`.
- Stable code/evidence head: `196106541d7dae3c555cd141461109cd50817ccd`.
- This later commit updates review metadata only; the evidence head stays fixed.
- Execution/aggregation: **COMPLETE**,24/24 u300 trajectories,312/312 stages,
  624 size rows and624 complete packed Boolean banks; numerical failures0.
- Formal verdict: `NO_CONDITIONED_RELIABILITY_QUALIFICATION`.
- The separate earlier RRC-v0 partial snapshot remains INCOMPLETE; this update
  does not complete or reinterpret that experiment.

## Minimal reading order

1. [Results](evidence/port_relation_20261007_02/RESULTS.md),
   [summary](evidence/port_relation_20261007_02/summary.json), and
   [formal metrics](evidence/port_relation_20261007_02/final_metrics.csv).
2. [All stage metrics](evidence/port_relation_20261007_02/metrics.csv),
   [validation](evidence/port_relation_20261007_02/validation.json),
   [interruption/recovery](evidence/port_relation_20261007_02/interruption_and_recovery.json).
3. [Protocol](new/port_relation/PROTOCOL.md), [cell](new/port_relation/cells.py),
   [runner](new/port_relation/run.py), [reporter](new/port_relation/reporting.py).
4. [Reproduction](evidence/port_relation_20261007_02/REPRODUCTION.md),
   [engineering qualification](evidence/port_relation_20261007_02/runtime_qualification.json),
   [publication manifest](PORT_RELATION_PUBLICATION_MANIFEST.json).

Do not open the block ZIP archives or compressed dense records first. They
retain every recorded per-stage/per-map result, packed correctness trace,
relation-activity record, training curve and completion marker. Checkpoints and
private launch receipts stay local; checkpoint/source/bank/plan hashes remain.
The evidence package is about324.48MiB, compressed without dropping trace times.

## New decision-relevant evidence

Current/Constant/Conditioned are each0/8 joint-ready and0/8 old Full at u300.
Conditioned-current has8/8 pairs,0 wins,0 losses,net0,p=1; the all-tie contrast
does not establish equivalence. Size32 block-mean strict pooled T64 coverage is
.110424/.137410/.050948; retention is.548940/.429450/.475381, respectively.
Constant is ever-ready1/8 and loses readiness; the other arms are never ready
on the saved checkpoint grid. Intermediate success cannot select an endpoint.

Both candidate arms retain the historical full writer and add only a learned
port-matrix action:16 constant entries versus144 Z-conditioned coefficients,
initialized to zero. Counts5033/5049/5177 are not matched. Primary is only
Conditioned-current; other contrasts/dense stages/size64/old Full are secondary.
Runtime ID `port_relation_v1_full_writer_native_k8` retains the frozen protocol
header alias `port_relation_conditioned_v1`. See reproduction for the explicit
32-backward-window/one-optimizer-step implementation of the training wording.

The parent remains ERROR/INCOMPLETE after2 trajectories/33 stages: diagnostic
float32 squaring overflowed after finite state/logit checks. The repaired child
inherits33 stages, restores conditioned u150 model/Adam/RNG and replays unsaved
work. Qualified cached evaluation preserves full u175/u150 report dictionaries
and packed traces, including live parameter reload; eager/captured gradients
match exactly in the two-update qualification. Backend nondeterminism means
this is not a bitwise uninterrupted-replay claim. No new experiment was launched
for publication. Saved-data verification and CPU cell/activity/report checks pass.

## Reviewer questions and claim boundary

Does retaining the original full writer isolate the tested port-relation
addition sufficiently? What can be inferred from all arms failing the endpoint,
given low continuous reach/retention and a transient Constant success?
Does the preserved recovery bridge change interpretation beyond the stated
nondeterministic-replay limit? Review raw activity only for a specific question.

The frozen candidate failed this finite reliability screen. The evidence does
not establish a general relation-algebra failure, a unique formation mechanism,
a population success rate, or a solution to short-BPTT credit assignment.
