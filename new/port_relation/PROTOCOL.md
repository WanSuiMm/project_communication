# State-conditioned port relation: paired formation screen

Protocol ID: `port_relation_conditioned_v1`  
Status: predeclared experiment specification; no training result is claimed here.

## Question and scope

Does a learned state-conditioned relation between the four carrier ports improve
reliable formation over the historical carrier writer, and does it outperform a
state-independent learned relation, under the same paired K8 training recipe?

The experiment has three arms with the historical carrier and recurrent update
as the shared base:

| Arm | Carrier relation | Parameters |
|---|---|---:|
| `current` | Historical writer; no added port relation | 5,033 |
| `constant` | One global, state-independent learned 4×4 matrix `K`, shared across cells and time; 16 unconstrained parameters initialized to zero | 5,049 |
| `conditioned` | A per-cell 4×4 matrix produced by a zero-initialized `Conv2d(8,16,1)` from `tanh(Z)`; 128 weights and 16 bias values | 5,177 |

The conditioned matrix is reshaped from 16 output channels to four output rows
and four input ports at each cell. Its convolution weights and bias both start
at zero; the bias is the constant component `Theta0`, so the conditioned arm
starts with `K(Z)=0` and does not add a second constant matrix. Neither relation
uses a softmax or a separate gate. The constant matrix is optimized directly
with AdamW; the conditioned matrix generator is optimized with the same
optimizer. Common encoder, writer, readout, and recurrent parameters are copied
identically within each paired block. Parameter counts differ; this is not a
parameter-matched comparison.

For all arms, first apply the existing pure masked port transport to obtain
`U = T_M(C)`. The historical writer uses the pre-stream neighborhood
`L_M(C)`. In the two relation arms, add the port relation to that writer:

```text
K_current = 0
K_constant = one learned global 4x4 matrix
K_conditioned(x,y) = reshape_4x4(Conv8_to_16(tanh(Z))(x,y))
C_next = U + 0.1 * [F_original(U, Z, L_M(C), L_M(Z), X) + K(Z) U]
Z_next = Z + 0.5 * Q_original(C_next, Z, L_M(C_next), L_M(Z), X)
```

`K(Z)U` contracts over the four-port axis at each cell, carrying each lane's six
payload channels independently. `F_original` receives the streamed carrier `U`
and the original pre-stream neighborhood features `L_M(C)` and `L_M(Z)`;
`Q_original` retains the historical update. No new spatial stream, post-stream
neighborhood feature, topology controller, regularizer, or Q redesign is
introduced. The constant matrix starts at zero, as does the conditioned map.
This intervention tests the specified candidate; it does not by itself prove a
causal or algebraic property of port relations generally.

## Paired blocks, data, and optimization

There are eight independent paired blocks. Block `b=0..7` uses initialization
seed `130001+b` and schedule seed `131001+b`; all three arms in a block share
their common initialization and minibatch schedule. Each arm trains on the same
512-map size-32 training bank (seed `10002`) and batch size 8. Reuse the existing
RRC training bank and data-generation implementation; do not regenerate or
modify frozen bank artifacts.

Each arm receives 300 optimizer super-updates using the existing reset64×4
procedure. A super-update reuses the same minibatch `X` for four cold 64-step
segments. Each segment has eight K8 windows; compute BCE at each window endpoint,
sum/divide the 32 window losses by 32, backpropagate once, clip global gradient
norm to 1, then take one AdamW step (learning rate 0.001, weight decay 0.0001,
betas `(0.9,0.999)`, epsilon `1e-8`). Parameters are fixed across the four
segments and all 32 windows. This is not full 256-step backpropagation.

Evaluate the same held-out banks for every block and arm: 32 size-32 maps from
seed `122032` and 32 size-64 maps from seed `122064`, using the existing bank
artifacts. Save and evaluate updates `0,25,...,300`. This yields 13 stages per
trajectory, 24 trajectories, and 312 expected stage records. Update 300 is the
sole formal endpoint. No best-checkpoint selection is allowed.

## Readiness and primary decision

Use the unchanged `continuous_coverage.metrics.joint_readiness(summary, size=32)`
definition. Size-32 joint readiness passes only when all of these hold:

1. Strict changed-pixel coverage at T64, with graph distance `16 < d < 32`, is
   at least 0.80 for both the pooled rate and mean per-map rate.
2. Retention from pixels correct at T64 through T256 is at least 0.95.
3. The retention reference includes at least 16 maps and at least 100 paired
   correct pixels.

Missing metrics and insufficient support fail readiness; a pass on an
intermediate stage cannot replace the update-300 result.

The sole primary contrast is conditioned minus current at update 300. Across the
eight paired blocks, let `w` be conditioned-ready/current-not-ready blocks and
`l` be current-ready/conditioned-not-ready blocks. Qualification requires both
`w-l >= 6` and the exact two-sided binomial sign-test p-value over the `w+l`
discordant blocks to be at most 0.05. Ties count toward paired readiness totals
but not the sign-test denominator. A 6-to-0 contrast has exact p=0.03125.
Qualification is a finite eight-block reliability screen, not population-level
or task-general proof.

`conditioned − constant` and `constant − current` are secondary descriptive
paired contrasts only. They do not qualify the primary outcome, receive no
multiple-testing-based promotion, and cannot substitute for it. The original
Full phenotype gate remains auxiliary. Size-64 distance/coverage and longer
dense R/S formation trajectories may be reported as diagnostics; they do not
select checkpoints or alter the primary decision.

A complete run that fails to learn the task or misses the primary gate is a
negative result for reliable formation of this candidate under this protocol.
It does not close the broader port-relation research direction. Incomplete
paired blocks or a missing predeclared stage remain `INCOMPLETE`, with observed
and missing records retained; partial execution is not a negative scientific
result.

## Reporting contract

`reporting.py` exposes `aggregate(final_records, dense_records,
expected_blocks=8)` and `report(out, final_records, dense_records, status={})` for
the runner. Rows carry `block`, `arm`, `update`, `metrics`, and `joint`, where
`joint` is the frozen size-32 readiness object with a Boolean `pass`. Formal rows
are the update-300 rows. Dense rows cover all 13 declared updates. The aggregate
uses the eight paired blocks as the independent statistical units, preserves
missing/duplicate/invalid-record counts, and returns `INCOMPLETE` unless all
three arms have all formal endpoints and the complete 312-record dense grid.

The report writes `RESULTS.md`, `final_metrics.csv` (24 formal rows × two sizes),
and `metrics.csv` (312 stage rows × two sizes). Formation summaries are
diagnostic and never choose a checkpoint. The report keeps execution status
separate from the scientific endpoint and preserves negative and incomplete
outcomes.

## Numerical failure and incomplete execution

The new `K(Z)U` term is not bounded by this protocol; a rollout may become
nonfinite before T256. Catch only an `E.evaluate` `FloatingPointError` whose
message is exactly `Nonfinite paired rollout` as a stage-level `NUMERICAL_FAILURE`.
Make one record for that predeclared evaluation attempt, write `failure.json`,
and continue with the remaining scheduled stages and blocks. Its compact metrics
use the `continuous-coverage-metrics-v1` schema with all rates and support values
null; at update 300 set `full_pass=false`. Record `joint.pass=false`,
`trace_complete=false`, and `numerical_failure=true`. This counts as non-ready
under the frozen readiness rule, but it is not a completed Boolean correctness
trace. Stage-record counts refer to scheduled evaluation attempts; the explicit
trace flags preserve whether a full bank trace was obtained.

Do not convert other exceptions into numerical failures. An `AssertionError`,
I/O error, CUDA error, or other engineering exception is `ERROR` and stops the
run. A nonfinite training loss, gradient, or update also stops as `ERROR`; if
update 300 was not reached, the experiment remains `INCOMPLETE` and no synthetic
u300 endpoint is created. These failures are not task-readiness negatives.

## Dispatch boundaries

The runner must record the block and schedule seeds, arm, update, checkpoint,
last loss, stage metrics, and formal endpoint marker in its run artifacts. Keep
the training and held-out bank provenance attached to the run. This protocol
sets no wall-clock cap, automatic retry policy, or recurring status tracking.
The experiment can be dispatched under its authorized scope; monitoring is a
separate request.
