# Historical seed4 credit-locality formation audit v1

Protocol: credit_locality_formation_v1. Frozen before new gradient or
continuation outcomes. Zero training, no optimizer, no architecture changes,
no new seeds, no time cap, watchdog, or continuing monitor.

## Question and inputs

Do short-credit fidelity and continuation preservation/progress collapse and
recover together within ONE historical training trajectory? This is a
descriptive synchrony test, not an intervention on training or proof of either
causal direction. Fixed optimizer settings do not exclude optimizer-dependent
effects across training stages. The selected checkpoints are not independent
training replications.

Use existing original StreamingCell checkpoints at updates
140,145,175,180,190,195,200 from the exact replay in
runs/transition_20261003_seed4_dense01. Initialization4, training-bank seed10002,
512 size32 maps, schedule20002, original K8 training. Bind every checkpoint
file and state-dict hash to training.json and its prior publication manifest;
require the archived exact replay qualification. Never regenerate checkpoints.

Reuse the published held-out banks from continuation_interface_20261004:
test32 seed94032,32maps and test64 seed94064,32maps. Their previously checked
training-bank exclusion and file/data hashes are rechecked. No sample, batch,
checkpoint, loss, or module is chosen after new results. These banks were used
in a previous audit; this is not a new independent confirmation dataset.

Gradient batches are the four ordered size32 groups [0:8],[8:16],[16:24],[24:32],
original cues only. The original balanced BCE is averaged over maps and loss
times8,16,...,64. K8 and full64 use the same numerical state trajectory,
parameters, input and scalar objective; only detach boundaries differ. Reuse
the unchanged historical backward_trajectory. Compare state SHA256 at every
step0..64 and forward loss (tolerance1e-6); no clipping, optimizer, or updates.

## Credit metrics

For overall and E=encoder,F=f_in/f_out,Q=q_in/q_out,R=readout:

    cosine = dot(g8,g64)/(norm(g8)*norm(g64))
    C_parallel = dot(g8,g64)/norm(g64)^2
    C_miss = norm(g64-g8)/norm(g64)

Also report both norms, norm ratio, zero flags, and the recurrent aggregate
excluding R. Primary values are computed from the mean of FOUR gradient
vectors, not the mean of four metric values. All batch-wise vectors and
metrics are retained, so a single batch's influence can be inspected.
C_parallel is a signed projection coefficient, not an information percentage;
it may exceed1. C_parallel=1 alone does not imply vector fidelity. Literal
near-equality is a separate descriptive flag: cosine>=.90 and C_miss<=.10.
Undefined zero-norm metrics are null and never count as a pass.

## Continuation and synchrony screen

Each checkpoint receives the same paired original/flipped bank and fresh state.
Rollout0..256 with the unchanged macro clock. Preserve all Boolean traces,
T64/128/256 logits/task scores and the existing Full predicate as secondary.
Primary continuation values use size32 all_changed cells:
preservation=correct at EVERY step64..256 given paired-correct at64;
sustained_progress=correct at EVERY step241..256 given paired-wrong at64.
Each checkpoint defines its own T64 cohort; cohort sizes/coverage are reported
to expose changes in conditioning. Integer counts, pooled and equal-map rates
are retained. Size64 is a secondary scale comparison, not a rescue gate.

Primary key sequence:175,180,195,200. Context140,145,190 is always reported.
The behavior episode qualifies if BOTH preservation and sustained_progress
drop by>=.10 at175->180, both rise by>=.10 at180->195, and neither drops by
more than.02 at195->200. If it does not reproduce on these banks, report
PHENOTYPE_UNQUALIFIED; do not infer a credit null from absent behavior contrast.

For a scope to synchronize, cosine must drop>=.10 and C_parallel>=.05 at
175->180; both must rise by those amounts at180->195; and at195->200 neither
may drop by more than.05 cosine or.02 C_parallel. Fixed outcomes:

- STRONG_SYNCHRONY: qualified behavior and overall credit satisfy all checks.
- MODULE_SYNCHRONY: behavior qualifies, overall fails, but predeclared F or Q
  satisfies all checks. This is a secondary module-specific signal, not
  evidence that the module alone causes failure.
- CREDIT_SYNCHRONY_NOT_SUPPORTED: behavior qualifies but overall/F/Q fail.
- PHENOTYPE_UNQUALIFIED: the behavior episode does not qualify.

These thresholds define this finite screen, not universal physical critical
points. All continuous deltas and C_miss are reported even if a gate fails.
No p-value treats checkpoints/cells as independent training runs. E/R and
recurrent results are descriptive, not alternate routes to a positive verdict.
The main figure uses true checkpoint coordinates, size32 preservation/progress
and overall cosine, with E/F/Q/R panels; all values and size64 data remain in
CSV/JSON. A passing screen establishes synchronous association, not
"continuation validity makes credit local" or an explanation of1/16 success.

## Qualification, outputs and stopping

One focused CPU vector check (including orthogonal error at C_parallel=1 and
zero norms), a synthetic synchrony gate fixture, and one frozen-checkpoint
CUDA smoke: K8/full16 exact forward-state agreement plus paired256-step
continuation on two maps at both sizes. Require source/checkpoint/bank hashes,
Torch2.5.1 FP32,threads2,cuDNN benchmark/deterministicFalse,cuDNN TF32True,
matmul TF32False, and at least3500MiB free GPU memory before launch.
Stop and keep failed outputs on source/reference drift, nonfinite values,
forward mismatch, or parameter changes. No rescue sweep.

New run directories only. Save source snapshots, configuration, checkpoint
bindings, per-checkpoint status, per-batch and mean gradient NPZ, native traces,
per-map summaries, summary.json, RESULTS.md, metrics.csv and figures. Machine
launch receipts remain local. Expected local work is roughly5-10minutes;
there is no enforced runtime cap. Authorized launch completion means verified
dispatch with a durable receipt; no monitor is installed.

Commands from repository root:

    python -X utf8 -B new/credit_formation/check.py --out analyses/NEW_CREDIT_CHECK.json
    pwsh -File tools/launch_credit_formation.ps1 -RunName NEW_CREDIT_RUN -Qualification analyses/NEW_CREDIT_CHECK.json
