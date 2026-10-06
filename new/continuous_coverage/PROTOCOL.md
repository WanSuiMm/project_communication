# Continuous execution-state coverage under K8

Protocol `continuous_execution_coverage_v1`, frozen before formal training.
The user authorized this comparison after the comprehensive audit and explicitly
distinguished it from the old half-batch random-age warmstart experiment.

Question: does continuously supervised execution across 0..256 help learn a
high-reach/high-retention recurrent rule, compared with repeated cold starts,
when architecture, data, backward-window count and optimizer-update count match?
Older-state exposure alone is not assumed to be the cause of previous failures.

## Intervention and clocks

Use the original 5033-parameter C24/Z8 StreamingCell without modification:

    C' = T_M(C) + .1 F(T_M(C), Z, L_M(C), L_M(Z), X)
    Z' = Z + .5 Q(C', Z, L_M(C'), L_M(Z), X)
    logits = readout(Z)

One super-update uses a single minibatch of eight maps. Parameters remain fixed
until its end. Both arms perform 256 forward macro-steps, 32 K8 backward windows,
32 original-world balanced-BCE losses divided by 32, clip once at norm1, and one
AdamW step. Accumulate gradients over all32 windows; never step inside a segment.

| Arm | Numerical execution before one optimizer step | Cold initializations |
|---|---|---:|
| reset64x4 | 0..64 four times, on exactly the same X/Y/mask | 4 |
| continuous256 | one continuous 0..256 trajectory | 1 |

After every loss, detach all recurrent state values; do not erase numerical
memory. Reset uses model.initial(X) at super-steps0,64,128,192. Continuous carries
the detached state through boundaries64,128,192. No cross-update warm states,
no optimizer moments/state transplant, no loss-free prefix, no independent data
for the four resets, and no paired/source-flip training loss.

The sole assigned intervention is the reset boundary. It necessarily changes
state-age visitation AND encoder-credit frequency: with K8, reset exposes four
first windows to the encoder, continuous only one. Shared recurrent parameters
receive32 windows in either arm. This is an induced part of the treatment, not
a separately controlled mechanism. A gain cannot uniquely distinguish numerical
visitation from that credit redistribution. Do not silently compensate encoder
gradients or add auxiliary losses. Repeating cold data is redundant compute,
not four times as many independent training examples.

## Frozen blocks and execution

Eight fresh paired blocks,16 trajectories, serial local execution. Initialization
seeds96001..96008, schedule seeds97001..97008. All model tensors at update0 match
bit-for-bit within a block. Both arms consume the same300x8 schedule, IID with
replacement, from the unchanged512-map size32 training bank seed10002. Arm order
is reset/continuous in even blocks and reversed in odd blocks, fixed in advance.
The independent statistical unit is the paired initialization/schedule block.

Exactly300 super-updates each: 76,800 forward steps,9,600 backward windows and300
optimizer steps per trajectory. Historical local FP32 Torch2.5.1/CUDA settings;
AdamW lr=.001,wd=.0001,betas(.9,.999),eps1e-8. Same seed and parameterization,
same batch, same supervised steps per super-update and same loss normalization.
Reset computes three extra encoder initializations; this small operation-count
difference is disclosed, and actual elapsed times are reported separately.

CUDA Graph acceleration may be used after a cheap actual-shape gradient/parameter/
Adam equivalence check for both new super-update paths. This changes execution
only, not the scientific arithmetic. A graph failure stops the run; no rescue
sweep or change of the training intervention. No runtime cap or watchdog.

## Dense observation and held-out maps

Save model AND optimizer checkpoints at u0,25,50,...,300. At each point, evaluate
256-step cold rollouts in original and flipped worlds on32 new maps each at
size32 and64; frozen evaluation seeds102032 and102064. These are shared by arms
and checkpoints, distinct from the reused99332/99364 cohorts. Maps/steps are not
independent training replicates. Evaluation never affects weights, optimizer,
schedules, learning rate or model selection.

Retain every integer0..256 Boolean correctness trace for both worlds and their
AND, with lossless bit packing. Record strict16<d<32 map-mean and pooled coverage
at T64/128/256, all-changed coverage, original/flip BA, endpoint retention64→256,
continuous survival of the T64-correct set, gains/losses and ever-regression.
Undefined ratios remain null, with denominators and eligible-map counts.

Dense points u0..275 are formation diagnostics ONLY. Compute the unchanged old
Full predicate, including matched-frontier support, only at u300. The lighter
intermediate summaries omit frontier/Full rather than labeling them failures.
Final traces also go through the frozen full summarizer/predicate. A small trace
replay check binds the new transfer-efficient recorder to the old recorder.

## Final endpoint and interpretation

Official checkpoint is u300 regardless of intermediate quality. Primary axis
R is size32 strict pooled paired coverage at T64 (also report map-mean). Primary
axis S is pooled all-changed T64-correct retention to T256. S is interpreted only
with at least16 contributing maps and100 T64-correct changed cells. Report
continuous S separately: endpoint return to correctness can conceal regression.

A separate, predeclared joint-readiness screen requires R pooled AND map-mean
>=.80, S>=.95 and the above S support. It does not require future coverage gain,
so an already-complete stable model is not rejected for having no headroom.
It is distinct from the unchanged old Full gate; report both without substitution.

The paired reliability qualification requires continuous-minus-reset joint
success net gain>=6/8 AND an exact two-sided discordant-pair binomial p<=.05.
No discordant pairs gives p1. Six wins/no losses gives p.03125. Otherwise retain
`NO_CONTINUOUS_COVERAGE_RELIABILITY_QUALIFICATION`. Report individual final R,S,
paired deltas/support, success counts and all negatives. Undefined S is not zero;
support-qualified S deltas use only pairs defined in both arms. No posthoc best
width, checkpoint, map subset or threshold may replace the primary.

Plot each block's predeclared checkpoint trajectory in the (R,S) plane. Formation,
collapse and recovery are descriptions on this finite observed grid, not a
phase-transition theorem. More late progress without sufficient reach/retention
is a partial diagnostic outcome, not the primary qualification. An improvement
supports this fixed continuous-training recipe, not a unique representation
mechanism, general short-credit solution, NCA superiority or 3D validation.

## Artifacts and failure

New run directory only. Freeze all schedules, bank tensor hashes, source hashes,
protocol, initial tensors and qualification before efficacy training. Save all
curves, checkpoints, summaries, traces and the final figure. Record capture setup,
training, evaluation and whole-run times separately. Partial/error records remain.
Nonfinite states/losses/gradients/parameters, broken provenance, graph mismatch or
evaluation mutation stops execution with ERROR. No automatic hyperparameter rescue.
Launch receipt contains host/PID/GPU/command locally; these and checkpoints remain
outside any future public export. Do not create a recurring monitor.

Repository-root commands:

    python -X utf8 -B new/continuous_coverage/run.py --check --out analyses/NEW_COVERAGE_CHECK.json
    pwsh -File tools/launch_continuous_coverage.ps1 -RunName NEW_COVERAGE_RUN -Qualification analyses/NEW_COVERAGE_CHECK.json

Qualification provides a measured duration/storage estimate. Estimated duration
is not a time limit. Dispatch verification covers the first real optimizer step
and a durable receipt; completion and result analysis are separate states.
