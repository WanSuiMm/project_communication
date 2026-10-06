# Outcome-blind tail calibration and fixed lane hard clipping

Protocol `hardclip_v1_tail_calibration_v1`, frozen before sampling or threshold
selection. This is the final planned test of unusually large persistent writes
for this recipe. It is not a final architecture or a test of all amplitude laws.

## Stage 1: fixed, outcome-blind calibration

Replay ALL eight neural u300 checkpoints from `hybrid_writer_20261006_01`, blocks
0..7 without success filtering. Verify file/parameter hashes against the existing
Hybrid Writer publication manifest. Use ONLY original-world input X from the
old training bank, indices0..15 in the fixed original order, size32, seed10002.
Every checkpoint receives the SAME16 maps and a cold64-step rollout at frozen
parameters, on the historical local Torch2.5.1 FP32 backend. No labels, readout,
correctness, losses, retention, or long-rollout performance select thresholds.
Do not construct an optimizer. The readout is never called.

At EVERY step1..64 collect actual raw carrier-write RMS `r=RMS(0.1*m)` over
the six payload coordinates of each of four lanes, on ALL open cells. Retain
the full FP32 samples, indexed by block/time/open-cell/lane, and the selected X.
Pool all block/time/open-cell events equally within each lane. Equal input maps
give each block and step the same count. Per-block diagnostics remain secondary.

For each lane d, solve by48-step float64 bisection:

    D_d(b) = sum(max(r_d-b,0))/sum(r_d) = 0.01.
    A_d(b) = count(r_d>b)/N_d.

The numerical dose target tolerance is1e-7. The largest threshold achieving
the dose minimizes the trigger rate among thresholds meeting at least that dose.
The four FP32 buffer values are frozen after selection; report their achieved
dose/trigger separately from the float64 root. Both must meet the tolerance and
ALL four lanes must have `A<=0.20`. Zero/nonfinite lanes stop as ERROR, not a pass.
These are engineering design cutoffs, not theoretically optimal quantities.

If any lane is too dense, verdict `NO_SPARSE_TAIL_AT_PREDECLARED_DOSE`: do not
start efficacy training, change the dose, relax the trigger ceiling, or select
another threshold by task performance. This says only that this fixed reference
distribution did not admit the prescribed sparse intervention. Save the negative
calibration and stop this amplitude route; it does not prove writes are harmless.

## Stage 2: conditional two-arm qualification

Only after calibration PASS, compare Neural vs HardClip. Same C24/Z8, original
MLP proposal, encoder, masked streaming, Q, Z update, readout, loss and optimizer.
No new learned parameters, controller, auxiliary loss or straight-through rule.
For each lane, with raw write `v=.1*m` and actual-write cap `b_d`:

    delta = v                              if RMS(v)<=b_d
    delta = b_d*v/RMS(v)                    otherwise
    C_new = T(C)+delta.

Use a numerically safe positive denominator floor only in the unused scaling
branch; the active below-threshold branch is byte-identical raw v, including
zero-point gradients. Differentiate the true clipping operation. Four caps are
buffers, fixed across blocks, times and sizes. Proposal-cap units, if required,
are `B=b/.1`; no epsilon-shifted clipping boundary is permitted.

Eight NEW paired initialization seeds116001..116008 and schedules117001..117008,
alternating arm order by block. Copy every trainable parameter identically.
Reuse the size32 training bank seed10002,512 maps, batch8. Repeat the SAME batch
four cold64-step segments:256 forwards,32 K8 BCE backward windows, detach after
each, divide losses by32, clip accumulated gradient norm1, ONE AdamW step after
the entire super-update.300 updates; lr.001, wd.0001, betas(.9,.999), eps1e-8.
Parameters stay fixed within each super-update. No warm states or paired loss.

Evaluate fresh32-map held-out banks at sizes32/64, seeds118032/118064. Save all
checkpoints u0,25,...,300 and Boolean original/flipped/paired traces0..256.
The fixed-u300 joint-readiness gate remains size32 strict16<d<32 coverage:
pooled AND map-mean>=.80, all-changed T64-correct retention to256>=.95, reference
support>=16 maps and100 cells. Old Full is separate, measured at u300 only.
The independent statistical unit is a paired training block, n=8.

ONE primary contrast: HardClip minus Neural. Reliability qualification requires
net candidate-only wins minus reference-only wins>=6/8 AND exact two-sided
discordant-pair binomial p<=.05. No three-test Holm family. Otherwise preserve
`NO_HARDCLIP_RELIABILITY_QUALIFICATION`. This screens large effects; a negative
does not exclude all small benefits. Checkpoint-grid ever-ready/loss-of-readiness
are secondary; no checkpoint selection replaces u300.

## Dose and execution records

During training retain clip trigger count/event count and sum RMS(v-delta)/sum
RMS(v) on open lane-cell events for every update. Both arms report these quantities:
HardClip actually applies clipping, Neural calculates its counterfactual dose.
At checkpoint evaluation separate t1..64 from t65..256, sizes and worlds; do
not merge them. Thresholds never adapt to the observed activation or outcomes.
If actual dose fades, disclose the weakened intervention; do not interpret the
negative as a general large-write mechanism refutation or start a threshold sweep.

Use new output directories, immutable source snapshots and input/checkpoint
bindings. Stop on nonfinite values, source/parameter mutation, or meaningful
implementation-check failures. One cheap CPU invariant suite precedes calibration;
if Stage1 passes, check actual-shape eager/CUDA Graph training before dispatch.
No runtime cap, watchdog or recurring monitor. Original evidence is read-only.
Machine/PID/command/time receipts stay local. Completion of a conditional training
launch means dispatch verified after tool yield, not a claim of finished training.

## Repository-root entry point

    python -X utf8 -B new/hardclip_v1/primitive.py
    python -X utf8 -u -B new/hardclip_v1/calibrate.py --out runs/NEW_HARDCLIP_CALIBRATION

Stage2 commands are added only if Stage1 qualifies; freeze the selected caps
and implementation before new efficacy training.
