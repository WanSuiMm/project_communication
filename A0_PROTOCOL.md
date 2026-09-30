# Frozen A0 protocol: rt_a0_2d3d_v2_1

Question: does confidence normalization improve learning a single-source bit
under the same spatial medium, emission and reaction? This is a new bounded
mechanism experiment. It does not replace or repair the frozen v1 evidence.

## Intervention and controls

All arms use C=64, r=32, four message groups, eight updates in two phases,
the existing local reaction and target-only readout. All common parameters
start identically for a given training seed. Per-group confidence c=sigmoid(f(s))
starts at 0.5 everywhere. Values v=emit(s); the emitted message is q=c*v.
Confidence receives no source supervision or oracle mask during training.
The constant medium has unit conductance. Learned conductance is 2*sigmoid of
the existing symmetric edge features, with zero final weights/bias initially:
it also starts at exactly one and remains positive and below two. Initial
medium scale is therefore matched. Confidence is a learned nonnegative weight,
not a claim of calibrated probability.

The four RT arms cross constant/learned symmetric medium with raw/normalized
receiver values. Raw uses n=T(q); normalized uses n/(T(c)+1e-6). Every arm's
reaction receives the same feature convention (s,local,e,m,m-q). For normalized
transport this feature difference is not claimed to be a linear diffusion delta.
Both arms pack numerator and confidence into the SAME group-specific solve,
including the raw arm whose denominator is unused by its reaction. Thus this
comparison isolates the receiver division at matched emission and solver work.
Raw A0 is confidence-gated and is not identical to raw v1. Runtime is measured
for these reference implementations, not an optimized raw-solver lower bound.

Attention learns queries and keys, uses the same emitted q as values, and has
no separate value/output projection. It shares the confidence/emission/reaction
design. It is a positive control, not a tuned ViT. Parameter counts are reported.

Keep the original axis-palindromic split in every RT arm to avoid combining an
axis intervention with normalization. No axis-equivariance claim is made.
Evaluate every spatial axis separately as well as pooled. No directed transport,
extra global pooling, global normalization or nonlocal decoder is introduced.

## Software and oracle qualification before training

Check grouped packed solves against separate numerator/confidence solves,
normalized gradients, same-value attention, identical pair initialization, and
finite 2D/3D forward/backward with live confidence and learned-edge gradients.
The original PCR solver and v1 source files remain unchanged.

For fixed oracle emission q=x[source_value] and confidence c=x[source_marker],
measure raw sign decoding, normalized decoding, magnitude, zero/underflow and
epsilon attenuation with tau=[1,16,256,4096]. Cover train d16 at length28, then
d16/32/64/128 at length140, all axes, batch8, fixed oracle seed424242+axis.
Zero raw output is an undecodable tie, not a correct prediction for label0.
The tau4096 group must decode every oracle bit with finite outputs; tiny-tau
underflow is reported and does not independently fail this qualification.
Oracle success only establishes the selected solver/readout path; no trained
model receives oracle emission, confidence or edges.

## Frozen training and stopping

- Both 2D (width14) and genuine narrow 3D (cross-section4x4).
- Seeds 1729,2718,31415,57721 are paired training replicates, not test cells.
- Exactly 480 AdamW updates per completed trial, batch8, lr0.003,
  weight_decay0.0001, gradient clipping1.0. No best-checkpoint selection.
- Train distances4/8/12/16, length28, axis=floor(step/4) mod dim. All
  distance/axis combinations occur equally often in each dimension.
- Evaluation:64 examples per axis for train-shape d16; large-shape d16;
  d32,d64,d128 on length140. Report seed and axis results, not just averages.
- Calibrate attention on all four seeds in both dimensions first. A dimension
  advances only if every seed has train-shape accuracy>=0.99 and every axis
  accuracy>=0.98. Failure is INCONCLUSIVE_POSITIVE_CONTROL; no RT arms run for
  that dimension. Training is not extended to force a positive-control pass.
- Within qualified dimensions run both raw/normalized arms for each medium,
  seed and dimension in the fixed runner order. A failed RT arm does not cancel
  its matched comparisons. Any nonfinite loss/gradient or total time limit stops
  the entire remaining schedule and records incompleteness.
- Maximum training/evaluation scheduling budget25 minutes. A final evaluation
  already begun may finish after the deadline. At most40 trials. No retries,
  substitute seeds, sweeps, B/C runs or recurring monitor.

## Endpoints and interpretation

Primary mechanistic endpoint is per-seed normalized-minus-raw accuracy in each
medium, reported separately for fitting, grid-size shift and every distance.
A0_SINGLE_SOURCE_PASS requires all four seeds to reach pooled accuracy>=0.99
and every axis>=0.98 in every evaluation condition. These are qualification
thresholds, not a significance test or a claim of architecture superiority.

Report denominator min/mean/p10, fraction<=epsilon, learned source/background
confidence, source's fraction of transported confidence mass, target message
RMS, tau, gradients, parameter count and synchronized batch1 latency. Epsilon
normalization does not inherit the original operator's Euclidean contraction.
Evaluation raises on nonfinite logits; reference latency includes this check.
An oracle pass plus failed learned models does not isolate emitter from reaction
optimization; a constant-normalized pass does not prove semantic routing.

Use a new run directory containing config, atomic per-trial results, checkpoints,
aggregate, Markdown summary, numeric/oracle checks, source snapshot and hashes,
and local host/PID/GPU/command/time launch receipt. Never overwrite old runs.
Training seeds are independent units. No formal uncertainty claim from four
seeds; report all paired outcomes. No natural-vision or general-3D claim.

## Commands from the repository root

```powershell
python a0.py check --out runs/NEW_A0_CHECK_DIRECTORY
python a0.py qualify --out runs/NEW_A0_RUN_DIRECTORY --checks runs/NEW_A0_CHECK_DIRECTORY/checks.json
```

The runner requires check success for the exact current Python source hashes.
CUDA is required for this frozen local protocol; it does not silently substitute
CPU timings. The qualification exits by itself and writes its final summary.
