# Frozen protocol: rt_2d3d_qualification_v1_1

## Question and primary comparison

Does learned symmetric directional transport improve useful communication over
an eight-update local network, and does its content dependence beat constant
transport? Qualify 2D and 3D separately before considering larger experiments.

This first look is a bounded qualification, with two paired training seeds
(1729, 2718). Training seeds are the independent replicate unit. Test examples,
voxels and repeated timings are not extra training replicates. No significance
or broad scientific generalization is claimed from two seeds.

## Model and implementation

- Width C=64, message width r=32, four groups, eight recurrent updates in two
  phases of four; parameters tied within a phase. This is a reduced-width
  qualification of the proposed C=128 model, not an evaluation of that exact size.
- Fixed pointwise input evidence E, persistent state H, channel-only LayerNorm,
  depthwise 3x3 or 3x3x3 local reaction, pointwise MLP and sigmoid write gate.
- Residual scale starts at 0.1, a frozen qualification choice instead of the
  proposal's illustrative 0.01. It is learned thereafter; no convergence claim.
- Learned undirected nearest-neighbor edges use symmetric endpoint sum and
  absolute difference; boundaries have zero outgoing conductance. Recompute
  edges only at each phase start. Constant transport has unit internal edges.
- Initial tau=[1,16,256,4096] per phase; positive learned scales use exp(log_tau).
- 2D uses X-half/Y/X-half; 3D uses X-half/Y-half/Z/Y-half/X-half.
- Portable float32 PyTorch PCR with factors reused across four updates and
  shared channel RHS. Backward solves the adjoint system and computes edge
  gradients. This is not a fused-kernel efficiency result.
- All models receive identical normalized coordinate channels. Readout gathers
  only the target state: no global pool, global normalization or global decoder.
- CNN uses eight unshared local cells; NCA uses two cells each repeated four
  times. The ViT-style control substitutes global PyTorch SDPA message transport
  at the same r. This is a controlled attention block, not a tuned standard ViT.
  Parameter counts are reported; these are width/depth-matched, not parameter-
  matched models.

## Data and training

Inputs have signed source values, source markers, a target marker, a binary
region mask and independent +/-1 local detail. Only those five channels plus
coordinates enter models; oracle geometry and labels are never passed to a
learned model. Source labels are random and independent of target-local inputs.

Training uses distances 4,8,12,16, main-axis length 28, rotating the main axis,
batch 8, 240 updates per model/seed, AdamW lr=0.003, weight decay=0.0001, gradient
clip=1.0. Every model sees matched seeded examples. 2D transverse width is 14;
3D transverse size is 4x4. Evaluation covers each axis with 64 independent
examples per condition, using identical test examples across models/seeds.

Axis schedule: axis=floor(step/4) mod dim, distance=[4,8,12,16][step mod 4].
This crosses all distances with all axes exactly evenly over 240 steps. The
initial v1 dispatch was stopped after discovering that axis=step mod dim
confounded axes and distances in 2D; its partial output remains excluded.

Report train-shape d16, large-shape d16, then d32,d64,d128 on common main-axis
length 140. Large-shape d16 separates grid-size shift from relation-distance
shift. Accuracy is reported by axis and pooled over axes. Geometry is thin and
axis-aligned, so results cannot establish arbitrary obstacle/bend handling.

Latency uses batch 1, d128, last axis, two warmups and seven synchronized wall
timings. Report median and allocated training memory. NCA also gets three
timings at 128 updates (same weights). The latter is a path-length-matched cost
reference only, not an assertion that recurrence extrapolation solves the task.

## Gate A: one source

One random source bit; target predicts it. Compare CNN, NCA, attention, constant
and learned transport, both dimensions and both seeds before any Gate B.

Positive control: attention train-shape d16 accuracy >=0.90 in both seeds.
Without it, report INCONCLUSIVE_POSITIVE_CONTROL. Learned transport must also
fit train-shape d16 >=0.90; otherwise report NOT_QUALIFIED_FIT_WITHIN_BUDGET.

Pass only if, in each seed, learned transport accuracy >=0.85 and exceeds
eight-step NCA by >=0.15 at each d32,d64,d128, and its eight-step d128 latency is
less than the 128-update NCA reference. Otherwise stop that dimension after A.
A failure means this frozen configuration did not qualify, not that all
reaction-transport architectures are impossible.

## Gate B: four sources, one relevant region

Four separated thin straight corridors, independent random source bits and one
randomly chosen target corridor. No corridor IDs appear in the input. Compare
constant, learned and oracle-edge transport (the oracle is a diagnostic only).

Oracle must reach d64 accuracy >=0.85 in both seeds; otherwise the gate is
inconclusive. Learned transport must reach >=0.80 and beat constant by >=0.10
at both d32,d64 in both seeds. Otherwise stop that dimension before C. Report
d128 as additional distance evidence, not as an extra pass condition.

## Gate C: distant source plus independent local detail

Four-way target label = 2*source_bit + target_detail_bit. Compare a deliberately
matched full-width pair: Q=H, r=C, identical transport/reaction/initialization.
`message_full` retains H as residual base; `state_full` replaces that base by
T(H). This isolates state retention at equal transport work. It is not a
comparison confounded by r=32 versus C=64 communication.

At d64, require retained-state joint accuracy >=0.75, detail marginal accuracy
at least 0.10 above state diffusion, and source marginal accuracy no more than
0.05 worse, in both seeds. Continued access to E is identical in both arms;
this may remove a retention advantage and a null result is retained.

## Bandwidth and stopping

Only after A/B/C all pass for a dimension, run Gate B learned transport with
r=4,8,16,32,64 (reuse the already completed r32 arm). This is a message-width
scan, not a theorem about whole-network rank, and cannot rescue a failed gate.

Maximum scheduled duration is 25 minutes, checked at training batch boundaries;
an already started final evaluation may finish after the deadline. Any nonfinite
loss/gradient stops the schedule and preserves evidence. No tuning, additional
seeds, changed thresholds, hidden sweeps, real-vision training or automatic
reruns are authorized by this protocol.

Every run stores config, source snapshots/hashes, host/PID/device launch receipt,
per-trial metrics/checkpoints, and an aggregate. Machine-specific receipts stay
local. Numerical checks establish software validity only, never gate passage.
