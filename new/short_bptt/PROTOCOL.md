# Short BPTT / matched long computation: screen v1

Frozen before efficacy training, 2026-10-02. User authorized a new bounded
experiment after the source-switch audit. Older stops remain historical.

## Question and single intervention

Can a shared cellular rule trained with eight-step temporal gradients solve
fresh-input source-conditioned propagation beyond a single gradient window,
relative to the same model trained with full64-step BPTT?

Use the existing frozen workspace additive and revision cells unchanged:
W24/Z8,5033 parameters,two sequential masked communication phases per macro-step.
Train both architectures with K8 and K64, paired model seeds0/1: eight runs.
No K16, state pool, new architecture, pretrained weights, warm-switch gate,
repair branch, hidden-state target, Jacobian audit or hyperparameter sweep.
This is a test of truncated training on existing cells, not a new algorithm.

## Matched training

- Forward trajectory always64 macro-steps from fresh initialization.
- Identical target BCE losses at t8,16,24,32,40,48,56,64, averaged equally.
- K64: retain the full graph and backpropagate the sum of eight losses once.
- K8: backpropagate loss/8 at each window end, accumulate parameter gradients,
  then detach BOTH W and Z before the next window. Never reset their values.
- No parameter update inside a trajectory. Clip the accumulated gradient once,
  then call optimizer.step once after all64 forward steps, in either condition.
- Same initial parameters, batch indices, data, optimizer and update count
  within each architecture/seed comparison. Record identities and schedules.
- Seed s:512 size32 training maps, bank10000+s; batch8, schedule RNG20000+s.
  AdamW lr0.001,weight decay0.0001,global gradient-norm clip1. FP32,no compile.
- Default600 updates/arm; all eight arms get the same frozen count. Budget-only
  selection BEFORE efficacy: preflight runs3 updates per architecture/K, using
  the median of the last two synchronized update durations for each arm. Let
  c be their maximum. Select the largest multiple of50 not exceeding
  min(600,1140/(8*1.25*c)). If this is below200, stop without formal training.
  This reserves approximately5 minutes inside the25-minute whole-run cap.
  No loss or evaluation metric participates in update-count selection.
- The encoder receives gradients only through windows connected to its initial
  output, as implied by actual truncation. Parameters are shared across time;
  windows are not independently parameterized models.

The trajectory training objective differs from the earlier reach/auxiliary
recipe, so prior checkpoints are not positive controls. This screen retrains
both K conditions from matched initialization.

## Evaluation and independent unit

Fresh original and flipped-source rollouts, sizes32/64, fixed16 maps each,
bank seeds30032/30064; evaluate T64/128/256. No state or intermediate target is
transferred into evaluation. BA averages per-map class recalls. Paired
correctness requires the same changed-component pixel to be correct under
BOTH fresh source alternatives. Average eligible per-map fractions, not pooled
pixels; pooled counts are secondary. Record raw per-map values and denominators.

Distance is graph geodesic distance from the changed component's source.
Report bins[0,8),[8,16),[16,32),[32,64),[64,infinity), plus primary d>16.
One macro-step has two communication phases, so the K8 window has a spatial
dependency bound of16 edges. Do not call d8 beyond its causal range or equate
distance directly with macro-steps. At T64,d>128 lies outside the forward
light cone and cannot be expected to have positive paired correctness.

Primary endpoint: size32,T64,paired correctness for d>16. All16 fixed maps
contain such pixels (1705 total); size64 has18641 across16 maps. These counts
were checked without running a model. Model seed is the independent unit,n=2
per architecture; maps,pixels,windows and timings are not training replicates.

## Frozen descriptive decisions, per architecture

All arms must complete their selected update count and finite evaluation.
For EACH seed:

1. FullK64 qualification: primary far-paired>=0.80 and BA64>=0.85.
2. K8 reach screen: far-paired>=0.80 and within5pp of its K64 control;
   BA64>=0.85 and within3pp of its K64 control.
3. K8 hold screen (reported separately): BA at bothT128/T256 no more than3pp
   below its own BA64; far-paired no more than5pp below its own far-paired64.

An architecture supports the short-gradient reach screen only if BOTH seeds
qualify and pass. If a full control fails, mark that pair BASELINE_UNQUALIFIED,
not evidence against short BPTT; show all short-run measurements anyway.
Hold cannot rescue failed reach and trivial constant prediction cannot pass
paired correctness. Size64 is secondary and cannot replace the primary endpoint.
Report per-seed and mean short-minus-full effects, with no significance claim.

## Systems measurements and limits

Record synchronized training duration, per-logged-update duration, gradient
clipping fraction, CUDA memory allocated before training and peak allocated
during training. Exclude evaluation from training peak. Peak-minus-baseline
includes optimizer state and gradient storage, not just activations. All arms
use the same batch/data and update count. These are descriptive observations
on one GPU; accuracy-matched systems superiority is not assumed.

Success would show compositional local computation learned with truncated
gradients on this task. It would not prove general long-range credit assignment,
stable useful internal computation, warm reopen, or a novel TBPTT method.
Training state lifetime is64 for all arms; inference lifetime extends to256.

## Execution and provenance

First a CPU check of forward identity, gradient accumulation and detach. Then
one four-arm CUDA preflight (seed0,3 updates per arm, abbreviated evaluation).
Formal launch requires matching source hashes and a passed preflight with a
budget-selected common update count. Alternating horizon order and reversing
architecture order across seeds reduce simple execution-order confounding.

Whole-run cap25 minutes, watchdog at26 minutes; save partials on failure. No
automatic retry, rescue sweep or ongoing monitor. Save source snapshots,
data/schedule/parameter hashes, configs, per-arm checkpoints, metrics, status
and aggregate. Separate local launch receipt records host/PID/GPU/command.
All existing weights and evidence stay unchanged.

Commands from repository root, using new output directories:

    python new/short_bptt/check.py
    python new/short_bptt/run.py --preflight --out runs/NEW_BPTT_PREFLIGHT
    pwsh -File tools/launch_short_bptt.ps1 -RunName NEW_BPTT_RUN -Preflight runs/NEW_BPTT_PREFLIGHT

Verified dispatch and its durable receipt complete the launch request.
