# AU-NCA v0: matched short-credit developmental screen

This protocol is frozen before qualification and training. The task is a
deterministic procedural RGBA reconstruction with stochastic cellular firing.
It tests one parameter-specific credit intervention on a fixed NCA cell; it
does not test generic NCA trainability.

## Question and arms

Does retaining direct credit for the final projection matrix W across
eight-step truncation improve learning over ordinary K8 BPTT, when the
forward functions are identical at matched parameters and each arm receives
one optimizer update per 64-step rollout? Full K64 is the positive control.

| Arm | Forward cell | Gradient window |
|---|---|---|
| original_k64 | Original NCA | All 64 steps |
| original_k8 | Original NCA | Last 8 steps at each boundary |
| au_k8 | Accumulated-feature reparameterization of the same NCA | Last 8 steps for feature parameters; accumulated direct W credit retained |

Each arm has 16 visible channels: RGBA followed by 12 hidden channels.
Perception concatenates identity, horizontal Sobel, and vertical Sobel
responses for each channel, giving 48 values per cell. A shared 128-unit
ReLU layer is followed by an appended constant one, giving a 129-value
feature h. A zero-initialized linear projection W maps h to 16 state
increments. Sobel filters use zero padding at the spatial boundary. A scalar
Bernoulli firing mask is shared across the 16 channels at each cell.

The alive mask is the 3-by-3 neighborhood maximum of alpha greater than
0.1. Compute the pre-update and post-update masks from their respective
visible states, then multiply the candidate state by their conjunction.
This gates the complete state, including hidden channels. Every arm uses
the same alive rule and same mask plan.

## Task and target

The only target is the fixed 32-by-32 image returned by
tasks.make_target(): four differently colored disk petals, a small yellow
center, a curved green stem, and a tilted asymmetric leaf. The background is
transparent. RGB values are premultiplied by alpha; all occupied pixels have
alpha one. The complete drawing lies within Chebyshev radius 9 of the center
seed cell. There is no external image, teacher, augmentation, or target
variation.

tasks.make_seed() creates 16-channel states with RGB zero, a one-pixel
alpha seed, and hidden channels 4 onward set to one at that same cell. All
off-center values are zero. The initial pool and every cold evaluation
trajectory use this seed.

## Frozen training procedure

Train three paired blocks. Initialization seeds are 190101, 190102, and
190103; pool-index schedule seeds are 190201, 190202, and 190203; held-out
evaluation seeds are 190401, 190402, and 190403. Within a block, all arms
start from bit-identical trainable parameters and use matched pool indices
and stochastic firing plans.

Each arm receives exactly 3,000 optimizer updates. Every update samples a
batch of eight visible states from a 512-entry pool and runs 64 steps. All
pool entries begin at the seed. For each block, generate each row of eight
pool indices from NumPy default_rng(schedule_seed).choice(512, 8,
replace=False), and save the complete schedule. Before each rollout, compute
the current visible-state RGBA MSE against the target for the eight selected
pool states. Reset the highest-MSE state to the seed; the other seven start
from their scheduled pool states.
Only visible 16-channel states are stored in the pool. After the update,
write the resulting visible states back to their selected entries. AU-NCA
starts each optimizer update with base b equal to the sampled visible states
and accumulator e equal to zero; e is never stored in the pool.

At each time step, each batch item and spatial cell fires independently with
probability 0.5. The firing decision is shared across channels. For
zero-based update u in 0 through 2999, seed a CUDA torch.Generator with
schedule_seed + 104729*u + 10000000 and draw a Boolean plan of shape
[64, 8, 1, 32, 32] by thresholding
torch.rand at 0.5. Use the same plan for all three arms in a paired block
and preserve the seed binding for exact resume. The target loss is terminal
mean squared error over all batch, channel, and pixel values in RGBA. There
are no intermediate losses.
Run one backward pass and one optimizer step after each 64-step rollout;
parameters remain fixed during the rollout.

Use AdamW with learning rate 0.002, weight decay 0, and default betas and
epsilon. Normalize each parameter tensor's gradient by its own L2 norm plus
1e-8 before the optimizer step. Apply this rule identically in all arms.
Use float32 without AMP, TF32, or CUDA Graphs. Do not tune hyperparameters
or add auxiliary losses.

original_k64 retains the full rollout graph. Since the only loss is at T64,
original_k8 can run steps 1 through 56 without a graph, detach the visible
state, and attach the graph for steps 57 through 64; this is equivalent to
detaching at steps 8, 16, 24, 32, 40, 48, and 56. au_k8 runs its first 56
steps without a graph while retaining the numeric base and accumulated
features. At step 56, it reconstructs the visible state from the detached
numeric accumulator and live W, then attaches the graph for the final eight
steps. It does not reset e at the boundary, so W remains connected to the
accumulated features from earlier steps. At the next optimizer update,
AU-NCA starts fresh with e=0.

Run block 0 original_k64 first. If its held-out T64 control fails the
qualification below, stop without starting the remaining arms. Otherwise
complete the remaining arms in frozen order. A later control failure does
not erase completed measurements or permit a rescue sweep.

## Held-out evaluation and endpoints

At updates 1000 and 2000, evaluate the first eight held-out firing
trajectories at T64 as diagnostics only. At update 3000, evaluate 32 fresh
independent firing trajectories from the cold seed at T64. For each block,
generate and save one Boolean 256-step mask bank from
NumPy default_rng(eval_seed), with shape [256, 32, 1, 32, 32]. Continue the
same 32 trajectories and mask bank to T128 and T256; these are secondary
retention measurements.

For the secondary damage/regrowth analysis, branch the same held-out
trajectories at T64. Preserve the actual intact state, including the AU
accumulator, for autonomous continuation. For the damaged branch only,
canonicalize to b equal to the visible state and e=0, then erase every
state channel in the fixed right half of the grid. The rebase preserves
the fixed-parameter forward function and prevents erased information
from remaining in an accumulator.
Continue intact and damaged branches with the same 192 suffix masks from
the cold-seed bank for 64 and 192 additional steps. Do not draw a separate
damage mask plan. Damage and regrowth are secondary and cannot affect the
primary decision. Report their NMSE and alpha IoU and paired differences
from the intact continuation at 64 and 192 post-damage steps.

Report per-trajectory normalized MSE and alpha IoU at threshold 0.5 at T64,
T128, and T256. Normalized MSE is RGBA MSE divided by the MSE of the all-zero
image against the target. The primary summary is the mean T64 normalized
MSE over the 32 held-out trajectories, with per-block values retained.
Alpha IoU and all individual values remain visible; they do not replace the
frozen NMSE decisions below.

Each paired training block is the independent unit. Trajectories, pixels,
time steps, and optimizer updates are not independent replicates. No
significance or population-reliability claim is made from three blocks.

## Qualification and frozen decisions

Each block's original_k64 positive control qualifies when its mean held-out
T64 normalized MSE is at most 0.10 and its mean alpha IoU is at least 0.80.
If block 0 fails, report a control-unqualified stop and do not interpret
K8/AU-NCA as a falsification or benefit. For later blocks, retain every
result; the overall comparison is control-qualified only if all three K64
controls qualify.

After all controls qualify, report AU-NCA as a developmental benefit only
when both conditions hold:

1. In at least two of three paired blocks, original_k8 mean T64 NMSE minus
   au_k8 mean T64 NMSE is at least 0.05.
2. Across all held-out trajectories, au_k8 pooled mean T64 NMSE is no more
   than 0.02 above original_k8.

A strong gap additionally requires, in the same paired blocks, original_k8
mean T64 NMSE at least 0.20 and au_k8 mean T64 NMSE at most 0.10 in at
least two of three blocks.
Report the continuous paired values regardless of category. A failed K64
control leaves the short-credit question unresolved; preserve that failure
without relabeling it as evidence against AU-NCA.

## Checkpoints and provenance

Save model checkpoints at updates 1000, 2000, and 3000. Save an atomic
latest-resume checkpoint every 25 updates containing model and optimizer
state, visible pool, RNG states, and the fixed schedule bindings. Keep the
schedule and firing-mask plans or their deterministic seed bindings with
the run. Resume only when source, configuration, seed, target, and schedule
bindings match. There is no wall-clock cap, automatic retry, or recurring
monitor. Preserve partial or failed runs as such.

Record source/configuration hashes, per-update training loss, checkpoints,
held-out predictions and metrics, paired block summaries, and run status.
The target and seed constructors live in new/au_nca/tasks.py.
