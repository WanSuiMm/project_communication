# ReLU-conditioned input lift: prospective matched developmental screen

This protocol specifies one new training arm for the fixed procedural RGBA
NCA task. It asks whether carrying a factored history of the ReLU layer's
augmented inputs across K8 boundaries improves on the frozen AU-NCA K8 arm.
Only `relu_lift_k8` is trained in this run. The three comparison arms are
read from the completed `runs/au_nca_20261009_01` experiment and remain frozen.
This is a three-block developmental screen, not a confirmatory or general
claim about truncated BPTT.

## Question and comparison

The original cell uses the same parameters and forward function as the
accumulated-update NCA (AU-NCA): 16 visible channels, fixed identity and Sobel
perception, a shared 128-unit ReLU layer, and a zero-initialized 16-by-129
projection. The proposed lift stores each active ReLU unit's accumulated
49-dimensional augmented input rather than only its scalar activation. At a
K8 boundary, the live first-layer matrix can therefore project that stored
history and receive a direct gradient from the terminal loss.

| Arm | Source | Role |
|---|---|---|
| `original_k64` | Frozen historical run | Positive-control reference |
| `original_k8` | Frozen historical run | Ordinary truncated-credit reference |
| `au_k8` | Frozen historical run | Existing accumulated-feature reference |
| `relu_lift_k8` | New run only | Input-lift treatment |

The new arm has the same trainable parameters, initialization, target, and
nominal optimizer settings as the historical arms. The independent unit is a
paired initialization/schedule block. The 32 evaluation trajectories within
each block are measurement samples, not additional independent replicates.

## Frozen historical reference

Use `runs/au_nca_20261009_01` as the only source for the three controls,
input-bank artifacts, target, and qualification record. Do not rerun or
rewrite those controls. The reference run is `COMPLETE`; its three `original_k64`
controls qualify at the existing thresholds (mean T64 NMSE at most 0.10 and
mean alpha IoU at least 0.80). The frozen per-block T64 means are:

| Block | `original_k64` | `original_k8` | `au_k8` |
|---:|---:|---:|---:|
| 0 | 0.000192 | 0.230793 | 0.129319 |
| 1 | 0.000442 | 0.248437 | 0.141727 |
| 2 | 0.004828 | 0.174778 | 0.202447 |

The historical aggregate is `AU_BENEFIT_DEVELOPMENTAL` under its own frozen
decision rule: AU-NCA improved over ordinary K8 in blocks 0 and 1, but was
worse in block 2. This new screen asks whether the input lift improves on
that mixed AU-NCA result.

These are historical executions, not simultaneous controls. The new arm must
load the exact target and block input banks from the reference and bind its
provenance to the reference manifest. In particular, verify the manifest's
SHA-256 values for `block00_inputs.npz`, `block01_inputs.npz`,
`block02_inputs.npz`, and `target_rgba.npy`, plus the recorded source hashes
for `new/au_nca/run.py`, `new/au_nca/cells.py`, and `new/au_nca/tasks.py`.
Do not regenerate pool-index schedules, held-out mask banks, target pixels, or
seed state.

The frozen reference manifest currently records these hashes:

| Artifact | SHA-256 |
|---|---|
| `new/au_nca/run.py` | `8cde026785aa00246c6517e8db205b81d941cfe8a02e92436776d26610d753b7` |
| `new/au_nca/cells.py` | `cd9cb6bee563c4285c1930876cd7bbdbbfdbf84e77270cb244e04e13aac5d3a6` |
| `new/au_nca/checks.py` | `4d0143b7264115f8961a7b462001a0e6be97b4ec54994231bcc083567e9f12df` |
| `new/au_nca/tasks.py` | `2c8dbde48326dde78d0987adfb923bd71d20bb184a1eb9481cc8d905f82cc8c6` |
| `block00_inputs.npz` | `ec7e83ef4a609a32255b9a52d67ac69df3ad07639d9e5e63b0152da265c559dc` |
| `block01_inputs.npz` | `1187148132fffea174cad9f44cfb6348af62a7beafb80775339014bd075d449e` |
| `block02_inputs.npz` | `ffe1a31e619812f35126ff19494935a31ca6ac2f390955e5d0895d7e3ddbcf24` |
| `target_rgba.npy` | `9e283574cdba132edaad5e049f3ed2c633be8206724d48473c237c3fa97a6d54` |

The historical input banks freeze the pool-index schedule and the held-out
mask banks; training firing masks remain bound to the recorded deterministic
seed rule below. The 512-entry visible-state pool is initialized from the
common seed and then evolves separately under each arm's own learned
parameters. Thus the input schedules are matched, while trained trajectories
and historical floating-point execution are not guaranteed to be identical.
Report this as a comparison to frozen historical controls, not as a same-run
implementation parity test. If the reference files, hashes, or qualification
record do not match, stop before interpreting a paired result.

## Task and model

Use the existing fixed 32-by-32 target from
`new/au_nca/tasks.py`: four colored disk petals, a yellow center, a curved green
stem, and a tilted asymmetric leaf, on a transparent background. RGB is
premultiplied by alpha. All arms start each pool entry from the same
16-channel seed: a one-pixel alpha seed, RGB zero, hidden channels 4 onward
equal to one at the seed cell, and zero elsewhere. There is no external image,
teacher, augmentation, or target variation.

The visible state has 16 channels: RGBA plus 12 hidden channels. Perception
concatenates identity, horizontal Sobel, and vertical Sobel responses for each
channel, with zero padding. Append a constant one to make
`p_t` in R^49. The first affine layer is `A` in R^(128 x 49); the output
projection is split into `W_h` in R^(16 x 128) and `w_0` in R^16, corresponding
to the 128 ReLU features and appended constant in the historical 16-by-129
projection. Preserve the historical initialization and trainable parameter
count (8,336).

At site `i`, let `m_t[i]` be the scalar Bernoulli firing mask, shared across
channels, and let `q_t[i]` be the conjunction of the pre-update and
post-candidate 3-by-3 alpha-alive masks (`alpha > 0.1`). Use the same mask
schedule and alive rule as the historical protocol. The treatment's lifted
state at each site is `b` in R^16, `U` in R^(128 x 49), and scalar `v`.
Initialize `b=x_0`, `U=0`, `v=0`; reconstruct the visible state as defined in
`THEORY.md`. On every step update the full lifted state with the same scalar
`q_t` at that site. The pool stores only visible 16-channel states; reset
`U=0` and `v=0` at the beginning of every optimizer rollout.

## Training procedure

Train three paired blocks with initialization seeds 190101, 190102, 190103;
pool schedule seeds 190201, 190202, 190203; and held-out evaluation seeds
190401, 190402, 190403. For each block, load its existing `blockXX_inputs.npz`
from the historical run. Each of the 3,000 updates samples eight distinct
indices from the 512-entry visible pool according to the stored pool-index
schedule. Before the rollout, reset the selected state with the largest
visible RGBA MSE to the seed. At each step and each cell, fire independently
with probability 0.5, shared across channels. Preserve the historical mask
binding: for one-based update `u=1,...,3000`, seed a CUDA generator with
`schedule_seed + 104729*u + 10000000` and draw `[64, 8, 1, 32, 32]` Boolean
masks by thresholding uniform draws at 0.5. The mask plan is shared with the
historical arms by seed and schedule definition. The frozen historical
protocol described a zero-based rule, but its actual runner called
`train_masks(block,u,...)` with `u=1,...,3000`. This experiment follows the
executed historical source, without editing that earlier frozen document.

Each update is one fixed-parameter, 64-step rollout and one terminal mean
squared error over batch, RGBA channels, and pixels. There are no intermediate
losses or optimizer steps inside the rollout. After the update, write the
visible results back to the selected pool entries. Use AdamW with learning
rate 0.002, weight decay 0, default betas and epsilon. Normalize each
parameter tensor's gradient by its own L2 norm plus 1e-8 before the optimizer
step. Use float32 without AMP, TF32, or CUDA Graphs. Do not tune parameters,
add losses, or add training arms.

K8 truncation detaches the numeric `b`, `U`, and `v` state every eight steps
without clearing it. Because the only training loss is at T64, the first 56
steps may run without an autograd graph. At the boundary after step 56,
reconstruct the visible state using live `A`, `W_h`, and `w_0`, then attach the
graph for steps 57–64. This retains the direct projection gradients through
the accumulated history while cutting gradients through the earlier history
that produced `U`.

## Evaluation

At updates 1000 and 2000, evaluate the first eight held-out trajectories at
T64 as diagnostics only. They cannot select a checkpoint or change the
protocol. At update 3000, evaluate 32 fresh trajectories from the cold seed at
T64 using the exact held-out mask bank in each historical block input file.
Continue those same trajectories and masks to T128 and T256 as secondary
retention measurements.

For the secondary damage/regrowth analysis, branch the same T64 trajectories.
Continue the intact branch with its actual lifted state. For the damaged
branch, canonicalize to `b=visible`, `U=0`, `v=0`, then erase every state
channel in the fixed right half of the grid. Continue both branches with the
same stored suffix masks for 64 and 192 additional steps. Damage/regrowth is
secondary and cannot affect the primary decision.

Report per-trajectory normalized MSE and alpha IoU (threshold 0.5) at T64,
T128, and T256. Normalize MSE by the MSE of the all-zero image against the
target. The primary statistic is the mean T64 normalized MSE over 32
trajectories, reported for each block and arm. Report alpha IoU, individual
trajectory values, and secondary intact/damaged NMSE and IoU alongside it.

## Qualification and frozen decisions

The historical controls already pass the K64 qualification in all three
blocks. Recheck the stored qualification and hashes before starting; do not
spend a new run reproducing the controls. The new cell's `--check` must verify
the fixed-parameter forward identity and K8 boundary-gradient identities on
the frozen mask rules before training. Record its input shapes, mask
mismatches, numeric tolerances, and parameter-gradient residuals. A failed
qualification check stops the new arm before training.

For each block `b`, define
`gain_b = NMSE(au_k8,b) - NMSE(relu_lift_k8,b)`. Positive values favor the
new arm. Label the result `RELU_LIFT_BENEFIT_DEVELOPMENTAL` only if all
conditions hold:

1. The mean of the three `gain_b` values is at least 0.03.
2. `gain_b` is strictly positive in at least two of the three blocks.
3. In no more than one block is the new arm worse than `au_k8` by more than
   0.02 NMSE (`gain_b < -0.02`).

Report the continuous paired values regardless of category. Label a result
`RELU_LIFT_RECOVERY_DEVELOPMENTAL` when the benefit conditions
above hold and a strong remaining-gap recovery also meets the following
conditions: the new arm's mean T64 NMSE is at most 0.10 and, in at least two
blocks,
`NMSE(relu_lift_k8) <= NMSE(original_k64) + 0.5 *
(NMSE(au_k8) - NMSE(original_k64))`.
Otherwise, if the benefit conditions fail, use
`NO_QUALIFIED_RELU_LIFT_BENEFIT`.

These are developmental decision thresholds for this one task and three
paired blocks. Do not report a significance or population-reliability claim.
A completed run, passing software check, or finite output is not itself a
benefit. Preserve a qualification failure, non-finite run, OOM, or partial
run without rescue tuning or outcome-based changes to thresholds.

## Checkpoints, stops, and reproduction

Save checkpoints at updates 1000, 2000, and 3000. Save an atomic latest-resume
checkpoint every 25 updates with model, optimizer, visible pool, and RNG
states. The run-level manifest binds the source, configuration, reference,
seeds, target, and input-bank hashes; validate it before loading a checkpoint.
Resume only when those bindings match. There is no runtime cap,
automatic retry, or recurring monitor. Stop only for a failed qualification
check, non-finite values, or OOM; preserve partial and failed runs as such.

The planned entry points are:

```powershell
python -X utf8 -B new/relu_input_lift/run.py --check --out analyses/NEW_RELU_CHECK
pwsh -File tools/start_protected_job.ps1 `
  -JobName relu_input_lift_20261009_01 `
  -Script new/relu_input_lift/run.py `
  -ScriptArguments @('--out','runs/NEW_RELU_RUN',
    '--qualification','analyses/NEW_RELU_CHECK/qualification.json',
    '--reference','runs/au_nca_20261009_01')
```

The check and run outputs must record new source/configuration hashes, the
historical reference manifest identity, each block's input hashes, training
losses, checkpoints, T64/T128/T256 metrics, and final run status. The lifted
state has 6,289 float32 values per cell, 43.4 times AU-NCA's 145. This
protocol makes no GPU-efficiency or end-to-end runtime claim.
