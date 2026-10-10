# Real-task AU-NCA developmental screen on FIVES

This is a single-block, developmental comparison of accumulated projection
credit on real retinal-vessel segmentation. It does not establish a
multi-seed effect, clinical performance, or a native-resolution result. The
model is developed and evaluated at **512 × 512**. FIVES images and labels
are published at 2048 × 2048, so this protocol must not be described as a
2048-pixel experiment.

FIVES contains 800 color fundus photographs with vessel masks; the official
archive provides 600 training and 200 test images in four diagnosis groups.
See the [official Figshare record](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169)
and the [FIVES dataset paper](https://doi.org/10.1038/s41597-022-01564-3).
The task and label source are FIVES only. Do not add cross-dataset training
or validation on CHASEDB1 or another vessel dataset.

This protocol adapts the frozen real-task controls in
[the DRIVE protocol](../real_task_nca/PROTOCOL.md) and the short-credit
comparison in [the AU-NCA protocol](../au_nca/PROTOCOL.md). The historical
AU-NCA outcome is recorded in
[its results report](../../evidence/au_nca_20261009_01/RESULTS.md); it is
motivation and context, not evidence about FIVES.

## Dataset, resolution, and split

Use the official Figshare file **34969398** from item **19688169**. Before
extraction, verify the downloaded archive against the registered file size
of **1,764,974,308 bytes** and MD5
`789c80dd5376a82063e27fa49192bac9`. The dataset is licensed CC BY 4.0; cite
the FIVES paper and credit the dataset in any report or release. Acquire the
official archive only. Do not construct substitute masks or source images.

Only files under the official `train/Original` and `train/Ground Truth`
directories enter the experiment. The acquisition helper may extract the
official archive as a whole, but dataset discovery and image decoding must
remain within the official training tree; do not open or decode the 200 test
masks, compute test metrics, or use test data for any model, threshold, gate,
or checkpoint decision. Do not use `Quality Assessment.xlsx` to alter this
protocol's split or select images.

The four image-name suffixes are `A` (AMD), `D` (diabetic retinopathy), `G`
(glaucoma), and `N` (normal). Within each suffix, parse the numeric image ID
and sort numerically. In each 150-image class, use zero-based sorted positions
`[4::5]` as validation: 30 images per class, 120 total. The remaining 120 per
class, 480 total, are the only development-training images. Do not reshuffle
this split. Use image ID as the split unit. The paper reports 573 patients,
but this protocol has no verified image-to-patient mapping; therefore it
makes no patient-disjoint claim.

Resize RGB images directly from 2048 × 2048 to 512 × 512 with Lanczos
downsampling. Resize binary vessel masks with nearest-neighbor interpolation,
then threshold at 0.5. Convert RGB to float in `[0,1]`. No official field-of-
view (FOV) masks are supplied for this task. Set the fixed FOV to ones over
the complete 512 × 512 image for every model, loss, and metric; never infer a
mask from the vessel target or image darkness. Where a training crop extends
beyond the image boundary, zero-pad RGB, coarse probabilities, and target, and use a
fixed support mask that is one on real image pixels and zero only in padding.
The padding support mask is not an estimated retinal FOV.

## Coarse teacher and frozen predictions

Train a width-16, four-level U-Net with GroupNorm and ReLU. Use random
128 × 128 crops, batch size 8, AdamW with learning rate 0.001 and zero weight
decay, and loss `0.5 * masked BCE + 0.5 * soft Dice`, with `+1` smoothing.
Each teacher uses 1200 optimizer updates except the final teacher trained on
all 480 training images, which uses 1600. The five teacher seeds are
201101–201105: seeds 201101–201104 belong to folds 0–3 and seed 201105 to the
full-training teacher.

Create four fixed, class-balanced folds from the 480 development-training
images. Within each diagnosis class, sort the non-validation IDs numerically
and assign ranks modulo four to folds 0–3. Each fold therefore contains 120
images (30 per class). For each fold, train a teacher on the other 360 images
and predict the held-out 120, giving one out-of-fold (OOF) prediction for
every NCA training image. Separately train the fifth teacher on all 480 and
predict the 120 validation images. Never use an in-sample full-training
teacher prediction as an NCA training input. Freeze and hash the teacher
weights, OOF predictions, validation predictions, split, and preprocessing
configuration before starting NCA training.

At 512 × 512, obtain each teacher's probability map with overlapping
256 × 256 tiles at stride 128 and Gaussian overlap weights, without resizing
the 512-pixel grid. Channel 0 of the NCA initial state is the logit of the
corresponding coarse probability after clamping it to `[1e-4, 1-1e-4]`. The
other 15 channels start at zero. The coarse model is initialization only; it
is not concatenated as a separate static NCA input and is not recomputed
during recurrence.

## NCA cell and matched arms

The visible work state has 16 channels. At each step, a learned 3 × 3 state
perception maps the 16 state channels to 64 features, and a separate learned
3 × 3 perception maps static RGB to 64 features. Concatenate these features,
apply a 128-unit ReLU layer, and project to 16 state increments. The RGB image
is the only static conditioning input and remains fixed throughout a rollout.
The same learned transition and parameter shapes are used by all arms.

For state `x`, learned increment `F`, fire mask `m`, and fixed support mask
`q`, one step is `x_next = q * (x + m * F(x, RGB))`. Each cell fires with
probability 0.5 at each step; one scalar fire decision is shared across all
16 channels. On real image pixels `q` is always one. There is no alpha/alive
rule and no label-derived FOV. Inference for every arm uses the common
materialized 16-channel transition, including checkpoints trained with AU.

| Arm | Training rollout and gradient rule |
|---|---|
| Original K64 | 64 steps and terminal-only loss. Use exact activation checkpointing in 8-step chunks; retain the full 64-step gradient. |
| Original K8 | 64 steps and terminal-only loss. Detach the complete recurrent state at each 8-step boundary; only the final 8-step window carries recurrent feature-parameter gradients. |
| AU-K8 | Same visible transition as Original K8, with a 16-channel base plus a 129-channel accumulator containing 128 post-ReLU features and a constant-one feature. Keep the accumulator's numeric value across K8 boundaries while detaching its feature history, so the output projection `W` retains direct historical credit. Perception and hidden-feature parameters remain K8-truncated; this does not restore their historical gradients. |
| T8 reference | Separate 8-step, terminal-only training run with full gradients through its 8 steps. It is used only for the frozen K64 depth-qualification comparison. |

For AU-K8, initialize base `b` from the coarse-start state and accumulator
`e` to zero at each optimizer update. Within a rollout, the materialized
visible state is represented as `b + W e`. The accumulator retains numeric
features across truncation boundaries; gradients through earlier feature
computations are detached. No parameter is warm-started from a different
arm. At matched parameters, AU and Original have the same forward function.

## Training schedule

Use one paired block with model-initialization seed **200101** and schedule
seed **200201**. All arms start from bit-identical initial parameters. For
Original K64, Original K8, and AU-K8, bind the same training crops and 64-step
fire masks update by update. The T8 reference uses the same sampled training
windows and the first eight fire masks from the bound schedule. Record the
RNG implementation and seed bindings in the run manifest.

For NCA training, sample a 192 × 192 crop containing a 64 × 64 scored center
and a 64-pixel halo on each side. Choose the scored-center location uniformly
from the 512 × 512 training image; use the fixed zero-padding rule above at
image boundaries. The scored loss uses only real pixels in the center. Use
batch size 2 and shared quarter-turn rotations and horizontal/vertical flips
on RGB, coarse input, target, and support mask. Do not apply photometric
augmentation or target-derived crop sampling.

For update `u`, sample windows from `numpy.default_rng(200201 + 104729*u)`.
Generate one 64-step fire plan from a device-local PyTorch generator seeded
with `200201 + 10000000 + 104729*u`; the T8 reference uses that plan's first
eight steps. Reuse these stateless per-update bindings across paired arms.

Train each started model for exactly 1500 optimizer updates with AdamW,
learning rate 0.001, and zero weight decay. Normalize each parameter
tensor's gradient by its own L2 norm plus `1e-8` before the optimizer step.
For each update, run the arm's full rollout and apply one terminal loss only;
do not add intermediate-step losses. Compute BCE over valid scored-center
pixels. For each image in the batch, compute soft Dice loss
`1 - (2 * sum(p*y) + 1) / (sum(p) + sum(y) + 1)` over those pixels, average
the image Dice losses, and use `0.5 * BCE + 0.5 * Dice`. Do not use class
weights or other auxiliary training losses.

Save model checkpoints at updates 250, 500, 1000, and 1500, with an atomic
latest-resume checkpoint every 25 updates. Preserve the configuration,
model/optimizer state, source and data hashes, update count, schedule
bindings, and run status needed to distinguish completed, interrupted, and
failed runs. Use new output directories for this experiment; do not overwrite
the DRIVE screen or the historical AU-NCA run.

## Evaluation and aggregation

Evaluate the validation set at full 512 × 512 development resolution. At
updates 250, 500, and 1000, report T64 diagnostics on a fixed 16-image cohort:
the first four numerically sorted validation IDs in each diagnosis class.
The T8 reference is evaluated only at T8 on this cohort. At update 1500,
report all frozen gates and endpoints on all 120 validation images.
Intermediate checkpoints describe formation only; the selected endpoint is
fixed at update 1500.

For each validation image and repeat index `r` in `{0,1}`, seed the device
fire generator with `200401 + int(SHA256(image_id)[:8], 16) + 104729*r` and
generate a 256-step plan. Share each plan across all arms and checkpoints,
and reuse its prefix at T8, T16, T32, T64, T128, and T256. These repeated
plans are stochastic measurements, not independent samples. The three main
arms (K64, K8, and AU-K8) receive all six horizons at update 1500 on all 120
images and T64 only at updates 250, 500, and 1000 on the 16-image cohort. The
separate T8 reference is evaluated only at T8, on all 120 images at update
1500 and on the 16-image cohort at diagnostic checkpoints. Keep all
per-image, per-plan, and per-checkpoint results.

The primary endpoint is equal-image-weight mean clDice at T64. Mean per-image
Dice at T64 is the non-degradation guard. Threshold predictions and target
masks at 0.5; do not tune the threshold. Report each image's plan-averaged
metrics before computing the equal-image mean. Also report precision, recall,
and absolute errors in beta0 and beta1. For topology counts, beta0 is the
number of 8-connected foreground components; beta1 counts enclosed holes
using 4-connected background, with image-boundary background treated as
exterior. These binary-mask metrics do not establish anatomical correctness.
The [clDice paper](https://arxiv.org/abs/2003.07311) describes the centerline
metric; the metric implementation and version must be recorded with the run.

For the evaluation-only hidden-state intervention on K64, K8, and AU-K8,
preserve the T32 output, then zero hidden state channels 1–15 while keeping
channel 0 and static RGB. Continue with the same suffix of the matched fire
plan at the final update-1500 checkpoint. Run this diagnostic only for the
fixed 16-image cohort, retain the intact and reset trajectories, and do not
use it for checkpoint selection or the primary gate. The T8 reference has no
hidden-reset evaluation.

## Run order and frozen decisions

All thresholds below are design choices for this developmental screen, not
clinical standards or universal cutoffs.

1. Check the full-training-teacher coarse baseline on all 120 validation
   images. Require mean Dice at least 0.60 and mean clDice at least 0.50. If
   either floor fails, retain the results and stop.
2. Train Original K64. Require its update-1500 T64 clDice to exceed the coarse
   baseline by at least 0.01, with mean Dice no more than 0.005 below coarse.
   Also require its T64 clDice to exceed the same checkpoint's T8 clDice by
   at least 0.01. If either immediate control fails, retain the results and
   stop before the T8 reference or short-credit comparison.
3. Train the independent T8 reference for 1500 updates. Complete K64
   qualification by requiring K64 T64 clDice to exceed the T8 reference's
   T8 clDice by at least 0.005. If it fails, retain all completed evidence
   and stop before Original K8 or AU-K8.
4. Only after complete K64 qualification, train Original K8 and then AU-K8
   for 1500 updates each. Define the credit gap as K64 T64 mean clDice minus
   Original K8 T64 mean clDice. It is decision-relevant only when at least
   0.01. Give AU-K8 the developmental-benefit label only if all conditions
   hold: its T64 clDice exceeds Original K8 by at least 0.005; it closes at
   least 25% of a positive, decision-relevant K64-to-K8 gap; and its T64
   mean Dice is no more than 0.005 below Original K8. Report continuous
   paired values and each gate whether it passes or fails.

A failed coarse or K64 control leaves the AU comparison unqualified; it is
not evidence for or against AU-K8. One paired block supports only a
single-block developmental conclusion, with no seed-population or
statistical-significance claim. Do not add a seed, tune a threshold, expand
the cohort, change resolution, or run another dataset under this protocol.
