# Real-task NCA developmental screen on DRIVE

This frozen, single-block developmental screen asks whether an accumulated
projection-credit variant improves a local recurrent vessel refiner on a
small real-image task. It is not a confirmatory result, an anatomical-validity
claim, or an exact reproduction of rNCA. The [official DRIVE site](https://drive.grand-challenge.org/)
defines the retinal vessel task, FOV masks, and hidden test-label evaluation;
its test leaderboard reports per-image Dice inside the FOV, not clDice. The
[rNCA paper](https://arxiv.org/abs/2512.13397) motivates iterative local mask
refinement. The authors' [rNCA repository](https://github.com/maltesilber/rnca)
currently documents a minimal synthetic training example, so this protocol
does not claim a DRIVE rNCA reproduction. The [clDice paper](https://arxiv.org/abs/2003.07311)
and [reference implementation](https://github.com/jocpae/clDice) define the
hard centerline metric used here.

## Frozen split and data boundary

Use DRIVE training-image IDs 21 through 40. The outer validation set is
21, 26, 31, and 36; the other 16 images are the only NCA training set. Make
four fixed inner folds of four images. For each inner fold, train the coarse
teacher on the other 12 images and produce predictions for the held-out four,
giving out-of-fold coarse predictions for all 16 NCA training images. Train a
separate coarse teacher on all 16 training images and use it to predict the
four outer-validation images. Record the fold assignment, teacher
configuration, and predictions. Train the NCA only on each training image's
out-of-fold coarse prediction paired with its training vessel mask; do not use
in-sample full-16 teacher predictions for those training images. The outer
validation labels may be used only
for the frozen qualification and endpoint calculations below. Do not read
test ground truth, use the test set for selection, or tune a threshold or
checkpoint against it.

Use the coarse teacher's logit as work-state channel 0 and initialize the
other 15 work-state channels to zero. The RGB image is the only static image
conditioning input and is supplied at every NCA step. Training uses shared
quarter-turn rotations and horizontal/vertical flips on RGB, coarse map,
target and FOV. There is no photometric augmentation, simulated label-derived
corruption or validation-based checkpoint selection.

The coarse teacher is a width16, four-level U-Net (three downsamplings), with
GroupNorm and ReLU. Each inner teacher receives1200 updates, batch8, patch128;
the full16-image teacher receives1600 updates. AdamW uses lr0.001 and zero
weight decay; the loss is0.5 masked BCE plus0.5 soft Dice with+1 smoothing.
Predict full-resolution probabilities with overlapping256-pixel tiles,
stride128 and Gaussian overlap weights, without resizing. Teacher seeds are
201101 through201105. Freeze coarse weights and predictions before NCA training.

## Shared cell and arms

The common work state has 16 channels: the vessel logit in channel 0 and 15
hidden channels. A learned 3-by-3 state perception maps those 16 channels to
64 features; a separate learned 3-by-3 RGB perception maps the three static
image channels to 64 features. Concatenate the two feature maps, apply a
128-unit ReLU layer, and project to 16 state increments. All arms use the
same parameterization and forward function at matched parameters:

At each cell, add the projected increment multiplied by the scalar fire
decision and fixed FOV mask. The FOV mask gates updates and losses; it is never
recomputed from the evolving state.

| Arm | Gradient treatment during a 64-step training rollout |
|---|---|
| Original K64 | Full backpropagation through all 64 steps. Use exact activation checkpointing in 8-step chunks; checkpointing must preserve the full gradient and must not truncate it. |
| Original K8 | Detach the recurrent state at 8-step boundaries; with a terminal-only loss, only the final 8-step window carries a recurrent feature-parameter gradient. |
| AU-K8 | Use the same visible transition with a 16-channel base b and a 129-channel accumulated-feature state e. Keep e's numeric value across K8 boundaries while detaching its history, so the final projection retains direct credit through accumulated features. Feature-extractor gradients remain K8-truncated. |

In AU-K8, e stores the 128 post-ReLU features plus the constant-one feature;
the projected visible state is represented as b + W e. At an optimizer-update
boundary, initialize b from the coarse-start state and e to zero. Within a
rollout, apply the same fixed FOV mask and scalar cellwise fire mask in all
arms. Fire independently with probability 0.5 at each step and spatial cell,
sharing each fire decision across all 16 channels. There is no alpha or RGBA
alive rule. Use paired fire masks and paired training windows for the arms.

An additional evaluation-only intervention resets the15 hidden channels at
T32, after recording its prediction, while preserving channel0 and static RGB.
It uses the identical subsequent firing plan as the unperturbed evaluation.
No reset is applied during training or the primary evaluation. All inference
uses the common materialized16-channel transition at fixed weights, including
AU-trained models. The AU-specific accumulator can be eliminated exactly at
fixed parameters. Resetting only its representation to b=x,e=0 would preserve
inference and would not be a meaningful learned-state intervention.

## Training procedure

Use one paired block with model-initialization seed 200101 and schedule seed
200201. Each training input is a 192-by-192 crop consisting of a 64-by-64
scored center and a 64-pixel halo on every side. Compute losses only over the
center pixels inside their fixed FOV. Each optimizer update uses batch size 2,
a64-step rollout, and one terminal loss; do not apply intermediate-step
losses. BCE is averaged over all valid center FOV pixels in the batch. Soft
Dice for each image is1-(2 sum(p*y)+1)/(sum(p)+sum(y)+1), where p is the sigmoid
probability and all sums cover the same valid center pixels. Average Dice
losses across images and use0.5 BCE+0.5 Dice. Use no class weights or extra losses.
Choose crop centers uniformly from FOV pixels without using the target mask.

Train each started model for exactly 1,500 optimizer updates. Use AdamW with
learning rate 0.001 and weight decay 0. For each parameter tensor, divide its
gradient by its own L2 norm plus 1e-8 before the optimizer step. Save
checkpoints at updates 250, 500, 1,000, and 1,500. The update schedule binds
training windows and fire masks across paired arms; record the schedule and
its seed binding with the run.

Run the control sequence in this order:

1. Check the coarse support floor, then train Original K64 and evaluate it.
2. If the coarse baseline passes the support floor and K64 has no immediate
   failure on its own control checks, train a separate T8 auxiliary model for
   the same 1,500 updates using an 8-step rollout and terminal T8 loss.
3. Apply the complete K64 qualification below. Start Original K8 and AU-K8
   only if that qualification passes. Train Original K8 before AU-K8.

If qualification fails, preserve every completed checkpoint and metric, mark
the task unqualified, and stop before the short-credit comparison. Do not
interpret an unqualified task as evidence for or against AU-K8. Do not add a
fresh seed or automatically expand this single-block screen.

## Evaluation and aggregation

Evaluate each started checkpoint from the cold coarse initialization on the
full outer-validation images at T8, T16, T32, T64, T128, and T256, with no
gradient computation. Generate two matched 256-step fire plans per outer
validation image, reuse each plan's prefix at every horizon, and use the same
plans for all arms and checkpoints. These are repeated measurements for
stochastic firing, not independent samples. Threshold sigmoid probabilities
at 0.5 for every metric; do not tune the threshold. For each original image,
average the two plan-level values,
then average the four image-level values with equal image weight. Model
selection is fixed to update1500; intermediate checkpoints only describe
formation. Never pool
pixels, tiles, plans, or time steps as independent observations. The primary
endpoint is mean per-image clDice at T64; mean Dice is the non-degradation
guard. Keep all per-image, per-plan, and per-checkpoint values.

The per-image API is
segmentation_metrics(prob, target, fov) in metrics.py. It accepts normalized
NumPy arrays or torch tensors with one image's spatial dimensions (singleton
batch/channel axes are accepted). It returns dice, cldice, precision, recall,
betti0_error, and betti1_error. It thresholds probability and target at 0.5,
skeletonizes each full binary mask before restricting counts to the FOV, and
uses the fixed FOV for every metric. The runner clips probabilities and targets
to the FOV before passing them to this API, including the coarse baseline, so
outside-FOV logits cannot change the skeleton. If both foregrounds are empty inside the
FOV, Dice, clDice, precision, and recall are 1; if only one is empty, they are
0. NaN and infinite input values are rejected.

For topology counts, beta0 is the number of 8-connected foreground
components. Beta1 is the number of enclosed background holes using the dual
4-connected background convention; a background component touching the FOV
boundary is exterior. Report absolute prediction-to-target beta0 and beta1
errors as auxiliary metrics. These counts and clDice describe binary masks;
they do not guarantee anatomical correctness.

## Frozen qualification and decisions

All thresholds below are design choices for this developmental screen, not
theorems or universal clinical cutoffs. First require the coarse baseline on
the four outer-validation images to have mean Dice at least 0.60 and mean
clDice at least 0.50. Then require all of the following for the K64 control:

1. K64 T64 mean clDice exceeds the coarse baseline by at least 0.01, and its
   mean Dice is no more than 0.005 below the coarse baseline.
2. K64 T64 mean clDice exceeds the same K64 checkpoint's T8 mean by at least
   0.01.
3. K64 T64 mean clDice exceeds the separately trained T8 model's T8 mean by
   at least 0.005.

Any failed condition leaves the task unqualified; do not run or interpret the
short-credit comparison as an AU result.

After K64 qualification, define the credit gap as K64 T64 mean clDice minus
Original K8 T64 mean clDice. The gap is decision-relevant only when it is at
least 0.01. AU-K8 receives the frozen developmental-benefit label only when
its T64 mean clDice exceeds Original K8 by at least 0.005, closes at least 25%
of that positive K64-to-K8 gap, and its T64 mean Dice is no more than 0.005
below Original K8. Report the continuous paired values and all component
gates whether they pass or fail. One paired block supports only a
single-block developmental conclusion; it supports no seed-population or
statistical-significance claim.
