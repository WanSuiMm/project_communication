# Spatial Nonlinear Lift v0: three-block developmental screen

Frozen before qualification and scientific training. New experiment; all
earlier results and both user-provided sanity scripts remain unchanged.

## Question and intervention

Can exact parameter deferral remove a terminal K8-versus-K64 learning gap
for genuinely state-nonlinear local writes on a cyclic 2D grid, at an
acceptable measured state/memory/time cost? See THEORY.md for the induction.
Original and lifted cells have identical nine parameters a,b,gamma and
identical real-arithmetic forward functions. Only persistent state and
parameter placement differ. The lifted state is locally updated online,
not provided as precomputed input. No custom backward or eligibility trace.

Three arms per paired block: original_k8, original_k64, lifted_k8. Train
exactly 300 updates with batch16, terminal T64 dense mean squared error,
one optimizer step per rollout. AdamW lr=.01, weight_decay=.0001,
betas=(.9,.999), eps=1e-8, gradient clip1. No AMP/TF32/compile/CUDA Graph,
auxiliary supervision, hyperparameter sweep, or teacher parameter transfer.
Init seeds180101,180102,180103; schedule seeds180201,180202,180203.
Model selection is fixed at u300; u100/u200 are diagnostics only.

Original K8 runs the first56 updates under no_grad, detaches state, and
differentiates steps57..64. Original K64 differentiates all64 steps. Lifted
K8 cuts the raw moment graph at the same point, performs the remaining local
updates, and contracts the terminal moments with live a,b,gamma. There is
no readout EMA. All three student parameter initializations match exactly.

## Task, input support, and labels

Features have shape [B,T,H,W,3]. Both spatial operators are periodic local
permutations: P=one-row forward shift, Q=one-column backward shift.
In the first half of an episode, channels0/1 carry independent per-map
Gaussian source variables plus static spatial variation and fresh noise;
channel2 has weak noise. In the second half, channel2 carries an independent
query variable with spatial variation and fresh noise; channels0/1 have
weak fresh noise. Late noise keeps every parameter's suffix path active,
without repeatedly revealing the earlier source. Inputs are independent of
all teacher/student parameters; latent diagnostic arrays are never student
inputs. Exact coefficients are in tasks.make_bank.

A hidden, fixed original-cell teacher uses a=(.8,-.6,.2), b=(.1,.15,.9),
gamma=(.05,.4,.8). Labels are its terminal dense tanh(z_T/T) field. Teacher
parameters are recorded for reproducibility but never loaded into a student.
This is an intentionally task-matched mechanism test, not an unrestricted
vision task. It tests learnability of shared projections and polynomial
coefficients; it does not train a nonlinear encoder or content routing.

Banks are shared across blocks and arms:

| Bank | Maps | Side | Horizon | Seed |
|---|---:|---:|---:|---:|
| train | 256 | 8 | 64 | 180301 |
| heldout (primary) | 64 | 8 | 64 | 180401 |
| long128 | 32 | 8 | 128 | 180501 |
| long256 | 32 | 8 | 256 | 180502 |
| spatial16 | 32 | 16 | 64 | 180601 |

Each longer episode retains a source phase lasting half its own horizon;
it is a new teacher-labeled sequence, not a continuation-stability test.
Maps are independent examples. Paired training block, not cell, is the
independent unit for arm comparisons. Three blocks give a developmental
screen, not a population reliability estimate.

## One bounded qualification before dispatch

At initial and one fixed perturbed parameter setting, CPU float64 checks
all64 projected states/outputs, original-full versus lifted-full gradients,
and original-full versus lifted-K8 gradients (relative L2<=1e-9).
CUDA float32 checks actual batch16/grid8 shape with state/output
atol1e-4,rtol1e-4 and gradient relative L2<=1e-4. Tolerances stay frozen.
The state Hessian probe must be nonzero. Lifted K8 must have exactly zero
early-input gradient, nonzero suffix-input gradient, and nonzero gradients
for all three parameter groups. Original K8 must also have nonzero gradients
for each group; this rules out an all-paths-dead short-credit condition.
One matched optimizer step per arm measures execution and peak allocation.

On a fixed16-map teacher probe, target variance must exceed1e-4 and the
RMS output change from removing gamma_2 must be at least10% of full target
RMS. These are support checks, not tuned task-performance thresholds.
Failure stops dispatch and preserves qualification failure evidence.

## Frozen endpoints, qualification, and decisions

Primary: heldout dense R2 at u300, computed across the saved64 maps/cells.
Report per-map MSE, all predictions, R2/MSE, every update loss, intermediate
metrics, all9 individual trajectories, and paired differences.
Secondary: corresponding R2 on long128,long256,spatial16; no endpoint mixing.

Run block00 original_k64 first as an actual scientific arm. If its primary
R2<.5, stop with POSITIVE_CONTROL_UNQUALIFIED; retain the completed arm and
do not start the other eight. This is a predeclared stop, not runtime error.
If it qualifies, complete all9 arms. The overall comparison requires all
three original_k64 endpoints R2>=.5. Otherwise it remains control-unqualified.

For qualified controls, let paired B-A be original_k64 minus original_k8,
and C-B be lifted_k8 minus original_k64:

- If max absolute(C-B)>.02: EQUIVALENT_TRAINING_DIVERGENCE_DEVELOPMENTAL.
- Otherwise if mean(B-A)>=.10 and at least2/3 differences are positive:
  SHORT_CREDIT_RECOVERY_DEVELOPMENTAL.
- Otherwise: NO_MATERIAL_SHORT_CREDIT_GAP_DEVELOPMENTAL.

Report continuous values regardless of category. These thresholds are
design choices. Matching B and C is predicted by the theorem; the empirical
questions are whether A has a meaningful gap and the measured state/systems
tradeoff. Neither success nor failure establishes generic NCA trainability.

## Execution, protection, and provenance

Local CUDA float32 on the established workstation, device0. Fresh run
directory only, exact source/config/data/schedule hashes and copied source.
Atomic latest checkpoint each update; named u100/u200/u300 checkpoints.
An explicit --resume requires unchanged source/config/data bindings; no
automatic restart. The existing independent on-demand protected worker has
no runtime limit, temporarily prevents idle sleep, and installs no recurring
trigger. Launch receipts and machine identifiers stay local.

From repository root:

```powershell
python -X utf8 -B new/spatial_nonlinear_lift/run.py --check --out analyses/NEW_SPATIAL_CHECK
& ./tools/start_protected_job.ps1 -JobName NEW_SPATIAL_RUN -Script new/spatial_nonlinear_lift/run.py -ScriptArguments @('--out','runs/NEW_SPATIAL_RUN','--qualification','analyses/NEW_SPATIAL_CHECK/qualification.json')
```
