# C8 read parameterization and R2 factorial qualification

Protocol: `latent_read_sidecar_factorial_v1`. Frozen before efficacy training.
32 fresh paired initialization/schedule blocks, four arms each, executed
serially on the local historical CUDA backend. No runtime cap, watchdog,
recurring monitor, efficacy early exit, replacement, sweep or rescue training.

## Question and interventions

Does read-matrix factorization, a stationary self-conditioned R2 sidecar, or
their combination make long continuation more reliably learnable with K8
credit? This tests parameterization and state structure, not a human-readable
latent, recovery of a source bit, or a universally sufficient dimension.

| Arm | Read parameterization | Persistent learned state | Parameters |
|---|---|---|---:|
| A: native | original dense C8 read | C8 + Z8 | 2521 |
| B: factorized | independent F/Q identity-initialized U8 | C8 + Z8 | 2649 |
| C: native_r2 | original dense C8 read | C8 + Z8 + R2 | 3275 |
| D: factorized_r2 | independent F/Q identity-initialized U8 | C8 + Z8 + R2 | 3403 |

C8 retains four fixed directional lanes, two channels each; the masked
streaming permutation and Laplacian are unchanged. Z8 stays stationary and
owns the readout. F/Q hidden widths remain 40/16, eta=.1, alpha=.5. All arms
copy exactly the same native core tensors within a block. C and D additionally
copy identical sidecar tensors. Constructing widened layers from the same RNG
seed alone is insufficient because it changes fan-in and RNG consumption.

For B/D, only the C and L(C) blocks of each first read layer are factored:

    M_F,C = B_F,C U_F; M_F,L = B_F,L U_F
    M_Q,C = B_Q,C U_Q; M_Q,L = B_Q,L U_Q

U_F and U_Q are independent trainable 8x8 matrices, initialized to identity.
B blocks, remaining read blocks, biases, encoder, output heads and readout
start as exact native copies. No new nonlinearity appears between U and B.
U is applied to the incoming carrier at the original read clock: an arbitrary
U does not commute with directional transport. It never transforms the
persistent carrier or changes its directional identity. A/B (and C/D) have
the same representable first-layer read maps; their optimizer geometries differ.

C/D add the following update, with beta=.1, G hidden width16:

    R+ = R + .1 G(C,R,Z,L(C),L(Z),X)
    C' = T(C) + .1 F(T(C),Z,L(C),L(Z),X,R+)
    Z' = Z + .5 Q(C',Z,L(C'),L(Z),X,R+)
    logits = readout(Z')

G is 37->16->2 with tanh hidden activation. R0=0 and the entire G output
head starts at zero. F/Q R columns are normally initialized, before tanh;
the existing C/Z/L(C)/L(Z)/X blocks remain exact native copies. G reads R;
there is no L(R), direct R transport, direct R readout or output injection.
R is readout-decoupled, not proven execution-only. This differs from the old
H12 additive-output sidecar. It adds parameters, nonlinear computation and
memory simultaneously, so a gain cannot isolate memory necessity or capacity.
Z already contains possible readout-nullspace memory. Full recurrent dynamics
have no stability or continuation guarantee. Maximum macro-step radius stays2.

## Training, matching and runtime qualification

Initialization seeds94001..94032; schedule seeds95001..95032. Extension tensors
use a separate deterministic RNG stream, shared by C/D. Every 300x8 batch
schedule is fixed before training, drawing with replacement from the unchanged
size32 512-map bank seed10002. Arm order cyclically rotates by block index.

Exactly300 updates, batch8, FP32, historical Torch2.5.1 backend, AdamW lr=.001,
betas=(.9,.999), eps=1e-8, global clip1. Ordinary tensors including B retain
wd=.0001. Only the two U matrices have wd=0, avoiding double decay of a
factored effective read. Each update initializes task state, executes64 steps,
averages balanced losses at8/16/.../64, performs eight backwards with state
detach after each8-step window, then makes one optimizer step. No new loss,
curriculum, state pool, distillation, state/moment transplant or checkpoint
selection. Evaluate only update300.

Zero F/Q output heads initially suppress the new paths' task gradients. The
first gradient opens Q_out; other paths can open on later updates. We do not
require first-update U/G gradients or optimizer-equivalent trajectories.
Before launch, one combined qualification checks copied tensors, nontrivial
initial projected dynamics with temporarily nonzero matched F/Q tails,
factor orientation, sidecar/self-consumption and readout isolation, plus five
eager versus captured CUDA updates for each arm on the actual batch8/size32
shape. It records when the U/G gradients first become nonzero. Failure to
open within five smoke updates blocks launch pending a concrete diagnosis.
The qualification uses nonformal seed97999 and an eight-map seed97432 bank;
it is plumbing/runtime evidence, never a scientific efficacy endpoint.

CUDA Graph captures only the fixed K8 forward/backward arithmetic. AdamW,
clip1 and finite checks remain eager. Every learned state component, including
R, is checked for finiteness. Compare gradients, final parameters and Adam
states/groups for five updates; require bitwise equality. This finite test
qualifies the tested shape and short replay, not all training trajectories.
No AMP, fused optimizer, numerical backend change or silent eager fallback.

## Evaluation and frozen endpoints

All arms use the same existing evaluation banks, exactly32 maps each at
sizes32/64, seeds99332/99364. These previously used banks are fixed evaluation
cohorts; blocks are fresh but maps are not a newly sampled test cohort. No
evaluation gradients or training feedback. Save paired source-flip Boolean
traces at every integer time0..256, existing T64/128/256 metrics, reach,
retention, sustained progress/regression and matched frontier support.
The unmodified `new/seed4_followup/phenotype.py:predicate` Full gate is binding.

Independent unit is a paired training block, not pixels, maps, horizons or
timing repetitions. Primary D-A: paired Full wins/losses, success frequencies,
delta and exact two-sided McNemar/binomial p. Reliability qualification
requires net gain>=8/32 AND p<=.05. Report Wilson95 for each arm and all paired
outcomes. Primary failure stays failure even if B or C performs better.

Secondary B-A and C-A report raw and Holm-adjusted p over those two contrasts.
The factorial interaction is the mean block-level binary contrast
D-C-B+A, reported with its complete distribution and a descriptive t-based
95% interval; it is not a second confirmatory gate. A D win alone does not
prove synergy. Systems measurements report actual warmed batch8 forward64
latencies, peak allocation, parameters and all persistent learned bytes at
both sizes, separately from training variability.

Formal failure is no D-A reliability qualification. Better T64 accuracy with
failed Full is not the intended success. A negative result rejects this
particular factorized-read/R2 recipe, not all sidecars, latents or NCA. A
positive result concerns this task and fixed evaluation cohort and still
needs independent replication and capacity/stateless controls.

## Outputs and dispatch

New output directory only; prior source/evidence is read-only. Bind complete
source dependency hashes, protocol, plans, data and initial/final tensors.
Retain all128 curves, final checkpoints, compact aggregates, component
summaries, compressed traces and frontier CSVs. Local receipts also record
host, PID, GPU and command; they are not publication-ready. No full floating
state trajectories are saved. Nonfinite values, source/data/plan mismatch or
CUDA failure stop with ERROR and retained partial data; no efficacy rescue.

Repository-root commands:

    python -X utf8 -B new/latent_factorial/run.py --check --out analyses/NEW_FACTORIAL_CHECK.json
    pwsh -File tools/launch_latent_factorial.ps1 -RunName NEW_FACTORIAL_RUN -Qualification analyses/NEW_FACTORIAL_CHECK.json

Start reading run RESULTS.md and summary.json, then perarm.json/metrics.csv;
raw traces are secondary. Verified dispatch confirms the first real optimizer
update, not scientific completion. Runtime/storage estimates are refined from
the combined check and preceding local measurements, not used as cutoffs.
