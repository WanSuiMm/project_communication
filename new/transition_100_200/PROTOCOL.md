# Historical seed4 update100--200 transition audit v1

Protocol: seed4_dense_transition_v1. Authorized on 2026-10-03. One historical
training replay, no new architecture, objective, schedule, initialization,
state pool, topology or parameter sweep. No wall-clock limit or watchdog.

## Question and independent unit

Locate the training interval in which original Streaming seed4 acquires its
observed combination of continued acquisition and low destruction. This is a
descriptive within-training audit of ONE selected initialization on ONE fixed
bank/schedule. Checkpoints, maps, cells and rollout times are not independent
training replications. A physical phase transition, latent commitment,
contextual type equivalence or causal handoff is not established by this run.

## Frozen replay

Original StreamingCell, W24/Z8,5033 parameters, original two-hop macro clock.
Initialization4;512 size32 training maps seed10002; batch schedule20002,
300 updates,batch8; original new/short_bptt/training.py backward_trajectory
with K8,64 forward steps, losses8,16,...,64 divided by8, detach every8 steps.
AdamW lr.001,weight_decay.0001, clip norm1. FP32 PyTorch2.5.1,threads2,
cudnn benchmarkFalse/deterministicFalse/TF32True,matmul TF32False.
No discarded warm prefix is needed: the original historical helper is called
directly. No diagnostics or test-map forwards are interleaved with training.

Save parameters at0,100,105,...,200,300. Every saved tensor is cloned to CPU.
After training, require exact initial/final historical parameter hashes,
training-bank and full schedule identities, and exact100/200/300 anchors from
runs/warmstart_20261003_paired01. Verify saved reference checkpoint file hashes
against their original record. Reproduce all six published historical
size32/64 x T64/128/256 endpoint payloads. Any drift invalidates the audit;
do not change backend, select another seed, threshold, schedule or rescue run.

## Fixed audit inputs and output

Audit each checkpoint100,105,...,200 and the300 control (22 checkpoints).
Size32:16 maps,seed61032, identical to previous exploratory stage bank.
Size64:16 maps,seed61064, descriptive scale extension. Each map is paired with
its source-flipped counterpart. Record macro steps0..256 from fresh state.
Save Boolean original/flipped/paired correctness, paired signed margin,
changed/source/distance fields, per-step state RMS, endpoint task scores and
raw bank bindings. Check finite states/logits and the two-hop causal cone.
No hidden-state ground truth or transplant intervention is used.

Primary descriptive profiles on size32:
1. All-changed G,D and exact weighted net coverage gain64->128.
2. First-exit destruction and continuous survival64->128 and64->256,
   distinguished from endpoint destruction/retention.
3. Strict16<BFSdistance<32 paired coverage at64/128/256, equal-map and pooled.
4. First-correct survival at fixed follow-ups8/16/32/64/128, with explicit
   eligible denominators and horizon censoring; current-correct-run-age
   hazard with resets after regression. Distance and age strata are fixed.
Finite-horizon first-to-terminal-stable lag is secondary and conditioned on
final correctness. It is NOT a latent commitment time. Report never-correct,
final-wrong and late-entry counts. Size64 uses the same reporting definitions.

Report per-map rates/equal-map summaries and pooled count descriptions.
Undefined rates are null; no epsilon ratio is a primary order parameter.
All transition counts must satisfy delta_correct=acquired-destroyed exactly.

## Prespecified descriptive screen and claim boundary

A checkpoint meets the size32 solve-and-preserve screen only if strict pooled
AND equal-map T128 coverage>=.80, all-changed pooled64->128 endpoint G>=.20
(conditional acquisition among cells wrong at64, not a20pp coverage gain),
pooled first-exit destruction64->128<=.01 and continuous survival64->256>=.95.
This is a NEW descriptive screen, not the historical formal phenotype gate.
Report the earliest saved checkpoint followed by two more consecutive saved
checkpoints meeting this screen (three-checkpoint persistence), or NONE.
Report every raw profile even if this screen fails. Do not refine the5-update
grid or alter the screen after seeing results. A narrow crossing locates an
operational behavioral onset; it does not identify an invariant representation
or distinguish maturation from survivor heterogeneity.

## Qualification and execution

Before formal dispatch: focused hand-computed CPU metric fixtures, verify
source/reference hashes and all anchor checkpoint tensors;3-update CUDA
smoke with all six learned groups receiving finite/nonzero gradients by3;
full256-step paired smoke on both fixed16-map banks; available GPU headroom.
Bind formal launch to these exact source hashes. Stop on nonfinite values,
reference/source drift or historical replay mismatch. Keep failed records.
Publish no machine identifiers, execution receipts or checkpoints by default.
The runner saves source snapshots, durable status, checkpoints and summaries
under a new owning-project runs/ directory. Dispatch verification is separate
from ongoing monitoring; no monitor is installed.

Commands from repository root (fresh output names):

    python -X utf8 -B new/transition_100_200/check.py --out analyses/NEW_TRANSITION_CPU.json
    python -X utf8 -u -B new/transition_100_200/run.py --preflight --qualification analyses/NEW_TRANSITION_CPU.json --out runs/NEW_TRANSITION_PREFLIGHT
    pwsh -File tools/launch_transition_100_200.ps1 -RunName NEW_TRANSITION_RUN -Preflight runs/NEW_TRANSITION_PREFLIGHT
