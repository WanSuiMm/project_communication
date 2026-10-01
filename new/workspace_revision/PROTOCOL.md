# Workspace + revision: paired exploratory screen v1

Frozen before efficacy training, 2026-10-02. The user authorized this new
experiment. Earlier stops on training are superseded only for this bounded
screen. No theoretical stability guarantee is a prerequisite.

## Question and intervention

Does candidate revision of task state improve reach, hold and reopen compared
with additive task-state updates, under the same architecture and training?

Both arms have W24 (workspace), Z8 (task state), a W encoder, and readout from
Z only. Initial Z is zero. On the supplied binary four-neighbor medium:

    Wnew = W + 0.1 F(W, Z, Lm W, Lm Z, X)
    Q = Qnet(Wnew, Z, Lm Wnew, Lm Z, X)
    additive: Znew = Z + 0.5 Q
    revision: Znew = Z + 0.5 (Q - Z)

F is 67 -> 40 -> 24; Qnet is 67 -> 16 -> 8. Both use pointwise convolutions,
Tanh hidden activation and zero-initialized final layers. Encoder is 3 -> 24;
readout is 8 -> 1 with zero initial bias. Alpha is fixed0.5. Q has no imposed
output range: no output squashing, clamping, learned gate, energy or extra
supervision. Because Q uses a Tanh hidden layer, fixed finite weights do imply
a finite candidate bound (absolute final weights plus bias); this bounds Z
under revision but does not imply stable decisions or bounded workspace.
Both arms have identical parameters and seed-specific initial parameter tensors.
Proposal scaling matches at Z=0; no old checkpoint is reused.

One macro-step has TWO sequential masked communication phases in both arms.
Do not compare macro-step speed directly with the old one-phase NCA. This
isolates update parameterization within a shared workspace split, not the split
itself. No full-system contraction, idempotence or repair guarantee is claimed.
Prior checkpoints/negative results are read-only contextual evidence, not
matched controls for this training recipe.

## Data and training

- Frozen seeded-component task and masked Laplacian. Input is mask plus
  positive/negative sources, never component IDs, distances or targets.
- Paired seeds0,1; 512 training maps at size32, bank seed10000+s; batch8.
  600 AdamW updates per arm; lr0.001, weight decay0.0001, gradient norm clip1.
  Float32, two CPU threads, no compilation or mixed precision.
- One precomputed schedule per seed from RNG20000+s, shared by both arms.
  Record exact tensor, schedule and initial parameter hashes.
- Each update trains reach from fresh initialization, T in {32,48,64},
  averaging balanced target BCE at T-4 and T.
- Detach reached state; advance under no_grad for U in {0,32,64} steps with
  unchanged weights, then detach again for the auxiliary branch.
- Cycle hold/switch/repair equally. Auxiliary K in {16,32};
  average target BCE at K-4 and K. Hold uses original input/target. Switch
  uses x_flip/y_flip WITHOUT resetting state. Repair zeros a side8 patch in
  Z or both W/Z with equal scheduled probability; original input/target.
  Patch coordinates are scheduled per example.
- Total loss = reach_loss + auxiliary_loss. Backpropagate both graphs before
  one optimizer step; no gradient crosses detach. Longest gradient chain64;
  largest auxiliary absolute age160. Same objective in both arms.
- No state pool, hidden oracle target, synthetic gradient, rescue sweep,
  extra seed or retuning.

## Evaluation

Use original fixed16 held-out maps per size32/64, seed30000+size, shared across
both training seeds. Maps are not independent model replicates. No evaluation
example selects weights or changes the protocol.

1. Fresh original/source-flipped rollouts at T16/32/64/96/128/192/256.
   Record BA, BCE, paired correctness on changed component, per-map metrics,
   distance bins, open W/Z RMS, next-update RMS and output change.
   Primary paired metric averages per-map fractions where BOTH source
   counterfactuals are correct; pooled pixels are secondary.
2. Hold = minimum aggregate BA at T128/192/256. Report drop from BA64.
   This concerns sampled horizons, not all-time stability.
3. Source switch from original T64 state, K8/16/32/64/128: new-target BA,
   changed/unchanged component accuracy, cold restart with equal extra steps.
   Warm state is never re-encoded.
4. Damage from original T64, fixed side round(size/4) patch, RNG50000+size:
   Z-only and W/Z together, immediate and same K endpoints, clean continuation
   and cold restart controls. Unconditional metrics plus conditional BA for
   maps with original BA>=0.95, eligibility counts; none eligible means null.
   Primary damage endpoint Z-only K64; W/Z damage is secondary.
5. Batch1 size32 rollout64 timing: two warmups, three repeats, CUDA synchronized;
   includes initialization/readout, records memory. No accuracy-matched speedup.

## Frozen decision (size32)

Model seed is independent unit: two paired replicates, descriptive only.
Primary effect = mean paired revision-minus-additive hold BA. Pass requires:

- All four runs finish600 updates and all prescribed evaluation is finite.
- Mean hold improvement >=5pp, positive in both seeds.
- EACH seed: revision BA64>=0.85, paired64>=0.50, BA64 at most3pp below additive.
- EACH seed: revision hold BA at most3pp below its own BA64.
- EACH seed: warm-switch K64 changed accuracy>=0.80, unchanged accuracy>=0.85,
  and neither is more than3pp below additive.
- EACH seed: Z-only repair K64 BA>=0.85, at most3pp below additive.

Report every failed predicate and effect. Small differences are not equivalence.
Both models succeeding supports the shared recipe without a revision advantage.
Failure to reach is qualification failure for this recipe, not impossibility.
Size64, long-horizon paired curves, conditionals, workspace motion and latency
are secondary and cannot rescue the primary gate.

## Execution and provenance

Local available CUDA device; order seed0 additive, seed0 revision, seed1
revision, seed1 additive. Whole-run scheduling cap25 minutes; retain partial
work and missing arms. Daemon watchdog at cap+60s writes a timeout marker and
exits if normal checks are blocked. Nonfinite loss/gradient/state stops its arm.
No automatic retry or monitoring.

Pre-dispatch engineering amendment: the first CUDA preflight measured about
0.655 seconds/update on the longest branch after initial CUDA startup.
The initially proposed800 updates/arm were reduced to600 for BOTH arms so
all four jobs should fit the25-minute cap. No efficacy training had started.
The original preflight is retained; the amended source hashes require a fresh
preflight. No model, objective, thresholds, evaluation maps or seeds changed.

Before launch: one tiny CPU model check, then three full-width CUDA updates per
arm covering hold/switch/repair and abbreviated evaluation. Preflight has no
efficacy interpretation. Formal launch requires matching passed source hashes.

New outputs under runs/workspace_revision_*. Save source snapshot/hashes, config,
data/schedule hashes, parameter hashes, checkpoints, raw arm JSON, training log,
aggregate and RESULTS. Separate local host/PID/GPU/command receipt stays out of Git.

Commands from repository root:

    python new/workspace_revision/check.py
    python new/workspace_revision/run_revision.py --preflight --out runs/NEW_REVISION_PREFLIGHT
    pwsh -File tools/launch_revision.ps1 -RunName NEW_REVISION_RUN -Preflight runs/NEW_REVISION_PREFLIGHT

Verified dispatch completes this launch request. No ongoing monitor is created.
