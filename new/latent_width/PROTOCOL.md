# Native execution-state width qualification

Protocol: `native_latent_width_v1`. Frozen before efficacy measurements.
The user authorized widths 16, 4 and 2 in addition to the proposed W24,
native W8 and parameter-matched W24 control. Run all six arms serially in
16 fresh initialization/schedule blocks: 96 training trajectories. No time
limit, watchdog, continuous monitor, efficacy-based early exit or replacement.

## Question and architecture

Does reducing the native execution carrier improve the frequency of learning
long continuation with K8 credit under the existing task/training recipe?
There is no encoder/decoder fitted to old W, distillation, sidecar, semantic
channel assignment, new loss, gate, pool, curriculum or optimizer search.
Every model directly initializes, persists, transports and consumes its native
carrier. Z remains stationary with eight channels and owns the readout.

| Arm | Total carrier channels | F/Q hidden widths | Parameters | Transport |
|---|---:|---:|---:|---|
| w24 | 24 | 40/16 | 5033 | existing four-lane streaming |
| w8 | 8 | 40/16 | 2521 | existing four-lane streaming |
| w24_capacity | 24 | 16/12 | 2521 | existing four-lane streaming |
| w16 | 16 | 40/16 | 3777 | existing four-lane streaming |
| w4 | 4 | 40/16 | 1893 | existing four-lane streaming |
| w2_alternating | 2 | 40/16 | 1579 | two opposing lanes, alternating axes |

Widths 24/16/8/4 differ only in carrier width when F/Q stay 40/16. Their
channels divide equally into fixed N/E/S/W lanes. W24 capacity is a diagnostic
parameter-count control; changing its F/Q widths also changes the computational
parameterization, so it is not a complete mechanism isolation.

Two total channels cannot retain four independent directional lanes.
`w2_alternating` is therefore a separately identified exploratory transport
condition, not a width-only contrast. Its two lanes point along +/- of the
active axis; step 1 is vertical, step 2 horizontal, repeated thereafter.
Each axis operator is a masked port permutation with blocked-link bounce and
fixed wall values. No averaging, wrapping or 4/24-channel carrier rebuild.
The Markov state is `(C2,Z8,phase)`: phase is one global, nonlearned CPU scalar
bit, reset to zero at each initialization, passed through K8 detaches and
flipped out-of-place after each step. It is not a per-cell learned channel.
Two-channel results cannot identify a carrier-width effect alone.

All arms retain eta=.1, alpha=.5, zero-initialized F/Q output heads, and
the original residual feature clock:

    C' = T(C) + .1 F(T(C), Z, L(C), L(Z), X)
    Z' = Z + .5 Q(C', Z, L(C'), L(Z), X)
    logits = readout(Z')

Pure transport is an isometry; the learned complete update has no stability
or continuation guarantee. Existing W24 already is a learned continuous
latent. This experiment concerns width/parameterization, not discovering
latent representations or proving an intrinsic minimal dimension.

## Frozen training and evaluation

Fresh block initialization seeds 91001..91016, independent batch schedule
seeds 92001..92016. Shapes differ, so initial parameters cannot be identical
across all arms: use the same initialization seed and historical PyTorch
initialization policy, record every initial tensor hash, and describe this as
blocked training with common data schedules rather than identical weights.
Each schedule is 300x8 IID draws with replacement from the unchanged 512-map
size32 training bank seed10002. Freeze all schedules and their hashes before
any efficacy training. Arm order is cyclically rotated across blocks, decided
from block index rather than results, to reduce systematic timing/order bias.

FP32, historical local Torch2.5.1 backend settings, AdamW lr=.001, wd=.0001,
betas=(.9,.999), eps=1e-8, clip1, batch8, exactly300 updates. Each update starts
a fresh task state, executes64 numerical steps, averages task losses at
8/16/.../64, detaches all state tensors every8 steps and makes one optimizer
update. K8 is credit length;64 is supervised execution length. No Adam-state
transplants. Only the final update300 checkpoint is evaluated.

Identical fresh evaluation banks for every arm: exactly32 maps each at sizes
32/64 with seeds99332/99364. All paired source-flip correctness traces at
integer times0..256 are retained losslessly compressed. Evaluate T64/128/256
and the existing frozen `phenotype.predicate` Full gate unchanged, including
reach, preservation, sustained progress/regression and frontier support.
Reuse the original Boolean/frontier evaluator, not a new accuracy-only gate.

## Endpoints and interpretation

Independent statistical unit: one fresh initialization plus batch schedule
block, not cells, maps, rollout times or systems timings. Primary contrast is
prespecified w8 versus concurrent w24: Full success counts, paired wins/losses,
delta, Wilson95 intervals and exact two-sided McNemar/binomial p. Qualification
requires net gain>=4/16 AND p<=.05. Report all16 paired outcomes. Primary
failure is retained even if another width looks better.

W16/W4/W2 and W8-vs-capacity are secondary diagnostics. Report all unadjusted
paired p values plus Holm-adjusted p over the four width-vs-w24 contrasts
(w8,w16,w4,w2_alternating); do not substitute the selected best width into
the primary gate. Even an adjusted W2 signal is transport-confounded.
Report continuous per-seed reach, retention, regression and frontier effects
at both scales without selecting endpoints from these curves.

Systems quantities: parameters, learned persistent channels/bytes, optional
global clock bit, actual warmed median GPU step/forward64 latency on batch8
at both sizes and incremental CUDA peak allocation above the preloaded
model/input. Timings are secondary fixed-workload measurements, not training
replicates or promised speedups. W8 has half the learned full state channels
(C8+Z8 versus W24+Z8), not a threefold full-state memory reduction.

B beats A and the capacity control: evidence for this native-width recipe,
requiring independent confirmation before generalization. B and capacity both
beat A: capacity/optimization alternatives remain. Better short accuracy but
failed Full is not the intended success. No gain rejects qualification for
this width/recipe, not all latents or the NCA paradigm. A successful run does
not establish a universal short-credit solution or minimal sufficient width.

## Execution, provenance and failure

One bounded combined sanity check covers transport/clock interfaces, same-shape
W24 equivalence, exact parameter counts, K8 cadence, finite CUDA updates in all
arms, a tiny W2 source-flip trace and statistical fixtures. Freeze source hashes,
all data hashes, schedules, environment and initial/final model hashes. Retain
per-arm training curves, final checkpoints, compressed traces and frontier CSVs
in a new run. Existing evidence is read-only. Checkpoint/model-state and
machine launch records remain local pending a separately requested export.

Scientific failures still complete all96 arms. Nonfinite states/gradients,
CUDA failure or a source/data/plan binding mismatch stops execution and records
ERROR without changing the frozen recipe. No runtime cutoff or automatic
rescue sweep. Expected local work is roughly3-5 hours, refined from the small
sanity timings; this is an estimate, not a limit.

Commands from this standalone repository root:

    python -X utf8 -B new/latent_width/run.py --check --out analyses/NEW_LATENT_CHECK.json
    pwsh -File tools/launch_latent_width.ps1 -RunName NEW_LATENT_RUN -Qualification analyses/NEW_LATENT_CHECK.json

For result inspection, start with the run's RESULTS.md and summary.json;
perarm.json, curves and raw compressed traces are secondary. Launch verification
only confirms dispatch and the first actual optimizer update; it is not a
completed scientific result or an authorization for recurring monitoring.
