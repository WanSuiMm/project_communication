# Independent multi-source geodesic-distance task

## Question and claim boundary

On a task with a continuous shortest-path target, does the unchanged
StreamingCell repeatedly learn accurate distance propagation over 256 steps,
with low error at T=64 and continued refinement on cells farther than 32
geodesic steps from every source?

This is an independent, baseline-only task test. All eight arms train the same
model and recipe from fresh initialization. The runs do not transfer weights,
select a recipe from the second experiment, or change any earlier experiment.
The task gate is a proxy for recurrence behavior on a different problem; it is
not the original paired-flip Full gate, evidence of causal benefit from
StreamingCell, or a population-level claim beyond this fixed task bank and
sampling protocol.

## Task construction

Use the original region task's seeded mask sampler from
`new/nca_inertial_wind_tunnel/tasks.py`. Retain each generated traversability
mask, then discard that sampler's region labels, original single sources, and
distances. In every connected component, draw two distinct open cells from the
task RNG. The input is three channels: traversability mask, the union of all
sources, and an all-zero channel. No components, labels, distances, or source
pair identities are given to the model.

The target is the exact unweighted graph distance to the nearest source in the
same component. Walls are excluded from loss and metrics. The model predicts a
raw scalar with no sigmoid or other output activation. For training only, divide
oracle distance by the fixed constant 32; all reported errors and gates convert
the prediction back to raw cell units by multiplying by 32.

The oracle is multi-source breadth-first search. A local min-plus Bellman
identity checks each non-source open cell against its open neighbors. The CPU
check also verifies a hand-computed two-source branched-corridor fixture and a
small K=8 loss/backward/detach cadence. On the full training bank, exact BFS
distances are checked against the local Bellman equation and are required to be
at most 256. Both held-out banks additionally run the 256-step local min-plus
propagation and must equal the BFS solution at every open cell. These checks
validate the task algorithm only; they are not a learned-model baseline or
evidence of NCA efficacy.

## Frozen training

- Architecture: unchanged `StreamingCell(streaming=True)` from
  `new/streaming_carry/stream_cells.py`, W=24, Z=8, 5033 trainable parameters.
- One size-32 training bank: 512 maps, seed 81032.
- Eight fresh initialization seeds: [41001, 41002, 41003, 41004, 41005, 41006,
  41007, 41008]. Their fixed schedule seeds are [42001, 42002, 42003, 42004,
  42005, 42006, 42007, 42008]. Sample 300 batches of 8 map indices with
  replacement; each arm uses its schedule for all updates.
- For every update, run forward steps 1–64 from a fresh state. Compute masked
  per-map MSE at steps 8, 16, ..., 64 after scaling the target by 1/32. Average
  cells within each map, maps equally, and the eight checkpoint losses equally.
- Backpropagate each checkpoint's one-eighth loss at that checkpoint. Let
  gradients accumulate across checkpoints, detaching both carried state
  tensors at every K=8 boundary. Clip once to norm 1 and make one AdamW update
  after step 64.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001, gradient clip 1.
  Exactly 300 updates per arm; no efficacy-based stopping, adaptation, or
  checkpoint selection. No runtime cap is applied.
- FP32. cuDNN benchmark off, cuDNN deterministic off, cuDNN TF32 on, and CUDA
  matmul TF32 off. These settings do not claim full device determinism.

## Fixed evaluation and reporting

Evaluate the complete held-out bank for size 32 (32 maps, seed 82032) and size
64 (32 maps, seed 82064). Do not select maps or models. Record T=0, 8, 16, 32,
64, 128, and 256 metrics, and save a Boolean correctness trace at every integer
step from 0 through 256. A cell is correct within one cell when the absolute
raw prediction error is at most one.

At each reported checkpoint, retain per-map raw cell MAE and within-one-cell
coverage, their equal-map means, pooled-cell metrics, and per-map open-cell
denominators. Also report the strict d>32 band (d is exact source distance),
including per-map and pooled errors, coverage, eligible map count, and cell
counts. A d>32 map is eligible when that band contains at least one open cell.
State workspace and latent RMS values are retained at each reported
checkpoint; nonfinite values are recorded as evaluation failure.

Retention is computed from Boolean within-one-cell correctness: among all
traversable pixels correct at T=64, count the fraction that remain correct at
T=256. Report pooled pixel retention, equal-map retention, counts, and the
per-map values. An empty denominator fails.

## Predeclared task proxy gate

Each initialization seed is evaluated independently. All of these conditions
must pass for that arm:

1. Size-32 mean-map MAE at T=64 is at most 2 cells.
2. At both sizes, mean-map MAE at T=256 is at most T=64 MAE + 0.25 cells.
3. At both sizes, pooled correct64-to-correct256 retention is at least 0.95.
4. On size 64, the strict d>32 band has at least 16 eligible maps.
5. On size 64, mean eligible-map MAE improves by at least 0.5 cells from T=64
   to T=256, and mean eligible-map within-one-cell coverage rises by at least
   0.05.

Every empty or missing metric group fails its corresponding gate. Publish all
eight arm outcomes and continuous metrics; do not use the gate to select an
arm. The gate is only a per-initialization proxy on this alternate task. It
does not establish that the original paired-flip task's Full qualification
passes. If no arm meets size-32 mean-map MAE ≤2 cells at T=64, classify the
result as `SECOND_TASK_TRAINING_UNQUALIFIED`. If at least one arm meets that
checkpoint criterion but none passes the full task gate, classify it as
`NO_SECOND_TASK_LONG_ROLLOUT_QUALIFICATION`. If one or more arms pass the full
gate, classify it as `SECOND_TASK_POSITIVE_SCREEN`, which is not an architecture
GO. A numerical state, gradient, parameter, or evaluation failure stops later
arms and leaves a durable `NUMERICAL_FAILURE` record.

## Artifacts

`distance_task.py` provides `bank`, `backward_distance`, `evaluate_distance`,
`cpu_check`, and `run_stage(out, progress_callback)`. The stage writes its
configuration and task boundaries to `manifest.json`, lifecycle to
`status.json`, exact oracle checks to `oracle_checks.json`, fixed schedules to
`schedule_seed*.npy`, source coordinates for held-out maps to
`source_coordinates.json`, per-arm training curves to `curves/`, final
checkpoints to `checkpoints/`, full per-size Boolean traces with distances,
masks, and source sets to compressed NPZ files, and aggregate results to
`summary.json` and `RESULTS.md`. The manifest binds the exact train and held-out
banks. Each arm records its initialization parameter hash, schedule hash, final
parameter hash, and checkpoint-file hash. The output directory must be new.
Local output paths, host identifiers, and launch receipts are not part of the
task result.
