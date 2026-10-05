# Execution amendment: local CUDA Graph continuation

Scientific protocol remains `native_latent_width_v1`; execution amendment is
`local_cuda_graph_k8_v1`. The user requested local acceleration on 2026-10-05.
This changes runtime implementation only: all six models, 16 blocks, original
plans/data, FP32/backend, K8, losses at8..64,300 AdamW updates, clip1, final
checkpoint selection, evaluation and Full/primary/statistical rules remain.
No efficacy-based selection, additional seed or optimizer change.

## Qualification before replacement

The isolated pilot showed exact six-update replay for W24 and W2. Before
switching, replay all300 updates of every arm in the already completed first
block from its saved initial checkpoint and original schedule. Require bitwise
equality of final parameters and every Adam tensor/group with the saved eager
reference. All six must pass; a close result never substitutes for exact replay.
These are runtime checks, not new efficacy trials. Existing reference outcomes
are not a selector. The finite first-block check is not a universal guarantee
for all possible initializations or hardware environments.

Capture numerical forward64 and eight backward windows once per model with
static input/parameter/gradient addresses. Keep original eager norm clipping and
AdamW outside the capture. Every new minibatch copies x/y/mask into static GPU
buffers. The graph's first backward overwrites static gradients; later windows
accumulate. No set-to-None after capture. W2's phase schedule starts0 and ends0
each64-step training trajectory, so its CPU control is statically unrolled.
Loss/final-state finite checks are combined before clipping, norm checked before
AdamW, and parameter finiteness checked after it. This moves nonfinite detection
to the end of a trajectory rather than each loss; no nonfinite update is allowed.

## Handover and evidence preservation

After qualification, briefly wait for the next durable completed-arm row,
verify the original Python process identity/start time, then stop that exact
worker. This is a bounded migration action, not recurring monitoring. Keep the
original run unchanged; write a handover receipt outside it. A stop can race
with the next arm's startup. An arm absent from durable perarm.json is incomplete
even if it has some checkpoint files; preserve that attempt locally and rerun
its same seed and full schedule from update0, with no seed replacement.

New output is a separate run directory. Copy every completed arm's entire
artifact directory only after hashes, checkpoint clock300, full curve and
evaluation bindings pass. Copy original plans/config/source byte-for-byte and
record imported eager rows as such. The aggregate has exactly one final result
per planned `(block,arm)`, with provenance distinguishing imported and accelerated
execution. Runtime/qualification sources have their own hashes/snapshot. Partial
original attempts cannot enter the aggregate. Old status may remain RUNNING at
its last write after external stop: the handover receipt records actual stop.

Graph compilation/capture cost is reported separately from update timing.
Inference/evaluation and systems-forward timings still use the original eager
implementation. End-to-end time includes capture, evaluation, saving and import;
the pilot's4.8x/7.4x core speedups are not promised wall-time improvements.
No maximum runtime, watchdog, recurring monitor or architectural rescue.

Commands from repository root with new output names:

    python -X utf8 -B new/latent_runtime/qualify.py --original runs/latent_width_20261005_01 --out analyses/NEW_GRAPH_QUALIFICATION.json
    pwsh -File tools/launch_latent_accelerated.ps1 -OriginalRun latent_width_20261005_01 -RunName NEW_ACCELERATED_RUN -Qualification analyses/NEW_GRAPH_QUALIFICATION.json

Read the new run's RESULTS.md/summary.json first; perarm.json contains execution
provenance. launch/handover receipts and model checkpoints remain local.
