# Incremental review: initial geometry and continuation formation

- Review base: `483e86773c890aecca2a8109f80fe24c683b02a6`.
- Evidence head: `5db6d8184777cf617ea469fcbce9d056b3381259`.
- The later handoff/entry-clarification commit changes reader metadata only.

Two completed2D CPU diagnostics, zero new training. Initialization uses four
exact historical update0 checkpoints and previously inspected maps. Formation
reads44 saved trajectories from one selected historical seed4 run, with no
new inference. Existing architecture and warm-start negative results stand.

## Minimal reading order

1. [Combined results](evidence/seed4_learning_geometry_20261004/RESULTS.md).
2. [Initialization report](evidence/initial_geometry_20261004/RESULTS.md) and
   [formation report](evidence/formation_gate_20261004/RESULTS.md), then
   [interpretation](evidence/formation_gate_20261004/INTERPRETATION.md).
3. [Initial profiles](evidence/initial_geometry_20261004/profiles.csv),
   [formation profiles](evidence/formation_gate_20261004/profiles.csv), and
   [source/metric map](GPT_CONTEXT.md). Large per-map tables and the complete
   scalar/spectral summaries are secondary; do not open them first.
4. [Initialization protocol](new/initial_geometry/PROTOCOL.md),
   [formation protocol](new/formation_gate/PROTOCOL.md),
   [publication bindings](LEARNING_GEOMETRY_PUBLICATION_MANIFEST.json), and
   the two [initialization](evidence/initial_geometry_20261004/REPRODUCTION.md)/
   [formation](evidence/formation_gate_20261004/REPRODUCTION.md) reproduction notes.

## New decision-relevant evidence

- Seed4 has more balanced RAW lane RMS and greater measured available relative
  source-flip feature separation at K8 endpoint64. Centered balance and spectral
  rankings vary. Seed3 has better paired kernel-label alignment on all four
  inspected banks; initial Q-feature effective rank is highest for seed5.
  There is no consistent initialization advantage or causal success criterion.
- Initial state-Jacobian isometry is shared by all seeds. The first historical
  batch gives zero encoder/F/Q-in/readout-weight task gradients; Q-out weight
  has the appreciable gradient, rank one up to FP32 roundoff. Balanced-loss bias
  gradients are near-zero cancellation. Full-history and K8 tangents differ.
- Early retention/progress, measured with rollout t<=64, passes at TRAINING
  updates140/145/190/200. The long screen passes only200 in the dense window.
  Same-checkpoint classification: TP1/FP3/TN17/FN0. Forecasting the next saved
  training update (+5): TP0/FP3/TN16/FN1. Checkpoint300 is a nonadjacent control.
- At175 the early survival/G/new-state survival are .9291/.2751/.9204;
  at180 .4907/.0617/.4071; at195 .9377/.3324/.9520;
  at200 .9996/.3762/.9970. They describe changing short-rollout behavior.
  The margin-reserve heuristic does not forecast the175->180 optimizer change.
- All four early continuous quantities rank200 highest in this selected dense
  scan, but ordinary T64 output coverage also does. No incremental prediction
  beyond an output-only baseline, prospective training forecast or latent
  continuation representation has been demonstrated.

## Validation and reproduction boundary

Original initialization measurement took30.31 CPU seconds. Historical parameter,
training-bank, schedule and source hashes match; analytic tangent checks against
autograd have maximum error4.77e-7. Publication independently recomputes304
paired feature Gram spectra/alignments from saved arrays, max spectrum error
4.44e-16, without repeating inference. Public per-map grouping checks pass.

Formation analysis took8.86 CPU seconds, checking all44 raw archive bindings and
matching prior long rates. Publication verifies880 per-map rate/count records,
both confusion tables and all published hashes. It launches no model execution.

From a fresh clone, standard-library public verification is:

    python -X utf8 -B tools/export_learning_geometry.py --verify-only

Full remeasurement needs excluded original feature/input/checkpoint/trajectory
archives. Public verification is not exact model reproduction. Raw arrays and
machine receipts remain local. No new reliability,3D or architecture gate claim.

## What remains unestablished

The particular seed4 quotient/closure mechanism remains a low-confidence
candidate. These diagnostics did not construct pi, an abstract transition F,
a raw-domain certificate, or an independent consumer contract. Short behavior
and signed margin are not independent hidden-mechanism evidence. The conditional
execution implication remains intact; no training-discovery theorem is supplied.

## Reviewer questions

1. Which initialization findings survive centering, scale and payload-coordinate
   dependence, without implying a success probability from four reused seeds?
2. Do any reported formation quantities add information beyond current output
   behavior? What could be identified from these saved arrays alone?
3. Is downgrading the specific closure mechanism interpretation supported,
   while preserving the conditional execution argument and earlier negatives?
