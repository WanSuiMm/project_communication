# Incremental review: Hybrid Writer v0 code and prelaunch checks

- Review base: `90888de0ca869392686cda40aee6fbe36f43caa3`.
- Code/evidence head: `7373aa4d1f3c64bd1e84af108b9cc2ac617eab46`.
- This later commit changes only the handoff; the code/evidence head stays stable.
- Formal efficacy: `PENDING_NOT_PUBLISHED`. This is a code release, not a result.

## Read first

1. [Frozen protocol](new/hybrid_writer/PROTOCOL.md).
2. [Writer implementation](new/hybrid_writer/cells.py) and
   [CPU checks](new/hybrid_writer/check_cells.py).
3. [Runner](new/hybrid_writer/run.py), [paired reporting](new/hybrid_writer/reporting.py),
   and [configuration](evidence/hybrid_writer_prelaunch_20261006/config.json).
4. [Prelaunch validation](evidence/hybrid_writer_prelaunch_20261006/validation.json),
   [historical dispatch verification](evidence/hybrid_writer_prelaunch_20261006/dispatch.json),
   and [source bindings](evidence/hybrid_writer_prelaunch_20261006/manifest.json).

No need to reread earlier large state packages for this implementation review.
The previous causal motivation is the [block7 audit](evidence/block7_collapse_20261006/RESULTS.md).

## New implementation

C24 remains four directional lanes of six payload coordinates; Z8, encoder,
masked permutation, Q and readout keep their original architecture. Common
initial tensors are copied explicitly from a canonical seeded StreamingCell.

| Arm | Proposal | Writer | Parameters |
|---|---|---|---:|
| neural | original two-layer MLP | original free residual | 5033 |
| budget | same MLP | radial bound with four constant learned gates | 5037 |
| hybrid | same MLP | radial bound with36 state-conditioned coefficients | 5069 |
| affine_hybrid | direct affine67->24 | same Hybrid writer | 2997 |

Bounded arms use delta=.1*a*m/sqrt(.5^2+mean_lane(m^2)).
The uniform per-lane RMS write budget is .1; zero proposals preserve C transport.
The Hybrid gate is a convex hat interpolation at fixed knots0,.5,1 over lane
RMS of incoming T(C) and pre-stream L(C). It has no direct target/readout/time
input, but hidden magnitudes can carry semantics indirectly. All gates start
at.5 and proposal outputs at zero; initial numerical functions and local
proposal Jacobian match. No claim of semantic closure or full stability.

Eight fresh paired initialization/schedule blocks,300 reset64x4 super-updates
each, same batch8 and32 K8 losses/backwards per optimizer step. Official J
endpoint remains u300, with three primary paired contrasts and Holm correction.
Checkpoint-grid ever-ready and gate/write telemetry are secondary only.

## Verification and status

CPU checks passed hats, copied initialization, affine inactive-module removal,
bitexact randomized neural parity, zero-proposal carry, proposal Jacobian,
large-message write bounds, finite gradients and observer behavior.
All four actual training-shape GPU comparisons passed three AdamW updates;
gradient maximum absolute difference is zero in every arm. Loss, states,
parameters and Adam moments were also checked against eager arithmetic.

A later reporting-only source rebinding added Wilson intervals and error labels;
training cells, runtime, evaluator and launcher remained unchanged. This is
documented in validation.json. Launch verification checked the saved u25 model
and optimizer steps after tool yield. Published dispatch status is historical,
not live progress or a final scientific verdict.

Original checkpoints, live training/evaluation outputs, machine manifests,
launch receipts and logs are excluded. No additional model training or inference
was performed for this publication.

## Unchanged conclusions

The selected block7 audit supports successful continuation of earlier states
by G300; it does not identify which cold-prefix component caused failure.
The prior fixed-u300 continuous-coverage reliability qualification remains
negative. This code release supplies no new efficacy, general BPTT solution,
closure theorem, population reliability result or 3D result.

## Reviewer questions

1. Do the lane RMS axes, pre-stream feature clock and bounded write implement
   the declared intervention without a free residual bypass?
2. Do the Budget and Hybrid controls distinguish constant from state-conditioned
   regulation, while treating affine_hybrid as a capacity ablation?
3. Do formal u300 and observed ever-ready remain distinct throughout aggregation?

CPU reproduction from repository root:

    python -X utf8 -B new/hybrid_writer/check_cells.py

GPU qualification and fresh-run commands are in the protocol; they require
the historical Torch2.5.1 CUDA environment. Existing output directories are
never overwritten.
