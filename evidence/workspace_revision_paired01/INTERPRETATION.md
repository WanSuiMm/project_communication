# Workspace + Revision: completed paired screen

**Decision: `NO_JOINT_SCREEN_PASS`.** All four arms completed 600 updates in
1112.28 seconds. The mean paired primary hold effect (revision minus additive)
was **-9.02 percentage points**, against a frozen requirement of at least +5 pp
and a positive effect in both seeds. No rescue training or hyperparameter sweep
was performed for this publication.

## Read the joint failure before the single-seed signal

All accuracy entries below are percentages. Hold is the minimum aggregate BA
at T128, T192 and T256; it is not a minimum over individual maps. The primary
evaluation is size 32, with 16 shared maps and two independent model seeds.

| Model | Seed | BA at T64 | Hold minimum | Paired accuracy at T64 | Warm source-change accuracy at K64 | Z-only repair BA at K64 |
|---|---:|---:|---:|---:|---:|---:|
| Additive | 0 | 94.29 | 94.18 | 65.02 | 13.85 | 94.07 |
| Revision | 0 | 99.19 | 100.00 | 98.00 | 0.00 | 100.00 |
| Additive | 1 | 78.54 | 73.86 | 26.13 | 41.59 | 78.55 |
| Revision | 1 | 50.00 | 50.00 | 0.00 | 37.50 | 50.00 |

The paired hold effects are **+5.82 pp** and **-23.86 pp**. Revision seed 0
fails the required source-change accuracy and its noninferiority check.
Revision seed 1 also fails reach, paired accuracy, hold improvement and repair
requirements. Its flat 50% BA satisfies the relative hold predicate trivially;
it is not successful memory. The exact failed predicates are in
[analysis.json](analysis.json); the unchanged runner aggregate is
[aggregate.json](aggregate.json).

## Seed 0 reaches and holds, but does not reopen

Revision seed 0 reaches 100% BA at T96 and retains it at the measured points
through T256 on size 32. On the secondary size-64 evaluation, BA rises from
95.99% at T64 to 99.44% at T256, with a 98.93% hold minimum. These are useful
single-seed observations; size 64 cannot rescue the failed primary gate.

After changing the source at T64 while retaining the existing state, the same
model scores **0% on the changed component at K64 and K128**, while the
unchanged component remains 100% correct at K64. Cold initialization for the
new source achieves **100% changed-component accuracy at K64 and K128**.
On size 64, warm accuracy is 2.13% at K64 and 1.64% at K128, versus 87.39% and
99.49% for cold initialization. Thus learning the new-source task from a fresh
state is possible, while revising an established state fails.

This is behavioral evidence of history dependence. It does not establish a
particular attractor, hysteresis mechanism or Jacobian explanation. Fresh
original/flip paired accuracy is a different test from warm source revision.

## Repair is conditional on what survives

For revision seed 0 on size 32, zeroing a Z patch changes BA from 99.19% to
97.62% immediately; after 64 steps it reaches 100%, matching the clean
continuation. Zeroing both W and Z instead yields only **90.51%** after 64
steps. On size 64, Z-only repair is **96.91%**, below clean continuation
98.93%; joint W/Z repair is **80.42%**.

The patch occupies only one quarter of the side length, so whole-map BA can
understate local damage. Z-only repair retains W and should not be presented
as general state regeneration. Conditional repair metrics include only maps
whose clean T64 BA was at least 95%: revision seed 0 has 15 eligible maps at
size 32 and 12 at size 64; revision seed 1 has none, and its conditional
metrics are null, not a recovery success.

## Stable output does not imply stable full state

Revision seed 0's size-32 W RMS grows from **2.942 to 9.718** between T64 and
T256, while Z RMS changes from **1.103 to 1.120**. W continues to update at
T256. All recorded values are finite, but neither a full-state fixed point nor
asymptotic stability was demonstrated. The revision update has a conditional
bounded-input guarantee for Z, not a guarantee for W or useful task dynamics.

## What this experiment actually isolates

Both arms use the same W24/Z8 split, encoder, readout, F/Q networks, 5033
parameters, alpha=0.5, two masked communication phases per macro-step and
paired initial parameters/data/schedules. The single intervention is
`Z' = Z + 0.5 Q` versus `Z' = Z + 0.5 (Q - Z)`. This tests the revision rule
**within** a shared workspace architecture; it does not isolate the benefit
of introducing W/Z or prove that a relation bottleneck is necessary.

Training supervises reach plus alternating hold, source-switch and repair
branches. The 800-to-600 update amendment was frozen before efficacy training
using preflight runtime alone. There is no learned gate, output clamp or extra
output tanh. Q nevertheless has bounded hidden tanh units, so fixed finite
weights imply a finite candidate bound; this is not full recurrent stability.

Earlier Masked State/Momentum runs used different training and communication
schedules. They are historical context, not matched controls for this screen.
Even the current additive seed 0 retains high BA at long rollout, so the new
results do not establish that revision alone fixes the earlier collapse.

Two model seeds are insufficient for a reliable population effect or a
significance claim. Across seeds, initialization, training bank and schedule
all change; the failed seed cannot be attributed to initialization alone.
Per-map scores are repeated observations within each model, not additional
independent training replicates. Timing is descriptive, not a speedup result.
No 3D, very-long-rollout, idempotence or new Jacobian audit is included here.

## Evidence and verification

1. [Overview figure](overview.png): both seeds, both sizes, warm/cold switch.
2. [Compact analysis](analysis.json) and [runner results](RESULTS.md).
3. [Frozen protocol](../../new/workspace_revision/PROTOCOL.md) and
   [cell code](../../new/workspace_revision/revision_cells.py).
4. [Validation](validation.json) and
   [publication manifest](../../REVISION_PUBLICATION_MANIFEST.json).

Raw arm JSON and schedules are byte-identical copies of the completed run.
The publication checks recreate data banks and initial parameter hashes on
CPU, verify saved checkpoint hashes, recompute per-map aggregates and all
primary predicates, and compare executed source snapshots. They do not rerun
training or model evaluation. Public manifest/completion copies omit local
process metadata. Checkpoints and launch receipts remain local.
