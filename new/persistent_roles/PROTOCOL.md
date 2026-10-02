# Persistent local, carrier and task roles: development screen

Frozen before efficacy training, 2026-10-02. This authorized experiment tests
the third user proposal: a stationary computation workspace, a persistent
moving carrier, and stationary task state, updated by one shared local rule.
The question is whether this complete parameterization makes sustained K8
computation easier to learn across initialization seeds. Previously inspected
seeds and maps are reused; this is development, not independent confirmation.
Earlier negative decisions and sources remain unchanged.

## Exact candidate and comparison boundary

State per cell is H12+C12+Z8=32. C has four N/E/S/W lanes with three payload
channels each. X has three channels: traversable mask, positive source and
negative source. The task is seeded connected-region identity; labels are
computed from graph connectivity, not from a PDE. No component IDs, distances
or targets are exposed to the model.

Initialize the original RevisionCell('ws_additive') first, retain its
encoder3->24 and readout8->1 (including the original RNG advancement), and
replace the old F/Q modules by one pointwise Tanh MLP R35->72->32. Its last
layer is zero initialized. Split encoder(X) into H12 followed by C12 once;
Z0=0. Keep common encoder/readout draws exactly equal to both controls.
Complete candidate initialization is different because its residual module
has different shapes. Record full and common hashes separately.

Each phase, using the incoming C at the same cell:

```math
(dH,dC,dZ)=R_\theta(H,C,Z,X),
\qquad
(H,C,Z)\leftarrow(H+0.1dH,\;T_M(C+0.1dC),\;Z+0.5dZ).
```

The SAME phase and parameters are applied twice per macro-step. Z alone
feeds the readout. R has4928 parameters, encoder96 and readout9: exactly5033,
matching both old controls. There are no dummy matching parameters.

T_M is the existing validated masked port permutation: open ports move to
the adjacent open cell in the same direction; blocked/exterior links bounce
into the opposite lane; wall ports stay fixed. No averaging, dropping or
periodic wrapping. Only C crosses cells. R has no Laplacian, raw-neighbor,
ephemeral emitter, normalization, learned gate, auxiliary loss or momentum.

With R=0, H/Z stay local and C after p phases is T_M^p(C0), or T_M^(2t)(C0)
after t macro-steps. For fixed mask the base map I_H\oplus T_C\oplus I_Z is
isometric. The full trained recurrence is not guaranteed lossless/stable.
Tensor/interface closure does not guarantee semantic preservation. Detach
preserves numerical state, never restores cross-window credit assignment.
Zero residual-head initialization gives an isometric state Jacobian but can
initially block task gradients to the encoder/hidden residual layer; this is
a recorded initialization choice, not evidence of easy optimization.

The candidate's causal clocks must remain explicit: after p pointwise phases,
an initial carrier can reach at most p graph edges; H/Z can consume it at
most p-1 edges away because reading precedes streaming. A K8 window has16
phases, carrier radius<=16 and task-readout radius<=15 from its initial state.
At T64, carrier radius<=128 and task-readout radius<=127. These are upper
bounds, not useful-information guarantees. Losses/evaluation T still count
macro-steps. Old normalized evaluator bands based on2K are retained solely
for exact historical comparison; they are not exact candidate readout radii.

Controls are unchanged RevisionCell('ws_additive') and StreamingCell(). Old
Streaming already has T_W+I_Z, a moving W24 and stationary Z8, with Laplacian
perception. The candidate adds a dedicated stationary H workspace while
reducing persistent transport width24->12, removing Laplacian perception,
sharing the local rule and changing readout timing. Matching5033 parameters,
32 persistent scalars and two phases does not isolate H causality, prove a
RelationFirst mechanism, or match FLOPs, activation memory or latency.
No arbitrary addressing, editing, long-horizon stability, 3D or natural-data
claim is tested. No mechanism of old Streaming seed4 is assumed.

## Fixed recipe, endpoints and decisions

- Variants baseline/stream/roles; initialization seeds2,3,4,5;12 arms.
  Orders: seed2 baseline/stream/roles; seed3 stream/roles/baseline;
  seed4 roles/baseline/stream; seed5 roles/stream/baseline.
- Exact historical training bank10002 (512 size32 maps), batch schedule20002
  (300x8), evaluation banks40032/40064 (32 maps each). Bind data, schedule,
  full control initial parameters and common encoder/readout hashes to
  published historical evidence.
- Fixed300 updates for every arm. Fresh64-macro-step trajectories; eight
  equally weighted task losses at8,16,...64. Existing K8 trainer detaches
  every state member (H/C/Z or W/Z), preserves its values, accumulates
  gradients across windows, clips once at norm1 and updates AdamW once after
  the complete trajectory. lr0.001, weight_decay0.0001, batch8, FP32;
  backend flags unchanged. No pool, optimizer change or efficacy-based stop.
- Fresh original/source-flipped evaluations at sizes32/64, T64/128/256.
  Primary: size32/T64 STRICT16<d<32 paired per-map mean AND pooled>=80%,
  original AND flipped balanced accuracy>=85%. This primary is beyond the
  single K8 carrier/readout bounds; it does not count exact temporal handoffs.
- Hold at BOTH T128 and T256: each BA falls<=3pp and each primary paired
  statistic falls<=5pp from its own T64 value. Hold alone can preserve an
  incorrect endpoint and never counts as success without reach.
- All eight control final parameter hashes AND complete evaluation payloads
  must exactly reproduce their historical records. Baseline2/5 and stream4
  must retain reach+hold. Any completed mismatch is CONTROL_REPRODUCTION_DRIFT,
  not an interpretable qualified architecture comparison.
- DEVELOPMENT_GO requires all12 arms complete, qualified controls,
  roles reach+hold>=3/4 including2/5, and its count exceeds BOTH concurrent
  controls. Otherwise a complete matched run is DEVELOPMENT_NO_GO.
  Any error, time cap or incompletion is INCOMPLETE, never scientific success.
- Keep every seed, size, horizon and distance band, integer correct counts,
  denominators, per-map mean and pooled accuracy. Report roles-minus-baseline
  and roles-minus-stream separately. Empty bands are null; farther gains or
  an isolated positive seed cannot replace the frozen primary/hold gate.
- The independent unit is initialization seed n=4, conditional on one fixed
  training bank/schedule. No p-value or population reliability claim. The
  already inspected seeds/maps make any positive result developmental.

## Minimal engineering gate, budget and launch boundary

One focused CPU suite checks shapes/parameter/common initialization, an
independent port-scatter and nonzero residual forward/backward reference,
zero-residual local identities and transport/Jacobian isometry, phase readout
lag and component isolation, and three-state K8 detach/gradient/optimizer
clock compatibility. Decision truth cases and strict hold failure cases are
checked without GPU efficacy. Do not repeat an exhaustive transport audit or
select initialization/capacity by task performance.

One CUDA preflight:3 updates of all3 variants at seed2, abbreviated evaluation.
Let c be the largest median of the last2 synchronized update times. Require
12*300*c*1.20+180<=2400 seconds, peak allocated<3GiB, finite quantities,
identities matching and primary coverage>=16 maps/500 pixels. Otherwise stop
before formal training; do not reduce300 updates or inspect accuracy to tune.
Preflight is engineering only. Fixed hard cap40 minutes, watchdog41 minutes.

Source bindings validate all45 paths from LOCAL_INTERFACE_PUBLICATION_MANIFEST.json
and add that manifest, this cell/runner/check/protocol and the launcher (51
paths total). Sources must match the passed preflight exactly. Preserve all
executed sources/hashes, data/schedule/initialization identities, checkpoints,
integer raw metrics, clipping/timing/memory and aggregate/curves/effects.
New output directories only. Private host/PID/GPU/command/time receipt stays
local. Historical generated artifacts remain read-only.

Authorized launch completion means verified dispatch plus a durable receipt.
No automatic retry, architecture rescue, sweep, confirmation or monitor.
Inspecting the final results is a separate request.

Commands from repository root:

    python new/persistent_roles/check.py
    python new/persistent_roles/run.py --preflight --out runs/NEW_ROLES_PREFLIGHT
    pwsh -File tools/launch_persistent_roles.ps1 -RunName NEW_ROLES_RUN -Preflight runs/NEW_ROLES_PREFLIGHT
