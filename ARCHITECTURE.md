# Tensor and operator map

## Reparam-GCR: raw-state lift with late learned projection

For x[B,N,8], keep count[B,N,1], U[B,N,8], V[B,N,8,8] and endpoint z[B,1].
With the same predecessor gather and active-cell mask as old GCR:

```math
n'=n_{pred}+1,\qquad U'=U_{pred}+x,\qquad
V'=V_{pred}+U_{pred}x^\top.
```

At the endpoint, the bias-free W[4,8] produces S1=WU and S2=WVW^T.
The unchanged21->256->1 interpreter consumes normalized count/S1/S2;
z'=.5z+.5*interpreter and y=tanh(z). These projected states and outputs
equal old GCR in exact arithmetic, including full64 parameter gradients.
Only the location of W and fixed-state dimension change:5921 parameters
remain, per-cell state rises21->73 scalars.

`ReparamGCR.initialize/step/project_workspace`:
[cells.py](new/reparam_gcr/cells.py). `forward` and `check`:
[run.py](new/reparam_gcr/run.py). K8 detaches raw count/U/V/z after step56;
live W is applied throughout the final window. Since the first three variables
are independent of learned parameters, their cut loses no parameter path.
The residual endpoint EMA cut contributes the known255/256 gradient scale in
the bounded check; early input credit remains absent.

[Completed evidence](evidence/reparam_gcr_20261009_01/RESULTS.md) recovers
the old one-block GCR learning gap. The construction depends on linear
projection and degree-two sufficient statistics, not arbitrary neural dynamics.

## Learnable GCR: a separate directed-path task

This task's input is x[B,N,8] and a static predecessor index pred[B,N] on a
simple directed Manhattan path. A trainable linear phi produces p[B,N,4].
State consists of count[B,N,1], workspace[B,N,20] and endpoint z[B,1]. Starting
from zeros, both variants read the preceding cell's previous state; missing
predecessors supply zeros and inactive grid cells remain zero.

GCR uses n'=n_pred+1, S1'=S1_pred+p, S2'=S2_pred+S1_pred outer p. Workspace
contains S1[4] and row-major S2[16]; evidence is not decayed. Full Writer uses
h'=h_pred+.1*MLP(h_pred,p,log-count), with a25->16->20 ReLU MLP. Both endpoint
interpreters consume(log1p(n)/log97,workspace/n), using21->256->1 ReLU;
z'=.5z+.5*interpreter, y=tanh(z). Endpoint-only decoding is exact pruning
because z does not influence any workspace update. GCR's S2/n matches the
teacher scale. Local inputs remain available each step.

`OrderedCell.initialize/step/feature/predict`: [cells.py](new/learnable_gcr/cells.py).
`forward_credit`: [run.py](new/learnable_gcr/run.py), with56 no-grad steps plus
8 differentiable steps, versus64 differentiable steps. Exact phi caching
inside a gradient window retains the sum of parameter gradients from every
use. Shared phi/interpreter initialization matches across arms; model counts
are5921/GCR and6677/Full Writer. [Completed evidence](evidence/learnable_gcr_20261009_01/RESULTS.md)
is control-unqualified; this structure does not establish a generic NCA or
short-credit learning guarantee.

## Addressed delta writer: replace the carrier head

C[B,24,H,W] is reshaped into four6D directional ports; stationary Z has8 channels.
The masked permutation gives U=T_M(C). Writer features are
`[U,Z,L_pre(C),L(Z),X]` (67 channels), preserving the original feature clock.
A shared67-to-40 tanh perception feeds a key head40-to-16 (4 reads x4 ports)
and a value head40-to-24 (4 reads x6 payload). Each cell normalizes
`K=A/sqrt(1+||A||_F^2)` independently. Candidate writes are:

```math
C'_{add}=U+0.1K^\top V,\qquad
C'_{delta}=U+0.1K^\top(V-KU).
```

Both retain `Z'=Z+0.5Q(C',Z,L(C'),L(Z),X)` and readout(Z'). The original
f_in/f_out modules are removed and their perception weights copied into the
new writer. Common encoder/Q/readout match Current; Additive/Delta match each
other in all initial tensors. K starts zero; V is ordinarily initialized.
Counts are Current5033 and candidates5689 each. Initial carry preservation
does not imply full learned-recurrence stability or semantic preservation.

`AddressedDeltaCell.step` and `make_model`: [cells.py](new/addressed_delta/cells.py).
`AddressedDeltaWriter`: [writer.py](new/addressed_delta/writer.py).
`run`/`save_stage`/`committed`: [runner](new/addressed_delta/run.py).
`aggregate`: [reporter](new/addressed_delta/reporting.py).
Evaluation reuses frozen CUDA rollout/metrics helpers and persists compact
metrics only; the [protocol](new/addressed_delta/PROTOCOL.md) defines the
four-block developmental gate. [Completed results](evidence/addressed_delta_20261008_02/RESULTS.md)
are negative for both candidates. The compatibility recovery helper changes
JSON config representation only, with its source bound in child provenance.

## Current195/200 audit: unchanged cell, separate intervention axes

E=encoder, F=f_in/f_out, Q=q_in/q_out, R=readout. The frozen audit replaces
these blocks independently between two checkpoints of one training trajectory;
it does not add an architecture. At64, a paired state `(W,Z)` is produced by
one checkpoint, continued by another, and scored under either R. R never enters
the recurrent rule. Spatial swaps use the same reciprocal plan in both cue
worlds, keeping W's four directional lanes separate when testing a lane.
Readout-null Z deltas obey `R_weight * deltaZ approximately0` within1e-6;
that preserves the immediate score numerically, not future semantic content.

`instrument.parts` follows the original two-hop clock exactly: pre-stream
L(W)/L(Z) enter F; post-F W and its Laplacian enter Q. The fixed-order
Q(W,Z)->Q(TW,Z)->Q(TW+etaF,Z) telescope is exact linear-readout bookkeeping.
It is an order-specific mediation split; nonlinear Q-input knockouts do not
form additive semantic attributions. `probes.pulse` modifies selected cells
for64->65 then restores the original rule; F/transport pulses blend the
intermediate W before Q reads its neighbors.

[Frozen protocol](new/audit_195_200/PROTOCOL.md),
[source map](GPT_CONTEXT.md), and
[current evidence](evidence/joint195_200_20261003/RESULTS.md).

## Current follow-up: original StreamingCell unchanged

The A/B/C follow-ups keep W[B,24,H,W], four six-channel carrier lanes, and
stationary Z[B,8,H,W],5033 parameters. One macro-step is

```math
I=T_M(W),\qquad W'=I+0.1F_\theta(I,Z,L_M(W),L_M(Z),X),
\qquad Z'=Z+0.5Q_\theta(W',Z,L_M(W'),L_M(Z),X).
```

Encoder3->24, F67->40->24, Q67->16->8 and pointwise readout8->1 are unchanged.
The macro-step has a two-hop upper radius. Only fixed transport T_M is a
masked port permutation; learned recurrence has no losslessness/stability
guarantee. A changes only batch schedules. B changes starting parameters
along paired global Gaussian rays. Forward64/loss8/detach8 remains fixed.

C restores both W/Z at q to their own preceding macro-step, separately for
original/flipped source worlds. A different wrong neighbor r receives its
own rollback direction scaled per block/world to q's norm. Each event and
condition uses an independent full-map clone at the same snapshot time.
Only that site may differ initially; p/input/parameters stay fixed. Off-cone
s receives q's delta at Manhattan distance>2 and must leave next-step p logits
unchanged. This is a whole-cell state intervention, not an edge-message mask.

Source: StreamingCell/stream in new/streaming_carry/stream_cells.py;
initial_model, training, phenotype and causal helpers under new/seed4_followup/.
[Frozen protocol](new/seed4_followup/PROTOCOL.md) and
[current evidence](evidence/seed4_followup_20261003/RESULTS.md).

## Preceding architecture experiment: exactly nested stationary sidecar

Keep original W[B,24,height,width], Z[B,8,height,width] and add H[B,12,height,width].
W has four six-channel carrier lanes, Z/H remain stationary. G67->32->12
reads only original pre-stream [W,Z,L_M(W),L_M(Z),X]; pointwise feedback
P_F12->24 and P_Q12->8 have no bias. For one macro step:

```math
U=0.1G_\theta(W,Z,L_M(W),L_M(Z),X),
H^*=H+U\ \text{(memory)},\qquad H^*=U\ \text{(stateless)},
I=T_M(W),
W'=I+0.1\{F_\theta(I,Z,L_M(W),L_M(Z),X)+P_FH^*\},
Z'=Z+0.5\{Q_\theta(W',Z,L_M(W'),L_M(Z),X)+P_QH^*\}.
```

Store H'=H* for memory, H'=0 for stateless; both consume H* immediately.
Readout uses Z only. The G input does not contain H/LH, so the stateless
arm has no dormant H feature columns. Only original T_M/L_M cross cells;
the original upper bound of two hops per macro and16 per K8 remains.

Construct original modules first, preserving all5033 core parameter draws.
Side arms share identical initialization: H0=0, G_in default, G_out weight
N(0,.02)/bias0, P_F/P_Q weight0. G adds2572 parameters and feedback384,
total7989. The extra persistent H carry is the single structural difference.
Old stationary Z remains; names do not establish enforced semantic roles.

P_F=P_Q=0 nests projected W/Z/logits and core derivatives for arbitrary finite
H, including nonzero original F/Q tails. Full states have different dimensions.
Raw gradients do not imply identical joint-clipped AdamW updates. Original
Q_out=0 blocks initial P_F task gradients; P_Q opens first, P_F/G by update3.
H values survive K8 cuts but autograd history does not. Extra accumulation
can change scale; no full recurrence stability or losslessness guarantee.

Source map: SidecarCell.initial/step/logits/metadata and core_state_dict in
new/stationary_sidecar/sidecar_cells.py; neutral_check, reference_check,
clock_and_gradient_entry, geometry_check and decision_check in check.py.
Training/gates are in run.py; saved arithmetic and CPU checkpoint verification
in analyze.py. [Frozen protocol](new/stationary_sidecar/PROTOCOL.md) and
[completed negative result](evidence/stationary_sidecar_init2345/RESULTS.md).

## Current diagnostic: switches on the original Streaming cell

This audit changes no learned parameters. State W[B,24,height,width] has
four six-channel lanes; Z[B,8,height,width] remains stationary. Keep original
encoder3->24, F67->40->24, Q67->16->8 and readout8->1. Define K as T_M or
identity and P as masked Laplacian or the zero map. One macro-step is:

$$
\begin{aligned}
W^+ &= K(W)+0.1F_\theta(K(W),Z,P(W),P(Z),X),\\
Z^+ &= Z+0.5Q_\theta(W^+,Z,P(W^+),P(Z),X).
\end{aligned}
$$

The zero map preserves feature dimensions and zeroes both neighborhood slots
in both networks. F uses pre-stream W/Z perception; Q sees post-F W and
pre-update Z. Replacing T also replaces the incoming feature F reads.
All four combinations use the same weights and cold source/source-flip inputs.
Maximum graph hops per step are2 with perception,1 with transport alone,
and0 with neither. This is fixed-checkpoint intervention, not a communication-
depth-matched training comparison. Numeric legacy2K distance bands retain their
historical definitions rather than claiming the same dependency bound.

Full six historical evaluations replay exactly. Every knockout loses primary
paired correctness and near-cue behavior. This supports sensitivity of this
selected solution, with no unique semantic mechanism, training-cause or
retrained-model conclusion. See [results](evidence/stream_path_audit_seed4/RESULTS.md)
and [limits](evidence/stream_path_audit_seed4/INTERPRETATION.md).
Source: `step` in new/stream_path_audit/operators.py; frozen original
`StreamingCell.step` remains unchanged in new/streaming_carry/stream_cells.py.

## Previous architecture: Persistent Roles versus additive K8 and Streaming Carry

Input X has shape[B,3,height,width]: mask, positive source, negative source.
Persistent state is H[B,12,height,width], C[B,12,height,width] and
Z[B,8,height,width]. H is stationary computation workspace; Z is stationary
task-facing state; C has four N/E/S/W lanes, each with three payload channels.
These names provide structural default paths, not enforced semantic roles.

Initialize encoder(X) using the original pointwise3->24 encoder and split
its first12 channels into H, remaining12 into C; Z0=0. One shared pointwise
Tanh MLP R35->72->32 receives[H,C,Z,X]. Each phase is:

$$
\begin{aligned}
(\Delta H,\Delta C,\Delta Z)&=R_\theta(H,C,Z,X),\\
H'&=H+0.1\Delta H,\\
C'&=T_M(C+0.1\Delta C),\\
Z'&=Z+0.5\Delta Z.
\end{aligned}
$$

Repeat the same phase twice per macro-step. Readout8->1 uses Z alone.
R has4928 parameters, encoder96 and readout9, total5033. Module creation
first instantiates the default original cell, retaining encoder/readout
draws and RNG advancement; its F/Q modules are not retained in the candidate.
R is newly initialized and its final layer is zero. Full initial parameters
therefore differ despite common encoder/readout identity.

T_M is the validated masked port permutation with blocked/exterior bounce-back
and fixed wall ports. For fixed mask, the zero-residual base map
B=I_H\oplus T_C\oplus I_Z is isometric. After p phases C=T_M^p(C0), or
T_M^(2t)(C0) after t macro-steps. The general phase Jacobian is
J=B(I+D*dR/dS), with D having block scales0.1/0.1/0.5; trained stability,
losslessness and semantic payload preservation are not guaranteed.
Zero residual-head initialization gives J=B but initially blocks upstream
gradients through that head. Numerical state survives detach; gradients do not.

Only C crosses cell boundaries. R has no Laplacian or raw-neighbor inputs.
Since H/Z read incoming C BEFORE the phase streams, after p phases the
initial-carrier radius is<=p while the task-readout radius is<=p-1.
K8 has16 phases: carrier<=16 and task-readout<=15. The primary strict16<d<32
is beyond either bound. Two macro phases do not imply identical output
causal clocks or FLOPs to the old F/Q cell.

Old Streaming already has T_W\oplus I_Z and Laplacian residual perception.
This candidate adds a dedicated stationary H workspace while reducing
transported width24->12 and changing perception, rule sharing and readout
timing. Matching parameter/state counts does not isolate H causality.
The [completed development screen](evidence/persistent_roles_init2345/RESULTS.md)
is DEVELOPMENT_NO_GO: roles0/4 reach+hold, baseline2/4 and stream1/4;
all eight controls reproduce exactly. No mechanism, broad reliability,
family-wide impossibility, arbitrary addressing or3D result is established.

| Concept | Exact symbol | Source |
|---|---|---|
| One-time local/carrier initialization | PersistentRoleCell.initial | new/persistent_roles/role_cells.py |
| Shared pointwise collision and carrier stream | PersistentRoleCell.phase | new/persistent_roles/role_cells.py |
| Two phases per macro-step | PersistentRoleCell.step | new/persistent_roles/role_cells.py |
| Default-path, gradient and clock checks | check_independent_forward_gradient_and_base, check_readout_clock_and_isolation, check_three_state_K8_clock | new/persistent_roles/check.py |
| Frozen comparison | — | new/persistent_roles/PROTOCOL.md |

## Previous architecture: Local Interface versus additive K8 and Streaming Carry

The task and controls are the masked seeded-region system described in
[the frozen protocol](new/local_interface/PROTOCOL.md). Input X has shape
[B,3,H,W], persistent workspace W has 24 channels, and local state Z has 8.
Both state blocks retain identity paths at their cells. The candidate has a
transient message field M with 24 channels, arranged as four six-channel
directional ports. It is emitted afresh in each phase and is never stored as
an additional persistent state.

Let E be one shared pointwise affine map from [W,Z,X] (35 channels) to M
(24 channels). Let T_M be the masked four-port permutation, F the workspace
residual, and Q the local-state residual:

$$
\begin{aligned}
M_1 &= E_\theta(W,Z,X), & U_1 &= T_M M_1, \\
W' &= W+0.1F_\theta(W,Z,U_1,X), \\
M_2 &= E_\theta(W',Z,X), & U_2 &= T_M M_2, \\
Z' &= Z+0.5Q_\theta(W',Z,U_2,X).
\end{aligned}
$$

F and Q each receive 59 channels [W or W', Z, U, X] and use pointwise Tanh
MLPs 59->31->24 and 59->21->8. E is shared between phases; the message is
consumed by that phase's residual and discarded. F and Q do not read L(W),
L(Z), raw neighboring states or component IDs. X remains available at each
update. The two sequential phases allow at most two graph hops per macro-step,
or radius 16 within a K8 window.

The fixed masked permutation moves open ports to their neighbors and bounces
blocked or exterior links into the opposite directional lane; wall ports stay
fixed. T_M preserves global Euclidean norms. E and the learned recurrence do
not inherit that guarantee: for F=Q=0, the local state is unchanged regardless
of the emitted messages, while general emission, consumption, and
re-emission need not preserve task information or remain stable. This is an
ephemeral interface rather than a persistent carrier bank.

There are 5,033 trainable parameters: encoder 96, E 864, F 2,628, Q 1,436,
and readout 9. The common encoder/readout draws match the controls; the full
initial parameter set does not, because E/F/Q are new or differently shaped.
State allocation, neighbor representation and hidden widths change together,
so the experiment compares complete parameterizations and does not isolate
the effect of interface factorization.

The completed [development screen](evidence/local_interface_init2345/RESULTS.md)
is **DEVELOPMENT_NO_GO**: Local Interface reaches and holds in 0/4 seeds,
versus 2/4 for the additive baseline and 1/4 for Streaming Carry. This is
conditional developmental evidence on inspected seeds/maps. It does not
identify a failure mechanism, establish a family-wide limit, or support a
3D claim.

| Concept | Exact symbol | Source |
|---|---|---|
| Shared emitter and phase-specific transport | InterfaceCell._message | new/local_interface/interface_cells.py |
| Local state residuals and two-phase recurrence | InterfaceCell.step | new/local_interface/interface_cells.py |
| Three-arm factory | make_variant | new/local_interface/interface_cells.py |
| Frozen protocol and endpoint | — | new/local_interface/PROTOCOL.md |

## Previous comparison: Streaming Carry versus additive K8

The task and medium are the masked seeded-region system in
[the protocol](new/streaming_carry/PROTOCOL.md). Inputs X have shape
[B,3,H,W] (mask, positive source, negative source); persistent state W has 24
channels and local state Z has 8. W is divided into four six-channel
directional lanes N/E/S/W. The model keeps 32 persistent channels and 5,033
trainable parameters, matching the additive baseline.

For each open cell i and direction d, T_M sends lane (i,d) to the neighboring
open cell (i+delta_d,d). If that edge is blocked by a wall or exterior boundary,
the lane bounces to (i,opposite(d)). Wall-cell ports stay fixed, and isolated
open cells reverse directions in place. Every destination port has one
predecessor, so T_M is a permutation of position/direction registers. If B
reverses lane labels, T_M^-1=B T_M B and T_M^T T_M=I. The fixed transport
therefore preserves global Euclidean norms and distances. Directional ports
identify incoming lanes, not complete source provenance.

```math
\begin{aligned}
U &= T_M W,\\
W'_{\rm baseline} &= W+0.1F_\theta(W,Z,L_MW,L_MZ,X),\\
W'_{\rm stream} &= U+0.1F_\theta(U,Z,L_MW,L_MZ,X),\\
Z' &= Z+0.5Q_\theta(W',Z,L_MW',L_MZ,X).
\end{aligned}
```

The pointwise F input has 67 features `[U,Z,L_M W,L_M Z,X]`:
24+8+24+8+3 channels. F maps 67->40->24; Q maps 67->16->8 with Tanh hidden
activations and linear outputs. Readout remains a 1x1 map from Z to one logit.
`StreamingCell.step` computes old-state L(W) and L(Z) in parallel with T_M,
then uses incoming U as F's first feature. The old Laplacian features remain
available to the residual branch. Q applies the second local communication
phase, giving at most two graph hops per macro-step; applying L_M(T_M W) inside
F would add a third hop and change the protocol's spatial radius.

The fixed operator alone is lossless. The nonlinear residual can mix, amplify
or erase messages, and the local-memory update has no norm-preservation
guarantee. This is not a pure collision model: old neighbor features remain in
F. It adds neither learned routing nor carrier channels, and it does not restore
gradients across K8 detach points. No arbitrary-routing, causal failure
mechanism, broad robustness or 3D claim follows from the completed
[development screen](evidence/streaming_carry_init2345/RESULTS.md), which is
negative at the frozen gate (baseline reaches and holds in 2/4 seeds, stream
in 1/4).

Implementation lives in [stream_cells.py](new/streaming_carry/stream_cells.py);
the CPU operator and clock checks are in
[check.py](new/streaming_carry/check.py). The saved-result analysis and public
evidence exporter are [analyze.py](new/streaming_carry/analyze.py) and
[export_streaming_carry_evidence.py](tools/export_streaming_carry_evidence.py).

## Previous comparison: Direct Spatial Carry versus additive K8

The task and medium are the masked seeded-region system described in
[the protocol](new/direct_spatial_carry/PROTOCOL.md). Inputs X have shape
[B,3,H,W] (mask, positive source, negative source); persistent states W and Z
have24 and8 channels. The pointwise feature tensor has67 channels:
[W,Z,L_M W,L_M Z,X]. F uses67->40->24 channels, Q uses67->16->8, with Tanh
hidden activations and linear outputs; readout is a1x1 map from Z to one logit.

```math
\begin{aligned}
P_M &= I-\tfrac12D_M^\dagger L_M,\\
W'_{\rm baseline} &= W+0.1F_\theta(W,Z,L_MW,L_MZ,X),\\
W'_{\rm carry} &= P_MW+0.1F_\theta(W,Z,L_MW,L_MZ,X),\\
Z' &= Z+0.5Q_\theta(W',Z,L_MW',L_MZ,X).
\end{aligned}
```

D_M counts real four-neighbor open edges, excluding exterior ghost nodes.
Degree-zero nodes retain W. P_M acts independently on each channel; it
preserves constants, cannot cross closed edges and is maximum-norm
nonexpansive. It is lazy diffusion and mixes spatial values. These properties
do not guarantee Euclidean contraction, full-cell stability or useful message
preservation. The modified term replaces part of identity retention; it is
not an additional protected state bank. Both arms retain two communication
phases per step,5033 parameters and identical initialization draws.

`DirectCarryCell.step` in [carry_cells.py](new/direct_spatial_carry/carry_cells.py)
reuses the exact old L(W) feature at channel offset32. F receives pre-carry
features; Q receives the updated W. The baseline factory returns the unchanged
`RevisionCell('ws_additive')`. The [CPU check](new/direct_spatial_carry/check.py)
compares with an explicit neighbor loop and tests rho0 forward/gradient
equivalence. K8 detaches BOTH states while keeping their values; carry does
not restore gradients across those cuts.

[The completed development screen](evidence/direct_spatial_carry_init2345/RESULTS.md)
is negative: baseline reaches and holds in2/4 seeds, carry in0/4. All earlier
architecture definitions and results below are historical and unchanged.

## Historical 2D inertial wind-tunnel cell (seed 0)

The separate seeded-region task supplies three channels at every update:
open-region mask, positive seed and negative seed. Each connected region gets
a binary identity; BFS creates the target. Models are not given component IDs
or graph distances. Counterfactual evaluation flips the largest component's
seed while leaving the map geometry fixed. `X` stays available, so this is
conditional state repair. The ordinary five-point Laplacian is applied over
the whole grid: it does not remove transport across walls or gate edges by
connected component.

The candidate has 16 content channels and a persistent 16-channel velocity.
Its update is

\[
V_{t+1}=B V_t+\eta R_\theta(H_t;X)-D\mathcal L H_t,\qquad
H_{t+1}=H_t+V_{t+1}.
\]

Here `R_theta` is a two-layer, pointwise Tanh MLP over `[H,X]`; it does not
read neighboring cells. `B` and `D` are learned, spatially constant,
per-channel coefficients, initialized at beta=0.9 and d=0.1; eta=0.1. The
cell uses a replicated-edge, positive five-point Laplacian with unit grid
spacing. The coefficient bounds come from the pure-transport calculation;
they do not certify stability of the complete nonlinear cell.

| Arm | Per-step update | Comparison role |
|---|---|---|
| `inertial_rd` | `V'=B V+eta R(H,X)-D LH; H'=H+V'` | Explicit diffusion and inertia candidate |
| `momentum_nca` | `V'=B V+eta F(H,LH,X); H'=H+V'` | Generic momentum control with neighbor perception |
| `rd_nca` | `H'=H+eta R(H,X)-D LH` | No-inertia reaction–diffusion control |
| `nca_state_matched` | `U'=U+eta F(U,LU,X)`, with `U` having 2C channels | Persistent-state-capacity control |

All use 16 base channels and local inputs. Hidden widths are chosen for
approximate parameter-count matching; this does not match FLOPs, activation
memory, latency or representational capacity exactly. The generic momentum
arm learns a reaction that reads `LH`; the candidate's explicit Laplacian
term is not a mask-aware or component-aware transport operator.

In the seed-0 screen, at 32x32 and T=64, balanced accuracy was 69.94% for the
candidate, 84.44% for generic momentum and 85.17% for the state-matched arm.
At T=256, the candidate scored 50% balanced accuracy and zero paired-source
correctness at both 64x64 and 128x128. All sustained-95% thresholds were null;
candidate repair had zero eligible samples at every size and is therefore
unevaluable. These results support only a one-seed negative screen under this
recipe. They do not establish a family-level refutation, multi-seed advantage,
3D result, repair success/failure or speedup.

| Concept | Exact symbol | Source |
|---|---|---|
| Operators, arm updates and rollout | `laplacian`, `Cell.step`, `Cell.coefficients`, `Cell.rollout`, `make_cell` | `new/nca_inertial_wind_tunnel/cells.py` |
| Task generation and damage | `components`, `sample`, `bank`, `damage`, `metrics` | `new/nca_inertial_wind_tunnel/tasks.py` |
| Training and measured evaluation | `sustained_threshold`, `evaluation`, `gradient_probe`, `benchmark`, `train_one`, `main` | `new/nca_inertial_wind_tunnel/run_wind_tunnel.py` |
| Pure-transport linear checks | `roots`, `radius`, `block_matrix`, `run` | `new/nca_inertial_wind_tunnel/math_checks.py` |

The [compact evidence summary](evidence/inertial_seed0/summary.json),
[per-arm results](evidence/inertial_seed0/RESULTS.md), and
[publication manifest](INERTIAL_PUBLICATION_MANIFEST.json) are the entry
points for this screen.

The original operator/reaction below is unchanged. Current A0 extensions are
described in the final section and [A0_PROTOCOL.md](A0_PROTOCOL.md).

For dim=2 or 3, let the spatial grid be S and N=product(S). Batch input has shape
`[B,5,*S]`: signed source, source marker, target marker, binary region and local
detail. Normalized coordinates add dim channels. Pointwise stem and initializer
produce fixed E and persistent H with shape `[B,C,*S]`, C=64.

Two phases each repeat a shared local cell four times. Phase-start endpoint
features produce one undirected outgoing edge array `[B,G,*S]` per axis, G=4;
the terminal edge on each axis is zero. Positive tau is learned per phase/group.
Symmetric endpoint sum/absolute difference preserve edge reciprocity.

Each update computes channel-only normalization S=Norm(H), Q=Emit(S) with shape
`[B,r,*S]`, r=32, and M=T(Q). Transport is X-half/Y/X-half in 2D and
X-half/Y-half/Z/Y-half/X-half in 3D. Every factor is `(I+s*L_axis)^(-1)`.
The palindromic product is nonnegative, symmetric, doubly stochastic and
nonexpansive for a fixed medium; it is not the exact full multidimensional
resolvent.

The reaction reads `[S,DWConv(S),E,M,M-Q]`; a pointwise MLP produces delta and
a sigmoid gate reads `[S,E,M]`. Update is `H += gamma*gate*delta`. Target-only
gathered H enters a linear classifier (2 classes for A/B; 4 for C). There is no
global readout shortcut. Transport stability does not imply recurrent stability.

PCR solves batched independent lines with channel RHS sharing factorization.
The adjoint uses the same factors and propagates gradients to every edge and
tau. Factors are reused through each phase; dependence on phase-start H remains
in the graph. Only first-order differentiation is supported by the custom
adjoint; higher-order differentiation is not qualified.

For the unrun full-width retention comparison, `message_full` and `state_full`
both set Q=H, r=C and use the same computation. Residual base is H in the first
and M=T(H) in the second. This isolates retention without changing transported
width. B/C forward/backward smoke alone does not support a task-level advantage.

## A0 confidence-normalization intervention

`A0Model` imports the original sources without modifying them. Values
v=emit(Norm(H)) have shape `[B,r,*S]`; confidence c=sigmoid(f(Norm(H))) has
shape `[B,G,*S]`. Each group weight multiplies its r/G value channels, giving
q=c*v. Confidence begins at 0.5, receives no oracle source supervision and is
not claimed to be calibrated probability. Common initial parameters match.

`transport_pair` interleaves each group's r/G numerator channels and one
confidence channel into `[B,G*(r/G+1),*S]`. The same kernel transports both,
yielding n=T(q) and d=T(c). Raw uses m=n; normalized uses
m=n/(repeat_groups(d)+1e-6). Both compute n and d. The reaction still reads m
and m-q; in the normalized arm, this is not a linear diffusion-delta claim.

Constant conductance is 1. Learned conductance is 2*sigmoid of the original
symmetric endpoint features; its zero-initialized final head also yields 1.
Initial scale and output match in corresponding raw/normalized settings.
Reciprocity, phase caching and original fixed axis order are retained.

`SharedValueAttention` predicts query/key only, takes supplied q as values and
has no value/output projection. Persistent state and target readout are shared.

For fixed c/kernel and zero epsilon, positive-denominator normalization is a
row-normalized average; it does not generally inherit T's Euclidean
nonexpansion. For one active source, m_i=K_is*v_s/(K_is+epsilon). Epsilon,
underflow and background weights matter. Denominator differentiation is live
and checked with respect to q,c,edges and tau. No full-network stability or
arbitrary-message preservation theorem is claimed.

Diagnostics sample the final update. The source-mass statistic is
T(c*source_mask) at the target divided by clamped T(c), averaged over target
samples/groups, with per-axis records retained. It is not attribution of all
information: previous updates can place content in values emitted elsewhere.

## Masked inertial follow-up: neutral component means

For the explicit masked model only, `masked_cells.masked_laplacian` uses
edge weights m_i*m_j. On each open connected component C, sum_C L_m H=0.
With spatially constant, channelwise diagonal B and D, the component means obey

    mean_C(V_next) = B mean_C(V) + eta mean_C(R(H,X))
    mean_C(H_next) = mean_C(H) + mean_C(V_next).

This cancellation is exact algebra; it does not imply that the learned reaction
mean is constant, nonzero, or stabilizing. Nor is the mean equation closed in
the mean alone, since R depends on the full spatial state. If the reaction mean
approaches a nonzero vector r, then mean velocity can approach
eta*(I-B)^(-1)*r, permitting linear mean-state drift. A componentwise constant
target is in the operator nullspace, but the learned hidden representation
need not be componentwise constant. Isolated wall nodes also contribute neutral
modes and must be separated in diagnostics.

The observed rise in BA, BCE and H RMS at longer rollout establishes neither
fixed-point convergence nor exponential blow-up. Check component-projection
energy, residual variation, reaction/velocity means and direction changes.
Positive scalar logit-temperature adjustment cannot change decision signs;
held-out improvement after independent fitting probes scale sensitivity without
proving the cause of hidden-state drift.
