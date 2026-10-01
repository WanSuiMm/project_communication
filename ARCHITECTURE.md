# Tensor and operator map

## Current 2D inertial wind-tunnel cell (seed 0)

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
