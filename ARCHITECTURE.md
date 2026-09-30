# Tensor and operator map

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
