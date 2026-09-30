# Tensor and operator map

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
