# Exact lift of a state-nonlinear recurrence on a cyclic 2D grid

This closes the two separate gaps in the supplied sanity scripts: true
quadratic dependence on recurrent state and cyclic two-dimensional transport
are present in the same cell. It is a finite-degree, triangular construction.

Let e_t be a parameter-independent 3D feature at every grid cell. P rolls
one row forward and Q rolls one column backward, with periodic boundaries.
They act on spatial axes only. Time t starts at zero and d_t=max(1,t).
The original cell has scalar fields u,z and trainable vectors a,b,gamma in R^3:

```math
v_t=P u_t/d_t,
u_{t+1}=P u_t+a^T e_t,
z_{t+1}=Q z_t+(b^T e_t)(gamma_0+gamma_1 v_t+gamma_2 v_t^2).
```

Unlike the bilinear state-input example, this transition has nonzero second
derivative with respect to u whenever gamma_2*(b^T e_t) is nonzero.
The fixed initial state is zero. Terminal dense prediction is tanh(z_T/T).
There is no EMA, learned input feature map, or learned transport.

Use parameter-independent fields U,M0,M1,M2 of dimensions 3,3,9,27:

```math
V_t=P U_t/d_t,
U_{t+1}=P U_t+e_t,
M0_{t+1}=Q M0_t+e_t,
M1_{t+1}=Q M1_t+V_t outer e_t,
M2_{t+1}=Q M2_t+V_t outer V_t outer e_t.
```

All moments start at zero. The exact projections are

```math
u_t=<a,U_t>,
z_t=gamma_0<b,M0_t>+gamma_1<a outer b,M1_t>
    +gamma_2<a outer a outer b,M2_t>.
```

Proof: the U identity follows by induction because P acts only on space.
Likewise Q commutes with each channel contraction. Contracting the three
moment increments gives (b^T e_t), (a^T V_t)(b^T e_t), and
(a^T V_t)^2(b^T e_t), respectively. Since a^T V_t=P u_t/d_t,
their weighted sum is precisely the original z write. This proves every
projected state and output equal at every time and at every parameter value.
P and Q do not need to commute with each other. Spatial cycles cause no
failure of this induction; it describes walk accumulation, not unique-node
set aggregation or duplicate removal.

For fixed inputs, each raw field has zero parameter Jacobian. Thus for a
terminal loss, differentiating the current contraction gives the same
parameter gradient as full BPTT of the original cell. A cut before the last
K>=1 steps changes neither the raw field values nor their parameter Jacobian:

```math
grad_theta L_lift,K = grad_theta L_original,full.
```

This is an exact function identity in real arithmetic, not a bijective
change of the original 2D state coordinates. An identical deterministic
optimizer therefore has identical original-full and lifted-short training
trajectories in exact arithmetic. Floating-point implementations can diverge.
Early input gradients are still cut; the theorem concerns a,b,gamma only.

The construction costs 42 rather than 2 scalars per cell, plus one shared
integer time counter. It does not allow arbitrary learned feedback into the
raw fields. Moment growth, floating-point conditioning, learned routing,
nonlinear learned feature extraction, and stable autonomous computation are
not resolved by this identity. These limitations remain even if training
on the matched teacher succeeds.

The previous nested typed hierarchy is not used here. M2 explicitly stores
the square of the same normalized pre-update prefix used by the original
writer, rather than a nested third-order iterated sum.
