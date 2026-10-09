# Forward identity and W-specific credit in AU-NCA

Consider a fixed-parameter NCA rollout on a two-dimensional grid. Let x_t
contain 16 visible channels. Perception applies identity and two fixed Sobel
filters channelwise, then a shared learned affine map and ReLU produce 128
features. Appending a constant one gives h_t in R^129. A learned matrix
W in R^(16 by 129) produces the state increment. For a scalar firing mask
m_t at each cell, shared across channels, the candidate update is

$$
\widetilde{x}_{t+1}=x_t + m_t \odot (W h_t).
$$

The alive mask is the local 3-by-3 maximum of alpha greater than 0.1. Compute
the pre-update and post-update alive masks from their respective visible
states, then gate the candidate state by their conjunction. Perception uses
zero-padded fixed Sobel filters.

## Accumulated-feature representation

Represent the same visible state by a 16-channel base field b_t and a
129-channel accumulator e_t:

$$
x_t=b_t+W e_t,\qquad
\widetilde{x}_{t+1}=x_t+m_t\odot(W h_t),\qquad
q_t=\operatorname{alive}(x_t)\land\operatorname{alive}(\widetilde{x}_{t+1}),
$$

$$
b_{t+1}=q_t\odot b_t,\qquad
e_{t+1}=q_t\odot(e_t+m_t\odot h_t).
$$

Initialize b_0=x_0 and e_0=0. Since m_t is a scalar per cell, it commutes
with W. The next projected state is

$$
\begin{aligned}
b_{t+1}+W e_{t+1}
&= q_t\odot\bigl(b_t+W e_t+W(m_t\odot h_t)\bigr)\\
&= q_t\odot\widetilde{x}_{t+1}.
\end{aligned}
$$

This is exactly the original post-gated candidate update. By induction,
both systems produce the same visible trajectory for every fixed parameter
value, initial state, and firing-mask plan.

The augmented representation stores 129 accumulator scalars plus 16 base
scalars per cell, or 145 scalars. The original visible state stores 16.
At the start of each optimizer update, set b_0 to the sampled visible pool
state and e_0 to zero. The pool therefore stores only visible states. At a
K8 boundary, detach the graph of e while retaining its numeric value; do not
reset the accumulator. The base carries no parameter history from the
rollout. Canonicalizing at an evaluation branch as b_t=x_t and e_t=0 also
preserves x_t exactly.

## What the truncated gradient changes

With no detach, the equations above are a differentiable reparameterization
of the original cell. Full BPTT therefore gives the same parameter
gradients in exact arithmetic. Under K8, the numeric accumulator carries
features from all earlier windows into the current visible state, but
detaching e cuts gradients from later losses into the earlier computations
that produced those features.

The projection remains x_t=b_t+W e_t after a cut. Consequently a later loss
still differentiates through W's projection of the complete accumulated
feature value, including contributions written in earlier windows. The
feature extractor that produced those earlier values remains outside the
gradient graph. AU-NCA therefore changes the direct credit available to W
while retaining K8-truncated credit for the feature parameters. It does not
restore the full BPTT gradient for the whole recurrent system.

For one terminal loss and the final K8 window beginning at boundary tau=56,
let delta_tau be the derivative of that loss with respect to the visible
state at the boundary, after propagating through the final eight steps.
Relative to original K8, AU-NCA adds the direct projection term

$$
\nabla_W L_{\mathrm{AU,K8}}-\nabla_W L_{\mathrm{original,K8}}
=\delta_\tau e_\tau^{\mathsf T}.
$$

The feature-parameter gradient is the same as original K8. The added W term
can help or oppose the suffix gradient; retaining earlier features does not
guarantee a better update.

The direct W credit depends on the specified additive state write, scalar
per-cell firing mask, unchanged W throughout each rollout, and the
accumulated-feature state. A channel-specific firing mask, an extra learned
state write outside W h_t, or an optimizer step inside the rollout would
invalidate this identity as written.

This result concerns one fixed NCA parameterization and one local target
task. It does not establish that eight-step truncated BPTT is generally
repaired, that all parameters receive full temporal credit, or that the
augmented state has favorable memory or runtime cost. The empirical screen
must separately establish that the K64 positive controls learn the target.
