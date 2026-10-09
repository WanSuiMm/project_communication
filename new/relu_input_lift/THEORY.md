# ReLU-conditioned input lift and its K8 gradient

## Original cell and exact lifted state

At each grid site let `x_t` contain 16 visible channels. The fixed perception
operator concatenates the identity, horizontal Sobel, and vertical Sobel
responses. Appending a constant one gives `p_t` in R^49. The first affine
layer has `A` in R^(128 x 49), and the output projection is split into
`W_h` in R^(16 x 128) and `w_0` in R^16. Define

$$
z_{t,i}=A p_{t,i},\qquad d_{t,i,j}=\mathbf{1}[z_{t,i,j}>0],\qquad
h_{t,i,j}=d_{t,i,j}z_{t,i,j}.
$$

For a fixed ReLU branch `d_t`, the hidden-layer and output projection form the
activation-conditioned local map
$$
K_{t,i}=W_h\operatorname{diag}(d_{t,i})A,
\qquad W_hh_{t,i}=K_{t,i}p_{t,i}.
$$
`K_t` changes when the preactivation branch changes; it is not an additional
trainable parameter.

For a scalar firing mask `m_{t,i}` shared across channels, the candidate
state at site `i` is
$$
\widetilde{x}_{t+1,i}
=x_{t,i}+m_{t,i}\left(W_h h_{t,i}+w_0\right).
$$
Let `q_{t,i}` be the scalar per-site mask formed from the conjunction of the
pre-update and post-candidate 3-by-3 alpha-alive masks. The original state
update is `x_{t+1,i}=q_{t,i} \widetilde{x}_{t+1,i}`. The alive test is a hard branch
and is treated as fixed in the gradient, as in the historical cell.

The lift stores a base `b_{t,i}` in R^16, an input accumulator
`U_{t,i}` in R^(128 x 49), and a scalar count `v_{t,i}`. For hidden unit `j`, define
the projected feature accumulator
$$
e_{t,i,j}=\langle A_j,U_{t,i,j,:}\rangle,
\qquad x_{t,i}=b_{t,i}+W_h e_{t,i}+w_0v_{t,i}.
$$
Initialize `b_{0,i}=x_{0,i}`, `U_{0,i}=0`, and `v_{0,i}=0`. Let `p_{t,i}` include its appended
constant coordinate. The lifted recurrence is
$$
\begin{aligned}
b_{t+1,i}&=q_{t,i} b_{t,i},\\
U_{t+1,i,j,:}&=q_{t,i}\left(U_{t,i,j,:}+m_{t,i}d_{t,i,j}p_{t,i}^\mathsf{T}\right),\\
v_{t+1,i}&=q_{t,i}(v_{t,i}+m_{t,i}).
\end{aligned}
$$
All masks in these equations are scalar per site, so they commute with
`A` and `W_h`. Therefore
$$
\begin{aligned}
b_{t+1,i}+W_h e_{t+1,i}+w_0v_{t+1,i}
&=q_{t,i}\left(b_{t,i}+W_h e_{t,i}+w_0v_{t,i}
  +m_{t,i}W_h(d_{t,i}\odot A p_{t,i})+m_{t,i}w_0\right)\\
&=q_{t,i}\left(x_{t,i}+m_{t,i}(W_h h_{t,i}+w_0)\right)\\
&=x_{t+1,i}.
\end{aligned}
$$
Thus, in exact arithmetic, the original and lifted cells have the same
visible trajectory for every fixed parameter set, starting state, and firing
plan. They also compute the same alive masks when the arithmetic follows the
same branch. Re-association in finite precision can change values near an
alive or ReLU threshold; a qualification check must report numerical errors
and mask disagreements rather than asserting bitwise equality.

## What K8 retains

At a K8 boundary, detach the numeric values of `b`, `U`, and `v` without
clearing them. Reconstructing the visible state with live parameters keeps
`A`, `W_h`, and `w_0` connected to the terminal loss. With the final window
beginning at `\tau=56`, let `\delta_\tau` be the derivative of the terminal
loss with respect to the visible state at that boundary after propagating
through steps 57–64. Relative to ordinary K8, the boundary reconstruction
adds direct projection gradients. Summing over batch and spatial sites gives
these terms:
$$
\Delta\nabla_{W_h}L=\sum_i\delta_{\tau,i} e_{\tau,i}^\mathsf{T},\qquad
\Delta\nabla_{w_0}L=\sum_i\delta_{\tau,i} v_{\tau,i},
$$
and the explicit first-layer contribution
$$
\Delta\nabla_{A_{j,:}}L
=\sum_i\left(W_h^\mathsf{T}\delta_{\tau,i}\right)_j U_{\tau,i,j,:}.
$$
The `W_h` and `w_0` terms are the same direct accumulated-feature projection
available in AU-NCA. The input lift adds direct credit to `A` through the
stored per-unit inputs. The gradient through the final eight steps remains
ordinary autograd; earlier contributions stored in `U` are numeric values,
not a retained autograd graph.

To expose the remaining truncation, hold each ReLU branch `d` fixed and define
the tensor
$$
R_{i,j,k}=\left(W_h^\mathsf{T}\delta_{\tau,i}\right)_j A_{j,k}
$$
for each site `i`. If `U_\tau` were differentiated through its entire history,
there would also be the term
$$
\left(\partial_\theta\operatorname{vec}U_\tau\right)^\mathsf{T}
\operatorname{vec}(R_\tau),
$$
where `\theta` denotes the trainable parameters and the derivative includes
the historical dependence of perceptions on earlier visible states. K8
detaches `U_\tau`, so this historical term is absent. The explicit
`A`-through-`A U` gradient is not a replacement for that residual. The lift
therefore carries a direct, local eligibility-like statistic for `A`; it does
not implement full RTRL and gives no general guarantee about delayed feedback
credit or learning quality.

## Scope and costs

The equality assumes that parameters are fixed throughout the 64-step
rollout; the firing mask is scalar per site and shared across channels; the
same alive and ReLU branches are used; and the cell update consists only of
the stated additive `W_h h + w_0` write followed by the stated mask. A
channel-specific firing mask, another state-writing path, or an optimizer
step inside the rollout breaks this identity as written. The ReLU and alive
branches are nondifferentiable at their thresholds; the derivation is
piecewise, with hard branches treated as fixed. Floating-point
re-association means practical equivalence is numerical rather than
bitwise.

The lifted state stores 16 base values, `128 x 49 = 6,272` input-accumulator
values, and one count per cell: 6,289 float32 values. AU-NCA stores 16 base
values and 129 accumulated features: 145 total. The lift therefore uses
about 43.4 times as many recurrent state scalars per cell as AU-NCA, before
autograd and optimizer memory. This is a memory-cost statement, not a measured
GPU-efficiency result. The derivation concerns one NCA parameterization and
one procedural target; it does not establish universal BPTT recovery, full
parameter credit, stability, or a general benefit from the additional
state.
