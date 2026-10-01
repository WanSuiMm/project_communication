# Generic NCA dynamics audit: frozen seed-0 checkpoints

Question: do finite-window amplification, driven state drift, and task-information
readout explain the observed early improvement and late deterioration? A zero
crossing of a finite-time exponent is a hypothesis, not a required outcome.
This is an exploratory post-training diagnostic, not a preregistered efficacy gate.

## Fixed scope and controls

No optimizer, training, checkpoint edits, new model, calibration fit or new seed.
Use the four original 800-update checkpoints: whole-grid State, whole-grid
Momentum, Masked State, Masked Momentum. SHA256 identities are pinned in audit.py.
Verify the published source/reference hashes and all five old horizons at sizes
32 and 64 against the original 16-map evaluation banks (seeds 30032 and 30064).
BA, BCE, H/V RMS and paired-source correctness must replay within 5e-5.
Stop interpretation on a mismatch; preserve partial output. Size128 is outside
this bounded diagnostic, not an assertion of scale generalization.

Trajectory anchors: 16,32,48,64,96,128,192,256. Window lengths: 1,8,16, so the
last diagnostic window ends at272. Curves use all16 maps. Expensive tangents use
fixed map indices0,1,2,3, selected without inspecting performance. One trained
seed is the independent replicate; maps and timing windows are not new seeds.

## Exact derivatives and norm convention

For generic force f=eta W2 tanh(W_H H+W_L L H+W_X X+b)+b2, use the exact
matrix-free tangent D=partial_H f and transpose, with tanh slopes at the saved
trajectory. Implemented in operators.LinearStep, checked against autograd on a
tiny float64 grid with nonzero final weights and nonuniform beta.

    State J = I+D
    Momentum J = [[I+D, B], [D, B]]
    Momentum J*(u,w) = (u+D*(u+w), B*(u+w))

B is channelwise beta replicated over sites. L is the actual frozen graph
operator; the masked Laplacian and whole-grid Laplacian are symmetric.
Products apply different per-time Jacobians and reverse order for the adjoint.
Power iteration uses16 iterations,2 predetermined random starts. Store midpoint
and final estimates, singular residual and inter-start spread for each map.
Mark converged only if relative residual<=.01 and inter-start spread<=.02.
All estimates remain lower estimates, never certified upper bounds. No extra
iterations or restarts are chosen to obtain a preferred result.

Use raw Euclidean H/V units, no time-dependent whitening. Report both full
state products and P_open J_product P_open, projecting only the two endpoints
and both H/V blocks. This does not remove intermediate wall paths in whole-grid
models. Report log(sigma_estimate)/K as finite-window log gain, not an
asymptotic Lyapunov exponent or proof of instability. Include the analytic
free-momentum baseline (D=0, actual learned beta) to expose transient shear.
For Momentum, estimate the HH and VH block norms and report the exact HV/VV
norm max(beta). These are block diagnostics, not a decomposition of causality.

## Task information, drift and nonlinear checks

For every anchor report all-map BA, BCE, paired correctness, signed margins,
H/V norms, H update norm, and component-constant energy fractions for H,V,deltaH.
Component labels and target labels are used only for diagnostics, never fed to
the cell. Preserve per-map metrics so Jacobian maps can be compared directly
instead of correlating their statistics only with the full16-map average.

The binary source flip changes only positive/negative source channels; geometry
is held fixed. Propagate encoder tangent and the repeated direct source input:

    u_next = J u + partial_X F deltaX.

Report total, initial-only and drive-only tangents separately. Renormalize
vectors while tracking log magnitude to avoid overflow; report readout magnitude,
signed target response and alignment to the actual finite binary source flip.
Finite and infinitesimal interventions need not agree. A mask-channel change
would require partial L_m/partial m and is explicitly NOT part of this audit.

At every anchor, test16-step open-state perturbations in a random direction and
the estimated leading direction, using relative state RMS scales1e-4 and1e-3
(reference RMS floored at1). Compare observed and linear-predicted amplification,
linearization error and task metrics. Recompute the unperturbed window with the
same batch shape; separately record its difference from the full-bank trajectory.
Large numerical/linearization errors invalidate local causal interpretation of
that perturbation, not the archived original model evaluation.

## Qualification, budget and interpretation

Run one derivative-check command, then one CUDA preflight with all four arms,
size32,anchor32,map0,4 power iterations and2 starts. The preflight is engineering
validation and timing only. Formal settings are not tuned to preflight scores.
Formal cap25 minutes; preflight cap3 minutes. Stop on nonfinite values, replay
failure or time cap. Save partial data and explicit failure state; do not retry
the formal run or reduce the protocol after observing results.

A useful result may show early amplification, no zero crossing, or state drift
without rising finite-window gain. Do not force a criticality narrative.
Power spectra, source tangents and readout sensitivity are diagnostic evidence;
they do not by themselves establish causal medium-momentum interaction.
State and Momentum differ in H/program widths and parameter count. No isolated
velocity effect, asymptotic convergence, Jacobian upper bound, performance
speedup or new architecture is claimed. No recurring monitoring is authorized.

From the repository root, use new output directories:

```powershell
python new/dynamics_audit/check.py
python new/dynamics_audit/audit.py --preflight --out runs/NEW_DYNAMICS_PREFLIGHT
python new/dynamics_audit/audit.py --out runs/NEW_DYNAMICS_AUDIT
```

Manifest binds code, protocol, checkpoint identities and runtime. Local receipts
contain execution identity and are not public evidence. Read RESULTS.md first,
then per-arm curves and dynamics; summary.json retains the complete audit.
