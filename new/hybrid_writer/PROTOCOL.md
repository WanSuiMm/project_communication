# Explicit lane write budget under short credit

Protocol `hybrid_lane_write_budget_v1`, frozen before efficacy training.
Question: does a uniform, explicit persistent-write budget improve reliable
cold-start formation under K8? Does a small state-conditioned coefficient table
add benefit beyond a learned constant budget? The selected block7 audit motivates
this experiment but does not identify C, encoder, or Q as the unique cause.

## Architecture and assigned interventions

C24 has four directional lanes with six payload coordinates each; Z8 remains
stationary. Use the unchanged masked permutation T and pre-stream L(C), L(Z).
Proposal input chi=[T(C),Z,L(C),L(Z),X] has 67 channels. The common encoder,
Q (67->16->8), Z'=Z+.5Q, and Z readout remain trainable neural modules.

| Arm | Proposal | Persistent writer |
|---|---|---|
| neural | original 67->40 tanh->24 MLP | C'=U+.1m |
| budget | same MLP | bounded radial write, a_d=sigmoid(beta_d) |
| hybrid | same MLP | bounded radial write, lane-specific 3x3 table |
| affine_hybrid | affine 67->24 | same table/writer; secondary capacity ablation |

Bounded arms use delta_d=.1*a_d*m_d/sqrt(.5^2+mean_k(m_dk^2)).
The mean is over six payload coordinates at each cell, never over batch/space.
Thus RMS(delta_d)<=.1*a_d<=.1 regardless of proposal scale. m=0 preserves the
C transport exactly; this does not mean Z is unchanged. C corrections can
accumulate; no semantic closure, contraction or full-Jacobian guarantee follows.
The original tanh-hidden proposal already has a weight-dependent output bound;
the intervention is a fixed write budget independent of learned output weights.

For hybrid/affine_hybrid, phi1=r(T(C)_d)/(1+r(T(C)_d)) and
phi2=r(L(C)_d)/(1+r(L(C)_d)), r(v)=sqrt(mean_k(v_k^2)+1e-12).
Both axes have fixed piecewise-linear hat knots 0,.5,1. Weights are nonnegative
and sum to one. a_d=sum_pq h_p(phi1)h_q(phi2)sigmoid(beta_dpq).
The 36 coefficients are lane-specific and shared across cells, time and the
six payload coordinates. Features have no direct target/readout/cue/proposal/time
input; hidden amplitudes can still indirectly carry semantic information.

All beta=0, so a=.5. The proposal output layer (or affine map) is zero initialized.
Initial numerical outputs match all arms. At m=0, d(delta)/dm=.1I, matching the
original proposal interface. Writer-beta gradient is initially zero; proposal
can learn first, then writer. No auxiliary writer loss or random gate bias.
Copy shared E/Q/readout tensors from a canonical original StreamingCell per
block, plus the proposal MLP in its three arms. Do not infer matching from an
identical seed when module creation order changes. Counts are reported, not matched
by padding unused parameters. Proposal/writer Jacobian matching is local; affine
has a different tangent kernel and function class.

## Fixed training and independent units

Eight fresh paired initialization/schedule blocks, serial on the historical
local Torch2.5.1 FP32 backend. Init seeds110001..110008; schedules111001..111008.
Use the unchanged 512-map size32 training bank seed10002; one shared 300x8 IID
schedule per block. Rotate the four-arm order by block modulo4.

Every optimizer super-update repeats the SAME minibatch four times, cold-reset
64 steps each:256 forwards,32 K8 loss/backward windows, detach after each window,
mean of32 original-world balanced BCE losses, clip accumulated gradient norm1,
then one AdamW step. Parameters stay fixed throughout. Exactly300 updates.
AdamW lr=.001, wd=.0001, betas(.9,.999), eps1e-8, including writer parameters.
No warm states, paired loss, new transport, dimension change or rescue sweep.
CUDA Graph execution reuses the qualified coverage runtime. Actual-shape eager
versus captured loss/gradient/state/Adam arithmetic is checked on all four arms
before launch (atol1e-6, rtol1e-5); failure stops efficacy dispatch.

## Observation, endpoints and failure criteria

Save model+optimizer at u0,25,...,300. Shared fresh held-out banks:32 maps each
at sizes32/64, seeds112032/112064. Cold original/flipped rollout0..256, all integer
Boolean traces retained losslessly. Evaluation never updates model parameters.
Use the unchanged coverage evaluator, strict16<d<32 reach, T64-correct all-changed
retention toT256, continuous survival, acquisition/regression and T64/128/256 BA.
Old Full is measured at u300 separately; it never replaces joint readiness.

Primary checkpoint is u300. Joint readiness J uses size32 strict T64 pooled AND
map-mean coverage>=.80, retention64->256>=.95, and retention support>=16 maps
and100 reference pixels. Undefined ratios stay null. Independent unit is a
paired training block, not maps, cells, checkpoints or telemetry samples.

Primary comparisons: budget-minus-neural, hybrid-minus-budget, hybrid-minus-neural.
Report all8 paired outcomes, success rates/Wilson intervals and exact two-sided
discordant-pair binomial p. Formal improvement requires net gain>=6/8 AND
Holm-adjusted p<=.05 across the three primary comparisons, after all32 trajectories
complete. Affine-minus-hybrid is secondary without a primary qualification claim.
Otherwise retain NO_PRIMARY_RELIABILITY_QUALIFICATION; partial/error is INCOMPLETE.
No posthoc width, coefficient feature, budget, checkpoint or cohort changes.

Secondary formation statistic: any J on the predeclared u0,25,...,300 grid,
first observed ready update, and loss of readiness thereafter. This is observed
ever-ready, not a claim about unsaved updates and not model selection. It cannot
replace u300. Record gate and write-RMS distributions on open cells at steps
1,8,16,32,64,128,256 in both worlds/sizes. Summaries/quantiles only, no hidden
trajectory dump; telemetry is descriptive and supplies no correctness detector.

If budget and hybrid improve similarly, favor the simpler constant-budget arm.
If hybrid alone improves, retain state conditioning as a candidate. Any gain is
for this finite K8 recipe and parameterization; it does not uniquely identify
write magnitude as the original collapse cause or demonstrate general BPTT/3D.
Affine failure does not distinguish inadequate capacity from optimization.

## Execution and provenance

New run directory only; freeze source snapshots/hashes, banks, all schedules,
shared initial tensors, runtime qualification and configuration before training.
Stop on nonfinite values, disconnected gradients, broken source/input bindings,
graph mismatch or evaluation parameter mutation. Keep ERROR/partial outputs.
No runtime cap, watchdog or recurring monitor. Measured duration is an estimate.
Use a persistent tool-owned foreground session, with local PID/GPU/command/time
receipt and verification of the first real optimizer update after tool yield.
Checkpoints and machine/session receipts stay local until an explicit publication
request. Scientific packed evidence and small telemetry are retained.

Repository-root commands:

    python -X utf8 -B new/hybrid_writer/check_cells.py
    python -X utf8 -u -B new/hybrid_writer/run.py --check --out analyses/NEW_HYBRID_QUALIFICATION.json
    pwsh -File tools/launch_hybrid_writer.ps1 -RunName NEW_HYBRID_RUN -Qualification analyses/NEW_HYBRID_QUALIFICATION.json

Qualification estimates32 trajectories and416 checkpoint evaluations. Dispatch
verification is completion of the launch request; result inspection is separate.
