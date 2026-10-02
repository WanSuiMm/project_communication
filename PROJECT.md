# Reaction-Transport: local 2D / 3D qualification

Current completed screen (2026-10-02): matched short-BPTT training on the
existing workspace additive/revision cells, K8 versusK64, paired seeds0/1.
Eight runs; every training trajectory starts fresh, executes64 macro-steps,
has identical equally weighted losses at8,16,...64, and makes one optimizer
update only after all64 steps. K8 cuts BOTH W/Z history while retaining values.
No state pool, new model, warm-switch gate or pretrained weights. Frozen
protocol: new/short_bptt/PROTOCOL.md. Primary endpoint:size32/T64 paired
correctness beyond graph distance16; two communication phases per macro-step
are accounted for. Full controls must qualify before judging truncation.
Hold atT128/T256 and size64 are reported separately; model seed is the unit.

CPU forward/gradient/detach checks passed, including exact full-gradient
agreement with a reference and parameter-gradient accumulation across windows.
Four-arm CUDA preflight passed in9.375 seconds. Its worst steady-update
estimate was0.32699 seconds; the predeclared runtime-only budget rule selected
300 updates per arm, equally for all eight runs, before efficacy training.
Preflight: runs/short_bptt_20261002_preflight/. All paired initial-parameter,
data and schedule hashes match. Preflight training peaks were510.69MiB for
K64 and87.14MiB forK8 in both cells; these are engineering observations only.

All eight arms completed 300 updates in 730.797 seconds, finishing at
10:29:14 +08:00 on 2026-10-02. Output: runs/short_bptt_20261002_paired01/.
All four full-K64 controls failed the frozen far-paired qualification;
both architecture comparisons are BASELINE_UNQUALIFIED. Additive K8 seed1
meets its short reach/hold predicates: far paired 80.87% at T64 and 78.24%
at T256 on size32. The T64 pooled-pixel value is only 59.53%, and size64/T64
far paired is 28.40%. This is one positive case, not an architecture pass.
Peak allocated CUDA memory falls from 514.04 to 90.49 MiB (82.4%); training
times are similar. No general superiority or reliable scale extrapolation.

Public review: evidence/short_bptt_paired01/RESULTS.md and analysis.json;
BPTT_PUBLICATION_MANIFEST.json binds sources and evidence. CPU analysis at
analyses/short_bptt_20261002_review/ verified snapshots, parameter/data/schedule
hashes, 96 BA aggregates, 336 paired aggregates and all four pair decisions.
No extra training or inference was used for publication. Checkpoints and
machine receipts stay local; no further experiment or monitor is scheduled.

Commands from repository root, new output names only:

    python new/short_bptt/check.py
    python new/short_bptt/run.py --preflight --out runs/NEW_BPTT_PREFLIGHT
    pwsh -File tools/launch_short_bptt.ps1 -RunName NEW_BPTT_RUN -Preflight runs/NEW_BPTT_PREFLIGHT

Previous audit (2026-10-02): completed zero-training candidate/workspace source
switch audit on the existing revision seed0 checkpoint, sizes32/64,16 maps
each. Frozen protocol: new/switch_audit/PROTOCOL.md. Output:
runs/switch_audit_20261002_seed0/; completed in9.609 seconds. All232 historical
replay comparisons and the candidate step reconstruction have zero error.
Public report: evidence/switch_audit_seed0/RESULTS.md, analysis.json and
overview.png; SWITCH_PUBLICATION_MANIFEST.json binds raw inputs and sources.
Local analysis is preserved at analyses/switch_audit_20261002_review/.
Post-execution scope clarification: new/switch_audit/REVIEW_NOTES.md. Fresh-flip
replay verifies BA/BCE, while W/Z RMS replay is for original-input curves only.
Warm candidate readout remains old-aligned atK64/K128. Single-block W or Z
reset/transplant does not reliably recover source revision; reset-both cold
does. At size32/K128 all four single-block interventions predict negative
everywhere on the changed components, giving37.5% solely because6/16 target
labels are negative. Candidate-only W/Z cross-interventions show contributions
from both blocks and an interaction; a W-only stale-workspace mechanism is
not isolated. Mixed donor states may be off-distribution. This is one frozen
model, no new training, seed1/BPTT audit or architecture selection. The prior
NO_JOINT_SCREEN_PASS result is unchanged. No further experiment is scheduled.

Commands from repository root, with new output names:

    python new/switch_audit/audit.py --check
    python new/switch_audit/audit.py --out runs/NEW_SWITCH_AUDIT
    python new/switch_audit/analyze.py --run runs/NEW_SWITCH_AUDIT --out analyses/NEW_SWITCH_REVIEW

Authorized new screen (2026-10-02): workspace/task-state additive versus
candidate revision, paired seeds0/1. This new user request supersedes the
earlier training stop for this bounded experiment only. Both arms share W24/Z8,
5033 parameters, initialization, binary masked medium, two communication phases
per macro-step, training data and reach/hold/switch/repair supervision.
Only Z update changes: Z+0.5Q versus Z+0.5(Q-Z). Read the frozen protocol at
new/workspace_revision/PROTOCOL.md; entry is run_revision.py there.
The comparison tests revision within a shared state split, not the split itself.
Size32 training, size32/64 held-out evaluation through T256; 600 updates per
arm, two model seeds, 25-minute whole-run cap. No hyperparameter sweep.
CPU model checks and CUDA preflight passed. Based on preflight runtime,
formal updates were reduced from800 to600 equally before any efficacy run.
The final matching-source preflight completed in10.09 seconds at
runs/workspace_revision_20261002_dispatch_check/. Its two arms have identical
initial-parameter, data and schedule hashes. Read-only review found no scientific
control blocker; formal failure exit codes were corrected before dispatch.
Formal dispatch was verified on2026-10-02 at00:26:56 +08:00 on the local
RTX4060 Laptop GPU, including the child's first completed training update.
Output: runs/workspace_revision_20261002_paired01/. Completed2026-10-02 at
00:45:32 +08:00 in1112.28 seconds; all four arms completed600 updates.
Scientific status: NO_JOINT_SCREEN_PASS, mean paired hold effect-9.02pp.
Revision seed0 reaches99.19% BA64 and100% hold at size32, but warm source
revision scores0% on the changed component atK64 versus100% from cold start.
Seed1 remains50% BA. Z-only repair is stronger than joint W/Z repair; W keeps
growing. This does not validate the full reach/hold/reopen hypothesis.
Public review: evidence/workspace_revision_paired01/INTERPRETATION.md,
analysis.json and overview.png; REVISION_PUBLICATION_MANIFEST.json binds
code and evidence. Publication recomputed400 BA aggregates and all primary
predicates on CPU; no new training or GPU rollout. The local launch receipt
stays private. Old sources/evidence/checkpoints remain unchanged. No monitoring
or further experiment is scheduled.

Commands from this repository root:

    python new/workspace_revision/check.py
    python new/workspace_revision/run_revision.py --preflight --out runs/NEW_REVISION_PREFLIGHT
    pwsh -File tools/launch_revision.ps1 -RunName NEW_REVISION_RUN -Preflight runs/NEW_REVISION_PREFLIGHT

The historical entries below retain the stopping boundaries in effect at their
dates; they do not cancel the newly authorized screen above.

Authorized follow-up (2026-10-01): inference-only dynamical audit of the four
existing generic State/Momentum checkpoints, completing the medium by recurrence
comparison. No additional training or model change. Frozen protocol:
`new/dynamics_audit/PROTOCOL.md`; implementation: `operators.py` and `audit.py`
there. Sizes32/64; all16 original evaluation maps for curves; fixed maps0..3
for matrix-free Jacobian products at8 anchors, windows1/8/16,2 starts and16
power iterations. Report convergence diagnostics, free-momentum shear baseline,
drift, repeated source drive and readout sensitivity; do not assume a critical
zero crossing. Full/open endpoint norms use fixed raw H/V units.
The derivative/adjoint/finite-difference gate passed, including encoder plus
repeated source injection; the four-arm CUDA preflight passed in4.27 seconds.
Formal dispatch verified on2026-10-01 at16:26:16 +08:00, local RTX4060 Laptop.
Output: `runs/dynamics_audit_20261001_seed0/`; completed at16:28:11 +08:00 in
112.859 seconds. All40 historical replay comparisons have zero error; code and
checkpoint hashes verified. Public review: `evidence/dynamics_audit_seed0/INTERPRETATION.md`,
analysis.json and dynamics_overview.png there; `DYNAMICS_PUBLICATION_MANIFEST.json`
binds executed sources and public evidence. CPU-only post-hoc analysis command
(requires retained local audit output and checkpoints):
`python new/dynamics_audit/analyze.py --run runs/dynamics_audit_20261001_seed0 --out analyses/NEW_DYNAMICS_REVIEW`.
All256 open K16 log-gain estimates are positive already; late accuracy collapse
usually accompanies declining local gain, not a near-zero-to-positive crossing.
Only148/256 open K16 estimates satisfy convergence criteria; large finite
worst-direction perturbations often leave the linear regime. Generic Momentum
shows persistent updates and growing wrong confidence; its energy is largely
within-component variation, so the older explicit-inertial mean-drift explanation
cannot simply be reused. Neither asymptotic stability nor a causal mechanism is
established. The private launch receipt remains local. No additional experiment
or monitoring was started for the result inspection.
This follow-up changes only the diagnostic scope; the stop on new training holds.

Final authorized arm (2026-10-01): `masked_state_nca` supplies the missing
state-matched generic NCA/Momentum by whole-grid/masked medium 2x2 comparison.
Protocol: `new/masked_state/PROTOCOL.md`; model:H32, no velocity, hidden48,
4993 parameters, original state-NCA initialization/RNG. Frozen seed0/800-update
recipe and evaluator are reused. Main comparison is32/T64 BA and paired-source
correctness; all horizons and difference-in-differences are retained. State
scalar counts match, but program/readout widths and parameter counts differ.
Three implementation tests and the three-update CUDA preflight passed.
The run saved 800 updates and all 15 scientific endpoints, but the original
process exited without a final completion marker; cause is unknown. Original
run: `runs/masked_state_20261001_seed0/`, retained unchanged. Missing size128
timings and gradient probes were recovered without training in
`runs/masked_state_20261001_tail_recovery/`; two endpoints replay exactly.
Public evidence: `evidence/masked_state_seed0/RESULTS.md`, comparison.json and
recovery.json; hash bindings: `STATE_PUBLICATION_MANIFEST.json`.
At 32/T64, Masked State reaches 90.43% BA / 56.37% paired correctness versus
Masked Momentum 96.45% / 77.25%. Masking improves both primary metrics in both
generic recipes, but State BCE worsens and long-rollout BA collapses to 11.38%
at 32/T256. This supports a same-medium recipe advantage in one seed, not
isolated velocity causality or stable cellular computation.
Stop after this final arm. No extra seeds, architecture changes or recurring monitor.
From repository root, use new output directories:

```powershell
python new/masked_state/test_state.py
python new/masked_state/run_state.py --preflight --out runs/NEW_STATE_PREFLIGHT
python new/masked_state/run_state.py --out runs/NEW_STATE_CONTROL
```

Completed work (2026-10-01): inference-only trajectory audit of the existing
masked inertial checkpoint and one `masked_momentum_nca` training control.
The audit is now complete: all15 reference checkpoints replay exactly.
Read `evidence/trajectory_audit_seed0/INTERPRETATION.md` and RESULTS.md there.
Open-pixel norms grow, remaining wrong margins amplify, and independently fitted
temperature removes much of the32-scale BCE deterioration without changing BA.
Large-domain wrong decisions persist; fixed-point or projective convergence is
not established. Reproduce with
`python new/trajectory_audit/audit.py --out runs/NEW_TRAJECTORY_AUDIT`.
The control protocol is `new/masked_momentum/PROTOCOL.md`; same seed0,800 updates,
data and evaluator, with matched original momentum initialization. Three tests
passed (weight/RNG and all-open equivalence, isolation, finite backward).
The formal control finished at14:08:11 +08:00 on2026-10-01, all800 updates and
all evaluations valid. At32/T64 it reaches96.45% BA/77.25% paired correctness,
beating masked inertial RD by7.66/40.02pp. At32/T256 BA declines to77.68%
versus91.55% for the explicit candidate, exposing a horizon tradeoff.
Local run: `runs/masked_momentum_20261001_seed0/`; public evidence:
`evidence/masked_momentum_seed0/RESULTS.md` and comparison.json there.
Reproduce from this repository root with
`python new/masked_momentum/run_momentum.py --out runs/NEW_MOMENTUM_CONTROL`.
No ongoing monitor or additional seed/retuning is scheduled. Both recipes lack
evidence of convergent computation; read the audit and primary comparison together.

Previous follow-up (2026-10-01): the user authorized the bounded masked-medium
diagnostic in `new/masked_medium/PROTOCOL.md`. Only masked RD and masked inertial
RD are added, paired with the frozen unmasked seed-0 controls. The intervention
is input-mask edge weights m_i*m_j; initialization, data, 800 updates and all
evaluation settings are retained. Original sources and evidence stay unchanged.
The main outcome is the paired change at 32x32/T64; this does not test superiority
over a generic momentum model with the same masked graph.

Entry commands from this repository root (new output directories required):

```powershell
python new/masked_medium/test_masked.py
python new/masked_medium/run_masked.py --preflight --out runs/NEW_MASKED_PREFLIGHT
python new/masked_medium/run_masked.py --out runs/NEW_MASKED_SCREEN
```

Four operator/initialization/isolation checks passed, followed by a three-update
CUDA preflight for each masked arm. The formal run was dispatched on 2026-10-01
at 12:09 +08:00 on the local RTX4060 Laptop GPU; the child manifest was verified.
The formal run finished at 12:12:23 +08:00; both arms completed 800 updates.
At the prespecified 32x32/T64 endpoint, masked inertial RD improved balanced
accuracy from 69.94% to 88.79% and paired-source correctness from 17.81% to
37.22%, passing the descriptive joint +5 percentage-point criterion. Masked RD
declined from 81.27% to 73.21% and from 37.81% to 17.13%, respectively.
At T256, masked inertial RD reached 91.55% BA at 32x32 and 75.69% at 64x64,
but returned to 50.00% at 128x128. No sustained aggregate 95% endpoint was
reached. This single-seed diagnostic supports an operator-dependent improvement
within the inertial recipe; scale generalization and superiority over a masked
generic-momentum control remain unestablished.
Run directory: `runs/masked_medium_20261001_seed0/`. Read its status.json
for execution state and RESULTS.md / paired_comparison.json for completed arms.
No automatic additional seed, architecture change or continuous monitor is scheduled.

Previous screen (2026-10-01): the user-supplied inertial NCA wind tunnel is integrated
under `new/nca_inertial_wind_tunnel/`. Read its `WIND_TUNNEL.md` and
`LOCAL_INTEGRATION.md`. The latter records the original archive hash, limited
metric/output fixes and the frozen local 2D screen. Eight unit tests, the linear
checks and four full-size CUDA preflight arms passed. Seed 0 learning completed
on 2026-10-01 at 00:46:51 +08:00 on the local RTX 4060 Laptop GPU: all four arms
finished 800 updates and all three evaluation sizes without a numerical-failure
or budget-limit status. The
local run directory is `runs/inertial_20261001_seed0/`. Public results are in
`evidence/inertial_seed0/`; read its RESULTS.md and summary.json first.
INERTIAL_PUBLICATION_MANIFEST.json binds source, protocol and evidence hashes.
Private local status files and receipts are not required for GitHub review.

This is a negative exploratory result for the current inertial_rd recipe, not a
rejection of all PDE/NCA architectures. At 32x32 and T=64, balanced accuracy was
85.17% (state-matched NCA), 84.44% (momentum NCA), 81.27% (RD) and 69.94%
(inertial RD). Inertial RD declined to 53.50% at T=256, and to 50.00% at both
64x64 and 128x128 at T=256, where paired-source correctness was zero. No arm
reached sustained aggregate 95% BA. Inertial RD had zero pre-damage eligible
examples at every size, so its conditional repair outcome is unevaluable.
There is no benefit over generic momentum in this single-seed screen. Each size
has 16 held-out maps; no multi-seed superiority or universal impossibility claim
is established. The separately authorized masked follow-up above tests one
operator intervention; it does not revise this completed negative result.

Run from this repository root with CUDA (new output directory required):

```powershell
python new/nca_inertial_wind_tunnel/run_wind_tunnel.py --device cuda --arms all --seed 0 --steps 800 --minutes 25 --out runs/NEW_INERTIAL_SCREEN
```

This tests explicit transport plus inertia against generic momentum and state
capacity controls. It does not test redesigned graph substrates or 3D, and does
not revise the older A0 evidence. No automatic extra seeds or monitor is scheduled.

Previous work: A0 protocol `rt_a0_2d3d_v2_1` is implemented in `a0_models.py`
and `a0_runner.py`; see `A0_PROTOCOL.md`. It adds same-emission attention,
oracle transport diagnostics and a paired 2x2 normalization/medium comparison.
The new schedule has four seeds, 480 updates per trial, a 25-minute cap and
per-dimension positive-control stopping. It does not schedule B/C or monitoring.
The original v1 training sources and evidence remain unchanged.
Completed on 2026-09-30: 24/24 trials in the conditional A0 schedule. In 2D,
all four attention seeds pass and every RT arm is NOT_QUALIFIED_FIT. In 3D,
one attention seed fails, so its sixteen RT trials are skipped. The initial
40-trial maximum was conditional, not a remaining-work count.
Public evidence is in `evidence/a0_v2_1/`; see `RESULTS.md` first and
`A0_PUBLICATION_MANIFEST.json` for hashes. Numerical checks passed. No B/C,
further training or recurring monitor is scheduled by this publication.

Historical status: the first frozen local 2D/3D qualification completed. Gate A is
INCONCLUSIVE_POSITIVE_CONTROL in both dimensions: one attention seed solves
the task and one does not fit; learned transport remains near chance in both
seeds. B/C and the width sweep were not launched. See `RESULTS.md` and
`RUN_MANIFEST.md`. That historical run has no ongoing training or monitoring.
This is an architecture screen, not a natural-vision or foundation-model claim.

The question is whether persistent local state plus narrow, symmetric implicit
message transport can learn long-distance dependence, select relevant sources,
and preserve local detail at a useful measured cost.

Read `RESULTS.md` first, `PROTOCOL.md` for frozen gates and scope, and `THEORY.md` for the four short
checks. Source map: `data.py` generates tasks; `transport.py` implements PCR
and its implicit adjoint; `models.py` defines controls; `runner.py` trains,
evaluates and writes per-run `RESULTS.md` plus `aggregate.json`.

Run commands from this standalone repository root (Python with PyTorch and CUDA):

```powershell
python run.py check
python run.py smoke --out runs/NEW_SMOKE_DIRECTORY
python run.py qualify --out runs/NEW_RUN_DIRECTORY --minutes 25
python a0.py check --out runs/NEW_A0_CHECK_DIRECTORY
python a0.py qualify --out runs/NEW_A0_RUN_DIRECTORY --checks runs/NEW_A0_CHECK_DIRECTORY/checks.json
```

Output directories must be new. A timed-out or failed qualification is retained.
The original v1 uses two seeds; A0 uses four. Both use the local GPU and a
25-minute scheduling budget. No external datasets, remote jobs,
DEQ, directed transport, adaptive recurrence or generation tasks are involved.

The 3D task uses genuine Conv3d and three-axis transport on narrow volumes.
It does not establish performance on general 3D object geometry.
