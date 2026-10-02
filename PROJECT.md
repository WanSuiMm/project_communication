# Reaction-Transport: local 2D / 3D qualification

Latest completed experiment (2026-10-02): local persistent state plus a learned
ephemeral communication interface. Frozen protocol: new/local_interface/PROTOCOL.md.
W24 and Z8 both retain local identity. One shared pointwise E35->24 emits
four6-channel directional messages before each of two residual phases; only
those messages cross open edges through the existing masked port permutation.
F59->31->24 and Q59->21->8 give exactly5033 parameters with32 persistent
channels and24 transient message channels. Encoder/readout draws match the
controls; differently shaped F/Q and E do not share complete initialization.
The candidate has no direct Laplacian/raw-neighbor input or persistent carrier.
This changes the complete parameterization, including hidden-width allocation;
it does not isolate a causal effect of role separation or match FLOPs/latency.

Matched development screen: baseline/stream/interface, seeds2/3/4/5,
300 updates per arm (12 arms), unchanged K8 training and exact historical
data/schedule. Same size32/T64 strict16<d<32 mean+pooled and BA reach gate,
with hold at bothT128/T256. Control full evaluation/final parameters must
reproduce history. Continue only if interface reach+hold>=3/4 including2/5,
and its count exceeds both concurrent controls. No mechanism is inferred
from prior stream seed4. Frozen estimate/hard cap40 minutes, watchdog41.
CPU checks passed:5033 parameters, common encoder/readout initialization,
independent nonzero forward/gradient reference, local zero-residual identity,
two-hop light cone/component isolation and unchanged K8 gradient clock.
Runner fresh-process imports,45 source bindings, seven decision truth cases,
eight corrupted hold cases and historical control identities/anchors passed.
Three-arm CUDA preflight passed in8.547 seconds at
runs/local_interface_20261002_preflight/. Peak allocated memory was87.14/88.70/
89.49MiB (baseline/stream/interface). Worst steady update0.4295 seconds gives
2035.44 seconds (33.92 minutes) under the frozen estimate, within the40-minute
cap. Primary coverage:32 maps and2918 pixels. The launcher verified the first
training update and child manifest. Original outputs and private launch
metadata remain local under runs/local_interface_20261002_init2345/.
All12 arms completed300 updates at18:18:29 +08:00 in1281.532 seconds
(21.36 minutes). DEVELOPMENT_NO_GO: baseline reach+hold2/4, stream1/4,
interface0/4. All8 control final parameter hashes and complete evaluation
payloads exactly reproduce their historical records. Interface primary
size32 strict16<d<32 mean and pooled accuracy are0% for every seed at
T64/T128/T256 (0/2918 paired-correct pixels). Size64 d>32 is also0% for
every seed/horizon (0/27436). Near-cue behavior remains seed-dependent;
longer rollout does not recover the required remote behavior. Interface
seed4 Hold=True retains a failed endpoint, not successful computation.
Saved-result CPU audit passed:45 executed/current source bindings,12 checkpoint
parameter hashes,1008 paired aggregates reconstructed from integer per-map
counts,936 curve rows,624 paired contrasts and all12 reach/hold decisions.
This negative result concerns the complete ephemeral-interface parameterization
under this developmental recipe; it does not isolate the cause or establish
that local-state/interface separation is generally impossible. No follow-up
training or monitor is started. Public review starts at
evidence/local_interface_init2345/RESULTS.md, analysis.json and validation.json;
source/evidence bindings are in LOCAL_INTERFACE_PUBLICATION_MANIFEST.json.
Commands from repository root:

    python new/local_interface/check.py
    python new/local_interface/run.py --preflight --out runs/NEW_INTERFACE_PREFLIGHT
    pwsh -File tools/launch_local_interface.ps1 -RunName NEW_INTERFACE_RUN -Preflight runs/NEW_INTERFACE_PREFLIGHT

Previous completed experiment (2026-10-02): lossless-path Streaming Carry,
new/streaming_carry/PROTOCOL.md. W24 becomes four directional6-channel lanes;
Z8 remains stationary local state. Fixed streaming follows open edges and
reverses direction on blocked edges; wall ports stay fixed. This is a
permutation of position/direction registers, not normalized neighbor averaging.
Learned modules,5033 parameters, initialization and the K8 recipe are matched.
F reads incoming T(W), Z and OLD L(W)/L(Z); Q reads updated W. This explicit
clock retains two-hop macro-step locality while allowing residual modification
of arriving messages. The residual branch retains old neighborhood perception;
this is not a model with all Laplacian features removed.

Development only: rerun baseline/stream on inspected seeds2/3/4/5, same
train bank10002/schedule20002/eval40032,40064;300 updates per arm,8 arms.
Same primary reach/hold gate: stream>=3/4, retain2/5, exceed current baseline
count; baseline2/5 must reproduce. No efficacy tuning, automatic confirmation,
new topology, sweep or monitoring. Prior averaging-carry failure is unchanged.

CPU checks passed: all64 binary2x3 masks agree with independent port scatter,
bijectivity/inverse/norm/adjoint hold, no wall/component leakage, identical
initial parameters, T=I old-cell forward/gradient equivalence, nonzero residual
reference, pure-stream zero-residual rollout and two-hop/K8 dependency checks.
Two-arm CUDA preflight passed in7.359 seconds at
runs/streaming_carry_20261002_preflight/. Peak allocated memory baseline87.14
and stream88.70MiB. Worst steady update0.406s gives1289.28s (21.49 minutes)
under the frozen estimate, within the25-minute cap (26-minute watchdog).
Historical initial/data/schedule/evaluation identity bindings are checked.
All8 arms completed300 updates at16:27:07 +08:00 in828.297 seconds
(13.80 minutes), within the25-minute cap. DEVELOPMENT_NO_GO: baseline
reach+hold2/4 versus stream1/4. Stream loses historical positive seeds2/5
and rescues seed4. Seed4 primary pooled accuracy rises85.92 ->99.35 ->100%
atT64/128/256; size64 d>32 pooled rises12.48 ->38.60 ->59.43%.
This sustained positive case does not replace the failed multi-seed gate.
Outputs: runs/streaming_carry_20261002_init2345/. Public entry:
evidence/streaming_carry_init2345/RESULTS.md, analysis.json and validation.json.
Source/evidence bindings: STREAMING_CARRY_PUBLICATION_MANIFEST.json.
CPU saved-result audit passed:35 source bindings,8 checkpoint parameter hashes,
96 BA and672 paired aggregates with reconstructed denominators,8 arm decisions,
312 paired contrasts and936 CSV rows. All four baseline final parameter hashes
and complete evaluation records exactly reproduce Phase II. No model outputs
were regenerated for publication; original run artifacts remain unchanged.
Fixed T is lossless; the full learned recurrence has no such guarantee.
No follow-up training, confirmation, rescue or monitoring is scheduled.
Commands from repository root:

    python new/streaming_carry/check.py
    python new/streaming_carry/run.py --preflight --out runs/NEW_STREAM_PREFLIGHT
    pwsh -File tools/launch_streaming_carry.ps1 -RunName NEW_STREAM_RUN -Preflight runs/NEW_STREAM_PREFLIGHT

Previous completed experiment (2026-10-02): Direct Spatial Carry development
screen. Frozen protocol: new/direct_spatial_carry/PROTOCOL.md. Only the W
identity term changes to W-0.5*D_M^dagger*L_M(W); F/Q, additive Z, W24/Z8,
5033 parameters, two communication phases and K8 training remain unchanged.
Rerun baseline and carry for initialization seeds2/3/4/5, 300 updates each,
with the exact Phase-II bank/schedule/evaluation maps (8 arms). These already
inspected seeds/maps make this a development screen, not confirmation.
Primary reach/hold thresholds are unchanged. Continue only if carry reaches
and holds in>=3/4, keeps seeds2/5, and exceeds the concurrent baseline count;
baseline2/5 must reproduce, otherwise report BASELINE_REPRODUCTION_DRIFT.
Report every distance band, size32/64, T64/128/256 and per-seed paired effects.
No normalization, routing, sweep, automatic confirmation or monitoring.

CPU implementation/gradient/decision checks passed. Read-only independent
review found a final-evaluation deadline gap, now closed; no remaining
correctness blocker was found within the review's stated scope. The final
matching-source CUDA preflight passed in6.375 seconds at
runs/direct_spatial_carry_20261002_dispatch_check/. Both arms completed3
updates; peak allocated memory baseline87.14MiB and carry88.11MiB. Worst
steady update0.406 seconds gives1289.28 seconds (21.49 minutes) under the
frozen runtime rule. Source, initialization, training-bank and evaluation-bank
identity checks passed. Earlier engineering preflight is retained separately.
Formal dispatch verified at13:53:26 +08:00, local RTX4060 Laptop GPU,
including its first completed training update. Run:
runs/direct_spatial_carry_20261002_init2345/. Private launch receipt:
runs/direct_spatial_carry_20261002_init2345_launch_20261002_135326/launch_receipt.json.
Completed at14:06:46 +08:00 in797.797 seconds (13.30 minutes), all8 arms
at300 updates. Status: DEVELOPMENT_NO_GO; baseline reach+hold2/4, carry0/4.
All four baseline final parameters and complete evaluation records exactly
reproduce Phase II. Carry primary pooled values by seed2/3/4/5 are
0.00/3.12/11.41/4.42%, below baseline92.97/17.27/44.76/95.68% respectively.
At size64/T256 carry d>32 paired correctness is0% in all four seeds.
No confirmation, rescue experiment or monitoring is scheduled.
CPU review at analyses/direct_spatial_carry_20261002_review/ verifies29 source
snapshots,8 checkpoints,96 BA and672 paired aggregates/denominators,8 arm
decisions,312 contrasts and936 CSV rows; no new training or inference.
Public entry: evidence/direct_spatial_carry_init2345/RESULTS.md and analysis.json.
Code/evidence bindings: DIRECT_CARRY_PUBLICATION_MANIFEST.json.
The carry alone is maximum-norm nonexpansive; full-cell stability, useful
information preservation and gradients across detach are not guaranteed.
Earlier evidence and claims remain unchanged. Commands from repository root:

    python new/direct_spatial_carry/check.py
    python new/direct_spatial_carry/run.py --preflight --out runs/NEW_CARRY_PREFLIGHT
    pwsh -File tools/launch_direct_carry.ps1 -RunName NEW_CARRY_RUN -Preflight runs/NEW_CARRY_PREFLIGHT
    python new/direct_spatial_carry/analyze.py --run runs/direct_spatial_carry_20261002_init2345 --out analyses/NEW_CARRY_REVIEW

Previous completed experiment (2026-10-02): short-BPTT Phase II, additive only,
K8/K16/K64, four new initialization seeds2/3/4/5. Fixed training bank10002 and
schedule20002 separate initialization variability from changing training data.
All12 arms use300 updates,64-step fresh trajectories, identical losses every8
steps and one optimizer update afterward. Frozen prospective protocol:
new/short_bptt_phase2/PROTOCOL.md. New evaluation banks40032/40064,32 maps each.
Primary size32/T64 band STRICT16<d<32, both per-map and pooled paired>=80%,
both original/flipped BA>=85%; at least3/4 K8 seeds for descriptive replication.
Hold, full-K64 qualification, old d>16, farther bands and size64 stay separate.
This selected narrow-band follow-up does not replace the old failed gate.

CPU K8/K64 equivalence and independent K16 gradient-reference checks passed;
forward values match exactly and weights remain unchanged within trajectories.
CUDA preflight passed in14.265 seconds at runs/short_bptt_phase2_20261002_preflight/.
All three K arms completed3 updates with matched identities and finite outputs.
The primary band covers32 held-out maps and2918 pixels. Worst steady-update
estimate0.367 seconds gives1765.44 seconds (29.42 minutes) under the frozen
12*300*c*1.20+180 rule, within30 minutes. Budget stays300 updates per arm.
Formal dispatch verified at11:19:17 +08:00 on the local RTX4060 Laptop GPU,
including the first completed training update. Run:
runs/short_bptt_phase2_20261002_init2345/. All12 arms completed300 updates at
11:37:34 +08:00 in1095.047 seconds (18.25 minutes); process exited and stderr
is empty. K8 narrow reach2/4 and reach+hold2/4, below the frozen>=3/4 threshold:
NOT_REPLICATED for both properties. K16 reaches3/4 but reach+hold only1/4;
all four K64 controls remain unqualified (BASELINE_UNQUALIFIED).

K8 seeds2/5 are positive cases: primary mean/pooled93.70/92.97% and96.45/95.68%
at size32/T64, respectively. Their pooled primary remains93.21/93.59% atT256,
and89.65/97.25% at size64/T256. Farther distance>32 at size64/T256 is only
36.78/14.83% pooled. Successful narrow propagation does not establish reliable
global propagation. K16's primary band fits inside its single-window radius.
All K8 clipping fractions are<=6%, including reach failures; low clipping is
not sufficient. Peak allocated memory K8/K16/K64:95.12/155.81/519.41MiB.

CPU result review: analyses/short_bptt_phase2_20261002_review/RESULTS.md,
analysis.json, curves.csv and provenance.json. Entry:
new/short_bptt_phase2/analyze.py. Verified18 source snapshots,12 checkpoints,
data/schedule/initial-parameter identities,144 BA aggregates,1008 paired
aggregates with independently reconstructed denominators and all12 decisions.
No extra training or inference. Conditional n=4 initialization replication,
not a claim across training banks. Original Phase-I gate stays unchanged.
Public review: evidence/short_bptt_phase2_init2345/RESULTS.md, analysis.json and
curves.csv; BPTT_PHASE2_PUBLICATION_MANIFEST.json binds code and public evidence.
Durable private launch receipt:
runs/short_bptt_phase2_20261002_init2345_launch_20261002_111917/launch_receipt.json.
All18 current sources match their preflight snapshots. An additional CPU gate
check confirms absolute replication and control qualification are independent,
and incomplete runs cannot pass. Cap30 minutes; emergency watchdog31 minutes.
No automatic retry or continuous monitoring.
Existing code, evidence and checkpoints remain unchanged. Commands from repo root:

    python new/short_bptt_phase2/check.py
    python new/short_bptt_phase2/run.py --preflight --out runs/NEW_PHASE2_PREFLIGHT
    pwsh -File tools/launch_bptt_phase2.ps1 -RunName NEW_PHASE2_RUN -Preflight runs/NEW_PHASE2_PREFLIGHT
    python new/short_bptt_phase2/analyze.py --run runs/short_bptt_phase2_20261002_init2345 --out analyses/NEW_PHASE2_REVIEW

The analysis script's interpretation is specific to the recorded screen;
use the runner's aggregate and gate report to interpret a new training run.

Previous completed screen (2026-10-02): matched short-BPTT training on the
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
