# Context for incremental scientific review

## Current: matched short-BPTT screen completed, all full controls unqualified

- Execution: `COMPLETE`; eight arms, 300 updates each, 730.797 seconds.
- Scientific status:
  `BASELINE_UNQUALIFIED_ALL_FOUR_PAIRS_WITH_ONE_SHORT_WINDOW_POSITIVE_CASE`.
- Design: existing additive/revision cells, K8/K64, paired seeds0/1; W24/Z8,
  5033 parameters and two masked communication phases per macro-step.
- Scope: 2D size32 training; fresh original/flipped evaluation on sizes32/64,
  16 fixed maps each, at T64/128/256. Model seed is the unit, n=2 per cell.
- Primary: size32/T64 paired correctness at graph distance >16, averaged
  per map. Full control requires BA>=85% and far paired>=80%; all four fail.
- Stop: the bounded screen is complete. No new training, retuning or monitor.

Every training trajectory starts fresh, runs64 steps, averages the same eight
losses at8,...64, and performs one optimizer update afterward. K8 accumulates
parameter gradients across eight backwards and detaches BOTH W/Z seven times
without resetting values. K64 retains the graph and backpropagates once.
Weights remain fixed inside each trajectory; clipping happens once at its end.
The encoder only receives gradients through windows connected to initialization.
Matched initialization, data, schedules, optimizer and update count were verified.

Additive K8 seed1 reaches BA98.03%, far paired80.87% and keeps far paired78.24%
atT256. Its short reach/hold predicates pass, but architecture qualification
requires both seeds and qualified full controls. The primary per-map80.87%
corresponds to pooled-pixel59.53%; size64/T64 far paired is28.40%. Far-distance
bins deteriorate. This is one positive example of compositional local behavior
with short gradients, not general long-horizon credit assignment or a new
TBPTT algorithm. It does not establish warm reopening or robust extrapolation.

K8 peak allocated CUDA memory is90.49 versus514.04 MiB for K64 (82.4% lower),
with similar training times. Peak allocation includes more than activations.
Clipping is much more frequent in additive K64, but causality is not isolated.
The runtime-only preflight selected300 updates before efficacy; the objective
differs from the old600-update reach/auxiliary recipe. Old qualification cannot
be imported. Do not interpret the runner's `decision: COMPLETE` as scientific
success, or treat failed full controls as evidence against truncation.

| Concept | Exact symbol | Source |
|---|---|---|
| Matched loss, detach and backward clock | `backward_trajectory` | `new/short_bptt/training.py` |
| Forward and gradient invariants | `main` | `new/short_bptt/check.py` |
| Paired metric, distance bins, training and gates | `paired`, `evaluate`, `train_one`, `aggregate` | `new/short_bptt/run.py` |
| Independent aggregate/hash verification and figure | `main` | `new/short_bptt/analyze.py` |
| Evidence copying and publication bindings | `main` | `tools/export_bptt_evidence.py` |

Start with [results](evidence/short_bptt_paired01/RESULTS.md),
[compact analysis](evidence/short_bptt_paired01/analysis.json),
[frozen protocol](new/short_bptt/PROTOCOL.md), and
[validation](evidence/short_bptt_paired01/validation.json).
[Publication hashes](BPTT_PUBLICATION_MANIFEST.json) bind the executed sources,
analysis and public evidence. All96 BA and336 paired aggregates and four gate
decisions were recomputed from stored per-map values. Raw arm JSONs and schedules
are byte-identical; local PID metadata is omitted from public status/manifests.
Checkpoints stay local; no extra inference was used for publication.

## Previous: source-switch audit completed, candidate persistence with W/Z interaction

- Execution: `COMPLETE`,9.609 seconds; no training, no Jacobian computation.
- Checkpoint: original revision seed0 from the paired screen, unchanged.
- Scope: two sizes32/64,16 shared maps each, K0/1/2/4/8/16/32/64/128 after
  switching source at T64. One trained model; sizes/maps are not new replicates.
- Scientific status:
  `OLD_ALIGNED_CANDIDATE_AND_W_Z_INTERACTION_WITHOUT_SINGLE_BLOCK_RECOVERY`.
- Validation:232 old metric comparisons replay exactly; reconstructed step
  error0; affine readout identity error<=7.153e-7. BA/BCE replay covers original
  and fresh-flip trajectories; W/Z RMS only the original trajectory. The frozen
  protocol overstates norm coverage; [post-execution notes](new/switch_audit/REVIEW_NOTES.md)
  explicitly correct the scope without editing the executed snapshot.

Candidate timing: from (W_K,Z_K), compute Wplus_K with the real F update,
then Q_K=Qnet(Wplus_K,Z_K,X), which drives Z_(K+1). O(Q_K) uses the existing
affine readout, including bias. O(Z_(K+1))=.5O(Z_K)+.5O(Q_K) is checked.
This is a projection of Q, not an interpretation of all latent coordinates.

Six full-rollout branches compare warm, W-only reset, Z-only reset, cold reset
of both, W-only mature donor and Z-only mature donor. Mature donors come from
new-input T64 and contain extra computation. Candidate-only factorials cross
post-F warm/new-aged W, pre-update warm/new-aged Z, and old/new X. No extra F
is run inside those queries. Thus direct X contrasts omit other input paths.

At size32/K128 the warm candidate/output is0% correct on the changed component;
cold output is100%. All four single-block interventions reach37.5%, which is
all-negative prediction across all changed pixels, not source-based recovery.
At size64 the corresponding warm output is1.64%, cold99.49%, single-block
37.17%-45.40%. The candidate function responds to both donor blocks; continuous
signed-margin and accuracy contrasts, including interactions, are saved per map.

Do not claim stale W alone, generally necessary simultaneous reset, an on-manifold
causal mechanism, successful conditional invalidation, or stable useful computation
solely from W motion. This audit leaves the original two-seed joint failure
and seed1 qualification failure unchanged. Short BPTT was still untested at
the time of this audit; the subsequent screen is described above.

| Concept | Exact symbol | Source |
|---|---|---|
| Candidate clock and affine identity | `probe`, `check` | `new/switch_audit/audit.py` |
| Natural replay and intervention branches | `natural`, `audit_size` | `new/switch_audit/audit.py` |
| Per-map metrics and source-distance bins | `describe` | `new/switch_audit/audit.py` |
| Contrasts, label-bias derivation and figure | `positive_fraction`, `main` | `new/switch_audit/analyze.py` |
| Lossless raw publication and aggregate verification | `main` | `tools/export_switch_evidence.py` |

Start with [audit results](evidence/switch_audit_seed0/RESULTS.md),
[compact analysis](evidence/switch_audit_seed0/analysis.json),
[protocol](new/switch_audit/PROTOCOL.md), and
[validation](evidence/switch_audit_seed0/validation.json).
[Publication hashes](SWITCH_PUBLICATION_MANIFEST.json) bind code and evidence.
Raw JSON is losslessly compacted, not byte-identical; decoded equality is verified.
No new inference or training was performed for publication. No further run scheduled.

## Previous: Workspace + Revision paired screen completed, joint gate failed

- Execution status: `COMPLETE`; four arms, seeds0/1, 600 updates each, 1112.28s.
- Scientific status: `NO_JOINT_SCREEN_PASS`; mean primary hold effect -9.02 pp.
- Scope: 2D seeded-component task; size32 training; size32 primary evaluation,
  size64 secondary; 16 shared held-out maps per size. Independent unit: model
  seed, n=2, descriptive only. No 3D result or significance claim.
- Provenance: eight executed source files match snapshots; per-seed initial
  parameters, data and schedules match between arms; saved checkpoint hashes
  verified. Raw records and schedules are copied without alteration.
- Stop: completed bounded screen. No rescue sweep, new seed or monitoring.

Input X is `[batch,3,height,width]`: supplied binary medium and two signed
source channels. Persistent state is W24/Z8. Encoder initializes W from X,
Z starts at zero, readout uses only Z. Both arms have 5033 trainable parameters.

```math
\begin{aligned}
W' &= W+0.1F(W,Z,L_MW,L_MZ,X),\\
Q &= Q_\theta(W',Z,L_MW',L_MZ,X),\\
Z'_{\mathrm{add}} &= Z+0.5Q,\\
Z'_{\mathrm{rev}} &= Z+0.5(Q-Z).
\end{aligned}
```

F and Q have pointwise hidden tanh and zero-initialized last layers. Each
macro-step has two masked communication phases in both arms. Alpha is fixed;
there is no learned gate or output clamp. Fixed finite Q weights give a bounded
candidate due to hidden tanh, hence a bound on revision Z, but not on W or task
correctness. This screen isolates the Z update within the shared W/Z split.

Paired training supervises reach plus alternating hold/switch/repair branches.
Reach gradients span at most 64 steps; auxiliary states are detached, optionally
advanced without gradients, then supervised for 16/32 more steps. No hidden-state
ground truth or checkpoint from the old recipes is used. Runtime-only preflight
reduced the initial 800-update proposal to 600 before efficacy training.

Primary hold is min BA at T128/T192/T256, not a map minimum or an asymptotic
statement. Revision seed0 gives +5.82 pp hold, seed1 -23.86 pp. Seed0 reaches
99.19% BA64 and 100% hold; seed1 stays at 50%. Seed0's warm source-change
accuracy is 0% at K64/K128 versus 100% from cold initialization on size32.
Cold fresh-pair success cannot replace warm revision. This is history dependence
without an identified attractor mechanism. Z-only repair retains W; joint W/Z
repair is worse. W continues to grow despite stable output. Size64 is secondary
and cannot rescue the failed joint gate. Across seeds, data/schedules also change;
do not attribute failure specifically to initialization.

| Concept | Exact symbol | Source |
|---|---|---|
| Matched cell and initialization | `RevisionCell.initial`, `step`, `logits`, `make_cell` | `new/workspace_revision/revision_cells.py` |
| Frozen geometry operator | `masked_laplacian` | `new/masked_medium/masked_cells.py` |
| Shared schedule and branch objective | `schedule`, `branch_loss`, `train_one` | `new/workspace_revision/run_revision.py` |
| Fresh/warm/cold/damage evaluation and gate | `evaluate`, `summary` | `new/workspace_revision/run_revision.py` |
| CPU implementation check | `run_checks` | `new/workspace_revision/check.py` |
| CPU evidence verification/publication | `main` | `tools/export_revision_evidence.py` |

Start with [results](RESULTS.md),
[interpretation](evidence/workspace_revision_paired01/INTERPRETATION.md),
[analysis](evidence/workspace_revision_paired01/analysis.json), then
[protocol](new/workspace_revision/PROTOCOL.md) and
[validation](evidence/workspace_revision_paired01/validation.json).
The [publication manifest](REVISION_PUBLICATION_MANIFEST.json) binds scientific
evidence and code; checkpoints and machine receipts stay local. Older training
recipes differ in objective and per-step communication, so are contextual only.

## Previous: generic dynamics audit completed, with diagnostic limitations

- Execution: `COMPLETE`,112.859 seconds; training:false; four existing seed0
  generic State/Momentum checkpoints, two media, sizes32/64, eight anchors.
- Provenance: all40 historical comparisons replay exactly, sources and
  checkpoint hashes match; no earlier frozen code or evidence changed.
- Spectral scope: windows1/8/16, raw Euclidean H/V units, full state and open
  endpoint compression. Fixed maps0..3; performance curves retain all16 maps.
- Precision:562/1536 product estimates and148/256 open K16 estimates satisfy
  the frozen residual/spread criterion. Others are not certified top norms.
- Scientific status: `NO_OBSERVED_MAXIMUM_GAIN_ZERO_CROSSING_WITH_DIAGNOSTIC_LIMITS`.
- Stop: no new training, inference rerun, architecture or monitor scheduled.

All256 open K16 log-gain estimates are positive, and usually decline while
late accuracy deteriorates. At size32, Masked Momentum BA goes96.45% to77.68%
from T64 toT256 while g16 changes0.27742 to0.17582; Masked State BA goes90.43%
to11.38% while g16 changes0.14666 to0.10359. This does not identify a
near-zero-to-positive transition or establish asymptotic stability/instability.

Generic Masked Momentum shows continued updates, growing H norms and wrong
confidence. Its source-flip response grows while paired correctness drops.
Mean per-map component-constant H energy is only0.20% at32/T256; the earlier
explicit-inertial component-mean interpretation cannot simply be reused.
Worst-direction finite perturbations often leave the linear regime: at1e-4
relative RMS, median linearization error46.44%, versus1.19% for random directions.
No causal velocity isolation, medium-spectrum mechanism or stable recurrence
design is established. The original primary Momentum advantage is unchanged.

| Concept | Exact symbol | Source |
|---|---|---|
| Exact tangent and adjoint | `LinearStep.force`, `force_adj`, `apply`, `adj`, `input_apply` | `new/dynamics_audit/operators.py` |
| Ordered Jacobian product, norm estimate | `product`, `estimate` | `new/dynamics_audit/operators.py` |
| Drift and continuously driven source tangent | `trajectories`, `scaled_update`, `summarize_tangent` | `new/dynamics_audit/audit.py` |
| Analytic shear and nonlinear checks | `free_baseline`, `perturbation`, `diagnose` | `new/dynamics_audit/audit.py` |
| Autograd, adjoint, finite-difference and exact SVD gate | `main` | `new/dynamics_audit/check.py` |
| CPU-only derived analysis | `main` | `new/dynamics_audit/analyze.py` |

For Momentum, J=[[I+D,B],[D,B]], with D the learned force derivative and B
channelwise beta. The adjoint is (u+D*(u+w),B*(u+w)). Source tangents include
encoder initialization and repeated direct X injection. Geometry is fixed;
mask-channel derivatives are excluded. Open compression applies only at
product endpoints, so whole-grid intermediate paths may still cross walls.
The finite binary source flip is distinct from its infinitesimal tangent.

Start with [interpretation](evidence/dynamics_audit_seed0/INTERPRETATION.md),
[table](evidence/dynamics_audit_seed0/RESULTS.md),
[compact analysis](evidence/dynamics_audit_seed0/analysis.json),
[protocol](new/dynamics_audit/PROTOCOL.md), then
[validation](evidence/dynamics_audit_seed0/validation.json) and
[publication hashes](DYNAMICS_PUBLICATION_MANIFEST.json). Per-map curve/dynamics
JSON under `evidence/dynamics_audit_seed0/raw/` is secondary. No checkpoint or
local machine receipt is required to inspect the published scientific evidence.

## Previous final 2x2: scientific endpoints saved, auxiliary tail recovered

- Scientific status: `MOMENTUM_JOINT_5PP_ADVANTAGE`, descriptive, one training seed.
- Original execution: 800 updates and all 15 scientific endpoints saved; no
  final completion marker, process no longer present, cause unknown.
- Auxiliary status: `PLANNED_AUXILIARIES_RECOVERED`; missing size128 timings and
  initial-state gradient probes added from the existing checkpoint, no retraining.
- Verification: all 15 four-way comparisons recomputed from raw; two checkpoint
  endpoints replay exactly; original fields and executed sources unchanged.
- Stop: final authorized arm; no further run, seed, architecture or monitor scheduled.

At 32/T64, State goes from whole-grid 85.17% BA / 49.31% paired to masked
90.43% / 56.37%. Momentum goes from 84.44% / 42.75% to 96.45% / 77.25%.
Same-medium Momentum advantages are 6.02pp BA and 20.88pp paired correctness;
descriptive interactions are +6.75pp and +27.43pp. State's masked BCE is worse
(0.9871 versus unmasked 0.2579). Masked State collapses at 32/T256 to 11.38%
BA, paired 9.38%, BCE 63.4853; Masked Momentum also declines, to 77.68% BA.
Neither reaches sustained 95%. These are finite-horizon observations, not proof
of convergence, asymptotic divergence or scale-robust cellular computation.

This compares established recipes: State H32/hidden48/4993 parameters versus
Momentum H16+V16/hidden88/4689. Equal persistent scalar capacity does not isolate
velocity: content/readout/program widths differ. Within each recipe masking is
the intervention, with original initialization/RNG preserved. All four logged
rollout/damage schedules match; only one training seed remains the replicate.
The input mask supplies task-specific geometry and changes degree and spectrum.

| Final concept | Exact source symbol | File |
|---|---|---|
| Generic masked state update | `MaskedState.step`, `make_cell` | `new/masked_state/state_cells.py` |
| Frozen references and final contrasts | `verify`, `summarize`, `main` | `new/masked_state/run_state.py` |
| Missing diagnostic recovery, no training | `main` | `new/masked_state/recover_tail.py` |
| Pairing, isolation, backward checks | `Tests` | `new/masked_state/test_state.py` |
| Scientific-field and publication audit | `main` | `tools/export_state_evidence.py` |

Start with [current results](RESULTS.md), [comparison](evidence/masked_state_seed0/comparison.json)
and [protocol](new/masked_state/PROTOCOL.md). Then inspect
[recovery](evidence/masked_state_seed0/recovery.json),
[validation](evidence/masked_state_seed0/validation.json) and
[hash bindings](STATE_PUBLICATION_MANIFEST.json). Raw original/augmented records
are secondary. The publication does not claim that the original process finished
normally. Missing-finalization cause is not inferred from the large gradients.
The earlier audit below applies to Masked Inertial RD, not either generic model.

## Previous same-medium control: completed

Status: valid exploratory single-seed comparison; seed0;800 updates; all15
size/horizon endpoints. `new/masked_momentum/momentum_cells.py::MaskedMomentum.step`
uses F(H,L_m H,X) with generic momentum and no explicit diffusion force. The
original momentum factory, initialization/RNG and frozen trainer are reused.
Hidden width88/parameters4689 versus128/4737 for explicit masked inertial RD;
generic force starts at zero, explicit diffusion is nonzero at initialization.

At32/T64, Momentum BA96.45%, paired77.25%, BCE0.0512 versus the explicit
candidate's88.79%,37.22%,0.2503. Outcome `MOMENTUM_JOINT_5PP_ADVANTAGE`:
the explicit candidate loses7.66pp BA and40.02pp paired correctness. At32/T256,
Momentum BA falls77.68%, below candidate91.55%, while paired correctness is
42.78% versus41.92%. Neither has sustained95% BA. Do not hide this horizon
tradeoff or select the late horizon to replace the frozen primary endpoint.

Read [comparison](evidence/masked_momentum_seed0/comparison.json),
[protocol](new/masked_momentum/PROTOCOL.md), and
[publication manifest](MOMENTUM_AUDIT_PUBLICATION_MANIFEST.json). Raw JSON contains
damage, revision and timing metrics. Different repair eligibility cohorts prevent
direct conditional-repair superiority claims. No cross-run speedup or multi-seed
claim. All historical sources and evidence are unchanged.

## Completed masked-inertial trajectory audit and previous medium intervention

Interpretation update: classification improvement with extended rollout is
not evidence of hidden-state convergence. At32, T64/128/256 H RMS is
17.19/50.88/126.95, V RMS0.483/0.575/0.617 and BCE0.250/0.657/1.452.
Global RMS includes walls, while BCE is traversable-only. Directional alignment,
component-mean drift and confidence amplification are hypotheses to audit.
Conditional damage recovery is a decision metric, not proof of an attractor.
The new generic masked momentum protocol is `new/masked_momentum/PROTOCOL.md`;
it retains known hidden-width and initial-transport differences from the candidate.
The [completed trajectory audit](evidence/trajectory_audit_seed0/INTERPRETATION.md)
replays all15 historical checkpoints exactly. At32/T256, H RMS on open pixels
is120.61 and component-constant energy fraction97.04%; wrong-margin median
is -18.42. Independent calibration reduces BCE1.4519 to0.2492 without changing
BA. At128/T256 calibrated BCE is0.6903 and BA stays50%. These finite-horizon
observations support amplification/mean drift, not asymptotic convergence.

Status: completed exploratory operator intervention; dimension: 2D; seed: 0;
updates: 800 per arm; held-out maps: 16 per size. The original sources and
controls are frozen. `new/masked_medium/masked_cells.py::masked_laplacian`
computes sum over four neighbors of m_i*m_j*(H_i-H_j), using only input mask.
`MaskedCell.step` wraps the original structured cells with that operator.
`run_masked.py` reuses the original training/evaluation functions and verifies
initialization/RNG, source hashes and logged training schedule pairing.

Primary 32x32/T64: masked inertial RD gains +18.85pp BA and +19.42pp paired
correctness, satisfying the prespecified descriptive joint +5pp criterion.
Masked RD loses 8.06pp and 20.68pp. Candidate BA at T256 is 91.55%, 75.69%,
50.00% for sizes 32, 64, 128. No sustained aggregate 95% endpoint is reached.
Masking changes connectivity, degree and spectrum; the specific causal
mechanism is not isolated. The new masked momentum arm above is a separate
follow-up. Multiple-seed, 3D, controlled-speedup and general architecture-pass
claims remain unestablished.

Read [masked results](evidence/masked_medium_seed0/RESULTS.md),
[paired comparisons](evidence/masked_medium_seed0/paired_comparison.json),
[protocol](new/masked_medium/PROTOCOL.md) and
[manifest](MASKED_PUBLICATION_MANIFEST.json). Raw per-arm records are exact
copies; host/PID/receipts/checkpoints remain local. Earlier claims below remain
bound to their original experiments.

## Previous 2D inertial wind-tunnel screen (seed 0)

This separate experiment asks whether explicit five-point reaction–transport
with inertia learns seeded-region identity better than generic momentum NCA,
with a state-capacity control. Four arms completed 800 AdamW updates each on
512 fixed 32x32 training maps. Evaluation uses 16 held-out maps per size
(32/64/128) at horizons 16/32/64/128/256. This is one exploratory seed.

At 32x32 and T=64, `inertial_rd` reached 69.94% balanced accuracy, compared
with 84.44% for `momentum_nca` and 85.17% for `nca_state_matched`. At T=256,
`inertial_rd` was at 50% balanced accuracy with zero paired-source correctness
on sizes 64 and 128. Every arm/size 95%-sustained threshold is null. The
candidate has zero repair-eligible examples at every size, so conditional
repair is unevaluable; this is not a repair failure estimate.

The result is negative for this one-seed recipe. It does not refute an
architecture family or support multi-seed superiority, a 3D claim, a repair
claim, or a speedup claim. The result also does not establish that the task
and training recipe have a reliable positive control.

Read the [compact summary](evidence/inertial_seed0/summary.json),
[per-arm results](evidence/inertial_seed0/RESULTS.md), [configuration](evidence/inertial_seed0/config.json),
[linear checks](evidence/inertial_seed0/linear_checks.json),
[validation](evidence/inertial_seed0/validation.json), and
[publication manifest](INERTIAL_PUBLICATION_MANIFEST.json). The detailed
[candidate](evidence/inertial_seed0/arms/inertial_rd_seed0.json),
[generic momentum control](evidence/inertial_seed0/arms/momentum_nca_seed0.json),
and [state-matched control](evidence/inertial_seed0/arms/nca_state_matched_seed0.json)
retain their per-horizon records. The source protocol is
[WIND_TUNNEL.md](new/nca_inertial_wind_tunnel/WIND_TUNNEL.md), with analytic
scope in [THEORY.md](new/nca_inertial_wind_tunnel/THEORY.md) and local evidence
notes in [LOCAL_INTEGRATION.md](new/nca_inertial_wind_tunnel/LOCAL_INTEGRATION.md).

| Concept | Exact symbol | Source |
|---|---|---|
| Cell family, five-point operator, arm updates and coefficients | `ARMS`, `laplacian`, `Cell.step`, `Cell.coefficients`, `Cell.rollout`, `make_cell` | `new/nca_inertial_wind_tunnel/cells.py` |
| Region maps, labels, damage and task metrics | `components`, `sample`, `bank`, `damage`, `metrics` | `new/nca_inertial_wind_tunnel/tasks.py` |
| Training, evaluation, thresholds and timing | `sustained_threshold`, `evaluation`, `gradient_probe`, `benchmark`, `train_one`, `main` | `new/nca_inertial_wind_tunnel/run_wind_tunnel.py` |
| Pure-transport analytic checks | `roots`, `radius`, `block_matrix`, `run` | `new/nca_inertial_wind_tunnel/math_checks.py` |

## Current A0 status and minimum route

Protocol rt_a0_2d3d_v2_1 completed 24/24 conditional trials (40 maximum; 16 3D RT
trials skipped). In 2D, attention is perfect for all four seeds and all reported
conditions; all four RT arms are NOT_QUALIFIED_FIT. In 3D, attention seed 1729
fails and no RT treatment estimate exists. Numerical/oracle checks pass but
do not establish learned-model qualification. B/C and width sweeps remain unrun.

Read [RESULTS.md](RESULTS.md), [A0_PROTOCOL.md](A0_PROTOCOL.md), then the
[small A0 summary](evidence/a0_v2_1/summary.json). Use the
[per-seed table](evidence/a0_v2_1/RESULTS.md) before opening
[aggregate.json](evidence/a0_v2_1/aggregate.json) for histories, axis metrics
and diagnostics. [checks.json](evidence/a0_v2_1/checks.json) and
[A0_PUBLICATION_MANIFEST.json](A0_PUBLICATION_MANIFEST.json) establish validation
and provenance. Independent units are seeds 1729, 2718, 31415, 57721.

Every A0 arm emits q=c*v, with learned sigmoid group weights c initialized at 0.5.
Raw uses T(q); normalized uses T(q)/(T(c)+1e-6). Both perform the same packed
grouped solve. Initial common parameters and medium scale match; only the
receiver division changes within a medium/seed pair. Constant conductance is 1;
learned conductance is 2*sigmoid(local symmetric features), initialized at 1.
Attention projects query/key and uses q directly as values, with no separate
value/output projection. C=64, r=32, G=4, K=8; two phases of four updates. A0 raw is
not identical to v1 raw. Fixed axis order and target-only readout are retained.

| A0 concept | Exact symbol | Source |
|---|---|---|
| Standalone entry | package/module dispatch | `a0.py`, `run.py` |
| Forward and diagnostics | `A0Model.forward` | `a0_models.py` |
| Same-value attention | `SharedValueAttention.forward` | `a0_models.py` |
| Grouped numerator/confidence solve | `transport_pair` | `a0_models.py` |
| Config, trial and conditional gate | `config`, `trial`, `decisions`, `main` | `a0_runner.py` |
| Paired estimates | `paired_effects` | `a0_runner.py` |
| Software and oracle checks | `run_checks`, `oracle_diagnostics` | `a0_checks.py` |
| Public snapshot | `summarize`, `main` | `tools/export_a0_evidence.py` |

The 2D result is negative under a reliable fitting control. It neither proves
diffusion-family impossibility nor causally separates emitter/readout failures.
Sampled normalized-model confidence remains dense on background; this is not
proof that a different confidence/emission mechanism cannot learn. Confidence-
mass contribution is not a total-information attribution through recurrence.
Changes between v1 and A0 cannot isolate why attention fitting improved.

The sections below describe historical v1 and its unchanged code/evidence.

## Historical v1 formal status

- Scientific status: `INCONCLUSIVE_POSITIVE_CONTROL` in both 2D and 3D.
- Endpoint reached: completed Gate A, 20 model/seed/dimension runs.
- RT fit status: not qualified even at training-shape distance 16.
- Software status: numerical and end-to-end smoke checks passed.
- Not run: Gate B, Gate C, r=4/8/16/32/64 width scan, real vision, foundation
  pretraining, directed transport, DEQ, generation or video.
- Independent replicate: training seed (1729, 2718). Test examples, voxels and
  timing repetitions do not create additional training replicates.

## Hypothesis and variants

Main hypothesis: retain local H, emit Q with r<C channels, exchange Q through a
content-dependent symmetric nearest-neighbor medium, then update H locally.
Width C=64, r=32, four groups and eight updates form a reduced-width screen of
the earlier illustrative C=128 design. Residual scale starts at 0.1.

Controls: CNN has eight unshared local cells; NCA has two cells repeated four
times each; attention substitutes global SDPA messages at the same r; constant
transport fixes internal conductance to one. Learned transport predicts local
conductance from E and normalized H at each phase start. All get the same
normalized coordinate inputs and target-only readout. Parameter counts differ.

Gate A has one random source bit. Gate B would add four sources on separated
straight regions and an oracle-edge diagnostic. Gate C would compare equal-
width Q=H arms differing only in residual base H versus T(H), using joint remote
source/local-detail labels. These B/C implementations are smoke-tested, not
scientifically evaluated.

## Exact source map

| Concept | Symbol | Source |
|---|---|---|
| Synthetic tasks, independent source/detail labels | `make_batch` | `data.py` |
| Symmetric endpoint edge prediction | `_EdgeBuilder.forward` | `models.py` |
| Local reaction and write gate | `_Reaction.forward` | `models.py` |
| SDPA message control | `_GlobalMessage.forward` | `models.py` |
| Phase sharing, persistent state and target readout | `GateModel.forward` | `models.py` |
| PCR factor reuse | `factorize`, `solve_factors` | `transport.py` |
| Implicit RHS/edge adjoint | `_ImplicitSolve.backward` | `transport.py` |
| Palindromic 2D/3D splitting | `prepare_transport`, `apply_transport` | `transport.py` |
| Matched training schedule, evaluations | `trial`, `evaluate` | `runner.py` |
| Frozen decisions | `verdicts` | `runner.py` |
| Numerical checks | `run_checks` | `test_transport.py` |

## Evidence route and boundaries

Read `RESULTS.md` and the small `evidence/qualification_v1_1/summary.json` first.
Use `config.json` to establish the protocol; use `aggregate.json` only for
per-axis evaluations, losses, gradients, memory and individual timing records.
`PUBLICATION_MANIFEST.json` binds evidence to hashes of the code actually run.

Do not interpret chance-level RT scores as distance extrapolation failure after
successful fitting: it did not fit the training evaluation. Attention's one
successful seed demonstrates task solvability, but its failed second seed
violates the frozen positive-control requirement. Neither successful solver
checks nor a formal global receptive field establish useful learned routing.

Timing scope: batch one, fixed large thin grids, float32 portable PyTorch
implementation, synchronized wall time including Python, two seeds. The
128-update NCA timing is a path-length cost reference; accuracy at 128 recurrent
updates was not qualified. The evidence does not establish the speed ceiling
of a fused transport kernel or a tuned attention backbone.

## Useful reviewer questions

1. Which missing training qualification would most cheaply distinguish sparse
   message dilution from generic optimization failure?
2. Is any claimed communication advantage confounded by target markers,
   coordinates, local evidence paths or differing parameter counts?
3. After a reliable fitting control, what intervention would isolate medium
   learning from message emission and reception?

Those were v1 review questions. A0 above supplies the registered paired
normalization experiment, but does not causally isolate the remaining failure.
