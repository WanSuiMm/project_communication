# Context for incremental scientific review

## Latest completed audit: continuation interface

- Run: `continuation_interface_v1`, `COMPLETE`, 50 transfer cells, 439 seconds;
  no model training or optimizer steps. Common native Full qualification is
  true for S1/S2, and the gauge control passes.
- Raw S1↔S2 off-diagonal transfer fails; raw S1→F1 passes. With the frozen
  100-coefficient linear alignment, S2→S1 and S2→F2 pass, while S1→S2 fails.
  This is a selected-checkpoint interface result, not proof of a common
  algorithm. Restricted off-diagonal alignment qualification is **0/20**;
  matched short-observable flags are **0/6**. Transfer proxies and these
  qualification tests remain separate.
- Start with [results](evidence/continuation_interface_20261004/RESULTS.md),
  [summary](evidence/continuation_interface_20261004/summary.json),
  [validation](evidence/continuation_interface_20261004/validation.json),
  [configuration](evidence/continuation_interface_20261004/config.json), and
  [reproduction notes](evidence/continuation_interface_20261004/REPRODUCTION.md).
  Then read [selection](evidence/continuation_interface_20261004/selection.json),
  [alignment validation](evidence/continuation_interface_20261004/alignment/validation.json),
  [phase-0 comparisons](evidence/continuation_interface_20261004/phase0/comparisons.json),
  [cross decomposition](evidence/continuation_interface_20261004/cross/decomposition.json),
  and [gradient metadata](evidence/continuation_interface_20261004/gradients/metadata.json).
  Full records and larger arrays are secondary. Verify from the repository root
  with `python -X utf8 -B tools/export_continuation_interface.py --verify-only`.
- Frozen protocol: [PROTOCOL.md](new/continuation_interface/PROTOCOL.md).

| Concept | Exact symbols | Source |
|---|---|---|
| Run orchestration and saved evidence | `run`, `source_hashes` | [run.py](new/continuation_interface/run.py) |
| Restricted linear state alignment and gauge control | `fit_alignment`, `apply_alignment`, `exact_gauge_clone` | [alignment.py](new/continuation_interface/alignment.py) |
| Handoff summaries and matrix decomposition | `summarize_handoff`, `decompose` | [metrics.py](new/continuation_interface/metrics.py) |
| K8/full64 gradient diagnostics | `audit_gradients` | [gradients.py](new/continuation_interface/gradients.py) |
| CPU/GPU qualification checks | `check` | [check.py](new/continuation_interface/check.py) |

## Previous completed suite: serial training-state and second-task qualification

- Protocol: `trajectory_qualification_v1`; formal execution `COMPLETE`,
  48/48 arms (stage 1: 8/8, stage 2: 32/32, stage 3: 8/8), 6,610.797 seconds.
- Stage 1 is one conditional update-3 donor-state family on the selected H/S
  suffixes. The native controls exactly reproduce their saved endpoints; the
  frozen selector chose `shadow_joint`. This selects the whole donor package,
  not a parameter-only or moment-only mechanism.
- Stage 2 tests that selected recipe on 16 fresh initialization/schedule pairs.
  Baseline and treatment each pass the unchanged Full gate on 1/16 pairs;
  there are 0 treatment-only and 0 baseline-only passes (1 both-pass and 15
  both-fail ties; paired delta 0, exact two-sided p=1). Decision:
  `NO_RELIABILITY_QUALIFICATION`.
- Stage 3 is the independent multi-source geodesic-distance task using the
  unchanged StreamingCell and 300-update recipe. No arm met the size-32 T64
  short-task gate (mean-map MAE <=2 cells), so the decision is
  `SECOND_TASK_TRAINING_UNQUALIFIED`; 0/8 arms pass the full distance proxy.
  This second-task proxy is separate from the original paired-source-flip Full
  gate.
- Across the suite, the original cell and K8 credit window stay fixed. Stage 1
  is conditional on one selected update-3 donor package and two suffixes;
  stage 2 uses 16 independent fresh pairs; stage 3 tests a distinct task. These
  records do not establish a suffix-independent component mechanism, recipe
  reliability, geometric basin volume, or the original task gate on stage 3.
- Start with [suite results](evidence/trajectory_qualification_20261004/RESULTS.md),
  [suite summary](evidence/trajectory_qualification_20261004/summary.json),
  and [validation](evidence/trajectory_qualification_20261004/validation.json).
  Then read the [stage 1 results](evidence/trajectory_qualification_20261004/stage1_state_cross/RESULTS.md)
  and [summary](evidence/trajectory_qualification_20261004/stage1_state_cross/summary.json),
  the [stage 2 results](evidence/trajectory_qualification_20261004/stage2_fresh_recipe/RESULTS.md)
  and [summary](evidence/trajectory_qualification_20261004/stage2_fresh_recipe/summary.json),
  and the [stage 3 results](evidence/trajectory_qualification_20261004/stage3_distance/RESULTS.md)
  and [summary](evidence/trajectory_qualification_20261004/stage3_distance/summary.json).
  Raw arm records, curves, and traces are secondary; checkpoints remain local. The frozen
  [suite protocol](new/trajectory_qualification/PROTOCOL.md),
  [distance protocol](new/trajectory_qualification/DISTANCE_PROTOCOL.md), and
  [reproduction notes](evidence/trajectory_qualification_20261004/REPRODUCTION.md)
  define the procedure and public artifact limits.

| Concept | Exact symbols | Source |
|---|---|---|
| Suite orchestration and provenance | `run`, `source_hashes`, `reference_hashes` | [run.py](new/trajectory_qualification/run.py) |
| Stage 1 cross and recipe selector | `run_stage`, `select_recipe`, `specs` | [state_cross.py](new/trajectory_qualification/state_cross.py) |
| Stage 2 paired outcomes and reliability test | `run_stage`, `_paired_summary`, `wilson_interval`, `exact_two_sided_binomial_p` | [fresh_recipe.py](new/trajectory_qualification/fresh_recipe.py) |
| Stage 3 distance gate and evaluation | `run_stage`, `_gate`, `evaluate_distance`, `backward_distance` | [distance_task.py](new/trajectory_qualification/distance_task.py) |
| Shared original update path and replay checks | `train_updates`, `load_state`, `compare_replay` | [common.py](new/trajectory_qualification/common.py) |

## Previous result: conditional seed4 bootstrap-path screen

- Run: `seed4_bootstrap_path_v1`, COMPLETE, 23 arms × 300 updates, 3,068.407 s.
- Inputs: init4, unchanged historical StreamingCell/AdamW/K8; H20002 and S20012/S20022; prefix lengths1/3/8. Paired source-flip evaluation uses32 maps each at spatial sizes32/64, seeds50032/50064, reused from the earlier screen.
- Publication is data and code, without a new mechanism analysis. All arms, failures, denominators, training curves, frontier CSVs and Boolean NPZ traces are retained. First-step cases: init2/3/4/5 on H plus init4 on both S first batches; gradient and actual parameter-delta spectra are separate fields.
- Read [table](evidence/bootstrap_path_20261004/RESULTS.md), [aggregate](evidence/bootstrap_path_20261004/summary.json), [first-step data](evidence/bootstrap_path_20261004/first_step_summary.json), then [protocol](new/bootstrap_path/PROTOCOL.md) and [validation](evidence/bootstrap_path_20261004/validation.json). `traces/` arrays are secondary, not the starting point. Full local reference requirements are in [reproduction](evidence/bootstrap_path_20261004/REPRODUCTION.md).

| Concept | Exact symbols | Source |
|---|---|---|
| Provenance and training loop | `source_hashes`, `actual_first_steps`, `train_one`, `aggregate` | [run.py](new/bootstrap_path/run.py) |
| Frozen schedules and intervention construction | `suite`, `check_suite` | [schedules.py](new/bootstrap_path/schedules.py) |
| Actual first-step gradient and AdamW audit | `audit_first_step` | [first_step.py](new/bootstrap_path/first_step.py) |
| Full phenotype evaluation | `evaluate`, `predicate` | [phenotype.py](new/seed4_followup/phenotype.py) |

## Previous diagnostic: update-zero geometry and continuation formation

- Scope: two zero-training diagnostics in 2D. Initial geometry uses the four
  exact historical update-zero checkpoints (seeds 2/3/4/5) and fixed primary
  16 plus diagnostic consistency 8 maps per size at 32/64. All banks were already
  inspected; the second bank checks consistency and is not fresh confirmation.
  Parameter hashes match historical records. Tangent/autograd checks pass
  (maximum absolute error 4.77e-7); no optimizer update occurs. Runtime:
  30.31 CPU seconds.
- Initial geometry: seed4 has the most balanced raw lane RMS and largest
  measured available relative source-flip separation at K8 endpoint64 across
  all four banks. Centered lane balance and cross-lane correlation do not
  uniformly favor seed4; seed3 has the highest paired kernel-label alignment,
  while seed5 has the highest initial Q-feature effective rank. Other spectral
  rankings vary by bank/window. Available separation is normalized by hidden
  state norms and is not accuracy or an invariant information measure. These
  are random task features rather than the state Jacobian or a learned
  continuation quotient. On the first historical
  training batch, encoder, F, Q-in and readout.weight task gradients are exactly
  zero. Balanced-loss bias gradients are about 2e-7 from FP32 cancellation;
  Q-out weight is the only appreciable group-level gradient and is rank one up
  to FP32 roundoff. This is not an AdamW update or a success criterion for
  initialization.
- Formation: saved-array CPU analysis of one selected seed4 trajectory, no new
  forward inference, took 8.86 seconds over 44 arrays. Predictors use t<=64;
  the long screen uses T128/T256. The early screen
  (survival32->64>=.95 and G32->64>=.20) passes at updates 140/145/190/200;
  the frozen long screen passes only at200. Same-update classification is
  TP1/FP3/TN17/FN0; prediction at the next saved update (+5) is
  TP0/FP3/TN16/FN1.
  Every continuous early observable ranks200 highest, but ordinary strict T64
  output coverage also ranks200 highest (.7063). No held-out prospective
  training outcome or incremental predictive value is established. The margin
  reserve at175 is not a certificate for behavior after the optimizer changes
  at180.
- Claim boundary: no latent `pi`, abstract transition, quotient defect, or
  independent consumer contract was measured. The specific seed4 closure
  mechanism remains unsupported by these diagnostics; the conditional
  execution theorem is not refuted. Results are descriptive for existing
  seeds, checkpoints and previously seen maps, not a fresh confirmation or a
  causal initialization explanation.

| Concept | Exact symbols | Source |
|---|---|---|
| Lane balance and initial task features | `lane_stats`, `feature`, `local_support`, `relational` | `new/initial_geometry/audit.py` |
| Paired tangent construction and numerical check | `collect`, `tangent_sanity`, `main` | `new/initial_geometry/audit.py` |
| Saved geometry aggregation | `main` | `new/initial_geometry/summarize.py` |
| Early continuation proxies and long screen | `predictors`, `long_screen` | `new/formation_gate/analyze.py` |
| Confusion counts, rankings and saved-array report | `confusion`, `ranking`, `main` | `new/formation_gate/analyze.py` |
| Published snapshot creation and CPU verification | `build`, `verify`, `--verify-only` | `tools/export_learning_geometry.py` |

Canonical entry: [integrated results](evidence/seed4_learning_geometry_20261004/RESULTS.md),
[initial geometry](evidence/initial_geometry_20261004/RESULTS.md) and
[summary](evidence/initial_geometry_20261004/summary.json), then
[formation results](evidence/formation_gate_20261004/RESULTS.md),
[interpretation](evidence/formation_gate_20261004/INTERPRETATION.md), and
[summary](evidence/formation_gate_20261004/summary.json). The public verification
command is `python -X utf8 -B tools/export_learning_geometry.py --verify-only`;
it checks the exported snapshot without NPZ files or checkpoints. Full replay
requires original archives excluded from GitHub, so a fresh clone cannot
recreate the source runs. See the [initial geometry protocol](new/initial_geometry/PROTOCOL.md)
and [formation protocol](new/formation_gate/PROTOCOL.md) for definitions and
dependencies, and the [initial geometry](evidence/initial_geometry_20261004/REPRODUCTION.md)
and [formation](evidence/formation_gate_20261004/REPRODUCTION.md) notes for
replay dependencies.

## Preceding diagnostic: joint195/200 conditional mechanism audit

- Protocol audit195_200_v1;2D only, zero training. COMPLETE492 conditions in
  885.469 seconds. Four full257-step primary endpoint Boolean and paired-margin
  traces replay exactly, maximum error0. No counterfactual arm is nonfinite.
- One selected training trajectory, same original W24/Z8 StreamingCell. Primary
  banks16 maps each at32/64 (61032/61064); independent diagnostic banks8 each
  (62032/62064). Use equal-map means; prior dense reports also showed pooled
  cell rates, which differ. These maps do not replicate training seeds.
- The eight-cell cross cube separates state producer at64, continuation rule
  and post-hoc readout. Primary32 strict T256: S200/D195/R195=1.000;
  S195/D200/R195=.93333, baseline195=.93268. Production before64 accounts for
  much of this contrast; changing the continuation rule still affects timing.
- Replacing only E or F with200 weights in a195 background gives primary32
  strict T256 .99967/1.000, while Q alone gives.89528. Full16 coalitions and
  all24 replacement orders expose coadaptation. A negative average Q marginal
  does not mean the trained200 Q is intrinsically harmful.
- Interpolation primary32 first-exit64->128 .04281 atlambda.4 to.000406 at.5.
  The removed failure is concentrated in map10. Independent32 baseline195
  already has survival1.000; no general state phase boundary is identified.
- Sparse same-role swaps, readout-null/span changes, local cross-checkpoint
  transplants and one-step pulses keep both cue worlds paired. Fixed-cohort
  all-steps-correct includes immediate swap damage. Small localized effects
  cannot identify a unique hidden-state contract or rule out global structure.
- Original180MB summary and420MB duplicate analysis remain local; public
  column-encoded per-map cases preserve scientific values, all negative arms,
  update probes and plans. Begin with the small summary/profiles, not all files.

| Concept | Exact symbols | Source |
|---|---|---|
| Replay, factorial/interpolation, cross continuation and roles | Runner.replay, experiments, suffix | new/audit_195_200/run.py |
| Fixed-cohort risk, survival and turnover | summarize, _interval, _cohort_summary | new/audit_195_200/metrics.py |
| Sparse paired reciprocal swaps, readout-relative projections | build_swaps, apply_swap, apply_z_projection_swap, transplant_between | new/audit_195_200/state_interventions.py |
| Original-clock step accounting and order-specific logit telescope | parts, semantic_decomposition, preactivation_parts | new/audit_195_200/instrument.py |
| Label-signed update statistics and masked single-step knockouts | summary, pulse | new/audit_195_200/probes.py |
| All24-order effects and saved-mask baseline comparisons | analyze, _factorial_summary, _state_pulse_effects | new/audit_195_200/analyze.py |
| Offline saved-trace validation and compact public export | main | new/audit_195_200/validate_saved.py; tools/export_joint195_evidence.py |
| Explicit public-archive reproduction adapter | main | tools/replay_joint195_public.py |

Canonical entry: [results](evidence/joint195_200_20261003/RESULTS.md),
[interpretation](evidence/joint195_200_20261003/INTERPRETATION.md),
[summary](evidence/joint195_200_20261003/summary.json).
Earlier multi-seed architecture/warm-start negative verdicts remain unchanged.

## Preceding diagnostic: historical seed4 dense update100--200 replay

- Protocol: seed4_dense_transition_v1,2D only. Execution COMPLETE300 updates,
  44 audits,353.141 seconds; training112.687 seconds. One selected seed4
  initialization, one fixed bank10002 and schedule20002; checkpoints/maps/
  cells/times are not independent training replications.
- Original5033-parameter W24/Z8 StreamingCell, two-hop macro clock and
  original K8 backward_trajectory are unchanged. No test forwards during
  training. Save0,100,105,...,200,300; audit21 dense checkpoints plus300 control.
- Exact initial/final and100/200 anchors; six complete historical endpoints
  replay with5628 integer and2636 floating leaves, maximum error0. Frozen
  source and independent input/reference bindings are retained.
- Fixed16-map banks: size32 seed61032 (same earlier stage bank), size64
  seed61064 (secondary scale description). Paired original/source-flipped
  rollout0..256; raw Boolean traces and signed margins are now preserved.
- G is endpoint wrong64->correct128 conditional on wrong64; endpoint D is
  correct64->wrong128 conditional on correct64. First-exit D includes any
  intermediate wrong step. Exact netgain=(1-q64)*G-q64*D uses shared counts.
- Size32 update195->200: strict pooled T12885.65%->97.55%; first-exit
  D64->1285.15%->0.032%; continuous survival64->25694.53%->99.97%.
  Update180 shows a substantial temporary collapse. Retention is already
  good on the then-correct subset at120/125, so this is not a demonstrated
  single monotone onset of preservation.
- Primary descriptive screen requires strict T128 equal-map AND pooled>=.80,
  G64->128>=.20, first-exit64->128<=.01 and continuous survival64->256>=.95.
  Only200 passes inside100--200;300 control also passes. Three consecutive
  five-update checkpoints are required for persistent onset.205/210 are
  unobserved, so operational_onset=null is inconclusive about persistence,
  not a failed architecture qualification.
- Terminal-stable minus first-correct lag is final-correct-conditioned and
  finite-horizon. Its median is already0 at120 and remains0 at180 despite
  the collapse (only94/1345 strict pixels final-correct). At2001345/1345
  are final-correct. It cannot identify latent commitment by itself.
- No contextual transplantation, causal handoff, infinite-horizon invariant,
  physical phase transition, seed-population reliability or3D experiment
  is supplied. Earlier multi-seed and warm-start negative decisions remain.
- Canonical entry: evidence/transition_20261003/RESULTS.md and summary.json.
  Detailed counts are secondary; raw NPZ/checkpoints remain local. Publication
  performs saved-artifact CPU checks only, without new training/GPU inference.
- Fresh-clone CPU/reference reproduction: tools/replay_transition_public.py.
  Public mode compares new weights to published parameter hashes and actual
  public stage JSON bytes. It explicitly does not reverify unavailable
  original checkpoint file bytes; the executed run used the private archive.

| Concept | Exact symbols | Source |
|---|---|---|
| Frozen historical K8 loss/detach cadence | backward_trajectory | new/short_bptt/training.py |
| Dense replay, fixed offline paired trajectories, screen | main, collect, report | new/transition_100_200/run.py |
| Exact turnover, first exit, first-entry survival, current-run-age hazard | summarize, _interval_summary, _fixed_lag_summary, _age_hazard | new/transition_100_200/metrics.py |
| Finite-horizon terminal lag and its exclusions | _finite_horizon_summary | new/transition_100_200/metrics.py |
| Independent saved-result validation | main | tools/validate_transition_results.py |
| Public-reference reproduction adapter | main | tools/replay_transition_public.py |
| Allowlisted public evidence and CPU verification | main | tools/export_transition_evidence.py |

## Preceding screen: detached on-policy warm-start of original StreamingCell

- Protocol: detached_warmstart_v1_nocap. 2D only; execution COMPLETE,8 arms x300
  updates,1629.375 seconds. Decision DEVELOPMENT_NOT_QUALIFIED.
- Same5033-parameter W24/Z8 original masked StreamingCell, two-hop macro
  clock, K8,64-step trained suffix, eight equally weighted task losses,
  training bank10002 and original schedule20002. Initialization seeds2/3/4/5
  are four paired units conditional on this bank/schedule.
- Both arms compute a no-grad prefix for the last four examples. Baseline
  discards it and calls the untouched historical K8 helper for all eight.
  Warmstart keeps the detached state for those four examples and starts the
  other four fresh. Prefix age choices32/64/128/192 share frozen seed60002.
  Older supervised state ages and the detached encoder history are part of
  the intervention; it is not an isolated credit-assignment intervention.
- Four baseline initial/final parameter hashes and full six historical
  endpoint records exactly reproduce. Fresh32-map banks at sizes32/64 use
  new seeds60032/60064. Baseline seed4 qualifies on the fresh full gate.
- Full phenotype passes: baseline1/4(seed4), warmstart0/4. Positive screen
  required warmstart>=3/4 and at least two more passes than baseline, with
  qualified historical/fresh controls. No threshold or seed rescue occurred.
- The same reach/hold/retention/growth/per-step-regression/matched-frontier
  conjunction from seed4_followup is reused. Partial improvements are
  retained separately: seed2 gains some output retention/growth; seed4 degrades.
  Full-gate failure is not uniform component failure or a general theorem.
- Stage diagnostics at updates0/100/200/300 use16 separate held-out size32
  maps(seed61032),128 paired steps, acquisition/destruction integer accounting
  and continuous survival. These are exploratory; no order parameter,
  computational phase transition or latent continuation closure is established.
- Before formal training the user removed the time cap. Failed preflight01
  remains recorded; CPU02/preflight02 qualified the amended no-cap source.
  Scientific conditions are unchanged. Prefix finite flags cover every state
  with sticky detection and eight-step host checks. No new GPU training or
  inference is used for this publication.

| Concept | Exact symbols | Source |
|---|---|---|
| Detached prefix, mixed fresh/warm suffix and original baseline | _prefix_state, backward_warmstart, _backward_mixed_trajectory | new/warmstart/training.py |
| Paired training, original replay and development decision | arms, train_one, replay, aggregate | new/warmstart/run.py |
| Exploratory integer turnover and continuous survival | diagnose, summarize_correct_traces | new/warmstart/diagnostics.py |
| Unchanged fresh full phenotype | evaluate, summarize_from_traces, predicate | new/seed4_followup/phenotype.py |
| Saved CPU checkpoint/trace validation | main | tools/validate_warmstart.py |
| Sanitized evidence and public arithmetic | main | tools/export_warmstart.py |

Start with [results](evidence/warmstart_20261003/RESULTS.md),
[compact aggregate](evidence/warmstart_20261003/summary.json), and
[protocol](new/warmstart/PROTOCOL.md). Individual full summaries and gate
checks are secondary; frontier integer CSVs and stage turnover records are
for verification. Full Boolean traces, checkpoints and machine receipts
remain local. [Reproduction](evidence/warmstart_20261003/REPRODUCTION.md)
distinguishes public arithmetic from full local saved-trace validation.

## Preceding follow-up: independent A/B/C on original Streaming seed4

- A/B execution COMPLETE,13 arms x300 updates,1866.672 seconds. Same5033
  parameter StreamingCell/W24/Z8, K8,64-step forward, eight losses, fixed
  512-map bank10002. Only A schedules or B starting parameters change.
- Control: initial/final parameter hashes and full six historical endpoints
  exactly reproduce; fresh32-map banks at seeds50032/50064 qualify. Fresh
  data are shared across arms and separate from the old40032/40064 maps.
- A: fixed seed4 initialization; four independent schedules20012/22/32/42,
  full phenotype0/4. B: four CPU Gaussian directions70002..5, paired global
  L2 radii.01/.05; full phenotype0/4 at each radius. Noise includes originally
  zero output weights/biases. B does not preserve the zero-output manifold.
- Gate: size32/T64 strict16<d<32 mean/pooled>=.80 and BA>=.85; hold at128/256;
  each-size all-changed64->256 retention>=.95, coverage gain>=.05,
  ever-regressed/ever-correct<=.15, and matched frontier>=.05 with coverage
  minima. Retention alone cannot qualify a small stalled solved region.
- All 12 A/B arms pass both matched-frontier gates but fail the conjunction.
  A20032 passes hold/retention/gain, fails reach and per-step regression;
  B70003/.05 passes reach+hold at32, fails parts of64 dynamics and regression.
  The sampled radius pattern is not monotone. No basin radius or reliability
  estimate follows from four schedules/directions conditional on one bank.
- C: old trained seed4,93/257 selected events across31 maps each at32/64,
  full replay/controls PASS. Step1 native-minus-sender14.52/7.00pp and
  wrong-sham-minus-sender3.76/.44pp miss the frozen joint threshold.
  Step4 secondary effects33.53/28.60pp and20.18/15.33pp remain secondary.
  Whole-cell W/Z rollback is local temporal-state sensitivity, not an edge
  message knockout or unique sender/flood-fill law. The threshold miss is
  not a statistical null test. C n=1 selected checkpoint, historical maps.
- Earlier architecture no-go decisions unchanged. Publication uses saved
  evidence, CPU checkpoint/trace validation and arithmetic; no new GPU work.

| Concept | Exact symbols | Source |
|---|---|---|
| Original cell and masked streaming | StreamingCell.step, stream | new/streaming_carry/stream_cells.py |
| Fixed init and paired Gaussian rays | initial_model | new/seed4_followup/initialization.py |
| Training/control replay and A/B decisions | arms, train_one, control_replay, aggregate | new/seed4_followup/train.py |
| Fresh traces, censoring, retained/acquired sets, all gates | evaluate, summarize_from_traces, predicate | new/seed4_followup/phenotype.py |
| Selected local rollback, matched sham, homogeneous-time copies | select_events, event_batches, run_event_outcomes, aggregate | new/seed4_followup/causal.py |
| Full saved-evidence CPU validation | main | tools/validate_seed4_followup_training.py, tools/validate_seed4_followup_causal.py |
| Sanitized delivery and public arithmetic | main, verify | tools/export_seed4_followup.py |

Begin with [results](evidence/seed4_followup_20261003/RESULTS.md),
[interpretation](evidence/seed4_followup_20261003/INTERPRETATION.md),
[summary](evidence/seed4_followup_20261003/summary.json), and
[protocol](new/seed4_followup/PROTOCOL.md). Individual summaries retain all
checks/denominators/profiles. Raw arm JSON, matched CSVs and C events are
secondary. Full Boolean traces and checkpoints remain local, bound in
[provenance](evidence/seed4_followup_20261003/provenance.json). Public arithmetic
does not replace full local trace/checkpoint validation.

## Preceding descriptive diagnostic: original seed4 retained correctness and local acquisition

- Execution: COMPLETE,31.813 seconds, zero training. Same original Streaming
  seed4 checkpoint;32 historical evaluation maps at each size32/64. Every
  macro step0..256 saved. Model replication n=1, selected successful checkpoint,
  maps reused. There is no new architecture qualification or intervention.
- Replay: six complete historical T64/128/256 records exactly reproduce,
  including5628 integer leaves and2636 float leaves, maximum error0.
- Endpoint retention: all changed pixels correct at64 remain correct at256
  at6856/6856 (100%) for size32 and18827/19084 (98.65%) for size64.
  Wrong-at64 to correct-at256 acquisitions are1446/13275 respectively.
- Every-step qualification:186/8303 (2.24%) ever-correct pixels regress at
  size32;2768/33328 (8.31%) at size64. Endpoint retention is not per-step
  monotonicity. One of the size32 correct-at64 pixels briefly errs then recovers.
  Terminal-stable means correct at EVERY subsequent step through256 only.
- Distance censoring:392/8695 and10860/44188 changed pixels never become paired
  correct through256. Final wrong counts393/12086 include transiently correct
  pixels. Primary-band success does not equal full-component completion.
- Frontier: wrong pixels with a currently correct open one-hop neighbor versus
  wrong pixels without one, matched within map/time/exact source BFS distance.
  Acquisition t->t+1 at starts0,8,...,248. Both groups required; within-map
  weight nf*nn/(nf+nn), then equal-map mean. Eligible maps31/32 and32/32;
  common strata1081/5801; differences+30.92/+22.44 percentage points.
- Interpretation: mostly retained paired output correctness and local acquisition
  in this checkpoint. Shared latent propagation may cause both the neighbor
  and target outputs; a macro step can cross two hops. No causal handoff,
  flood-fill, invariant latent closure, bounded state, basin, seed reliability,
  fresh-map confirmation, resolution scaling law or3D claim is established.
  Earlier architecture no-go decisions and the latest sidecar verdict remain.

| Concept | Exact symbol | Source |
|---|---|---|
| Every-step original/flipped predictions and replay gate | trace, main | new/frontier_audit/audit.py |
| First/suffix time, loss, matched open-neighbor acquisition | analyze_trace, _frontier_interval | new/frontier_audit/metrics.py |
| Compact behavior report and fixed map0 figures | build_summary, save_report | new/frontier_audit/report.py |
| Independent full saved-trace CPU accounting and source/BFS checks | main, times, bfs | new/frontier_audit/validate.py |
| Synthetic metric fixtures | run_checks | new/frontier_audit/check_metrics.py |
| Sanitized copies and public CSV arithmetic | main, verify | tools/export_frontier_audit.py |

Read [results](evidence/frontier_audit_seed4/RESULTS.md),
[diagnosis](evidence/frontier_audit_seed4/DIAGNOSIS.md),
[summary](evidence/frontier_audit_seed4/summary.json),
[protocol](new/frontier_audit/PROTOCOL.md),
[validation](evidence/frontier_audit_seed4/validation.json), and
[publication bindings](FRONTIER_AUDIT_PUBLICATION_MANIFEST.json).
Public summary drops only redundant per-map transition rows. Matched integer
count CSVs reproduce frontier effects; per-map first/stable summaries and
exact-distance timing/censor profiles remain public. Full traces and large
behavior JSON remain local, bound by provenance. The full saved-trace check
was performed locally; public arithmetic verification does not replace it.
Run `python tools/export_frontier_audit.py --verify-only` from repository root
without checkpoints. Full trajectory replay requires local checkpoint/source
artifacts excluded from GitHub. No training or model inference for publication.

## Current experiment: exactly nested stationary sidecar is a no-go

- Execution: COMPLETE,12 arms x300 updates,1676.281 seconds. Formal status:
  DEVELOPMENT_NO_GO; inspected seed n=4, one fixed training bank/schedule,
  reused32 evaluation maps per size32/64. This is a2D development screen.
- Controls: four original Streaming final parameter hashes and complete
  evaluations exactly reproduce history. All core initial parameters match
  across three arms; the two side-arm full initial states also match.
- Intervention: original W24/Z8, F67->40->24, Q67->16->8, streaming,
  Laplacians and perception clocks remain. Extra G67->32->12 writes H12;
  zero-initialized bias-free P_F/P_Q feed active H into the old residuals.
  Memory retains H; stateless consumes the same instantaneous write and
  resets only H. Both side arms7989 params, original5033; not FLOP matched.
- Frozen gate: T64 size32 strict16<d<32, both paired mean/pooled>=80%, both
  source-orientation BA>=85%; hold bounds declines at BOTH T128/T256.
  GO needs memory>=3/4 including4 and strictly above both controls.
  Reach+hold: stream1/4, memory0/4, stateless0/4. All three reach only seed4.
- Seed4 primary pooled T64/T128/T256: stream85.92/99.35/100%,
  memory94.24/0/0.38%, stateless100/100/88.49%. Memory loses primary and BA
  atT128. Stateless fails paired and BA hold atT256. Seed2 stateless reaches
  the primary threshold only later; its failed T64 cannot be rescued.
- Meaning: the additional carried-sidecar recipe does not stabilize or
  reproduce the rare positive long-rollout regime. Some finite-horizon gains
  are descriptive. Original Z is already local memory; this does not reject
  stationary state in general, identify a Jacobian/overwrite mechanism,
  restore gradients across detach, prove significance, or claim3D results.
- Neutral nesting concerns projected W/Z/logits and raw core derivatives.
  It does not preserve optimizer updates under joint clipping. Additional
  H carry changes accumulation and activation scale together; stateless W/Z
  remain recurrent. Side gradient paths open by update3, not all at update1.

| Concept | Exact symbol | Source |
|---|---|---|
| Extra local carry and immediate stateless feedback | SidecarCell.step | new/stationary_sidecar/sidecar_cells.py |
| Original-core initial identity | core_state_dict, cpu_initialization_reference | sidecar_cells.py, run.py in new/stationary_sidecar |
| Three-state detach without reset | backward_trajectory | new/short_bptt_phase2/training.py |
| Gradient-entry, reach, hold and GO | gradient_entry_check, predicates, decision | new/stationary_sidecar/run.py |
| Independent saved counts, control hashes and hold failures | verify_metric, gate_row, hold_failures, main | new/stationary_sidecar/analyze.py |
| Sanitized publication | main, public | tools/export_stationary_sidecar_evidence.py |

Read [results](evidence/stationary_sidecar_init2345/RESULTS.md),
[analysis](evidence/stationary_sidecar_init2345/analysis.json),
[protocol](new/stationary_sidecar/PROTOCOL.md),
[validation](evidence/stationary_sidecar_init2345/validation.json),
[CPU qualification](evidence/stationary_sidecar_init2345/cpu_validation.json)
and [publication bindings](STATIONARY_SIDECAR_PUBLICATION_MANIFEST.json).
Analysis fields primary_trajectories, seed4_trajectory and hold_failure_reasons
contain the exact rollout metrics and failed constraints. Raw JSON is secondary.

## Previous diagnostic: fixed-weight Streaming seed4 operator sensitivity

- Execution: COMPLETE, 12.36 seconds, zero training. Four cold-rollout
  conditions use the same original seed4 weights, initialization and inputs.
  Two sizes, 32 historical maps each, T8/16/32/64/128/256. Model n=1 selected
  from the prior failed multi-seed screen; maps are not training replicates.
- Scientific status: SELECTED_CHECKPOINT_OPERATOR_SENSITIVITY. No new GO gate.
  Full original evaluations at size32/64 and T64/128/256 replay exactly:
  5628 integer leaves and2636 float leaves, maximum absolute error0.
- Switches: T_M versus identity in the workspace carry and first F feature;
  masked Laplacians versus zero tensors in BOTH F and Q input slots. F retains
  pre-stream perception clock and Q retains post-F W/pre-update Z clock.
  Parameters, W24/Z8 widths, residual scales and readout remain unchanged.
- Primary size32 strict16<d<32 pooled paired correctness at T64/128/256 is
  85.92/99.35/100% for full and0% (0/2918) for every knockout. Size64 d>32
  full is12.48/38.60/59.43%; every knockout is0% there as well.
- Nearby computation also fails. Without perception, d<8 has0/1706 hits
  already atT8. Without transport it has906/1706 atT8 and1067/1706 atT16,
  then zero byT32. Primary failure is not solely a slower remote wave.
- Scope: knockouts change learned feature/state distributions. Upper-bound
  graph hops per macro-step are2/2/1/0 for full/no_transport/no_perception/
  neither. Equal steps do not equal matched communication opportunity.
  Loss measures intervention sensitivity of this solution, not unique semantic
  path attribution, the cause of H/C/Z failure or inability to retrain.
  The prior Streaming DEVELOPMENT_NO_GO and positive seed4 case are unchanged.

| Concept | Exact symbol | Source |
|---|---|---|
| Fixed-weight F/Q switches | step, CONDITIONS, hops_per_step | new/stream_path_audit/operators.py |
| Independent nonzero CPU reference | reference_step, reference_stream, reference_laplacian, check | new/stream_path_audit/check.py |
| Replay gate, cold rollouts and paired contrasts | compare_tree, evaluate, contrasts, main | new/stream_path_audit/audit.py |
| Saved denominator/count/CSV/source verification | main | new/stream_path_audit/analyze.py |
| Public copies and sanitization | main, public | tools/export_stream_path_audit.py |

Start at [results](evidence/stream_path_audit_seed4/RESULTS.md),
[interpretation](evidence/stream_path_audit_seed4/INTERPRETATION.md) and
[summary](evidence/stream_path_audit_seed4/summary.json). Then read
[protocol](new/stream_path_audit/PROTOCOL.md),
[validation](evidence/stream_path_audit_seed4/validation.json) and
[publication bindings](STREAM_PATH_AUDIT_PUBLICATION_MANIFEST.json).
[Raw conditions](evidence/stream_path_audit_seed4/raw_conditions.json) are
secondary; original checkpoint and machine metadata remain local.

## Previous: Persistent Roles development screen is a no-go

- Execution: COMPLETE;12 matched arms completed300 updates each in1204.234
  seconds (20.07 minutes). Scientific decision: DEVELOPMENT_NO_GO.
- Reach+hold: baseline2/4, stream1/4, roles0/4. All eight controls exactly
  reproduce their historical final parameter hashes and complete evaluations.
- Frozen primary: size32/T64, strict16<d<32. Paired per-map mean AND pooled
  correctness>=80%, original AND flipped BA>=85%. Hold at BOTH T128/T256
  permits BA declines<=3pp and paired-statistic declines<=5pp from T64.
  Continuation additionally needs at least3/4 candidate reach+hold seeds,
  including2/5, and a count exceeding BOTH controls. Every roles seed misses
  reach. All four hold=True values preserve failed endpoints.
- Candidate primary pooled correctness by seed2/3/4/5 atT64 is
  0.00/0.31/0.00/1.58% (0/9/0/46 correct out of2918 paired pixels per seed).
  These require the SAME changed-region pixel to be correct under BOTH source
  alternatives; they are not ordinary whole-grid pixel accuracy.
  AtT256 pooled is0.00/0.58/0.00/2.09%; no horizon recovers reach.
  Size64/d>32 remains<=0.24% pooled throughT256 (max65/27436).
- Architecture: H12 stationary computation, C12 persistent four-direction
  carrier (three channels per lane), Z8 stationary task state. One pointwise
  R35->72->32 with Tanh hidden and zero-initialized output is shared across
  two collision-then-stream phases per macro-step. Each phase updates
  (H,C,Z)<-(H+.1*dH,T(C+.1*dC),Z+.5*dZ). Only C moves; no Laplacian input.
  H/C split the old encoder3->24 once; Z0=0; readout uses Z alone.
- Matched:5033 parameters,32 persistent scalars, common encoder/readout
  initialization, historical data/schedule, K8 losses,300 updates and the
  evaluator. Full candidate initialization differs. K8 carrier radius<=16,
  task-readout radius<=15; reused normalized2K bands are historical comparison
  bands, not exact candidate readout bounds.
- Scope: complete parameterization comparison. Relative to old Streaming,
  a dedicated H workspace is added while carrier width24->12, perception,
  rule sharing and readout clock also change. Old Streaming already has
  stationary Z. No H-specific failure cause, family-wide impossibility,
  restored cross-detach gradients, arbitrary addressing, stable trained
  recurrence, fresh confirmation or3D claim follows.
- Independent unit: initialization seed n=4, conditional on one fixed bank
  and schedule. The same32 evaluation maps per size are reused across seeds.
  Rollup eligible-map totals count map-seed evaluations, not extra samples.

| Concept | Exact symbol | Source |
|---|---|---|
| One-time H/C initialization | PersistentRoleCell.initial | new/persistent_roles/role_cells.py |
| Shared collision then carrier stream | PersistentRoleCell.phase, PersistentRoleCell.step | new/persistent_roles/role_cells.py |
| Three-state K8 training | backward_trajectory | new/short_bptt_phase2/training.py |
| Endpoint, hold and development decision | predicates, decision, aggregate | new/persistent_roles/run.py |
| Saved arithmetic, denominators and CPU parameter hashes | main | new/persistent_roles/analyze.py |
| Sanitized publication | main, public | tools/export_persistent_roles_evidence.py |

Start with [results](evidence/persistent_roles_init2345/RESULTS.md) and
[compact analysis](evidence/persistent_roles_init2345/analysis.json), then
[protocol](new/persistent_roles/PROTOCOL.md), [architecture](ARCHITECTURE.md),
[validation](evidence/persistent_roles_init2345/validation.json),
[completion](evidence/persistent_roles_init2345/completion.json) and
[publication bindings](PERSISTENT_ROLES_PUBLICATION_MANIFEST.json).
[Curves](evidence/persistent_roles_init2345/curves.csv),
[paired effects](evidence/persistent_roles_init2345/paired_effects.csv) and
[raw records](evidence/persistent_roles_init2345/raw/) are secondary.

## Previous: Local Interface development screen is a no-go

- Execution: COMPLETE; all 12 matched arms completed 300 updates each in
  1281.532 seconds (21.36 minutes).
- Scientific status: DEVELOPMENT_NO_GO; reach+hold is baseline 2/4, stream
  1/4, and interface 0/4.
- Frozen endpoint: size32/T64, strict graph distance 16<d<32; paired per-map
  mean and pooled accuracy must both reach 80%, and original/flipped BA must
  each reach 85%. At both T128 and T256, declines must stay within 3 pp for BA
  and 5 pp for each paired statistic relative to T64.
- Interface outcome: all four seeds have 0.00% per-map mean and 0.00% pooled
  paired correctness across 32 maps and 2,918 eligible pixels. All four also
  miss the BA threshold. Seed 4 has hold=True but reach=False, so it does not
  pass the combined endpoint.
- Controls: all eight baseline/stream arms exactly reproduce historical final
  parameter hashes and complete evaluation payloads. Baseline reaches and
  holds in 2/4 seeds; stream does so in 1/4.
- Scope: previously inspected seeds and maps are reused; the independent unit
  is initialization seed, n=4 conditional on one training bank and schedule.
  This is developmental evidence, not fresh confirmation.
  Each size uses 32 fixed evaluation maps across seeds. The compact analysis
  rollups sum `eligible_maps` over seed evaluations; that field counts repeated
  map evaluations, not additional maps or independent statistical units.

| Seed | Interface mean / pooled % | Interface reach / hold |
|---:|---:|---|
| 2 | 0.00 / 0.00 | False / False |
| 3 | 0.00 / 0.00 | False / False |
| 4 | 0.00 / 0.00 | False / True |
| 5 | 0.00 / 0.00 | False / False |

- Intervention: W24 and Z8 stay locally resident. One shared pointwise E maps
  [W,Z,X] (35 channels) to a fresh 24-channel message M, split into four
  six-channel directional ports and transported by the fixed masked
  permutation. F updates W; a second emission uses W' and the same E before Q
  updates Z. The message is consumed within its phase and discarded.
- Matched: 5,033 parameters, 32 persistent state channels, two communication
  phases, K8 training, data/schedule, evaluator and 300 updates. The candidate
  alone adds 24 transient message channels. Only the common encoder/readout
  initialization draws are shared;
  the complete candidate initialization differs because E/F/Q are new or
  differently shaped parameters.
- Claim boundary: this is a bundled parameterization comparison. State
  allocation, neighbor representation and hidden widths change together, so
  the result does not isolate interface factorization. The pure port
  permutation is norm-preserving, but E, residual updates and repeated
  receive/store/re-emit computation have no losslessness or stability
  guarantee. There is no family-wide, mechanism, arbitrary-routing or 3D claim.

| Concept | Exact symbol | Source |
|---|---|---|
| Fresh two-phase interface messages and local state updates | InterfaceCell.step, InterfaceCell._message | new/local_interface/interface_cells.py |
| Variant construction | make_variant | new/local_interface/interface_cells.py |
| Training, reach/hold and arm decisions | train_one, predicates, decision, aggregate | new/local_interface/run.py |
| Independent saved-result analysis | main | new/local_interface/analyze.py |
| Sanitized evidence export | main, public | tools/export_local_interface_evidence.py |

Read the [full report](evidence/local_interface_init2345/RESULTS.md),
[compact analysis](evidence/local_interface_init2345/analysis.json),
[validation](evidence/local_interface_init2345/validation.json),
[completion record](evidence/local_interface_init2345/completion.json),
[protocol](new/local_interface/PROTOCOL.md), and
[publication bindings](LOCAL_INTERFACE_PUBLICATION_MANIFEST.json).
Use [curves](evidence/local_interface_init2345/curves.csv),
[paired effects](evidence/local_interface_init2345/paired_effects.csv) and
[sanitized raw records](evidence/local_interface_init2345/raw/) for detailed
denominators and per-seed checks. The full three-arm table is in the report.

## Previous: Streaming Carry development screen is a no-go

- Execution: `COMPLETE`; eight matched arms, 300 updates each, 828.297 seconds.
- Scientific status: `DEVELOPMENT_NO_GO`; reach+hold is baseline 2/4 and
  stream 1/4, below the frozen stream threshold of 3/4. Only stream seed 4
  passes; seeds 2, 3 and 5 fail reach.
- Baseline reproduction: all four baseline final parameter hashes and complete
  evaluation records exactly match Phase II.
- Scope: development with previously inspected initialization seeds2-5 and
  maps; independent unit is seed, n = 4 conditional on one training bank and
  schedule. This is not fresh confirmation.
- Primary: size 32/T64, strict graph distance 16<d<32; both per-map mean and
  pooled paired correctness must reach 80%, with both balanced accuracies >=85%.
  Hold requires T128 and T256 drops <=3 percentage points in balanced accuracy
  and <=5 percentage points in both paired statistics relative to T64.

| Seed | Baseline mean/pooled % | Stream mean/pooled % | Baseline reach+hold | Stream reach+hold |
|---:|---:|---:|---|---|
| 2 | 93.70 / 92.97 | 1.08 / 1.47 | pass | fail |
| 3 | 21.85 / 17.27 | 8.28 / 5.62 | fail | fail |
| 4 | 52.40 / 44.76 | 88.89 / 85.92 | fail | pass |
| 5 | 96.45 / 95.68 | 16.05 / 13.02 | pass | fail |

Seed 4 stream's fixed size 32 primary band has paired mean/pooled correctness
88.89/85.92% at T64, 99.64/99.35% at T128, and 100/100% at T256. At size 64,
the same narrow band is 93.01/93.50%, 93.39/94.33%, and 93.75/94.85% at
T64/128/256; size 64 is descriptive and does not enter the frozen hold gate.
For d>32 at size 64, stream paired mean at T64/128/256 is
26.32/55.78/74.99%, while pooled correctness is 12.48/38.60/59.43%. These
single-seed measurements do not change the frozen size 32 primary decision.

- Intervention: a fixed masked port permutation streams the 24 W channels as
  four six-channel N/E/S/W lanes; Z8 stays local. Open links move each lane to
  its neighbor, blocked links bounce it into the opposite lane, and wall ports
  stay fixed. No learned transport parameters are added.
- Matched: same additive baseline, initialization, data, schedule, evaluator,
  K8 trainer, 300 updates, 64-step trajectories, eight losses and one optimizer
  update; both arms retain 5,033 parameters and two communication phases.
- Claim boundary: the fixed permutation preserves global Euclidean norms and
  distances. The nonlinear residual and local-memory updates do not inherit
  that guarantee. F's first feature is incoming T(W), while old L(W) and L(Z)
  remain in its input, so this is not a pure collision model. No arbitrary
  routing, causal failure mechanism, broad robustness or 3D claim follows.
- Stop: the screen is complete; no retuning, confirmation run or monitor is
  scheduled.

| Concept | Exact symbol | Source |
|---|---|---|
| Masked port permutation and inverse | `stream`, `inverse_stream` | `new/streaming_carry/stream_cells.py` |
| Incoming carrier and residual update | `StreamingCell.step` | `new/streaming_carry/stream_cells.py` |
| CPU operator, gradient and clock checks | `main` | `new/streaming_carry/check.py` |
| Paired metrics and distance-band evaluation | `p2.paired`, `p2.evaluate` | `new/short_bptt_phase2/run.py` |
| Reach/hold, baseline reproduction and decision | `predicates`, `decision`, `aggregate` | `new/streaming_carry/run.py` |
| Saved-result arithmetic and validation | `main` | `new/streaming_carry/analyze.py` |
| Sanitized evidence package | `main`, `public` | `tools/export_streaming_carry_evidence.py` |

Read [results](evidence/streaming_carry_init2345/RESULTS.md),
[compact analysis](evidence/streaming_carry_init2345/analysis.json),
[protocol](new/streaming_carry/PROTOCOL.md),
[validation](evidence/streaming_carry_init2345/validation.json), and
[publication bindings](STREAMING_CARRY_PUBLICATION_MANIFEST.json). Use
[curves](evidence/streaming_carry_init2345/curves.csv),
[paired effects](evidence/streaming_carry_init2345/paired_effects.csv) and
[raw arm records](evidence/streaming_carry_init2345/raw/) for detailed
denominators and per-seed checks. Checkpoints and machine receipts are excluded
from the public package. Baseline comparison identities are part of the
validation record.

## Previous: Direct Spatial Carry development screen is negative

- Execution: `COMPLETE`;8 arms,300 updates,797.797 seconds.
- Scientific status: `DEVELOPMENT_NO_GO`.
- Narrow reach AND hold: baseline2/4, carry0/4. Every primary mean and pooled
  value decreases with carry. Both old positives2/5 are lost.
- Intervention: only W identity -> W-0.5*D_M^dagger*L_M(W), fixed rho0.5,
  degree counts real open neighbors; isolated/wall nodes retain identity.
- Matched: F/Q, additive Z, W24/Z8,5033 parameters, two communication phases,
  K8,300 updates,64 forward steps, eight equally weighted losses and one
  optimizer step. Same Phase-II bank10002/schedule20002/eval40032,40064.
- Baseline reproduction:4/4 final parameter hashes and complete evaluations
  exactly match the earlier Phase-II K8 records. No baseline drift here.
- Scope: development with previously inspected seeds2-5/maps, n=4
  initialization variation conditional on one bank/schedule; not confirmation.
- Stop: no continuation to fresh-seed confirmation, new architecture or monitor.

Primary size32/T64 strict16<d<32: mean AND pooled paired>=80%, both BA>=85%.
Hold requires T128 AND T256 drops<=3pp BA and<=5pp paired relative to T64.
Continue only with>=3 carry successes, preservation of2/5 and more successes
than concurrent baseline, with baseline2/5 reproduced. Full-K64 qualification
is not a gate in this two-arm architecture screen.

Primary pooled baseline -> carry: seed2 92.97->0%, seed3 17.27->3.12%,
seed4 44.76->11.41%, seed5 95.68->4.42%. At size64/T256, carry d>32 paired
is0% for all four. Seed4 has a transient T128 narrow-band benefit: pooled
74.64% versus43.59% at size32 and82.54% versus37.06% at size64. It falls below
baseline again byT256 and fails hold; this does not rescue the fixed endpoint.
Do not mistake carry seed2 hold=True for retained ability: its reach is false
and primary stays zero. Whole-grid BA and paired far-pixel correctness differ.

Interpretation: the tested lazy-average path harms this K8 recipe. It replaces
part of temporal identity with mixing, so averaging-induced dilution or other
mechanisms are hypotheses, not identified causes. Channel preservation does
not ensure task-information preservation. Other carry operators remain untested.

| Concept | Exact symbol | Source |
|---|---|---|
| Real-edge degree and fixed carry | `masked_degree`, `spatial_carry` | `new/direct_spatial_carry/carry_cells.py` |
| Sole W intervention with old F/new-W Q | `DirectCarryCell.step` | `new/direct_spatial_carry/carry_cells.py` |
| Unchanged gradient accumulation | `backward_trajectory` | `new/short_bptt_phase2/training.py` |
| Original paired evaluator | `paired`, `selections`, `evaluate` | `new/short_bptt_phase2/run.py` |
| Reach/hold, baseline reproduction, comparison | `predicates`, `decision`, `aggregate` | `new/direct_spatial_carry/run.py` |
| Independent CPU result arithmetic | `main` | `new/direct_spatial_carry/analyze.py` |

Read [results](evidence/direct_spatial_carry_init2345/RESULTS.md),
[compact analysis](evidence/direct_spatial_carry_init2345/analysis.json),
[protocol](new/direct_spatial_carry/PROTOCOL.md),
[validation](evidence/direct_spatial_carry_init2345/validation.json), and
[publication bindings](DIRECT_CARRY_PUBLICATION_MANIFEST.json).
Raw records/schedule/aggregate/CSV are byte-identical; public manifest/status
copies omit PID only. Checkpoints, machine receipts and logs remain local.
CPU verification reconstructs denominators and aggregates saved counts;
it does not regenerate logits. Earlier evidence is unchanged.

## Previous: Phase II completed, K8 replication threshold not met

- Execution: `COMPLETE`; 12 arms, 300 updates each, 1095.047 seconds.
- K8 narrow reach: `NOT_REPLICATED`, 2/4 versus required3/4.
- K8 narrow reach+hold: `NOT_REPLICATED`, 2/4 versus required3/4.
- Comparison: `BASELINE_UNQUALIFIED`; K64 reaches0/4.
- K16 context: reaches3/4; reaches+holds1/4.
- Scope: additive only, K8/16/64, initialization seeds2/3/4/5. Shared train
  bank10002 and batch schedule20002; new eval banks40032/40064,32 maps each.
- Unit: initialization seed,n=4 conditional on one data bank and schedule.
  Backend nondeterminism is not claimed to have been eliminated.
- Stop: bounded run complete; no rescue training, new architecture or monitor.

Primary is STRICT16<d<32 at size32/T64: paired correctness must be>=80% both
per map and pooled, and original/flipped BA must both be>=85%. This band and
architecture were selected after Phase I, then frozen for new evaluation maps.
The previous broader gate is unchanged. K16 covers this primary band within
its own radius32, so its reach is not cross-window evidence. Normalized d/(2K)
bands are secondary and must not replace the shared absolute endpoint.

Forward64, losses every8, and one optimizer update after64 are shared. K16 sums
two losses before each16-step backward; K8 uses one, K64 uses all eight.
Parameter gradients accumulate across windows. Both W/Z histories are cut
without changing state values; clipping is once at the trajectory end.
Model architecture remains W24/Z8,5033 parameters,two masked phases per step.

K8 seeds2/5 reach and hold: pooled primary92.97/95.68% atT64,93.21/93.59%
atT256. The other seeds are17.27/44.76% atT64. At size64/T256, the positive
models retain89.65/97.25% pooled in the primary band, but d>32 is36.78/14.83%.
There are no distance>=64 changed pixels in size32 evaluation; these are null
measurements, not failures. All K8 clip<=6%, including failures; K16 seed2
clips60% and passes. Large/small gradients alone do not explain success.

Public records preserve all12 raw arm files, integer hits/denominators and
the shared schedule byte-identically. Only PID metadata is removed from
analysis/status/manifest/preflight copies. The CPU review independently
reconstructed denominators and144 BA/1008 paired aggregates and verified all
12 decisions, parameters, data and source hashes. No new training/inference.

| Concept | Exact symbol | Source |
|---|---|---|
| Loss cadence versus detach cadence | `backward_trajectory` | `new/short_bptt_phase2/training.py` |
| K16 independent gradient reference and K8/K64 equivalence | `main` | `new/short_bptt_phase2/check.py` |
| Integer paired counts and absolute/normalized bands | `paired`, `selections`, `evaluate` | `new/short_bptt_phase2/run.py` |
| Absolute replication versus control qualification | `reach_predicate`, `aggregate` | `new/short_bptt_phase2/run.py` |
| Independent saved-result verification | `main` | `new/short_bptt_phase2/analyze.py` |
| Sanitized publication | `main`, `public` | `tools/export_bptt_phase2_evidence.py` |

Start with [results](evidence/short_bptt_phase2_init2345/RESULTS.md),
[compact analysis](evidence/short_bptt_phase2_init2345/analysis.json),
[protocol](new/short_bptt_phase2/PROTOCOL.md),
[validation](evidence/short_bptt_phase2_init2345/validation.json), and
[publication bindings](BPTT_PHASE2_PUBLICATION_MANIFEST.json).
Do not infer general credit assignment, a new RelationFirst representation,
optimal K, broad distance extrapolation or robust training across datasets.

## Previous: matched short-BPTT Phase I completed, all full controls unqualified

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
