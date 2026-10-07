# Results: inertial NCA screen, with preserved A0/v1 evidence

## Latest: full-writer port-relation screen

**COMPLETE:24/24 trajectories,312/312 predeclared stages,624 size-metric rows.**
The fixed u300 size32 joint-readiness endpoint remains the only formal endpoint.

| Arm | Joint-ready u300 | Old Full u300 | Dense ever-ready | Size32 strict pooled T64 mean | T64→T256 retention mean |
|---|---:|---:|---:|---:|---:|
| Current | 0/8 | 0/8 | 0/8 | .110424 | .548940 |
| Constant | 0/8 | 0/8 | 1/8 | .137410 | .429450 |
| Conditioned | 0/8 | 0/8 | 0/8 | .050948 | .475381 |

Conditioned-minus-Current has8/8 complete pairs,0 candidate-only wins,0 losses,
net0,exact two-sided p=1: **`NO_CONDITIONED_RELIABILITY_QUALIFICATION`**.
This all-tie contrast does not establish equivalence. Constant's one dense
success was lost by u300 and cannot replace that endpoint. Means above are
across8 blocks, with insufficient initial support explicitly null in raw rows.

Read [full result](evidence/port_relation_20261007_02/RESULTS.md),
[aggregate](evidence/port_relation_20261007_02/summary.json),
[all checkpoint metrics](evidence/port_relation_20261007_02/metrics.csv), and
[saved-data validation](evidence/port_relation_20261007_02/validation.json).
The [recovery record](evidence/port_relation_20261007_02/interruption_and_recovery.json)
preserves the original diagnostic-overflow interruption and33 inherited stages.
No general architecture rejection or unique collapse mechanism follows.

## Earlier: Hybrid Writer v0 reliability screen

`hybrid_lane_write_budget_v1`: execution and formal aggregation **COMPLETE**,
32/32 trajectories and 416/416 predeclared checkpoint records in 8,864.187
seconds (about 2 h 28 min). The official endpoint is fixed u300; intermediate
checkpoints do not replace it.

| Arm | Valid blocks | Joint-ready at u300 | Wilson 95% interval |
|---|---:|---:|---:|
| neural | 8/8 | 1/8 | [0.0224175, 0.470888] |
| budget | 8/8 | 0/8 | [0, 0.324408] |
| hybrid | 8/8 | 0/8 | [0, 0.324408] |
| affine_hybrid | 8/8 | 0/8 | [0, 0.324408] |

| Primary contrast (candidate − reference) | Complete pairs | Candidate-only wins | Losses | Holm-adjusted exact p | Verdict |
|---|---:|---:|---:|---:|---|
| budget − neural | 8 | 0 | 1 | 1 | NO_RELIABILITY_QUALIFICATION |
| hybrid − budget | 8 | 0 | 0 | 1 | NO_RELIABILITY_QUALIFICATION |
| hybrid − neural | 8 | 0 | 1 | 1 | NO_RELIABILITY_QUALIFICATION |

The frozen rule requires at least 6/8 net wins and Holm-adjusted exact two-sided
p<=0.05. Overall verdict: `NO_PRIMARY_RELIABILITY_QUALIFICATION`. Observed
ever-ready counts on the predeclared grid match u300; readiness-loss is 0/1 for
neural and undefined for budget, hybrid and affine_hybrid, which never became
ready. This fixed lane-write-budget recipe did not improve frozen reliability.
The finite result gives no causal root diagnosis and does not reject general
Hybrid or NCA designs.

1. [Results](evidence/hybrid_writer_20261006/RESULTS.md),
   [summary](evidence/hybrid_writer_20261006/summary.json),
   [saved-data validation](evidence/hybrid_writer_20261006/validation.json), and
   [publication manifest](HYBRID_WRITER_PUBLICATION_MANIFEST.json).
2. [Final metrics](evidence/hybrid_writer_20261006/final_metrics.csv),
   [dense records](evidence/hybrid_writer_20261006/dense.json),
   [per-arm records](evidence/hybrid_writer_20261006/perarm.json), and
   [reach/retention trajectory](evidence/hybrid_writer_20261006/reach_retention_trajectory.png).
3. [Reproduction and artifact map](evidence/hybrid_writer_20261006/REPRODUCTION.md),
   then the [frozen protocol](new/hybrid_writer/PROTOCOL.md) and
   [implementation](new/hybrid_writer/cells.py).

The publication includes 832 packed trace banks, 416 summaries, 416 telemetry
records and 32 curves as losslessly compressed secondary data. Checkpoints are
not published. Verify saved evidence from the repository root:

    python -X utf8 -B tools/export_hybrid_writer.py --verify-only

## Selected block7 formation/collapse continuation audit

COMPLETE32/32 producer-consumer matrix units and12/12 same-state single-step
units,90.97seconds. No training, optimizer update or fitted state alignment.
Eight native diagonals replay all Boolean bits exactly. This evaluates one
selected training trajectory, not 44 independent success trials.

Fixed R275, source-paired all-changed coverage atT256:

|Producer toT64|Consumer toT256|size32|size64|
|---:|---:|---:|---:|
|275|275|.950144|.840536|
|275|300|.970374|.933684|
|300|275|.489719|.111351|
|300|300|.425492|.059384|

G300 preserves u275-correct cells continuously at.999731/.999708 and makes
more sustained progress. Its one-step destruction on u275 states is below
.0003 at all tested times/sizes. G275 only partly improves u300-produced states.
The distinction points to cold-prefix state formation (encoder plus64 updates),
without identifying a unique module or excluding representation compatibility.
This zero-training audit did not evaluate Hybrid; see the separate formal result
above. Earlier formal u300 negative qualification is unchanged.

[Results](evidence/block7_collapse_20261006/RESULTS.md),
[summary](evidence/block7_collapse_20261006/summary.json),
[matrix summaries](evidence/block7_collapse_20261006/matrix_summary.csv),
[single-step summaries](evidence/block7_collapse_20261006/single_step_summary.csv),
[validation](evidence/block7_collapse_20261006/validation.json),
[protocol](new/collapse_audit/PROTOCOL.md).
Secondary: [provenance](evidence/block7_collapse_20261006/provenance.json),
[full matrix](evidence/block7_collapse_20261006/matrix.json.gz),
[full single-step records](evidence/block7_collapse_20261006/single_step.json.gz).

## Previous: matched reset64x4 versus continuous256 training

COMPLETE16/16 trajectories, eight fresh paired blocks,208 saved checkpoint
records;3439.86seconds. Architecture unchanged(C24/Z8,5033 parameters).
Each super-update uses the SAME batch8,256 steps,32 K8 backward windows,
32 losses/32 and one AdamW step. Reset reinitializes at64-step boundaries;
continuous carries state through256. No optimizer update occurs within a rollout.

|Fixed-u300 arm|Joint readiness|Unchanged old Full|
|---|---:|---:|
|reset64x4|0/8|0/8|
|continuous256|0/8|0/8|

Frozen verdict: `NO_CONTINUOUS_COVERAGE_RELIABILITY_QUALIFICATION`.
Joint net gain0/8, exact paired p1. Pooled strict reach delta-0.0709758;
retention delta+0.1171954 with4/8 positive pairs. Dense points describe finite
formation/decline and cannot replace u300. Encoder-credit frequency also changes
under reset. This negative recipe qualification is not a universal impossibility.

[Results](evidence/continuous_coverage_20261006/RESULTS.md),
[summary](evidence/continuous_coverage_20261006/summary.json),
[all checkpoint metrics](evidence/continuous_coverage_20261006/metrics.csv),
[validation](evidence/continuous_coverage_20261006/validation.json),
[protocol](new/continuous_coverage/PROTOCOL.md).

## Latest diagnostics: initialization geometry and continuation formation

Two completed CPU diagnostics with zero new training. Initialization uses four
exact historical update0 checkpoints and four reused map banks per seed;
formation uses44 saved traces from one selected historical training trajectory.
Seed4 has favorable raw lane balance and relative source-flip separation, but
no consistent task-tangent advantage. Early retention/progress passes at
updates140/145/190/200; the long behavior screen passes only200 in the dense
window. Predicting update u+5 produces0 TP,3 FP,16 TN,1 FN. No latent quotient
or training-time precursor is identified. Earlier architecture verdicts stand.

[Combined result](evidence/seed4_learning_geometry_20261004/RESULTS.md),
[initialization evidence](evidence/initial_geometry_20261004/RESULTS.md),
[formation evidence](evidence/formation_gate_20261004/RESULTS.md),
[interpretation](evidence/formation_gate_20261004/INTERPRETATION.md).

## Preceding diagnostic: joint195/200 zero-training causal audit

COMPLETE492 conditions,885.469 seconds; four full endpoint traces match
historical Boolean correctness and margins exactly. All arms remain finite.
Primary32 cross-continuation (equal-map strict T256):

| State at64 | Continuation | Readout | Strict T256 |
|---|---|---|---:|
| 195 | 195 | 195 | .93268 |
| 195 | 200 | 195 | .93333 |
| 200 | 195 | 195 | 1.00000 |
| 200 | 200 | 195 | 1.00000 |

E/F/Q/R full factorial effects are context dependent; readout replacement is
small. The primary32 interpolation jump is dominated by map10, while195
already preserves solved cells on the independent size32 bank. This supports
conditional state-production/coadaptation sensitivity; no population phase
transition or unique local semantic gate is established.

[Canonical report](evidence/joint195_200_20261003/RESULTS.md),
[interpretation](evidence/joint195_200_20261003/INTERPRETATION.md),
[all492 profiles](evidence/joint195_200_20261003/profiles.csv),
[frozen protocol](new/audit_195_200/PROTOCOL.md).

## Earlier follow-up: full seed4 behavior is not reproduced by sampled A/B changes

**COMPLETE,13 arms x300 updates,31.11 minutes.** Historical control exactly
reproduces initial/final parameters and six endpoint records and qualifies
on fresh32-map banks at both sizes. A full phenotype0/4 schedules; B0/4
directions at each of two paired radii (.01/.05). All 12 A/B arms pass
both matched-frontier gates but fail the complete behavior gate. Partial
reach/hold outcomes and the failed stability criteria are preserved.

C is an independent zero-training selected-checkpoint temporal rollback audit:
350 events,31 maps at each size. Its one-step joint threshold is not met;
four-step secondary effects are larger and do not rescue the primary.
Neither A/B nor C establishes a basin radius, unique semantic handoff or
general failure of cellular computation. Earlier architecture verdicts stand.

[Canonical report](evidence/seed4_followup_20261003/RESULTS.md),
[interpretation](evidence/seed4_followup_20261003/INTERPRETATION.md),
[all-arm aggregate](evidence/seed4_followup_20261003/summary.json),
[frozen protocol](new/seed4_followup/PROTOCOL.md), and
[publication bindings](SEED4_FOLLOWUP_PUBLICATION_MANIFEST.json).

## Preceding descriptive diagnostic: selected seed4 retains correctness while acquiring regions

**COMPLETE, zero training,31.813 seconds.** Original Streaming seed4,
historical32 maps per size32/64, every macro step0..256. All six full historical
T64/128/256 evaluation payloads replay exactly. Model n=1; this is descriptive
behavior, not a new architecture gate.

| Changed-component behavior | size32 | size64 |
|---|---:|---:|
| Correct-at64 retention at256 | 6856/6856 (100%) | 18827/19084 (98.65%) |
| Wrong at64, correct at256 | 1446 | 13275 |
| Ever-correct pixels with any one-step regression | 186/8303 (2.24%) | 2768/33328 (8.31%) |
| Never correct through256 | 392/8695 | 10860/44188 |
| Matched frontier one-step acquisition difference | +30.92 pp | +22.44 pp |

The frontier comparison matches map, time and exact BFS distance, weights
common strata within each map, then averages eligible maps equally. This
association does not establish neighbor causal handoff or flood-fill. Endpoint
retention also hides short regressions; terminal stability ends at256. The
maps are reused and the checkpoint selected. Earlier no-go verdicts remain.

[Report and figures](evidence/frontier_audit_seed4/RESULTS.md),
[diagnosis](evidence/frontier_audit_seed4/DIAGNOSIS.md),
[summary](evidence/frontier_audit_seed4/summary.json), and
[full saved-trace validation](evidence/frontier_audit_seed4/validation.json)
preserve the negative and censored observations. The local CPU validation
checked45 source bindings,64 maps/BFS,2240 transitions,123296 frontier strata
and192 paired aggregates. Public matched CSVs are independently reproducible
by `python tools/export_frontier_audit.py --verify-only`.
[Bindings](FRONTIER_AUDIT_PUBLICATION_MANIFEST.json) distinguish compact and
sanitized copies from untouched local evidence. No new model rollout for upload.

## Latest experiment: stationary sidecar does not improve reach+hold

**COMPLETE, DEVELOPMENT_NO_GO.** All12 arms completed300 updates in
1676.281 seconds (27.94 minutes). Original Streaming controls reproduce
historical final parameter hashes/full evaluations in4/4 seeds.

| Variant | Parameters | T64 reach | Reach+hold |
|---|---:|---:|---:|
| Original Streaming | 5033 | 1/4 | 1/4 |
| Persistent H sidecar | 7989 | 1/4 | 0/4 |
| Same-capacity stateless side branch | 7989 | 1/4 | 0/4 |

Every T64 reach is seed4; only original Streaming holds. Its size32 primary
pooled T64/T128/T256 is85.92/99.35/100%, versus memory94.24/0/0.38% and
stateless100/100/88.49%. The memory primary loses all2918 paired hits atT128.
Stateless drops to BA78.06/76.13% atT256; its paired metrics also fail hold.
Seed2 stateless improves from67.07% pooled atT64 to80.36/86.46% later, but
the frozen endpoint remains a failure. No later metric replaces the gate.

The old core is nested exactly at zero feedback, with all original modules,
operators and clocks retained. Memory/stateless differ only by extra H carry
and have identical parameters and initialization. Carry changes accumulation
and scale together, and additional gradients can change global clipping.
This negative recipe result does not reject local memory (old Z8 already
provides it), identify a failure mechanism, or establish population/3D claims.

Generated [full report](evidence/stationary_sidecar_init2345/RESULTS.md) and
[analysis](evidence/stationary_sidecar_init2345/analysis.json) retain all seeds,
rollout horizons, mean/pooled distinctions, failed hold checks and denominators.
[Validation](evidence/stationary_sidecar_init2345/validation.json) checks45
source bindings,12 CPU checkpoint hashes,4 exact controls,144 BA means,
1008 paired aggregates,936 curves and624 contrasts. The focused CPU
qualification was rerun; no new scientific training or trained-checkpoint
inference was launched for publication. [Bindings](STATIONARY_SIDECAR_PUBLICATION_MANIFEST.json)
distinguish unchanged raw evidence from sanitized metadata copies.

## Latest diagnostic: Streaming seed4 depends on both tested pathways

Zero-training fixed-checkpoint audit completed in12.36 seconds. All six
historical size32/64 by T64/128/256 evaluations replay with zero error.
Four cold-rollout conditions switch fixed T_M versus identity and both F/Q
Laplacian inputs versus zero tensors; all weights and data remain unchanged.

| Condition | Size32 primary pooled T64 / T128 / T256 % | Size64 d>32 pooled T256 % |
|---|---:|---:|
| Full Streaming seed4 | 85.92 / 99.35 / 100.00 | 59.43 |
| T_M replaced by identity | 0.00 / 0.00 / 0.00 | 0.00 |
| Direct Laplacian inputs removed | 0.00 / 0.00 / 0.00 | 0.00 |
| Both removed | 0.00 / 0.00 / 0.00 | 0.00 |

Each primary knockout result is0/2918 hits. Near-cue behavior also collapses;
the loss cannot be attributed solely to slower remote communication.
This selected frozen solution does not retain success under either knockout.
Interventions alter learned feature/state distributions and available hop
depth (2/2/1/0); they do not isolate training causality or establish that
restoring Laplacians would rescue H/C/Z. One checkpoint, n=1; no architecture
superiority, significance or retrained-knockout conclusion. Earlier gates
and the seed4 positive observation are unchanged.

Public canonical outputs: [report](evidence/stream_path_audit_seed4/RESULTS.md),
[interpretation](evidence/stream_path_audit_seed4/INTERPRETATION.md) and
[summary](evidence/stream_path_audit_seed4/summary.json). The independent saved-result validation
passed39 source bindings,672 paired aggregates/denominators,96 BA means,
672 curve rows and600 contrasts. Frozen protocol and implementation are
new/stream_path_audit/PROTOCOL.md and operators.py; analyze.py verifies saved
arithmetic without additional inference. [Validation](evidence/stream_path_audit_seed4/validation.json)
and [publication bindings](STREAM_PATH_AUDIT_PUBLICATION_MANIFEST.json) preserve
the distinction between original private run hashes and sanitized public files.

## Previous experiment: Persistent Roles fails the development gate

Execution **COMPLETE**:12 matched arms completed300 updates each in1204.234
seconds (20.07 minutes). Scientific decision **DEVELOPMENT_NO_GO**:
baseline reaches and holds in2/4 seeds, Streaming Carry in1/4 and Persistent
Roles in0/4. All eight control final parameter hashes and full evaluation
records exactly reproduce history.

Candidate primary size32/T64 strict16<d<32 pooled paired correctness for
seeds2/3/4/5 is0.00/0.31/0.00/1.58% (0/9/0/46 correct out of2918 pixels
per seed), versus the frozen80% threshold for BOTH mean and pooled metrics.
AtT256 it is0.00/0.58/0.00/2.09%. All candidate Hold=True values retain a
failed reach endpoint. Size64 d>32 peaks at65/27436 pooled hits (0.24%)
throughT256. Modest later gains cannot replace the frozen primary.

H12 stationary workspace, C12 persistent directional carrier and Z8 task
state are updated by shared R35->72->32, twice per macro-step. The candidate
matches5033 parameters and32 persistent scalars, but role allocation,
carrier width, perception, sharing and readout timing change together.
Its K8 carrier radius is<=16 and task-readout radius<=15. This is a negative
development result for this recipe, not an isolated H effect or a general
rejection of persistent local/communication state. Inspected seeds/maps,
one training bank/schedule and n=4 do not support population or3D claims.

The canonical per-seed table and integer denominators are generated from
saved records in the [full report](evidence/persistent_roles_init2345/RESULTS.md)
and [analysis](evidence/persistent_roles_init2345/analysis.json).
See [validation](evidence/persistent_roles_init2345/validation.json),
[completion](evidence/persistent_roles_init2345/completion.json),
[curves](evidence/persistent_roles_init2345/curves.csv),
[paired effects](evidence/persistent_roles_init2345/paired_effects.csv),
[protocol](new/persistent_roles/PROTOCOL.md) and
[source/evidence bindings](PERSISTENT_ROLES_PUBLICATION_MANIFEST.json).
The12 [raw arm records](evidence/persistent_roles_init2345/raw/) are secondary;
checkpoints, machine receipts and transient logs remain local.

## Previous: learned ephemeral Local Interface fails the development gate

Execution **COMPLETE**: all 12 matched arms completed 300 updates in 1281.532
seconds (21.36 minutes). Scientific decision **DEVELOPMENT_NO_GO**: baseline
reaches and holds in 2/4 seeds, Streaming Carry in 1/4, and Local Interface in
0/4. Every baseline and stream control exactly reproduces its historical final
parameter hash and complete evaluation payload.

The frozen primary is size32/T64, strict graph distance 16<d<32. Reach requires
both per-map mean and pooled paired correctness >=80%, with original and
flipped BA >=85%; hold also limits declines at both T128 and T256. Each
interface seed has 0.00% mean and 0.00% pooled paired correctness over 32 maps
and 2,918 eligible pixels. Original/flipped BA by seed is 72.22/67.53%,
58.82/56.43%, 76.55/71.44%, and 70.60/65.49%. Seed 4 has hold=True but
reach=False, so no interface seed passes the combined endpoint.

| Seed | Variant | Primary mean / pooled % | Reach / hold |
|---:|---|---:|---|
| 2 | Baseline | 93.70 / 92.97 | True / True |
| 2 | Streaming Carry | 1.08 / 1.47 | False / False |
| 2 | Local Interface | 0.00 / 0.00 | False / False |
| 3 | Baseline | 21.85 / 17.27 | False / False |
| 3 | Streaming Carry | 8.28 / 5.62 | False / False |
| 3 | Local Interface | 0.00 / 0.00 | False / False |
| 4 | Baseline | 52.40 / 44.76 | False / True |
| 4 | Streaming Carry | 88.89 / 85.92 | True / True |
| 4 | Local Interface | 0.00 / 0.00 | False / True |
| 5 | Baseline | 96.45 / 95.68 | True / True |
| 5 | Streaming Carry | 16.05 / 13.02 | False / True |
| 5 | Local Interface | 0.00 / 0.00 | False / False |

The candidate keeps W24 and Z8 at their spatial positions. A shared learned
encoder emits 24 transient message channels, six per N/E/S/W port; messages are
transported in two phases and discarded after each local update. The fixed
transport alone is lossless. The encoder, nonlinear residuals and repeated
receive/store/re-emit recurrence have no losslessness or stability guarantee.
The candidate changes state allocation, neighbor representation and hidden
widths together, so this comparison does not isolate an interface mechanism.
Seeds/maps were previously inspected, n=4 is conditional on one training
bank/schedule, and the result is 2D only. No family-wide or causal claim follows.

Read the [full report](evidence/local_interface_init2345/RESULTS.md),
[compact metrics](evidence/local_interface_init2345/analysis.json),
[validation](evidence/local_interface_init2345/validation.json),
[completion record](evidence/local_interface_init2345/completion.json),
[all curves](evidence/local_interface_init2345/curves.csv),
[paired effects](evidence/local_interface_init2345/paired_effects.csv),
[frozen protocol](new/local_interface/PROTOCOL.md), and
[publication bindings](LOCAL_INTERFACE_PUBLICATION_MANIFEST.json).
The sanitized raw arm records are available under
[raw/](evidence/local_interface_init2345/raw/); they are secondary to the
report and compact analysis.

## Previous: lossless Streaming Carry has one sustained positive case, but fails the gate

Execution **COMPLETE**: all8 matched arms completed300 updates in828.297 seconds
(13.80 minutes). Scientific decision **DEVELOPMENT_NO_GO**: baseline reaches
and holds in2/4 seeds; stream in1/4. The baseline reproduces all four Phase-II
final parameter hashes and complete evaluation records exactly.

W24 is now four directional6-channel lanes, with Z8 stationary local state.
Fixed T moves along open edges and reverses lanes at blocked links; wall
ports stay fixed. This is a norm-preserving permutation. F reads incoming
T(W), Z and old L(W)/L(Z); Q reads updated W. Parameters (5033), initialization,
data/schedule, K8 training and two-hop macro-step locality are matched.
This compares the full specified recipe, including the incoming F feature.

Primary size32/T64 STRICT16<d<32 paired correctness, mean / pooled percentages:

| Seed | Baseline | Stream | Stream reach and hold |
|---|---:|---:|---|
| 2 | 93.70 / 92.97 | 1.08 / 1.47 | Fail |
| 3 | 21.85 / 17.27 | 8.28 / 5.62 | Fail |
| 4 | 52.40 / 44.76 | 88.89 / 85.92 | Pass |
| 5 | 96.45 / 95.68 | 16.05 / 13.02 | Fail |

The gate requires both paired statistics>=80%, both original/flipped BA>=85%,
and bounded declines at BOTH T128/T256. Continuation additionally requires
stream reach+hold>=3/4, retention of positive seeds2/5 and a count exceeding
the concurrent baseline. Stream meets none of those continuation requirements.

Seed4 is a sustained positive case, unlike the previous average-carry
transient. Its size32 primary mean / pooled rises from88.89/85.92% atT64 to
99.64/99.35% atT128 and100/100% atT256. On size64, d>32 pooled accuracy rises
12.48 ->38.60 ->59.43%; per-map mean rises26.32 ->55.78 ->74.99%.
These measurements use different aggregations and do not imply general
large-grid qualification. Stream seeds2/3 deteriorate substantially byT256;
seed5 remains weak. Both original successful baseline seeds2/5 are lost.

This rejects this matched streaming recipe as a robust improvement under
the frozen development criterion. It preserves evidence that the recipe
can learn useful sustained propagation in one initialization. Pure transport
isometry does not certify task-signal preservation or whole-cell stability;
the experiment does not identify the failure mechanism or refute every
lossless transport architecture. Seeds/maps were previously inspected,
n=4 is conditional on one training bank/schedule, and this is2D only.

Peak allocated memory baseline95.12MiB, stream97.08MiB. Systems measurements
are descriptive, without an accuracy-matched efficiency claim. No extra
training or checkpoint inference was performed for publication. No automatic
confirmation or rescue experiment is scheduled. Earlier results are unchanged.

[Full report](evidence/streaming_carry_init2345/RESULTS.md),
[compact metrics](evidence/streaming_carry_init2345/analysis.json),
[all curves](evidence/streaming_carry_init2345/curves.csv),
[validation](evidence/streaming_carry_init2345/validation.json),
[frozen protocol](new/streaming_carry/PROTOCOL.md), and
[publication bindings](STREAMING_CARRY_PUBLICATION_MANIFEST.json).

## Previous: fixed Direct Spatial Carry fails the development gate

**DEVELOPMENT_NO_GO**. All8 matched runs completed300 updates in797.797 seconds
(13.30 minutes). Baseline reach+hold is2/4; carry is0/4. Every baseline final
parameter hash and full evaluation record exactly reproduces Phase II.

Only the W skip path changes from identity to W-0.5*D_M^dagger*L_M(W).
F/Q, additive Z, initialization,5033 parameters, data/schedule and K8 training
are matched. Primary size32/T64 is strict16<d<32, with paired correctness
required under BOTH source alternatives. Mean / pooled percentages:

| Seed | Baseline | Carry |
|---|---:|---:|
| 2 | 93.70 / 92.97 | 0.00 / 0.00 |
| 3 | 21.85 / 17.27 | 2.87 / 3.12 |
| 4 | 52.40 / 44.76 | 14.39 / 11.41 |
| 5 | 96.45 / 95.68 | 7.59 / 4.42 |

Carry loses the original successful seeds2/5 and rescues neither3 nor4.
At size64/T256 its d>32 paired correctness is0% in all four seeds. A meaningful
transient exception is seed4 atT128: strict16<d<32 pooled accuracy is74.64%
versus baseline43.59% at size32, and82.54% versus37.06% at size64. ByT256 those
carry values fall to19.50% and26.25%, below their concurrent baselines. This
is delayed, transient narrow performance, not sustained rescue; it cannot
replace the fixed T64 endpoint. Size64/T128 d>32 pooled is only0.70% versus0%
for this seed. Carry seed2's
hold=True is stability of failure: its primary stays at0%, not task retention.

This is a negative result for this fixed-average parameterization. Replacing
half of local retention with neighbor mixing does not isolate a failure
mechanism or refute all spatial carry. Previously inspected seeds/maps make
this development, conditional on one train bank and schedule. Do not advance
the recipe to confirmation or interpret whole-grid BA as far paired accuracy.
Earlier scientific conclusions remain unchanged.

CPU verification:29 source snapshots,8 checkpoints,96 BA aggregates,
672 paired aggregates with reconstructed denominators,8 arm decisions,
312 contrasts and936 CSV rows. No new training or checkpoint inference.
Peak allocated memory is95.12MiB baseline versus96.08MiB carry.

[Detailed report](evidence/direct_spatial_carry_init2345/RESULTS.md),
[compact metrics](evidence/direct_spatial_carry_init2345/analysis.json),
[all contrasts](evidence/direct_spatial_carry_init2345/paired_effects.csv),
[protocol](new/direct_spatial_carry/PROTOCOL.md), and
[publication hashes](DIRECT_CARRY_PUBLICATION_MANIFEST.json).

## Previous: Phase-II initialization replication misses its K8 threshold

Execution: **COMPLETE**, 12 arms at 300 updates each, 1095.047 seconds.
**K8 narrow reach and sustained replication are both `NOT_REPLICATED`: 2/4
seeds pass, below the frozen 3/4 criterion.** All four K64 controls remain
unqualified, so the comparison is separately `BASELINE_UNQUALIFIED`.

| Gradient window | Narrow reach | Narrow reach and hold |
|---|---:|---:|
| K8 | 2/4 | 2/4 |
| K16 | 3/4 | 1/4 |
| K64 | 0/4 | 0/4 |

Only additive was tested. Initialization seeds2/3/4/5 share training bank10002
and schedule20002. New evaluation banks40032/40064 contain32 maps per size.
Primary size32/T64 uses **strict16<d<32**, with both per-map and pooled paired
accuracy>=80% and both original/flipped BA>=85%. Hold limits the decline at
T128 and T256. This selected narrow endpoint was frozen prospectively after
Phase I; it does not replace or pass Phase I's broader d>16 gate.

Primary paired percentages (per-map mean / pooled pixels):

| Initialization seed | K8 | K16 | K64 |
|---|---:|---:|---:|
| 2 | 93.70 / 92.97 | 99.06 / 98.83 | 0.16 / 0.14 |
| 3 | 21.85 / 17.27 | 90.19 / 87.77 | 76.44 / 70.56 |
| 4 | 52.40 / 44.76 | 57.56 / 51.75 | 0.00 / 0.00 |
| 5 | 96.45 / 95.68 | 92.90 / 90.47 | 12.88 / 9.49 |

K8 seeds2/5 remain positive examples: primary pooled accuracy is93.21/93.59%
at size32/T256 and89.65/97.25% at size64/T256. Yet d>32 pooled accuracy on
size64/T256 is only36.78/14.83%. Narrow propagation transfers better than broad
far propagation. Size32 has no changed-component pixels at distance>=64 in
this bank; null entries must not be read as zero accuracy.

K16's primary band lies within its single-window radius32, so its3/4 reach
does not prove cross-window composition. Only seed2 also holds. K8 is worse
than K64 on seed3; there is no general superiority or optimal-K conclusion.
All K8 clipping rates are<=6%, including both reach failures; low clipping is
not sufficient for success. K16 seed2 clips60% and still reaches and holds.
The clipping observation does not identify a causal mechanism.

Peak allocated CUDA memory K8/K16/K64 is95.12/155.81/519.41MiB; per-arm
training times are84.28-92.86 seconds. These are descriptive systems numbers,
not activation-only memory or an accuracy-matched speedup result.
Verification covered18 source snapshots,12 checkpoints, shared identities,
144 BA aggregates,1008 paired aggregates/denominators and all12 arm decisions.
No extra training or inference. n=4 is conditional on one training bank and
schedule, without a significance or cross-distribution robustness claim.

[Full report](evidence/short_bptt_phase2_init2345/RESULTS.md),
[compact metrics](evidence/short_bptt_phase2_init2345/analysis.json),
[all curve rows](evidence/short_bptt_phase2_init2345/curves.csv),
[frozen protocol](new/short_bptt_phase2/PROTOCOL.md), and
[publication hashes](BPTT_PHASE2_PUBLICATION_MANIFEST.json).

## Previous: matched short-BPTT Phase I, one positive case with unqualified controls

Execution: **COMPLETE**, eight arms at 300 updates each, 730.797 seconds.
Scientific status: **BASELINE_UNQUALIFIED for both architectures**; all four
full-K64 controls fail the predeclared far-paired threshold. This prevents a
qualified comparative verdict, not publication of the observed outcomes.

Primary size32/T64 percentages; far means graph distance >16. Paired correctness
requires the same pixel to be correct under both fresh source alternatives.
The statistic averages eligible per-map fractions, with model seed the
independent unit (n=2 per architecture).

| Cell | Seed | K64 BA | K8 BA | K64 far paired | K8 far paired | K8 reach / hold predicates |
|---|---:|---:|---:|---:|---:|---|
| Additive | 0 | 87.20 | 91.22 | 15.83 | 23.18 | Fail / Fail |
| Additive | 1 | 87.73 | 98.03 | 0.00 | 80.87 | Pass / Pass |
| Revision | 0 | 78.04 | 83.90 | 21.44 | 0.67 | Fail / Fail |
| Revision | 1 | 86.16 | 64.38 | 0.00 | 17.81 | Fail / Fail |

Additive K8 seed1 retains 79.20%/78.24% far paired at T128/T256. This is an
example of computation beyond the single-window dependency radius (two
communication phases per macro-step, hence radius16 for K8). It is not an
architecture-level pass: the other short runs fail reach, all full controls
are unqualified, and **80.87% per-map mean corresponds to only 59.53% pooled
far-pixel accuracy**. The same seed scores 45.60% in distance [32,64), and
0% at distance >=64 on size32/T64 (the latter has only one eligible map).
Size64 far paired is 28.40% at T64 and 41.51% at T256. Reliable long-distance
and size extrapolation are not established.

Peak allocated CUDA memory is 514.04 MiB for K64 versus 90.49 MiB for K8
across both cells and seeds, **82.4% lower**. This includes data, optimizer
and gradient storage. Training times are similar (86-96 seconds per arm).
Additive K64 clips 88.67%/90.67% of updates versus 6.67%/6.67% for K8;
this does not establish clipping as the cause of full-control failure.

Forward trajectories, eight equally weighted losses and one optimizer update
per trajectory are matched. The new objective and runtime-selected 300-update
budget cannot borrow qualification from the older 600-update recipe.
CPU publication analysis verified executed sources, parameter/data/schedule
hashes, 96 BA aggregates, 336 paired aggregates and all four pair decisions.
No extra training or inference, rescue tuning, 3D test, warm source-switch
test or significance claim is included.

Read [full report](evidence/short_bptt_paired01/RESULTS.md),
[compact analysis](evidence/short_bptt_paired01/analysis.json),
[figure](evidence/short_bptt_paired01/overview.png),
[frozen protocol](new/short_bptt/PROTOCOL.md), and
[publication hashes](BPTT_PUBLICATION_MANIFEST.json).

## Previous: candidate/workspace source-switch audit, zero training

Revision seed0, sizes32/64,16 fixed maps each; completed in9.609 seconds. All232
saved replay comparisons and candidate-step reconstruction have zero error.
Warm candidate readout itself is old-aligned: size32 new-target accuracy is0%
at K64/K128; size64 is2.09%/1.63%. Slow averaging of an otherwise correct
candidate is therefore not the observed failure.

Full-rollout changed-component accuracy at K128 (%):

| Intervention at old T64 | Size32 | Size64 |
|---|---:|---:|
| Retain W and Z, switch input | 0.00 | 1.64 |
| Reset W to new-input encoding, retain Z | 37.50 | 43.36 |
| Reset Z to zero, retain W | 37.50 | 37.17 |
| Transplant W from new-input T64, retain Z | 37.50 | 45.39 |
| Transplant Z from new-input T64, retain W | 37.50 | 45.40 |
| Cold reset of both | 100.00 | 99.49 |

At size32/K128, **all four single-block interventions predict negative on every
changed-component pixel**;37.5% equals the six negative labels among16 maps.
This is fixed-label behavior, not partial recovery. Resetting Z also harms
unchanged-component accuracy (77.57%/61.70% at sizes32/64). Candidate-only
cross-interventions at fixed new X show dependence on both W and Z and a
nonadditive response. Donors from different trajectories can be off-distribution;
the result does not prove a W-only mechanism or that both blocks must generally
be reset. Mature donors include extra computation and are diagnostic controls.

The earlier `NO_JOINT_SCREEN_PASS` remains unchanged. No seed1 training-failure
diagnosis, short-BPTT test, new architecture or retraining is included.
Read [audit report](evidence/switch_audit_seed0/RESULTS.md),
[figure](evidence/switch_audit_seed0/overview.png),
[compact metrics](evidence/switch_audit_seed0/analysis.json),
[exact replay scope and contrasts](new/switch_audit/REVIEW_NOTES.md), and
[publication hashes](SWITCH_PUBLICATION_MANIFEST.json).

## Previous: Workspace + Revision paired screen fails the joint gate

Execution: **COMPLETE**, all four arms at 600 updates, 1112.28 seconds.
Scientific decision: **`NO_JOINT_SCREEN_PASS`**. Primary size32 hold is minimum
aggregate BA across T128/T192/T256; the mean paired revision-minus-additive
effect is **-9.02 pp** (required >=+5 pp and positive in both seeds).

| Seed | Additive BA64 (%) | Revision BA64 (%) | Additive hold (%) | Revision hold (%) | Hold effect (pp) | Revision warm changed K64 (%) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 94.29 | 99.19 | 94.18 | 100.00 | +5.82 | 0.00 |
| 1 | 78.54 | 50.00 | 73.86 | 50.00 | -23.86 | 37.50 |

Seed 0 supplies a real reach/hold observation: revision BA is 100% at measured
T96 through T256 on size32, and 99.44% at size64/T256. Yet after a source switch,
size32 changed-component accuracy is 0% at K64 and K128 versus 100% for a cold
restart; unchanged-component accuracy stays 100% at K64. **Holding an answer
has not become revising an answer.** Seed 1 never qualifies, so the result does
not support a reliable revision advantage. Two model seeds are the independent
units; 16 shared evaluation maps per size are not additional training replicates.

Seed 0's size32 Z-only repair reaches 100%, but W/Z joint repair only 90.51%
after 64 steps; at size64 these are 96.91% and 80.42%. W RMS still grows from
2.942 to 9.718 at size32/T64 to T256. Thus neither general regeneration nor
full-state stability is established. Seed 1 has no eligible clean95 maps for
conditional repair, so its null conditionals must not be presented as recovery.

Both arms have 5033 parameters, W24/Z8 and two communication phases per step,
with matching initial parameters, data and schedules within each seed. The
only intervention is additive `Z+0.5Q` versus revision `Z+0.5(Q-Z)`. This tests
the rule within a shared workspace split, not the value of the split itself.
Older State/Momentum recipes are not matched controls for this new training.
The preflight-based 800-to-600 amendment occurred before efficacy training.

Read the [interpretation](evidence/workspace_revision_paired01/INTERPRETATION.md),
[compact metrics and failed predicates](evidence/workspace_revision_paired01/analysis.json),
[overview](evidence/workspace_revision_paired01/overview.png) and
[frozen protocol](new/workspace_revision/PROTOCOL.md). The
[validation](evidence/workspace_revision_paired01/validation.json) independently
recomputes 400 BA aggregates and all 22 seed-level gate predicates.
[Publication hashes](REVISION_PUBLICATION_MANIFEST.json) bind executed sources,
raw records and schedules. No new training or GPU rollout was run for publication.

## Previous: completed zero-training generic dynamics audit

Four existing generic State/Momentum checkpoints, whole-grid and masked media,
sizes32/64, eight anchors and windows1/8/16. Completed in112.859 seconds, without
training. All40 historical replay comparisons have zero error; source and
checkpoint hashes match. Performance curves use16 maps; Jacobian diagnostics
use fixed maps0..3. One trained seed remains the independent replicate.

**No near-zero-to-positive crossing of maximum finite-window gain is observed.**
All256 open K16 estimates are positive, already large before the performance
peak, and generally decline while late performance deteriorates. This does
not imply improving stability or exclude task-specific neutral modes, longer
product instability or behavior in unsampled windows.

Masked Momentum, size32; g16=log(sigma_estimate)/16 for the window starting at T:

| T | BA %, all16 | BCE | Paired % | Median g16, maps0..3 | Converged /4 |
|---|---:|---:|---:|---:|---:|
| 32 | 91.47 | 0.1598 | 49.82 | 0.70763 | 3 |
| 64 | 96.45 | 0.0512 | 77.25 | 0.27742 | 3 |
| 128 | 86.17 | 0.9152 | 53.28 | 0.16949 | 2 |
| 256 | 77.68 | 6.8694 | 42.78 | 0.17582 | 4 |

The four diagnostic maps themselves also decline: BA97.14%,86.40%,77.87% at
T64,T128,T256. Masked State shows BA90.43% to11.38% from T64 toT256 while
median g16 decreases0.14666 to0.10359. Short-window maximal gain is therefore
not a sufficient account of the observed performance trend.

Persistent motion and growing wrong confidence are directly observed. For
Masked Momentum at32, mean per-map open H RMS grows42.57 to258.87 from T64
toT256, while V RMS remains approximately1.1. The median source-flip logit RMS
on the changed component grows25.26 to63.75, despite declining paired
correctness. A growing response to source identity is not necessarily useful
task information. The component-constant H-energy fraction falls1.79% to0.20%
(mean of per-map fractions), so the earlier explicit-inertial mean-drift account
must not be transferred wholesale to generic Momentum.

**Diagnostic limits:** only562/1536 product estimates converge by the frozen
criteria;148/256 converge in the main open K16 subset. Unconverged values are
not certified maxima or upper bounds. At relative RMS perturbation1e-4,
worst-direction linearization error has median46.44%, versus1.19% for random
directions. Large infinitesimal gains often fail to quantitatively predict
these finite perturbations. Free momentum alone yields positive finite-window
gain (masked K16 sigma8.0003, g16=0.12997). None of these measurements is an
asymptotic Lyapunov exponent or proof of a causal failure mechanism.

The primary same-medium Momentum advantage is unchanged. No new stability
parameterization is validated. Read [full interpretation](evidence/dynamics_audit_seed0/INTERPRETATION.md),
[all64 table rows](evidence/dynamics_audit_seed0/RESULTS.md),
[compact metrics](evidence/dynamics_audit_seed0/analysis.json), and
[figure](evidence/dynamics_audit_seed0/dynamics_overview.png) first.
[Validation](evidence/dynamics_audit_seed0/validation.json) and
[hash bindings](DYNAMICS_PUBLICATION_MANIFEST.json) establish provenance.
The16 byte-identical [raw files](evidence/dynamics_audit_seed0/raw/) retain
per-map power iterations, perturbations, source tangents and block diagnostics.
No additional experiment or monitor is scheduled.

## Final: generic recurrence by medium, 2D seed 0

The last authorized arm, Masked State-matched NCA, saved all 800 training updates
and all 15 scientific evaluation points. The original process did not finalize;
only missing size128 timings and initial-state gradient probes were subsequently
recovered from its checkpoint, with no retraining. The 32/T64 and 128/T256 BA,
BCE and H RMS replay exactly. Exit cause is unknown; original records remain
unchanged. [Recovery](evidence/masked_state_seed0/recovery.json) and
[validation](evidence/masked_state_seed0/validation.json) document this distinction.

Frozen primary endpoint: 32x32/T64; one training seed, 16 held-out maps.
BA averages class recall within each map then averages maps; paired correctness
pools changed-component pixels that are correct under both original and flipped
source labels. Neither maps nor pixels are independent training replicates.

| Generic recipe | Whole-grid BA (%) | Masked BA (%) | Whole-grid paired (%) | Masked paired (%) |
|---|---:|---:|---:|---:|
| State-matched NCA | 85.17 | 90.43 | 49.31 | 56.37 |
| Momentum NCA | 84.44 | 96.45 | 42.75 | 77.25 |

| Primary contrast | BA difference (pp) | Paired difference (pp) |
|---|---:|---:|
| Masked Momentum minus Masked State | +6.02 | +20.88 |
| Masking effect, State | +5.26 | +7.06 |
| Masking effect, Momentum | +12.01 | +34.49 |
| Difference-in-differences | +6.75 | +27.43 |

Outcome: `MOMENTUM_JOINT_5PP_ADVANTAGE`, a descriptive threshold specified before
the new arm ran, not statistical significance. The other three cells were already
known when the [final protocol](new/masked_state/PROTOCOL.md) was written.
Within this seed, both generic recipes benefit from the task-respecting medium
on the primary metrics, and Momentum retains an advantage on that same medium.
This neither makes the medium sufficient nor shows that velocity alone causes
the advantage. State uses H32/hidden48/4993 parameters, Momentum H16+V16/hidden88/
4689 parameters: equal persistent scalars, differing widths and 6.48% more
parameters for State. Masking also changes connectivity, degree and spectrum.

The negative secondary findings are substantial. At 32/T64, State masking
improves BA while worsening BCE from 0.2579 to 0.9871. With longer rollout:

| Size | Masked State BA T64 to T256 (%) | Masked Momentum BA T64 to T256 (%) |
|---|---:|---:|
| 32 | 90.43 to 11.38 | 96.45 to 77.68 |
| 64 | 75.93 to 25.93 | 84.75 to 64.15 |
| 128 | 63.93 to 38.59 | 75.20 to 51.90 |

At 32/T256, Masked State paired correctness falls to 9.38%, BCE reaches 63.4853,
and H RMS rises from 5.99 at T64 to 23.87. Its 87.38% training clip fraction
and initial-state task-gradient norm of 9.06e8 at T64 flag sensitivity, but the
single-example loss-gradient probe is not a Jacobian spectral norm or a proof
of asymptotic instability. No sustained 95% endpoint is reached. Conditional
repair cohorts differ (State 5 eligible maps versus Momentum 12 at size32).
Timings span separate runs, including a separate recovery process for size128;
no controlled speedup or conditional-repair superiority is established.

Read [final generated results](evidence/masked_state_seed0/RESULTS.md),
[all 15 comparisons](evidence/masked_state_seed0/comparison.json), then
[publication hashes](STATE_PUBLICATION_MANIFEST.json). The
[original arm](evidence/masked_state_seed0/original_arm.json) is byte-identical
to the saved run; the [augmented arm](evidence/masked_state_seed0/arms/masked_state_nca_seed0.json)
only adds the documented diagnostics. Earlier results below remain intact.
This completes the final authorized comparison. No further experiment is scheduled.

## Previous: same-medium generic momentum control and trajectory audit

Masked Momentum completed800 updates and all three evaluation sizes on2026-10-01.
Exact initial parameters/RNG match the old generic control; logged rollout and
damage schedules match both references. All evaluation statuses are valid.

Primary endpoint: size32, T64, one model seed and16 held-out maps:

| Model | BA (%) | Paired-source correctness (%) | BCE |
|---|---:|---:|---:|
| Masked Momentum | 96.45 | 77.25 | 0.0512 |
| Masked Inertial RD | 88.79 | 37.22 | 0.2503 |
| Original unmasked Momentum | 84.44 | 42.75 | 0.3217 |

Masked Momentum beats the explicit candidate by7.66pp BA and40.02pp paired
correctness, satisfying the prespecified descriptive reverse joint5pp criterion.
Relative to the original generic control, masking gains12.01pp and34.49pp.
This is a negative primary result for explicit-factorization superiority in
this seed/recipe; it is not a multi-seed or isolated causal factorization result.

Long rollout reverses the BA ranking at sizes32 and64:

| Size | Momentum BA T64 → T256 (%) | Inertial RD BA T64 → T256 (%) |
|---|---:|---:|
| 32 | 96.45 → 77.68 | 88.79 → 91.55 |
| 64 | 84.75 → 64.15 | 75.39 → 75.69 |
| 128 | 75.20 → 51.90 | 57.47 → 50.00 |

At32/T256, paired correctness is42.78% for Momentum versus41.92% for Inertial
RD, despite the opposite BA ranking. BA alone is insufficient to rank source
information retention. Momentum BCE rises0.0512 to6.8694 and H RMS36.33 to219.81
fromT64 toT256. Both models lack a sustained95% endpoint; a single96.45% peak
does not satisfy it. Momentum's changed-component accuracy after a source switch
and64 extra steps is29.04% at32. Conditional repair cohorts differ (12 versus7
eligible maps), so their conditional percentages are not a matched repair comparison.

Read [full control results](evidence/masked_momentum_seed0/RESULTS.md) and
[all comparisons](evidence/masked_momentum_seed0/comparison.json). The separate
[candidate audit](evidence/trajectory_audit_seed0/INTERPRETATION.md) finds open
state growth, increasingly confident errors and component-mean drift. It supports
a finite-horizon amplification interpretation, not a stable attractor. Independent
temperature fitting does not change decisions or fix the128-scale failure.
No corresponding detailed trajectory audit of generic Momentum was performed.

No automatic retuning, extra seeds, new architecture or follow-up run is scheduled.

## Previous follow-up: masked medium, 2D seed 0

**Interpretation correction:** improved classification accuracy with longer
rollout does not establish a stable attractor or fixed-point convergence.
For masked inertial RD at32, T64/128/256 BA is88.79/89.96/91.55%, while
BCE rises0.250/0.657/1.452, H RMS rises17.19/50.88/126.95 and V RMS
rises0.483/0.575/0.617. The existing RMS includes walls; BCE uses only open
pixels. These observations warrant an inference-only trajectory audit, not
an assertion of exponential divergence or calibrated convergence. Conditional
decision recovery is distinct from return to a stable hidden state. Source
revision failure does not by itself identify a deep attractor basin.

The separately authorized [trajectory audit](evidence/trajectory_audit_seed0/INTERPRETATION.md)
is complete, with exact replay of all15 stored checkpoints. Open-pixel H RMS
at32 rises14.48 to120.61 fromT64 toT256; wrong-margin median changes -1.889
to -18.421. Independently calibrated test BCE atT256 is0.2492 versus raw1.4519,
with unchanged BA. Componentwise-constant H energy is97.04% at32/T256, but
also99.51% at128/T256 where BA is50%. These support finite-horizon amplification
and mean drift, not convergence or correct propagation throughout the domain.
The completed same-medium generic momentum comparison is reported above.

Both masked arms completed 800 updates with matched initialization, RNG and
logged rollout/damage schedules relative to their frozen unmasked controls.
Only the structured transport operator changes to input-mask edge weights.

| Model | BA at 32x32/T64, old → masked | Paired correctness, old → masked | Prespecified descriptive outcome |
|---|---:|---:|---|
| RD | 81.27% → 73.21% | 37.81% → 17.13% | NO_JOINT_5PP_GAIN |
| Inertial RD | 69.94% → 88.79% | 17.81% → 37.22% | JOINT_GAIN_5PP |

At T256, masked inertial RD reaches 91.55% BA at size 32, 75.69% at size 64,
and 50.00% at size 128. No sustained aggregate 95% threshold is reached.
This supports an operator-dependent improvement within the tested inertial
recipe, with unresolved scale generalization. The opposite RD effect prevents
a blanket claim that masking helps both dynamics. One seed is not statistical
significance; no same-medium momentum comparison or architecture superiority
claim is established. Repair eligibility counts alone do not prove repair.

Read [compact results](evidence/masked_medium_seed0/RESULTS.md),
[all paired horizons](evidence/masked_medium_seed0/paired_comparison.json), and
[publication manifest](MASKED_PUBLICATION_MANIFEST.json). Detailed per-arm JSON
under `evidence/masked_medium_seed0/arms/` is secondary. Original evidence below
remains unchanged.

## Previous: inertial NCA, 2D seed 0

**Completed, negative exploratory screen for the current recipe.** Four arms
completed 800 updates and all planned evaluations in approximately 7 minutes
38 seconds. No numerical-failure or budget-limit status was recorded. Read the
[new per-arm tables](evidence/inertial_seed0/RESULTS.md),
[summary](evidence/inertial_seed0/summary.json) and
[configuration](evidence/inertial_seed0/config.json) first.

Balanced accuracy (%), 16 held-out maps at each size, one training seed:

| Model | 32x32, T64 | 32x32, T256 | 64x64, T256 | 128x128, T256 |
|---|---:|---:|---:|---:|
| State-matched NCA | 85.17 | 82.85 | 73.03 | 55.54 |
| Momentum NCA | 84.44 | 73.38 | 74.47 | 67.18 |
| RD NCA | 81.27 | 74.79 | 56.89 | 50.33 |
| Inertial RD | 69.94 | 53.50 | 50.00 | 50.00 |

The candidate already underperforms at the training size and within the training
rollout range. At 32x32 its accuracy drops from 72.89% at T32 to 53.50% at T256.
At 64x64 and 128x128, paired-source correctness at T256 is zero. The tested
reaction/transport parameterization shows no benefit over generic momentum.

No arm reaches sustained aggregate 95% BA, so matched-quality time-to-solution
is null. Inertial RD has zero pre-damage eligible maps at all sizes: its main
conditional-repair endpoint is **unevaluable**, not a measured zero repair rate.
Other models learn partial structure, but this recipe has not established a
robust high-quality reference solution. Gradient clipping and latency are recorded;
they do not establish trainability or systems superiority.

This is one exploratory training seed, not a significance claim, an NCA/PDE-family
refutation, or a 3D result. It is a different task and operator from A0; their
accuracy values must not be compared as successive versions of one benchmark.
No follow-up seed, new architecture or experiment was launched after this result.

[Validation](evidence/inertial_seed0/validation.json) and
[linear checks](evidence/inertial_seed0/linear_checks.json) establish implementation
checks only. [INERTIAL_PUBLICATION_MANIFEST.json](INERTIAL_PUBLICATION_MANIFEST.json)
binds the frozen executed sources and byte-identical per-arm output files.

## Earlier A0 experiment

Earlier protocol: [rt_a0_2d3d_v2_1](A0_PROTOCOL.md). Read the
[compact summary](evidence/a0_v2_1/summary.json),
[full per-seed table](evidence/a0_v2_1/RESULTS.md) and
[configuration](evidence/a0_v2_1/config.json) first. The
[raw aggregate](evidence/a0_v2_1/aggregate.json) retains histories, axis metrics
and diagnostics. [A0_PUBLICATION_MANIFEST.json](A0_PUBLICATION_MANIFEST.json)
binds current code and evidence to the run.

## Final A0 qualification

The conditional schedule completed 24/24 trials: eight attention trials across
two dimensions/four seeds, then sixteen 2D RT trials. Sixteen potential 3D RT
trials were skipped after failed calibration. This is a completed schedule,
not an unfinished 24/40 run or budget exhaustion.

| Dimension | Attention control | RT outcome |
|---|---|---|
| 2D | 4/4 seeds reach 100% in every evaluation condition | All four arms NOT_QUALIFIED_FIT |
| 3D | Seed 1729 has 47.92% train d16; other three seeds reach 100% | INCONCLUSIVE_POSITIVE_CONTROL; RT not run |

2D accuracy (%), seeds 1729 / 2718 / 31415 / 57721:

| Arm | Train-shape d16 | d128 |
|---|---:|---:|
| Attention | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| Constant raw | 50.78 / 46.09 / 55.47 / 43.75 | 50.78 / 45.31 / 54.69 / 46.09 |
| Constant normalized | 50.78 / 50.78 / 55.47 / 43.75 | 50.78 / 47.66 / 55.47 / 46.88 |
| Learned raw | 50.78 / 46.09 / 55.47 / 43.75 | 50.00 / 44.53 / 54.69 / 46.09 |
| Learned normalized | 50.78 / 45.31 / 55.47 / 44.53 | 50.78 / 45.31 / 56.25 / 45.31 |

There is no fitted RT arm whose distance generalization can be qualified.
Confidence normalization does not resolve learning failure in this frozen
configuration. The reliable 2D control makes this a clearer negative result
than v1, but it does not establish an architecture-family impossibility or
identify emission versus reaction/readout as the cause.

## Paired A0 normalization effect

Mean of four paired seed differences, normalized minus raw, percentage points:

| Medium | Train d16 | Large-grid d16 | d32 | d64 | d128 |
|---|---:|---:|---:|---:|---:|
| Constant | +1.172 | -0.586 | +1.953 | -1.953 | +0.977 |
| Learned | +0.000 | -0.195 | +0.977 | -0.977 | +0.586 |

At d128, constant-medium seed differences are 0.000 / +2.344 / +0.781 / +0.781
points; learned-medium differences are +0.781 / +0.781 / +1.562 / -0.781 points.
Small positive averages at one distance do not establish useful communication
when both arms remain near chance and fail fitting. Four seeds are replicate
units; these are descriptive estimates, not significance or equivalence claims.

Both arms use q=c*v, identical initial parameters within each pair, unit
initial conductance and matched packed solver work. Only receiver division
changes within a pair. A0 raw differs from v1 raw. Changes between experiments
prevent attributing improved attention calibration to any single intervention.

## A0 diagnostics and cost

[Checks](evidence/a0_v2_1/checks.json) passed for grouped packing, normalized
q/confidence/edge/tau gradients, supplied-value attention, paired initialization,
initial medium matching and finite model gradients in both dimensions.

In the fixed oracle cases, tau4096 permits correct raw-sign decoding through
d128 on every tested axis. At d128, normalized value error is at most 0.00445;
at tau1, all 40 sampled bits become undecodable zeros. This is recoverability in
the tested large-scale oracle setting, not learned communication at every scale.

The normalized arms' sampled final-update d128 probes retain mean background
confidence about 0.34-0.48, versus source confidence about 0.47-0.53. Thus the
source-only oracle confidence assumption is not realized in those probes.
None of their sampled target denominators is <=epsilon. Confidence-mass share
is not a measure of total bit information: earlier recurrent updates can move
content into values emitted elsewhere. These observations do not isolate cause.

| 2D arm | Mean of seed-median batch-one d128 latency, ms |
|---|---:|
| Attention | 22.13 |
| Constant raw / normalized | 35.90 / 37.20 |
| Learned raw / normalized | 36.58 / 37.33 |

A0 includes a finite-logit evaluation check and matched packed solver work;
cross-version timings are not kernel improvement/regression estimates. These
are portable reference implementations, not optimized ceilings. A0 has no
speed-pass gate. Models are not parameter-matched; exact counts are in the
summary. Narrow straight geometry and fixed axis splitting remain limitations.

Decision: do not escalate this configuration to B/C, rank sweeps or real vision.
Any further experiment needs a distinct bounded diagnostic. No new experiment
or monitor is scheduled by this publication. 3D normalization remains unmeasured.

## Historical first local 2D / 3D result

Canonical evidence: [per-seed results](evidence/qualification_v1_1/RESULTS.md),
[raw aggregate](evidence/qualification_v1_1/aggregate.json), and
[configuration](evidence/qualification_v1_1/config.json).

Published evidence is an allowlisted copy of the frozen local run. Its original
training-source hashes are recorded in `PUBLICATION_MANIFEST.json`.

Twenty Gate A model/seed/dimension runs completed on the local RTX 4060 Laptop
GPU in approximately 4.32 minutes. Each used 240 training updates, C=64, r=32,
eight recurrent updates, training distances <=16, and tests through d=128.
3D uses actual three-axis transport/Conv3d on a narrow 4x4 transverse volume,
rotated across all three axes. It is not general 3D geometry qualification.

### Historical v1 accuracy at distance 128

Values are the separate training seeds 1729 / 2718, in percent.

| Model | 2D | 3D |
|---|---:|---:|
| CNN | 55.5 / 49.2 | 52.1 / 50.5 |
| NCA | 53.1 / 49.2 | 51.6 / 53.1 |
| Attention control | 100.0 / 49.2 | 100.0 / 48.4 |
| Constant transport | 57.8 / 49.2 | 52.1 / 51.0 |
| Learned transport | 52.3 / 49.2 | 51.6 / 49.5 |

Models are width/depth-matched, not parameter-matched. CNN has 274,370 / 283,650
parameters in 2D / 3D, versus 74,402 / 76,770 for learned transport. The attention
control is a shared SDPA message variant rather than a tuned standard ViT.

Learned transport also fails to fit the training-scale d16 evaluation
(2D 54.7 / 49.2; 3D 51.0 / 47.9). Consequently, its chance-level long-distance
scores cannot be isolated as an extrapolation failure after successful fitting.

### Historical v1 cost and validity

Batch-one d128 latency, mean of each seed's synchronized median:

| Model | 2D ms/image | 3D ms/image |
|---|---:|---:|
| Attention control | 10.05 | 12.13 |
| Learned transport | 35.43 | 40.89 |

This portable float32 PyTorch implementation is slower than the attention
control at these sizes; neither number establishes an optimized-kernel ceiling.
Solver agreement, RHS/edge gradients, 2D/3D constant preservation, symmetry,
nonexpansion and GPU float32 large-scale checks passed. End-to-end smoke found
nonzero learned-edge gradients in both dimensions. These validate software
paths, not the scientific hypothesis.

See [the numerical check record](evidence/validation/transport_checks.json),
[the original model smoke record](evidence/validation/smoke.json), and the
exact numerical test `test_transport.py::run_checks`. The numerical check was
rerun during packaging; the original training sources retain their recorded hashes.

Both dimensions receive INCONCLUSIVE_POSITIVE_CONTROL because attention did
not fit reliably across both seeds. Gates B/C and the message-width sweep were
not run. The current configuration provides no evidence of an RT advantage;
the qualification does not refute the architecture family. Further scientific
comparison first requires a reliable training qualification. No extra steps,
hyperparameter changes or rescue sweep were run after observing these scores.

The initial v1 dispatch was interrupted for a distance/axis scheduling defect;
its outputs are preserved and excluded, as recorded in `RUN_MANIFEST.md`.
The user requested no continued tracking; no background training or monitoring
remains scheduled.
