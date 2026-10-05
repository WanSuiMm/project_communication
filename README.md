# Cellular computation: reaction-transport and inertial NCA screens

## Latest result: native execution-state width qualification

`native_latent_width_v1`: **COMPLETE**, 96/96 planned arms across 16 blocks.
All six arms recorded 0/16 Full passes: W24, W8, the parameter-matched W24
control, W16, W4, and exploratory W2 alternating-axis. The frozen W8-versus-W24
primary contrast has 0 wins, 0 losses, net gain 0/16, and exact two-sided
`p=1`; the verdict is `NO_W8_RELIABILITY_QUALIFICATION`. Concurrent W24 also
records 0/16. This is a negative result for the frozen native-width recipe,
not a universal impossibility claim about latent widths, architectures, or
BPTT.

Runtime provenance: all six block00 arms passed exact 300-update CUDA Graph
replay against the eager reference, including bitwise-equal final parameters
and Adam state/groups. The completed aggregate imports 13 eager arms and has
83 CUDA Graph arms. This finite replay check qualifies the runtime only. The
reported 2771.016 seconds excludes the original eager run and the separate
replay qualification, so it is not end-to-end wall time for all 96 arms.

Start with [results](evidence/latent_width_20261005/RESULTS.md),
[summary](evidence/latent_width_20261005/summary.json),
[validation](evidence/latent_width_20261005/validation.json),
[configuration](evidence/latent_width_20261005/config.json), and
[reproduction notes](evidence/latent_width_20261005/REPRODUCTION.md); then
inspect [per-arm results](evidence/latent_width_20261005/perarm.json) and
[metrics](evidence/latent_width_20261005/metrics.csv). Losslessly compressed
per-block evaluation arrays, summary records, frontier CSVs and curves are
under `block00/` through `block15/`; `plans.json.gz` contains the frozen block
plans. W2 alternates its transport axis and is not a width-only contrast.
The frozen [protocol](new/latent_width/PROTOCOL.md),
[execution amendment](new/latent_runtime/ACCELERATION.md), and code map in
[GPT_CONTEXT.md](GPT_CONTEXT.md) define the procedure. Verify the saved-data
snapshot from the repository root with:

    python -X utf8 -B tools/export_latent_width.py --verify-only

## Previous result: binary-carrier causal compression

`binary_carrier_causal_compression_v1`: **COMPLETE**, 24/24 cells in
473.469 seconds. The W replication anchors qualify, but all three combined
primary gates (`bit_once`, `bit_repeat8`, and `source_once`) fail at both
sizes. The run fits one local linear decoder, uses fixed calibration centroids,
and makes zero recurrent updates; the full keep/progress table is in the
results report.

The oracle-only centroid arm has sustained progress of 0.9967/0.6390 but
keep-continuous of 0.1193/0.0896, so it misses the keep requirement at both
sizes. Under the frozen protocol this is evidence against the tested
centroid/consumer combination, not a universal bit-code impossibility claim.
The run concerns this selected producer, consumer, and finite projected
continuation; it does not establish a quotient, architecture, or BPTT guarantee.

Start with [results](evidence/binary_carrier_20261005/RESULTS.md),
[summary](evidence/binary_carrier_20261005/summary.json),
[validation](evidence/binary_carrier_20261005/validation.json), and
[configuration](evidence/binary_carrier_20261005/config.json); then read
[decoder fit](evidence/binary_carrier_20261005/decoder/fit.json),
[decoder probes](evidence/binary_carrier_20261005/decoder/probes.json), and
[metrics](evidence/binary_carrier_20261005/metrics.csv). Raw arrays and
per-map records are secondary. The frozen
[protocol](new/binary_carrier/PROTOCOL.md) and code map in
[GPT_CONTEXT.md](GPT_CONTEXT.md) define the procedure. Verify the saved-data
snapshot from the repository root with:

    python -X utf8 -B tools/export_binary_carrier.py --verify-only

## Previous result: W medium qualification

`w_medium_qualification_v1`: **COMPLETE**, 36/36 intervention cells in
570.406 seconds, with zero training or optimizer updates. Stage 1 is
`QUALIFIED_W_REPLICATION_CONTROLS_ENABLED` at sizes 32 and 64; native Full is
true for S1 and false for F1, and the historical localization episode predicate
qualifies at both sizes. On fixed fresh 128-map banks, SF passes the frozen
rescue proxy at both sizes: keep-continuous is 1.0000/0.9899 and sustained
progress is 0.5726/0.5925.

The fixed-consumer controls remain distinct. RMS-normalized SF passes at both
sizes while the reverse FF normalization fails. Raw payload permutation fails
progress (0.0053/0.0134), while its exact consumer-gauged control reproduces
SF; raw lane permutation passes at both sizes. Spatial scrambling preserves
progress (0.9966/0.9147), but its size-64 keep-continuous value of 0.9029 is
below the 0.95 gate, so that rescue proxy fails. Cue swaps fail at both sizes;
donor times 48, 80, and 96 pass at both sizes, while time 32 fails the keep
gate. These selected-checkpoint sensitivity results do not classify an
exclusive mechanism, establish a learned algorithm, or prove that W is solely
for execution and Z solely for the answer. Native Full, localization, and
per-arm rescue flags are reported separately.

Start with [results](evidence/w_medium_20261004/RESULTS.md),
[summary](evidence/w_medium_20261004/summary.json),
[validation](evidence/w_medium_20261004/validation.json),
[configuration](evidence/w_medium_20261004/config.json), and
[reproduction notes](evidence/w_medium_20261004/REPRODUCTION.md); then inspect
the [metrics](evidence/w_medium_20261004/metrics.csv) and
[figure](evidence/w_medium_20261004/w_medium.png). Raw per-arm records are
secondary. The frozen [protocol](new/w_medium/PROTOCOL.md) and exact code map in
[GPT_CONTEXT.md](GPT_CONTEXT.md) define the procedure. Verify the published
snapshot from the repository root with:

    python -X utf8 -B tools/export_w_medium.py --verify-only

## Previous result: output-preserving state factorization

`output_preserving_state_factorization_v1`: **COMPLETE**, 16/16 intervention
cells in 74.172 seconds, with zero training or optimizer updates. The primary
episode is `LOCALIZATION_EPISODE_UNQUALIFIED`; confirmation is `UNQUALIFIED`.
On the fixed shared-solved cohort, SF continuous preservation is 1.000 at both
sizes; on the fixed shared-unsolved cohort, its sustained progress is 0.5152
at size 32 and 0.6730 at size 64. The SS rescue proxy
is unqualified at both sizes: its progress cohort has only 6 valid maps at size
32, and its keep cohort has only 9 at size 64, below the frozen 16-map support
minimum. This is a selected-checkpoint finite state-localization intervention;
it does not establish encoded task information, a consistency relation, or
population reliability.

Start with [results](evidence/state_factorization_20261004/RESULTS.md),
[summary](evidence/state_factorization_20261004/summary.json),
[validation](evidence/state_factorization_20261004/validation.json),
[configuration](evidence/state_factorization_20261004/config.json), and
[reproduction notes](evidence/state_factorization_20261004/REPRODUCTION.md);
then inspect the [metrics](evidence/state_factorization_20261004/metrics.csv)
and [figure](evidence/state_factorization_20261004/state_factorization.png).
Raw outputs and per-arm records are secondary. The frozen
[protocol](new/state_factorization/PROTOCOL.md) and code map in
[GPT_CONTEXT.md](GPT_CONTEXT.md) define the procedure. Verify the published
snapshot from the repository root with:

    python -X utf8 -B tools/export_state_factorization.py --verify-only

## Previous result: credit-locality formation audit

`credit_locality_formation_v1`: **COMPLETE**, 7/7 checkpoints, 138.5 seconds,
with zero training or optimizer updates. The frozen verdict is
**CREDIT_SYNCHRONY_NOT_SUPPORTED**. At size 32, preservation/progress fall from
0.9471/0.5007 at update 175 to 0.4958/0.0432 at 180, while overall gradient
cosine and `C_parallel` rise from -0.0352/-0.0019 to 0.2243/0.0122. Overall,
F and Q synchrony qualifications fail. Native Full passes are 0/7 checkpoints
on the reused held-out banks; update 200 misses strict T64 pooled reach
(0.76648 versus 0.80). This is a finite,
within-trajectory association screen; it establishes neither causality nor a
broad impossibility result.

Read the public [results](evidence/credit_formation_20261004/RESULTS.md),
[summary](evidence/credit_formation_20261004/summary.json),
[validation](evidence/credit_formation_20261004/validation.json),
[configuration](evidence/credit_formation_20261004/config.json), and
[reproduction notes](evidence/credit_formation_20261004/REPRODUCTION.md), then
the [metrics](evidence/credit_formation_20261004/metrics.csv) and
[figure](evidence/credit_formation_20261004/credit_formation.png). Per-update
gradient metadata, continuation, and phenotype records are secondary. The
frozen [protocol](new/credit_formation/PROTOCOL.md) and [source map](GPT_CONTEXT.md)
define the procedure and code entry points. Verify the published snapshot from
the repository root with:

    python -X utf8 -B tools/export_credit_formation.py --verify-only

## Previous result: continuation-interface audit

`continuation_interface_v1`: **COMPLETE**, 50 transfer cells, 439 seconds.
The common native Full qualification passes for S1/S2, and the gauge control
passes. Raw S1↔S2 off-diagonal transfer fails while raw S1→F1 passes. Under the
frozen 100-coefficient linear alignment, S2→S1 and S2→F2 pass while S1→S2
still fails. This selected-checkpoint audit does not establish a common
algorithm. Restricted off-diagonal alignment qualification is **0/20**, and
matched short-observable flags are **0/6**; these remain separate from transfer
proxy passes. No model training or optimizer steps were run.

Read in this order:

1. [Results](evidence/continuation_interface_20261004/RESULTS.md),
   [summary](evidence/continuation_interface_20261004/summary.json),
   [validation](evidence/continuation_interface_20261004/validation.json),
   [configuration](evidence/continuation_interface_20261004/config.json), and
   [reproduction notes](evidence/continuation_interface_20261004/REPRODUCTION.md).
2. [Selection](evidence/continuation_interface_20261004/selection.json),
   [alignment validation](evidence/continuation_interface_20261004/alignment/validation.json),
   [short-objective comparisons](evidence/continuation_interface_20261004/phase0/comparisons.json),
   [cross-consumer decomposition](evidence/continuation_interface_20261004/cross/decomposition.json),
   and [gradient metadata](evidence/continuation_interface_20261004/gradients/metadata.json).
3. Frozen [protocol](new/continuation_interface/PROTOCOL.md) and the
   [source map](GPT_CONTEXT.md). Full per-cell records and larger arrays are
   secondary. Verify the published snapshot from the repository root with
   `python -X utf8 -B tools/export_continuation_interface.py --verify-only`.

## Previous result: serial training-state and second-task qualification

`trajectory_qualification_v1`: **COMPLETE**, 48/48 arms across the three stages
(8 + 32 + 8; 6,610.797 seconds). Stage 1 selected `shadow_joint` as a
conditional update-3 donor-state recipe. Stage 2 recorded 1/16 Full passes for
both baseline and treatment, with no reliability qualification. Stage 3 was
`SECOND_TASK_TRAINING_UNQUALIFIED`: 0/8 arms met the size-32 T64 short-task
training gate, and 0/8 passed the full distance-task proxy. The original
StreamingCell and K8 credit window remain fixed; stage 3 is a separate task
proxy, not the original Full gate. This is a data release for independent
review, without new mechanism analysis.

Read in this order:

1. [Suite results](evidence/trajectory_qualification_20261004/RESULTS.md),
   [summary](evidence/trajectory_qualification_20261004/summary.json), and
   [validation](evidence/trajectory_qualification_20261004/validation.json).
2. Stage 1 [results](evidence/trajectory_qualification_20261004/stage1_state_cross/RESULTS.md)
   and [summary](evidence/trajectory_qualification_20261004/stage1_state_cross/summary.json);
   stage 2 [results](evidence/trajectory_qualification_20261004/stage2_fresh_recipe/RESULTS.md)
   and [summary](evidence/trajectory_qualification_20261004/stage2_fresh_recipe/summary.json);
   stage 3 [results](evidence/trajectory_qualification_20261004/stage3_distance/RESULTS.md)
   and [summary](evidence/trajectory_qualification_20261004/stage3_distance/summary.json).
3. Frozen [suite protocol](new/trajectory_qualification/PROTOCOL.md),
   [distance-task protocol](new/trajectory_qualification/DISTANCE_PROTOCOL.md),
   [reproduction notes](evidence/trajectory_qualification_20261004/REPRODUCTION.md),
   and the [source map](GPT_CONTEXT.md).

## Previous result: conditional seed4 bootstrap-path screen

`seed4_bootstrap_path_v1`: COMPLETE, 23 arms × 300 updates, 3,068.407 s.
This publication supplies data and code for independent review.

1. [All-arm table](evidence/bootstrap_path_20261004/RESULTS.md), [aggregate](evidence/bootstrap_path_20261004/summary.json), and [six first-step measurements](evidence/bootstrap_path_20261004/first_step_summary.json).
2. [Frozen protocol](new/bootstrap_path/PROTOCOL.md) and [source map](GPT_CONTEXT.md).
3. Per-arm `arms/` JSON, `training/` curves, `frontier/` CSV, and `traces/` NPZ under `evidence/bootstrap_path_20261004/`; large raw arrays are secondary. [Validation](evidence/bootstrap_path_20261004/validation.json) and [reproduction](evidence/bootstrap_path_20261004/REPRODUCTION.md) describe checks and archive requirements.

## Previous diagnostic: update-zero geometry and continuation formation

The initialization audit covers historical seeds 2/3/4/5 and previously inspected
16-map primary plus 8-map consistency banks per size at 32/64; historical
parameter hashes match, and the CPU audit took 30.31 seconds. Seed 4 has the
best raw lane balance and measured available source-flip pair separation at
K8 step 64 across the bank types, but centered balance and cross-lane
correlation are mixed; seed 3 has the highest paired kernel-label alignment.
At the first historical batch, Q-out weight is the only parameter group with
appreciable task gradient; it is rank one up to FP32 roundoff. The readout
weight gradient is exactly zero, while balanced-loss bias gradients are at
FP32 cancellation scale. These feature statistics do not give a
successful-initialization criterion.

The saved-trajectory formation analysis recomputes 44 arrays in 8.86 CPU
seconds. Its early behavior screen passes at updates 140/145/190/200, while
the long behavior screen passes only at 200. The early screen gives 1 true
positive and 3 false positives at the same update, then misses 200 when asked
to predict the next saved update. Early continuous measures and ordinary T64
output coverage both rank 200 highest. This retrospective single-trajectory
result does not show independent predictive value or a formed closure
mechanism. No latent `pi`, abstract transition, or independent consumer was measured; the
conditional execution theorem remains unchanged. The second bank was already
inspected and supplies consistency checking, not fresh confirmation.

Start here for these diagnostics:

1. [Integrated results](evidence/seed4_learning_geometry_20261004/RESULTS.md).
2. [Initial geometry results](evidence/initial_geometry_20261004/RESULTS.md),
   [summary](evidence/initial_geometry_20261004/summary.json),
   [profiles](evidence/initial_geometry_20261004/profiles.csv), and
   [validation](evidence/initial_geometry_20261004/validation.json).
3. [Formation results](evidence/formation_gate_20261004/RESULTS.md),
   [interpretation](evidence/formation_gate_20261004/INTERPRETATION.md),
   [summary](evidence/formation_gate_20261004/summary.json),
   [profiles](evidence/formation_gate_20261004/profiles.csv),
   [formation figure](evidence/formation_gate_20261004/formation_profiles.png), and
   [validation](evidence/formation_gate_20261004/validation.json).
4. [Frozen protocols](new/initial_geometry/PROTOCOL.md) and
   [formation protocol](new/formation_gate/PROTOCOL.md),
   [geometry audit](new/initial_geometry/audit.py),
   [geometry summary](new/initial_geometry/summarize.py),
   [formation analysis](new/formation_gate/analyze.py),
   [geometry reproduction](evidence/initial_geometry_20261004/REPRODUCTION.md),
   [formation reproduction](evidence/formation_gate_20261004/REPRODUCTION.md),
   and [publication manifest](LEARNING_GEOMETRY_PUBLICATION_MANIFEST.json).

Verify the published snapshot from the repository root with
`python -X utf8 -B tools/export_learning_geometry.py --verify-only`. This
check needs no NPZ files or checkpoints. Full replay depends on the original
archives, which are excluded from GitHub and are not recreated by this command.

Preceding diagnostic: **joint195/200 zero-training causal audit completed**,492
conditions in14.76 minutes, with all four full historical endpoint traces
replayed exactly. On the primary size32 bank, the state produced at64 by200
reaches strict T256 coverage100% even under the195 continuation rule; the
195 state reaches93.33% under the200 rule. Readout replacement has little
effect. Encoder/workspace changes and their interaction with Q matter;
the average Q block effect is not an independent verdict on Q. The sharp
interpolation improvement on primary32 is concentrated in one map, and195
already performs well on the independent size32 bank. This locates conditional
state-production/coadaptation sensitivity, without establishing a general
computational phase transition or a unique local commitment mechanism.

Preceding results: [joint195/200 report](evidence/joint195_200_20261003/RESULTS.md),
[interpretation](evidence/joint195_200_20261003/INTERPRETATION.md), the
[all492 profiles](evidence/joint195_200_20261003/profiles.csv), and its
[frozen protocol](new/audit_195_200/PROTOCOL.md).

Preceding diagnostic: **historical seed4 dense update100--200 replay completed**.
One unchanged K8 training trajectory exactly reproduces the historical
parameters at0/100/200/300 and all six final historical endpoint payloads.
The21 checkpoints100,105,...,200 plus300 control give44 paired256-step
audits on fixed16-map size32/64 banks. Total execution353.141 seconds.
The curve improves unevenly, with a marked regression atupdate180. Between
195 and200, size32 strict paired T128 coverage rises85.65%->97.55%,
first-exit destruction64->128 falls5.15%->0.032%, and continuous survival
64->256 rises94.53%->99.97%. Only200 meets the frozen descriptive screen
within the dense window;205/210 were not sampled, so the three-consecutive
checkpoint onset remains unestablished. This is a finite-window behavioral
change in one selected run, not proof of latent commitment or a phase transition.
[Latest results](evidence/transition_20261003/RESULTS.md),
[compact profiles](evidence/transition_20261003/summary.json),
[frozen protocol](new/transition_100_200/PROTOCOL.md), and
[reproduction notes](evidence/transition_20261003/REPRODUCTION.md).

Preceding 2D screen: **detached warm-start — DEVELOPMENT_NOT_QUALIFIED**. All eight
paired arms completed300 updates in27.16 minutes. The four historical baseline
replays pass exactly, and baseline seed4 qualifies on the new evaluation banks.
Full phenotype passes are **baseline1/4 versus warmstart0/4**. Same original
StreamingCell and K8; half of each warm-start batch begins at a detached
on-policy state aged32/64/128/192 steps. Some components improve at seed2,
while seed4 loses reach and long-rollout retention. This fixed recipe does not
improve full-phenotype reproducibility in the sampled four initializations;
it does not identify a mechanism or reject warm-start methods in general.
[Current results](evidence/warmstart_20261003/RESULTS.md),
[all-arm aggregate](evidence/warmstart_20261003/summary.json),
[frozen protocol and runtime amendment](new/warmstart/PROTOCOL.md), and
[reproduction notes](evidence/warmstart_20261003/REPRODUCTION.md).

Preceding follow-up: **original seed4 reproduces, full phenotype does not reproduce
in the sampled A/B interventions**. All13 arms completed300 updates in31.11
minutes. The control exactly matches historical initial/final parameters and
all six endpoint records, and passes the new phenotype on fresh32-map banks
at sizes32/64. A passes0/4 new schedules; B passes0/4 directions at each of
two paired radii (.01/.05). All 12 A/B arms pass both matched-frontier
gates, but none meets the full reach/hold/retention/coverage/relapse gate.
Some arms have partial long-rollout behavior. This is conditional schedule
and directional sensitivity, not a basin-radius theorem or general NCA verdict.
The independent C rollback audit misses its one-step joint threshold; larger
four-step effects remain secondary, without identifying a unique handoff law.
[Current results](evidence/seed4_followup_20261003/RESULTS.md),
[interpretation](evidence/seed4_followup_20261003/INTERPRETATION.md), and
[compact aggregate](evidence/seed4_followup_20261003/summary.json).

Preceding architecture experiment: **Stationary Sidecar — DEVELOPMENT_NO_GO**. All12 arms
completed300 updates in27.94 minutes. Four original Streaming controls exactly
reproduce historical final parameter hashes and full evaluations. Reach+hold
is **stream1/4, persistent H0/4, stateless0/4**. Both side branches preserve
the original W24/Z8 core and add the same7989-parameter recipe; only extra H
carry differs. Stream has5033 parameters. Seed4 primary pooled T64/T128/T256:
stream **85.92/99.35/100%**, memory **94.24/0/0.38%**, stateless
**100/100/88.49%**. Both side branches fail hold. This recipe improves some
finite-horizon endpoints but does not make beneficial long rollout more
reproducible. Carry changes temporal accumulation and scale together; no
general rejection of stationary memory or failure mechanism follows.
[Current report](evidence/stationary_sidecar_init2345/RESULTS.md),
[compact analysis](evidence/stationary_sidecar_init2345/analysis.json), and
[frozen protocol](new/stationary_sidecar/PROTOCOL.md).

Preceding descriptive diagnostic: a **zero-training behavior audit of original Streaming
seed4**. All six full historical endpoints replay exactly. On32 maps per
size32/64, pixels correct at64 are retained at256 at **100% / 98.65%**.
Every-step traces also reveal regressions: **2.24% / 8.31%** of ever-correct
pixels regress at least once. Matching map, time and exact BFS distance gives
one-step frontier acquisition differences of **+30.92 / +22.44 pp** (equal-map
means). This selected checkpoint exhibits mostly retained correctness and
local acquisition, but the association does not identify causal handoff,
flood-fill, latent closure or training reliability. The horizon ends at256;
maps are reused, model n=1, and earlier architecture no-go decisions remain.
[Behavior report](evidence/frontier_audit_seed4/RESULTS.md),
[diagnosis](evidence/frontier_audit_seed4/DIAGNOSIS.md), and
[compact summary](evidence/frontier_audit_seed4/summary.json).

The preceding diagnostic was a **zero-training operator audit of Streaming seed4**.
All six historical size/horizon evaluations replay exactly. Replacing fixed
streaming by identity or removing direct Laplacian inputs in both F/Q reduces
primary pooled paired correctness from **85.92 / 99.35 / 100%** at
T64/T128/T256 to **0 / 0 / 0%**; near-cue behavior also collapses.
This selected frozen solution is sensitive to both interventions. They alter
learned feature/state distributions and hop depth, so the result does not
identify training causality or prove that restoring Laplacians repairs H/C/Z.
No new model was trained. Earlier multi-seed gate decisions are unchanged.
[Audit results](evidence/stream_path_audit_seed4/RESULTS.md) and
[interpretation](evidence/stream_path_audit_seed4/INTERPRETATION.md).

This repository tests whether changing the medium and update rule improves
persistent cellular computation. The preceding **Persistent Roles** development
screen is **DEVELOPMENT_NO_GO**: reach+hold is **0/4**, versus **2/4** for the
additive baseline and **1/4** for Streaming Carry. All12 matched runs completed
300 updates in1204.234 seconds (20.07 minutes), and all eight old controls
exactly reproduce their historical final parameters and complete evaluations.
At size32/T64, strict16<d<32, candidate pooled paired correctness is
0.00/0.31/0.00/1.58% across seeds2/3/4/5. Longer rollout does not recover the
primary; size64/d>32 remains at most0.24% pooled throughT256. All four
candidate Hold=True values preserve weak endpoints, not successful computation.
The candidate uses H12 stationary workspace, C12 persistent directional
carrier and Z8 stationary task state, with one shared local residual rule
followed by streaming twice per macro-step. It matches5033 parameters and32
persistent scalars, but changes role allocation, carrier width, perception,
parameter sharing and readout timing together; no H-specific cause is isolated.
[Persistent Roles results](evidence/persistent_roles_init2345/RESULTS.md).

The preceding **Local Interface** development
screen is **DEVELOPMENT_NO_GO**: reach+hold is **0/4** for the ephemeral
interface, versus **2/4** for the additive baseline and **1/4** for Streaming
Carry. All 12 matched runs completed 300 updates each in 1281.532 seconds
(21.36 minutes). Every interface seed scores 0% mean and pooled paired
correctness on the primary size32/T64 band, strict graph distance 16<d<32
(2,918 eligible pixels across 32 maps). Interface seed 4 has hold=True but
fails reach, so it does not pass the combined endpoint. All eight baseline and
stream controls exactly reproduce their historical final parameter hashes and
complete evaluation records. This development comparison changes state
allocation, neighbor representation and hidden widths together; it does not
isolate an interface effect. [Local Interface results](evidence/local_interface_init2345/RESULTS.md).

The preceding **Streaming Carry** development screen remains
**DEVELOPMENT_NO_GO**: reach+hold is **1/4** for fixed masked permutation
streaming versus **2/4** for the additive baseline, below the frozen 3/4
streaming gate. The stream arm passes only seed 4; the other three fail reach.
All four baseline arms exactly reproduce their Phase-II final parameters and
complete evaluation records. Eight matched K8 runs completed 300 updates each
in 828.297 seconds (13.80 minutes). This is a bounded development result on
inspected seeds and maps. The fixed transport preserves Euclidean norms, but
the nonlinear residual and local-memory updates do not inherit that guarantee.
[Streaming Carry results](evidence/streaming_carry_init2345/RESULTS.md).

The previous **Direct Spatial Carry** development screen remains
**DEVELOPMENT_NO_GO**: fixed normalized neighbor averaging reduced reach+hold
from baseline **2/4 to 0/4**. All four seeds lost primary paired accuracy; its
baseline exactly reproduced Phase II in final parameters and evaluation
records. The tested rho=0.5 recipe failed without identifying a mechanism or
rejecting all spatial carry. [Direct Carry results](evidence/direct_spatial_carry_init2345/RESULTS.md).

The previous **short-BPTT Phase II** completed
12 matched runs: additive cells, K8/K16/K64, four new initialization seeds,
one fixed training bank and schedule. **K8 reaches and holds in 2/4 seeds,
below the frozen 3/4 replication criterion.** K16 reaches in 3/4 but reaches
and holds in only 1/4; all four K64 controls remain unqualified. Two new K8
positive cases retain high accuracy in the selected narrow distance band,
including on larger grids, while farther propagation is weak. This prospective
follow-up tests strict graph distance **16<d<32** on new maps; it does not amend
Phase I's broader failed gate. [Phase-II results](evidence/short_bptt_phase2_init2345/RESULTS.md).

The previous **matched short-BPTT Phase-I screen**
completed eight 300-update runs: additive/revision cells, K8/K64 gradients,
two paired seeds, identical 64-step forward trajectories and losses.
**All four full-BPTT controls failed qualification**, so neither architecture
passes the frozen comparison. One additive K8 seed reaches **80.87% far paired
accuracy** at size32/T64 and retains 78.24% at T256. This is a per-map average;
pooled far-pixel accuracy is 59.53% at T64, with weak distance/size extrapolation.
Peak allocated CUDA memory is **82.4% lower** for K8; training times are similar.
This is a useful positive case and a systems observation, not a general
short-BPTT advantage. [Phase-I results](evidence/short_bptt_paired01/RESULTS.md).

The preceding **zero-training source-switch audit** confirms that revision
seed0's candidate continues to support the old
answer. W-only and Z-only resets or transplants fail to restore reliable
switching, while cold reset of both succeeds. At size32/K128, all four
single-block interventions predict negative everywhere on the changed region;
their 37.5% accuracy is label bias, not partial recovery. This implicates both
state blocks in the tested interventions, without isolating a unique mechanism.
[Audit results](evidence/switch_audit_seed0/RESULTS.md).

The preceding **Workspace + Revision paired
screen** completed all four 600-update runs. It **fails the joint gate**:
revision's mean hold effect is **-9.02 pp** across two paired seeds. Seed 0
reaches and holds excellent accuracy, including at doubled spatial size, but
retains the old answer after a source change (0% changed-component accuracy
versus 100% from a cold start at size32/K64). Seed 1 stays at 50% BA.
This supports a narrower single-seed reach/hold observation, not a successful
general revision rule. Both arms share the workspace split; only their Z
update differs. [Full interpretation](evidence/workspace_revision_paired01/INTERPRETATION.md).

## Start here

For incremental review from `e142f224ad03f9bde9894dfcf80cf41411744f6e`, begin with [GPT_HANDOFF.md](GPT_HANDOFF.md).

1. [Latest dense-checkpoint results](evidence/transition_20261003/RESULTS.md) and
   [compact profiles](evidence/transition_20261003/summary.json).
2. [Frozen protocol](new/transition_100_200/PROTOCOL.md),
   [unchanged K8 helper](new/short_bptt/training.py),
   [executed runner](new/transition_100_200/run.py),
   [trace metrics](new/transition_100_200/metrics.py), and [GPT_CONTEXT.md](GPT_CONTEXT.md).
3. [Saved-artifact validation](evidence/transition_20261003/validation/local_validation.json),
   [publication bindings](TRANSITION_PUBLICATION_MANIFEST.json), and
   [reference provenance](evidence/transition_20261003/input_provenance.json).
4. [Reproduction notes](evidence/transition_20261003/REPRODUCTION.md).
   Verify without a checkpoint/GPU using
   `python tools/export_transition_evidence.py --verify-only` or run the focused
   metric fixtures with `python new/transition_100_200/check_metrics.py`.
   Public-reference CPU qualification uses
   `python tools/replay_transition_public.py --check --out analyses/NEW_PUBLIC_TRANSITION_CPU.json`.
   Per-checkpoint map counts and profiles are secondary; full NPZ trajectories,
   checkpoints, logs and machine receipts remain local, with hashes bound.

Previous context, if needed:

1. [Previous behavior report](evidence/frontier_audit_seed4/RESULTS.md),
   [diagnosis](evidence/frontier_audit_seed4/DIAGNOSIS.md), and
   [compact summary](evidence/frontier_audit_seed4/summary.json). The report
   includes the distance/acquisition figures; full arrays remain local.
2. [Frozen behavior protocol](new/frontier_audit/PROTOCOL.md),
   [trajectory/replay runner](new/frontier_audit/audit.py),
   [per-map metrics](new/frontier_audit/metrics.py), and [GPT_CONTEXT.md](GPT_CONTEXT.md).
3. [Full local-trace validation record](evidence/frontier_audit_seed4/validation.json),
   [public arithmetic validation](evidence/frontier_audit_seed4/publication_validation.json),
   [size32 matched counts](evidence/frontier_audit_seed4/frontier_matches_size32.csv),
   [size64 matched counts](evidence/frontier_audit_seed4/frontier_matches_size64.csv),
   [provenance](evidence/frontier_audit_seed4/provenance.json), and
   [publication bindings](FRONTIER_AUDIT_PUBLICATION_MANIFEST.json).
   Recompute exported effects with `python tools/export_frontier_audit.py --verify-only`.
4. [RESULTS.md](RESULTS.md), the latest training screen's
   [Sidecar report](evidence/stationary_sidecar_init2345/RESULTS.md), the preceding
   [seed4 operator audit](evidence/stream_path_audit_seed4/RESULTS.md),
   [Persistent Roles report](evidence/persistent_roles_init2345/RESULTS.md),
   [Local Interface report](evidence/local_interface_init2345/RESULTS.md),
   [Streaming Carry report](evidence/streaming_carry_init2345/RESULTS.md),
   and earlier evidence: [Direct Spatial Carry](evidence/direct_spatial_carry_init2345/RESULTS.md),
   [Phase II](evidence/short_bptt_phase2_init2345/RESULTS.md),
   [Phase I](evidence/short_bptt_paired01/RESULTS.md),
   [source-switch audit](evidence/switch_audit_seed0/RESULTS.md),
   [prior training screen](evidence/workspace_revision_paired01/INTERPRETATION.md),
   and [generic dynamics audit](evidence/dynamics_audit_seed0/INTERPRETATION.md).

The sanitized Sidecar package includes its [completion summary](evidence/stationary_sidecar_init2345/completion.json)
and omits checkpoints, local machine receipts and transient logs. Earlier
publication manifests remain valid.

Minimal sidecar reproduction from repository root with the existing requirements:

```text
python new/stationary_sidecar/check.py --out analyses/NEW_SIDECAR_CHECK.json
python new/stationary_sidecar/run.py --preflight --out runs/NEW_SIDECAR_PREFLIGHT
pwsh -File tools/launch_stationary_sidecar.ps1 -RunName NEW_SIDECAR_RUN -Preflight runs/NEW_SIDECAR_PREFLIGHT
python new/stationary_sidecar/analyze.py --run runs/NEW_SIDECAR_RUN --preflight runs/NEW_SIDECAR_PREFLIGHT --out analyses/NEW_SIDECAR_REVIEW
```

Training requires CUDA; analysis reads locally generated checkpoints on CPU.
Existing public JSON/CSV and integer denominators support saved-result review
without downloading checkpoints or launching training.

## Previous evidence

The seed-0 medium/recurrence screen favors generic Masked Momentum over Masked
State at32/T64, but both deteriorate at long rollout. This compares recipes
with different widths, not velocity alone. Its missing auxiliary tail was
recovered from saved weights without training; [recovery provenance](evidence/masked_state_seed0/recovery.json)
retains that distinction. The subsequent generic dynamics audit finds positive
finite-window gain already early, with limited numerical precision and no
observed near-zero-to-positive crossing. Earlier explicit-inertial drift
diagnostics do not directly transfer to generic Momentum. These results and
the earlier 2D/3D transport failures remain unchanged in [RESULTS.md](RESULTS.md).
No further training or monitor is scheduled.

## Reproduce the seed4 operator audit

Use the same Python/PyTorch/NumPy environment as the existing training screen.
The CPU smoke needs no checkpoint. The CUDA audit requires the retained
original `stream_K8_seed4.pt` and original run/source bindings under
`runs/streaming_carry_20261002_init2345/`; checkpoints are excluded from Git.
The published metrics and CPU reference can be reviewed without those files.
The preceding Streaming Carry reproduction section gives the original
training commands; no training was performed for this publication.

    python new/stream_path_audit/check.py
    python new/stream_path_audit/audit.py --out runs/NEW_STREAM_PATH_AUDIT
    python new/stream_path_audit/analyze.py --run runs/NEW_STREAM_PATH_AUDIT --out analyses/NEW_STREAM_PATH_REVIEW

The saved-result analyzer performs CPU arithmetic/provenance checks and does
not run checkpoint inference. The [exporter](tools/export_stream_path_audit.py)
publishes byte-identical metrics and sanitized metadata. Public validation's
`input_sha256` refers to original run bytes; the publication manifest separately
binds the public manifest/completion copies. Begin with the summaries above,
then inspect the 2.1 MB raw condition file only for detailed per-map evidence.

## Reproduce Persistent Roles

Install [requirements.txt](requirements.txt), run from this repository root,
and use new output directories. No prior weights are needed for training.
The CPU suite checks default continuation, nonzero gradients, phase readout
lag, disconnected components and three-state K8 cuts. Training requires CUDA.

    python new/persistent_roles/check.py
    python new/persistent_roles/run.py --preflight --out runs/NEW_ROLES_PREFLIGHT
    pwsh -File tools/launch_persistent_roles.ps1 -RunName NEW_ROLES_RUN -Preflight runs/NEW_ROLES_PREFLIGHT

The fixed screen trains baseline/stream/roles on seeds2/3/4/5,300 updates
per arm. A failed runtime preflight stops without reducing that budget.
The hard cap is40 minutes. The runner writes all raw metrics, aggregate,
curves, paired effects and its frozen decision. Previously inspected data
and initialization seeds make this development, not fresh confirmation.

The result-specific CPU audit needs the completed local run, its matching
preflight and retained checkpoints. It performs no checkpoint inference:

    python new/persistent_roles/analyze.py --run runs/RECORDED_ROLES_RUN --preflight runs/RECORDED_ROLES_PREFLIGHT --out analyses/NEW_ROLES_REVIEW

All published measurements can be reviewed without weights. The
[exporter](tools/export_persistent_roles_evidence.py) retains raw saved
metrics byte-identically and sanitizes private manifest/completion copies.
Cross-device bitwise reproduction is not promised; exact control matches
describe this completed local run.

## Reproduce the previous Local Interface

Install [requirements.txt](requirements.txt) and run from this repository root
with new output directories. The candidate keeps W (24 channels) and Z (8
channels) at their cells and emits a fresh 24-channel message field for each of
two communication phases. The CPU check covers parameter/state sizes, shared
initialization, the explicit port operator, gradients, light-cone and K8 clock
properties. Training requires CUDA.

    python new/local_interface/check.py
    python new/local_interface/run.py --preflight --out runs/NEW_INTERFACE_PREFLIGHT
    pwsh -File tools/launch_local_interface.ps1 -RunName NEW_INTERFACE_RUN -Preflight runs/NEW_INTERFACE_PREFLIGHT

The frozen screen uses four previously inspected initialization seeds and 300
updates for each of three variants. The runner checks saved identities and
decisions; [analysis](new/local_interface/analyze.py) reviews saved results on
CPU, and the [exporter](tools/export_local_interface_evidence.py) writes a
sanitized evidence package. The candidate and controls each have 5,033
parameters, but the candidate's changed state allocation, message path and
hidden widths form one bundled comparison.

For the CPU saved-result audit, point it to the completed local run and its
matching preflight, then use a new output directory:

    python new/local_interface/analyze.py --run runs/RECORDED_INTERFACE_RUN --preflight runs/RECORDED_INTERFACE_PREFLIGHT --out analyses/NEW_INTERFACE_REVIEW

This audit checks saved metrics, control identities and checkpoint state hashes
without model inference or training. The private run and preflight directories
are not part of the public evidence package; the published result can be
reviewed from the sanitized summaries and curves.

## Reproduce the previous Streaming Carry screen

Install [requirements.txt](requirements.txt) and run from this repository root
with new output directories. The stream arm divides W (24 channels) into four
six-channel directional lanes. Both arms keep stationary Z (8 channels), 5,033
parameters and the fixed K8 training budget. The CPU check verifies the masked
port permutation, inverse and norm preservation, plus baseline identity.
Training requires CUDA.

```powershell
python new/streaming_carry/check.py
python new/streaming_carry/run.py --preflight --out runs/NEW_STREAM_PREFLIGHT
pwsh -File tools/launch_streaming_carry.ps1 -RunName NEW_STREAM_RUN -Preflight runs/NEW_STREAM_PREFLIGHT
```

The formal screen uses four paired initialization seeds and 300 updates per arm.
The analysis command verifies saved metrics and gate decisions on CPU; the
exporter creates the sanitized evidence package.

```powershell
python new/streaming_carry/analyze.py --run runs/RECORDED_STREAM_RUN --out analyses/NEW_STREAM_REVIEW
```

## Reproduce the previous Direct Spatial Carry screen

Install [requirements.txt](requirements.txt) and use new output directories
from this repository root. Both arms use the same historical initialization,
data, schedule,5033 parameters and K8 trainer; no prior weights are needed.
CPU checks compare carry with explicit neighbor averaging and verify baseline
equivalence at rho=0, including gradients. Training requires CUDA.

```powershell
python new/direct_spatial_carry/check.py
python new/direct_spatial_carry/run.py --preflight --out runs/NEW_CARRY_PREFLIGHT
pwsh -File tools/launch_direct_carry.ps1 -RunName NEW_CARRY_RUN -Preflight runs/NEW_CARRY_PREFLIGHT
```

Fixed300 updates per arm; preflight must fit the25-minute estimate, otherwise
abort without reducing the budget. Formal run writes all8 raw records,
`aggregate.json`, `curves.csv`, `paired_effects.csv` and `RESULTS.md`.
`COMPLETE` describes execution; the frozen scientific decision is separate.
The recorded screen finished in13.30 minutes on an RTX4060 Laptop GPU.
Peak allocation baseline95.12MiB, carry96.08MiB; these are systems measurements,
not matched-accuracy benefits. No follow-up confirmation is scheduled.

The CPU analysis command below verifies saved results and contains the
recorded screen's interpretation. It needs local checkpoints but does no
checkpoint inference. All published measurements can be reviewed without them.

```powershell
python new/direct_spatial_carry/analyze.py --run runs/RECORDED_CARRY_RUN --out analyses/NEW_CARRY_REVIEW
```

## Reproduce the previous short-BPTT Phase II

Install [requirements.txt](requirements.txt), run from this repository root,
and use new output directories. No prior weights are needed. The CPU check
verifies the new K16 gradients against an independent window reference and
K8/K64 against the frozen trainer. It also verifies exact forward identity.

```powershell
python new/short_bptt_phase2/check.py
python new/short_bptt_phase2/run.py --preflight --out runs/NEW_PHASE2_PREFLIGHT
pwsh -File tools/launch_bptt_phase2.ps1 -RunName NEW_PHASE2_RUN -Preflight runs/NEW_PHASE2_PREFLIGHT
```

CUDA preflight must pass the fixed 300-update budget estimate; it aborts if the
estimate exceeds 30 minutes. It never changes the count or selects settings
using efficacy. The formal runner writes per-arm measurements, `aggregate.json`
and `RESULTS.md`. Execution completion, absolute K8 replication and K64 control
qualification are separate fields. This run took 18.25 minutes on an RTX 4060
Laptop GPU with PyTorch 2.5.1 and NumPy 1.26.4; bitwise portability is not promised.

The separate analysis script verifies the recorded screen and contains its
result-specific interpretation. Use the runner's gate report for a new run.
All published measurements can be reviewed without weights. Checkpoints remain
local; no extra training or checkpoint inference was used for publication.

## Reproduce the previous short-BPTT Phase I

Install [requirements.txt](requirements.txt) and run from this repository root,
using new output directories. The CPU check verifies forward identity, actual
gradient cuts, parameter-gradient accumulation and a single optimizer clock.
CUDA is required for training; no prior weights are needed.

```powershell
python new/short_bptt/check.py
python new/short_bptt/run.py --preflight --out runs/NEW_BPTT_PREFLIGHT
pwsh -File tools/launch_short_bptt.ps1 -RunName NEW_BPTT_RUN -Preflight runs/NEW_BPTT_PREFLIGHT
```

The frozen runtime-only preflight rule selected 300 updates on the recorded
GPU; another machine may select a different budget. To replicate the recorded
300-update budget with the exact published sources, the runner also accepts
the [published passed preflight](evidence/short_bptt_paired01/preflight/):

```powershell
python new/short_bptt/run.py --preflight-dir evidence/short_bptt_paired01/preflight --out runs/NEW_BPTT_300
```

The runner writes all metrics and frozen gate decisions. The separate analysis
script verifies the specific recorded 300-update screen and its observed
qualification outcomes; it is not a generic evaluator for new runs. Training
took 12.18 minutes on an RTX 4060 Laptop GPU; cross-device bitwise identity is
not promised. `COMPLETE` is execution status, not a scientific pass: read pair
and architecture qualifications. New training/inference was not needed to
publish the present evidence. No follow-up run is scheduled.

## Reproduce the previous source-switch audit

The audit uses the retained `ws_revision_seed0.pt` at the repository-relative
location pinned in [audit.py](new/switch_audit/audit.py); the checkpoint hash is
in [SWITCH_PUBLICATION_MANIFEST.json](SWITCH_PUBLICATION_MANIFEST.json).
Weights stay local. Reviewing all published measurements requires no weights.
The CPU check needs only [requirements.txt](requirements.txt); inference needs
the exact checkpoint and CUDA. Use new output directories:

```powershell
python new/switch_audit/audit.py --check
python new/switch_audit/audit.py --out runs/NEW_SWITCH_AUDIT
python new/switch_audit/analyze.py --run runs/NEW_SWITCH_AUDIT --out analyses/NEW_SWITCH_REVIEW
```

The completed audit took 9.61 seconds, with no training. All 232 historical
comparisons replay exactly. Both natural trajectories' BA/BCE are checked;
W/Z RMS replay covers the original trajectory only because the fresh-flip
reference did not store norms. The original protocol's broader wording is
clarified in [review notes](new/switch_audit/REVIEW_NOTES.md), without rewriting
its frozen snapshot. New mixed donors may be off-distribution. One trained
seed and two sizes do not establish population-level or causal claims.

## Reproduce the previous paired screen

From this repository root, install [requirements.txt](requirements.txt), then
use new output directories. Training requires CUDA; the first check is CPU-only.
The recorded environment used PyTorch 2.5.1 and NumPy 1.26.4; exact data,
schedule, initial parameter and executed-source hashes are published. Four
600-update runs took 18.54 minutes on an RTX 4060 Laptop GPU. Cross-device
bitwise identity is not promised.

```powershell
python new/workspace_revision/check.py
python new/workspace_revision/run_revision.py --preflight --out runs/NEW_REVISION_PREFLIGHT
pwsh -File tools/launch_revision.ps1 -RunName NEW_REVISION_RUN -Preflight runs/NEW_REVISION_PREFLIGHT
```

The launcher verifies matching passed preflight sources, records a local
receipt and returns after dispatch verification. Read the final `status.json`
and `aggregate.json` to distinguish execution completion from scientific pass.
The run has a 25-minute cap; no ongoing monitor is created. Review of the
published results requires no weights or new run. Publication used only CPU
checks and saved measurements; the original evidence remains unchanged.

## Reproduce the previous dynamics audit

Dependencies are in [requirements.txt](requirements.txt). Reviewing published
metrics needs no checkpoints. This CPU-only check validates tangents, adjoints,
the driven source derivative and an exact two-dimensional SVD case:

```powershell
python new/dynamics_audit/check.py
```

Exact checkpoint replay requires the four retained local `.pt` files whose
hashes are in [the manifest](DYNAMICS_PUBLICATION_MANIFEST.json). Their
repository-relative locations are pinned in `new/dynamics_audit/audit.py::ARMS`.
Checkpoints are excluded from GitHub. With those files present, use new outputs:

```powershell
python new/dynamics_audit/audit.py --preflight --out runs/NEW_DYNAMICS_PREFLIGHT
python new/dynamics_audit/audit.py --out runs/NEW_DYNAMICS_AUDIT
python new/dynamics_audit/analyze.py --run runs/NEW_DYNAMICS_AUDIT --out analyses/NEW_DYNAMICS_REVIEW
```

The analysis command only summarizes saved output on CPU; it also verifies
local checkpoint/source hashes. Freshly retrained weights may differ and will
be rejected by the frozen audit's identity gate. The old training recipes below
remain available, but cross-device checkpoint identity is not promised. No
training or GPU audit was rerun for this publication.

## Reproduce the previous training screen

Run from the repository root using Python with NumPy and CUDA PyTorch. The
recorded environment used PyTorch 2.5.1, NumPy 1.26.4 and an RTX 4060 Laptop GPU;
backend settings are recorded in the publication manifest. No dataset download
is needed. Every output directory must be new.

```powershell
python new/masked_state/test_state.py
python new/masked_state/run_state.py --preflight --out runs/NEW_STATE_PREFLIGHT
python new/masked_state/run_state.py --out runs/NEW_STATE_CONTROL
```

The runner reuses the original trainer/evaluator and verifies frozen source and
reference hashes. Existing three controls are included as public evidence.
The separate `new/masked_state/recover_tail.py` helper only completes missing
planned auxiliaries from a local checkpoint; it never trains. Its published
recovery record identifies exactly which fields were added. Checkpoints remain
local and are not needed to review the scientific endpoints.

The audit requires the existing masked inertial checkpoint, whose SHA is in
the audit manifest. Checkpoints remain local. On a fresh clone, generate it with
`python new/masked_medium/run_masked.py --out runs/masked_medium_20261001_seed0`,
then run `python new/trajectory_audit/audit.py --out runs/NEW_TRAJECTORY_AUDIT`.
The audit reconstruction gate stops on a mismatch; cross-device bitwise replay
is not promised. All published audit measurements can be reviewed without a
checkpoint. This publication schedules no new runs.

For the earlier unmasked four-arm screen:

```powershell
python new/nca_inertial_wind_tunnel/test_core.py
python new/nca_inertial_wind_tunnel/math_checks.py --out runs/NEW_INERTIAL_CHECKS/linear_checks.json
python new/nca_inertial_wind_tunnel/run_wind_tunnel.py --device cuda --arms all --preflight --minutes 5 --out runs/NEW_INERTIAL_PREFLIGHT
python new/nca_inertial_wind_tunnel/run_wind_tunnel.py --device cuda --arms all --seed 0 --steps 800 --minutes 25 --out runs/NEW_INERTIAL_SCREEN
```

Preflight is three optimizer updates per arm and is not efficacy evidence.
The last command starts the full screen. Cross-device bitwise determinism and
speedup are not claimed. This publication does not schedule another run.

## Earlier implicit-message experiments

This project tests whether a visual field model can retain local state, exchange
narrow messages through a learned symmetric medium, and learn distant dependence.
**The earlier A0 result is a negative qualification in 2D:** attention solves all
reported conditions in four seeds, while raw and confidence-normalized RT both
fail training-scale fitting. **3D remains inconclusive:** one attention seed
fails, so all 3D RT arms are skipped. There is no evidence of an RT advantage.

The frozen conditional schedule completed **24/24 trials**. Forty was the maximum
if both dimensions passed calibration; the protocol excluded 16 3D RT trials.
Read [current results](RESULTS.md) and the [A0 per-seed table](evidence/a0_v2_1/RESULTS.md).
Gates B/C and width sweeps were not run. Historical v1 evidence remains unchanged.

## Historical v1 excerpt

At test distance 128, accuracy (%) for training seeds 1729 / 2718:

| Model | 2D | 3D |
|---|---:|---:|
| NCA, 8 updates | 53.1 / 49.2 | 51.6 / 53.1 |
| Attention control | 100.0 / 49.2 | 100.0 / 48.4 |
| Constant transport | 57.8 / 49.2 | 52.1 / 51.0 |
| Learned transport | 52.3 / 49.2 | 51.6 / 49.5 |

The 3D screen uses genuine three-axis transport and Conv3d on narrow volumes
with a 4x4 transverse cross-section, rotating the long axis. It does not test
general curved 3D geometry. CNN is width/depth-matched rather than parameter-
matched; attention is a controlled SDPA message block rather than a tuned ViT.

## Historical A0/v1 reading route

The A0 update followed `c669e18`; its evidence head is `74d7725`.

1. [RESULTS.md](RESULTS.md): the A0 and historical v1 sections follow the latest inertial result.
2. [GPT_CONTEXT.md](GPT_CONTEXT.md): claim boundaries, code symbols and evidence routing.
3. [A0_PROTOCOL.md](A0_PROTOCOL.md): current frozen intervention and stopping rules;
   [PROTOCOL.md](PROTOCOL.md) describes historical v1.
4. [ARCHITECTURE.md](ARCHITECTURE.md) and [THEORY.md](THEORY.md): tensor flow and four short checks.
5. Read the [A0 summary](evidence/a0_v2_1/summary.json),
   [configuration](evidence/a0_v2_1/config.json) and
   [checks](evidence/a0_v2_1/checks.json) before the
   [A0 raw aggregate](evidence/a0_v2_1/aggregate.json).
   Historical v1 has its [small summary](evidence/qualification_v1_1/summary.json) and
   [configuration](evidence/qualification_v1_1/config.json) before opening the
   [full aggregate](evidence/qualification_v1_1/aggregate.json), which includes
   all 20 runs, histories, axis-level evaluations and timings.

## Run from the repository root

Reproducing A0 (requires CUDA, no silent CPU substitution):

```powershell
python a0.py check --out runs/new_a0_checks
python a0.py qualify --out runs/new_a0_run --checks runs/new_a0_checks/checks.json
```

Directories must be new. A0 requires successful checks for the exact current
Python sources, runs four seeds at 480 updates and stops by its 25-minute
scheduling cap. No recurring monitor is installed. A0 raw and normalized both
emit q=c*v, start with matched parameters and medium scale, and perform packed
numerator/confidence transport; only the receiver division changes within a
pair. A0 raw differs from v1 raw. Fixed axis splitting is retained and evaluated
by axis; no axis-equivariance claim is made.

Historical v1 reproduction:

Python 3.12 and PyTorch 2.5.1 were used. Install the matching CUDA-enabled
PyTorch build for your machine; `requirements.txt` lists the Python dependencies.
CPU execution works, but its timing and 25-minute budget are not comparable to
the recorded GPU run. No datasets or model downloads are required.

```bash
python run.py check
python run.py smoke --out runs/new_smoke
python run.py qualify --out runs/new_qualification --minutes 25
python run.py plot --run runs/new_qualification --out analyses/new_qualification
```

Each output directory must be new. Qualification launches training; `check`
validates the solver and adjoint, and `smoke` checks model forward/backward paths.
The default frozen schedule covers both dimensions and stops escalation when
a gate does not pass. Reproduction is subject to device/library differences;
cross-device bitwise determinism is not claimed.

## Included evidence

[A0_PUBLICATION_MANIFEST.json](A0_PUBLICATION_MANIFEST.json) binds the A0 code,
protocol and exported completed run. Its [numerical/oracle checks](evidence/a0_v2_1/checks.json)
establish software and fixed-signal recoverability, not learned-model success.
Checkpoints, private machine receipts, transient logs and caches remain local.
The following manifest and figure describe only the preserved historical v1.

[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) identifies the canonical
run, original training-source hashes and sanitized runtime provenance. Original
source hashes match the published training code; `run.py` is a packaging-only
entry point added afterward. [Validation](evidence/validation/smoke.json) records
the original 2D/3D smoke checks. Local checkpoints, caches, launch receipts and
the interrupted orientation-confounded pilot are excluded; its exclusion reason
is preserved in [RUN_MANIFEST.md](RUN_MANIFEST.md).

The [solver and adjoint check record](evidence/validation/transport_checks.json)
was rerun during packaging. These checks establish software validity only.

The [recorded figure](evidence/qualification_v1_1/gate_a.png) shows means and the
range across two seeds, not confidence intervals. Prefer the per-seed table
when interpreting the unstable attention control. Portable float32 PyTorch PCR
was slower than attention at these grid sizes; no optimized-kernel advantage is
claimed.
