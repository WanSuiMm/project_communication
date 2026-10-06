# Reaction-Transport: local 2D / 3D qualification

## Hybrid Writer v0: completed formal result

Frozen protocol [hybrid_lane_write_budget_v1](new/hybrid_writer/PROTOCOL.md),
[cells and mechanical write contract](new/hybrid_writer/cells.py), and
[serial runner](new/hybrid_writer/run.py). Four arms compare neural, budget,
hybrid and affine_hybrid with C24/Z8 and unchanged masked transport/E/Q/readout.
The bounded writer limits per-lane single-step RMS correction to .1; it does not
guarantee semantic closure or full recurrent stability. Eight fresh paired blocks
completed 300 matched reset64x4 super-updates each, with K8.

**Execution and formal aggregation are COMPLETE:** 32/32 trajectories and
416/416 predeclared checkpoint records in 8,864.187 seconds (about 2 h 28 min).
At fixed u300, joint readiness is neural 1/8 and budget, hybrid and affine_hybrid
0/8. Each primary paired contrast has zero candidate-only wins and Holm-adjusted
exact p=1; none passes the frozen qualification rule. Verdict:
`NO_PRIMARY_RELIABILITY_QUALIFICATION`. This fixed lane-write-budget recipe did
not improve frozen reliability; the result gives no causal root diagnosis and
does not reject general Hybrid or NCA designs. Ever-ready counts match u300;
readiness-loss is undefined for the three arms with no ready trajectory.

The formal run saves u0,25,...,300 checkpoints and shared held-out cohort
measurements. CPU initialization/transport/budget checks and all four pre-run
actual-shape eager/CUDA Graph comparisons passed; they are implementation checks,
not efficacy evidence. The published package is under
[evidence/hybrid_writer_20261006](evidence/hybrid_writer_20261006/RESULTS.md),
with the [publication manifest](HYBRID_WRITER_PUBLICATION_MANIFEST.json).
It retains 832 packed trace banks, 416 summaries, 416 telemetry records and 32
curves losslessly compressed as secondary evidence. Checkpoints and machine
receipts remain local. Historical launch details are in
[dispatch](evidence/hybrid_writer_prelaunch_20261006/dispatch.json),
[validation](evidence/hybrid_writer_prelaunch_20261006/validation.json) and
[configuration](evidence/hybrid_writer_prelaunch_20261006/config.json).

Canonical saved-evidence verification from the repository root:

    python -X utf8 -B tools/export_hybrid_writer.py --verify-only

The implementation check is `python -X utf8 -B new/hybrid_writer/check_cells.py`;
the frozen protocol and published [reproduction guide](evidence/hybrid_writer_20261006/REPRODUCTION.md)
describe a fresh run.

Completed zero-training audit (2026-10-06): block7/reset formation and
collapse, [protocol](new/collapse_audit/PROTOCOL.md). The four existing u225,
u250,u275,u300 checkpoints hand off full (C,Z) states at T64 in a4x4
producer/consumer matrix on the same32-map size32/64 banks. Three fixed readout
views separate immediate readout switching from later recurrent effects.
Also apply G275/G300 for one step to the SAME u275 states at t64,128,192,
observed with R275. Total32 matrix and12 single-step units. Eight native
diagonals must exactly reproduce saved Boolean traces. No model training,
optimizer updates, state alignment, or Hybrid implementation in this audit.
Small two-map CUDA handoff smoke passed with zero differing bits. The local run
is `block7_collapse_audit_20261006_01`; its machine/session receipts remain local.
No time cap, watchdog, recurring monitoring or change to frozen prior evidence.
The audit completed32/32 matrix and12/12 single-step units in90.97seconds;
all eight native replay diagonals have zero differing bits. Under fixed R275,
u275 states continued by G300 achieve T256 paired coverage.970374(size32) and
.933684(size64), versus.950144/.840536 under G275. G275 only partially improves
u300-produced states (.489719/.111351 versus G300's.425492/.059384).
The result points to deficient cold-prefix state formation rather than loss
of continuation on already-successful states; it does not isolate encoder
versus recurrent writes or prove a unique contract. See the new
[results](evidence/block7_collapse_20261006/RESULTS.md) and
[raw report](evidence/block7_collapse_20261006/frozen_RESULTS.md). This audit did
not evaluate Hybrid; its separate formal result is summarized above.

Repository-root entry commands:

    python -X utf8 -B new/collapse_audit/check_metrics.py
    python -X utf8 -u -B new/collapse_audit/run.py --out runs/NEW_COLLAPSE_AUDIT
    python -X utf8 -B tools/export_block7_collapse.py --verify-only

Re-running GPU inference requires the original local checkpoints referenced
by published hashes. Saved public-data verification needs no checkpoints.

Completed experiment (2026-10-06): continuous execution-state coverage,
protocol [new/continuous_coverage/PROTOCOL.md](new/continuous_coverage/PROTOCOL.md).
Original C24/Z8 StreamingCell, eight fresh paired initialization/schedule blocks,
reset64x4 versus continuous256. Both use the same minibatch eight maps,256
forward macro-steps,32 K8 backward windows,32 BCE terms divided by32, and one
AdamW update at the end. Parameters are fixed for the whole super-update.
No old warm states cross optimizer updates. Reset necessarily changes encoder
credit frequency as well as numerical state ages; this is disclosed, not corrected.
Checkpoints u0,25,...,300 retain model/optimizer states and paired Boolean traces
on fresh32-map size32/64 cohorts. Formal endpoint stays u300; dense checkpoints
only describe formation/decline. Final reach/retention joint readiness and the
unchanged old Full gate are reported separately. No runtime cap or monitor.
First dispatch: `runs/continuous_coverage_20261006_01` was interrupted after its
first optimizer update. Its process disappeared without a Python error artifact;
the preserved status is stale RUNNING, not proof of a live worker. A CPU lifetime
control reproduced child reclamation after the launching tool shell exits.
See [execution repair](new/continuous_coverage/EXECUTION_FIX.md). The replacement
launcher runs Python in a persistent foreground tool session and verifies it from
a separate tool call. The initial run and receipt remain unchanged.
Replacement run: `runs/continuous_coverage_20261006_02`, passed
[qualification](evidence/continuous_coverage_20261006/runtime_qualification.json). The existing scientific
source hashes are unchanged; only the launcher changed and execution notes were
added. The persistent execution owner is recorded in the local launch receipt.
Advancing progress u250 to u275 was verified across separate tool calls. In this
isolated tool environment a new shell cannot enumerate the other session's PIDs;
use the bounded progress verifier below rather than the launcher's OS-process
`-Verify` option. Neither verifier changes model state or creates a monitor.

Repository-root commands:

    python -X utf8 -B new/continuous_coverage/run.py --check --out analyses/NEW_COVERAGE_CHECK.json
    pwsh -File tools/launch_continuous_coverage.ps1 -RunName NEW_COVERAGE_RUN -Qualification analyses/NEW_COVERAGE_CHECK.json
    pwsh -File tools/verify_continuous_coverage_progress.ps1 -RunName NEW_COVERAGE_RUN -Qualification analyses/NEW_COVERAGE_CHECK.json -OwnerSessionId 12345

The launch command stays running and returns a tool exec session; replace12345
with that session ID in the separate verification call. Do not end that session
before the worker completes. A finished chat reply is separate from the live
execution session. Replacement execution finished COMPLETE16/16 with208/208
dense records, in3439.86 seconds (about57.3 minutes); worker exit code0.
Fixed-u300 joint readiness and unchanged old Full were0/8 in both arms.
Frozen verdict: `NO_CONTINUOUS_COVERAGE_RELIABILITY_QUALIFICATION`.
Public [results](evidence/continuous_coverage_20261006/RESULTS.md),
[summary](evidence/continuous_coverage_20261006/summary.json),
[dense metrics](evidence/continuous_coverage_20261006/metrics.csv), and
[reproduction](evidence/continuous_coverage_20261006/REPRODUCTION.md).
Verify the public saved-data export with
`python -X utf8 -B tools/export_continuous_coverage.py --verify-only`.

Completed zero-training audit (2026-10-06): semantic-write source protection,
protocol [new/semantic_write_audit/PROTOCOL.md](new/semantic_write_audit/PROTOCOL.md).
Execution is COMPLETE396/396 on the historical local CUDA backend.
The first detached dispatch exited before its
first measurement. The second dispatch saved75/396 units, then a transient
Windows file lock interrupted atomic status-file replacement. Both interrupted
records remain unchanged. The local writer now retries transient reader locks;
its Windows lock test passed. The same scientific protocol resumes in a new
directory after validating and importing all75 completed units byte-identically.
See [execution repair](new/semantic_write_audit/EXECUTION_FIX.md).
Take factorial blocks00..07, all four final-u300 arms, plus the selected original
Streaming seed4 reference. Two existing32-map cohorts (sizes32/64), six conditions,
396 model/size/condition units; no parameter updates, runtime cap or monitor.
The primary endpoint is unprotected strict16<d<32 paired coverage atT256 in the
eight native blocks, comparing source-only negative semantic-write protection
against natural rollout and a same-norm orthogonal source perturbation. Oracle
solved-cell protection and full nullspace ablation are auxiliary; protected-cell
retention is a manipulation check, never evidence for the causal hypothesis.
Eighteen CPU projection/wiring checks, short CUDA intervention smoke and complete
natural256-step replay at both actual32-map shapes passed before dispatch.
Both preflight natural replays have zero mismatched Boolean bits. All64 formal
factorial natural replays also pass; the selected seed4 reference qualifies on
the common cohort. Primary source-natural mean delta=0.0005684167,
source-orthogonal=0.0004133940, positive blocks2/8:
`NO_PRIMARY_SOURCE_DRIVER_SIGNAL`. Source preservation does not rescue distant
computation in this intervention. Oracle gains cannot rescue this verdict.
Continuation worker elapsed1318.938seconds excludes earlier imported-unit work.
Results remain in `runs/semantic_write_20261006_03/`; public
[results](evidence/semantic_write_20261006/RESULTS.md),
[summary](evidence/semantic_write_20261006/summary.json) and
[validation](evidence/semantic_write_20261006/validation.json) retain this negative
conclusion without changing previous frozen verdicts.

Repository-root entry commands:

    python -X utf8 -B new/semantic_write_audit/run.py --check --out analyses/NEW_SEMANTIC_CHECK.json
    python -X utf8 -u -B new/semantic_write_audit/run.py --out runs/NEW_SEMANTIC_RUN --qualification analyses/NEW_SEMANTIC_CHECK.json

Validated continuation of an interrupted run:

    python -X utf8 -u -B new/semantic_write_audit/run.py --out runs/NEW_SEMANTIC_RUN --qualification analyses/NEW_SEMANTIC_CHECK.json --resume-from runs/INTERRUPTED_SEMANTIC_RUN

Completed experiment (2026-10-05): C8 read parameterization / R2 factorial,
protocol [new/latent_factorial/PROTOCOL.md](new/latent_factorial/PROTOCOL.md).
Four arms (native, factorized, native_r2, factorized_r2) share copied native
C8/Z8 core tensors within each of 32 fresh paired training blocks: 128
trajectories, 300 updates each, original K8 credit and unchanged Full gate.
The primary contrast is factorized_r2 minus native; qualification requires
net gain >=8/32 and exact paired two-sided p<=.05. Sidecar effects bundle
memory, extra computation and parameters. No runtime cap or recurring monitor.
Execution is **COMPLETE**,128/128 in4361.047seconds. Each of the four arms
records0/32 Full passes; D-A has0 wins/0 losses, delta0 and exact two-sided p=1,
with verdict `NO_D_MINUS_A_RELIABILITY_QUALIFICATION`. This preserves the
negative frozen-recipe result without claiming model equivalence or universal
latent/sidecar failure. Combined qualification passed: nontrivial initial-state matching, four CPU
checks, five actual-shape eager/graph updates per arm with bitwise-equal
losses/gradients/parameters/Adam states, and the three-state evaluation trace.
All128 trajectories used the captured K8 runtime. The short five-update
qualification is not a full300 equivalence claim. Raw results remain locally
in `runs/latent_factorial_20261005_01/`. Start with the public
[results](evidence/latent_factorial_20261005/RESULTS.md),
[summary](evidence/latent_factorial_20261005/summary.json),
[validation](evidence/latent_factorial_20261005/validation.json), then
[per-arm records](evidence/latent_factorial_20261005/perarm.json) and
[reproduction](evidence/latent_factorial_20261005/REPRODUCTION.md).

Repository-root entry commands:

    python -X utf8 -B new/latent_factorial/run.py --check --out analyses/NEW_FACTORIAL_CHECK.json
    pwsh -File tools/launch_latent_factorial.ps1 -RunName NEW_FACTORIAL_RUN -Qualification analyses/NEW_FACTORIAL_CHECK.json
    python -X utf8 -B tools/export_latent_factorial.py --verify-only

Completed experiment (2026-10-05): native execution-state width qualification,
protocol [new/latent_width/PROTOCOL.md](new/latent_width/PROTOCOL.md), execution
amendment [new/latent_runtime/ACCELERATION.md](new/latent_runtime/ACCELERATION.md).
All 96 planned arms completed across 16 blocks. W24, W8, the parameter-matched
W24 control, W16, W4 and exploratory W2 alternating-axis each have 0/16 Full
passes. The frozen W8-versus-W24 primary verdict is
`NO_W8_RELIABILITY_QUALIFICATION` (0 wins, 0 losses, net 0/16, exact two-sided
`p=1`); the concurrent W24 arm is also 0/16. This result is limited to the
frozen native-width recipe and does not establish universal impossibility.

Runtime provenance: 13 completed eager arms were imported and 83 ran with CUDA
Graph acceleration. Exact 300-update block00 replay passed for all six arms,
with bitwise-equal final parameters and Adam state/groups. That finite check
qualifies the runtime path only. The 2771.016-second elapsed field excludes the
original eager run and separate replay qualification, so it is not end-to-end
time for all 96 trajectories.

Read the public package in this order: [results](evidence/latent_width_20261005/RESULTS.md),
[summary](evidence/latent_width_20261005/summary.json),
[validation](evidence/latent_width_20261005/validation.json),
[configuration](evidence/latent_width_20261005/config.json), and
[reproduction](evidence/latent_width_20261005/REPRODUCTION.md), then
[per-arm results](evidence/latent_width_20261005/perarm.json) and
[metrics](evidence/latent_width_20261005/metrics.csv). Full losslessly
compressed per-block evaluation arrays, summaries, frontier CSVs and curves
are in `block00/`–`block15/`; frozen plans are in `plans.json.gz`. Verify the
saved snapshot with `python -X utf8 -B tools/export_latent_width.py --verify-only`.

Previous completed audit (2026-10-05): `binary_carrier_causal_compression_v1`,
defined in [new/binary_carrier/PROTOCOL.md](new/binary_carrier/PROTOCOL.md).
Formal execution is **COMPLETE**, 24/24 cells in 473.469 seconds. The W
replication anchors qualify, but all three combined primary gates
(`bit_once`, `bit_repeat8`, `source_once`) fail at both sizes. One local linear
decoder was fit on calibration W; class-centroid templates are fixed. Recurrent updates and template
optimization are zero.

The oracle-only centroid arm has sustained progress 0.9967/0.6390 and
keep-continuous 0.1193/0.0896 at sizes 32/64, missing the keep gate at both
sizes. This rejects the tested centroid/consumer combination under the frozen
protocol, not all possible binary codes. The results concern a selected
producer, consumer and finite projected continuation.

Read the public package under `evidence/binary_carrier_20261005/` in this order:
`RESULTS.md`, `summary.json`, `validation.json`, `config.json`, then
`decoder/fit.json`, `decoder/probes.json`, and `metrics.csv`. Raw arrays and
per-map records are secondary. Verify the saved-data snapshot from the
repository root with
`python -X utf8 -B tools/export_binary_carrier.py --verify-only`.

For a new run with fresh output names, use these repository-root commands:

    python -X utf8 -B new/binary_carrier/check.py --out analyses/NEW_BINARY_CHECK.json
    pwsh -File tools/launch_binary_carrier.ps1 -RunName NEW_BINARY_RUN -Qualification analyses/NEW_BINARY_CHECK.json

Previous completed suite (2026-10-04): `w_medium_qualification_v1`, defined in
[new/w_medium/PROTOCOL.md](new/w_medium/PROTOCOL.md). Formal execution is
**COMPLETE**, 36/36 intervention cells in 570.406 seconds, with zero training
or optimizer updates. Stage 1 qualified W replication and enabled the controls
at both sizes. Native Full is true for S1 and false for F1; the historical
localization episode predicate qualifies at both sizes. On fixed fresh
128-map banks, SF keep-continuous is 1.0000/0.9899 and sustained progress is
0.5726/0.5925. Read the canonical public package under
`evidence/w_medium_20261004/`: `RESULTS.md`, `summary.json`,
`validation.json`, `config.json`, `REPRODUCTION.md`, then `metrics.csv` and
`w_medium.png`. Per-arm proxy flags and support counts remain in the evidence.
Verify the published snapshot from the repository root with
`python -X utf8 -B tools/export_w_medium.py --verify-only`.

Canonical new-suite commands from the repository root:

    python -X utf8 -B new/w_medium/check.py --out analyses/NEW_W_MEDIUM_CHECK.json
    pwsh -File tools/launch_w_medium.ps1 -RunName NEW_W_MEDIUM_RUN -Qualification analyses/NEW_W_MEDIUM_CHECK.json

Completed zero-training audit (2026-10-04):
`output_preserving_state_factorization_v1`, 16/16 intervention cells in
74.172 seconds. Fixed S1/F1 update300 producers, F1-only consumer, T64 handoff,
and eight W/Z/readout-span/readout-null combinations on fresh 32-map banks at
sizes 32 and 64. On fixed shared-solved cells, SF continuous preservation is
1.000 at both sizes; fixed shared-unsolved sustained progress is 0.5152/0.6730.
The primary episode is
`LOCALIZATION_EPISODE_UNQUALIFIED`; confirmation is `UNQUALIFIED`. SS rescue
fails the frozen support gate because the progress cohort has 6 valid maps at
size 32 and the keep cohort has 9 at size 64 (minimum 16). The finite
selected-checkpoint intervention does not establish encoded information, a
consistency relation, or population reliability.

Public route: [results](evidence/state_factorization_20261004/RESULTS.md),
[summary](evidence/state_factorization_20261004/summary.json),
[validation](evidence/state_factorization_20261004/validation.json),
[configuration](evidence/state_factorization_20261004/config.json), and
[reproduction notes](evidence/state_factorization_20261004/REPRODUCTION.md);
then metrics CSV and figure in the same directory. Raw outputs are secondary.
The frozen [protocol](new/state_factorization/PROTOCOL.md) and exact source
symbols are mapped in [GPT_CONTEXT.md](GPT_CONTEXT.md). Verify the published
snapshot from the repository root with:

    python -X utf8 -B tools/export_state_factorization.py --verify-only

Completed zero-training audit (2026-10-04): `credit_locality_formation_v1`,
output `runs/credit_formation_20261004_01`; 7/7 checkpoints in 138.5 seconds.
The verdict is `CREDIT_SYNCHRONY_NOT_SUPPORTED`: at size 32, preservation and
progress collapse between updates 175 and 180, but overall gradient cosine and
`C_parallel` rise. Overall, F, and Q synchrony qualifications fail. Native Full
passes are 0/7 checkpoints on the reused held-out banks; update 200 misses
strict T64 pooled reach (0.76648 versus 0.80). This finite within-trajectory association screen establishes neither
causality nor broad impossibility. No model training or optimizer updates ran.

Public route: [results](evidence/credit_formation_20261004/RESULTS.md),
[summary](evidence/credit_formation_20261004/summary.json), validation,
configuration, reproduction notes, metrics CSV and figure in the same directory.
Per-update gradient metadata, continuation and phenotype records are secondary.
The frozen [protocol](new/credit_formation/PROTOCOL.md) and exact source symbols
are mapped in [GPT_CONTEXT.md](GPT_CONTEXT.md). Verify from repository root:

    python -X utf8 -B tools/export_credit_formation.py --verify-only

Completed zero-training audit (2026-10-04): `continuation_interface_v1`,
output `runs/continuation_interface_20261004_01`; formal run `COMPLETE`, 50
transfer cells in 439 seconds. Common native Full qualification passes for
S1/S2, and the exact gauge control passes. Raw S1↔S2 off-diagonal transfer
fails while S1→F1 passes. With the frozen 100-coefficient linear alignment,
S2→S1 and S2→F2 pass while the reverse S1→S2 direction fails. This is a
selected-checkpoint interface audit; it does not establish a common algorithm.
No model training or optimizer steps were run. Restricted off-diagonal alignment
qualification is 0/20 and matched short-observable flags are 0/6, separate from
the transfer proxies; all flags remain in the raw data.

Public reading route: [results](evidence/continuation_interface_20261004/RESULTS.md),
[summary](evidence/continuation_interface_20261004/summary.json),
[validation](evidence/continuation_interface_20261004/validation.json),
[configuration](evidence/continuation_interface_20261004/config.json), and
[reproduction notes](evidence/continuation_interface_20261004/REPRODUCTION.md);
then `selection.json`, `alignment/validation.json`, `phase0/comparisons.json`,
`cross/decomposition.json`, and `gradients/metadata.json` in that evidence
directory. Full per-cell records and larger arrays are secondary. The frozen
protocol is [new/continuation_interface/PROTOCOL.md](new/continuation_interface/PROTOCOL.md);
code entry points are `run.py`, `alignment.py`, `metrics.py`, `gradients.py`,
and `check.py` in `new/continuation_interface/`. Verify the public snapshot
from the repository root with
`python -X utf8 -B tools/export_continuation_interface.py --verify-only`.

Previous completed suite (2026-10-04): `trajectory_qualification_v1`. Formal
execution is `COMPLETE`, 48/48 arms (stage 1: 8/8, stage 2: 32/32, stage 3:
8/8; 6,610.797 seconds). Stage 1 selected the conditional update-3 donor
recipe `shadow_joint`. Stage 2 has 1/16 baseline and 1/16 treatment Full
passes and is `NO_RELIABILITY_QUALIFICATION`. Stage 3 is
`SECOND_TASK_TRAINING_UNQUALIFIED`: 0/8 arms met the size-32 T64 short-task
training gate, and 0/8 passed the separate distance-task proxy. The original
cell and K8 credit window are unchanged; stage 3 is not the original Full gate.

Canonical public entries: [suite results](evidence/trajectory_qualification_20261004/RESULTS.md),
[suite summary](evidence/trajectory_qualification_20261004/summary.json),
[validation](evidence/trajectory_qualification_20261004/validation.json),
and [reproduction notes](evidence/trajectory_qualification_20261004/REPRODUCTION.md).
Read the stage results and summaries under `stage1_state_cross/`,
`stage2_fresh_recipe/`, and `stage3_distance/`; the frozen definitions are in
`new/trajectory_qualification/PROTOCOL.md` and
`new/trajectory_qualification/DISTANCE_PROTOCOL.md`. A stage 1 rerun requires
the preserved update-3 checkpoints. Verify the published snapshot from the
repository root with:

    python -X utf8 -B tools/export_trajectory_qualification.py --verify-only

Preceding completed experiment (2026-10-04): seed4 early-bootstrap versus
training-path sensitivity, new/bootstrap_path/PROTOCOL.md
(`seed4_bootstrap_path_v1`). Formal run: runs/bootstrap_20261004_seed4_01.
COMPLETE23/23 arms x300 updates in3068.407 seconds. Historical H, two selected schedules,
first1/3/8 prefix/suffix crosses, same-donor pre-180 replacements and two exact
batch-multiset row swaps. Original cell, K8 training, AdamW, banks and complete
phenotype thresholds remain unchanged; optimizer state is never reset at a
splice. No runtime cap, watchdog or continuous monitoring. Historical H and
both S controls reproduce their archived results exactly. CPU/CUDA preflight
checks pass. Publication is data and code for independent analysis:
evidence/bootstrap_path_20261004/RESULTS.md, summary.json, first_step_summary.json,
all23 arm summaries/curves, frontier CSVs,46 Boolean trace NPZs and six
first-step array files. Configuration and validation are in the same directory.
No new mechanism interpretation is supplied. Checkpoints and machine records
remain local. Public hashes: BOOTSTRAP_PATH_PUBLICATION_MANIFEST.json.
Canonical commands from repository root, new output names required:

    python -X utf8 -B new/bootstrap_path/check.py --out analyses/NEW_BOOTSTRAP_CPU.json
    python -X utf8 -u -B new/bootstrap_path/run.py --preflight --qualification analyses/NEW_BOOTSTRAP_CPU.json --out runs/NEW_BOOTSTRAP_PREFLIGHT
    pwsh -File tools/launch_bootstrap_path.ps1 -RunName NEW_BOOTSTRAP_RUN -Preflight runs/NEW_BOOTSTRAP_PREFLIGHT

Latest saved-artifact diagnostic (2026-10-04): continuation-formation proxies,
new/formation_gate/PROTOCOL.md. CPU8.86 seconds,44 saved trace records, no
training or inference. At all21 dense checkpoints100..200 plus300 control,
predictors use only rollout t<=64 and retain old T128/T256 behavior screen.
Size32 early survival>=.95/acquisition>=.20 passes140/145/190/200; long screen
passes only200. Next-update(+5) comparison has0 TP,3 FP,16 TN,1 FN.
All continuous early observables rank200 highest, as does ordinary T64 output
coverage; no incremental or prospective mechanism prediction is established.
No quotient representation or independent consumer closure was measured.
Local entry: runs/formation_gate_20261004_saved01/RESULTS.md and INTERPRETATION.md.
Reproduce saved-artifact arithmetic with a new directory:

    python -X utf8 -u -B new/formation_gate/analyze.py --out runs/NEW_FORMATION_GATE

Latest local diagnostic (2026-10-04): update-zero initialization geometry,
new/initial_geometry/PROTOCOL.md. Zero training, CPU30.31 seconds, exact
historical initial checkpoints2/3/4/5 and four reused size32/64 map banks.
Parameter/source/data/schedule bindings pass; analytic full8/full16/K8-detached16
tangents match autograd within4.77e-7. Seed4 has the most balanced raw lane RMS
and largest measured available relative source-flip separation at endpoint64,
but centered lane balance and feature-spectrum rankings differ. Seed3 has
the best endpoint64 K8 paired kernel-label alignment on all four banks.
This is mixed exploratory initialization evidence, not a causal success rule.
Original first-batch task gradients initially open Q-out weight only; encoder,
F, Q-in and readout weight gradients are zero. Full/K8 tangents are distinct.
No optimizer step, new training gate or GitHub publication was performed.
Local entry: runs/initial_geometry_20261004_update0_01/RESULTS.md and summary.json.
Reproduce with new output names:

    python -X utf8 -u -B new/initial_geometry/audit.py --out runs/NEW_INITIAL_GEOMETRY
    python -X utf8 -B new/initial_geometry/summarize.py --run runs/NEW_INITIAL_GEOMETRY

Latest completed experiment (2026-10-03): joint historical u195/u200
zero-training causal audit, new/audit_195_200/PROTOCOL.md (audit195_200_v1).
Original StreamingCell and checkpoints remain unchanged. The suite records492
conditions across primary16-map and independent diagnostic8-map banks at32/64:
11-point weight interpolation, all16 E/F/Q/R coalitions and all24 block-order
effects, producer/continuation/readout cross cube, sparse role-matched W/Z/lane/
channel/readout-null swaps, local bidirectional transplants, fixed-cohort
survival and one-step update/feature pulses with continuation through256.
Five CPU fixtures pass. GPU preflight01 exactly replays four full257-step
endpoint traces (all Boolean arrays and paired margins, maximum error0).
Actual-state null-readout errors stay below1e-6, instrumented steps and suffix
shams are exact, and semantic telescopes pass. Peak allocated234273792 bytes.
Measured runtime estimate798.47 seconds is descriptive; no cap or watchdog.
Formal dispatch verifies the first completed inference and saves a durable
local launch receipt. Run: runs/audit195_200_20261003_joint01.
COMPLETE492/492 conditions in885.469 seconds; empty stderr and launch-session
exit0, process gone. All four full historical Boolean/margin traces match
exactly and all counterfactual arms remain finite. Primary32 cross-continuation
strict T256: state200/rule195/readout195=1.000; state195/rule200/readout195=.93333.
Readout changes are small; E/F/Q block interactions are large. The apparent
primary32 interpolation jump removes a failure concentrated in map10;195 is
already strong on the independent size32 bank. This supports conditional
state-production/coadaptation sensitivity, not a universal closure mechanism.
One selected training trajectory; independent task maps do not replicate
training seeds. Public entry: evidence/joint195_200_20261003/RESULTS.md.
Saved-artifact CPU validation03 passes all492 cases,1940 fixed cohorts,
80 live/snapshot source bindings,12 reference input hashes,16 swap plans,
and all16-coalition/24-order factorial metrics; maximum numeric error0.
Earlier validator reports are retained locally; fixes changed no run evidence.
Canonical commands from repository root, new output names required:

    python -X utf8 -u -B new/audit_195_200/run.py --preflight --out runs/NEW_AUDIT_PREFLIGHT
    pwsh -File tools/launch_audit_195_200.ps1 -RunName NEW_AUDIT -Preflight runs/NEW_AUDIT_PREFLIGHT

Preceding completed experiment (2026-10-03): historical seed4 dense
update100--200 transition audit, new/transition_100_200/PROTOCOL.md
(seed4_dense_transition_v1). Original architecture and original K8 helper,
bank10002/schedule20002/init4 unchanged. Replay300 updates to verify exact
historical anchors0/100/200/300 and all six final historical endpoints;
save100,105,...,200 plus0/300. After training, audit22 checkpoints on fixed
16-map size32/64 banks61032/61064, paired rollout0..256. Preserve raw Boolean
traces, signed margins, state RMS, exact turnover and censoring-aware survival.
This is one selected training trajectory; no latent commitment, type closure,
causal handoff or physical phase transition is established by a profile change.
CPU01 passed67 source bindings, four historical reference checkpoint hashes
and five hand-computed metric fixtures. CUDA preflight01 passed three updates,
six finite/nonzero gradient groups and both full256-step paired smoke audits;
peak allocated110959616 bytes, estimated formal357.03 seconds (descriptive).
Formal dispatch verified the first optimizer update. No runtime limit,
watchdog or continuing monitor. Run: runs/transition_20261003_seed4_dense01.
Independent post-dispatch CPU receipt at
analyses/transition_20261003_dispatch_bindings01.json additionally verifies
both historical evaluation-bank hashes, both saved audit banks, three prior
stage-summary checkpoint bindings and the recorded historical GPU model;
this supplements evidence without modifying the frozen running source.
COMPLETE300 updates,44 audits in353.141 seconds; training112.687 seconds.
Process ended with empty stderr and successful launch-session exit. Anchors
0/100/200/300 and all six complete historical endpoint payloads replay
exactly (5628 integer,2636 floating leaves, maximum error0). Size32 shows
noisy improvement, a marked collapse at180, then strong preservation at200:
195->200 strict T12885.65%->97.55%, first-exit64->1285.15%->0.032%, continuous
survival64->25694.53%->99.97%. Only200 passes the descriptive screen inside
the dense window;205/210 were not observed. Persistent three-checkpoint
onset is therefore unestablished; no new architecture qualification follows.
Canonical public entry: evidence/transition_20261003/RESULTS.md and summary.json.
Saved-artifact CPU validation passes 451 checks: 67 live source and 67 snapshot
hashes, 23 checkpoints, 44 raw trace hashes and behavior recomputations, all
historical anchors/endpoints and three prior stage summaries. Counts match
exactly; maximum numeric error is 6.38e-8 within 2e-7. The first validator report
is retained locally: an absent-field lookup in the older public manifest and
a check-status serialization collision were corrected by cheap binding
finalization. All 44 metric recomputations come from the first full pass;
experimental outputs are unchanged. Recorded RMS/BCE are finite but cannot
be recomputed without hidden-state/logit arrays.
Full trajectory/checkpoint data stay local. Canonical executed commands:

    python -X utf8 -B new/transition_100_200/check.py --out analyses/NEW_TRANSITION_CPU.json
    python -X utf8 -u -B new/transition_100_200/run.py --preflight --qualification analyses/NEW_TRANSITION_CPU.json --out runs/NEW_TRANSITION_PREFLIGHT
    pwsh -File tools/launch_transition_100_200.ps1 -RunName NEW_TRANSITION_RUN -Preflight runs/NEW_TRANSITION_PREFLIGHT

For a fresh clone without original checkpoint archives, use the explicitly
declared public-reference adapter described in
evidence/transition_20261003/REPRODUCTION.md. It checks published parameter
hashes and public stage counts, not unavailable original checkpoint bytes.

Preceding completed experiment (2026-10-03): detached warm-start training-state
distribution control for the ORIGINAL StreamingCell, frozen at
new/warmstart/PROTOCOL.md (detached_warmstart_v1_nocap). Four paired
initializations2/3/4/5, baseline versus warmstart, eight arms300 updates each.
Both arms compute the same no-grad prefix; baseline discards it, warmstart
keeps the detached prefix state for half the batch (ages32/64/128/192), while
the other half stays fresh. Architecture, K8, trained64-step suffix, historical
training bank/schedule and full phenotype gates are fixed. Fresh primary
evaluation uses32 maps each at size32/64, seeds60032/60064; exploratory
stage diagnostics use16 held-out size32 maps, seed61032, at updates0/100/200/300.

CPU02 qualification passed70 source bindings, exact historical baseline
loss/state/all-gradient checks and transient-NaN recovery detection. GPU
preflight02 completed both three-update maximum-age arms; identity, finite
gradient and memory gates passed. Estimated formal time2661.13 seconds
(44.35 minutes) is descriptive. The user removed the runtime limit before
formal launch; no timing gate, deadline or termination watchdog is enabled.
The earlier preflight01 failure under the former cap remains preserved;
new/warmstart/PROTOCOL.md records the execution amendment for public review.
Nonfinite values, source drift and historical control mismatch still stop
execution without rescue. COMPLETE8/8 arms,300 updates each in1629.375 seconds
(27.16 minutes). All four historical baseline initial/final hashes and six
full endpoint records match exactly. Baseline seed4 qualifies on the new
full phenotype. Full passes: baseline1/4(seed4), warmstart0/4; decision
DEVELOPMENT_NOT_QUALIFIED. Partial components change in both directions:
seed2 gains some retention/growth while seed4 loses its full phenotype.
No mechanism, phase transition, or general warm-start rejection follows.
Run: runs/warmstart_20261003_paired01; local checkpoints/traces remain intact.
Public entry: evidence/warmstart_20261003/RESULTS.md and summary.json.
No ongoing monitor was created. Publication uses saved artifacts on CPU,
without new training or GPU inference.
Independent saved-result CPU validation passes583 checks:70 current source/
reference bindings and70 snapshots,8 final plus32 stage checkpoints,24 exact
saved historical replay records,8 fresh phenotype gates from16 Boolean
trace files,32 stage summaries with4096 turnover identities and384 finite
recorded RMS values. Stage Boolean traces were not saved; their survival
statistics are checked from recorded summaries rather than independently
replayed from raw stage trajectories. Public verification and reproduction
notes are at evidence/warmstart_20261003/REPRODUCTION.md.
Canonical commands from repository root, new output names required:

    python -X utf8 -B new/warmstart/check.py --out analyses/NEW_WARMSTART_CPU.json
    python -X utf8 -u -B new/warmstart/run.py --preflight --qualification analyses/NEW_WARMSTART_CPU.json --out runs/NEW_WARMSTART_PREFLIGHT
    pwsh -File tools/launch_warmstart.ps1 -RunName NEW_WARMSTART_TRAINING -Preflight runs/NEW_WARMSTART_PREFLIGHT

Preceding completed follow-up (2026-10-03): three independent original-seed4
follow-ups, frozen at new/seed4_followup/PROTOCOL.md. A changes only four
batch schedules at fixed initialization/bank; B probes four parameter-space
directions at two paired radii with fixed training; C uses the old trained
checkpoint for local temporal-state rollback versus a norm-matched wrong
neighbor and off-cone/no-op controls. Architecture and K8 training unchanged.
Training has one historical replay control plus4 A and8 B arms, each300
updates; one local GPU,40-minute cap. Fresh paired correctness/acquisition
phenotype is measured on32 maps per size32/64 at new seeds50032/50064.
C is a separate zero-training five-minute audit of the historical maps.
Focused CPU qualification passed56 source bindings; three-update CUDA smoke
passed gradient/memory gates and timed a full fresh256-step evaluation at
11.14 seconds. Conservative formal estimate2263.21 seconds (37.72 minutes),
below the40-minute cap. A/B COMPLETE13/13 arms,300 updates each in1866.672
seconds (31.11 minutes). Historical control initial/final hashes and fullsix
endpoint records match exactly; fresh control phenotype QUALIFIED. A0/4
schedules pass the full phenotype; B0/4 directions at each paired radius.
All 12 A/B arms pass both matched-frontier gates but fail other full-gate checks.
Run: runs/seed4_followup_20261003_training01. Publication starts at
evidence/seed4_followup_20261003/RESULTS.md and summary.json. No basin,
population reliability or new architecture qualification follows.
C completed in29.89 seconds at runs/seed4_followup_20261003_causal01.
Six complete historical endpoints replay exactly. Selected93/257 events at
size32/64, each31 eligible maps; frozen population minima passed. Primary
NO_PRIMARY_THRESHOLD_SIGNAL: one-step native-minus-sender effects+14.52/
+7.00pp and wrong-sham-minus-sender+3.76/+0.44pp; thresholds not met at both
sizes. Four-step secondary effects are+33.53/+28.60pp and+20.18/+15.33pp
respectively; these do not rescue the failed primary. CPU saved-result
validation passed55 source bindings, published-reference/checkpoint hashes,
350 event geometries, paired logits, controls and independent equal-map
contrasts at analyses/seed4_followup_20261003_causal01/validation_bound.json.
This is selected local temporal-state sensitivity, not unique semantic handoff.
Training saved-result review passes383/383 checks:56 source/snapshot bindings,
13 CPU checkpoint hashes, all bank/schedule/initialization identities and
Boolean-trace metrics/gates. Public verification passes125459 matched strata
and350 C events. [Public reproduction notes](evidence/seed4_followup_20261003/REPRODUCTION.md)
cover A/B on a fresh clone; exact C replay requires the original local checkpoint.
Commands from repository root, new output names required:

    python new/seed4_followup/check.py --out analyses/NEW_FOLLOWUP_CPU.json
    python new/seed4_followup/train.py --preflight --qualification analyses/NEW_FOLLOWUP_CPU.json --out runs/NEW_FOLLOWUP_PREFLIGHT
    python new/seed4_followup/causal.py --out runs/NEW_FOLLOWUP_CAUSAL
    python tools/validate_seed4_followup_causal.py --run runs/NEW_FOLLOWUP_CAUSAL --out analyses/NEW_FOLLOWUP_CAUSAL_VALIDATION.json
    pwsh -File tools/launch_seed4_followup.ps1 -RunName NEW_FOLLOWUP_TRAINING -Preflight runs/NEW_FOLLOWUP_PREFLIGHT

Preceding descriptive diagnostic (2026-10-03): original Streaming seed4
zero-training behavior audit, protocol new/frontier_audit/PROTOCOL.md.
Recorded every macro step0..256 on the historical32 maps per size32/64.
One selected checkpoint, no optimizer/backward, new model, training or altered
operator. Completed in31.813 seconds; all six complete historical endpoint
payloads replay exactly (5628 integer leaves,2636 floating leaves, max error0).
At size32, all6856 pixels correct at64 remain correct at128 and256; at size64,
18827/19084 (98.65%) remain correct at256. Endpoint retention is not perfect
trajectory monotonicity: across0..256,186/8303 (2.24%) ever-correct pixels
regress at size32, and2768/33328 (8.31%) at size64. Matched map/time/exact-BFS
distance one-step frontier acquisition differences average+30.92/+22.44 pp
over eligible maps, descriptive only. This supports investigating a retained
correctness/acquisition behavior in this checkpoint, not a flood-fill rule,
causal handoff, invariant latent closure, training basin or seed reliability.
Canonical public outputs: evidence/frontier_audit_seed4/RESULTS.md,
summary.json, DIAGNOSIS.md, behavior_curves.png and acquisition_maps.png.
Independent CPU validation at evidence/frontier_audit_seed4/validation.json
passed45 source bindings,64 map/BFS checks,2240 transitions,123296 frontier
strata and192 paired aggregates. Public DIAGNOSIS.md records interpretation
and the additional saved-trace within-interval count. Compact matched-stratum
CSVs preserve the integer frontier counts and independently reproduce effects.
FRONTIER_AUDIT_PUBLICATION_MANIFEST.json binds public evidence and all45
executed sources; metadata copies omit machine identifiers. Original local
run runs/frontier_audit_20261003_seed4_01 and its analyses were preserved;
large traces/behavior JSON and checkpoints remain local.
Five synthetic CPU metric checks passed before the single CUDA audit.
Original evidence/checkpoint preserved; earlier architecture verdicts unchanged.
Commands from repository root, new output names required:

    python new/frontier_audit/check_metrics.py
    python new/frontier_audit/audit.py --out runs/NEW_SEED4_FRONTIER_AUDIT
    python new/frontier_audit/validate.py --run runs/NEW_SEED4_FRONTIER_AUDIT --out analyses/NEW_SEED4_FRONTIER_VALIDATION
    python tools/export_frontier_audit.py --verify-only

Latest completed experiment (2026-10-03): exactly nested stationary-sidecar
development screen, protocol new/stationary_sidecar/PROTOCOL.md.
All12 arms completed300 updates in1676.281 seconds (27.94 minutes).
Scientific decision DEVELOPMENT_NO_GO: stream1/4 reach+hold, memory0/4,
stateless0/4. All four original Streaming final parameter hashes and complete
evaluations exactly reproduce history. Every arm reaches only at seed4.
Seed4 primary pooled T64/T128/T256: stream85.92/99.35/100%,
memory94.24/0/0.38%, stateless100/100/88.49%. Both side branches fail hold;
later stateless seed2 gains cannot replace its failed frozen T64 endpoint.
Original W24/Z8, F/Q, operators, clocks and K8 trainer retained; side arms
have7989 parameters versus stream5033. Extra H carry is the single structural
memory/stateless difference, also changing accumulation/scale. Old Z already
provides local memory. No general memory rejection or causal failure mechanism.
Canonical public report and analysis: evidence/stationary_sidecar_init2345/
RESULTS.md and analysis.json. Validation independently checks45 sources,
12 CPU checkpoint hashes,4 exact controls,144 BA means,1008 paired metrics,
936 curves,624 contrasts, data/schedule/preflight and all gates.
CPU qualification rerun passed; no new scientific training or trained-checkpoint
inference for publication. STATIONARY_SIDECAR_PUBLICATION_MANIFEST.json binds
the unchanged raw evidence and sanitized metadata. Checkpoints/receipts remain
local. Original run runs/stationary_sidecar_20261002_init2345_01 and preflight
runs/stationary_sidecar_20261002_preflight01 were preserved. No new experiment
or continuous monitor was started for result review.
Commands from repository root; new output names required:

    python new/stationary_sidecar/check.py --out analyses/NEW_SIDECAR_CHECK.json
    python new/stationary_sidecar/run.py --preflight --out runs/NEW_SIDECAR_PREFLIGHT
    pwsh -File tools/launch_stationary_sidecar.ps1 -RunName NEW_SIDECAR_RUN -Preflight runs/NEW_SIDECAR_PREFLIGHT
    python new/stationary_sidecar/analyze.py --run runs/NEW_SIDECAR_RUN --preflight runs/NEW_SIDECAR_PREFLIGHT --out analyses/NEW_SIDECAR_REVIEW

Latest completed diagnostic (2026-10-02): zero-training operator audit of
the original Streaming seed4 checkpoint. New frozen protocol/code:
new/stream_path_audit/PROTOCOL.md, operators.py, check.py, audit.py.
Fixed four-way cold rollout switches T_M versus identity and direct masked
Laplacian inputs versus zeros in BOTH F/Q. Same parameters and historical
32-map evaluation banks per size32/64; no other model seeds or training.
All six historical T64/128/256 evaluations replay exactly (maximum error0).
Completed all four conditions in12.36 seconds under the five-minute cap.
Full primary pooled85.92/99.35/100%; either single knockout and both-off
give0/2918 at all three horizons, and size64 d>32 also drops to zero.
Near-cue paired behavior collapses too, so a pure slower-propagation account
is not established. The frozen solution is sensitive to both pathways;
knockouts change learned distributions and effective hop depth (2/2/1/0),
and do not prove the cause of H/C/Z failure or inability to retrain a knockout.
One selected model, n=1; all earlier architecture gate decisions unchanged.
Outputs: runs/stream_path_audit_20261002_seed4_01/RESULTS.md,
INTERPRETATION.md, summary.json, curves.csv, contrasts.json; raw_conditions.json
is secondary. Saved-result validation at
analyses/stream_path_audit_20261002_seed4_01/validation.json passed39 source
bindings,672 paired aggregates/denominators,96 BA means,672 curve rows and
600 contrasts. Original code/checkpoints/evidence were preserved.
Public review begins at evidence/stream_path_audit_seed4/RESULTS.md,
INTERPRETATION.md, summary.json and validation.json. The
STREAM_PATH_AUDIT_PUBLICATION_MANIFEST.json binds all public evidence and code;
manifest/completion copies omit machine metadata. No new inference or training
was performed for publication.
Commands from repository root, new output names only:

    python new/stream_path_audit/check.py
    python new/stream_path_audit/audit.py --out runs/NEW_STREAM_PATH_AUDIT
    python new/stream_path_audit/analyze.py --run runs/NEW_STREAM_PATH_AUDIT --out analyses/NEW_STREAM_PATH_REVIEW

Latest completed experiment (2026-10-02): persistent local, carrier and
task roles. Frozen protocol: new/persistent_roles/PROTOCOL.md. The candidate
has H12 stationary computation, C12 persistent four-direction carrier and Z8
stationary readout state. A shared pointwise R35->72->32 is evaluated twice
per macro-step: (H,C,Z)<-(H+.1*dH,T(C+.1*dC),Z+.5*dZ). H/C split the original
encoder3->24 once; Z0=0; R output layer is zero initialized. Exactly5033
parameters and32 persistent channels match the unchanged baseline/stream.
Only C crosses cells; no Laplacian or ephemeral emitter. Old Streaming
already has stationary Z; this candidate adds dedicated local computation
capacity while changing carrier width, perception, parameter sharing and
readout timing. It does not isolate H causality or match FLOPs/latency.
Candidate K8 has16 phases, carrier radius<=16 and task-readout radius<=15.

Fixed development recipe: seeds2/3/4/5, baseline/stream/roles (12 arms),
300 updates per arm, unchanged historical data/schedule and K8 losses.
Same size32/T64 strict16<d<32 mean+pooled and BA reach gate; hold at both
T128/T256. All eight concurrent controls must reproduce historical full
evaluation and final parameter hashes. DEVELOPMENT_GO requires roles
reach+hold>=3/4 including2/5 and exceeding both controls. The shared seeds
and maps were already inspected: no independent confirmation claim.
Formal hard cap40 minutes; watchdog41. No efficacy tuning, sweep or monitor.
Five focused CPU checks passed: common initialization and5033 parameters,
independent nonzero forward/backward reference and base isometry, phase
readout clock/component isolation, three-state K8 gradient/detach clock,
and decision/hold truth cases. CPU record:
runs/persistent_roles_20261002_cpu_check/checks.json. Three-arm CUDA
preflight passed in8.219 seconds at runs/persistent_roles_20261002_preflight/.
Peak allocated memory baseline87.14/stream88.70/roles84.77MiB. Worst last-two
median update0.4375 seconds gives a frozen estimate2070 seconds (34.50
minutes), within the40-minute hard cap. All51 source bindings, initialization
identities and32-map/2918-pixel primary coverage passed. All12 formal arms
completed300 updates at21:28:55 +08:00 in1204.234 seconds (20.07 minutes).
DEVELOPMENT_NO_GO: baseline reach+hold2/4, stream1/4, roles0/4. All eight
control final parameter hashes and complete evaluations reproduce history.
Candidate primary pooled accuracy by seed2/3/4/5 is0.00/0.31/0.00/1.58%
atT64 and0.00/0.58/0.00/2.09% atT256. Every candidate Hold=True preserves
a failed endpoint. Size64 d>32 remains at most0.24% pooled across all seeds
and measured horizons. This is a negative result for the complete tested
parameterization, not an isolated H effect or family-wide impossibility.
Original outputs/checkpoints/private receipt remain local under
runs/persistent_roles_20261002_init2345/. Public review starts at
evidence/persistent_roles_init2345/RESULTS.md, analysis.json and validation.json;
PERSISTENT_ROLES_PUBLICATION_MANIFEST.json binds source and public evidence.
Saved-result CPU audit passed:51 source hashes against current/formal/preflight
copies,12 checkpoint parameter hashes, all eight exact historical controls,
144 BA means,1008 paired aggregates/independent denominators,12 decisions,
936 curve rows and624 paired contrasts. No new training or checkpoint
inference was performed for publication; originals remain unchanged.
No follow-up training or continuous monitor is scheduled.
Commands from repository root:

    python new/persistent_roles/check.py
    python new/persistent_roles/run.py --preflight --out runs/NEW_ROLES_PREFLIGHT
    pwsh -File tools/launch_persistent_roles.ps1 -RunName NEW_ROLES_RUN -Preflight runs/NEW_ROLES_PREFLIGHT

Previous completed experiment (2026-10-02): local persistent state plus a learned
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
