# Cellular computation: reaction-transport and inertial NCA screens

Latest experiment: **Stationary Sidecar — DEVELOPMENT_NO_GO**. All12 arms
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

Latest diagnostic: a **zero-training behavior audit of original Streaming
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

For incremental review from `2e550abfeb1bf53e1348dae7f5653fa6ccd8e47b`, begin with [GPT_HANDOFF.md](GPT_HANDOFF.md).

1. [Current behavior report](evidence/frontier_audit_seed4/RESULTS.md),
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
