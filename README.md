# Cellular computation: reaction-transport and inertial NCA screens

This repository tests whether changing the medium and update rule improves
persistent cellular computation. The latest **short-BPTT Phase II** completed
12 matched runs: additive cells, K8/K16/K64, four new initialization seeds,
one fixed training bank and schedule. **K8 reaches and holds in 2/4 seeds,
below the frozen 3/4 replication criterion.** K16 reaches in 3/4 but reaches
and holds in only 1/4; all four K64 controls remain unqualified. Two new K8
positive cases retain high accuracy in the selected narrow distance band,
including on larger grids, while farther propagation is weak. This prospective
follow-up tests strict graph distance **16<d<32** on new maps; it does not amend
Phase I's broader failed gate. [Current results](evidence/short_bptt_phase2_init2345/RESULTS.md).

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

For incremental review from `7d1e326`, begin with [GPT_HANDOFF.md](GPT_HANDOFF.md).

1. [RESULTS.md](RESULTS.md) and
   [current screen report](evidence/short_bptt_phase2_init2345/RESULTS.md).
2. [Compact metrics](evidence/short_bptt_phase2_init2345/analysis.json) and
   [all 72 distance/rollout rows](evidence/short_bptt_phase2_init2345/curves.csv).
3. [Frozen protocol](new/short_bptt_phase2/PROTOCOL.md),
   [gradient accumulation and detachment](new/short_bptt_phase2/training.py),
   [runner and decisions](new/short_bptt_phase2/run.py), and
   [GPT_CONTEXT.md](GPT_CONTEXT.md).
4. [Validation](evidence/short_bptt_phase2_init2345/validation.json),
   [analysis code](new/short_bptt_phase2/analyze.py), and
   [publication hashes](BPTT_PHASE2_PUBLICATION_MANIFEST.json).
5. Historical context: [Phase I](evidence/short_bptt_paired01/RESULTS.md),
   [source-switch audit](evidence/switch_audit_seed0/RESULTS.md),
   [prior training screen](evidence/workspace_revision_paired01/INTERPRETATION.md),
   and [generic dynamics audit](evidence/dynamics_audit_seed0/INTERPRETATION.md).

The twelve current [raw arm files](evidence/short_bptt_phase2_init2345/raw/) and
[shared training schedule](evidence/short_bptt_phase2_init2345/schedule.json) are byte-identical
copies of completed outputs. They are secondary. Start with the small
summaries above. Checkpoints, machine receipts, duplicate ZIPs
and transient logs are excluded. Original publication manifests remain valid.

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

## Reproduce the current short-BPTT Phase II

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
