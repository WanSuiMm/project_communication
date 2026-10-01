# Cellular computation: reaction-transport and inertial NCA screens

This repository tests whether changing the medium and update rule improves
persistent cellular computation. The latest **Workspace + Revision paired
screen** completed all four 600-update runs. It **fails the joint gate**:
revision's mean hold effect is **-9.02 pp** across two paired seeds. Seed 0
reaches and holds excellent accuracy, including at doubled spatial size, but
retains the old answer after a source change (0% changed-component accuracy
versus 100% from a cold start at size32/K64). Seed 1 stays at 50% BA.
This supports a narrower single-seed reach/hold observation, not a successful
general revision rule. Both arms share the workspace split; only their Z
update differs. [Full interpretation](evidence/workspace_revision_paired01/INTERPRETATION.md).

## Start here

For incremental review from `b992e2e`, begin with [GPT_HANDOFF.md](GPT_HANDOFF.md).

1. [RESULTS.md](RESULTS.md) and
   [current interpretation](evidence/workspace_revision_paired01/INTERPRETATION.md).
2. [Compact metrics](evidence/workspace_revision_paired01/analysis.json) and
   [both-seed overview](evidence/workspace_revision_paired01/overview.png).
3. [Frozen protocol](new/workspace_revision/PROTOCOL.md),
   [cell](new/workspace_revision/revision_cells.py),
   [trainer/evaluator](new/workspace_revision/run_revision.py), and
   [GPT_CONTEXT.md](GPT_CONTEXT.md).
4. [Validation](evidence/workspace_revision_paired01/validation.json) and
   [publication hashes](REVISION_PUBLICATION_MANIFEST.json).
5. Historical context: [generic dynamics audit](evidence/dynamics_audit_seed0/INTERPRETATION.md)
   and [final medium/recurrence comparison](evidence/masked_state_seed0/RESULTS.md).

The four current [raw arm records](evidence/workspace_revision_paired01/raw/)
and [training schedules](evidence/workspace_revision_paired01/schedules/) are
secondary, as are the older audit's large per-map files. Start with the small
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

## Reproduce the current paired screen

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
