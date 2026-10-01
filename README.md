# Cellular computation: reaction-transport and inertial NCA screens

This repository tests whether explicit reaction/transport structure improves
persistent cellular computation. The latest **zero-training dynamics audit**
compares four existing generic State/Momentum checkpoints at sizes32/64. It
finds positive finite-window gain already early in the rollout, usually declining
as accuracy deteriorates. No near-zero-to-positive maximum-gain crossing is
observed. Persistent updates and increasingly confident errors remain visible.
Forty historical endpoints replay exactly. Spectral precision is limited:
148/256 open K16 estimates converge, and finite worst-direction perturbations
often leave the linear regime. These are diagnostic observations, not an
asymptotic Lyapunov result or an identified causal mechanism.

The final **2D, seed-0, recurrence-by-medium
2x2 screen** adds Masked State-matched NCA. At the frozen 32x32/T64 endpoint,
Masked Momentum reaches **96.45% BA / 77.25% paired-source correctness**, versus
**90.43% / 56.37%** for Masked State. Masking improves both generic recipes on
these two primary metrics, with larger gains for Momentum. State masking also
worsens BCE (0.258 to 0.987), so the improvement is not uniform across metrics.
Both models deteriorate beyond the training rollout: at 32/T256 BA is 77.68%
for Masked Momentum and 11.38% for Masked State. Neither has sustained 95% BA.

The new arm saved 800 updates and all 15 scientific evaluation points, but its
original process left no final completion marker. Only missing size128 timings
and gradient probes were recovered from the existing checkpoint, without
retraining. Two representative endpoints replay exactly; original evidence is
preserved separately. See [recovery provenance](evidence/masked_state_seed0/recovery.json).

An inference-only [trajectory audit](evidence/trajectory_audit_seed0/INTERPRETATION.md)
of Masked Inertial RD exactly reproduces all15 historical checkpoints. It finds
open-pixel state growth, increasingly confident errors and substantial component
mean drift. Independent temperature calibration improves late BCE without
changing decisions. These observations do not establish fixed-point, directional
or asymptotic convergence. That calibration audit applies to explicit Inertial
RD. The newer generic audit above is a separate analysis: generic Masked
Momentum has predominantly within-component state variation, so the older
component-mean explanation cannot simply be transferred to it.

Persistent state capacity is equal (32 scalars), but State uses H32/hidden48/
4993 parameters and Momentum uses H16+V16/hidden88/4689 parameters. This is a
recipe comparison, not an isolated causal test of velocity. Single-seed results
do not establish population-level superiority, repair superiority, a general
PDE impossibility claim or a speedup. The earlier same-medium comparison still
favors generic Momentum over explicit Inertial RD at the primary endpoint.
The latest audit adds diagnostics only; no further training or monitor is scheduled.
Earlier implicit-message v1/A0 (2D/3D), unmasked inertial, and masked-medium
evidence are preserved unchanged.

## Start here

For incremental review from `aa2d50a`, begin with [GPT_HANDOFF.md](GPT_HANDOFF.md).

1. [RESULTS.md](RESULTS.md): current audit, then the preserved training comparisons.
2. [Audit interpretation](evidence/dynamics_audit_seed0/INTERPRETATION.md),
   [64-point table](evidence/dynamics_audit_seed0/RESULTS.md),
   [compact analysis](evidence/dynamics_audit_seed0/analysis.json), and
   [overview figure](evidence/dynamics_audit_seed0/dynamics_overview.png).
3. [Frozen audit protocol](new/dynamics_audit/PROTOCOL.md) and
   [GPT_CONTEXT.md](GPT_CONTEXT.md): derivative definitions and claim boundaries.
4. [Validation](evidence/dynamics_audit_seed0/validation.json) and
   [publication hashes](DYNAMICS_PUBLICATION_MANIFEST.json).
5. For context only: [final 2x2](evidence/masked_state_seed0/RESULTS.md),
   [earlier Momentum comparison](evidence/masked_momentum_seed0/RESULTS.md), and
   [explicit-inertial audit](evidence/trajectory_audit_seed0/INTERPRETATION.md).

The16 [new raw curve/dynamics files](evidence/dynamics_audit_seed0/raw/) and the
older6.2MB audit `components.json` are secondary: open only
for specific repair, revision, distance or component-vector questions. Start
with the small summaries above. Checkpoints, machine receipts, duplicate ZIPs
and transient logs are excluded. Original publication manifests remain valid.

## Reproduce the latest audit

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
