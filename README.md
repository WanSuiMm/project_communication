# Cellular computation: reaction-transport and inertial NCA screens

This repository tests whether explicit reaction/transport structure improves
persistent cellular computation. The latest **2D, seed-0, same-medium control**
favors generic Masked Momentum at the frozen 32x32/T64 endpoint: BA96.45%
and paired-source correctness77.25%, versus88.79% and37.22% for Masked Inertial RD.
The generic control completes800 updates and all evaluations. This is a negative
primary comparison for the explicit recipe, with a horizon-dependent tradeoff:
at32/T256 the generic control declines to77.68% BA, while explicit RD reaches
91.55%. Neither reaches sustained aggregate95% across subsequent checkpoints.

An inference-only [trajectory audit](evidence/trajectory_audit_seed0/INTERPRETATION.md)
of Masked Inertial RD exactly reproduces all15 historical checkpoints. It finds
open-pixel state growth, increasingly confident errors and substantial component
mean drift. Independent temperature calibration improves late BCE without
changing decisions. These observations do not establish fixed-point, directional
or asymptotic convergence. The detailed audit was not run on Masked Momentum;
its original evaluator also records increasing norms and late BCE deterioration.

Single-seed results do not establish population-level superiority, repair
superiority, a general PDE impossibility claim or a speedup. Hidden widths and
initial spatial coupling differ between the parameter-matched architecture
packages. No extra seed, architecture rescue or continuous monitor was launched.
Earlier implicit-message v1/A0 (2D/3D), unmasked inertial, and masked-medium
evidence are preserved unchanged.

## Start here

For incremental review from `c13b73f`, begin with [GPT_HANDOFF.md](GPT_HANDOFF.md).

1. [RESULTS.md](RESULTS.md): latest comparison, rollout tradeoffs and audit conclusions.
2. [Momentum results](evidence/masked_momentum_seed0/RESULTS.md) and
   [all paired horizons](evidence/masked_momentum_seed0/comparison.json).
3. [Audit interpretation](evidence/trajectory_audit_seed0/INTERPRETATION.md),
   [tables](evidence/trajectory_audit_seed0/RESULTS.md) and
   [summary](evidence/trajectory_audit_seed0/summary.json).
4. [Momentum protocol](new/masked_momentum/PROTOCOL.md),
   [audit protocol](new/trajectory_audit/PROTOCOL.md), and
   [GPT_CONTEXT.md](GPT_CONTEXT.md) for source symbols and claim boundaries.
5. [Publication manifest](MOMENTUM_AUDIT_PUBLICATION_MANIFEST.json) for code,
   reference and evidence hashes. [ARCHITECTURE.md](ARCHITECTURE.md) explains the equations.

Raw per-arm JSON and the6.2MB audit `components.json` are secondary: open only
for specific repair, revision, distance or component-vector questions. Start
with the small summaries above. Checkpoints, machine receipts, duplicate ZIPs
and transient logs are excluded. Original publication manifests remain valid.

## Reproduce the latest screen

Run from the repository root using Python with NumPy and CUDA PyTorch. The
recorded environment used PyTorch 2.5.1, NumPy 1.26.4 and an RTX 4060 Laptop GPU;
backend settings are recorded in the publication manifest. No dataset download
is needed. Every output directory must be new.

```powershell
python new/masked_momentum/test_momentum.py
python new/masked_momentum/run_momentum.py --preflight --out runs/NEW_MOMENTUM_PREFLIGHT
python new/masked_momentum/run_momentum.py --out runs/NEW_MOMENTUM_CONTROL
```

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
