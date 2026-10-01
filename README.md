# Cellular computation: reaction-transport and inertial NCA screens

Latest update: a completed **2D masked-medium diagnostic**, seed 0, replaces
only the structured cells' transport operator with input-mask edge weights.
Inertial RD improves at 32x32/T64 from 69.94% to 88.79% balanced accuracy and
from 17.81% to 37.22% paired-source correctness; first-order RD gets worse.
The inertial arm passes the prespecified descriptive joint +5pp criterion,
but 128x128/T256 remains at chance and no sustained aggregate 95% endpoint is
reached. No same-medium generic momentum control or multi-seed claim exists.
Start with [the masked results](evidence/masked_medium_seed0/RESULTS.md),
[paired comparisons](evidence/masked_medium_seed0/paired_comparison.json),
[protocol](new/masked_medium/PROTOCOL.md), then [GPT_HANDOFF.md](GPT_HANDOFF.md).
Earlier frozen results below are preserved.

Reproduce this diagnostic from the repository root:

```powershell
python new/masked_medium/test_masked.py
python new/masked_medium/run_masked.py --preflight --out runs/NEW_MASKED_PREFLIGHT
python new/masked_medium/run_masked.py --out runs/NEW_MASKED_SCREEN
```

This repository preserves two distinct architecture experiments: implicit
message transport (historical v1/A0, 2D/3D) and an explicit inertial NCA
(latest screen, **2D only**). The latest question is whether a prescribed
reaction/transport decomposition improves persistent spatial computation over
generic momentum NCA. **The completed seed-0 screen is negative for the current
recipe:** inertial RD has 69.94% balanced accuracy at 32x32/T64 versus 84.44%
for generic momentum, and degrades with longer rollouts. No arm reaches the
sustained aggregate 95% criterion; candidate repair is unevaluable because no
pre-damage map qualifies. This is not a family-wide impossibility result.

All four arms completed 800 updates and every planned evaluation. Read the
[new result table](evidence/inertial_seed0/RESULTS.md) and
[incremental handoff](GPT_HANDOFF.md) first. No extra seeds, 3D extension,
rescue sweep or recurring monitor was launched. Historical evidence is unchanged.

## Start here

For the latest incremental review from `48095e5`, begin with [GPT_HANDOFF.md](GPT_HANDOFF.md).

1. [RESULTS.md](RESULTS.md): latest result and limits, followed by earlier A0/v1.
2. [GPT_CONTEXT.md](GPT_CONTEXT.md): independent experiment scopes, status and exact code symbols.
3. [WIND_TUNNEL.md](new/nca_inertial_wind_tunnel/WIND_TUNNEL.md) and
   [LOCAL_INTEGRATION.md](new/nca_inertial_wind_tunnel/LOCAL_INTEGRATION.md): frozen recipe, metric fixes and sampler conventions.
4. [summary.json](evidence/inertial_seed0/summary.json),
   [config.json](evidence/inertial_seed0/config.json) and
   [validation.json](evidence/inertial_seed0/validation.json): all horizons and implementation checks.
5. [cells.py](new/nca_inertial_wind_tunnel/cells.py),
   [ARCHITECTURE.md](ARCHITECTURE.md) and
   [inertial theory](new/nca_inertial_wind_tunnel/THEORY.md): equations, shapes and limits.

Open the four per-arm JSON files under `evidence/inertial_seed0/arms/` only for
repair/revision controls, distance bins or detailed timings. Start with the
small summaries above. [INERTIAL_PUBLICATION_MANIFEST.json](INERTIAL_PUBLICATION_MANIFEST.json)
binds exact executed sources, protocol documents and byte-identical per-arm
metrics. Checkpoints, ZIPs, machine receipts and transient logs are excluded.

## Reproduce the latest screen

Run from the repository root using Python with NumPy and CUDA PyTorch. The
recorded environment used PyTorch 2.5.1, NumPy 1.26.4 and an RTX 4060 Laptop GPU;
backend settings are recorded in the publication manifest. No dataset download
is needed. Every output directory must be new.

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
