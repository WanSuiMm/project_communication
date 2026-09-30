# Reaction-Transport: communication qualification in 2D and 3D

This project tests whether a visual field model can retain local state, exchange
narrow messages through a learned symmetric medium, and learn distant dependence.
**The latest A0 result is a negative qualification in 2D:** attention solves all
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

## Start here

1. [RESULTS.md](RESULTS.md): current A0 outcomes first, followed by historical v1.
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

Current A0 (requires CUDA, no silent CPU substitution):

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

[A0_PUBLICATION_MANIFEST.json](A0_PUBLICATION_MANIFEST.json) binds the latest code,
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
