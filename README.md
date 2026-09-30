# Reaction-Transport: communication qualification in 2D and 3D

This project tests whether a visual field model can retain rich local state,
transport narrower messages through a learned symmetric medium, and use local
nonlinear reactions to learn distant dependence. The first small qualification
completed Gate A in both dimensions. **The result is inconclusive: the attention
positive control converged in one seed and failed in the other; RT failed to fit
even the training-scale evaluation.** It provides no evidence of an RT advantage.
Gates B/C and the message-width sweep were not run.

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

1. [RESULTS.md](RESULTS.md): per-seed outcomes, fitting failure, latency and verdict.
2. [GPT_CONTEXT.md](GPT_CONTEXT.md): claim boundaries, code symbols and evidence routing.
3. [PROTOCOL.md](PROTOCOL.md): frozen training budget, task definitions and stop rules.
4. [ARCHITECTURE.md](ARCHITECTURE.md) and [THEORY.md](THEORY.md): tensor flow and four short checks.
5. Read the [small summary](evidence/qualification_v1_1/summary.json) and
   [configuration](evidence/qualification_v1_1/config.json) before opening the
   [full aggregate](evidence/qualification_v1_1/aggregate.json), which includes
   all 20 runs, histories, axis-level evaluations and timings.

## Run from the repository root

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
