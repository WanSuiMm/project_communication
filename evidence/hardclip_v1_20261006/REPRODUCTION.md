# HardClip v1 package verification

Start with [RESULTS.md](RESULTS.md), [final_metrics.csv](final_metrics.csv), and [dose_summary.json](dose_summary.json). The formal endpoint is joint readiness at u300 across eight paired blocks. `metrics.csv` retains both evaluated sizes at all 208 checkpoints; `training_dose.csv` retains the four lane summaries for each of 300 updates in each trajectory.

From the repository root, verify this package with saved data only:

```powershell
python -X utf8 -B tools/export_hardclip_v1.py --verify-only
```

Verification does not need the local run directory, qualification analysis, model checkpoints, PyTorch, or a GPU. It checks package and source hashes, schedule and data-bank bindings, all 208 record keys and evaluator summaries, all packed trace file shapes, the 32 u300 trace banks and paired Boolean AND, the reporter aggregate, training cadence and dose accounting, all saved evaluation-dose rows, and calibration sample/cap bindings. It does not load checkpoint files or regenerate the Old Full evaluator gate.

The package includes the frozen HardClip and calibration summaries, all lossless evaluation summaries and dose files, all 416 trace NPZ files, all 16 training curves, the data banks, and the block schedules. `HARDCLIP_V1_PUBLICATION_MANIFEST.json` binds raw-file hashes, published hashes, current source hashes, calibration inputs, and checkpoint file/parameter hashes. Checkpoint and optimizer contents, private run manifests/status/PIDs, and launch receipts are excluded. One launcher source was sanitized to remove its local Python path; the exact permitted source transform and both hashes are recorded in the manifest.

HardClip had zero trigger events across all lanes in five of the eight paired blocks. Neural-arm dose is counterfactual. This weak observed dose limits interpretation of the negative efficacy result to the fixed tested intervention. No broad large-write-mechanism conclusion follows.

The frozen protocol and implementation are in [PROTOCOL.md](../../new/hardclip_v1/PROTOCOL.md) and `new/hardclip_v1/`. Training requires the qualified CUDA environment; package verification is CPU-only and uses Python with NumPy.
