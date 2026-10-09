# AU-NCA v0 evidence reproduction

Start with RESULTS.md, final_metrics.csv, per_episode_metrics.csv, aggregate.json, and validation.json. The public package includes the fixed target, three packed input banks, all nine arm results and training logs, 18 intermediate T64 diagnostics, all final saved RGBA predictions, continuation curves, gradient diagnostics, and 36 metadata-only checkpoint bindings. Checkpoint weights, optimizer/pool state, local receipts, and worker logs are omitted.

From the project repository root, run the independent public-package check:

```powershell
python -X utf8 -B tools/export_au_nca.py --verify-only
```

Verification uses public saved arrays and JSON, hashes, and current source files. It needs NumPy and does not read the private run or checkpoints, initialize CUDA, train, or run inference. The RGBA metrics are recomputed in float64 from saved predictions and target and checked against the frozen result JSON at absolute and relative tolerance 1e-12.

Training logs contain exactly updates 1..3000 for each arm (27,000 canonical update records). The separate training_before_resume_historical_partial.jsonl file preserves an interrupted prefix through update 1200; the committed checkpoint was at update 1175, so its 1176..1200 tail is historical partial evidence and is excluded from canonical counts. The aggregate 1,254.7 seconds covers the resumed execution only; it excludes the earlier interrupted attempt and the gap, so it is not total wall-clock time across attempts.

No prediction or training operation is needed to reproduce this verification. The three independent units are paired initialization/schedule blocks; the 32 firing-mask episodes are repeated measurements within each block.

## Fresh run commands

For a new experiment, use Windows PowerShell in the registered CUDA environment and choose fresh output names. The qualification command includes a small CUDA smoke; the training command launches a new run and can take substantially longer.

```powershell
python -X utf8 -B new/au_nca/run.py --check --out analyses/au_nca_qualification_20261010_01
python -X utf8 -B new/au_nca/run.py --out runs/au_nca_20261010_01 --qualification analyses/au_nca_qualification_20261010_01/qualification.json
```
