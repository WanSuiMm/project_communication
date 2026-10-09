# ReLU input-lift evidence reproduction

Start with RESULTS.md, final_metrics.csv, per_episode.csv, aggregate.json, validation.json, and reference_bindings.json. The package includes the fixed target, three packed input banks, all three new arm results, final saved predictions, intermediate diagnostics at updates 1000 and 2000, all 9000 training records, continuation curves, gradient diagnostics, and 12 metadata-only checkpoint bindings. It excludes checkpoint tensors, optimizer/pool state, worker logs, local receipts, and machine identifiers.

From the project repository root, verify the package offline:

```powershell
python -X utf8 -B tools/export_relu_input_lift.py --verify-only
```

Verification recomputes metrics from the saved predictions and target, checks the saved training and continuation records, recomputes the frozen decision, and verifies current source hashes plus the SHA-pinned public AU-NCA controls. It does not open the private run or checkpoint files, initialize CUDA, train, or run model inference. Historical controls are linked from their existing public package; their prediction arrays and input banks are not duplicated here.

## Fresh training

Checkpoint weights are not shipped. A fresh ReLU input-lift run needs a local frozen AU-NCA reference: first create that reference with the qualification and training steps in [the AU-NCA protocol](../../new/au_nca/PROTOCOL.md), then pass its run directory with the ReLU runner's --reference option while qualifying and training under [the ReLU input-lift protocol](../../new/relu_input_lift/PROTOCOL.md). The published scores use the historical controls pinned in reference_bindings.json; a newly generated reference is a new experiment.
