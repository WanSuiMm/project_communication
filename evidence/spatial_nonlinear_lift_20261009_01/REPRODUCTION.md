# Reproduce and review this result

Start with `RESULTS.md`, `aggregate.json`, `final_metrics.csv`, and
`systems.csv`. The aggregate is the decision summary; the nine per-arm
`result.json` files contain the same final metrics and system records. The
complete saved curves and predictions are retained beside them. Public result
copies omit learned a/b/gamma parameter values; checkpoint binding JSON files
publish only hashes, parameter shapes, and update metadata.

From the repository root, verify the exported arrays and artifact hashes on
CPU without training or model inference:

```powershell
python -X utf8 -B tools/export_spatial_nonlinear_lift.py --verify-only
```

The verifier checks all 2,700 finite loss records, all 36 final metric rows
including per-map MSE, all 27 heldout stage evaluations, all 36 checkpoint
bindings, the three regenerated CPU minibatch schedules, and current source
hashes. Predictions and targets are loaded from the saved NPZ files as
float64 for metric recomputation.

The original run reached 300 updates for each of three arms in each of three
matched initialization/schedule blocks. It has 42 versus 2 state scalars per
cell and nine trainable parameters per arm. The result is a three-block
developmental screen on a matched finite-degree teacher task. Long-horizon
rows are separately generated finite episodes; they do not establish
autonomous continuation or stability. Systems timings cover training plus
intermediate evaluation, and reported peak memory includes resident banks;
neither is a standalone kernel benchmark.

The public source and protocol are sufficient to rerun the qualification and
experiment without an older reference run. Any new training run must use a
fresh output directory and the declared CUDA environment; it does not belong
in this saved-array verification step.
