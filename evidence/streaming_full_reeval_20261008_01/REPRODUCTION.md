# Saved-data reproduction

Read RESULTS.md and summary.json first; full_gates.csv lists every threshold.
The six unit folders retain unchanged summaries, completion bindings, all frontier
strata and losslessly compressed paired/original/flipped Boolean trajectories.
No trace times, cells, maps or negative gates were removed. Model weights and
private process receipts remain local; checkpoint and parameter hashes are retained.

The scientific run was zero-training and made no optimizer updates. Primary
u300 checkpoints were selected by the earlier Joint metric; u275 is diagnostic.
The addressed_selection cohort is reused selection data. No formation-rate estimate.

From repository root, Python with NumPy verifies the full public package:

```powershell
python -X utf8 -B tools/export_streaming_full_reeval.py --verify-only
```

For a trace NPZ, load with `numpy.load(path, allow_pickle=False)`. Each key
`correct`, `original_correct`, `flipped_correct` has `_shape` and `_packed`
arrays. Decode using `numpy.unpackbits(packed, bitorder="little", count=prod(shape))`
and reshape to `[257,32,size,size]`; every integer time0..256 is present.
Input map NPZ banks are published too, preserving their exact tensor hashes.

The verifier reuses the frozen Full predicate, recomputes trace-derived endpoint
coverage and transitions, ever-regression, and frontier effects from saved CSV.
It does not run inference. Repeating model evaluation requires the three private
checkpoint contents and Torch2.5.1/CUDA; see the frozen protocol and run.py.
