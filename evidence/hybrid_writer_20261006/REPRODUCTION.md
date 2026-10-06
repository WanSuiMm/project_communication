# Hybrid Writer result verification and reproduction

Start with [RESULTS.md](RESULTS.md), [summary.json](summary.json), and the32-row
[final_metrics.csv](final_metrics.csv). The independent unit is the paired
training block, n=8; u300 joint readiness is the formal endpoint. Ever-ready
on u0,u25,...,u300 and writer telemetry are secondary diagnostics.

From the repository root, run the CPU-only saved-data verification:

```powershell
python -X utf8 -B tools/export_hybrid_writer.py --verify-only
```

This checks hashes, three data banks, schedules, all832 packed Boolean trace
banks, paired original/flipped AND, and R/S/survival for all416 checkpoints.
The saved reporter aggregate, primary paired counts/Holm correction and
checkpoint grid are checked. No Torch import, model inference or training is
performed. Old Full and frontier are preserved evaluator results, not rebuilt.

The complete scientific output is included. Large raw summaries, writer
telemetry, training curves, schedules and frontier CSVs use lossless gzip.
NPZ banks are unchanged; bitpack order is little-endian. Each trace contains
correct, original_correct and flipped_correct shape[257,32,size,size], with
times0..256. The dense records refer to original local artifact names; use
the root publication manifest's raw_artifact_bindings to resolve gzip paths.
Every checkpoint file SHA was checked locally and its parameter hash retained;
checkpoint/optimizer contents, launch receipts, private manifests and logs are
excluded. Prelaunch validation is in the separate historical evidence package.

The frozen protocol and training commands are in
[PROTOCOL.md](../../new/hybrid_writer/PROTOCOL.md). Training needs the qualified
PyTorch2.5.1 CUDA environment and a GPU. This export is self-contained for
saved-data checks with Python and NumPy and does not need local runs/checkpoints.
The plot and dense table are formation diagnostics; no peak selection replaces
the frozen u300 result. No general semantic-closure or recurrent-stability claim
is made by this recipe.
