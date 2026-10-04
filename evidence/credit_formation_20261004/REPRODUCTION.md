# Reading and saved-data verification

Start with RESULTS.md, summary.json, metrics.csv and credit_formation.png.
Per-update continuation.json and phenotype.json retain cohorts, integer counts,
equal-map summaries, Full flags and reasons. The 65 NPZ files retain all banks,
per-batch/mean gradients, paired correctness traces and endpoint logits.

From repository root (NumPy required; no model/GPU execution):

    python -X utf8 -B tools/export_credit_formation.py --verify-only

The verifier checks source/publication hashes, recomputes gradient metrics,
mean vectors, continuation counts and the frozen synchrony verdict. Recorded
forward equality and no-update checks are retained in gradient metadata.
Full rerun requires the historical checkpoint archives identified by hashes in
config.json, Torch2.5.1, CUDA and matplotlib; see new/credit_formation/PROTOCOL.md.
Those checkpoints, source snapshots and machine launch receipts remain local.
Existing checkpoints and the original run have not been changed.

The behavior episode qualifies but overall/F/Q synchrony fails. All seven
current-bank native Full flags are false; update200 fails the strict T64 pooled
reach threshold (0.76648 versus0.80). This is distinct from the synchrony screen,
and does not erase the original checkpoint's result on its earlier banks.
The study is one selected trajectory on reused held-out banks; no population
or causal mechanism claim follows. No additional model run was made to publish.
