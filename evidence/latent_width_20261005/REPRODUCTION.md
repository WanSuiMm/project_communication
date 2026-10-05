# Native-width saved evidence

Start with RESULTS.md, summary.json, validation.json and config.json, then
perarm.json and metrics.csv. Per-arm evaluation summaries contain every Full
component and all per-map metrics. NPZ and gzip data are secondary.

Saved-data verification from the standalone repository root:

    python -X utf8 -B tools/export_latent_width.py --verify-only

Requires NumPy and PyTorch; the verifier executes no model or CUDA operation.
It checks hashes, bank/schedule bindings, all 300-update curves, Boolean
pairing, coverage/retention/regression counts, saved Full predicates, all576
metric rows and paired statistics. It does not rerun logits, balanced accuracy
or matched-frontier strata. Read validation.json for the exact check scope.

For a fresh full run, use the source and commands in new/latent_width/PROTOCOL.md
with PyTorch2.5.1, NumPy1.26.4 and a compatible CUDA environment. Activate that
environment so the PowerShell launcher discovers the correct Python via PATH.
Run the combined sanity check and choose fresh output names; no historical
checkpoint is required. CUDA Graph continuation of an existing eager run uses
new/latent_runtime/ACCELERATION.md and requires its local reference checkpoints.
Runtime qualification is limited to the recorded six first-block replays.

Only the final update300 checkpoint is scored. The independent statistical
unit is an initialization/schedule block (16), not maps/cells/rollout times.
No efficacy selection or new sweeps were performed during publication.
