# Verification and archive requirements

From repository root:

    python -X utf8 -B tools/export_continuation_interface.py --verify-only

This standard-library check verifies published/source hashes and saved counts
and gate consistency. It does not run models. All scientific JSON, CSV and
NPZ files are retained at their original relative paths; RESULTS.md additionally
lists recorded qualification flags. Start with summary.json, selection.json,
alignment/validation.json, phase0/comparisons.json and cross/decomposition.json.
The larger state, logits, trace and gradient arrays are secondary evidence.

The frozen procedure and runnable source are in new/continuation_interface/.
Full rerun requires the original final checkpoint archives identified by the
file/parameter hashes in config.json, historical Torch2.5.1 and a CUDA device.
Checkpoints, source snapshots, machine manifests, launch receipts and PIDs stay
local. The new banks, calibration states, adapters and consumer traces are
published, enabling downstream saved-data analysis without those checkpoints.
No model training or optimizer update was performed in this audit or release.
