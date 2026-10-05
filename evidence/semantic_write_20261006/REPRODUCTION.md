# Semantic-write saved evidence

Read RESULTS.md, summary.json, validation.json, then config.json and perunit.json.
All396 unit NPZ/JSON files and compressed per-map CSVs are secondary records.
Each NPZ stores little-endian packbits of the flattened paired Boolean trace,
its shape, FP32 sampled changed-region margins, and intervention arrays.
Natural units additionally store per-step signed-write diagnostics and source
margin/proposal arrays. Unit JSON files define layouts and column names.

CPU saved-data verification from this repository root:

    python -X utf8 -B tools/export_semantic_write.py --verify-only

Dependencies: NumPy and PyTorch (see requirements.txt). No CUDA, local run folders
or checkpoint files are needed for this verification. It checks hashes, banks,
all396 endpoint and per-map counts, retention/regression, natural prefixes and
the frozen aggregate. Recorded natural replay checks were performed at runtime;
publication does not repeat model logits or signed-write calculations.

Scientific code: new/semantic_write_audit/run.py; projection/wiring checks:
new/semantic_write_audit/checks.py; transient Windows-lock test: check_io.py.
Follow PROTOCOL.md and EXECUTION_FIX.md in that directory for a new experiment.
Running new model inference requires the33 local checkpoints bound in config.json
and the historical Torch2.5.1 CUDA backend. Checkpoint files stay local, so a
clone can verify saved measurements but cannot rerun inference unassisted.
The successful dispatch used a managed execution session and the direct command:

    python -X utf8 -u -B new/semantic_write_audit/run.py --out runs/NEW_RUN --qualification analyses/NEW_CHECK.json

The original detached PowerShell launcher is retained as runtime-bound source;
it did not survive the initial dispatch in this environment. Source snapshots
and machine receipts stay local. Neither publication nor verification trains.
