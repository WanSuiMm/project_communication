# Execution repair; scientific protocol unchanged

The first detached dispatch ended before a completed measurement. A managed
execution session then computed75/396 units, but stopped when Windows returned
WinError5 while replacing `status.json`. The error handler successfully wrote
the final ERROR record. No model/nonfinite failure was reported.

The audit now uses a local atomic JSON writer that retries transient
PermissionError for at most10 seconds. Shared historical utilities and scientific
functions remain unchanged. A CPU test holds a destination file with a Windows
handle that denies replacement, releases it, and verifies retry/roundtrip.

Resume into a new run directory. Import only completed prefix units after
checking model/data/dependency bindings, scientific function AST equivalence,
unit metadata, NPZ hashes and byte-identical copies of all unit NPZ/JSON/CSV files.
Reuse the imported natural Boolean trace as the exact reference for remaining
conditions. The interrupted directory stays unchanged. No checkpoint selection,
new training, changed seeds, replacement models, altered intervention or new gate.

Use the managed execution session for dispatch:

    python -X utf8 -u -B new/semantic_write_audit/run.py --out runs/NEW_RUN --qualification analyses/NEW_CHECK.json --resume-from runs/INTERRUPTED_RUN

Report imported/newly computed counts separately. The current worker elapsed
field excludes work already computed by the interrupted worker. A restarted
process is not evidence of a scientific efficacy failure.
