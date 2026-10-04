# Reproduction boundaries

From the repository root, `python -X utf8 -B tools/export_bootstrap_path.py --verify-only`
checks public hashes, ratios and gate decisions using the standard library.
The original CPU sanity check and CUDA preflight/formal commands are in the
[frozen protocol](../../new/bootstrap_path/PROTOCOL.md). They require Python,
NumPy and the historical PyTorch2.5.1 CUDA environment; no new dependency.

Exact retraining also expects the excluded reference archives explicitly named
in `new/bootstrap_path/run.py`: initial checkpoints2/3/4/5, historical schedule
and prior control records/manifests. A fresh clone can verify this publication
but cannot run the frozen runner without those archives. Training/evaluation
maps and schedule interventions are procedurally generated from recorded seeds.
Initial/model/data/schedule hashes and all negative endpoints are published.
No excluded archive is silently regenerated or overwritten. The validation
recomputed endpoint, retention, regression, BA and frontier arithmetic from
all46 published Boolean-trace banks; it did not repeat model execution.
