# Verification and reproduction

From the repository root, run `python -X utf8 -B tools/export_trajectory_qualification.py --verify-only`.
This checks published SHA-256 bindings, source/reference hashes when available,
ratio consistency, saved gate decisions, stage counts, and stage 2 paired counts
and exact p-value with Python's standard library. It does not train or evaluate a model.

To rebuild the publication from the completed local run, pass the sanitized
validation artifact: `python -X utf8 -B tools/export_trajectory_qualification.py
--validation analyses/trajectory_publication_validation_20261004.json`.
The portable launcher resolves `python` from PATH; its runner still requires
the recorded Torch 2.5.1 environment. Exact stage 1 reproduction also needs the
archived initialization/reference checkpoints identified by the published
hashes. Checkpoint binaries and launch receipts are excluded, so a fresh clone
can verify the saved evidence but requires those local archives to rerun stage 1.
