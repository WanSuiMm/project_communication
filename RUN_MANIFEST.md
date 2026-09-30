# Run provenance

The `runs/` locations below identify local evidence and are not uploaded. Public
copies of the canonical aggregate/configuration are in `evidence/qualification_v1_1/`;
the original smoke is in `evidence/validation/smoke.json`. Training source hashes
and sanitized runtime provenance are in `PUBLICATION_MANIFEST.json`.

- `runs/smoke_20260930_2d3d_v1`: passed numerical forward/backward smoke for
  both dimensions and all transport/full-state/oracle variants.
- `runs/qual_20260930_2d3d_v1`: interrupted and excluded. A scheduling review
  found axis=step mod dim confounded distance and orientation in 2D. Partial
  outputs are preserved; do not merge them into qualification evidence.
- `runs/qual_20260930_2d3d_v1_1`: corrected axis-distance crossing. Hyperparameters,
  gates, thresholds and seed list are unchanged. Canonical first qualification.
- `runs/a0_checks_20260930_v2_1`: A0 numerical/gradient/initialization and 2D/3D
  forward/backward checks passed on the local GPU. Oracle diagnostics cover
  100 dimension/axis/distance/scale conditions. At tau4096, raw sign decoding
  is correct in every tested case, including d128; at tau1, d128 underflows
  to zero on every tested axis. This is an oracle-path result, not learned
  model qualification. Source hashes bind the report to the A0 code.
- `runs/a0_20260930_2d3d_v2_1`: COMPLETED_FROZEN_SCHEDULE, exported on
  2026-09-30 under `A0_PROTOCOL.md`. All 24 conditional trials completed;
  16 potential 3D RT trials were skipped after failed positive-control
  calibration. All four 2D RT arms are NOT_QUALIFIED_FIT. Public snapshot is
  `evidence/a0_v2_1/`, with `A0_PUBLICATION_MANIFEST.json` for hashes and export
  time. Local machine receipts and checkpoints remain excluded. No recurring
  monitor or further experiment is scheduled by publication.
