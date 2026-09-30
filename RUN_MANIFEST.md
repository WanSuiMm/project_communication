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
