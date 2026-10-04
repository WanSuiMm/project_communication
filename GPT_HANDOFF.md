# Incremental data delivery: seed4 bootstrap-path screen

- Review base: `bd1b80ba5946e4a6d79852cd1f83669f076b339e`.
- Evidence head: `0ee1e81d0581fc76195ac62bc29f1afe99c49e9f`.
- This later handoff commit changes navigation only. This delivery supplies
  data and code for independent analysis, without a new mechanism discussion.

## Read first

1. [All23-arm table](evidence/bootstrap_path_20261004/RESULTS.md),
   [aggregate](evidence/bootstrap_path_20261004/summary.json),
   [six first-step cases](evidence/bootstrap_path_20261004/first_step_summary.json).
2. [Run configuration](evidence/bootstrap_path_20261004/config.json),
   [frozen protocol](new/bootstrap_path/PROTOCOL.md), and [source map](GPT_CONTEXT.md).
3. [Per-arm summaries](evidence/bootstrap_path_20261004/arms),
   [training curves](evidence/bootstrap_path_20261004/training),
   [frontier counts](evidence/bootstrap_path_20261004/frontier).
   [46 Boolean NPZ banks](evidence/bootstrap_path_20261004/traces) and
   [first-step arrays](evidence/bootstrap_path_20261004/first_step_arrays)
   are secondary raw data; do not start with the binary files.

## Data scope and verification

COMPLETE23/23 arms x300 updates in3068.407 seconds. Original init4 cell,
optimizer and K8 recipe; H20002, selected S20012/S20022; prefix lengths1/3/8,
pre-180 replacements and exact-multiset row swaps. Evaluation is paired at
sizes32/64 on the reused32-map banks50032/50064. All failed arms and frozen
gate denominators remain in the data. Existing architecture claims are unchanged.

[Validation](evidence/bootstrap_path_20261004/validation.json) checks all46
saved trace banks, source snapshots, control replays and count arithmetic.
[Publication hashes](BOOTSTRAP_PATH_PUBLICATION_MANIFEST.json) identify
byte-identical summaries/arrays and allowlisted training metadata.
No model inference or new training was run for this upload.

From repository root: `python -X utf8 -B tools/export_bootstrap_path.py --verify-only`.
[Reproduction notes](evidence/bootstrap_path_20261004/REPRODUCTION.md) describe
the excluded reference archives needed by the frozen training runner.
Checkpoints, machine manifests, PIDs and logs were not uploaded.
