# Incremental review: semantic-write source causal audit

- Review base: `606265480e56cc1b88f81f451148c8274601180c`.
- Evidence head: `27eac03eec20cd7efc17a4a7c01adc0fede69d3d`.
- This later commit changes only this handoff. Earlier frozen protocols/evidence
  remain unchanged. Publication performs no new inference, training or updates.

## Read first

1. [Results](evidence/semantic_write_20261006/RESULTS.md),
   [summary including all block deltas](evidence/semantic_write_20261006/summary.json),
   [validation](evidence/semantic_write_20261006/validation.json).
2. [396 measurement records](evidence/semantic_write_20261006/perunit.json),
   [configuration](evidence/semantic_write_20261006/config.json),
   [natural replay](evidence/semantic_write_20261006/replay.json).
3. [Protocol](new/semantic_write_audit/PROTOCOL.md),
   [code map](GPT_CONTEXT.md),
   [reproduction](evidence/semantic_write_20261006/REPRODUCTION.md).

## New evidence and boundaries

COMPLETE396/396,33 frozen checkpoints x2 sizes x6 conditions; no training or
optimizer updates. First8 factorial blocks, all four arms, plus selected C24
Streaming seed4, on reused32-map evaluation cohorts. The independent primary
unit is the paired training block(n8), not pixels, maps or rollout steps.

Primary native size32/T256 strict16<d<32 deltas:

| Block | Source minus natural | Source minus orthogonal |
|---:|---:|---:|
|0|0|0|
|1|0|0|
|2|0.002893758|0.001653576|
|3|0|0|
|4|0|0|
|5|0.001653576|0.001653576|
|6|0|0|
|7|0|0|
|Mean|0.000568417|0.000413394|

Only2/8 source-natural differences are positive. Frozen exploratory thresholds
require both means>=0.05 and positives>=6/8. Verdict:
`NO_PRIMARY_SOURCE_DRIVER_SIGNAL`. Native source T256 coverage improves0.5820
to1.0000 but distant coverage changes only0.3217 to0.3223. This rejects the tested
source-driver intervention; it is not a general semantic-write impossibility.

Oracle solved-cell protection uses distant labels; its gains cannot qualify
communication or rescue the primary. Protected-source preservation is a
manipulation check. Removing all Q-nullspace components changes future state,
so it is an ablation. Selected seed4 qualifies on the common cohort, but has a
different width and is not a fresh matched replication.

All64 factorial natural traces replay exactly. Two seed4 records correctly say
NEW_COMMON_COHORT_REFERENCE. All396 packed Boolean traces and sampled FP32 margin
arrays,66 natural signed-write/source arrays, intervention arrays and152064
per-map endpoint rows are retained. NPZ is byte-exact compressed data; per-map
CSV uses lossless gzip. Package317.89MiB; raw arrays are secondary.

The completed worker imported75 validated units and computed321 new units.
All225 imported file hashes match; the science-function AST, model/data/source
bindings match.1318.938seconds covers this continuation worker only, excluding
earlier work/preflight/publication. A local transient Windows JSON replacement
failure interrupted the prior worker; machine records remain local.

Saved-data validation recomputes all endpoint/per-map coverage, retention,
regression and aggregate values, checks bank/file hashes and natural prefixes.
It does not rerun logits or signed-write diagnostics. CPU projection/wiring and
Windows-lock checks, plus natural CUDA replay, were recorded before dispatch.
Checkpoint files stay local;33 file hashes were checked during export.

    python -X utf8 -B tools/export_semantic_write.py --verify-only

Review questions: what does source preservation fail to change downstream;
which secondary observations remain interpretable without distant-label
injection; which narrower hypothesis survives this negative causal screen?
