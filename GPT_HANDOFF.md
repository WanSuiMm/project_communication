# Incremental review: continuous execution-state coverage

- Review base: `df5107b8d529a9753f09b1eec750bba36be67295`.
- Evidence head: `8ac3d7558d3d4bafb5f495e27e60db05d22f3d11`.
- This later commit changes only this handoff. Earlier protocols and evidence
  remain unchanged; publication performs no model fitting, training or inference.

## Read first

1. [Results](evidence/continuous_coverage_20261006/RESULTS.md),
   [final summary](evidence/continuous_coverage_20261006/summary.json),
   [validation](evidence/continuous_coverage_20261006/validation.json).
2. [All checkpoint metrics](evidence/continuous_coverage_20261006/metrics.csv),
   [dense records](evidence/continuous_coverage_20261006/dense.json),
   [reach/retention trajectories](evidence/continuous_coverage_20261006/reach_retention_trajectory.png).
3. [Protocol](new/continuous_coverage/PROTOCOL.md),
   [code map](GPT_CONTEXT.md),
   [runtime qualification](evidence/continuous_coverage_20261006/runtime_qualification.json),
   [reproduction](evidence/continuous_coverage_20261006/REPRODUCTION.md).

## New evidence

COMPLETE16/16 trajectories in eight fresh paired training blocks,208 saved
checkpoint records. Original C24/Z8 StreamingCell(5033 parameters), unchanged.
Reset64x4 repeats the SAME batch8 with four cold starts; continuous256 carries
one state through256 steps. Each super-update uses32 K8 backward windows,
32 balanced losses divided by32, one clip and one AdamW step after step256.
Parameters are fixed throughout the super-update; no state crosses updates.

|Fixed u300|Joint readiness|Unchanged old Full|
|---|---:|---:|
|reset64x4|0/8|0/8|
|continuous256|0/8|0/8|

Frozen verdict: `NO_CONTINUOUS_COVERAGE_RELIABILITY_QUALIFICATION`.
Joint wins0, losses0, net0/8, exact discordant-pair two-sided p1.
Continuous-minus-reset pooled strict T64 reach is-0.0709758255 (3/8 positive);
T64-correct retention to T256 is+0.1171953934 (4/8 positive). All eight pairs
meet the reference support floor. Mixed continuous metrics do not rescue the
primary. This is a negative qualification of the tested recipe, not a universal
failure of continuous training or cellular computation.

Joint readiness uses size32 strict16<d<32 pooled AND map-mean reach>=.80,
retention>=.95,>=16 reference maps and>=100 T64-correct changed cells. It does
not require additional coverage gain and is reported separately from old Full.
Checkpoint grid u0,25,...,300 is frozen; onlyu300 is the formal endpoint.
Intermediate Full is not evaluated. Dense points describe finite formation,
collapse or recovery; no peak selection or phase-transition theorem follows.

Reset also gives the encoder four credit-connected first windows versus one
for continuous. This induced difference prevents isolating numerical state-age
visitation from encoder-credit frequency. The experiment differs from old
half-batch random-age warmstart: it supervises every K8 window of a complete
continuous trajectory, with no loss-free prefix or cross-update state staleness.

## Verification and data scope

All416 packed Boolean trace banks, three data banks, schedules,16 full training
curves,208 full per-checkpoint summaries and16 final frontier tables are retained
losslessly. Public package245.79MiB; arrays and gzip files are secondary.
Saved-data validation recomputes R/S/continuous survival for every checkpoint,
all bank tensor hashes, packed roundtrips and the paired final aggregate.
Final frontier and old Full use saved evaluator outputs and are not regenerated.
All208 local checkpoint file hashes and52 source snapshot/current bindings were
checked during export; checkpoint contents and machine/session records stay local.

Three actual-shape updates per arm had bitwise-equal eager/graph arithmetic,
parameters and Adam states; this is not a300-update equivalence proof.
Formal worker3439.86seconds excludes preflight and publication. Initial launch
interruption was an execution-lifetime issue; the completed worker restarted
from the identical initialization and schedules, not a selected checkpoint.

    python -X utf8 -B tools/export_continuous_coverage.py --verify-only

Review questions: do the fixed dense records show transient joint readiness
that training later destroys; how much can the negative final qualification
say about trajectory visitation given the encoder-credit difference?
