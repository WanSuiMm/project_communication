# Incremental review: completed Hybrid Writer v0 experiment

- Review base: `283ebe5ae584c53e26f588d8230808349a66472d`.
- Prior implementation evidence: `7373aa4d1f3c64bd1e84af108b9cc2ac617eab46`.
- Result evidence head: `ecdc145e55bd5a21a1f719c84791f82357b714f4`.
- Execution: `COMPLETE`,32/32 trajectories and416/416 checkpoint evaluations.
- Frozen verdict: `NO_PRIMARY_RELIABILITY_QUALIFICATION`.
- This later handoff commit changes only review metadata; the evidence head stays stable.

## Read first

1. [Frozen final report](evidence/hybrid_writer_20261006/RESULTS.md),
   [aggregate](evidence/hybrid_writer_20261006/summary.json), and
   [32-row formal metrics](evidence/hybrid_writer_20261006/final_metrics.csv).
2. [Formation plot](evidence/hybrid_writer_20261006/reach_retention_trajectory.png),
   [checkpoint metrics](evidence/hybrid_writer_20261006/metrics.csv), and
   [dense records](evidence/hybrid_writer_20261006/dense.json).
3. [Saved-data validation](evidence/hybrid_writer_20261006/validation.json),
   [reproduction and artifact map](evidence/hybrid_writer_20261006/REPRODUCTION.md),
   [publication manifest](HYBRID_WRITER_PUBLICATION_MANIFEST.json).
4. Refer back to the unchanged [protocol](new/hybrid_writer/PROTOCOL.md),
   [writer](new/hybrid_writer/cells.py), [runner](new/hybrid_writer/run.py), and
   [reporter](new/hybrid_writer/reporting.py) only as needed.

Large packed arrays and full gzip JSON/CSV files are secondary evidence.
Earlier audit state packages need not be reread for this result update.

## New decision-relevant evidence

At the fixed u300 endpoint, joint readiness is neural1/8, budget0/8,
hybrid0/8 and affine_hybrid0/8. Candidate-only/reference-only outcomes are
0/1 for budget minus neural,0/0 for hybrid minus budget, and0/1 for hybrid
minus neural. All three exact and Holm-adjusted p-values are1; none meets
the predeclared net-gain and multiplicity requirements. Affine minus Hybrid
is a descriptive capacity comparison, not a fourth primary test.

The predeclared u0,u25,...,u300 grid yields the same ever-ready counts.
Neural has one trajectory first-ready before300 and no later readiness loss;
the other arms have zero first-ready trajectories, so readiness-loss rates
are undefined. Dense results never replace the final endpoint with a peak.

Execution took8864.187seconds, about2h28m. All scientific records are retained:
832 packed paired trace banks,416 full per-checkpoint summaries,416 writer
telemetry files,32 training curves,32 final frontier CSVs, three numeric banks,
all schedules and aggregates. Large text records use lossless gzip; NPZ files
are unchanged.416 local checkpoint file hashes and58 source snapshot/current
hashes were verified. Model/optimizer contents and private machine records
remain local. The package is about411MiB.

Publication verification recomputed R/S/continuous survival at all416 checkpoints
from saved Boolean traces, checked original/flipped AND and bitpack roundtrips,
and checked the final paired aggregate and checkpoint grid. No model inference,
training, optimizer update or new causal intervention was performed. Old Full
and frontier remain saved evaluator outputs, not regenerated measurements.

## Changed and unchanged claims

The efficacy-pending label is replaced by a completed formal negative result:
this fixed lane-write-budget/table recipe did not qualify a reliability gain
under its frozen K8 protocol. No training code, evaluation bank, gate, schedule,
checkpoint selection policy or historical evidence was changed for publication.

The result does not isolate why formation failed: the budget contrast bundles
radial bounds with regulation. It supplies no general rejection of Hybrid/NCA,
no proof of semantic closure or recurrent stability, and no3D or general BPTT
claim. Earlier block7 continuation evidence and previous formal negatives stay
unchanged.

## Reviewer questions

1. Do the final block table and formation curves suggest no readiness formation,
   or formation below the joint thresholds? Keep this descriptive and distinguish
   reach, retention and their support denominators.
2. Do the secondary lane-gate/write telemetry records reveal a consistent pattern
   worth testing, without treating a correlation as a causal diagnosis?
3. Does any proposed next intervention address a specific observed deficit rather
   than reinterpret this negative qualification as a positive result?

CPU-only verification from repository root:

    python -X utf8 -B tools/export_hybrid_writer.py --verify-only
