# Incremental review: binary carrier causal compression

- Review base: `e9a3b23c641c3d946baf8ba440c2823e0a0f6bfc`.
- Evidence head: `2f1582f14524ea184e3b2b1cba03824a42a5862b`.
- This later commit adds only this handoff. Earlier frozen experiments and
  their claim boundaries are unchanged. Publication used saved data only.

## Read first

1. [Results](evidence/binary_carrier_20261005/RESULTS.md),
   [compact summary](evidence/binary_carrier_20261005/summary.json),
   [validation](evidence/binary_carrier_20261005/validation.json), and
   [configuration](evidence/binary_carrier_20261005/config.json).
2. [Decoder fit](evidence/binary_carrier_20261005/decoder/fit.json),
   [decoder probes](evidence/binary_carrier_20261005/decoder/probes.json),
   and [metrics](evidence/binary_carrier_20261005/metrics.csv).
3. [Frozen protocol](new/binary_carrier/PROTOCOL.md),
   [source map](GPT_CONTEXT.md), and
   [reproduction notes](evidence/binary_carrier_20261005/REPRODUCTION.md).

No unchanged earlier evidence needs rereading. `raw_summary.json.gz` and the
full activation/trajectory arrays are secondary.

## New data and qualification

COMPLETE, 24/24 cells in 473.469 seconds. Same selected S1/F1 update300
checkpoints and F1 consumer; T64→256, fresh128 test map pairs at each size.
One W-only linear decoder was fitted on independent calibration maps; class
centroids are fixed. There were zero recurrent-model updates.

SF and SS anchors pass, with adequate fixed keep/prog support at both sizes.
All three primary compression gates fail at both sizes; retain this verdict.

| Arm | Keep32 | Prog32 | Keep64 | Prog64 |
|---|---:|---:|---:|---:|
| SF | 1.0000 | .6793 | .9890 | .6404 |
| One-shot predicted bit field | .2159 | .4327 | .1527 | .4614 |
| Every8-step predicted projection | .0064 | .0368 | .0372 | .1280 |
| Source-only tokens | .0000 | .0000 | .0000 | .0000 |
| Oracle-only class centroids | .1193 | .9967 | .0896 | .6390 |

Oracle progress does not overcome its preservation failure. These results
reject the tested decoder/centroid/fixed-consumer construction; they do not
prove universal binary-code insufficiency or identify a unique missing variable.
Neither the prior W effect nor this result proves a quotient or BPTT guarantee.

## Complete data, compression and verification

Full numeric data are retained:117 NPZ files,840 metric rows and all per-map
JSON. The package is about643MiB. The duplicate full summary uses lossless
gzip; two oversized state captures use eight lossless32-map paired-cue chunks.
All numeric precision is unchanged. Small entry reports/JSON/CSV are directly
readable. [State index](evidence/binary_carrier_20261005/state_chunks.json)
specifies exact reconstruction and tensor hashes.

Runtime projection files contain T72..256; the initial T64 bit fields and norms
were reconstructed from saved states/templates under `derived/`, explicitly
separate from runtime measurements. No new inference or fitting was performed.
The public launcher replaces its local Python path with PATH discovery;
scientific sources match execution exactly, with both launcher hashes recorded.
Checkpoints, host/PID records, launch receipts and logs remain local.

From the repository root:

    python -X utf8 -B tools/export_binary_carrier.py --verify-only

Saved-data checks pass: cohorts/counts/gates, centroid/probe/XOR arithmetic,
FF-native suffixes, W-only immediate outputs, projection clocks/bits, metric
rows, paired-map bootstrap records and lossless state hashes.

Review questions: what does the low prog-cohort decoder accuracy distinguish
from overall probe accuracy; which hypotheses survive oracle progress coupled
with preservation failure; what further claim is warranted by these finite
interventions? Keep proposed mechanisms separate from the frozen measurements.
