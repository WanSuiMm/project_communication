# Binary carrier evidence

Read RESULTS.md, summary.json, validation.json, config.json, decoder/fit.json,
decoder/probes.json and metrics.csv first. raw_summary.json.gz is secondary.

Saved-data verification from this repository root (NumPy, no model execution):

    python -X utf8 -B tools/export_binary_carrier.py --verify-only

The verifier checks file/source hashes, map-disjoint banks, lossless state
reconstruction, centroid arithmetic, held-out decoder/XOR scores, fixed cohorts,
24 arm traces/gates, 840 metric rows, FF-native equality, immediate W-only
output equality, paired-map bootstrap records and repeated template projection
bits/clocks. Native Full and parameter immutability are recorded runtime checks.
See RESULTS.md for the separate reconstructed T64 projection artifacts.

For a full rerun, obtain exact S1/F1 checkpoint hashes from config.json, install
NumPy and Torch2.5.1 with CUDA, and follow new/binary_carrier/PROTOCOL.md using
new output names. Run check.py first, then tools/launch_binary_carrier.ps1.
The public launcher discovers Python via PATH. Checkpoints remain local.

Test banks have128 map pairs per size; each pair is one independent map unit.
The single selected S1/F1 pair does not establish training reliability. All
three combined primary gates fail; oracle is diagnostic and excluded from
method claims. Failed class centroids do not rule out every possible bit code.
