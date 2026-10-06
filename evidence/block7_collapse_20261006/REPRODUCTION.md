# Reproduction and saved-data layout

Read RESULTS.md, summary.json, validation.json, then matrix_summary.csv and
single_step_summary.csv. Undefined CSV ratios are blank; raw JSON uses null.
This publication performs CPU saved-data checks only, with no model inference.

From repository root, with Python and NumPy:

    python -X utf8 -B tools/export_block7_collapse.py --verify-only

GPU audit code: new/collapse_audit/run.py; protocol and CPU metric checks are
beside it. Re-running inference requires the four original local checkpoint
files from continuous_coverage_20261006_02. Their hashes are in provenance.json;
checkpoint contents and all machine/session records remain local. The prior
published evaluation banks and native traces are reused by reference, with
their exact paths and hashes in provenance.json. No external workspace is used.

matrix.json.gz and single_step.json.gz are lossless copies of complete raw
records, including all per-map numerators/denominators. The matrix CSV contains
192 rows:32 cells x3 readout views x2 cohorts; single-step CSV has24 rows.
This audit evaluates one selected training trajectory, not independent rows.

matrix_traces/sizeSIZE_pPRODUCER_cCONSUMER.npz stores Boolean trajectories for
producer, consumer and fixed275 readout views. Each has correct,
original_correct, flipped_correct fields. Fields are stored as NAME_shape
(int32) and NAME_packed(uint8); unpack with numpy.unpackbits, bitorder=little,
count=product(shape), then reshape. Shape is [193,32,SIZE,SIZE], times64..256.
The paired correct field is the AND of the original and flipped fields.

single_step_traces/sizeSIZE_tTIME_cCONSUMER.npz uses before/after prefixes and
the same three correctness fields, shape[32,SIZE,SIZE]. TIME is64,128 or192.
Both observations use the fixed u275 readout.

states/ contains eight T64 producer packages and two u275 packages including
T64/T128/T192. Keys are tTIME_original_C/Z and tTIME_flipped_C/Z. Arrays are
FP32[32,24,SIZE,SIZE] for C and FP32[32,8,SIZE,SIZE] for Z. All54 NPZs are
byte-exact compressed copies; no quantization or scientific-data deletion.

Preservation uses the producer-native T64-correct changed cohort at every
time64..256. Sustained progress uses its T64-wrong cohort at all times241..256.
Three readout views only observe states, never affect the recurrent rule.
Native prefix/diagonal replay is exact at all eight checkpoint/size pairs.
Prior formal u300 qualification stays negative; Hybrid has not been run.
