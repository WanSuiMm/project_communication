# Incremental data delivery: W medium qualification

- Review base: `19c3a6d2142d1ac5edcb8132123da2ae63bd6cfd`.
- Evidence head: `84d5d327213c816c9b076b698a3bd716e4878bdd`.
- This later commit adds the handoff only. Earlier frozen experiments,
  including the 32-map support failure, remain unchanged. Publication used
  saved data only; no extra inference, training or optimizer updates.

## Read first

1. [Results](evidence/w_medium_20261004/RESULTS.md),
   [compact summary](evidence/w_medium_20261004/summary.json),
   [validation](evidence/w_medium_20261004/validation.json), and
   [configuration](evidence/w_medium_20261004/config.json).
2. [Metrics](evidence/w_medium_20261004/metrics.csv) and
   [figure](evidence/w_medium_20261004/w_medium.png).
3. [Frozen protocol](new/w_medium/PROTOCOL.md), [code map](GPT_CONTEXT.md),
   and [reproduction](evidence/w_medium_20261004/REPRODUCTION.md).

The compact summary is about148KiB. Do not open the37MB raw summary or large
arrays first; they are secondary evidence. No prior unchanged file needs rereading.

## Recorded outcomes and changed qualification

COMPLETE36/36 cells in570.406 seconds, zero training. Same selected S1/F1
update300 checkpoints and F1 consumer, T64 handoff, fixed128 fresh maps at
each size32/64. W-only transplants have exact FF immediate logits. Both SF and
SS pass the unchanged rescue thresholds with adequate support at both sizes.
Native Full is S1=True/F1=False; the separate historical localization
predicate also qualifies. Its native Full conditions were descriptive for
the new controls-stage eligibility, as frozen before outcomes.

Fixed keep/prog map support is118/33 at32 and39/123 at64. SF continuous keep
is1.0000/.9899 and sustained/delayed progress is.5726/.5925; FF progress is0.
Norm-matched SF passes, reverse norm-matched FF fails. Raw payload permutation
fails; raw lane permutation passes. All six gauge suffixes/logits exactly
reproduce SF. Spatial scrambling progress is.9966/.9147, but size64 keep
.9029 fails the.95 gate. Cue swaps fail. Donor times48/80/96 pass,32 fails
keep. Preserve every arm's support, proxy, preservation and progress fields.

These are selected-pair conditional sensitivities. They do not identify an
exclusive mechanism, prove task semantics or pure phase, assign W/Z exclusive
roles, establish a carrier loss, or show training-seed reliability. Prior
conditional Z effects and the negative credit-synchrony result remain unchanged.

## Data and verification

All scientific data retained:182 NPZ files,1260 metric rows, full arm/native
JSON, original raw summaries and PNG/PDF figures. The six state captures are
losslessly split into48 map/time chunks; [state index](evidence/w_medium_20261004/state_chunks.json)
records reconstruction and exact tensor hashes. No activation values changed.
Checkpoints, machine manifests, PIDs, launch receipts and logs stay local.

Verification recomputes fixed cohorts/counts/gates, paired-map differences and
bootstrap intervals, immediate W-only output equality, FF/native suffixes and
all six gauge suffixes/endpoints, and reconstructs original state hashes.
Native Full, parameter immutability and instrumented first steps are recorded
checks; this saved-data verifier does not rerun models.

Reviewer questions: which controls rule out the defined scalar-norm account;
what is inferable from payload sensitivity with exact gauge controls; how do
high spatial-scramble progress and the size64 preservation failure constrain
the next hypothesis? No new architecture is proposed here.

[Publication manifest](W_MEDIUM_PUBLICATION_MANIFEST.json). From repository root:

    python -X utf8 -B tools/export_w_medium.py --verify-only
