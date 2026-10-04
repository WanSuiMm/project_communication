# Incremental data delivery: serial trajectory qualification

- Review base: `62189a05815e0e5620a9fb9e288d9923e68b742f`.
- Evidence head: `77266893bb1c26eb29cf2d72ee2233297dcc6dfe`.
- This later handoff commit changes navigation only. The delivery supplies
  code and recorded data for independent analysis; prior architecture claims
  remain unchanged.

## Read first

1. [Suite table](evidence/trajectory_qualification_20261004/RESULTS.md),
   [summary](evidence/trajectory_qualification_20261004/summary.json), and
   [validation](evidence/trajectory_qualification_20261004/validation.json).
2. Stage summaries: [state cross](evidence/trajectory_qualification_20261004/stage1_state_cross/summary.json),
   [fresh recipe](evidence/trajectory_qualification_20261004/stage2_fresh_recipe/summary.json),
   [distance task](evidence/trajectory_qualification_20261004/stage3_distance/summary.json).
3. [Frozen suite protocol](new/trajectory_qualification/PROTOCOL.md),
   [distance protocol](new/trajectory_qualification/DISTANCE_PROTOCOL.md),
   [source map](GPT_CONTEXT.md), and
   [reproduction notes](evidence/trajectory_qualification_20261004/REPRODUCTION.md).

## Recorded outcomes and scope

COMPLETE: 48/48 arms (8 + 32 + 8), 6,610.797 seconds. Stage 1 reproduces
four native controls and selects the conditional `shadow_joint` recipe.
Stage 2 records 1/16 Full passes for both baseline and treatment, with no
discordant pairs: `NO_RELIABILITY_QUALIFICATION`. Stage 3 records 0/8
short-task qualifications: `SECOND_TASK_TRAINING_UNQUALIFIED`.

All failed arms, gate denominators and training curves are retained.
The 96 NPZ trace banks and 40 frontier CSVs are secondary raw evidence;
start with the summaries. The distance task is a distinct proxy, and its
unqualified training does not establish an architecture impossibility.

Reviewer questions: what does the conditional state cross identify; does
the fresh-pair result support reliability; and what remains untested when
the second-task training gate is unqualified?

No training or model inference was run for this upload. Checkpoints and
machine launch records remain local. [Publication hashes](TRAJECTORY_QUALIFICATION_PUBLICATION_MANIFEST.json)
bind the public files and record the sole portable launcher adaptation.
Verify from repository root:

    python -X utf8 -B tools/export_trajectory_qualification.py --verify-only
