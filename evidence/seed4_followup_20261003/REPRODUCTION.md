# Reproduction and available evidence

From the repository root, install requirements.txt. The frozen runtime was
PyTorch2.5.1 with FP32 CUDA, threads2, cudnn benchmark/deterministic False,
cudnn TF32 True, and matmul TF32 False. The protocol specifies all seeds,
sample sizes, optimizer, state dimensions, clocks, gates and caps.

## Public CPU verification

```
python tools/export_seed4_followup.py --verify-only
python new/seed4_followup/check.py --out analyses/NEW_FOLLOWUP_CPU.json
```

The first command verifies source/evidence/tool hashes, gate arithmetic,
matched integer CSV strata and equal-map C contrasts. The second runs the
focused initialization/metric/intervention fixtures. Neither trains a model
or performs GPU inference. training/validation.json is a compact record of
the full local Boolean-trace and CPU-checkpoint validation. It retains check
names/verdicts, all gate values and provenance; duplicate metric profiles and
successful-check details remain local, with the full report SHA bound in
provenance.json. Public arithmetic is narrower.

## A/B training on a fresh clone

```
python tools/prepare_seed4_followup_training_references.py
python new/seed4_followup/check.py --out analyses/NEW_FOLLOWUP_CPU.json
python new/seed4_followup/train.py --preflight --qualification analyses/NEW_FOLLOWUP_CPU.json --out runs/NEW_FOLLOWUP_PREFLIGHT
python new/seed4_followup/train.py --out runs/NEW_FOLLOWUP_TRAINING --preflight-dir runs/NEW_FOLLOWUP_PREFLIGHT
```

Use new output names. The preparation helper copies only the published
historical manifest, schedule and seed4 record into the expected archival
location; it refuses an existing directory. A/B do not require a checkpoint.
The runner retains a strict historical parameter/endpoint replay gate and
stops on drift. Exact replay across different GPU/backend configurations is
not guaranteed by these local checks. No training was repeated for upload.

## C and full local validation

The frozen C runner requires the original trained checkpoint and historical
source/run archives named by the protocol. They are excluded from GitHub.
For a local archive, use causal.py --out runs/NEW_CAUSAL and then
tools/validate_seed4_followup_causal.py with --run/--out. Full A/B saved-trace
validation uses tools/validate_seed4_followup_training.py; see its entry
arguments. Public C events retain coordinates, two-world logits, controls and
norm matching, and can reproduce reported equal-map contrasts without a model.

Raw arm records, model summaries, matched CSVs, schedules and C events are
preserved as secondary evidence. Checkpoints, full NPZ trajectories, source
snapshots, machine/process metadata, receipts and logs remain local. Their
hashes are bound in provenance.json. Read RESULTS.md and summary.json first.
