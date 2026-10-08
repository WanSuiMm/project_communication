# Saved-data reproduction

Read RESULTS.md, summary.json and final_metrics.csv first. The 12 unit folders
retain unchanged endpoint records, all per-map metrics and packed Boolean traces
for every valid integer time. Each u300 curve contains all 300 updates; earlier
curve files are redundant prefixes and are represented by their completion hashes.
All 72 checkpoint bindings, four minibatch schedules and three exact banks remain.
Model/Adam/RNG checkpoint contents and private process receipts stay local.

From repository root with NumPy and PyTorch (CUDA is not needed for verification):

```powershell
python -X utf8 -B tools/export_delayed_credit.py --verify-only
```

NPZ decoding: for correct/original_correct/flipped_correct use numpy.unpackbits
on the _packed array, bitorder="little", count=prod(_shape), then reshape.
valid_times and valid_through_step specify the finite prefix; never interpret a
discarded nonfinite suffix as an incorrect Boolean prediction.

The frozen protocol and source are under new/delayed_credit. A fresh experiment
uses run.py --check in a new qualification directory, then run.py --out in a new
run directory with --qualification pointing to that qualification.json. Training
requires the established Torch2.5.1 CUDA environment. No new training or inference
is performed by this exporter. The K64 control failed qualification, so these
results do not identify a K8-versus-K64 learnability gap or its mechanism.
