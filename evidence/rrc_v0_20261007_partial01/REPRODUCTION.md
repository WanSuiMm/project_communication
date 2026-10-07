# RRC partial snapshot verification

Read RESULTS.md and summary.json first. This package is incomplete and includes
only fully recorded stages. Training curves are frozen through each last recorded
checkpoint; a partly written subsequent evaluation is excluded. Dense and curve
JSON is losslessly gzipped; traces retain every integer T0..T256 as packed bits.
Model/optimizer contents, private receipts, logs and original machine manifest
stay local. Checkpoint hashes bind the omitted checkpoint contents.

From the repository root:

```
python -X utf8 -B tools/export_rrc_v0.py --verify-only
```

This is a NumPy/CPU saved-data check, with no inference, training or optimizer
updates. It checks source/file/raw hashes, all saved evaluator summaries, trace
shapes, u300 paired correctness and reach/retention, bank/schedule bindings,
training cadence, and the unchanged partial reporter aggregate. It does not rerun
the saved old Full gate. The full experiment recipe is in
[PROTOCOL.md](../../new/rrc_v0/PROTOCOL.md).
