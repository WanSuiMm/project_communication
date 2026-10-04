# W medium saved evidence

Read RESULTS.md, summary.json, metrics.csv, validation.json, config.json and
w_medium.png first. Then frozen protocol new/w_medium/PROTOCOL.md and source
map GPT_CONTEXT.md. raw_summary.json and all arrays are secondary.

Saved-data verification from repository root (NumPy; no Torch/model execution):

    python -X utf8 -B tools/export_w_medium.py --verify-only

The verifier reconstructs all six original state captures from48 paired-cue
map/time chunks and checks exact tensor hashes. It recomputes fixed cohorts,
all36 arm counts/support gates,1260 CSV rows, paired-map contrasts/2000-sample
bootstrap intervals, instantaneous W-only neutrality, native FF suffix equality
and all six coordinate-gauge suffix/logit equalities. Original raw files copied
without changes are bound by SHA256. Native Full and parameter/first-step
immutability remain recorded checks; this verifier does not execute models.

182 NPZ files:2 fresh banks,2 fixed cohorts,8 native trace/logit banks,
108 intervention trace/logit/norm banks,14 transform/index files and48 state
chunks. States are activations, not checkpoints. Every chunk stores32 original
maps then the same32 flipped maps. Reconstruct into original full-cue order
using state_chunks.json; donor-time chunks are indexed by32/48/80/96.

A full rerun requires the exact S1/F1 checkpoints listed by SHA in config.json,
Torch2.5.1, CUDA and matplotlib. Follow new/w_medium/PROTOCOL.md with NEW output
names. Source hashes bind the frozen implementation and preflight qualification.
The original local run is unchanged. No extra inference, training or optimizer
update was performed for publication. Maps are not independent training seeds.
