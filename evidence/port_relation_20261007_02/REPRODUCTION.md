# Completed port-relation screen: reproduction and evidence map

Read RESULTS.md, summary.json and final_metrics.csv first. The complete run has
24 formal trajectories,312 stages and624 complete Boolean banks, with no
NUMERICAL_FAILURE stage. No intermediate checkpoint replaces fixed u300.

From the repository root, a CPU saved-data check needs NumPy only:

```powershell
python -X utf8 -B tools/export_port_relation.py --verify-only
```

It verifies source/archive/file hashes, bank tensor hashes, schedule seeds,
completion-marker bindings, all312 evaluator-to-metric rows, the aggregate,
all624 packed shapes, all48 formal Boolean conjunctions/reach/retention gates,
and the24 curves'256 forward/32 backward/one optimizer cadence. Old Full is
retained from the saved evaluator; inference and optimizer updates are zero.

The raw/block00.zip through raw/block07.zip archives contain original bytes
with run-relative paths: all per-stage summaries, relation-activity records,
size32/64 packed traces, full-endpoint per-map CSVs, training curves and atomic
completion markers. Dense/perarm/plans JSON is losslessly gzipped. Numeric
banks are included. Checkpoint/optimizer contents, private launch metadata and
transient logs stay local; checkpoint hashes remain in records and markers.
PORT_RELATION_PUBLICATION_MANIFEST.json at the repository root binds archive
entries, sources, qualification, banks, plans and the recovery bridge.

For a NEW independent reproduction, install the repository requirements with
CUDA-enabled PyTorch2.5.1. Use unique output names and run from the repository
root (qualification below performs a small CUDA smoke, not the formal run):

```powershell
python -X utf8 -B new/port_relation/check_cells.py
python -X utf8 -B new/port_relation/check_activity.py
python -X utf8 -B new/port_relation/reporting.py --self-test
python -X utf8 -u -B new/port_relation/run.py --check --out analyses/port_relation_fresh_check.json
python -X utf8 -u -B new/port_relation/run.py --out runs/port_relation_fresh --qualification analyses/port_relation_fresh_check.json
```

Fresh execution generates the same bank seeds and300-update plans; no private
parent/checkpoint is needed. Windows independent execution can replace the last
line with `pwsh -NoProfile -File tools/launch_port_relation.ps1 -RunName port_relation_fresh -Qualification analyses/port_relation_fresh_check.json`.
The Windows helper requires an interactive scheduled-task session and available
CUDA/disk headroom. These instructions were source-inspected; a second formal
experiment was not launched for this publication. The completed experiment's
actual-shape qualification and saved-data checks are published separately.

The preserved protocol-header name port_relation_conditioned_v1 is an alias
for runtime ID port_relation_v1_full_writer_native_k8. The protocol's phrase
"backpropagate once" summarizes accumulation; the executed implementation
backpropagates32 K8 endpoint losses divided by32 and takes ONE optimizer step.
Model equations, endpoint, seeds and all256 physical forward steps are unchanged.

The original parent stopped after2 trajectories/33 stages on diagnostic float32
squaring overflow after finite T256 state/logit checks. The child inherited those
33 committed stages, restored conditioned u150 with Adam/RNG and replayed the
unsaved prefix. Qualification checks identical saved u175/u150 report dictionaries
and packed traces, including loading live parameters into cached CUDA Graphs.
This is not an uninterrupted bitwise claim; backend nondeterminism is retained.
Parent interruption and zero-training regression are not scientific endpoints.
Qualification timings are single engineering observations, not repeated systems
benchmarks. The resumed process lasted4799.797seconds (~80minutes), with original
partial process time recorded separately.
