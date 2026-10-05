# Incremental review: native execution-state width ladder

- Review base: `ffe8b22b1130992e8d1685ce31926040653b3cc6`.
- Evidence head: `e2ffb36a3963f6f0fabf5c7ad2eb1fcc513b2b28`.
- This later commit changes only this handoff. Earlier frozen evidence and
  claims are unchanged; no new training or model inference was run for export.

## Read first

1. [Results](evidence/latent_width_20261005/RESULTS.md),
   [summary](evidence/latent_width_20261005/summary.json), and
   [validation](evidence/latent_width_20261005/validation.json).
2. [Configuration](evidence/latent_width_20261005/config.json),
   [per-arm records](evidence/latent_width_20261005/perarm.json), and
   [576 metric rows](evidence/latent_width_20261005/metrics.csv).
3. [Frozen protocol](new/latent_width/PROTOCOL.md),
   [runtime amendment](new/latent_runtime/ACCELERATION.md),
   [exact replay record](evidence/latent_width_20261005/runtime_qualification.json),
   [code map](GPT_CONTEXT.md), and
   [reproduction](evidence/latent_width_20261005/REPRODUCTION.md).

Unchanged older files need no rereading. Per-map evaluation JSON, NPZ traces,
frontier tables and training curves under `block00/`–`block15/` are secondary.

## New evidence and boundaries

COMPLETE, 96/96 planned trajectories across 16 independent initialization and
schedule blocks. All six arms have 0/16 Full successes: W24, W8, equal-parameter
W24 capacity control, W16, W4 and exploratory W2 alternating-axis. The frozen
W8-versus-W24 primary verdict remains `NO_W8_RELIABILITY_QUALIFICATION`:
0 wins, 0 losses, net0/16, exact two-sided p=1. The concurrent W24 also fails
Full; these results do not establish intrinsic minimal dimension or reject all
continuous latents. W2 changes transport as well as width.

13 eager trajectories were imported; 83 used CUDA Graph training. Exact
300-update block00 replay qualified all six arms' final parameters and Adam
state/groups. That is a finite runtime check, not an efficacy qualification.
The 2771.016-second elapsed field covers the accelerated worker only, excluding
original training, replay qualification and handover preparation.

Full numeric measurement data are retained losslessly, about427MiB. All96
training curves, 192 paired trace banks, frontier tables, per-map summaries,
systems measurements and batch schedules are present. Checkpoints and machine
records remain local; checkpoint/parameter hashes remain in the public records.
CPU-regenerated task banks match runtime tensor hashes exactly. Six CPU tests
and the saved-data verifier pass; validation.json states what was recomputed.

    python -X utf8 -B tools/export_latent_width.py --verify-only

Review questions: which Full components fail across the widths and concurrent
baseline; does any continuous diagnostic change without meeting Full; which
conclusions are identifiable from this frozen recipe? Keep those interpretations
separate from the primary verdict and transport-confounded W2 condition.
