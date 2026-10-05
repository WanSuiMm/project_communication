# Latent factorial saved evidence

Start with RESULTS.md, summary.json, validation.json and config.json. Continue
with perarm.json and metrics.csv. The per-arm evaluation summaries contain the
Full components and per-map metrics. NPZ traces, training curves, runtime and
systems data, and frontier CSVs are the detailed records.

Saved-data verification from the repository root:

    python -X utf8 -B tools/export_latent_factorial.py --verify-only

The verifier needs NumPy and PyTorch and runs on CPU. It verifies source,
checkpoint and tensor hashes; regenerated banks; all 32 schedules; all 300
update clocks for each arm; exact pairing of all 256 Boolean trace banks;
coverage, retention and regression counts; the unchanged Full predicate;
all 768 metric rows; and the saved aggregate against the frozen runner. It
does not rerun logits or recompute matched-frontier strata. Read validation.json
for the exact verification scope.

For a new experiment, follow new/latent_factorial/PROTOCOL.md. The frozen
commands are:

    python -X utf8 -B new/latent_factorial/run.py --check --out analyses/NEW_FACTORIAL_CHECK.json
    pwsh -File tools/launch_latent_factorial.ps1 -RunName NEW_FACTORIAL_RUN -Qualification analyses/NEW_FACTORIAL_CHECK.json

A fresh run requires the historical PyTorch 2.5.1 environment and a compatible
CUDA device. The independent statistical unit is the paired training block
(32), not pixels, maps, rollout times or timing repetitions.
