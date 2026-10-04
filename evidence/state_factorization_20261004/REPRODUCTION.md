# State-factorization saved evidence

Start with RESULTS.md, summary.json, metrics.csv, validation.json, config.json
and state_factorization.png. Open raw_summary.json and per-arm JSON only after
the compact aggregate. Frozen protocol: new/state_factorization/PROTOCOL.md.

Saved-data verification from repository root (NumPy only; no model execution):

    python -X utf8 -B tools/export_state_factorization.py --verify-only

The verifier recomputes fixed cohorts from native paired traces, all16 arm
counts, support gates, immediate output comparisons and the560 CSV rows.
It checks FF/native suffix equality, source/data hashes and66 NPZ files.
Native parameter immutability and first-step equivalence are recorded checks,
not independently reproduced model executions by this verifier.

NPZ inventory:2 fresh banks,2 fixed-cohort masks,2 readout projectors,
4 native T64 state captures,4 native trace banks,4 native endpoint-logit banks,
16 intervention trace banks,16 intervention endpoint-logit banks and16 norm
trajectories. State captures are scientific activation arrays, not model
checkpoints. Projection is pointwise across Z channels using F1's weight.
Original/flip state order is all original maps followed by all flipped maps.

A full rerun requires the exact S1/F1 checkpoints identified by SHA in
config.json, historical Torch2.5.1, CUDA and matplotlib. Checkpoints and machine
receipts remain local. The original run and source snapshot were not changed.
The map-support failure remains frozen; no new maps, training or rescue run
was added during publication.
