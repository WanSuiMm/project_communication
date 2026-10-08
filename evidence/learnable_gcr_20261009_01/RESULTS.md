# Learnable GCR: one-block developmental screen

Execution: **COMPLETE**. Frozen verdict: **POSITIVE_CONTROLS_UNQUALIFIED**.

One paired initialization/schedule block, 150 updates, terminal T64 MSE only. This does not estimate architecture reliability.

| Arm | Held-out T64 MSE | Held-out T64 R2 | Length64 T128 R2 | Length96 T128 R2 |
|---|---:|---:|---:|---:|
| full_writer_k8 | 0.202610 | -0.0017 | -0.0023 | -0.0082 |
| full_writer_k64 | 0.202606 | -0.0016 | -0.0019 | -0.0108 |
| gcr_k8 | 0.149662 | 0.2601 | 0.4034 | 0.2364 |
| gcr_k64 | 0.093906 | 0.5358 | 0.6697 | 0.5385 |
| fixed_raw_evidence_mlp | 0.036286 | 0.8206 | 0.8590 | 0.8519 |

The fixed-evidence MLP is an offline readout diagnostic, not a matched NCA.

The continuous architecture-by-credit contrast and all systems/evaluation metrics are in [aggregate.json](aggregate.json).
Per-example predictions and exact data banks are compressed NPZ. Models and optimizer checkpoints remain local.

Qualification establishes path/algebra/gradient plumbing only. K64 learning qualification is decided by the scientific endpoint.
A GCR-K8 success would reflect shared local parameter learning from suffix uses; it would not recover gradients of individual distant early writes.
The task matches second-order evidence. Routing, cyclic duplicates, general learned write, other tasks and multiple seeds remain untested.
