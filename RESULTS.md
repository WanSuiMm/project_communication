# First local 2D / 3D result

Canonical evidence: [per-seed results](evidence/qualification_v1_1/RESULTS.md),
[raw aggregate](evidence/qualification_v1_1/aggregate.json), and
[configuration](evidence/qualification_v1_1/config.json).

Published evidence is an allowlisted copy of the frozen local run. Its original
training-source hashes are recorded in `PUBLICATION_MANIFEST.json`.

Twenty Gate A model/seed/dimension runs completed on the local RTX 4060 Laptop
GPU in approximately 4.32 minutes. Each used 240 training updates, C=64, r=32,
eight recurrent updates, training distances <=16, and tests through d=128.
3D uses actual three-axis transport/Conv3d on a narrow 4x4 transverse volume,
rotated across all three axes. It is not general 3D geometry qualification.

## Accuracy at distance 128

Values are the separate training seeds 1729 / 2718, in percent.

| Model | 2D | 3D |
|---|---:|---:|
| CNN | 55.5 / 49.2 | 52.1 / 50.5 |
| NCA | 53.1 / 49.2 | 51.6 / 53.1 |
| Attention control | 100.0 / 49.2 | 100.0 / 48.4 |
| Constant transport | 57.8 / 49.2 | 52.1 / 51.0 |
| Learned transport | 52.3 / 49.2 | 51.6 / 49.5 |

Models are width/depth-matched, not parameter-matched. CNN has 274,370 / 283,650
parameters in 2D / 3D, versus 74,402 / 76,770 for learned transport. The attention
control is a shared SDPA message variant rather than a tuned standard ViT.

Learned transport also fails to fit the training-scale d16 evaluation
(2D 54.7 / 49.2; 3D 51.0 / 47.9). Consequently, its chance-level long-distance
scores cannot be isolated as an extrapolation failure after successful fitting.

## Cost and validity

Batch-one d128 latency, mean of each seed's synchronized median:

| Model | 2D ms/image | 3D ms/image |
|---|---:|---:|
| Attention control | 10.05 | 12.13 |
| Learned transport | 35.43 | 40.89 |

This portable float32 PyTorch implementation is slower than the attention
control at these sizes; neither number establishes an optimized-kernel ceiling.
Solver agreement, RHS/edge gradients, 2D/3D constant preservation, symmetry,
nonexpansion and GPU float32 large-scale checks passed. End-to-end smoke found
nonzero learned-edge gradients in both dimensions. These validate software
paths, not the scientific hypothesis.

See [the numerical check record](evidence/validation/transport_checks.json),
[the original model smoke record](evidence/validation/smoke.json), and the
exact numerical test `test_transport.py::run_checks`. The numerical check was
rerun during packaging; the original training sources retain their recorded hashes.

Both dimensions receive INCONCLUSIVE_POSITIVE_CONTROL because attention did
not fit reliably across both seeds. Gates B/C and the message-width sweep were
not run. The current configuration provides no evidence of an RT advantage;
the qualification does not refute the architecture family. Further scientific
comparison first requires a reliable training qualification. No extra steps,
hyperparameter changes or rescue sweep were run after observing these scores.

The initial v1 dispatch was interrupted for a distance/axis scheduling defect;
its outputs are preserved and excluded, as recorded in `RUN_MANIFEST.md`.
The user requested no continued tracking; no background training or monitoring
remains scheduled.
