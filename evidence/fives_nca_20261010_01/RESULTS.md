# FIVES real-task NCA v0 results

Status: **COMPLETE**. Scientific verdict: **TASK_UNQUALIFIED**.

One developmental block trained `standard_k64` for 1,500 updates with a 64-step rollout and full 64-step credit. All final metrics use the same saved checkpoint at 512 × 512 evaluation horizons; T8 is an evaluation horizon for this K64-trained model, not a separately trained T8 reference.

| Evaluation | Mean Dice | Mean clDice |
|---|---:|---:|
| Coarse teacher | 0.870909 | 0.885511 |
| T8 | 0.871031 | 0.885600 |
| T16 | 0.871134 | 0.885729 |
| T32 | 0.871031 | 0.885885 |
| T64 | 0.869917 | 0.885686 |
| T128 | 0.864772 | 0.883164 |
| T256 | 0.840710 | 0.853819 |

Final horizon means use 120 validation images and two matched stochastic plans per image. The repeated plans are measurements, not independent samples. `final_per_image_metrics.csv` contains 720 image-by-horizon rows averaged over those plans; `aggregate.json` preserves all 1,440 plan-specific final rows and raw gate inputs.

Exactly two frozen gates failed: `k64_cldice_gain` (T64 clDice improvement over the coarse baseline was +0.000175, below +0.010) and `within_model_depth` (T64 minus T8 clDice was +0.000086, below +0.010). The coarse floors and T64 Dice guard passed. The overall run completed, but the task remained unqualified.

The official training split supplied 480 development images and 120 validation images, with image ID as the split unit. No patient mapping was available, so this is not a patient-disjoint result. The official archive was unpacked; test labels were excluded from discovery, decoding, training, and evaluation. No test labels were loaded.

The separate `standard_t8`, `standard_k8`, and `au_k8` arms were not run. This evidence supports no AU conclusion and does not establish that the task generally lacks long-computation benefit.

The process-level pipeline elapsed time was 1556.739 seconds. It covers the recovered pipeline process, excludes earlier acquisition/download/transfer work, and is not an end-to-end speed measurement.

Intermediate T64 checkpoints are formation diagnostics on the fixed 16-image cohort:

| Update | Mean Dice | Mean clDice |
|---:|---:|---:|
| 250 | 0.889178 | 0.894720 |
| 500 | 0.892804 | 0.898210 |
| 1000 | 0.888231 | 0.897809 |

Saved software-check results are in `software_check.json`. `training_samples.jsonl` contains 151 sampled update records, not a complete per-update log. `manifest.json` preserves the split, OOF/full-teacher provenance, and checkpoint/data hashes. The source hashes in `provenance.json` bind the run to `new/real_task_fives`.

Offline verification: `python tools/publish_fives_nca.py --verify`.
