# Results: inertial NCA screen, with preserved A0/v1 evidence

## Latest: same-medium generic momentum control and trajectory audit

Masked Momentum completed800 updates and all three evaluation sizes on2026-10-01.
Exact initial parameters/RNG match the old generic control; logged rollout and
damage schedules match both references. All evaluation statuses are valid.

Primary endpoint: size32, T64, one model seed and16 held-out maps:

| Model | BA (%) | Paired-source correctness (%) | BCE |
|---|---:|---:|---:|
| Masked Momentum | 96.45 | 77.25 | 0.0512 |
| Masked Inertial RD | 88.79 | 37.22 | 0.2503 |
| Original unmasked Momentum | 84.44 | 42.75 | 0.3217 |

Masked Momentum beats the explicit candidate by7.66pp BA and40.02pp paired
correctness, satisfying the prespecified descriptive reverse joint5pp criterion.
Relative to the original generic control, masking gains12.01pp and34.49pp.
This is a negative primary result for explicit-factorization superiority in
this seed/recipe; it is not a multi-seed or isolated causal factorization result.

Long rollout reverses the BA ranking at sizes32 and64:

| Size | Momentum BA T64 → T256 (%) | Inertial RD BA T64 → T256 (%) |
|---|---:|---:|
| 32 | 96.45 → 77.68 | 88.79 → 91.55 |
| 64 | 84.75 → 64.15 | 75.39 → 75.69 |
| 128 | 75.20 → 51.90 | 57.47 → 50.00 |

At32/T256, paired correctness is42.78% for Momentum versus41.92% for Inertial
RD, despite the opposite BA ranking. BA alone is insufficient to rank source
information retention. Momentum BCE rises0.0512 to6.8694 and H RMS36.33 to219.81
fromT64 toT256. Both models lack a sustained95% endpoint; a single96.45% peak
does not satisfy it. Momentum's changed-component accuracy after a source switch
and64 extra steps is29.04% at32. Conditional repair cohorts differ (12 versus7
eligible maps), so their conditional percentages are not a matched repair comparison.

Read [full control results](evidence/masked_momentum_seed0/RESULTS.md) and
[all comparisons](evidence/masked_momentum_seed0/comparison.json). The separate
[candidate audit](evidence/trajectory_audit_seed0/INTERPRETATION.md) finds open
state growth, increasingly confident errors and component-mean drift. It supports
a finite-horizon amplification interpretation, not a stable attractor. Independent
temperature fitting does not change decisions or fix the128-scale failure.
No corresponding detailed trajectory audit of generic Momentum was performed.

No automatic retuning, extra seeds, new architecture or follow-up run is scheduled.

## Previous follow-up: masked medium, 2D seed 0

**Interpretation correction:** improved classification accuracy with longer
rollout does not establish a stable attractor or fixed-point convergence.
For masked inertial RD at32, T64/128/256 BA is88.79/89.96/91.55%, while
BCE rises0.250/0.657/1.452, H RMS rises17.19/50.88/126.95 and V RMS
rises0.483/0.575/0.617. The existing RMS includes walls; BCE uses only open
pixels. These observations warrant an inference-only trajectory audit, not
an assertion of exponential divergence or calibrated convergence. Conditional
decision recovery is distinct from return to a stable hidden state. Source
revision failure does not by itself identify a deep attractor basin.

The separately authorized [trajectory audit](evidence/trajectory_audit_seed0/INTERPRETATION.md)
is complete, with exact replay of all15 stored checkpoints. Open-pixel H RMS
at32 rises14.48 to120.61 fromT64 toT256; wrong-margin median changes -1.889
to -18.421. Independently calibrated test BCE atT256 is0.2492 versus raw1.4519,
with unchanged BA. Componentwise-constant H energy is97.04% at32/T256, but
also99.51% at128/T256 where BA is50%. These support finite-horizon amplification
and mean drift, not convergence or correct propagation throughout the domain.
The completed same-medium generic momentum comparison is reported above.

Both masked arms completed 800 updates with matched initialization, RNG and
logged rollout/damage schedules relative to their frozen unmasked controls.
Only the structured transport operator changes to input-mask edge weights.

| Model | BA at 32x32/T64, old → masked | Paired correctness, old → masked | Prespecified descriptive outcome |
|---|---:|---:|---|
| RD | 81.27% → 73.21% | 37.81% → 17.13% | NO_JOINT_5PP_GAIN |
| Inertial RD | 69.94% → 88.79% | 17.81% → 37.22% | JOINT_GAIN_5PP |

At T256, masked inertial RD reaches 91.55% BA at size 32, 75.69% at size 64,
and 50.00% at size 128. No sustained aggregate 95% threshold is reached.
This supports an operator-dependent improvement within the tested inertial
recipe, with unresolved scale generalization. The opposite RD effect prevents
a blanket claim that masking helps both dynamics. One seed is not statistical
significance; no same-medium momentum comparison or architecture superiority
claim is established. Repair eligibility counts alone do not prove repair.

Read [compact results](evidence/masked_medium_seed0/RESULTS.md),
[all paired horizons](evidence/masked_medium_seed0/paired_comparison.json), and
[publication manifest](MASKED_PUBLICATION_MANIFEST.json). Detailed per-arm JSON
under `evidence/masked_medium_seed0/arms/` is secondary. Original evidence below
remains unchanged.

## Previous: inertial NCA, 2D seed 0

**Completed, negative exploratory screen for the current recipe.** Four arms
completed 800 updates and all planned evaluations in approximately 7 minutes
38 seconds. No numerical-failure or budget-limit status was recorded. Read the
[new per-arm tables](evidence/inertial_seed0/RESULTS.md),
[summary](evidence/inertial_seed0/summary.json) and
[configuration](evidence/inertial_seed0/config.json) first.

Balanced accuracy (%), 16 held-out maps at each size, one training seed:

| Model | 32x32, T64 | 32x32, T256 | 64x64, T256 | 128x128, T256 |
|---|---:|---:|---:|---:|
| State-matched NCA | 85.17 | 82.85 | 73.03 | 55.54 |
| Momentum NCA | 84.44 | 73.38 | 74.47 | 67.18 |
| RD NCA | 81.27 | 74.79 | 56.89 | 50.33 |
| Inertial RD | 69.94 | 53.50 | 50.00 | 50.00 |

The candidate already underperforms at the training size and within the training
rollout range. At 32x32 its accuracy drops from 72.89% at T32 to 53.50% at T256.
At 64x64 and 128x128, paired-source correctness at T256 is zero. The tested
reaction/transport parameterization shows no benefit over generic momentum.

No arm reaches sustained aggregate 95% BA, so matched-quality time-to-solution
is null. Inertial RD has zero pre-damage eligible maps at all sizes: its main
conditional-repair endpoint is **unevaluable**, not a measured zero repair rate.
Other models learn partial structure, but this recipe has not established a
robust high-quality reference solution. Gradient clipping and latency are recorded;
they do not establish trainability or systems superiority.

This is one exploratory training seed, not a significance claim, an NCA/PDE-family
refutation, or a 3D result. It is a different task and operator from A0; their
accuracy values must not be compared as successive versions of one benchmark.
No follow-up seed, new architecture or experiment was launched after this result.

[Validation](evidence/inertial_seed0/validation.json) and
[linear checks](evidence/inertial_seed0/linear_checks.json) establish implementation
checks only. [INERTIAL_PUBLICATION_MANIFEST.json](INERTIAL_PUBLICATION_MANIFEST.json)
binds the frozen executed sources and byte-identical per-arm output files.

## Earlier A0 experiment

Earlier protocol: [rt_a0_2d3d_v2_1](A0_PROTOCOL.md). Read the
[compact summary](evidence/a0_v2_1/summary.json),
[full per-seed table](evidence/a0_v2_1/RESULTS.md) and
[configuration](evidence/a0_v2_1/config.json) first. The
[raw aggregate](evidence/a0_v2_1/aggregate.json) retains histories, axis metrics
and diagnostics. [A0_PUBLICATION_MANIFEST.json](A0_PUBLICATION_MANIFEST.json)
binds current code and evidence to the run.

## Final A0 qualification

The conditional schedule completed 24/24 trials: eight attention trials across
two dimensions/four seeds, then sixteen 2D RT trials. Sixteen potential 3D RT
trials were skipped after failed calibration. This is a completed schedule,
not an unfinished 24/40 run or budget exhaustion.

| Dimension | Attention control | RT outcome |
|---|---|---|
| 2D | 4/4 seeds reach 100% in every evaluation condition | All four arms NOT_QUALIFIED_FIT |
| 3D | Seed 1729 has 47.92% train d16; other three seeds reach 100% | INCONCLUSIVE_POSITIVE_CONTROL; RT not run |

2D accuracy (%), seeds 1729 / 2718 / 31415 / 57721:

| Arm | Train-shape d16 | d128 |
|---|---:|---:|
| Attention | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| Constant raw | 50.78 / 46.09 / 55.47 / 43.75 | 50.78 / 45.31 / 54.69 / 46.09 |
| Constant normalized | 50.78 / 50.78 / 55.47 / 43.75 | 50.78 / 47.66 / 55.47 / 46.88 |
| Learned raw | 50.78 / 46.09 / 55.47 / 43.75 | 50.00 / 44.53 / 54.69 / 46.09 |
| Learned normalized | 50.78 / 45.31 / 55.47 / 44.53 | 50.78 / 45.31 / 56.25 / 45.31 |

There is no fitted RT arm whose distance generalization can be qualified.
Confidence normalization does not resolve learning failure in this frozen
configuration. The reliable 2D control makes this a clearer negative result
than v1, but it does not establish an architecture-family impossibility or
identify emission versus reaction/readout as the cause.

## Paired A0 normalization effect

Mean of four paired seed differences, normalized minus raw, percentage points:

| Medium | Train d16 | Large-grid d16 | d32 | d64 | d128 |
|---|---:|---:|---:|---:|---:|
| Constant | +1.172 | -0.586 | +1.953 | -1.953 | +0.977 |
| Learned | +0.000 | -0.195 | +0.977 | -0.977 | +0.586 |

At d128, constant-medium seed differences are 0.000 / +2.344 / +0.781 / +0.781
points; learned-medium differences are +0.781 / +0.781 / +1.562 / -0.781 points.
Small positive averages at one distance do not establish useful communication
when both arms remain near chance and fail fitting. Four seeds are replicate
units; these are descriptive estimates, not significance or equivalence claims.

Both arms use q=c*v, identical initial parameters within each pair, unit
initial conductance and matched packed solver work. Only receiver division
changes within a pair. A0 raw differs from v1 raw. Changes between experiments
prevent attributing improved attention calibration to any single intervention.

## A0 diagnostics and cost

[Checks](evidence/a0_v2_1/checks.json) passed for grouped packing, normalized
q/confidence/edge/tau gradients, supplied-value attention, paired initialization,
initial medium matching and finite model gradients in both dimensions.

In the fixed oracle cases, tau4096 permits correct raw-sign decoding through
d128 on every tested axis. At d128, normalized value error is at most 0.00445;
at tau1, all 40 sampled bits become undecodable zeros. This is recoverability in
the tested large-scale oracle setting, not learned communication at every scale.

The normalized arms' sampled final-update d128 probes retain mean background
confidence about 0.34-0.48, versus source confidence about 0.47-0.53. Thus the
source-only oracle confidence assumption is not realized in those probes.
None of their sampled target denominators is <=epsilon. Confidence-mass share
is not a measure of total bit information: earlier recurrent updates can move
content into values emitted elsewhere. These observations do not isolate cause.

| 2D arm | Mean of seed-median batch-one d128 latency, ms |
|---|---:|
| Attention | 22.13 |
| Constant raw / normalized | 35.90 / 37.20 |
| Learned raw / normalized | 36.58 / 37.33 |

A0 includes a finite-logit evaluation check and matched packed solver work;
cross-version timings are not kernel improvement/regression estimates. These
are portable reference implementations, not optimized ceilings. A0 has no
speed-pass gate. Models are not parameter-matched; exact counts are in the
summary. Narrow straight geometry and fixed axis splitting remain limitations.

Decision: do not escalate this configuration to B/C, rank sweeps or real vision.
Any further experiment needs a distinct bounded diagnostic. No new experiment
or monitor is scheduled by this publication. 3D normalization remains unmeasured.

## Historical first local 2D / 3D result

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

### Historical v1 accuracy at distance 128

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

### Historical v1 cost and validity

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
