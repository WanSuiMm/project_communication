# Results: inertial NCA screen, with preserved A0/v1 evidence

## Latest: Phase-II initialization replication misses its K8 threshold

Execution: **COMPLETE**, 12 arms at 300 updates each, 1095.047 seconds.
**K8 narrow reach and sustained replication are both `NOT_REPLICATED`: 2/4
seeds pass, below the frozen 3/4 criterion.** All four K64 controls remain
unqualified, so the comparison is separately `BASELINE_UNQUALIFIED`.

| Gradient window | Narrow reach | Narrow reach and hold |
|---|---:|---:|
| K8 | 2/4 | 2/4 |
| K16 | 3/4 | 1/4 |
| K64 | 0/4 | 0/4 |

Only additive was tested. Initialization seeds2/3/4/5 share training bank10002
and schedule20002. New evaluation banks40032/40064 contain32 maps per size.
Primary size32/T64 uses **strict16<d<32**, with both per-map and pooled paired
accuracy>=80% and both original/flipped BA>=85%. Hold limits the decline at
T128 and T256. This selected narrow endpoint was frozen prospectively after
Phase I; it does not replace or pass Phase I's broader d>16 gate.

Primary paired percentages (per-map mean / pooled pixels):

| Initialization seed | K8 | K16 | K64 |
|---|---:|---:|---:|
| 2 | 93.70 / 92.97 | 99.06 / 98.83 | 0.16 / 0.14 |
| 3 | 21.85 / 17.27 | 90.19 / 87.77 | 76.44 / 70.56 |
| 4 | 52.40 / 44.76 | 57.56 / 51.75 | 0.00 / 0.00 |
| 5 | 96.45 / 95.68 | 92.90 / 90.47 | 12.88 / 9.49 |

K8 seeds2/5 remain positive examples: primary pooled accuracy is93.21/93.59%
at size32/T256 and89.65/97.25% at size64/T256. Yet d>32 pooled accuracy on
size64/T256 is only36.78/14.83%. Narrow propagation transfers better than broad
far propagation. Size32 has no changed-component pixels at distance>=64 in
this bank; null entries must not be read as zero accuracy.

K16's primary band lies within its single-window radius32, so its3/4 reach
does not prove cross-window composition. Only seed2 also holds. K8 is worse
than K64 on seed3; there is no general superiority or optimal-K conclusion.
All K8 clipping rates are<=6%, including both reach failures; low clipping is
not sufficient for success. K16 seed2 clips60% and still reaches and holds.
The clipping observation does not identify a causal mechanism.

Peak allocated CUDA memory K8/K16/K64 is95.12/155.81/519.41MiB; per-arm
training times are84.28-92.86 seconds. These are descriptive systems numbers,
not activation-only memory or an accuracy-matched speedup result.
Verification covered18 source snapshots,12 checkpoints, shared identities,
144 BA aggregates,1008 paired aggregates/denominators and all12 arm decisions.
No extra training or inference. n=4 is conditional on one training bank and
schedule, without a significance or cross-distribution robustness claim.

[Full report](evidence/short_bptt_phase2_init2345/RESULTS.md),
[compact metrics](evidence/short_bptt_phase2_init2345/analysis.json),
[all curve rows](evidence/short_bptt_phase2_init2345/curves.csv),
[frozen protocol](new/short_bptt_phase2/PROTOCOL.md), and
[publication hashes](BPTT_PHASE2_PUBLICATION_MANIFEST.json).

## Previous: matched short-BPTT Phase I, one positive case with unqualified controls

Execution: **COMPLETE**, eight arms at 300 updates each, 730.797 seconds.
Scientific status: **BASELINE_UNQUALIFIED for both architectures**; all four
full-K64 controls fail the predeclared far-paired threshold. This prevents a
qualified comparative verdict, not publication of the observed outcomes.

Primary size32/T64 percentages; far means graph distance >16. Paired correctness
requires the same pixel to be correct under both fresh source alternatives.
The statistic averages eligible per-map fractions, with model seed the
independent unit (n=2 per architecture).

| Cell | Seed | K64 BA | K8 BA | K64 far paired | K8 far paired | K8 reach / hold predicates |
|---|---:|---:|---:|---:|---:|---|
| Additive | 0 | 87.20 | 91.22 | 15.83 | 23.18 | Fail / Fail |
| Additive | 1 | 87.73 | 98.03 | 0.00 | 80.87 | Pass / Pass |
| Revision | 0 | 78.04 | 83.90 | 21.44 | 0.67 | Fail / Fail |
| Revision | 1 | 86.16 | 64.38 | 0.00 | 17.81 | Fail / Fail |

Additive K8 seed1 retains 79.20%/78.24% far paired at T128/T256. This is an
example of computation beyond the single-window dependency radius (two
communication phases per macro-step, hence radius16 for K8). It is not an
architecture-level pass: the other short runs fail reach, all full controls
are unqualified, and **80.87% per-map mean corresponds to only 59.53% pooled
far-pixel accuracy**. The same seed scores 45.60% in distance [32,64), and
0% at distance >=64 on size32/T64 (the latter has only one eligible map).
Size64 far paired is 28.40% at T64 and 41.51% at T256. Reliable long-distance
and size extrapolation are not established.

Peak allocated CUDA memory is 514.04 MiB for K64 versus 90.49 MiB for K8
across both cells and seeds, **82.4% lower**. This includes data, optimizer
and gradient storage. Training times are similar (86-96 seconds per arm).
Additive K64 clips 88.67%/90.67% of updates versus 6.67%/6.67% for K8;
this does not establish clipping as the cause of full-control failure.

Forward trajectories, eight equally weighted losses and one optimizer update
per trajectory are matched. The new objective and runtime-selected 300-update
budget cannot borrow qualification from the older 600-update recipe.
CPU publication analysis verified executed sources, parameter/data/schedule
hashes, 96 BA aggregates, 336 paired aggregates and all four pair decisions.
No extra training or inference, rescue tuning, 3D test, warm source-switch
test or significance claim is included.

Read [full report](evidence/short_bptt_paired01/RESULTS.md),
[compact analysis](evidence/short_bptt_paired01/analysis.json),
[figure](evidence/short_bptt_paired01/overview.png),
[frozen protocol](new/short_bptt/PROTOCOL.md), and
[publication hashes](BPTT_PUBLICATION_MANIFEST.json).

## Previous: candidate/workspace source-switch audit, zero training

Revision seed0, sizes32/64,16 fixed maps each; completed in9.609 seconds. All232
saved replay comparisons and candidate-step reconstruction have zero error.
Warm candidate readout itself is old-aligned: size32 new-target accuracy is0%
at K64/K128; size64 is2.09%/1.63%. Slow averaging of an otherwise correct
candidate is therefore not the observed failure.

Full-rollout changed-component accuracy at K128 (%):

| Intervention at old T64 | Size32 | Size64 |
|---|---:|---:|
| Retain W and Z, switch input | 0.00 | 1.64 |
| Reset W to new-input encoding, retain Z | 37.50 | 43.36 |
| Reset Z to zero, retain W | 37.50 | 37.17 |
| Transplant W from new-input T64, retain Z | 37.50 | 45.39 |
| Transplant Z from new-input T64, retain W | 37.50 | 45.40 |
| Cold reset of both | 100.00 | 99.49 |

At size32/K128, **all four single-block interventions predict negative on every
changed-component pixel**;37.5% equals the six negative labels among16 maps.
This is fixed-label behavior, not partial recovery. Resetting Z also harms
unchanged-component accuracy (77.57%/61.70% at sizes32/64). Candidate-only
cross-interventions at fixed new X show dependence on both W and Z and a
nonadditive response. Donors from different trajectories can be off-distribution;
the result does not prove a W-only mechanism or that both blocks must generally
be reset. Mature donors include extra computation and are diagnostic controls.

The earlier `NO_JOINT_SCREEN_PASS` remains unchanged. No seed1 training-failure
diagnosis, short-BPTT test, new architecture or retraining is included.
Read [audit report](evidence/switch_audit_seed0/RESULTS.md),
[figure](evidence/switch_audit_seed0/overview.png),
[compact metrics](evidence/switch_audit_seed0/analysis.json),
[exact replay scope and contrasts](new/switch_audit/REVIEW_NOTES.md), and
[publication hashes](SWITCH_PUBLICATION_MANIFEST.json).

## Previous: Workspace + Revision paired screen fails the joint gate

Execution: **COMPLETE**, all four arms at 600 updates, 1112.28 seconds.
Scientific decision: **`NO_JOINT_SCREEN_PASS`**. Primary size32 hold is minimum
aggregate BA across T128/T192/T256; the mean paired revision-minus-additive
effect is **-9.02 pp** (required >=+5 pp and positive in both seeds).

| Seed | Additive BA64 (%) | Revision BA64 (%) | Additive hold (%) | Revision hold (%) | Hold effect (pp) | Revision warm changed K64 (%) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 94.29 | 99.19 | 94.18 | 100.00 | +5.82 | 0.00 |
| 1 | 78.54 | 50.00 | 73.86 | 50.00 | -23.86 | 37.50 |

Seed 0 supplies a real reach/hold observation: revision BA is 100% at measured
T96 through T256 on size32, and 99.44% at size64/T256. Yet after a source switch,
size32 changed-component accuracy is 0% at K64 and K128 versus 100% for a cold
restart; unchanged-component accuracy stays 100% at K64. **Holding an answer
has not become revising an answer.** Seed 1 never qualifies, so the result does
not support a reliable revision advantage. Two model seeds are the independent
units; 16 shared evaluation maps per size are not additional training replicates.

Seed 0's size32 Z-only repair reaches 100%, but W/Z joint repair only 90.51%
after 64 steps; at size64 these are 96.91% and 80.42%. W RMS still grows from
2.942 to 9.718 at size32/T64 to T256. Thus neither general regeneration nor
full-state stability is established. Seed 1 has no eligible clean95 maps for
conditional repair, so its null conditionals must not be presented as recovery.

Both arms have 5033 parameters, W24/Z8 and two communication phases per step,
with matching initial parameters, data and schedules within each seed. The
only intervention is additive `Z+0.5Q` versus revision `Z+0.5(Q-Z)`. This tests
the rule within a shared workspace split, not the value of the split itself.
Older State/Momentum recipes are not matched controls for this new training.
The preflight-based 800-to-600 amendment occurred before efficacy training.

Read the [interpretation](evidence/workspace_revision_paired01/INTERPRETATION.md),
[compact metrics and failed predicates](evidence/workspace_revision_paired01/analysis.json),
[overview](evidence/workspace_revision_paired01/overview.png) and
[frozen protocol](new/workspace_revision/PROTOCOL.md). The
[validation](evidence/workspace_revision_paired01/validation.json) independently
recomputes 400 BA aggregates and all 22 seed-level gate predicates.
[Publication hashes](REVISION_PUBLICATION_MANIFEST.json) bind executed sources,
raw records and schedules. No new training or GPU rollout was run for publication.

## Previous: completed zero-training generic dynamics audit

Four existing generic State/Momentum checkpoints, whole-grid and masked media,
sizes32/64, eight anchors and windows1/8/16. Completed in112.859 seconds, without
training. All40 historical replay comparisons have zero error; source and
checkpoint hashes match. Performance curves use16 maps; Jacobian diagnostics
use fixed maps0..3. One trained seed remains the independent replicate.

**No near-zero-to-positive crossing of maximum finite-window gain is observed.**
All256 open K16 estimates are positive, already large before the performance
peak, and generally decline while late performance deteriorates. This does
not imply improving stability or exclude task-specific neutral modes, longer
product instability or behavior in unsampled windows.

Masked Momentum, size32; g16=log(sigma_estimate)/16 for the window starting at T:

| T | BA %, all16 | BCE | Paired % | Median g16, maps0..3 | Converged /4 |
|---|---:|---:|---:|---:|---:|
| 32 | 91.47 | 0.1598 | 49.82 | 0.70763 | 3 |
| 64 | 96.45 | 0.0512 | 77.25 | 0.27742 | 3 |
| 128 | 86.17 | 0.9152 | 53.28 | 0.16949 | 2 |
| 256 | 77.68 | 6.8694 | 42.78 | 0.17582 | 4 |

The four diagnostic maps themselves also decline: BA97.14%,86.40%,77.87% at
T64,T128,T256. Masked State shows BA90.43% to11.38% from T64 toT256 while
median g16 decreases0.14666 to0.10359. Short-window maximal gain is therefore
not a sufficient account of the observed performance trend.

Persistent motion and growing wrong confidence are directly observed. For
Masked Momentum at32, mean per-map open H RMS grows42.57 to258.87 from T64
toT256, while V RMS remains approximately1.1. The median source-flip logit RMS
on the changed component grows25.26 to63.75, despite declining paired
correctness. A growing response to source identity is not necessarily useful
task information. The component-constant H-energy fraction falls1.79% to0.20%
(mean of per-map fractions), so the earlier explicit-inertial mean-drift account
must not be transferred wholesale to generic Momentum.

**Diagnostic limits:** only562/1536 product estimates converge by the frozen
criteria;148/256 converge in the main open K16 subset. Unconverged values are
not certified maxima or upper bounds. At relative RMS perturbation1e-4,
worst-direction linearization error has median46.44%, versus1.19% for random
directions. Large infinitesimal gains often fail to quantitatively predict
these finite perturbations. Free momentum alone yields positive finite-window
gain (masked K16 sigma8.0003, g16=0.12997). None of these measurements is an
asymptotic Lyapunov exponent or proof of a causal failure mechanism.

The primary same-medium Momentum advantage is unchanged. No new stability
parameterization is validated. Read [full interpretation](evidence/dynamics_audit_seed0/INTERPRETATION.md),
[all64 table rows](evidence/dynamics_audit_seed0/RESULTS.md),
[compact metrics](evidence/dynamics_audit_seed0/analysis.json), and
[figure](evidence/dynamics_audit_seed0/dynamics_overview.png) first.
[Validation](evidence/dynamics_audit_seed0/validation.json) and
[hash bindings](DYNAMICS_PUBLICATION_MANIFEST.json) establish provenance.
The16 byte-identical [raw files](evidence/dynamics_audit_seed0/raw/) retain
per-map power iterations, perturbations, source tangents and block diagnostics.
No additional experiment or monitor is scheduled.

## Final: generic recurrence by medium, 2D seed 0

The last authorized arm, Masked State-matched NCA, saved all 800 training updates
and all 15 scientific evaluation points. The original process did not finalize;
only missing size128 timings and initial-state gradient probes were subsequently
recovered from its checkpoint, with no retraining. The 32/T64 and 128/T256 BA,
BCE and H RMS replay exactly. Exit cause is unknown; original records remain
unchanged. [Recovery](evidence/masked_state_seed0/recovery.json) and
[validation](evidence/masked_state_seed0/validation.json) document this distinction.

Frozen primary endpoint: 32x32/T64; one training seed, 16 held-out maps.
BA averages class recall within each map then averages maps; paired correctness
pools changed-component pixels that are correct under both original and flipped
source labels. Neither maps nor pixels are independent training replicates.

| Generic recipe | Whole-grid BA (%) | Masked BA (%) | Whole-grid paired (%) | Masked paired (%) |
|---|---:|---:|---:|---:|
| State-matched NCA | 85.17 | 90.43 | 49.31 | 56.37 |
| Momentum NCA | 84.44 | 96.45 | 42.75 | 77.25 |

| Primary contrast | BA difference (pp) | Paired difference (pp) |
|---|---:|---:|
| Masked Momentum minus Masked State | +6.02 | +20.88 |
| Masking effect, State | +5.26 | +7.06 |
| Masking effect, Momentum | +12.01 | +34.49 |
| Difference-in-differences | +6.75 | +27.43 |

Outcome: `MOMENTUM_JOINT_5PP_ADVANTAGE`, a descriptive threshold specified before
the new arm ran, not statistical significance. The other three cells were already
known when the [final protocol](new/masked_state/PROTOCOL.md) was written.
Within this seed, both generic recipes benefit from the task-respecting medium
on the primary metrics, and Momentum retains an advantage on that same medium.
This neither makes the medium sufficient nor shows that velocity alone causes
the advantage. State uses H32/hidden48/4993 parameters, Momentum H16+V16/hidden88/
4689 parameters: equal persistent scalars, differing widths and 6.48% more
parameters for State. Masking also changes connectivity, degree and spectrum.

The negative secondary findings are substantial. At 32/T64, State masking
improves BA while worsening BCE from 0.2579 to 0.9871. With longer rollout:

| Size | Masked State BA T64 to T256 (%) | Masked Momentum BA T64 to T256 (%) |
|---|---:|---:|
| 32 | 90.43 to 11.38 | 96.45 to 77.68 |
| 64 | 75.93 to 25.93 | 84.75 to 64.15 |
| 128 | 63.93 to 38.59 | 75.20 to 51.90 |

At 32/T256, Masked State paired correctness falls to 9.38%, BCE reaches 63.4853,
and H RMS rises from 5.99 at T64 to 23.87. Its 87.38% training clip fraction
and initial-state task-gradient norm of 9.06e8 at T64 flag sensitivity, but the
single-example loss-gradient probe is not a Jacobian spectral norm or a proof
of asymptotic instability. No sustained 95% endpoint is reached. Conditional
repair cohorts differ (State 5 eligible maps versus Momentum 12 at size32).
Timings span separate runs, including a separate recovery process for size128;
no controlled speedup or conditional-repair superiority is established.

Read [final generated results](evidence/masked_state_seed0/RESULTS.md),
[all 15 comparisons](evidence/masked_state_seed0/comparison.json), then
[publication hashes](STATE_PUBLICATION_MANIFEST.json). The
[original arm](evidence/masked_state_seed0/original_arm.json) is byte-identical
to the saved run; the [augmented arm](evidence/masked_state_seed0/arms/masked_state_nca_seed0.json)
only adds the documented diagnostics. Earlier results below remain intact.
This completes the final authorized comparison. No further experiment is scheduled.

## Previous: same-medium generic momentum control and trajectory audit

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
