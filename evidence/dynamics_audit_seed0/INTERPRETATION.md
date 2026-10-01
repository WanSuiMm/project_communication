# Completed generic NCA dynamics audit: interpretation

The audit completed at 16:28:11 +08:00 on 2026-10-01 in112.859 seconds. No
training or checkpoint modification occurred. Four recipes, two sizes, eight
anchors and three window lengths were evaluated. Independent post-hoc checks
recomputed40 historical comparisons: BA, BCE, H/V RMS and paired correctness
all replay exactly. Source snapshots and all four checkpoint hashes match.
The same-batch unperturbed window replay error is also zero.

This review uses only saved output. Its computation is CPU-only summarization,
not a new model run. See [analysis.json](analysis.json), [provenance.json](provenance.json),
and [overview figure](dynamics_overview.png). All16 detailed [curve and dynamics files](raw/) are byte-identical to the original run.
The redundant combined local summary is excluded; its hash is retained in provenance.json.
Read this interpretation and analysis.json before opening the detailed files.

## 1. The proposed maximum-gain criticality crossing is not supported

All256 sampled open-input/output K16 log-gain estimates are positive; minimum
0.0827. They are already large before the accuracy peak and generally decline
through later performance deterioration. This concerns the stated raw Euclidean
full-state maximum-gain diagnostic. It does not exclude near-neutral task-specific
directions or instability in longer products, other metrics or unsampled windows.

Masked Momentum at size32:

| T | BA %, all16 maps | BCE | Paired % | Median g16, maps0..3 | Converged /4 |
|---|---:|---:|---:|---:|---:|
| 32 | 91.47 | 0.1598 | 49.82 | 0.70763 | 3 |
| 64 | 96.45 | 0.0512 | 77.25 | 0.27742 | 3 |
| 128 | 86.17 | 0.9152 | 53.28 | 0.16949 | 2 |
| 256 | 77.68 | 6.8694 | 42.78 | 0.17582 | 4 |

Here g16=log(sigma_estimate)/16 for the forward window starting at T, ending
at T+16. These are finite-window gains, not asymptotic Lyapunov exponents.
The same four diagnostic maps also decline: their BA is97.14%,86.40%,77.87%
at T64,T128,T256. Thus this observation is not solely a mismatch between the
16-map performance average and the four-map diagnostic cohort.

Masked State is an even clearer counterexample to a simple rising-local-gain
explanation: at size32, BA falls90.43% to11.38% from T64 toT256 while median
g16 decreases0.14666 to0.10359. This does not establish improving stability:
the gains remain positive and only short windows were measured.

## 2. Persistent motion and growing wrong confidence are directly observed

For Masked Momentum size32, mean per-map open H RMS increases42.57 to112.25
to258.87 at T64,T128,T256; open V RMS is1.071,1.151,1.168. Continuing updates
do not decay toward zero over the measured interval. Error confidence grows:
BCE increases0.0512 to6.8694. The median across maps with errors of each map's
wrong-margin median changes from -0.178 to -32.608. The error cohorts differ
between times, so this is not a tracked-same-error statistic.

Source influence does not simply vanish: the median across16 maps of the
changed-component binary source-flip logit RMS increases25.26 to63.75 from
T64 toT256, while paired correctness falls77.25% to42.78%. This shows that a
nonzero, growing response to source identity can coexist with worsening
correctness. It does not prove that information remains recoverable by another
readout, or isolate the readout from the rest of the learned recurrence.

The source tangent often disagrees with the finite binary intervention. It is
a local derivative along a fixed branch, with initial and repeated-drive paths
included; its sign or magnitude cannot replace finite counterfactual correctness.

## 3. Generic Momentum does not inherit the old component-mean explanation

At size32, Masked Momentum's mean per-map component-constant H energy fraction
falls1.79% to0.20% between T64 andT256. The corresponding constant fraction of
the next H update falls0.30% to0.09%. Most state/update energy therefore lies
in variations within components. Constant components can still grow in absolute
size; these fractions do not prove they are irrelevant.

This differs from the earlier explicit Masked Inertial RD audit. Its dominant
component-constant-energy interpretation must not be transferred to generic
Momentum. The new fractions average per-map fractions; the older audit reported
pooled energies, so their exact numeric values should not be equated.

## 4. Spectral and nonlinear diagnostic limits are material

- Only562/1536 product estimates satisfy the frozen convergence criteria.
  For the main open K16 subset,148/256 satisfy them. Unconverged values remain
  estimates from tested directions, not certified maxima or upper bounds.
- At relative RMS perturbation1e-4, worst-direction linearization error has
  median46.44%; only27/256 cases are below10%. Random-direction median error
  is1.19%, with193/256 below10%. At1e-3 the worst-direction median is77.33%.
  Thus the very large infinitesimal gains often do not quantitatively predict
  the tested finite perturbations. No smaller-epsilon rescue run was added.
- Masked Momentum's free-momentum K16 baseline already has sigma8.0003 and
  g16=0.12997. Positive short-window gain alone cannot establish asymptotic
  instability. At size32/T256 the HH block norm estimate is approximately1,
  VH approximately0.050, and exact HV/VV norm0.911; these local estimates
  coexist with continued state motion and decision deterioration.
- One training seed and four diagnostic maps do not establish a population
  mechanism. This audit does not isolate velocity, mask-induced spectral change,
  or finite-time amplification as the cause of the primary interaction.

## Decision

The primary same-medium Momentum advantage remains intact. The clean story
"performance peaks near zero maximal finite-time exponent, then fails after
the exponent crosses positive" is unsupported by the sampled diagnostics.
Persistent updates, growing state/logit magnitudes and increasingly incorrect
decisions are stronger direct observations. They motivate explaining why
continued computation damages answers, without yet selecting an antisymmetric,
damping or near-isometric recurrence as the remedy. No further experiment,
training, architecture change or monitor was started for this review.
