# Masked inertial trajectory audit

Frozen before execution. This is an inference-only audit of one seed-0
masked_inertial_rd checkpoint. It performs no training, parameter update,
sweep, or model intervention.

Replay the checkpoint on the evaluator's 16 held-out maps per size, generated
with seeds 30000 + size, at sizes 32, 64, and 128 and horizons 16, 32, 64,
128, and 256. Restore the checkpoint run's PyTorch backend settings. Before
mechanistic interpretation, compare balanced accuracy, balanced BCE, full
state H RMS, and full state V RMS against the stored evaluator JSON. The
maximum absolute difference must be no more than 5e-5; otherwise stop and keep
only the reconstruction report.

For calibration only, generate 32 independent maps per size with seeds
51000 + size. At each size and horizon, fit one positive scalar temperature
on calibration labels by minimizing the evaluator's per-map,
present-class-balanced BCE. Use deterministic derivative bisection (50 steps)
in inverse temperature over [0.001, 100], equivalent to T in [0.01, 1000],
including boundary-optimum checks, and report whether a bound was selected. Do not use
test labels to fit T. Report calibration and held-out BCE before and after
scaling and held-out balanced accuracy. Verify strict threshold invariance.

For each size and horizon, report traversable logit RMS; pooled traversable
signed margins (median, correct median, wrong median, q10, mean of the lowest
ceil(10%) margins, and wrong fraction); and original balanced BCE split into
correct and wrong contributions. Prediction is logit >= 0, so zero is correct
for a positive target and wrong for a negative target.

Report H and V RMS separately on traversable and wall pixels. For each pair
t,2t through (128,256), save per-map traversable H cosine, normalized H
distance norm(H_2t/||H_2t||-H_t/||H_t||), relative state change
norm(H_2t-H_t)/norm(H_t), and traversable V cosine. Undefined
zero-norm comparisons are null.

Using 4-neighbor components derived from the unchanged input mask for
diagnostics only, report component sizes and per-channel H, V, and current
reaction means at every horizon. Also report component-constant projection
energy fractions for H and V, signed component mean Laplacians and their
cancellation, wall pixels separately as singleton modes, and one-step state
and velocity equation residuals. Compare component-mean V at t and 2t as a
finite-horizon stationarity diagnostic. Beta is a per-channel vector and
reaction depends on the full H field; component means are not an autonomous
closed system. Wall singletons are not merged into traversable components.

Run projection-energy identity, relative component-Laplacian cancellation,
and positive-temperature sign-invariance checks. Keep all masks diagnostic
only; never modify model or state. Write compact summary.json, detailed
components.json, reconstruction.json, RESULTS.md, manifest.json, and a
local-only receipt. Record checkpoint and source hashes, data seeds,
configuration, runtime, and GPU model. The local receipt may contain host,
PID, and absolute paths; public summaries must not.

Execution is capped at10 minutes, checked at each rollout step and diagnostic
checkpoint. Test batches stay16 and calibration batches32 in separate rollouts.
Sanity checks cover weighting against the frozen balanced loss, projection
energy decomposition and mean-Laplacian cancellation. The final implementation
is root-owned after a setup-only quoting failure; that failure produced no
scientific result and is not counted as a model failure.

These are finite-horizon observations from one checkpoint. Drift, component
projection, or near directional alignment cannot establish convergence,
asymptotic behavior, architecture benefit, or a universal claim.
