# Candidate/workspace switch audit: completed

Read the [post-execution review notes](../../new/switch_audit/REVIEW_NOTES.md)
for exact replay scope and the quantified factorial contrasts.

Zero training, revision seed0 only, 16 fixed maps per size. All 232 saved
replay comparisons have zero error. Candidate reconstruction also has zero
error; affine readout identity maximum error is below 7.2e-7. Inference
completed in 9.61 seconds.

**The candidate retains the old answer, but the audit does not isolate W
as the sole cause. Single-block resets or mature-state transplants fail to
recover reliable switching; cold reset of both blocks succeeds.**

## Current output versus candidate

Percent correct on the changed component under the new target:

| Size | K | Warm output | Warm candidate | Cold output | Cold candidate |
|---|---:|---:|---:|---:|---:|
| 32 | 0 | 2.00 | 1.90 | 37.50 | 63.31 |
| 32 | 1 | 1.95 | 1.78 | 63.31 | 64.48 |
| 32 | 2 | 1.92 | 1.71 | 64.48 | 65.55 |
| 32 | 4 | 1.78 | 1.59 | 66.85 | 68.82 |
| 32 | 8 | 1.46 | 1.30 | 70.97 | 72.09 |
| 32 | 16 | 0.86 | 0.68 | 79.70 | 83.08 |
| 32 | 32 | 0.00 | 0.00 | 94.34 | 95.43 |
| 32 | 64 | 0.00 | 0.00 | 100.00 | 100.00 |
| 32 | 128 | 0.00 | 0.00 | 100.00 | 100.00 |
| 64 | 0 | 7.26 | 6.72 | 43.75 | 56.39 |
| 64 | 1 | 7.03 | 6.45 | 56.39 | 56.59 |
| 64 | 2 | 6.80 | 6.25 | 56.59 | 56.75 |
| 64 | 4 | 6.36 | 5.83 | 56.97 | 57.41 |
| 64 | 8 | 5.55 | 5.08 | 58.03 | 58.75 |
| 64 | 16 | 4.26 | 4.05 | 61.47 | 62.30 |
| 64 | 32 | 2.89 | 2.76 | 68.46 | 69.82 |
| 64 | 64 | 2.13 | 2.09 | 87.39 | 88.63 |
| 64 | 128 | 1.64 | 1.63 | 99.49 | 99.62 |

Q_K is computed after the W update and drives Z_(K+1). O(Q_K) is
the existing affine readout applied to the candidate. Persistent wrong
candidate readout is directly observed; slow exponential averaging alone
does not explain warm failure. This says nothing about a unique origin
inside the coupled W/Z recurrence.

## Full-rollout interventions

All branches use the new input. Accuracy columns are percentages at K128.

| Size | Intervention | Changed | Unchanged | BA | Predicted positive on changed | Recovery criterion |
|---|---|---:|---:|---:|---:|---|
| 32 | warm | 0.00 | 100.00 | 75.54 | 37.50 | False |
| 32 | reset_W | 37.50 | 100.00 | 84.33 | 0.00 | False |
| 32 | reset_Z | 37.50 | 77.57 | 66.38 | 0.00 | False |
| 32 | cold | 100.00 | 100.00 | 100.00 | 62.50 | True |
| 32 | transplant_W | 37.50 | 100.00 | 84.33 | 0.00 | False |
| 32 | transplant_Z | 37.50 | 100.00 | 84.33 | 0.00 | False |
| 64 | warm | 1.64 | 100.00 | 69.03 | 45.39 | False |
| 64 | reset_W | 43.36 | 100.00 | 83.41 | 1.12 | False |
| 64 | reset_Z | 37.17 | 61.70 | 55.38 | 9.84 | False |
| 64 | cold | 99.49 | 100.00 | 99.55 | 56.76 | True |
| 64 | transplant_W | 45.39 | 100.00 | 85.26 | 1.64 | False |
| 64 | transplant_Z | 45.40 | 100.00 | 85.27 | 1.65 | False |

`reset_W` uses encoder(new input); `reset_Z` uses zero. `cold` resets
both. `transplant_W/Z` uses one block from a separate 64-step new-input
rollout, retaining the other old block. Donors contain extra computation.

**At size32 the 37.5% result is not partial retrieval of the new source:**
all four single-block interventions predict negative on every pixel of
every changed component at K128. Six of the 16 new component labels are
negative. This label-bias observation is a post-hoc derivation from saved
per-map accuracy and the fixed labels, requiring no extra model rollout.

In the Z-donor transplant, size32 output starts 100% correct on the changed
component and falls to 37.5%; injecting a correct task-state block does not
sustain the new answer in the old workspace. Conversely a new W donor with
old Z also fails. Resetting Z alone also damages unchanged accuracy.

## Candidate-only W/Z factorial

All queries below use the same new X at K64. W donors are post-F, Z donors
are pre-update. New donors come from an intact fresh-new rollout at T128.

| Size | W donor | Z donor | Candidate changed accuracy (%) | New-target signed margin |
|---|---|---|---:|---:|
| 32 | warm | warm | 0.00 | -1.8852 |
| 32 | warm | new_aged | 37.50 | -0.3386 |
| 32 | new_aged | warm | 37.50 | -0.2182 |
| 32 | new_aged | new_aged | 100.00 | 2.3580 |
| 64 | warm | warm | 2.09 | -1.9138 |
| 64 | warm | new_aged | 45.51 | -0.1033 |
| 64 | new_aged | warm | 45.27 | -0.0434 |
| 64 | new_aged | new_aged | 99.62 | 2.2218 |

Both W and Z substitutions change the fixed candidate function; their
joint effect includes an interaction. Mixing blocks from incompatible
trajectories can be off-distribution. Neither these contrasts nor reset
failures establish a unique natural causal mechanism or prove that both
blocks must always be reset. The all-new donor combination is an intact
computed solution, not an equal-cost practical remedy.

Direct old/new X swaps inside Qnet barely change the shown aggregate
responses. This local query excludes X's paths through F, initialization
and repeated updates, and changes a single source location. It does not
show that the network generally ignores external evidence.

## What the feedback gets right and what remains unproven

- Confirmed: cold source solving versus warm source revision are distinct;
  the candidate itself remains old-aligned at K64/K128 in both sizes.
- Refined: the W-only stale-workspace explanation is insufficiently
  isolated. Z feedback and W/Z compatibility matter in the tested interventions.
- Not established: useful internal computation merely from continued W
  motion, a unique stale-state mechanism, general persistent-memory success,
  or conditional invalidation as the necessary next architectural primitive.
- The earlier repair contrast retains more state under Z-only damage; it
  did not by itself prove that repair and reopen failure share one cause.
- At the original zero-output-layer initialization the state Jacobian is
  block diagonal (I,0.5I). That is a code-level fact, not a diagnosis of
  seed1 failure or a measurement of parameter credit and later dynamics.
- No short-versus-long BPTT experiment, new training, seed1 audit or
  generalization across independently trained models was performed.

The prior paired screen's NO_JOINT_SCREEN_PASS and -9.02 pp mean hold effect
remain unchanged. This is a diagnostic result for one existing checkpoint.

See [compact analysis](analysis.json), [figure](overview.png),
[provenance](provenance.json) and the frozen protocol in the source tree.
