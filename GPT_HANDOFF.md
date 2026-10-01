# Incremental review: final masked-state control after 82635bb

- Review base: `82635bbf0063d11359fd2d0b6c6571616a01f89b`.
- Evidence head: `48497e147a9ee81dcb6027d51e103fe5c0156649`.
- The subsequent handoff-only commit changes review metadata only.

Only one new model was trained: generic Masked State-matched NCA. It completes
the exploratory state/Momentum by whole-grid/masked-medium 2x2. The other three
cells were already known when the final protocol was fixed. This is the final
authorized experiment; no further training, architecture change or monitor is scheduled.

## Minimal reading order

1. [Current results](RESULTS.md), then [final control results](evidence/masked_state_seed0/RESULTS.md)
   and [all 15 comparisons](evidence/masked_state_seed0/comparison.json).
2. [Frozen final protocol](new/masked_state/PROTOCOL.md),
   [MaskedState.step](new/masked_state/state_cells.py), and
   [runner and contrasts](new/masked_state/run_state.py).
3. [Recovery provenance](evidence/masked_state_seed0/recovery.json),
   [recovery implementation](new/masked_state/recover_tail.py),
   [validation](evidence/masked_state_seed0/validation.json), and
   [publication hashes](STATE_PUBLICATION_MANIFEST.json).

Do not begin with the original/augmented raw arm JSON or the previous audit's
components.json. The original arm is an exact saved snapshot; the augmented
arm adds only the explicitly listed diagnostics. Prior evidence is unchanged.

## New decision-relevant evidence

At the fixed 32/T64 endpoint:

| Recipe | Whole-grid BA / paired (%) | Masked BA / paired (%) |
|---|---:|---:|
| State-matched NCA | 85.17 / 49.31 | 90.43 / 56.37 |
| Momentum NCA | 84.44 / 42.75 | 96.45 / 77.25 |

Masked Momentum exceeds Masked State by 6.02pp BA and 20.88pp paired correctness,
meeting the descriptive joint 5pp threshold. Masking gains are 5.26pp/7.06pp
for State versus 12.01pp/34.49pp for Momentum; interactions are +6.75pp/+27.43pp.
Thus medium sensitivity is present in both generic recipes at this endpoint,
and the same-medium Momentum recipe retains an advantage in this seed.

Do not hide the failures: State masking worsens primary BCE from 0.2579 to
0.9871. At 32/T256 Masked State falls to 11.38% BA, paired 9.38%, BCE 63.4853;
its H RMS grows from 5.99 to 23.87. Masked Momentum also declines to 77.68% BA.
Neither has a sustained 95% endpoint. The large State initial-state gradient
probe (9.06e8 at T64) is a single-example loss-gradient measurement, not a
Jacobian spectral norm or proof of an asymptotic dynamical mechanism.

## Execution and claim boundaries

The original process saved 800 updates and all 15 scientific endpoints, then
left no final completion marker. It was no longer present when checked; its
exit cause is unknown. Original outputs and status remain unchanged. Missing
size128 timings and initial-state gradient probes were recovered from the same
checkpoint without retraining. BA/BCE/H RMS replay exactly at 32/T64 and
128/T256. All pre-existing fields are unchanged in the augmented record;
all 15 comparisons were recomputed independently from raw evidence.
This publication does not assert normal completion of the original process.
Timings come from different sessions and do not establish a speedup.

State has H32/hidden48/4993 parameters versus Momentum H16+V16/hidden88/4689:
same persistent scalar capacity, different content/readout/program widths.
The result does not isolate velocity or establish its universal necessity.
Masking changes connectivity, degree and spectrum using a supplied task mask.
One training seed remains the independent replicate; no population-level
significance, conditional-repair advantage, stable attractor or 3D claim.

The earlier explicit Inertial RD primary loss to generic Momentum and its
finite-horizon amplification/mean-drift audit are unchanged. No detailed
trajectory audit of the generic State or Momentum models has been performed.

## Reviewer questions

1. Do the pairing, source hashes and preserved scientific fields support the
   final descriptive comparison despite incomplete original process finalization?
2. Are both primary masking gains and the same-medium Momentum advantage
   reported without claiming an isolated velocity mechanism?
3. Do BCE deterioration and long-rollout failure rule out any stronger
   interpretation suggested by the primary accuracy table?

---

# Historical handoff: same-medium Momentum and audit after c13b73f

- Review base: `c13b73f7deadf6c6e0611659df5fa8d118af618c`.
- Evidence head: `d3038c3a1a462491a7715a4d5573520572f6ad5e`.
- The subsequent handoff-only commit changes review metadata only.

Two completed additions: one generic Masked Momentum training arm and an
inference-only trajectory audit of the existing Masked Inertial checkpoint.
Historical sources, results, checkpoints and protocols remain unchanged.

## Minimal reading order

1. [Current results](RESULTS.md), then
   [control results](evidence/masked_momentum_seed0/RESULTS.md) and
   [all15 paired horizons](evidence/masked_momentum_seed0/comparison.json).
2. [Audit interpretation](evidence/trajectory_audit_seed0/INTERPRETATION.md),
   [audit table](evidence/trajectory_audit_seed0/RESULTS.md), and
   [compact audit summary](evidence/trajectory_audit_seed0/summary.json).
3. [Control protocol](new/masked_momentum/PROTOCOL.md),
   [MaskedMomentum.step](new/masked_momentum/momentum_cells.py),
   [audit protocol](new/trajectory_audit/PROTOCOL.md), and
   [audit implementation](new/trajectory_audit/audit.py).
4. [Publication manifest](MOMENTUM_AUDIT_PUBLICATION_MANIFEST.json) and
   [validation](evidence/masked_momentum_seed0/validation.json).

Do not begin with the6.2MB components.json or raw arm JSON. They are secondary
sources for component-vector, repair, revision and distance questions.

## New decision-relevant evidence

Masked Momentum completes800 updates and all evaluations with matched original
generic initialization/RNG and logged schedules matching both references.
At the frozen32/T64 endpoint it reaches96.45% BA,77.25% paired correctness,
BCE0.0512 versus explicit masked inertial88.79%,37.22%,0.2503. Generic wins
by7.66pp/40.02pp, satisfying the reverse descriptive joint5pp criterion.
Masking also improves generic Momentum versus its own old control by12.01pp
BA and34.49pp paired correctness. A benefit from the medium is not unique to
the explicit decomposition.

The BA ranking reverses at32/T256: Momentum77.68%, explicit91.55%. However,
paired correctness is42.78% versus41.92%, and neither reaches sustained95%.
Momentum BCE rises to6.8694 and H RMS to219.81. Source revision remains weak.
Keep this horizon/metric tradeoff visible without replacing the primary endpoint.

The candidate audit exactly reproduces all15 historical evaluation checkpoints.
At32/T64→T256, open H RMS14.48→120.61, wrong-margin median -1.889→-18.421,
and component-constant energy89.10%→97.04%. Independent32-map calibration
gives held-out BCE0.2569→0.2492, versus raw0.2503→1.4519; BA is unchanged.
This supports amplification and component-mean drift over the tested horizon.
At128/T256, constant-mode energy is99.51% while BA remains50%; temperature
cannot fix those wrong decisions. Direction/velocity convergence is unproven.

## Claim changes and limits

The explicit candidate lacks a primary advantage over the new same-medium
generic control in this seed. Earlier within-candidate masking gains still hold.
Reject the interpretation that BA improvement demonstrates a stable attractor;
conditional decision recovery alone also does not establish attractor recovery.
The detailed audit applies only to explicit Masked Inertial RD. Generic Momentum
has existing evaluator norms/BCE, but no new component or temperature audit.

One training seed,16 held-out maps per size, differing hidden widths and initial
transport, and different conditional-repair cohorts limit attribution. No
population-level superiority, isolated factorization mechanism, controlled
speedup, general PDE rejection or3D claim. No new run or monitor is scheduled.

## Reviewer questions

1. Does any pairing or implementation issue invalidate the primary generic-control win?
2. How should the late-horizon BA/paired-correctness tradeoff constrain the next claim?
3. Does the audit justify finite-horizon amplification/mean drift, while withholding
   constant-force limits and fixed-point/projective convergence claims?

---

# Historical handoff: masked medium after 48095e5

- Review base: `48095e5b7fae82ec65d1fd1ce0b4e54726fe7df2`.
- Evidence head: `35bfd1e0a2fac7e38183cf6e0c7a03d8b95c41b0`.
- The subsequent handoff-only commit changes review metadata only.

Only the two structured cells receive an input-mask-respecting transport
operator. Same seed, initialization/RNG and logged rollout/damage schedule;
800 updates per arm. No original sources or frozen evidence were changed.

## Minimal reading order

1. [Masked results](evidence/masked_medium_seed0/RESULTS.md) and
   [paired comparison](evidence/masked_medium_seed0/paired_comparison.json).
2. [Frozen protocol](new/masked_medium/PROTOCOL.md).
3. [masked_laplacian / MaskedCell.step](new/masked_medium/masked_cells.py),
   [runner reuse and pairing](new/masked_medium/run_masked.py), and
   [four implementation tests](new/masked_medium/test_masked.py).
4. [Publication manifest](MASKED_PUBLICATION_MANIFEST.json) binds sources and
   byte-identical public results. Open detailed arm JSON only for a specific
   repair, revision, gradient or distance question.

## Changed and unchanged claims

At the prespecified 32x32/T64 endpoint, inertial RD gains +18.85pp BA and
+19.42pp paired correctness, passing the descriptive joint +5pp criterion.
RD loses 8.06pp and 20.68pp. Masked inertial BA at T256 is 91.55%, 75.69%,
50.00% for sizes 32, 64, 128. No sustained aggregate 95% threshold is reached.
This is a positive within-recipe operator intervention for the inertial arm,
with an opposite RD result. Connectivity, degree and spectrum all change;
wall leakage is not isolated as the sole mechanism. No multi-seed significance,
same-medium generic momentum superiority, repair success or speedup is claimed.
The original inertial negative screen and historical A0 claims are unchanged.
No new experiments or monitor are scheduled.

## Reviewer questions

1. Does the implementation preserve the intended single operator intervention
   and valid pairing with the frozen controls?
2. What mechanisms are consistent with the opposite RD and inertial effects,
   without inferring a mechanism from this single seed?
3. Given low paired correctness, failure at 128x128 and no masked momentum arm,
   what is the smallest defensible next comparison?

---

# Historical handoff: inertial NCA after eeb6514

- Review base: `eeb6514b35f4e860ffbe69bdecb627d5ea172b7e`.
- Evidence head: `e17bb5b04861996389fa263dc0d7041ae14cc4ff`.
- The later handoff-only commit does not change code, results or claims.

This update adds a separate 2D seeded-region experiment: explicit reaction and
five-point transport with velocity state, compared with generic momentum,
state-matched NCA and first-order RD. It does not rerun A0 or change its task.
All four arms completed 800 updates with seed 0, and all 32/64/128 evaluations.

## New evidence and claim boundary

At 32x32/T64, candidate BA is 69.94%, versus 84.44% for generic momentum and
85.17% for state-matched NCA. Candidate BA drops to 53.50% at 32x32/T256 and
50% at both larger sizes/T256, where paired-source correctness is zero.
No arm reaches sustained aggregate 95% BA. Candidate repair is unevaluable
because no pre-damage map qualifies. This is a negative single-seed screen,
not a family-wide impossibility result or a new preregistered binary gate.

Training sources and the two theory/protocol documents match the executed
snapshot byte-for-byte. Four published per-arm JSON files are exact copies of
the frozen run. Code/protocol/evidence hashes are bound by
[INERTIAL_PUBLICATION_MANIFEST.json](INERTIAL_PUBLICATION_MANIFEST.json).
Private receipts, checkpoints, duplicate ZIP material and logs remain local.
Earlier A0/v1 code, evidence and claims are unchanged. No new training is scheduled.

## Minimum reading order

1. [New result table](evidence/inertial_seed0/RESULTS.md), then
   [summary.json](evidence/inertial_seed0/summary.json) for all rollout horizons.
2. [Protocol](new/nca_inertial_wind_tunnel/WIND_TUNNEL.md) and
   [integration notes](new/nca_inertial_wind_tunnel/LOCAL_INTEGRATION.md): metric fixes and sampler conventions.
3. [cells.py](new/nca_inertial_wind_tunnel/cells.py), especially `Cell.step`,
   then [tasks.py](new/nca_inertial_wind_tunnel/tasks.py) if questioning geometry or targets.
4. Use individual JSON files under `evidence/inertial_seed0/arms/` only for
   specific distance, repair, revision or timing questions. Do not begin with
   all raw files or unchanged historical evidence.

## Reviewer questions

1. Does any implementation or measurement issue invalidate this scoped negative result?
2. What does the candidate's within-training-horizon deficit establish separately
   from its longer-rollout degradation, without causally attributing either to inertia alone?
3. Given zero repair eligibility and no solved reference arm at the 95% criterion,
   which claims remain unevaluable? Any suggested follow-up should be a separate
   bounded diagnostic, not a reinterpretation of the frozen result.

---

# Historical handoff: A0 after c669e18

- Review base: `c669e18cc6f1747ff99480c1e7a57ebc2866e956`.
- Evidence head: `74d77256daadfcd2bf90741b63fa6ba62a8eb65b`.
- The subsequent commit adding this handoff and its README link changes review
  metadata only. Use the evidence head above when comparing code and results.

## What changed

The registered A0 experiment completed its conditional schedule: 24/24 trials,
with 16 possible 3D RT trials skipped after failed positive-control calibration.
Forty was an upper bound, not the number still required after that failure.

A0 uses four seeds and 480 updates, compared with v1's two seeds/240 updates.
Every A0 arm emits q=c*v. Attention learns query/key and uses the same q as
values, with no separate value/output projection. The four RT arms cross
constant/learned medium with raw/normalized reception. Paired arms share initial
parameters, unit initial conductance and packed numerator/confidence solver
work; only receiver division differs within a pair. Learned conductance is
2*sigmoid of symmetric local features, with a zero-initialized final head.

Numerical/oracle checks cover the added grouped solve, normalization gradients,
attention values and paired initialization. Source/protocol hashes and all
completed trial metrics are included in the new public evidence directory.

## Decision-relevant new evidence

- 2D attention: four of four seeds are perfect in every reported condition.
- 2D RT: all four arms fail training-scale fitting in every seed. This is a
  negative qualification under a reliable control, not just the v1 control issue.
- Normalization's mean d128 effect is +0.977 percentage points for constant
  medium and +0.586 for learned medium, while both remain near chance. Effects
  change sign across conditions; no significance or equivalence claim is made.
- 3D: attention seed1729 fails (47.92% train d16); no RT arms or normalization
  treatment estimate exist for that dimension.
- Large-tau oracle emission remains decodable through d128; tiny-tau d128
  underflows. Oracle recoverability does not establish learned-model success.
- Sampled normalized-model final-update probes retain moderate background
  confidence and do not show denominators below epsilon. This observation does
  not causally identify why training fails or quantify total information flow.

## Unchanged claims and evidence

The original v1 data, model, runner and solver sources remain unchanged, as
verified by the original manifest. Its negative/inconclusive evidence is kept.
No B/C, width sweep, natural-vision, general-3D or optimized-kernel experiment
was added. Fixed axis splitting and narrow straight geometry remain limitations.
A0 raw is confidence-gated, so it is not identical to v1 raw. Multiple changes
between protocols prevent isolating the cause of improved attention calibration.

## Minimum reading order

1. [RESULTS.md](RESULTS.md), current A0 sections only.
2. [A0_PROTOCOL.md](A0_PROTOCOL.md), intervention and stopping rules.
3. [a0_models.py](a0_models.py), `transport_pair`, `SharedValueAttention`, `A0Model`.
4. [summary.json](evidence/a0_v2_1/summary.json), per-seed/paired metrics and probe summaries.
5. Use [aggregate.json](evidence/a0_v2_1/aggregate.json) only for a specific
   axis/history question; [checks.json](evidence/a0_v2_1/checks.json) for numerical
   validation. [A0_PUBLICATION_MANIFEST.json](A0_PUBLICATION_MANIFEST.json) binds
   these to the executed source. Do not begin by rereading large raw evidence.

## Concrete reviewer questions

1. Does any code or endpoint contradict the scoped 2D NOT_QUALIFIED_FIT verdict?
2. What one bounded intervention would distinguish emission learning from
   reaction/readout optimization, without treating oracle success as learned evidence?
3. Are confidence-mass diagnostics being interpreted at their actual final-
   update/group/sample scope, including the possibility of messages re-emitted
   elsewhere after earlier recurrent steps?

The current configuration is not being escalated. This handoff requests review,
not additional runs or a reinterpretation of the frozen negative result.
