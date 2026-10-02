# Incremental review: Direct Spatial Carry negative development screen

- Review base: `0decc2926b52967496d57a90f8f803383e383178`.
- Evidence head: `592762476340a26c01a29f16b8f7b0bb349f5e3b`.
- This subsequent handoff-only commit changes review metadata only.

All8 new matched K8 runs completed300 updates in797.797 seconds. The sole
intervention replaces the W identity path with fixed lazy averaging over the
same masked graph: W-0.5*D_M^dagger*L_M(W). F still receives old features;
Q receives updated W. Additive Z,5033 parameters, initialization, training
bank/schedule, loss/optimizer clocks and evaluator remain matched.

**DEVELOPMENT_NO_GO**: baseline reaches AND holds in2/4 seeds; carry in0/4.
The baseline reproduces all four historical final parameter hashes and whole
evaluation payloads exactly. This is an interpretable negative development
result, not a baseline-reproduction failure. Seeds2-5 and these maps were
already inspected; no fresh confirmation or new training is authorized here.

## Minimal reading order

1. [Current result and boundary](RESULTS.md) and
   [full carry report](evidence/direct_spatial_carry_init2345/RESULTS.md).
2. [Compact metrics](evidence/direct_spatial_carry_init2345/analysis.json),
   [curves](evidence/direct_spatial_carry_init2345/curves.csv), and
   [paired contrasts](evidence/direct_spatial_carry_init2345/paired_effects.csv).
3. [Frozen protocol](new/direct_spatial_carry/PROTOCOL.md),
   [cell](new/direct_spatial_carry/carry_cells.py),
   [runner and decisions](new/direct_spatial_carry/run.py), and
   [CPU checks](new/direct_spatial_carry/check.py).
4. [Independent arithmetic audit](new/direct_spatial_carry/analyze.py),
   [validation](evidence/direct_spatial_carry_init2345/validation.json), and
   [source/evidence bindings](DIRECT_CARRY_PUBLICATION_MANIFEST.json).

Read the8 raw arm JSONs only for detailed follow-up; summaries above are the
entry points. Raw records, schedule, aggregate and CSV are byte-identical.
Public manifest/completion/preflight copies omit PID metadata; private
originals, checkpoints, launch receipts and transient logs remain local.

## Decision-relevant delta

Primary size32/T64 STRICT16<d<32, paired mean AND pooled>=80%, both BAs>=85%.
Hold limits declines at BOTH T128 and T256. Continuation additionally needs
>=3 carry reach+hold seeds, retention of2/5, improvement over concurrent
baseline count, and baseline2/5 reproduction. None of the carry seeds reaches.

| Seed | Baseline primary mean / pooled % | Carry primary mean / pooled % |
|---|---:|---:|
| 2 | 93.70 / 92.97 | 0.00 / 0.00 |
| 3 | 21.85 / 17.27 | 2.87 / 3.12 |
| 4 | 52.40 / 44.76 | 14.39 / 11.41 |
| 5 | 96.45 / 95.68 | 7.59 / 4.42 |

Both primary statistics decline in every seed. Carry loses historical
positives2/5 and does not rescue3/4. At size64/T256, carry d>32 paired
correctness is0% in all four seeds. Do not mistake carry seed2 hold=True for
successful retention: its primary score was already zero.

An important secondary exception is seed4 atT128: pooled narrow accuracy is
74.64% versus baseline43.59% at size32, and82.54% versus37.06% at size64.
The corresponding carry values fall to19.50% and26.25% atT256, below baseline;
size32 original BA drops4.88pp fromT64, failing hold. This is transient narrow
performance, not a pass or robust far propagation. Preserve this exception
instead of saying carry is worse at every distance and time.

The fixed operator preserves channel coordinates and is maximum-norm
nonexpansive by itself. Replacing half of local retention with spatial mixing
does not guarantee semantic preservation or whole-cell stability. This result
rejects the tested rho0.5 average-carry recipe under the frozen protocol.
It does not isolate smoothing, dilution or any other mechanism, nor refute
directional carry, other spatial operators, short BPTT or NCA in general.

Peak allocated memory is95.12MiB baseline and96.08MiB carry; the small memory
overhead does not compensate for failed efficacy. CPU verification covers29
source snapshots,8 checkpoints,96 BA aggregates,672 paired aggregates with
reconstructed denominators,8 arm decisions,312 effects and936 CSV rows.
It verifies saved-result arithmetic/provenance, without regenerating logits.
No extra training or checkpoint inference was performed for publication.

Unchanged: all Phase-II/Phase-I qualification outcomes, original positive K8
cases, previous revision/source-switch/dynamics conclusions and earlier
transport failures. No automatic confirmation, architecture rescue or monitor.

## Questions for review

1. Does the code implement precisely the stated identity-to-lazy-average
   substitution, with real-edge degrees and unchanged F/Q inputs and clocks?
2. Are baseline exact reproduction, failure of BOTH primary statistics,
   seed2 zero-score hold and seed4 transient T128 gains all represented fairly?
3. Are the conclusion and stopping decision restricted to this frozen
   development recipe, without asserting a causal failure mechanism?

---

# Historical handoff: short-BPTT Phase-II initialization replication

- Review base: `7d1e32697144add1c063d4157bbd1ca62338a45f`.
- Evidence head: `73542a1a1e123e6f27aaad8c6991e14208901e15`.
- This subsequent handoff-only commit changes review metadata only.

All12 new runs completed300 updates in1095.047 seconds. Existing additive cell,
K8/K16/K64, initialization seeds2/3/4/5; one fixed training bank and shared
batch schedule. No cell modification or state pool. Losses remain every8 steps,
forward trajectories64, one optimizer update afterward. K16 accumulates two
losses before each backward; both W/Z states are detached only at its boundary.

## Minimal reading order

1. [Result and limitations](evidence/short_bptt_phase2_init2345/RESULTS.md).
2. [Compact metrics](evidence/short_bptt_phase2_init2345/analysis.json) and
   [all72 curve rows](evidence/short_bptt_phase2_init2345/curves.csv).
3. [Frozen protocol](new/short_bptt_phase2/PROTOCOL.md),
   [trainer](new/short_bptt_phase2/training.py),
   [runner/gates](new/short_bptt_phase2/run.py), and
   [CPU checks](new/short_bptt_phase2/check.py).
4. [Independent aggregate verification](new/short_bptt_phase2/analyze.py),
   [validation](evidence/short_bptt_phase2_init2345/validation.json), and
   [publication hashes](BPTT_PHASE2_PUBLICATION_MANIFEST.json).

Read raw arm JSON only for detailed follow-up. All12 arm records and the shared
schedule are byte-identical copies. Public analysis/status/manifest/preflight
copies omit PID metadata; frozen originals remain local. Checkpoints and machine
receipts are excluded. No additional training or checkpoint inference for publication.

## Decision-relevant delta

| K | Narrow reach | Narrow reach and hold |
|---|---:|---:|
| 8 | 2/4 | 2/4 |
| 16 | 3/4 | 1/4 |
| 64 | 0/4 | 0/4 |

**K8 replication is NOT_REPLICATED** for both reach and sustained reach, because
the frozen threshold is>=3/4 seeds. **Comparison is BASELINE_UNQUALIFIED**:
none of the full controls qualify. These are separate scientific decisions.

Primary now uses STRICT16<d<32 on new32-map evaluation banks, requiring>=80%
both per-map and pooled paired accuracy, and both original/flipped BA>=85%.
The architecture and band were selected from Phase I and frozen before this
follow-up. This is a narrower prospective replication, not a replacement for
the old d>16 gate. Do not combine the phases into one success rate: their data,
random-factor controls and endpoints differ. n=4 is conditional initialization
replication with one shared training bank/schedule, not dataset robustness.

K8 seeds2/5 provide two positive cases: primary pooled92.97/95.68% atT64 and
93.21/93.59% atT256. On size64/T256 they retain89.65/97.25% in that band, but
d>32 pooled accuracy is36.78/14.83%. K8 seeds3/4 score17.27/44.76% at the primary.
K8 is worse than K64 on seed3. There is no general superiority or optimal-K claim.
K16's primary band fits inside its single-window radius32;3/4 reach is not
cross-window evidence, and only1/4 also holds. Size32 has no changed-component
pixels at distance>=64; corresponding nulls are not zero accuracy.

All four K8 clipping fractions are<=6%, including the two failures. K16 seed2
clips60% and passes reach+hold. Low clipping is not sufficient and high clipping
is not invariably failure; the causal explanation remains open. Peak allocated
memory K8/K16/K64 is95.12/155.81/519.41MiB with84.28-92.86s training per arm.
No accuracy-matched systems claim or activation-only memory interpretation.

Verification covers18 executed source snapshots,12 checkpoint parameter sets,
initialization/data/schedule hashes,144 BA and1008 paired aggregates with
reconstructed denominators, and all12 decisions. CPU K16 gradient-reference and
K8/K64 equivalence checks pass. Frozen historical code/evidence remain unchanged.

Unchanged: Phase-I unqualified comparisons, previous revision joint-gate failure,
source-switch audit and earlier transport/dynamics results. This screen does not
test warm source changes, optimizer-crossing state lifetime, a RelationFirst
representation, arbitrary delayed credit assignment or a new NCA primitive.
No further experiment or monitor is scheduled.

## Questions for review

1. Are common supervision and optimizer clocks preserved forK16, including
   gradient accumulation and detachment of both states?
2. Are the selected narrow endpoint, two aggregation weights, fixed-data scope
   and separate full-control qualification kept explicit?
3. Do successful K8 cases support the stated limited propagation claim without
   hiding failed seeds, long-distance weakness or K16 hold failures?
4. Are the empty distance bins and clipping observations interpreted correctly?

---

# Historical handoff: matched short-BPTT Phase-I screen

- Review base: `c2d705fcb56fdc2b31c68a33ecc7a02610b3fdee`.
- Evidence head: `7af8b2d8df225c9db713cf00d931ba673d00ad01`.
- This subsequent handoff-only commit changes review metadata only.

Eight new training runs compare K8 and K64 gradients in the existing additive
and revision cells, paired seeds0/1. All completed300 updates in730.797 seconds.
Forward length64, eight equally weighted losses, fresh initialization, one
optimizer update per trajectory and all within-pair identities are matched.
K8 detaches both W/Z without changing their values, and accumulates gradients
from all windows before clipping/updating. No architecture change was made.

## Minimal reading order for this delta

1. [Result and limitations](evidence/short_bptt_paired01/RESULTS.md) and
   [distance/rollout figure](evidence/short_bptt_paired01/overview.png).
2. [Compact metrics](evidence/short_bptt_paired01/analysis.json).
3. [Frozen protocol](new/short_bptt/PROTOCOL.md),
   [backward implementation](new/short_bptt/training.py),
   [runner and decisions](new/short_bptt/run.py), and
   [CPU checks](new/short_bptt/check.py).
4. [Validation](evidence/short_bptt_paired01/validation.json),
   [analysis code](new/short_bptt/analyze.py), and
   [publication hashes](BPTT_PUBLICATION_MANIFEST.json).

Raw arm JSONs and schedules are secondary, byte-identical copies. Public
preflight records document the runtime-only choice of300 updates before
efficacy. Checkpoints, machine receipts and transient logs remain local.
Publication used CPU verification, with no new training or checkpoint inference.

## New evidence and claim boundary

**All four full-K64 controls fail qualification** at the frozen size32/T64
far-paired endpoint. Both architecture comparisons are BASELINE_UNQUALIFIED.
The runner's aggregate `decision: COMPLETE` means execution completion only.
It does not support either a general truncation failure or superiority claim.

One positive case remains meaningful: additive K8 seed1 reaches BA98.03% and
far paired80.87%, retaining79.20%/78.24% atT128/T256. Its short reach and hold
predicates pass. Additive seed0 and both revision K8 arms fail reach; hence
there is no reproducible two-seed architecture pass, even aside from controls.

Keep the denominator visible: **80.87% per-map mean is59.53% pooled far pixels**.
Size32/T64 accuracy is45.60% in distance[32,64), and0% at distance>=64, the latter
on just one map. At size64 far paired is28.40% atT64 and41.51% atT256. The result
supports one instance of shared-rule composition beyond a K8 window, not robust
long-distance/size extrapolation or arbitrary delayed credit assignment.
Two communication phases per step give the window a radius16, not8.

Peak allocated CUDA memory is514.04MiB for K64 and90.49MiB for K8:82.4% lower.
Training times are similar,86-96 seconds per arm, with no accuracy-matched
systems advantage established. Additive K64 clips88.67%/90.67% of updates;
K8 clips6.67%/6.67%. This suggests an optimization difference but does not
identify clipping as the cause of full-control failure.

Sources, checkpoints, data and schedules verify;96 BA and336 paired aggregates
and all four pair decisions were recomputed. The CPU gradient tests pass with
zero full-gradient reference error. The new loss recipe and300-update budget
cannot borrow positive-control qualification from the older600-update study.
This is ordinary TBPTT on existing cells, not a claimed new training algorithm.

Unchanged: the earlier NO_JOINT_SCREEN_PASS and the source-switch audit's
old-aligned candidate/W-Z interaction result. The present screen evaluates
fresh starts, not warm source revision; it neither repairs nor retests that
failure. The earlier negative transport and dynamics evidence is preserved.
No further experiment or monitor is scheduled.

## Questions for review

1. Are full and truncated gradients correctly matched in loss normalization,
   parameter accumulation and one-update timing, including encoder gradients?
2. Is the graph-distance>16 endpoint correct for two communication phases,
   and is paired correctness kept distinct from BA and fixed-label accuracy?
3. Do summaries retain the unqualified controls, seed dependence, per-map
   versus pooled discrepancy and sparse far-distance support?
4. Is the memory reduction separated from efficacy/speedup claims, and is the
   clipping observation kept descriptive rather than causal?

---

# Historical handoff: zero-training candidate/workspace switch audit

- Review base: `7cb1ca975c13104b3abb157cf5324ea977ad1051`.
- Evidence head: `a700da6e8579830c36cea8ad6ecd014198430766`.
- This subsequent handoff-only commit changes review metadata only.

The audit reuses revision seed0, sizes32/64,16 fixed maps each, with no training
or Jacobian computation. It measures the actual post-F candidate, full-rollout
W/Z interventions and candidate-only W/Z/X factorials. It completed in9.609s;
all232 historical metric comparisons replay exactly.

## Minimal reading order for this delta

1. [Audit results](evidence/switch_audit_seed0/RESULTS.md) and
   [overview figure](evidence/switch_audit_seed0/overview.png).
2. [Compact metrics and per-map contrasts](evidence/switch_audit_seed0/analysis.json).
3. [Frozen protocol](new/switch_audit/PROTOCOL.md),
   [post-execution review notes](new/switch_audit/REVIEW_NOTES.md), and
   [audit implementation](new/switch_audit/audit.py).
4. [Validation](evidence/switch_audit_seed0/validation.json) and
   [publication hashes](SWITCH_PUBLICATION_MANIFEST.json).

The two raw JSON files are secondary. They have been losslessly compacted,
with decoded equality and original/published hashes recorded. Checkpoints and
machine receipts remain local. No extra inference was used for publication.

## New evidence and corrected interpretation

Warm candidate readout itself retains the old answer: changed-component
accuracy0% at size32/K64 and K128, versus100% for cold. At size64/K128 the
warm candidate is1.63% correct, cold output99.49%. This rules out the narrow
account that an already-correct candidate is merely averaged too slowly.

Single-block W/Z resets and mature donor transplants do not recover reliable
switching. At size32/K128, all four predict negative across every changed
component;37.5% is exactly the six negative target labels among16 maps. It is
not partial retrieval. A correct Z transplant initially gives100% changed
accuracy, then drops to37.5% in the old workspace. A new W donor with old Z
also fails. Both blocks influence the candidate; the continuous signed-margin
factorial includes a nonzero interaction, with per-map contrasts published.

This refines the proposed stale-workspace explanation: W-only causality has
not been isolated. Mixed donor states can be off-distribution, and donors
contain extra computation. The result does not prove both blocks must always
be reset, identify an attractor mechanism, or validate conditional invalidation
as a new architecture. Only one trained checkpoint was audited.

The protocol overstates fresh-flip norm replay: the old record stores W/Z
norms only for the original curve. Both curves' BA/BCE and per-map BA were
replayed; original W/Z norms and warm/cold switch metrics were also replayed.
The clarification is explicit in the review notes; frozen code/protocol and
original generated evidence were not rewritten.

The previous two-seed `NO_JOINT_SCREEN_PASS`, -9.02pp hold effect, seed1
qualification failure, and untested short-BPTT hypothesis remain unchanged.

## Questions for review

1. Does Q_K use post-F W correctly, and are candidate-only interventions
   distinguished from full recurrent interventions?
2. Are both W and Z confounds, incompatible mixed donors and donor compute
   costs kept explicit before assigning a mechanism?
3. Are all-negative37.5% behavior and the single-checkpoint limitation retained
   alongside cold success? Is any claim stronger than this evidence permits?

---

# Historical handoff: completed Workspace + Revision screen

- Review base: `b992e2eac68d1bcfc81b70044bb89100bc664c2d`.
- Evidence head: `cf157cfb36adf1be4a2bb5914575746630e2d5c6`.
- This subsequent handoff-only commit changes review metadata only.

All four paired arms completed 600 updates in 1112.28 seconds. Scientific
decision: **NO_JOINT_SCREEN_PASS**, with mean revision-minus-additive hold
effect **-9.02 pp** across two model seeds. This is new authorized training;
publication itself used only CPU verification of saved evidence.

## Minimal new reading order

1. [Current results](RESULTS.md) and
   [interpretation](evidence/workspace_revision_paired01/INTERPRETATION.md).
2. [Compact metrics and failed predicates](evidence/workspace_revision_paired01/analysis.json)
   and [both-seed overview](evidence/workspace_revision_paired01/overview.png).
3. [Frozen protocol](new/workspace_revision/PROTOCOL.md),
   [cell](new/workspace_revision/revision_cells.py) and
   [trainer/evaluator](new/workspace_revision/run_revision.py).
4. [Validation](evidence/workspace_revision_paired01/validation.json) and
   [publication hashes](REVISION_PUBLICATION_MANIFEST.json).

Raw arm records and shared schedules are secondary. They are byte-identical
copies; checkpoints, local receipts and machine metadata are excluded.

## Changed and unchanged conclusions

Revision seed0 reaches 99.19% BA64 and 100% hold on size32; size64 BA256 is
99.44%. However, its warm changed-component accuracy is 0% at K64/K128 versus
100% from cold initialization on size32. It holds an old answer but fails to
revise it. Revision seed1 stays at 50% BA. Hold effects are +5.82 and -23.86 pp;
neither the single-seed positive result nor size64 rescues the joint gate.

Z-only repair retains W; joint W/Z damage is much less recoverable. W keeps
growing despite stable output. No full-state stability, general repair,
attractor mechanism, 3D result or speedup is established. Both arms share the
W/Z split and all 5033 initial parameters; only the Z recurrence changes. The
split's own contribution remains untested. Two seeds are descriptive, with
different training banks/schedules across seeds; failure cannot be attributed
specifically to initialization. The runtime-only amendment from 800 to 600
updates preceded efficacy training.

The earlier dynamics audit and medium/recurrence comparisons remain unchanged.
Their different training and one-phase communication recipes are not matched
controls for this two-phase screen. No rescue sweep or new run was launched.

## Reviewer questions

1. Does the matched intervention support only a within-workspace update-rule
   comparison, without crediting the shared split or altered training?
2. Are cold/fresh paired success and warm source revision correctly separated?
3. Are failed qualification, null repair eligibility, continuing W growth and
   the two-seed uncertainty retained alongside the seed0 reach/hold signal?

---

# Historical handoff: generic dynamics audit after aa2d50a

- Review base: `aa2d50a91e4ff02f74144dc699124bd886b5aa85`.
- Evidence head: `4bcb8fef29bb796c3e1f7376d4b16f53163ba158`.
- The subsequent handoff-only commit changes review metadata only.

This completed, zero-training audit reuses the four existing seed-0 State and
Momentum checkpoints, with whole-grid and masked media. It adds exact tangent
and adjoint operators, finite-window gain estimates, trajectory diagnostics and
finite-perturbation checks. No new architecture, training or monitoring is added.

## Minimal reading order

1. [Current results](RESULTS.md) and
   [audit interpretation](evidence/dynamics_audit_seed0/INTERPRETATION.md).
2. [Compact audit table](evidence/dynamics_audit_seed0/RESULTS.md),
   [structured aggregates](evidence/dynamics_audit_seed0/analysis.json) and
   [overview figure](evidence/dynamics_audit_seed0/dynamics_overview.png).
3. [Frozen protocol](new/dynamics_audit/PROTOCOL.md),
   [tangent/adjoint implementation](new/dynamics_audit/operators.py) and
   [audit runner](new/dynamics_audit/audit.py).
4. [Validation](evidence/dynamics_audit_seed0/validation.json) and
   [publication hashes](DYNAMICS_PUBLICATION_MANIFEST.json).

Do not start with the 16 raw JSON files. They support verification after the
compact interpretation; checkpoints and local launch receipts are not published.
The executed audit sources are bound separately from post-hoc analysis code.

## New decision-relevant evidence

All 40 historical endpoint comparisons replay with zero error. The audit
completed normally. Performance uses 16 original maps per size (32 and 64);
Jacobian diagnostics use the fixed first four maps. One training seed remains
the independent replicate.

The proposed near-zero-to-positive maximum-gain crossing was not observed:
all 256 open-state K16 estimates are positive, including early horizons, and
the estimates generally decrease while late performance deteriorates. For
Masked Momentum at size32, T64 to T256 changes BA from 96.45% to 77.68%, while
median open K16 log gain per step changes from 0.27742 to 0.17582. These are
finite-window, raw-coordinate measurements, not asymptotic Lyapunov exponents.

Continued state updates and growing source-flip logit effects coexist with
declining paired correctness and increasingly wrong confidence. The generic
masked models do not support directly transferring the earlier explicit
Inertial RD component-mean drift explanation: most measured state and update
energy is within-component variation. These observations do not establish a
causal failure mechanism or a stable recurrence remedy.

## Diagnostic limits and unchanged claims

Only 562/1536 total product estimates and 148/256 open K16 estimates meet the
frozen convergence criteria. Unconverged direction estimates are not certified
maxima or upper bounds. At relative RMS perturbation 1e-4, the estimated leading
direction has median linearization error 46.44%, versus 1.19% for random
directions. Finite binary source flips and infinitesimal source tangents are
distinct diagnostics. No rescue sweep, extra restarts or extra seeds were run.

The audit does not rule out task-specific neutral modes, longer-product
instability, unsampled windows or conclusions under another state metric.
Full-state and open-endpoint-compressed products are distinguished; the latter
still allow intermediate wall paths in whole-grid models. The supplied medium
is fixed during differentiation.

The previous same-medium Momentum advantage at the primary endpoint, both
recipes' long-rollout failures and the earlier negative results remain intact.
Different content/program widths still prevent causal isolation of velocity.
Neither the medium interaction mechanism, selective dynamical isometry as a
remedy, asymptotic stability nor a systems speedup is established.

## Reviewer questions

1. Do the exact operators, total driven-source tangent, endpoint compression
   and replay checks match the frozen checkpoint dynamics and evaluation cohort?
2. Are the convergence and finite-perturbation limits prominent enough to
   prevent interpreting estimated maximum gain as a proven failure mechanism?
3. Which mechanistic conclusions, if any, follow beyond rejecting the observed
   maximum-gain zero-crossing narrative, without selecting a new architecture?

---

# Historical handoff: final masked-state control after 82635bb

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
