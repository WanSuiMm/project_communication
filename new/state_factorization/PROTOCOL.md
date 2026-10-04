# Output-preserving state factorization v1

Frozen before fresh-map outcomes. Question: which parts of the selected S1
T64 state enable the fixed F1 consumer to preserve work and continue computing?
This is a finite localization experiment, not a RelationFirst mechanism proof.

## Fixed models, medium and interventions

Use exactly S1=historical H_u300 and F1=H_swap_early8_u300 from the published
continuation_interface_v1 configuration. Verify file and parameter SHA256s,
unchanged StreamingCell sources, update300, eval mode and no parameter updates.
F1 is the only intervention consumer, including its readout. Inputs, masked
streaming clock, eta and alpha remain unchanged. Handoff is absolute T64;
continue through T256, including the immediate T64 output before any step.

S/F denote native S1/F1 states. For F1's nonzero readout weight r (bias is
unchanged), form pointwise channel projections of dZ=Z_S-Z_F:

    dZ_span = r * (r dot dZ) / (r dot r)
    Z_span = Z_F + dZ_span
    Z_null = Z_F + dZ - dZ_span

Projection arithmetic is float64; resulting states are cast once to float32.
There is no fitted alignment, normalization, rescaling or optimization.
Eight arms: FF, F-span, F-null, FS, SF, S-span, S-null, SS, from
W in [W_F,W_S] and Z in [Z_F,Z_span,Z_null,Z_S]. Save r, projector,
native T64 states and all immediate logits. SF must have EXACT FF logits;
F-null/S-null must preserve FF output within normalized max error1e-6
and exact original/flipped decision agreement on all open cells. Span/S
must reproduce F1's readout of Z_S within the same numeric tolerance.
Save absolute errors, per-map errors and decision disagreement, even on failure.
A failed neutrality check labels that arm OUTPUT_NEUTRALITY_UNQUALIFIED;
do not interpret it as output-preserving or silently repair its state.

## Fresh banks and cohorts

Primary: size32,32maps,seed96032. Confirmation: size64,32maps,seed96064.
Original/flipped cues are both evaluated; paired correctness means BOTH are
correct at the same cell. Verify exact input hashes against the historical
training bank and ALL six previous continuation-interface banks, comparing
both cue versions. Smoke uses disjoint seeds96932/96964,2maps each.
No sample selection or new bank after seeing outcomes.

Before any transplant, fix cohorts inside the source-flipped changed component:

    keep = correct_S64 AND correct_F64
    rescue = correct_S64 AND NOT correct_F64
    prog = NOT correct_S64 AND NOT correct_F64
    F_only = NOT correct_S64 AND correct_F64   # secondary

All arms use these identical cell sets and denominators. Frontier is the
fixed prog subset adjacent by a valid four-neighbor edge to keep. Native S1
and F1 rollouts to256 are controls; originals remain unchanged. Record the
historical Full predicate separately on these fresh banks. No transfer proxy
is renamed Full qualification.

## Outcomes and interpretation fixed in advance

Report each cohort's immediate T64 correctness, continuous correctness64:256,
endpoint128/256, ever-correct after64 and sustained correctness241:256.
Also report delayed_sustained = wrong at transplanted64 AND correct at ALL
241:256, divided by the ORIGINAL fixed cohort denominator. This separates
instant answer replacement from later acquisition. Save integer pooled counts,
equal-map means, nonempty-map counts, per-map records and complete Boolean
traces; cells are not independent training replications.

The conditional rescue proxy requires keep continuous>=.95, prog sustained
>=.10, and prog sustained at least .10 above FF, with keep/prog each having
>=16 nonempty maps and >=100 cells. Size32 is primary. Size64 is independent
map/scale confirmation, not an alternate route to a primary pass. Incomplete
cohorts make the proxy unqualified. Preserve all continuous outcomes.

Require native S1 Full and native F1 NOT Full on the fresh banks plus SS rescue
pass at size32 for a QUALIFIED_LOCALIZATION_EPISODE. SS must improve fixed-cohort
progress over FF by the predeclared .10, rather than treating FF's necessarily
zero self-improvement as a meaningful negative control. Otherwise label
LOCALIZATION_EPISODE_UNQUALIFIED and retain all eight arms at both sizes.
Report confirmation separately. A failed SS anchor is not evidence that a
component contains no useful information. Do not choose another checkpoint.

An SF or F-null rescue with neutrality verified supports a conditional effect
of readout-invisible state changes on future computation. It does not prove
encoded task information rather than scale/control effects. Hybrid failures
do not establish a special W/Z relation; both successful hybrids do not prove
universal sufficiency. Same initialization does not ensure component-wise
coordinate compatibility after training. Findings apply to these checkpoints,
inputs, interventions and finite horizon only.

Record handoff W/Z RMS, donor differences, Z/W, first-step F/W and Q/Z ratios,
separately by original/flipped cue and open cells. Also record W/Z RMS over
rollout. These are descriptions, not gates or rescaling prescriptions.

## Checks, execution and delivery

One focused CPU fixture group tests fixed denominators, trace endpoints,
instant/delayed separation, frontier boundaries and empty eligibility. CPU
projector fixtures test arbitrary non-axis r, bias, span/null reconstruction
and zero-readout rejection. One two-map CUDA smoke at both sizes verifies
actual neutral logits, all eight suffixes through256, exact FF/native-F1
suffixes, immutable parameters and instrumented first-step equivalence.
Sources, model bindings and backend flags must match qualification at launch.

Use historical Torch2.5.1 FP32,threads2,cuDNN benchmarkFalse,
deterministicFalse,cuDNN TF32True,matmul TF32False,batch8 paired cues.
Require >=3500MiB free GPU memory. Stop on binding drift, parameter mutation,
nonfinite native state, or failed FF suffix plumbing. Nonfinite intervention
arms are saved as NONFINITE_CONTINUATION and do not erase the remaining arms.
No training, new seeds, runtime cap, watchdog, rescaling rescue or monitor.

New output directory only. Save protocol/source snapshot, bindings,
configuration, fresh banks, native/arm traces and logits, native states,
cohorts, projection, scales, metrics.csv, summary.json, RESULTS.md and figure.
Machine-specific manifests, launch receipts and checkpoints stay local.
Expected runtime is a few minutes, not an enforced limit. Launch completion
means verified dispatch and durable receipt; later monitoring is separate.

From repository root:

    python -X utf8 -B new/state_factorization/check.py --out analyses/NEW_STATE_CHECK.json
    pwsh -File tools/launch_state_factorization.ps1 -RunName NEW_STATE_RUN -Qualification analyses/NEW_STATE_CHECK.json
