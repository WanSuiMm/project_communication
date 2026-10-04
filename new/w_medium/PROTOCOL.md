# W medium qualification v1

Frozen before fresh outcomes. Question: does output-neutral W-only replacement
reproduce on adequate map support, and how sensitive is that conditional effect
to scalar amplitude, consumer coordinates, spatial organization and donor time?
This is one selected S1/F1 checkpoint pair, not a training-reliability study.
No architecture change, fitting, gradient calculation or optimizer update.

## Fixed context and stage 1

S1 = historical H_u300, F1 = H_swap_early8_u300, exactly as bound in the
published continuation-interface configuration. File and parameter hashes,
frozen StreamingCell code, F1 readout, inputs and FP32 backend are fixed.
Original/flipped cues are evaluated together. State ordering is all original
maps followed by all flipped maps. W has 24 channels = N/E/S/W lanes x6;
Z has8, and the readout depends only on Z.

Fresh primary size32:128 maps, seed97032. Confirmation size64:128 maps,
seed97064. Check both cue hashes against all six continuation-interface
banks, both state-factorization banks, and the four prior/new two-map smoke
banks. New smoke seeds97932/97964. No rejection sampling, extra bank, checkpoint
replacement or sample selection after outcomes.

Native S1/F1 controls run through256. Fix native T64 cohorts in the changed
source component before intervention, reusing state_factorization.metrics:
keep=S_correct & F_correct; prog=~S_correct & ~F_correct;
rescue=S_correct & ~F_correct; F_only=~S_correct & F_correct.
Paired correctness requires both original and flipped predictions correct.
Frontier is prog adjacent by a valid four-neighbor edge to keep.

Stage1 arms FF, SF, FS, SS. Every transplant enters F1 at absolute T64 and
continues to256, including the T64 output before any update. No normalization
in stage1. FF must exactly reproduce native F1's saved suffix. SF must have
exactly the same immediate logits and open-cell decisions as FF.

Keep the earlier rescue gate unchanged: keep continuously correct64:256>=.95;
prog correct at ALL241:256>=.10; prog sustained gain over FF>=.10; keep and
prog each >=16 nonempty maps and >=100 cells. Insufficient map/cell support
is UNQUALIFIED, not a behavioral FAIL. Native Full is recorded separately.

Stage2 runs automatically only if both SS and output-neutral SF pass the
rescue gate at BOTH sizes. Native Full is descriptive, not a controls gate:
historical T64 reach requirements are a different question from this conditional
W replacement effect. Separately retain the earlier localization episode
predicate (native S1 Full, native F1 not Full, SS rescue at primary size), and
its confirmation; a miss is never relabeled as historical Full qualification.
Otherwise COMPLETE_STAGE1_ONLY, all control cells SKIPPED_STAGE1_GATE; preserve
all stage1 data. No fallback or gate relaxation. Size64 cannot rescue size32.

## Stage 2: fixed W-only controls

All Z remain Z_F64 and all recipient inputs remain unchanged. Reference is SF.
Fourteen additional arms per size (28 cells), for 36 total intervention cells:

1. SF_norm_F: W_S64 multiplied by RMS(W_F64)/RMS(W_S64).
2. FF_norm_S: W_F64 multiplied by RMS(W_S64)/RMS(W_F64).
   One scalar per cue/map, RMS over all W channels on open cells, computed in
   float64 then cast once to float32. Scale the whole tensor, including walls;
   record that choice and all scales. Zero donor/reference open RMS or overflow
   stops execution without replacing the transform. This tests only this scalar norm account.
3. SF_payload: same nonidentity gather permutation (2,5,0,4,1,3) of six payload
   channels within every lane. No averaging or lane change.
4. SF_payload_gauge: same state permutation and exactly conjugated consumer.
5. SF_lane: gather lane permutation (1,2,3,0), retaining each six-channel block.
6. SF_lane_gauge: same state permutation and exactly conjugated consumer.
7. SF_spatial: scramble whole 24-channel cell vectors separately inside each
   four-connected changed component, seed97007+size. Outside component fixed;
   same spatial permutation for original/flip, all channels and all future use.
8. SF_spatial_gauge: same state permutation and exactly conjugated consumer.
9. SF_cue_swap: exchange original/flipped donor W for the same map, retaining
   Z_F64 and recipient cue. Geometry/channel convention unchanged; norms may
   differ and are recorded. Tests cue provenance, not a pure semantic axis.
10-13. SF_time32/48/80/96: W_S at those native times, Z_F64, F1 T64 clock.
14. SF_norm_F_cue_swap: exchange original/flipped SF donor W, then norm-match
   each cue/map to W_F64. Fixed paired provenance+scalar-scale control.

For each gauge control, P acts ONLY on W; the tested consumer is
P_W Phi_F1 P_W^-1, and its readout is unchanged. Permutations and their exact
inverses are saved. This is a coordinate/plumbing positive control, not a new
learned model or the original consumer. Gauge correctness traces must exactly
match SF and endpoint logits must be exact. A mismatch halts with ERROR.
Raw permutation failures test compatibility with a fixed consumer; they do
not alone identify semantics. Spatial scrambling also changes local gradients,
spectrum and alignment to X/geometry. Donor time changes content, reach, scale
and state age simultaneously; no pure-phase interpretation is allowed.

## Reporting and decision boundaries

Reuse all seven frozen cohort metrics: immediate64, continuous64_256,
endpoint128/256, ever_after64, sustained241_256, delayed_sustained; also changed
coverage. Denominators never depend on transplant outcomes. Save pooled integer
counts, equal-map means and per-map numerator/denominator, complete Boolean
traces, endpoint logits, open-cell W/Z RMS over the suffix, initial force/write
scale descriptors, and masked edge-difference descriptors for transformed W.
Report every arm's frozen rescue proxy and paired equal-map differences from
SF, plus descriptive 95% paired-map bootstrap intervals (2000 replicates,
seed97100+size+arm index); never treat cells as independent samples. Empty maps
are excluded for that metric and the count is explicit. No seed-level CI.

Controls are sensitivity tests, not an exclusive four-way mechanism classifier.
No predeclared result establishes W as solely execution or Z as solely answer:
earlier Z-null addition on Z-span had a conditional future effect. There is no
carrier loss or short-credit guarantee in this protocol.

## Verification, outputs and execution

One CPU fixture group verifies fixed-cohort arithmetic, norm match, exact
inverse permutations, cue ordering, closed/component exclusions, and the
stop gate. One two-map CUDA smoke at both sizes checks all transforms, actual
neutral logits, full suffixes through256, native FF equality and exact gauge
suffixes; parameters remain bitwise unchanged. Smoke does not qualify behavior.
Source/checkpoint hashes and backend must match at launch; >=3500MiB free GPU.

Torch2.5.1 FP32, threads2, cuDNN benchmarkFalse/deterministicFalse/TF32True,
matmul TF32False, batch8 paired maps. Stop on source/checkpoint drift, parameter
mutation, nonfinite native state, failed FF plumbing or gauge equivalence.
Nonfinite raw handoff/first step/suffix is recorded NONFINITE_CONTINUATION,
proxy false, with remaining prespecified cells retained. Invalid transform
construction (including zero RMS or overflow during norm matching) stops the
run as specified above; no transform replacement. No cap, watchdog or monitoring.

New run directory only. Save source/protocol snapshots, configuration,
validation bindings, fresh banks, native states/traces, fixed cohorts,
transforms/inverses, donor times, every arm's traces/logits/norms, metrics.csv,
summary.json, RESULTS.md and scientific figure. Checkpoints and machine
receipts stay local. Typical run: several minutes; no enforced runtime limit.
Completion of launch means verified native execution plus durable receipt;
checking subsequent progress is a separate request.

From the repository root:

    python -X utf8 -B new/w_medium/check.py --out analyses/NEW_W_MEDIUM_CHECK.json
    pwsh -File tools/launch_w_medium.ps1 -RunName NEW_W_MEDIUM_RUN -Qualification analyses/NEW_W_MEDIUM_CHECK.json
