# Historical continuation-formation diagnostic v1

User question: does a measurable continuation property precede and predict the
successful behavior on the seed4 training trajectory, especially175/180/195/200?
Saved-artifact CPU analysis only: no forward inference, gradient, training,
new architecture, or checkpoint selection. Preserve all prior frozen evidence.

Two clocks are distinct: u is training update; t is inference rollout step.
All22 available checkpoints100..200 every5, plus nonadjacent300 control, are
reported at32/64. Focus checkpoints175/180/195/200 are not the fitting set.
The original size32 descriptive long screen is retained verbatim: strict pooled
and equal-map coverage at T128>=.80, endpoint G64->128>=.20, first-exit64->128
<=.01, and continuous survival64->256>=.95. The300 point is not treated as
adjacent to200. Only one selected training trajectory is available.

Candidate predictors use t<=64 ONLY, for each fixed checkpoint:

1. Early continuation proxy: continuous survival32->64 among paired-correct
   at32, plus acquisition G32->64 among wrong at32. The simple candidate screen
   uses the existing .95 survival and .20 G thresholds, not new tuned cutoffs.
2. New-state continuation proxy: cells wrong at24, correct at32, subsequently
   correct at every step32..64. Compare with old-correct cells correct at every
   step24..32. These are observable cohorts, not certified commitment types.
3. Margin reserve: for correct-at64 cells, compare signed paired margin m64
   against8 times their maximum one-step destructive margin decrement over
   t56..64. Report the fraction with m64-8*max_decrement>0. This is a fixed,
   speculative local extrapolation, not a certified Lyapunov/contraction bound.
   Scale cancels under a positive constant readout rescaling; hidden semantics
   and future contexts are not specified by this scalar.

Use all-changed and strict16<distance<32 cohorts; retain per-map numerators
and denominators. Empty cohorts are undefined, never passes. Do not fit a
quotient/decoder, pick a threshold to isolate200, or add candidates after
seeing the result. Candidate validity at64 is compared to subsequent same-rule
behavior through128/256. Separately compare a candidate at u to the long-screen
outcome at u+5; checkpoint300 is excluded from this lagged comparison. A future
optimizer update can change the rule, so failure to predict u+5 does not refute
closure of a fixed learned rule. Neither retrospective comparison is out-of-
sample predictive validation. No p-values or independent-checkpoint inference.

Primary diagnostic verdict: whether the specified early candidate screen has
an observed lead over the200 long behavior and whether it also generates false
positive warnings. Continuous rankings/pairwise ordering are descriptive;
there is only one successful checkpoint in the dense window. Exact behavioral
closure is not measured: no proposed pi, shared abstract transition F, raw-domain
certificate, or held-out consumer intervention is supplied by these arrays.

If candidates merely redescribe success or fail prediction, lower the weight of
the specific seed4 mechanism hypothesis, not the valid conditional execution
theorem or every possible continuation representation. Report the evidence gap.

Outputs in a new run directory: RESULTS.md, summary.json, profiles.csv, figure,
artifact hashes and COMPLETE receipt. Canonical input:
runs/transition_20261003_seed4_dense01/{summary.json,checkpoints/uNNN_sizeNN.npz}.

    python -X utf8 -u -B new/formation_gate/analyze.py --out runs/NEW_FORMATION_GATE
