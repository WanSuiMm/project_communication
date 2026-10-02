# Candidate/workspace source-switch audit v1

Frozen before audit measurement, 2026-10-02. Zero training; existing
`ws_revision_seed0` checkpoint from the completed paired screen only. The
question is whether candidate readout keeps the old answer and how W and Z
interventions change this response. The old joint-gate failure is unchanged.

## Scope and provenance

- Sizes32/64, the same 16 evaluation maps each (bank seeds30032/30064).
- Old source at T64, then flipped source for K0/1/2/4/8/16/32/64/128.
- Size32 is primary, size64 a replication within the same trained model.
- One checkpoint, not independent training replication or a significance test.
- No training, gradients, Jacobian estimates, hyperparameter search or new model.
- Verify source/checkpoint/data hashes against REVISION_PUBLICATION_MANIFEST.json.
- Replay saved original and fresh-flip BA/BCE and W/Z RMS at all seven original
  horizons, then saved warm/cold switch endpoints. Stop on mismatch: accuracy
  absolute tolerance1e-6; continuous metrics atol1e-6, rtol1e-5.
- One CPU algebra check, then the bounded CUDA audit. Five-minute cap, stop on
  nonfinite state, failed replay, or numerical identity. Save a new run and
  separate analysis; preserve all prior code, raw records and weights.

## Candidate clock and measurements

At state (W_K,Z_K), first compute Wplus_K using F with the branch's current X.
Then Q_K=Qnet(Wplus_K,Z_K,X) drives the update to Z_(K+1). Record current
O(Z_K), candidate O(Q_K), W/Z/Q/update RMS and changed-component new-label
accuracy and signed logit margin, with unchanged accuracy, BA, BCE and per-map
values. Distance bins on the changed component are descriptive.

The actual affine readout must satisfy O(Z_(K+1))=0.5 O(Z_K)+0.5 O(Q_K),
including its bias. Candidate probing must exactly reproduce the real step
within float32 tolerance. Candidate correctness is a readout projection, not
a complete interpretation of the latent Q tensor.

## Full-rollout interventions at T64

All switched branches receive the same new X at every subsequent step:

1. warm: (W_old64,Z_old64).
2. reset_W: (encoder(Xnew),Z_old64).
3. reset_Z: (W_old64,0).
4. cold: (encoder(Xnew),0), the exact reset-both control.
5. transplant_W: (W_new64,Z_old64).
6. transplant_Z: (W_old64,Z_new64).

The new64 donors come from a fresh rollout of the same input for 64 steps;
they use no hidden labels, but contain extra computation and are diagnostics,
not deployable interventions or equal-cost baselines. Also retain intact old
and new trajectories at absolute age64+K as reference donors. Resets and mixed
donors can be outside the training-state distribution. A reset may destroy
useful information, so its failure cannot exonerate the replaced state block.

## Candidate-only factorial

At each K take post-F W from warm switching and from the age-matched fresh
new-input trajectory at T64+K. Cross those two Wplus donors with the two
corresponding pre-update Z donors and old/new X: 2x2x2 queries to Qnet.
Hold the other tensors fixed for each contrast. Do not run an additional F
inside this factorial: it tests Q's interface, not all input-to-state paths.
Also query freshly encoded W with warm Z under old/new X; these unprocessed
donors lack distributed computation and are labeled separately.

Report W and Z contrasts under new X plus the factorial interaction, per-map
as well as their means. Old/new direct-X differences do not measure repeated
evidence injection through F or initialization. Mixing donors is an intervention
on this frozen function, not proof of a unique natural causal mechanism.

## Interpretation decided before measurement

- Old-aligned candidate: changed-component new accuracy <=5% at both K64/128.
  This supports persistence at the candidate readout, not W-only causality.
- Report each intervention's changed/unchanged accuracy at K64/128 and gain
  relative to warm. Describe an intervention as recovering at K128 only if
  changed>=80%, unchanged>=85% and changed gain>=20pp. This is descriptive.
- If a donor replacement changes candidate outputs at fixed other inputs,
  record sensitivity to that donor; neither necessity nor sufficiency of an
  architecture principle follows. Inspect W, Z and their interaction together.
- Full-rollout controls distinguish immediate candidate sensitivity from
  sustained recovery. No architecture selection follows from one seed alone.
- Seed1 training failure and short-BPTT performance are outside this audit.
  At zero-output-layer initialization the state Jacobian is exactly block
  diagonal (I,0.5I), but that fact alone does not explain seed1 or parameter
  credit through shared weights and later W/Z coupling.

Commands from repository root (new output names only):

    python new/switch_audit/audit.py --check
    python new/switch_audit/audit.py --out runs/NEW_SWITCH_AUDIT
    python new/switch_audit/analyze.py --run runs/NEW_SWITCH_AUDIT --out analyses/NEW_SWITCH_REVIEW
