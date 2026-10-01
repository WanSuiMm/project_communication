# Final missing cell: masked state-matched NCA, seed 0

Question: in the existing synthetic seeded-region task, does generic momentum
retain an advantage after ordinary state-matched NCA receives the same input-mask
graph? Add ONLY `masked_state_nca`; then stop. Existing three corners are known.
This is an exploratory completion of a2x2 comparison, not a blind preregistration.

## Model and intervention

    H_next = H + eta*F_theta(concat(H, L_m H, X))
    L_m H[i] = sum_{j adjacent i} m[i]*m[j]*(H[i]-H[j])

Use frozen `cells.make_cell('nca_state_matched',16,128)` and replace only its
perception Laplacian with the frozen masked operator. H has32 channels, no V;
eta=.1, hidden width48, parameters4993. Preserve exact initial weights and RNG
consumption relative to old unmasked state NCA, including the factory's dummy
construction. Both generic models start with a zero final program layer.

Masked Momentum has H16/V16, hidden width88 and4689 parameters. Persistent
scalar count is exactly32 for both; parameter counts are approximate (state NCA
has6.48% more). H/readout width and program width also differ, so this is a
comparison of the established state-matched recipes, not isolated causal removal
of velocity or evidence that momentum is universally necessary/unnecessary.

## Frozen execution

Reuse original `run_wind_tunnel.train_one`, data, loss and evaluator unchanged.
Seed0,800 updates,512 training maps at32,batch8,lr1e-3,AdamW weight_decay1e-4,
grad clip1; horizons32/48/64, lossesT-4/T, half the batches receive6.25% square
state damage halfway. Train data seed10000 and schedule seed20000. Evaluate16
maps per size32/64/128 (seeds30032/30064/30128), horizons16/32/64/128/256, and
the same repair/revision/gradient/timing probes. No new audit or model changes.

Controls: unmasked state and unmasked Momentum in `evidence/inertial_seed0/arms/`,
masked Momentum in `evidence/masked_momentum_seed0/arms/`. Verify their hashes,
all referenced frozen source hashes, and new-vs-old state initialization/RNG.
Require matching logged schedules,800 updates, all evaluations valid. Use a
three-update CUDA preflight after minimal implementation checks. Stop on failure
or25-minute cap; no automatic retry/sweep/seed. Expected formal run2-5 minutes.

## Endpoints and interpretation

Primary:32/T64 BA and paired-source correctness. Report all four cells, masked
Momentum minus masked state, each model's masking effect, and the descriptive
difference-in-differences:

    interaction = (masked_momentum - unmasked_momentum)
                - (masked_state - unmasked_state).

Report differences in percentage points for both metrics. Momentum has a
descriptive joint5pp advantage only if BOTH primary differences>=5pp; state has
one if BOTH<=-5pp; otherwise label MIXED_OR_BELOW_JOINT_5PP. Small differences
are not statistical equivalence. No threshold is chosen after seeing the new
arm. One model seed is the statistical unit; pixels/maps do not become multiple
training replicates. A positive interaction is descriptive, not a population
causal mechanism. Masking changes connectivity, degree and spectrum, not only
cross-wall leakage; the provided mask is a task-specific input prior.

Secondary:all horizons and sizes, BCE,H RMS, sustained95% threshold, conditional
repair/revision, gradient clipping and initial-state gradient probes. Do not
replace the primary horizon with whichever later point favors a preferred model.
No cross-run speedup or matched conditional-repair superiority claim.

## Reproduction and stop

From repository root, new output directories required:

```powershell
python new/masked_state/test_state.py
python new/masked_state/run_state.py --preflight --out runs/NEW_STATE_PREFLIGHT
python new/masked_state/run_state.py --out runs/NEW_STATE_CONTROL
```

Manifest and source snapshot bind the executed code, protocol, config and
references. Read RESULTS.md and comparison.json first; raw arm metrics second.
Checkpoint and local launch receipt stay local. After this arm, stop: no v2,
damping, multiscale, attention, extra seeds, or recurring monitor.
