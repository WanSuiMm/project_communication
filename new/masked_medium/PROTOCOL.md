# Masked-medium diagnostic: frozen before training

Question: does replacing whole-grid transport with an input-mask-respecting
operator improve the two structured cells in the existing seeded-region task?
This is a separate, bounded follow-up to the negative inertial seed-0 screen.

## Intervention and controls

New arms: `masked_rd_nca` and `masked_inertial_rd`. Replace only the positive
five-point Laplacian by

\[
(L_mH)_i=\sum_{j\sim i}m_i m_j(H_i-H_j),\qquad m=X[:,0].
\]

The supplied binary input mask determines local edges. No component IDs, BFS
distances, labels or source signs are used to construct edges. Encoder, reaction,
readout, coefficient parameterization, optimizer and training/evaluation functions
are reused unchanged. Do not zero wall states or velocities: those would be
additional interventions. The wrapper calls the original model factory so both
parameters and RNG consumption match the original counterpart exactly.

Frozen controls are the `rd_nca` and `inertial_rd` outputs under
`evidence/inertial_seed0/arms/`. Verify their hashes and executed source hashes
against `INERTIAL_PUBLICATION_MANIFEST.json` before running. Original sources,
protocol and evidence remain unchanged. No old arm is retrained in this screen.

For binary mask edges, L_m is symmetric PSD with maximum weighted degree <=4,
so its eigenvalues are in [0,8]. Existing pure-transport coefficient bounds
remain sufficient. This does not certify the full reaction/rollout dynamics.
Each connected component has a constant zero mode; masked transport does not
guarantee bounded region means, convergence, or repair.

## Frozen budget and paired configuration

- 2D only; seed 0; both arms receive 800 AdamW updates.
- C=16, reaction width=128, eta=.1, initial d=.1; inertial beta=.9.
- 512 fixed 32x32 training maps, batch 8, lr1e-3, weight decay1e-4, clip norm1.
- Same generator and seeds as the controls: training bank10000, schedule20000;
  random rollout32/48/64 and identical minibatch/horizon/damage RNG sequence.
- Original late supervision and 50% batch damage recipe remain unchanged.
- 16 held-out maps at each size32/64/128, seeds30000+size; rollout16/32/64/128/256.
- Original conditional repair, seed revision, initial-state gradient probes and
  synchronized eager GPU timing are retained. Warm batch-eight time divided by
  eight is throughput-normalized time, not isolated batch-one latency.
- Local RTX4060 Laptop GPU, same PyTorch environment/backend settings. Expected
  duration about5-10 minutes; whole-run scheduling cap25 minutes. A cap or
  numerical failure yields incomplete/invalid evidence, not a scientific rejection.

One CPU operator/initialization check and a three-update, full-model CUDA
preflight precede training. Preflight uses abbreviated evaluation and is not
efficacy evidence. There is no sweep, additional seed or automatic follow-up.

## Outcomes and interpretation

Primary readout: paired changes versus the corresponding frozen unmasked arm in
32x32/T64 balanced accuracy and paired-source correctness. Both values and their
percentage-point differences must be reported; do not select the best horizon.
For a descriptive advance criterion, both differences >=5 percentage points
count as `JOINT_GAIN_5PP`. Otherwise report `NO_JOINT_5PP_GAIN`, retaining signed
differences. This cutoff is chosen for this screen before its first training run;
it is not a significance test or a claim that smaller differences equal zero.

Secondary: all size/horizon curves and distance bins, T64-to-T256 degradation,
sustained95% threshold, pre-damage eligibility and repair/revision controls.
Missing thresholds remain null. Zero repair-eligible maps remain unevaluable.

An improvement supports usefulness of this input-derived operator intervention
under this recipe. Masking changes graph connectivity, degree and spectrum;
it does not isolate cross-wall contamination from every other operator effect.
It does not establish an inertia advantage over generic momentum: that would
require a control given the same masked neighborhood. A failure means correcting
the medium alone was insufficient here, not that all reaction/transport
factorizations are impossible. Cross-run timing comparisons are descriptive.

Outputs: fresh `runs/masked_medium_20261001_seed0/`, source snapshot, manifest,
per-arm JSON/checkpoints, RESULTS.md and `paired_comparison.json`. Source hashes
and logged training schedules bind the historical comparison. Training seed is
the replicate unit; pixels and timing repeats are not independent model seeds.

Run from the standalone repository root:

```powershell
python new/masked_medium/test_masked.py
python new/masked_medium/run_masked.py --preflight --out runs/NEW_MASKED_PREFLIGHT
python new/masked_medium/run_masked.py --out runs/NEW_MASKED_SCREEN
```
