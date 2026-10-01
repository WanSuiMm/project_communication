# Same-medium momentum control: seed 0

Question: under the same supplied binary mask graph, does the existing explicit
reaction/transport inertial recipe retain an advantage over a generic learned
momentum update? This is a new exploratory comparison frozen before training
this control. Candidate results are already known, not a blind preregistration.

## Intervention and preserved recipe

Only one new arm, `masked_momentum_nca`, is trained:

    L_m H[i] = sum_{j adjacent i} m[i]*m[j]*(H[i]-H[j])
    V_next = beta*V + eta*F_theta(concat(H, L_m H, X))
    H_next = H + V_next

Use the frozen `cells.make_cell('momentum_nca',16,128)` then wrap its step.
No explicit -d*L_m H term is added. The original factory's parameter values
and RNG consumption are identical to the unmasked momentum control.
Persistent state is H16/V16, beta starts at .9 and is learned channelwise,
eta=.1. Parameters: 4689, versus 4737 for masked inertial RD. Program hidden
widths are 88 versus 128. The generic update starts with a zero final layer;
explicit RD already has d=.1 at initialization. These are known differences
between architecture packages, not an isolated causal test of factorization.
No wall-state reset, new normalization, clipping rule or initialization fix.

Reuse the frozen `run_wind_tunnel.train_one` and all its data/loss/evaluation
functions unchanged. Seed0, 512 training maps at32, batch8, 800 AdamW updates,
lr1e-3, weight_decay1e-4, grad clip1, rollout sampled from32/48/64, losses at
T-4 and T. Half of training batches receive a6.25% square H/V erasure halfway.
Data seed10000, schedule seed20000; evaluate16 maps per size32/64/128, seeds
30032/30064/30128, at T16/32/64/128/256. All repair/revision/gradient probes
and throughput timings retain their frozen semantics. No extra seeds/sweeps.

## Endpoints and decisions

Primary: at32/T64 report BA and paired-source correctness against
`masked_inertial_rd` in `evidence/masked_medium_seed0/arms/`. Signed differences
are candidate minus new generic control. A descriptive candidate advantage
requires BOTH differences >=5 percentage points. The reverse requires BOTH
<=-5pp; otherwise call it mixed or smaller than the joint criterion. This is
not a population/significance gate or an architecture pass. Report the actual
differences even when the criterion is not met. Paired correctness pools changed
pixels, which are not independent replicates; the training-seed unit is n=1.

Also report the new-minus-old generic momentum contrast, all horizons/sizes,
BCE, state norms, threshold nulls, conditional repair eligibility and source
revision. Differing eligible subsets cannot establish comparative repair skill.
No controlled speedup claim from jobs run at different times.

Validity: verify frozen source/evidence hashes and exact initial weights/RNG;
test all-open equivalence, cross-wall isolation after a nonzero program update,
and finite gradients. Run one three-update preflight before training. Require
800 updates, all three evaluation sizes and no nested failure statuses for an
interpretable endpoint. Any failure/25-minute cap stops this run and is retained;
no automatic rescue. Expected formal run 2-5 minutes on the local laptop GPU.

## Outputs and reproduction

Run from repository root, each output path must be new:

```powershell
python new/masked_momentum/test_momentum.py
python new/masked_momentum/run_momentum.py --preflight --out runs/NEW_MOMENTUM_PREFLIGHT
python new/masked_momentum/run_momentum.py --out runs/NEW_MOMENTUM_CONTROL
```

`RESULTS.md` and `comparison.json` are entry points; raw arm JSON is secondary.
Manifest binds source snapshot, protocol, reference evidence, backend, config
and initialization. Checkpoint and local PID/launch receipt remain local.
This experiment adds no 3D, multiscale transport, force constraint or architecture
repair. Trajectory audit of the existing candidate is independent and does not
change this control based on its findings.
