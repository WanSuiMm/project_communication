# Direct Spatial Carry: fixed K8 development screen

Frozen before efficacy training on 2026-10-02. This is a development screen
using previously inspected initialization seeds and maps, not confirmation.
Question: does a fixed channel-preserving spatial path make the existing
short-BPTT additive cell more consistently trainable?

## Single intervention

Baseline: W' = W + 0.1 F(W,Z,L_M W,L_M Z,X).
Carry: W' = W - 0.5 D_M^dagger L_M W + 0.1 F(same inputs).
Both: Z' = Z + 0.5 Q(W',Z,L_M W',L_M Z,X).

D_M counts real four-neighbor binary-mask edges, excluding exterior ghosts.
For degree zero (wall or isolated cell), carry is identity. Otherwise it is
half self plus half neighbor mean. Fixed rho=0.5; no sweep. F/Q, initialization,
readout, W24/Z8, 5033 parameters and two communication phases are unchanged.
The implementation reuses the existing L(W) feature; F sees the old state.
No revision, routing, momentum, normalization, new channels or auxiliary loss.

The carry alone preserves channel coordinates and constant fields, respects
components, and is nonexpansive in the maximum norm. It is lazy diffusion;
these properties do not prove Euclidean contraction, full-cell stability,
semantic preservation or restored gradients across detach. Spatial locality
and the two-edges-per-macro-step light cone remain unchanged.

## Matched training

Rerun BOTH arms for seeds2,3,4,5 (8 arms), rather than relying on historical
baseline scores. By seed the orders are baseline/carry, carry/baseline,
baseline/carry, carry/baseline. Independent seed is the descriptive unit n=4.
Use the exact Phase-II bank10002 (512 size32 maps), schedule20002 (300x8),
evaluation banks40032/40064 (32 maps each). Verify parameter, data, schedule
and evaluation hashes against the historical evidence and within matched arms.

All arms: 300 updates, fresh64-step trajectories, loss at8,16,...64 divided
by8, K8 detaching BOTH W/Z, parameter-gradient accumulation across windows,
one norm1 clipping and AdamW update after64 steps. AdamW lr0.001,
weight_decay0.0001, batch8, FP32, unchanged CUDA flags. No checkpoint selection,
state pool or training changes. Backend nondeterminism remains possible.

## Endpoints and decisions

Fresh original/flipped evaluation at sizes32/64 and T64/128/256. Reuse the
Phase-II evaluator including per-map BA and paired correctness on the changed
component: the same pixel must be correct for both source alternatives.
Primary: size32/T64, STRICT 16<d<32; both per-map mean and pooled paired>=80%,
both original/flipped BA>=85%. Hold: at BOTH T128/T256 each BA drops<=3pp
and each primary paired statistic drops<=5pp from its own T64.

Report reach and reach+hold separately. Development continuation requires:
1. All8 arms finish300 updates and evaluation, with verified identities.
2. Carry reaches AND holds in>=3/4 seeds, including historical positives2/5.
3. Carry reach+hold count exceeds the concurrently rerun baseline count.
4. Concurrent baseline seeds2/5 still reach AND hold. Otherwise mark
   BASELINE_REPRODUCTION_DRIFT and inspect that discrepancy before interpreting
   improvements; absolute carry outcomes remain visible.

If complete and matched but the carry conditions fail: DEVELOPMENT_NO_GO.
If they pass: DEVELOPMENT_GO (4/4 is a stronger descriptive signal).
Incomplete/numerical failure: INCOMPLETE, never a scientific pass. Do not
change gates, rho, seeds, horizon or budget after observing efficacy.

Secondary: ALL distance bands [0,8),[8,16),{16},(16,32),[32,64),[64,128),
[128,infinity), d>16,d>32, size64 and all three horizons. Retain denominators;
empty bands are null. Emit per-seed carry-minus-baseline effects for every
endpoint. Farther improvement is a prediction, not a rescue gate or proof
against smoothing/optimization explanations. T256 is retention, not a proof
of convergence. Systems: synchronized training time, peak allocation,
gradient norms/clipping. No significance or broad architecture claim.

## Engineering, runtime and scope

One CPU check covers the explicit neighbor-average reference (including outer
boundaries, walls and isolation), channel preservation, zero-rho equivalence
to the old cell including K8 gradients, parameter identity and unchanged F/Q.
One CUDA preflight runs3 updates of BOTH arms, seed2, abbreviated evaluation.
Let c be the maximum median of the last2 synchronized update times; require
8*300*c*1.20+120 <=1500 seconds and peak allocation<3GiB, finite results,
source/identity checks and primary coverage>=16 maps/500 pixels. Budget is
fixed300, without runtime-dependent reduction. If preflight fails, stop.

Formal cap25 minutes, emergency watchdog26 minutes. No automatic retries,
rescue sweeps, polling, confirmation or alternative architecture. New outputs
only. Preserve all earlier code/evidence. Record source snapshots/hashes,
parameter/data/schedule hashes, raw per-map counts, checkpoints, all arms,
aggregate, status and a private launch receipt. A launch request completes
with verified dispatch and receipt; later inspection is separate. A positive
development screen motivates fresh seeds/maps/bank confirmation as a future
separately dispatched stage; this runner does not start it automatically.

Commands from repository root:

    python new/direct_spatial_carry/check.py
    python new/direct_spatial_carry/run.py --preflight --out runs/NEW_CARRY_PREFLIGHT
    pwsh -File tools/launch_direct_carry.ps1 -RunName NEW_CARRY_RUN -Preflight runs/NEW_CARRY_PREFLIGHT
