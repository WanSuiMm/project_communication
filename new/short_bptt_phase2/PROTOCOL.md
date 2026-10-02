# Short-BPTT Phase II: initialization replication, fixed data

Frozen before efficacy training, 2026-10-02. This is a new bounded screen
authorized after the Phase-I interpretation. It does not amend the old gate.

## Question and scope

Does additive K8 repeatedly learn source-dependent propagation beyond its
single-window spatial radius when initialization varies but data and sampling
are fixed? Compare K8, K16 and K64 in the unchanged workspace-additive cell:
W24/Z8, 5033 parameters, two masked communication phases per macro-step.
No revision arm, state pool, learned gate, Jacobian audit or tuning sweep.

Phase I selected the promising architecture and distance band. Consequently
this is a prospective replication of a selected narrow-range observation on
new held-out maps, not an independent discovery or a rescue of its failed gate.

## Frozen training and random factors

- Initialization seeds 2,3,4,5; all three horizons per seed: 12 arms.
- One shared training bank: size32, 512 maps, data seed10002.
- One shared batch schedule: RNG20002, shape[300,8], reused in every arm.
- These data/schedule seeds are fixed choices, not selected from efficacy.
  Only initialization seed varies between replicates. Claims are conditional
  on this bank and schedule, not robustness across training distributions.
- Forward64 from fresh state; the same eight losses at8,16,...64, divided by8.
- K8 backpropagates one loss each8 steps. K16 backpropagates the sum of two
  losses each16 steps. K64 backpropagates all eight losses at step64.
- Cut BOTH W/Z history at window boundaries, retaining numerical values.
  Accumulate parameter gradients across windows. No parameter update inside
  a trajectory. Clip once at norm1, then optimizer.step once after step64.
- AdamW lr0.001, weight_decay0.0001, batch8, 300 updates for EVERY arm.
  No runtime-dependent reduction of this budget and no efficacy-based stopping.
- FP32, no compile; same CUDA backend flags as Phase I. Device/backend
  nondeterminism is not claimed to have been eliminated.
- Run orders by seed: [8,16,64], [16,64,8], [64,8,16], [64,16,8].
  Scheduling is a fixed operational choice, not another scientific variable.

## Evaluation and statistics

Fresh original and source-flipped trajectories; sizes32/64, 32 maps each,
data seeds40032/40064, measured at T64/128/256. Evaluation banks are disjoint
in seed from Phase I and from training. No checkpoint/seed selection.

BA averages per-map class recalls. Paired correctness requires the SAME
changed-component pixel to be correct under BOTH source alternatives.
Publish both per-map mean and pooled-pixel accuracy, integer correct counts,
denominators and eligible-map counts. Neither pixels nor maps are additional
training replicates; independent initialization seeds n=4, descriptive only.

Common absolute bands for ALL K: [0,8), [8,16), {16}, (16,32), [32,64),
[64,128), [128,infinity), plus the old primary d>16 and the farther d>32.
The new primary is STRICT 16<d<32 at size32/T64. This excludes distance16.
The evaluation bank must contain at least16 eligible maps and500 pixels in
this band, checked without a model. Otherwise stop, without replacing seeds.

As secondary diagnostics report normalized bands d<=2K, 2K<d<=4K and d>4K,
with counts. Different K correspond to different absolute-distance questions;
these bands must NEVER replace the common primary in model comparisons.
They indicate spatial range, not an exact number of temporal handoffs. At
T steps information cannot travel farther than2T; report the corresponding
paired score and logit difference, without treating an empty band as zero.

## Frozen decisions: keep absolute replication separate from comparisons

For every trained seed/K, the narrow-band reach predicate at size32/T64 is:
both original and flipped BA>=85%, AND primary mean>=80%, AND pooled>=80%.
Hold is separate: at BOTH T128/T256, each BA must remain within3pp of its
own T64 value, and both primary paired statistics within5pp of their own T64.

K8 initialization replication passes descriptively iff at least3 of4 seeds
meet reach. Sustained replication additionally requires at least3 of4 to meet
reach AND hold. Failure of either threshold is reported as NOT_REPLICATED for
that property. K16/K64 success counts are shown as context; K16 can cover this
primary band within one window, so its success is not cross-window evidence.
Farther bands and size64 cannot rescue a failed narrow primary, and a narrow
pass is not a robust far-distance, original-gate or size-generalization pass.

For matched comparative claims, K64 must qualify by the same narrow reach
predicate in at least3 of4 seeds. Otherwise label the comparison
BASELINE_UNQUALIFIED, while retaining the absolute K8 replication result.
Always show all four paired K8-K64 and K16-K64 effects for mean/pooled/BA;
no significance, noninferiority, superiority or general optimal-K claim.
This qualification rule belongs only to Phase II; Phase I stays unchanged.

## Engineering gate, budget and stop

One CPU check verifies K8/K64 gradient equivalence with the frozen trainer,
K16 against an independent window-gradient reference, exact forward identity,
both-state detachment and unchanged weights inside the trajectory. Then one
CUDA preflight: 3 updates at each K with initialization seed2, using abbreviated
evaluation. No preflight efficacy metric selects seeds, settings or budget.

Let c be the maximum across K of the median synchronized duration of the
last two preflight updates. Estimate seconds = 12*300*c*1.20 + 180.
Require this <=1800, peak allocated CUDA memory<3GiB, finite results and
all identity/coverage checks. If not, abort without formal training.
Formal cap30 minutes, emergency watchdog31 minutes. No automatic retry,
rescue sweep, continuation of partial runs, monitor or further experiment.
Any incomplete/numerical failure yields an incomplete scientific screen.

Record timing, CUDA peak allocation, clipping fraction and gradient norms;
they are descriptive systems/optimization measurements, not mechanisms.
Save sources/hashes, initial/final parameters, checkpoints, schedule, banks'
hashes, raw per-map evidence, live status and all partial aggregates. Local
launch receipt records host, PID, GPU, command and time, and remains private.
Use new output directories; frozen historical evidence stays unchanged.

Commands from repository root:

    python new/short_bptt_phase2/check.py
    python new/short_bptt_phase2/run.py --preflight --out runs/NEW_PHASE2_PREFLIGHT
    pwsh -File tools/launch_bptt_phase2.ps1 -RunName NEW_PHASE2_RUN -Preflight runs/NEW_PHASE2_PREFLIGHT

Completion of this launch request means verified dispatch and durable receipt.
Follow-up result inspection is separate; no continuous monitoring is scheduled.
