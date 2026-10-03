# Seed4 follow-up: three independent experiments v1

Frozen before outcome measurement. No architecture change. A schedule screen,
B initialization-direction screen and C local temporal-state intervention answer
different questions. Neither A nor B must pass before another experiment runs.
One local GPU executes jobs without simultaneous memory/compute contention.

## Shared references, execution and scope

Original StreamingCell,5033 parameters,W24/Z8, original two-hop macro clock.
Historical model initialization seed4, train bank512 maps of size32, seed10002.
300 updates per training arm, batch8, AdamW(lr=.001, weight_decay=.0001), clip
gradient norm1; forward64, loss every8, detach every8. Preserve all old evidence.
Bind the45 previously audited source/reference files, all new executed sources,
data/schedule/checkpoint hashes, software/backend and snapshots in NEW runs.
Local historical FP32 settings: threads2, cudnn benchmarkFalse/deterministicFalse,
cudnn TF32True, matmul TF32False. No nondeterminism or run-order sweep.

One focused CPU qualification and a three-update GPU smoke of control/A/B
before training launch. The three-update control also times one complete
fresh256-step phenotype evaluation for the budget estimate; its phenotype
gates are not a scientific endpoint. Training hard cap2400 seconds (40 minutes), watchdog
2460; expected order of magnitude30 minutes. Stop on source/preflight drift,
nonfinite state/loss/gradient, insufficient memory, cap, or historical control
reproduction drift. Save partial results; no automatic rescue/sweep. C cap300
seconds, watchdog330 and its own independent qualification. A C failure does
not invalidate A/B. No recurring monitoring or automatic follow-up.

## Training control

Retrain seed4 once with historical initialization, bank and schedule20002.
Compare initial/final parameter hashes and the FULL six historical size32/64,
T64/128/256 evaluation records. Replay drift invalidates A/B interpretation;
it does not get repaired by changing settings or choosing another control.
Also evaluate this control on the fresh frozen banks below. If the control
misses the fresh phenotype, report BASELINE_FRESH_PHENOTYPE_UNQUALIFIED and
keep all measurements; do not redefine thresholds after observing them.

## A: fixed starting parameters, changed batch schedule

Four schedule seeds20012,20022,20032,20042; each draws300x8 indices uniformly
with replacement from the SAME512-map training bank. Use the same seed4
starting parameter hash in every run. Optimizer, loss, updates and execution
settings unchanged. Unit: independently drawn schedule, n=4 conditional on
this bank and starting point. >=3/4 phenotype passes is a bounded conditional
schedule-robustness signal, not a population reliability or basin theorem.

## B: perturbed starting parameters, fixed training

Flatten ALL trainable parameters in sorted-name order, including originally
zero weights/biases. Let theta0 be the historical seed4 starting vector.
Draw four independent CPU Gaussian directions with seeds70002..70005 and
normalize each to unit L2. For each SAME direction at both epsilon=.01,.05:

    theta_start = theta0 + epsilon * ||theta0||2 * unit_direction

Noise changes originally zero-initialized blocks too; this is a Euclidean
parameter-space probe, not preservation of the zero-output initialization
manifold or layerwise relative noise. Record actual global/block norms and
initial SHA. Use the SAME historical schedule20002, bank and optimizer for
all eight arms. Unit: direction n=4; the two radii are paired, not eight
independent replicates. >=3/4 passes at a radius is sampled directional
tolerance for this fixed training recipe, not a certified ball or basin radius.
Control then interleave A and both B radii per direction; no scientific
dependence on order. No epsilon sweep or perturbation of final weights.

## Fresh behavior endpoint for control/A/B

Before training freeze32 evaluation maps per size32/64, seeds50032/50064.
These are fresh to this project, shared across arms and disjoint from the
historical seeds40032/40064. Model does not receive graph distance or labels.
Record original/flipped paired correctness every step0..256; save Boolean
traces. Confidence margins are not a new endpoint. Per-map first/stable/censor
profiles, transitions and frontier counts retain denominators.

Every following condition is required for the exploratory phenotype pass:

1. Reach at size32/T64: strict16<d<32 paired map mean AND pooled>=.80,
   original AND flipped whole-open-grid BA>=.85. At T128 AND T256, declines
   from T64 are bounded by.03 for both BA and.05 for paired mean/pooled.
2. At EACH size: all-changed correct-at64 retention at256>=.95; increase in
   all-changed pooled coverage64->256>=.05; ever-regressed/ever-correct<=.15.
3. At EACH size: exact-map/time/source-BFS-distance matched one-step frontier
   acquisition difference>=.05, eligible maps>=16 and common strata>=100.
   Starts0,8,...,248; within-map weights nf*nn/(nf+nn), then equal-map mean.

Empty denominators/groups fail qualification, never count as success. Retention
alone cannot qualify a stalled tiny solved region. Terminal-stable ends at256;
right-censored first passage is reported, not treated as arrival257. No model
reliability significance or independent-pixel tests. Keep historical metrics
separate from the fresh frozen endpoint and all prior architecture decisions.

## C: existing checkpoint, local temporal-state intervention

Use original trained Streaming seed4 with32 historical maps per size32/64.
Reproduce six full endpoints before interpretation. Snapshot t-1,t at
t=16,32,64,128. Select targets using ONLY these snapshots and task labels:
p currently paired wrong, d(p)>8; adjacent non-source q newly paired correct,
d(q)=d(p)-1; another adjacent r remains paired wrong at both snapshots and
d(r)=d(q). Require nonzero r rollback directions for all nonzero q block deltas.
No selection on native next-step acquisition. Lexical cap4 events/map/time.

Each event gets an independent cloned full-map state and original/flipped
world; never apply several interventions to one map instance. Fixed conditions:

- Native/no-op copy.
- Sender rollback: replace q's W/Z by its own t-1 values.
- Wrong-neighbor sham: r's own rollback direction, scaled separately in W/Z
  and each source world to equal the corresponding q rollback L2 magnitude.
- Off-cone control: add q's delta at a deterministic non-source changed cell s
  at Manhattan distance>2 from p. Step1 target logits must remain unchanged.

Inputs, weights and other state sites fixed. Check no-op agreement, target
state untouched at intervention, q logits restored to the previous snapshot,
block norm matching, finite values, and off-cone step1 max logit error<=1e-6.
Measure target paired acquisition after1 step(primary) and4 steps(secondary).
Maps receive equal weight after within-map averaging over selected events.
Need>=64 events and>=16 eligible maps at EACH size, else insufficient matched
events with no widening. Positive descriptive C signal requires native-minus-
sender>=.10 AND wrong-sham-minus-sender>=.05 at BOTH sizes after qualification.

Whole-cell temporal rollback is local-state sensitivity, not an edge-specific
message intervention. Equal-norm sham does not match direction, phase or full
activation context. Selection is conditional on newly correct sender/wrong
target/wrong comparison neighbor. C can test a causal effect of this state
perturbation on these targets, not identify a unique semantic handoff law,
flood-fill, latent closure or the cause of short-BPTT training success.

## Outputs and delivery

Separate run directories and summaries for A/B and C; new outputs only.
Training launch completes with verified dispatch and durable PID/command/GPU/
output/time receipt. Later result review is a separate request. No GitHub
upload is included in this launch request. Fresh maps and schedules remain
fixed after freezing; no adaptive iteration counts or architecture v2.
