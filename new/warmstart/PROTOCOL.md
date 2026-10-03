# Detached warm-start StreamingCell development screen v1 (runtime amendment)

Protocol ID: detached_warmstart_v1_nocap. On 2026-10-03, before any formal
training, the user explicitly removed the runtime limit. The earlier CPU01
and preflight01 records remain immutable; preflight01 failed only its former
2400-second launch gate (estimated2451.03 seconds). This amendment removes
time-based qualification and watchdog termination for smoke and formal runs.
It does not change the scientific recipe or phenotype thresholds. The new
source-bound CPU02/preflight02 records qualify the amended implementation.

Question: does exposing the same K8 recurrent rule to older on-policy partial
states make the full solve-and-preserve phenotype easier to learn? This is a
training-state-distribution control, not a new architecture, state pool across
optimizer updates, exact replication of Bansal's mixed progressive objective,
or a test proving a phase transition/continuation closure.

## Fixed recipe and paired units

Original StreamingCell:5033 parameters,W24/Z8, masked streaming and the original
two-hop macro clock. Four initialization seeds2,3,4,5. Each seed has baseline
and warmstart arms, eight arms total. SAME512-map size32 training bank10002,
historical batch-index schedule20002,300 updates,batch8, AdamW(lr=.001,
weight_decay=.0001), global clip norm1. FP32 PyTorch2.5.1, threads2, cudnn
benchmarkFalse/deterministicFalse, cudnn TF32True, matmul TF32False.

Gradient-bearing suffix64 steps; loss at suffix8,16,...,64, each divided by8;
detach every8 steps, one optimizer update after all eight losses. Parameters
are fixed throughout prefix and suffix. No long-gradient path, halting gate,
age curriculum, architecture change, epsilon sweep or adaptive rescue.

## Sole training intervention

Freeze300 prefix ages with NumPy default_rng(60002), uniform choices from
{32,64,128,192} macro rollout steps under the original two-hop clock.
All seeds/arms share the age list. Each update uses the last
four batch examples as the warm subset (indices4..7), and the first four as
fresh examples. Batch indices themselves remain the old uniform schedule.

Both conditions run the SAME no-grad on-policy prefix on that four-example
subset, starting from the current model's initial state and using the current
parameters. Prefix states are detached, losses are not applied in the prefix,
and no state survives across optimizer updates.

- Baseline discards the prefix and executes the untouched historical K8
  helper from fresh initialization for all eight examples. The discarded
  prefix matches forward compute; it must not change historical training.
- Warmstart retains the prefix state for the last four examples, and starts
  the first four fresh. Fresh examples retain their original encoder gradient
  through the first eight-step window; warm examples have detached prefix
  history. Both groups then execute the identical64-step K8 suffix.

Warm examples receive task losses at total ages a+8,...,a+64 (maximum256),
fresh examples at8,...,64. Different supervised state ages are the treatment.
Both arms have equal optimizer updates, examples, trained suffix steps,
loss count/weights and prefix forward compute. This does not equalize total
state age or claim that new age exposure is a pure credit-assignment effect.

## Historical control and new primary endpoint

Every baseline must exactly reproduce its corresponding published original
Streaming initial/final parameter SHA and FULL six size32/64 xT64/128/256
historical records on seeds40032/40064. Source, data and schedule hashes must
match. Drift stops execution and invalidates the screen; do not repair by
changing thresholds/backend or selecting another control.

Freeze NEW32-map evaluation banks per size32/64, seeds60032/60064. They are
shared across arms and distinct from prior400xx/500xx and training banks.
Evaluate from fresh state, original/source-flipped paired predictions, every
macro step0..256. Reuse the unmodified seed4_followup phenotype implementation
and its COMPLETE conjunction:

1. size32/T64 strict16<d<32 paired map mean AND pooled>=.80, original AND
   flipped open-grid BA>=.85; at BOTH T128/T256 drops fromT64<=.03 BA and
   <=.05 strict mean/pooled.
2. EACH size all-changed correct-at64 retention at256>=.95, pooled coverage
   gain64->256>=.05, ever-regressed/ever-correct over0..256<=.15.
3. EACH size exact-map/time/BFS-distance matched frontier effect>=.05,
   eligible maps>=16 and common strata>=100; starts0,8,...,248, within-map
   weights nf*nn/(nf+nn), then equal-map mean. Empty groups fail.

Known historical seed4 baseline must also pass this new full phenotype for
BASELINE_QUALIFIED; a miss produces BASELINE_FRESH_PHENOTYPE_UNQUALIFIED and
all measurements are retained without threshold changes.

Independent unit: paired initialization, n=4 conditional on one bank and
batch schedule. Report four paired gates and all components. A bounded
positive development signal requires qualified baseline/control replay,
warmstart full passes>=3/4 AND at least two more full passes than baseline.
Otherwise report DEVELOPMENT_NOT_QUALIFIED, or the baseline/engineering
qualification failure. No population significance or independent-pixel tests.
Partial reach/hold/dynamics improvements remain secondary, not gate rescue.

## Training-stage diagnostics (exploratory, separate from the endpoint)

Save CPU state_dict checkpoints at updates0,100,200,300 for every arm.
At those updates evaluate16 held-out size32 maps, seed61032, for128 paired
steps from fresh state. No training or arm choice uses diagnostic outputs.
Record correct-set size, wrong->right acquisition and right->wrong destruction
counts/denominators at every step, plus64->128 continuous survival, endpoint
retention, growth and ever-regression. Verify the exact integer identity
correct_new-correct_old=acquired-destroyed. Do not use G/D without denominators
or call complementary single-step survival/destruction independent evidence.

These observations locate changes across training updates; four paired models
do not establish an order parameter, abrupt phase transition, two attractors,
causal handoff or invariant latent representation. Hidden state need not stop.

## Execution and provenance

One local GPU to retain the historical backend/replay. One focused CPU check,
then three-update baseline/warmstart GPU smoke at maximum prefix age192.
Smoke times complete final phenotype and a stage diagnostic; neither smoke
phenotype is scientific. A conservative time estimate from worst-prefix update
timings plus endpoint/diagnostic timings is reported for planning only; it is
not a launch gate. No runtime deadline or termination watchdog is enabled.
Memory headroom>=3500MiB is still required. Updates/seeds/maps/ages stay fixed.
Stop on source/preflight drift, nonfinite prefix/suffix/parameter/gradient,
replay mismatch or memory insufficiency. Preserve
partial records; no automatic rescue. Formal order pairs seeds2..5, reversing
baseline/warmstart order for odd-indexed pairs to reduce order confounding.

Prefix finiteness is evaluated on every initial/stepped state. Device-side
failure flags stay sticky across each eight-step window, with host checks at
window boundaries and the final prefix step. This synchronization-only
engineering change preserves numerical states, gradients and transient-error
detection; CPU02 checks both exact historical equivalence and a NaN/recovery
fixture. The failed initial smoke is retained separately.

Bind frozen source/reference and new source hashes, snapshots, initial/stage/
final weights, bank/schedule/age hashes, software/backend, per-arm output and
elapsed time. New run directories only; older evidence stays read-only.
The launch request completes with verified dispatch and a durable host/PID/
GPU/command/output/time receipt. No recurring monitoring or GitHub upload is
included in this request. A later result check is separate.
