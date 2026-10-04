# Gauge-aware continuation interface audit v1

Frozen before any new outcome measurement. Zero model training, no optimizer
steps, no architecture search, no runtime cap, no watchdog and no monitor.
The only transformed checkpoint is an in-memory exact Z-gauge clone of H.

## Questions and scope

Do similar short supervised behaviors hide different continuation behavior?
Can a fixed 100-coefficient state-coordinate map enable cross-model future
computation, preserving a producer's solved cohort while advancing its unsolved
cohort? Is an apparent producer/consumer interaction explained by this restricted
linear coordinate family? K8/full64 gradient comparisons are diagnostic, not
proof that truncation, optimization or an objective caused the outcome.

The independent objects are selected trained trajectories, not pixels. Map-level
records describe these checkpoints on fixed banks; there are only two selected
historically successful trajectories. No population reliability, universal
identifiability theorem, common-algorithm theorem or architecture GO follows.

## Models and selection

- S1: bootstrap historical `H`, update 300.
- S2: stage-2 `init31006_schedule32006` baseline, update 300.
- F1: bootstrap `H_swap_early8`, update 300 (same initialization/data multiset).
- Eligible fresh failures: the other 15 baseline final checkpoints from the
  completed stage-2 suite. Eligibility uses their already published Full-gate
  label, not a new long-horizon score.
- F2: smallest short probability-MSE distance to S1 among eligible failures.
- F3: smallest short probability-MSE distance to S2, excluding F2. Ties break
  by increasing initialization seed. Selection sees only the new short bank
  at times 8,16,...,64, original and flipped cues, equal maps/times/cues, open
  cells averaged within each map. No new T128/T256 result enters selection.

Bind checkpoint file and parameter hashes to original records. Keep originals
read-only. S1/S2 must be requalified on common test banks using the unchanged
original Full predicate. A failed common native qualification does not trigger
replacement, new seeds or training; all audits remain descriptive and the
shared-success interpretation is labelled unqualified.

## Fixed disjoint banks and execution

Original training bank: size32, 512 maps, seed10002, original cues only for
the exact historical supervised objective. New independent RNG banks:

| Role | Size | Maps | Seed | Observations |
|---|---:|---:|---:|---|
| Short objective / failure selection | 32 | 32 | 91032 | 8,16,...,64; both cues |
| Alignment fitting | 32 | 16 | 92032 | 8,16,24,32,40,48; both cues |
| Alignment validation | 32 | 16 | 93032 | 56,64; both cues |
| Future test | 32 | 32 | 94032 | native0:256; handoff64:256 |
| Future scale test | 64 | 32 | 94064 | native0:256; handoff64:256 |
| Non-scientific startup smoke | 32 | 2 | 90032 | gauge/forward checks only |

The model receives only original three-channel input. Save bank hashes and
input/label/mask/distance arrays. Verify new sampled inputs do not coincide
across roles or with training samples; seeded banks alone do not imply this.
FP32 local historical Torch2.5.1; threads2, cuDNN benchmarkFalse,
deterministicFalse, cuDNN TF32True, matmul TF32False. Batch8 throughout;
original+flipped inference concatenates the cue batch. Record current device,
software, PID, commands and time in local receipts, not public summaries.

## Phase 0: actual frozen short observables

Evaluate all 18 models (S1,F1,16 fresh baselines) without gradients. On the
entire historical training bank and the new short bank, save per-map and mean
balanced BCE and BA at every supervised time. The bank loss is the equal-map
and equal-time average. Training-bank loss uses original cues; also report
the new short-bank flipped branch separately and its two-cue average.

Retain new-bank logits at all eight times; report per-map probability RMS
disagreement and decision disagreement on open cells for every model pair.
These finite-bank measurements cannot prove exact A(theta)=A(theta').
An exploratory matched-observables flag requires BOTH bank BCE gaps <=.02,
new-bank two-cue probability RMS <=.05 and decision disagreement <=.05.
A continuation coverage gap >=.25 at T256 on the same test bank is a
descriptive under-identification pattern only, not an optimizer impossibility.
Always publish continuous values even when this flag fails.

## Phase 1: exact gauge qualification

Draw one Z permutation with fixed seed95008. For Z'=PZ, conjugate q_out
output rows/bias, readout input columns, and Z/LZ input columns in f_in/q_in.
Keep W unchanged. Verify nontrivial CPU toy dynamics and known-coordinate
handoff before formal dispatch. Formal H/H^P control uses four alignment maps
with both cues, T0:256, raw and known-P handoff at64. Native clone and aligned
handoff must have normalized maximum W/Z/logit discrepancy <=1e-3 and paired
correctness agreement >=.999 on changed cells. Report raw handoff separately;
raw failure is NOT required. If the known-P control fails, save
GAUGE_CONTROL_UNQUALIFIED and stop the cross-model interpretation.

## Phase 2: one restricted alignment family

A(W,Z)=((I4 tensor A_W)W,A_Z Z), A_W6x6 and A_Z8x8. Same A_W in all four
directional lanes, no bias, context independence, no output-label fitting.
Fit each directed off-diagonal pair independently using all open calibration
cells/cues/times. Float64 no-bias ridge has fixed coefficient1e-4 times
trace(X^T X)/dimension in each block; a zero design uses scale1. No tuning.
Self maps are EXACT identity, avoiding ridge-induced self shrinkage.

Freeze maps before reading future tests. On the separate validation bank,
report W/Z relative RMS residuals, adapter singular values/condition numbers,
and reverse-forward cycle Frobenius error normalized by sqrt(dimension).
The restricted invertible-map qualification requires residuals <=.10,
cycle errors <=.10 and condition numbers <=100 in BOTH blocks. Failure
weakens this tested coordinate-map family; it does not exclude nonlinear
relations or a common abstract algorithm. Low-rank observed states may make
full-space cycle identification unqualified even when on-manifold fitting is
good. No additional alignment family is allowed in this run.

## Phase 3: complete cross-consumer matrices

All five producers × all five consumers, raw and aligned, on both future
test banks. Producer runs original/flipped cues to T64; transplant the ENTIRE
(W,Z), leave receiver parameters/readout/input fixed, then run to T256.
Diagonal is ordinary native continuation; its identical raw/aligned data may
be referenced rather than rerun. Retain every integer paired-correctness
trace, immediate logits and T128/T256 logits, native full traces and T64
states. Nonfinite continuation is explicitly recorded as such, never hidden
as an ordinary classification failure.

Cohorts are fixed by PRODUCER native paired correctness at64 and its changed
component. Bands: all_changed, 16<d<32, d>32, and initially unsolved cells
adjacent by one valid four-neighbor edge to a solved changed-component cell.
Save integer counts, pooled ratios, equal-map means and empty denominators.

- Preservation: correct at EVERY time64:256 among producer-correct-at64.
- Immediate retention and endpoint retention at128/256 use that same cohort.
- Progress: ever correct after64 among producer-unsolved-at64.
- Sustained progress: correct at EVERY time241:256 among that unsolved cohort.
- Coverage and gains at64/128/256 retain denominators.

A descriptive transfer qualification requires preservation>=.95 and
sustained progress>=.10 on all_changed at BOTH sizes; each progress cohort
must have >=16 nonempty maps and >=100 pixels. Also require that neither
metric is more than .05 below the producer's native continuation. Empty
groups fail qualification; do not call a frozen tiny solved set successful.
Report native/cross differences and continuous values for every cell, not only
the gate. Separately decompose complete scalar matrices into grand mean,
producer effect, consumer effect and centered interaction. This is descriptive
for selected models, with no independent-pixel ANOVA or causal population test.

## Frozen-gradient diagnostic

Selected five models, same four consecutive batches of8 from the new short
bank, original cues. Evaluate the same BCE objective with K8 detach and
full64 BPTT, no clipping or optimizer step. Save gradient vectors, group norms,
cosines and averaged gradients. For the same-init pair only, report dot
products with normalized theta_F1-theta_S1; this parameter difference is a
diagnostic direction, not a state-interface fit or future-loss descent proof.
Require forward losses to agree within1e-6 and checkpoint parameters to be
unchanged. No Adam moments are inspected. This final diagnostic does not
select models or adapters.

## Completion, failure and artifacts

One combined CPU sanity and a two-map GPU smoke before formal launch. Stop on
binding drift, failed exact gauge plumbing or inadequate memory; preserve the
receipt and partial outputs. Scientific negative results do not launch rescue
experiments. New outputs only under runs/continuation_interface_20261004_01.
Write status.json, manifest/source snapshots, model selection, short objective
records/arrays, frozen alignments/validation/cycles, gauge control, native
qualification, all matrix traces/per-map records, gradient arrays, summary.json
and concise RESULTS.md. Launch completion means verified dispatch plus a
durable receipt. No recurring status polling or automatic GitHub push.
