# Binary Carrier Causal Compression — frozen v1

Question: can a local decoded binary field, or source-only binary tokens, replace
the selected success producer's W while the fixed failure consumer continues?
The recurrent model is never trained or updated. A linear decoder is fitted once.

## Models, banks and fitting

Use the existing hash-bound S1/F1 update-300 checkpoints. All consumers are F1;
handoff is T64, final time T256, paired original/one-component-flipped cues,
FP32, eight maps per batch and the historical CUDA backend settings.

Fresh roles are: calibration 64 size32 maps (seed 98032); decoder validation
64 size32 maps (98132) and 64 size64 maps (98164); intervention tests 128 size32
maps (98232) and 128 size64 maps (98264). Smoke uses two maps at each size,
98332/98364. Split by map, keeping each original/flip pair together. Input and
geometry hashes must be disjoint between these roles and earlier published
continuation/interface, state-factorization and W-medium banks and known smokes.
No bank expansion or replacement in response to measured outcomes.

Capture S1 W at T64 for calibration and validation. Fit one logistic linear
decoder on local W24 only, shared across sizes. X, Z, position, geometry,
distance and component identities are forbidden decoder features. Open masks
only select calibration cells. Weight each map/cue equally: each open cell has
weight 1/(number of open cells in its map/cue), normalized globally. Standardize
with calibration weighted mean/std (floor 1e-6); L2=1e-3 on standardized
coefficients, unpenalized intercept. Deterministic Newton/backtracking, at most
100 iterations, gradient infinity norm <1e-7; no fit/regularization sweep.
Both classes and a converged finite fit are required for numerical qualification.
Poor validation accuracy remains a result, not a reason to tune or stop controls.

Use this same weighting for fixed class-centroid templates p0/p1. Never optimize
templates against continuation. Raw-score >=0 maps to bit1. Neutral template
is the fixed midpoint (p0+p1)/2. Report decoder predictions of both templates
and projection idempotence; neither is assumed. Validation/test labels are used
only for measurement, except the explicitly oracle-only diagnostic arm.

## Interventions: 12 arms at each size (24 measurement cells)

FF=(W_F,Z_F), SF=(W_S,Z_F), SS=(W_S,Z_S) are restart anchors, all under F1.
Keep/prog cohorts are fixed from paired native S1/F1 correctness at T64 on the
changed component, before any intervention. Every W intervention applies to
**all open cells**, with wall W untouched and Z_F unchanged.

1. `bit_once`: W_i=p[q(W_S_i)] at T64, then ordinary continuation.
2. `bit_repeat8`: the same restart, then decode/project current W after every
   eighth F1 transition, at T72,80,...,256, before that time's readout. Z is
   untouched. There are 25 projections including T64. Save per-projection bit
   fields and pre/post W norms; assert exact Z and immediate readout equality.
3. `swap_once`, `swap_repeat8`: same decoder with template assignments reversed,
   with the corresponding clocks. These are semantic identity controls.
4. `midpoint_once`, `midpoint_repeat8`: every open W replaced with the midpoint,
   on corresponding clocks. These are payload-removal controls.
5. `oracle_once`: W_i=p[y_i] only at T64. Explicit oracle diagnostic; excluded
   from primary claims and algorithm candidates.
6. `source_once`: source locations/bits derived exclusively from recipient
   X1/X2. Positive source gets p1, negative source p0, other open cells midpoint.
   No labels, changed mask, distance or component ID construct this arm.
7. `source_swap_once`: the same source construction with p0/p1 reversed.

Stage B is a source-only W restart with recipient Z_F64 retained and X supplied
at every step. It is not computation from an otherwise empty state.

## Measurements and qualification

Reuse frozen paired keep-continuous T64:256 and prog-sustained T241:256 metrics,
including delayed progress, immediate output, endpoints and per-map records.
Primary gates use pooled cell rates for continuity with earlier gates; map is
the independent unit for descriptive equal-map means and paired bootstrap
intervals (2,000 resamples), never treat cells/cues as independent replicates.

Qualification requires SS and SF to pass the earlier rescue gate on both sizes:
keep>=.95, prog>=.10, prog gain over FF>=.10; keep and prog each >=16 nonempty
maps and >=100 cells. Native Full flags are descriptive and separate. Measure
all controls even if this anchor gate fails, but label causal compression
UNQUALIFIED. No rescue by selecting a subset or extra seeds.

Each primary compression arm (`bit_once`, `bit_repeat8`, `source_once`) must
have keep>=.95 and prog>=.90*SF prog **at both sizes**, with the above support,
finite trajectories and exact immediate W-only readout neutrality. Report each
size separately and the conjunctive gate. Nonfinite interventions are FAIL;
nonfinite anchors or plumbing failures terminate with ERROR.

Report decoder accuracy on all open cells, changed/unchanged cells and fixed
shared-unsolved prog cells, for each cue and paired correctness. Report
q(W_original) XOR q(W_flip) against changed-mask, including changed flip rate
and unchanged no-flip rate. Record raw score summaries and per-map values.
Template-swap opposite-answer rates distinguish coherent inversion from mere
loss of correctness. No decoder-accuracy threshold retrospectively selects maps.

## Interpretation and execution

One-shot success is tested restart sufficiency; repeated success is finite
repeated projected continuation under this specific consumer. It does not prove
a global quotient, unrestricted closure, an architecture or a BPTT guarantee.
Source-only success is a source-coded restart effect with existing Z and X.
Oracle success with predicted failure implicates this decoder; oracle failure
rejects the tested centroids/consumer combination, not every possible bit code.
All conclusions concern this selected pair and task, not training reliability.

CPU fixtures and a two-map CUDA smoke precede dispatch. Bind source, checkpoint,
bank, fitted decoder and run-configuration hashes. Save full raw measurement
arrays, states used for fitting/restarts, fixed cohorts and aggregate JSON/CSV.
Expected work is roughly 10–25 minutes on the local RTX4060 (estimate only).
No runtime cap, watchdog, automatic monitoring or architecture follow-up.
