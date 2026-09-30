# Context for incremental scientific review

## Current 2D inertial wind-tunnel screen (seed 0)

This separate experiment asks whether explicit five-point reaction–transport
with inertia learns seeded-region identity better than generic momentum NCA,
with a state-capacity control. Four arms completed 800 AdamW updates each on
512 fixed 32x32 training maps. Evaluation uses 16 held-out maps per size
(32/64/128) at horizons 16/32/64/128/256. This is one exploratory seed.

At 32x32 and T=64, `inertial_rd` reached 69.94% balanced accuracy, compared
with 84.44% for `momentum_nca` and 85.17% for `nca_state_matched`. At T=256,
`inertial_rd` was at 50% balanced accuracy with zero paired-source correctness
on sizes 64 and 128. Every arm/size 95%-sustained threshold is null. The
candidate has zero repair-eligible examples at every size, so conditional
repair is unevaluable; this is not a repair failure estimate.

The result is negative for this one-seed recipe. It does not refute an
architecture family or support multi-seed superiority, a 3D claim, a repair
claim, or a speedup claim. The result also does not establish that the task
and training recipe have a reliable positive control.

Read the [compact summary](evidence/inertial_seed0/summary.json),
[per-arm results](evidence/inertial_seed0/RESULTS.md), [configuration](evidence/inertial_seed0/config.json),
[linear checks](evidence/inertial_seed0/linear_checks.json),
[validation](evidence/inertial_seed0/validation.json), and
[publication manifest](INERTIAL_PUBLICATION_MANIFEST.json). The detailed
[candidate](evidence/inertial_seed0/arms/inertial_rd_seed0.json),
[generic momentum control](evidence/inertial_seed0/arms/momentum_nca_seed0.json),
and [state-matched control](evidence/inertial_seed0/arms/nca_state_matched_seed0.json)
retain their per-horizon records. The source protocol is
[WIND_TUNNEL.md](new/nca_inertial_wind_tunnel/WIND_TUNNEL.md), with analytic
scope in [THEORY.md](new/nca_inertial_wind_tunnel/THEORY.md) and local evidence
notes in [LOCAL_INTEGRATION.md](new/nca_inertial_wind_tunnel/LOCAL_INTEGRATION.md).

| Concept | Exact symbol | Source |
|---|---|---|
| Cell family, five-point operator, arm updates and coefficients | `ARMS`, `laplacian`, `Cell.step`, `Cell.coefficients`, `Cell.rollout`, `make_cell` | `new/nca_inertial_wind_tunnel/cells.py` |
| Region maps, labels, damage and task metrics | `components`, `sample`, `bank`, `damage`, `metrics` | `new/nca_inertial_wind_tunnel/tasks.py` |
| Training, evaluation, thresholds and timing | `sustained_threshold`, `evaluation`, `gradient_probe`, `benchmark`, `train_one`, `main` | `new/nca_inertial_wind_tunnel/run_wind_tunnel.py` |
| Pure-transport analytic checks | `roots`, `radius`, `block_matrix`, `run` | `new/nca_inertial_wind_tunnel/math_checks.py` |

## Current A0 status and minimum route

Protocol rt_a0_2d3d_v2_1 completed 24/24 conditional trials (40 maximum; 16 3D RT
trials skipped). In 2D, attention is perfect for all four seeds and all reported
conditions; all four RT arms are NOT_QUALIFIED_FIT. In 3D, attention seed 1729
fails and no RT treatment estimate exists. Numerical/oracle checks pass but
do not establish learned-model qualification. B/C and width sweeps remain unrun.

Read [RESULTS.md](RESULTS.md), [A0_PROTOCOL.md](A0_PROTOCOL.md), then the
[small A0 summary](evidence/a0_v2_1/summary.json). Use the
[per-seed table](evidence/a0_v2_1/RESULTS.md) before opening
[aggregate.json](evidence/a0_v2_1/aggregate.json) for histories, axis metrics
and diagnostics. [checks.json](evidence/a0_v2_1/checks.json) and
[A0_PUBLICATION_MANIFEST.json](A0_PUBLICATION_MANIFEST.json) establish validation
and provenance. Independent units are seeds 1729, 2718, 31415, 57721.

Every A0 arm emits q=c*v, with learned sigmoid group weights c initialized at 0.5.
Raw uses T(q); normalized uses T(q)/(T(c)+1e-6). Both perform the same packed
grouped solve. Initial common parameters and medium scale match; only the
receiver division changes within a medium/seed pair. Constant conductance is 1;
learned conductance is 2*sigmoid(local symmetric features), initialized at 1.
Attention projects query/key and uses q directly as values, with no separate
value/output projection. C=64, r=32, G=4, K=8; two phases of four updates. A0 raw is
not identical to v1 raw. Fixed axis order and target-only readout are retained.

| A0 concept | Exact symbol | Source |
|---|---|---|
| Standalone entry | package/module dispatch | `a0.py`, `run.py` |
| Forward and diagnostics | `A0Model.forward` | `a0_models.py` |
| Same-value attention | `SharedValueAttention.forward` | `a0_models.py` |
| Grouped numerator/confidence solve | `transport_pair` | `a0_models.py` |
| Config, trial and conditional gate | `config`, `trial`, `decisions`, `main` | `a0_runner.py` |
| Paired estimates | `paired_effects` | `a0_runner.py` |
| Software and oracle checks | `run_checks`, `oracle_diagnostics` | `a0_checks.py` |
| Public snapshot | `summarize`, `main` | `tools/export_a0_evidence.py` |

The 2D result is negative under a reliable fitting control. It neither proves
diffusion-family impossibility nor causally separates emitter/readout failures.
Sampled normalized-model confidence remains dense on background; this is not
proof that a different confidence/emission mechanism cannot learn. Confidence-
mass contribution is not a total-information attribution through recurrence.
Changes between v1 and A0 cannot isolate why attention fitting improved.

The sections below describe historical v1 and its unchanged code/evidence.

## Historical v1 formal status

- Scientific status: `INCONCLUSIVE_POSITIVE_CONTROL` in both 2D and 3D.
- Endpoint reached: completed Gate A, 20 model/seed/dimension runs.
- RT fit status: not qualified even at training-shape distance 16.
- Software status: numerical and end-to-end smoke checks passed.
- Not run: Gate B, Gate C, r=4/8/16/32/64 width scan, real vision, foundation
  pretraining, directed transport, DEQ, generation or video.
- Independent replicate: training seed (1729, 2718). Test examples, voxels and
  timing repetitions do not create additional training replicates.

## Hypothesis and variants

Main hypothesis: retain local H, emit Q with r<C channels, exchange Q through a
content-dependent symmetric nearest-neighbor medium, then update H locally.
Width C=64, r=32, four groups and eight updates form a reduced-width screen of
the earlier illustrative C=128 design. Residual scale starts at 0.1.

Controls: CNN has eight unshared local cells; NCA has two cells repeated four
times each; attention substitutes global SDPA messages at the same r; constant
transport fixes internal conductance to one. Learned transport predicts local
conductance from E and normalized H at each phase start. All get the same
normalized coordinate inputs and target-only readout. Parameter counts differ.

Gate A has one random source bit. Gate B would add four sources on separated
straight regions and an oracle-edge diagnostic. Gate C would compare equal-
width Q=H arms differing only in residual base H versus T(H), using joint remote
source/local-detail labels. These B/C implementations are smoke-tested, not
scientifically evaluated.

## Exact source map

| Concept | Symbol | Source |
|---|---|---|
| Synthetic tasks, independent source/detail labels | `make_batch` | `data.py` |
| Symmetric endpoint edge prediction | `_EdgeBuilder.forward` | `models.py` |
| Local reaction and write gate | `_Reaction.forward` | `models.py` |
| SDPA message control | `_GlobalMessage.forward` | `models.py` |
| Phase sharing, persistent state and target readout | `GateModel.forward` | `models.py` |
| PCR factor reuse | `factorize`, `solve_factors` | `transport.py` |
| Implicit RHS/edge adjoint | `_ImplicitSolve.backward` | `transport.py` |
| Palindromic 2D/3D splitting | `prepare_transport`, `apply_transport` | `transport.py` |
| Matched training schedule, evaluations | `trial`, `evaluate` | `runner.py` |
| Frozen decisions | `verdicts` | `runner.py` |
| Numerical checks | `run_checks` | `test_transport.py` |

## Evidence route and boundaries

Read `RESULTS.md` and the small `evidence/qualification_v1_1/summary.json` first.
Use `config.json` to establish the protocol; use `aggregate.json` only for
per-axis evaluations, losses, gradients, memory and individual timing records.
`PUBLICATION_MANIFEST.json` binds evidence to hashes of the code actually run.

Do not interpret chance-level RT scores as distance extrapolation failure after
successful fitting: it did not fit the training evaluation. Attention's one
successful seed demonstrates task solvability, but its failed second seed
violates the frozen positive-control requirement. Neither successful solver
checks nor a formal global receptive field establish useful learned routing.

Timing scope: batch one, fixed large thin grids, float32 portable PyTorch
implementation, synchronized wall time including Python, two seeds. The
128-update NCA timing is a path-length cost reference; accuracy at 128 recurrent
updates was not qualified. The evidence does not establish the speed ceiling
of a fused transport kernel or a tuned attention backbone.

## Useful reviewer questions

1. Which missing training qualification would most cheaply distinguish sparse
   message dilution from generic optimization failure?
2. Is any claimed communication advantage confounded by target markers,
   coordinates, local evidence paths or differing parameter counts?
3. After a reliable fitting control, what intervention would isolate medium
   learning from message emission and reception?

Those were v1 review questions. A0 above supplies the registered paired
normalization experiment, but does not causally isolate the remaining failure.
