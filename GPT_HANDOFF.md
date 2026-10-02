# Incremental review: exactly nested stationary sidecar

- Review base: `723dd4aae6dc082ee998dd647b9df9a49b58962f`.
- Evidence head: `33ced27d9f4966f6aa776e53cd521b9f5c98cbba`.
- This following handoff-only commit changes review metadata, not code or evidence.

All12 arms completed300 updates in1676.281 seconds. Four original Streaming
controls reproduce historical final parameter hashes and full evaluations
exactly. Frozen scientific status: **DEVELOPMENT_NO_GO**. Reach+hold is
stream1/4, carried H0/4, instantaneous side branch0/4.
This is a2D development comparison on inspected seeds2/3/4/5 and reused maps.

## Minimal reading order

1. [Current results](evidence/stationary_sidecar_init2345/RESULTS.md) and
   [compact analysis](evidence/stationary_sidecar_init2345/analysis.json).
   Start with primary_rows, seed4_trajectory, primary_trajectories and
   hold_failure_reasons; raw per-map JSON is secondary.
2. [Frozen protocol](new/stationary_sidecar/PROTOCOL.md),
   [exact sidecar cell](new/stationary_sidecar/sidecar_cells.py),
   [architecture](ARCHITECTURE.md), and
   [runner/gates](new/stationary_sidecar/run.py).
3. [Saved-result validation](evidence/stationary_sidecar_init2345/validation.json),
   [CPU qualification](evidence/stationary_sidecar_init2345/cpu_validation.json),
   [provenance](evidence/stationary_sidecar_init2345/provenance.json), and
   [publication bindings](STATIONARY_SIDECAR_PUBLICATION_MANIFEST.json).
   The analyzer is [CPU-only saved arithmetic](new/stationary_sidecar/analyze.py).
4. [All curves](evidence/stationary_sidecar_init2345/curves.csv),
   [paired memory-minus-controls effects](evidence/stationary_sidecar_init2345/paired_effects.csv),
   and12 [raw arm records](evidence/stationary_sidecar_init2345/raw/).

## Changed evidence and claim

The original W24/Z8, F/Q, operators and clocks are retained. G67->32->12
writes extra H12; zero-initialized P_F/P_Q feed it into the old residuals.
Memory retains H; stateless consumes the same instantaneous write and resets
only H. Side arms have identical7989 parameters and full initialization;
stream has5033. Neutral projected W/Z/logits/core derivatives are verified,
but extra gradients can change joint norm clipping and optimizer updates.
All side paths open by update3; K8 retains the16-hop graph-radius upper bound.

| Seed4 primary pooled | T64 | T128 | T256 |
|---|---:|---:|---:|
| Original Streaming | 85.92% | 99.35% | 100.00% |
| Persistent H | 94.24% | 0.00% | 0.38% |
| Stateless side branch | 100.00% | 100.00% | 88.49% |

Only original seed4 passes hold. Memory loses all2918 primary paired hits
atT128. Stateless seed4 drops to BA78.06/76.13% atT256 and also fails paired
hold. Stateless seed2 improves at later horizons but misses the frozen T64
threshold; later gains do not change its gate.

The carried-sidecar recipe did not improve reliable continued computation
on these four seeds. Carry simultaneously changes temporal accumulation and
activation scale; this does not isolate a scale-independent memory mechanism.
Old Z already provides stationary memory. No general memory rejection,
Jacobian/overwrite mechanism, population reliability, fresh confirmation,
FLOP equality, trained stability guarantee or3D claim is established.

## Unchanged claims and validation

The selected-seed4 operator audit remains an intervention-sensitivity result.
Earlier architecture no-go decisions and positive original seed4 behavior
remain unchanged; their frozen files and checkpoints were preserved.

Saved-result verification passed45 source bindings/current and both snapshots,
12 CPU checkpoint hashes,4 exact historical controls, banks/schedules/preflight,
144 BA means,1008 paired aggregates with independent denominators,936 curve
rows,624 contrasts and all frozen decisions. The focused CPU qualification
was rerun, including its documented tiny untrained-model smoke. There was no
new scientific training or trained-checkpoint inference for publication.
Raw arm/schedule/aggregate/CSV files are byte-identical; metadata copies remove
machine identifiers. Checkpoints and launch receipts remain local.

## Concrete reviewer questions

1. Do the zero-feedback nesting, shared core draws, immediate stateless
   consumption and unchanged clocks implement the stated carry contrast?
2. Do the frozen reach/hold rules explain why both seed4 side arms fail,
   despite higher T64 scores and stateless T128 perfection?
3. Are claim boundaries sufficient to separate this failed carried-sidecar
   recipe from generic local memory, activation-scale effects and identified
   long-rollout failure mechanisms?

Do not open the large raw files first or infer a new architecture proposal
from this negative screen.
