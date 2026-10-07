# Incremental review: completed full-writer port-relation screen

- Review base: `d3a95be5230b48532bad51370c6d1d021e4e9775`.
- Stable code/evidence head: `196106541d7dae3c555cd141461109cd50817ccd`.
- This later commit updates review metadata only; the evidence head stays fixed.
- Execution/aggregation: **COMPLETE**,24/24 u300 trajectories,312/312 stages,
  624 size rows and624 complete packed Boolean banks; numerical failures0.
- Formal verdict: `NO_CONDITIONED_RELIABILITY_QUALIFICATION`.
- The separate earlier RRC-v0 partial snapshot remains INCOMPLETE; this update
  does not complete or reinterpret that experiment.

## Minimal reading order

1. [Results](evidence/port_relation_20261007_02/RESULTS.md),
   [summary](evidence/port_relation_20261007_02/summary.json), and
   [formal metrics](evidence/port_relation_20261007_02/final_metrics.csv).
2. [All stage metrics](evidence/port_relation_20261007_02/metrics.csv),
   [validation](evidence/port_relation_20261007_02/validation.json),
   [interruption/recovery](evidence/port_relation_20261007_02/interruption_and_recovery.json).
3. [Protocol](new/port_relation/PROTOCOL.md), [cell](new/port_relation/cells.py),
   [runner](new/port_relation/run.py), [reporter](new/port_relation/reporting.py).
4. [Reproduction](evidence/port_relation_20261007_02/REPRODUCTION.md),
   [engineering qualification](evidence/port_relation_20261007_02/runtime_qualification.json),
   [publication manifest](PORT_RELATION_PUBLICATION_MANIFEST.json).

Do not open the block ZIP archives or compressed dense records first. They
retain every recorded per-stage/per-map result, packed correctness trace,
relation-activity record, training curve and completion marker. Checkpoints and
private launch receipts stay local; checkpoint/source/bank/plan hashes remain.
The evidence package is about324.48MiB, compressed without dropping trace times.

## New decision-relevant evidence

Current/Constant/Conditioned are each0/8 joint-ready and0/8 old Full at u300.
Conditioned-current has8/8 pairs,0 wins,0 losses,net0,p=1; the all-tie contrast
does not establish equivalence. Size32 block-mean strict pooled T64 coverage is
.110424/.137410/.050948; retention is.548940/.429450/.475381, respectively.
Constant is ever-ready1/8 and loses readiness; the other arms are never ready
on the saved checkpoint grid. Intermediate success cannot select an endpoint.

Both candidate arms retain the historical full writer and add only a learned
port-matrix action:16 constant entries versus144 Z-conditioned coefficients,
initialized to zero. Counts5033/5049/5177 are not matched. Primary is only
Conditioned-current; other contrasts/dense stages/size64/old Full are secondary.
Runtime ID `port_relation_v1_full_writer_native_k8` retains the frozen protocol
header alias `port_relation_conditioned_v1`. See reproduction for the explicit
32-backward-window/one-optimizer-step implementation of the training wording.

The parent remains ERROR/INCOMPLETE after2 trajectories/33 stages: diagnostic
float32 squaring overflowed after finite state/logit checks. The repaired child
inherits33 stages, restores conditioned u150 model/Adam/RNG and replays unsaved
work. Qualified cached evaluation preserves full u175/u150 report dictionaries
and packed traces, including live parameter reload; eager/captured gradients
match exactly in the two-update qualification. Backend nondeterminism means
this is not a bitwise uninterrupted-replay claim. No new experiment was launched
for publication. Saved-data verification and CPU cell/activity/report checks pass.

## Reviewer questions and claim boundary

Does retaining the original full writer isolate the tested port-relation
addition sufficiently? What can be inferred from all arms failing the endpoint,
given low continuous reach/retention and a transient Constant success?
Does the preserved recovery bridge change interpretation beyond the stated
nondeterministic-replay limit? Review raw activity only for a specific question.

The frozen candidate failed this finite reliability screen. The evidence does
not establish a general relation-algebra failure, a unique formation mechanism,
a population success rate, or a solution to short-BPTT credit assignment.
