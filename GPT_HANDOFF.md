# Incremental review: completed addressed-delta development screen

- Review base: `f080e69a79451fd6341e3b9fa3e5a23070d863b5`.
- Stable code/evidence head: `7a5ee01886c1a8d214625bb278217f4d10a4056a`.
- This later commit changes review metadata only; the evidence head stays fixed.
- Execution and aggregation: **COMPLETE**,12/12 u300 endpoints,156/156 stages.
- Frozen verdict: `NO_DEVELOPMENTAL_SIGNAL`.

## Read this update first

1. [Results](evidence/addressed_delta_20261008_02/RESULTS.md),
   [summary](evidence/addressed_delta_20261008_02/summary.json), and
   [formal metrics](evidence/addressed_delta_20261008_02/final_metrics.csv).
2. [All-stage metrics](evidence/addressed_delta_20261008_02/metrics.csv),
   [saved-data validation](evidence/addressed_delta_20261008_02/validation.json),
   [recovery](evidence/addressed_delta_20261008_02/interruption_and_recovery.json).
3. [Protocol](new/addressed_delta/PROTOCOL.md), [cells](new/addressed_delta/cells.py),
   [writer](new/addressed_delta/writer.py), [source map](ARCHITECTURE.md).
4. [Reproduction](evidence/addressed_delta_20261008_02/REPRODUCTION.md),
   [qualification](evidence/addressed_delta_20261008_02/runtime_qualification.json),
   [publication manifest](ADDRESSED_DELTA_PUBLICATION_MANIFEST.json).

Do not open the four raw block archives first. The approximately3.09MiB evidence
package retains all9,984 per-map metric rows,156 evaluations and completion
markers,12 training curves, banks and schedules, with lossless compression.
Checkpoint contents and private process metadata stay local. Full Boolean
trajectories were not persisted; old Full was not evaluated.

## New evidence and claim boundary

At fixed u300, Current is joint-ready2/4, Additive0/4, Delta0/4. Neither candidate
is ever-ready on the saved checkpoint grid. Primary Delta-minus-Additive has
0 wins,0 losses,4 ties and descriptive p=1; ties do not establish equivalence.
Size32 block-mean strict pooled T64 coverage is.5292/.0992/.1826; retention is
.9657/.0573/.1614 (Current/Additive/Delta). Four blocks are developmental only.

Candidates replace the entire carrier writer with matched initially identical
K/V heads: C'=U+.1K^T V versus C'=U+.1K^T(V-KU), K=A/sqrt(1+||A||_F^2).
They retain C24/Z8, E/Q/readout, transport, pre-stream perception and K8 training.
Current5033 and candidates5689 parameters are not capacity matched. The feedback
term includes structured damping; this is not a semantic duplicate proof or
a general failure theorem for addressed writing.

The cancelled parent retained Current u300 and Additive u200. After one failed
pre-training restart, the child inherited22 stages and resumed Additive u200
model/Adam/RNG with a JSON tuple/list compatibility fix. Numerical training source
was unchanged; helper source is separately bound. Recovered process time is
2570.515seconds. Nondeterminism prevents an uninterrupted bitwise-replay claim.
Qualification bound71 source files individually; the protocol's integrated
qualification-time ZIP was not generated. The public record states this deviation.

The saved-data exporter verifies all source/bank/archive hashes, records,
per-map count-to-aggregate agreement, gates, report tables and marker/curve
bindings, with no model inference or optimizer updates. CPU primitive/cell/report
checks pass; publication launched no new task training campaign.

Reviewer questions: What does the matched zero-success primary contrast establish,
given the positive contextual Current results? Can writer replacement be cleanly
distinguished from the feedback subtraction using these controls? Previous
port-relation and other frozen conclusions below remain unchanged; skip them
unless comparing experiments.

---

## Previous handoff: full-writer port-relation screen

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
