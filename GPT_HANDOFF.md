# Incremental review: detached warm-start of original StreamingCell

- Review base: `91be5bcb957ea9c33399e06201ada65cc3132732`.
- Evidence head: `f296f67909116aacbcfc8aa60122cc6053297510`.
- The following handoff-only commit changes reader metadata, not code or evidence.

The completed 2D screen asks whether older detached on-policy starting states
make the existing solve-and-preserve phenotype easier to learn. Original
StreamingCell and K8 are unchanged. Eight arms completed300 updates each in
1629.375 seconds (27.16 minutes). Decision: **DEVELOPMENT_NOT_QUALIFIED**.

## Minimal reading order

1. [Current results and all-arm partial metrics](evidence/warmstart_20261003/RESULTS.md)
   and [compact aggregate](evidence/warmstart_20261003/summary.json).
2. [Frozen protocol and runtime amendment](new/warmstart/PROTOCOL.md),
   [training helper](new/warmstart/training.py),
   [paired runner](new/warmstart/run.py), and
   [unchanged phenotype](new/seed4_followup/phenotype.py).
   [GPT_CONTEXT.md](GPT_CONTEXT.md) maps exact source symbols.
3. [Saved-artifact validation](evidence/warmstart_20261003/validation/local_validation.json),
   [public arithmetic](evidence/warmstart_20261003/publication_validation.json),
   [provenance](evidence/warmstart_20261003/provenance.json), and
   [publication bindings](WARMSTART_PUBLICATION_MANIFEST.json).
4. [Reproduction notes](evidence/warmstart_20261003/REPRODUCTION.md).
   Verify without a model/GPU with
   `python tools/export_warmstart.py --verify-only`.

Complete arm summaries retain all gates and map/distance profiles. Raw records,
matched-frontier CSVs and training-stage turnover rows are secondary; do not
open them first. Checkpoints, NPZ traces, source snapshots, full diagnostic
JSON and private execution receipts remain local, with hashes bound.

## Intervention and primary result

Four paired initialization seeds2/3/4/5 share one fixed training bank10002 and
historical batch schedule20002. Both variants compute the same no-grad prefix
for the last four examples. Baseline discards it and invokes the untouched
historical K8 helper; warmstart keeps the detached prefix state, with the first
four examples fresh. Both use64 trained suffix steps and eight K8 task losses.
Frozen prefix ages32/64/128/192 share seed60002; no state crosses optimizer
updates. Older supervised state ages and detached encoder history are part of
this intervention, so it does not isolate a pure credit-assignment effect.

Fresh primary evaluation uses32 maps per size32/64 at seeds60032/60064.
All four historical baseline initial/final parameters and all24 saved
size-by-horizon records exactly match their published originals. Baseline
seed4 also passes the fresh full phenotype, qualifying the control.

| Initialization | Baseline full phenotype | Warmstart full phenotype |
|---:|---|---|
| 2 | Fail | Fail |
| 3 | Fail | Fail |
| 4 | Pass | Fail |
| 5 | Fail | Fail |

The frozen positive criterion required warmstart>=3/4 AND at least two more
passes than baseline, with qualified controls. Actual passes are1/4 versus0/4.
Independent unit: paired initialization, n=4 conditional on one bank/schedule.
No pixel/time-step significance, population reliability or3D claim follows.

Partial changes are nonuniform. Warmstart seed2 improves some output
retention, coverage growth and open-grid BA, but misses the full gate.
Warmstart seed4 loses reach and retention; its size32 strict pooled
T64/T128/T256 is46.96/40.75/1.49%, compared with85.12/99.84/100% baseline.
The report retains these partial metrics; they cannot rescue the full gate.
The negative decision applies to this recipe, not all warm-start methods.

## Validation, exploratory diagnostics and unchanged evidence

Saved-artifact CPU validation passes583 checks:70 current source/reference
bindings and70 snapshots,8 final plus32 stage checkpoints,24 exact saved
historical records,8 recomputed fresh phenotype gates from16 Boolean trace
files,32 stage summaries,4096 integer turnover identities and384 finite
recorded RMS values. Numeric phenotype comparisons allow2e-7 for FP32/NumPy
roundoff; counts, denominators, thresholds and gate booleans match exactly.
Historical saved replay records require exact numeric equality and recorded
maximum error0.

The first publication-checker attempt is preserved locally. Its false failures
came from comparing formatted floating reason strings and confusing
start-intersection-end retention with total endpoint correctness. The checker
was corrected; experimental outputs and frozen gates were unchanged.

Stage diagnostics at updates0/100/200/300 use16 separate size32 maps(seed61032)
for128 paired steps. These are exploratory. Their raw Boolean trajectories
were not saved: recorded accounting, denominators, checkpoint bindings and
RMS finiteness are verified, but full stage survival cannot be independently
replayed from raw traces. This does not establish an order parameter,
computational phase transition, attractor or latent continuation closure.

Public arithmetic verifies8 full gate summaries,114952 matched strata,
32 stage snapshots and4128 turnover rows (including32 initial rows), together
with all source/tool/evidence hashes. An isolated checkout of the staged
standalone repository passes both public verification and the focused CPU
qualification without private run archives. No GPU training or inference was
performed to prepare publication.

Before formal training, the user explicitly removed the runtime limit.
Preflight01's failed former timing gate remains recorded. CPU02/preflight02
qualified the amended source, including the sticky prefix-finiteness
synchronization check. Scientific conditions stayed fixed. All earlier
architecture decisions and evidence remain unchanged.

## Concrete reviewer questions

1. Which frozen gate components explain each paired failure, especially the
   different seed2 and seed4 changes?
2. Does this screen support only the insufficiency of the sampled age-exposure
   recipe, rather than a general claim about warm-start or continuation learning?
3. What descriptive training-stage behavior is supported by the saved summaries,
   and which proposed phase/latent-state mechanisms remain unidentified?
