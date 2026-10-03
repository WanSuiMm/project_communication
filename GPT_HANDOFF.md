# Incremental review: seed4 dense update100--200 audit

- Review base: `e142f224ad03f9bde9894dfcf80cf41411744f6e`.
- Evidence head: `04f9e774ed11b74038b00171157dc8225398cb65`.
- The following handoff-only commit changes reader metadata, not code or evidence.

One historical seed4 training trajectory was replayed with the original
StreamingCell, K8 helper, data bank and batch schedule unchanged. All 300
updates and 44 offline paired audits completed. Checkpoints/maps/cells/times
are nested observations, not independent training replications.

## Minimal reading order

1. [Latest results](evidence/transition_20261003/RESULTS.md),
   [compact summary](evidence/transition_20261003/summary.json), and
   [complete 44-row profiles](evidence/transition_20261003/profiles.csv).
2. [Frozen protocol](new/transition_100_200/PROTOCOL.md),
   [executed runner](new/transition_100_200/run.py),
   [trace metrics](new/transition_100_200/metrics.py), and
   [GPT_CONTEXT.md](GPT_CONTEXT.md) for exact symbols.
   Architecture and [K8 training helper](new/short_bptt/training.py) are unchanged.
3. [Saved-artifact validation](evidence/transition_20261003/validation/local_validation.json),
   [publication bindings](TRANSITION_PUBLICATION_MANIFEST.json), and
   [excluded-artifact provenance](evidence/transition_20261003/input_provenance.json).
4. [Reproduction notes](evidence/transition_20261003/REPRODUCTION.md).
   Public verification uses
   `python tools/export_transition_evidence.py --verify-only`.
   Public-reference CPU qualification uses
   `python tools/replay_transition_public.py --check --out analyses/NEW_PUBLIC_TRANSITION_CPU.json`.

Begin with the small summary and profiles. Distance/age tables and 44
per-checkpoint JSON files are secondary. Original checkpoints, raw NPZ traces,
source snapshots and private execution receipts remain local, with hashes bound.

## New decision-relevant evidence

The dense sequence saves updates 100,105,...,200; update 300 is a separate
endpoint control. Both fixed audit banks contain 16 paired maps, rolled out
through step 256. Size 32 is primary; size 64 is a secondary projection.

Selected size32 pooled profiles (rates, not percentages):

| Update | Strict T128 | G64->128 | First-exit64->128 | Continuous survival64->256 |
|---:|---:|---:|---:|---:|
| 120 | 0.1249 | 0.1825 | 0.0030 | 0.9964 |
| 180 | 0.1026 | 0.0287 | 0.5231 | 0.3566 |
| 195 | 0.8565 | 0.4037 | 0.0515 | 0.9453 |
| 200 | 0.9755 | 0.5316 | 0.0003 | 0.9997 |
| 300 | 0.9978 | 0.6143 | 0.0003 | 0.9997 |

Retention can be high while reach is low at 120/125. Update 180 shows a real
collapse after stronger profiles at 175. The recovery at 195->200 sharply
improves preservation: first-entry survival at lag 64 also rises 0.8250->0.9531,
with changing eligible/censored denominators retained.

Only 200 passes the prespecified size32 screen within the dense window.
The 300 control also passes, but 205/210 are unmeasured. Three consecutive
five-update passes are therefore unestablished; operational_onset=null
does not establish either persistent onset or architecture failure.

G is acquisition conditional on being wrong at 64. First-exit destruction
counts any intervening wrong step; endpoint destruction is different.
At 200, one initially correct cell briefly becomes wrong and recovers.
Exact netgain uses weighted fluxes, not G-D.

The final-correct-conditioned lag median is 0 at 120,180 and200 despite
very different coverage: at 180 only 94/1345 strict-cohort cells finish correct
on one eligible map. This statistic alone cannot identify commitment.

## Validation and unchanged claims

Saved-artifact CPU validation passes 451 checks: 67 live source and 67 snapshot
bindings, 23 checkpoints, 44 raw trace hashes and behavior recomputations,
exact anchors 0/100/200/300, six historical endpoint payloads and three prior
stage summaries. Counts match exactly; maximum numeric error is 6.38e-8
within 2e-7. Recorded state RMS/BCE are finite, but hidden-state tensors and
separate logits were not saved, so those quantities cannot be rebuilt.

The first validator report is retained locally with its SHA. An absent-field
lookup in an older public manifest and a check-status serialization collision
were repaired. All 44 metric recomputations come from the first full CPU pass;
only cheap bindings/completion serialization were finalized afterward.
The validation report distinguishes the full-pass and finalizer source hashes.

An isolated checkout of the staged repository passes public verification and
public-reference CPU qualification without private archives. Publication
runs no new GPU training or inference. The public wrapper compares new weights
to published parameter hashes and actual public stage counts; it explicitly
does not reverify excluded original checkpoint file bytes.

This supplies a finite-window behavioral change in one selected trajectory.
It does not identify physical phase transition, latent commitment, contextual
type equivalence, causal handoff, infinite-horizon closure, population
reliability or 3D behavior. All earlier architecture and warm-start negative
decisions remain unchanged.

## Concrete reviewer questions

1. Does the evidence support a nonmonotone reach/preservation trajectory,
   rather than one clean first onset of an invariant representation?
2. What observations or context interventions would distinguish reusable
   downstream state from finite-horizon prediction correctness? Persistence
   at 205/210 and independent-seed reliability remain untested.
