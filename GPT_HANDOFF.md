# Incremental review: original Streaming seed4 behavioral audit

- Review base: `2e550abfeb1bf53e1348dae7f5653fa6ccd8e47b`.
- Evidence head: `af99c9ecf83391bac25150a2ffba71a928b88273`.
- This following handoff-only commit changes review metadata, not code or evidence.

Zero training, one selected original checkpoint,32 historical maps per
size32/64, every macro step0..256. Completed in31.813 seconds. All six full
historical endpoint payloads replay exactly. Earlier architecture no-go
decisions, including the stationary-sidecar screen, remain unchanged.

## Minimal reading order

1. [Current report and figures](evidence/frontier_audit_seed4/RESULTS.md),
   [diagnosis](evidence/frontier_audit_seed4/DIAGNOSIS.md), and
   [compact summary](evidence/frontier_audit_seed4/summary.json).
   Begin with sizes.*.transitions, ever_regressed_after_being_correct,
   frontier, distance_profile and endpoints. Timing medians condition on
   acquisition; censor counts are present beside them.
2. [Frozen protocol](new/frontier_audit/PROTOCOL.md),
   [trajectory/replay runner](new/frontier_audit/audit.py),
   [per-map metrics](new/frontier_audit/metrics.py), and
   [independent saved-trace validator](new/frontier_audit/validate.py).
3. [Full local-trace validation record](evidence/frontier_audit_seed4/validation.json),
   [public arithmetic validation](evidence/frontier_audit_seed4/publication_validation.json),
   [provenance](evidence/frontier_audit_seed4/provenance.json), and
   [publication bindings](FRONTIER_AUDIT_PUBLICATION_MANIFEST.json).
4. [Size32 matched counts](evidence/frontier_audit_seed4/frontier_matches_size32.csv)
   and [size64 matched counts](evidence/frontier_audit_seed4/frontier_matches_size64.csv).
   Verify exported effects without checkpoints using
   `python tools/export_frontier_audit.py --verify-only`.

## Changed evidence

| Changed-component behavior | size32 | size64 |
|---|---:|---:|
| Correct-at64 retention at256 | 6856/6856 (100%) | 18827/19084 (98.65%) |
| Wrong at64, correct at256 | 1446 | 13275 |
| Ever-correct pixels with any regression0..256 | 186/8303 (2.24%) | 2768/33328 (8.31%) |
| Never correct through256 | 392/8695 | 10860/44188 |
| Matched frontier one-step acquisition difference | +30.92 pp | +22.44 pp |

Retention at endpoints is not per-step monotonicity: one size32 pixel correct
at64 briefly errs and recovers by256. Matched acquisition compares currently
wrong pixels with versus without a correct open one-hop neighbor at starts
0,8,...,248, using t->t+1. Only common map/time/exact-BFS-distance strata
contribute, weighted within maps by nf*nn/(nf+nn), then equal-map averaged.
There are1081/5801 common strata and31/32 /32/32 eligible maps respectively.

The selected solution exhibits mostly retained paired correctness and local
acquisition in the observed interval. This is output behavior, not proof of
causal neighbor handoff, flood-fill, invariant latent closure or bounded state.
A macro step can use two graph hops; neighboring outputs can respond to a
common latent signal. Model n=1, maps reused, terminal stability ends at256.
No resolution scaling law, training basin or cross-checkpoint reliability
claim follows. Size64 distant regions remain partly unsolved.

## Validation and unchanged material

The local CPU check verified45 source bindings,64 maps/BFS,2240 transitions,
123296 frontier strata and192 paired aggregates. Replay compares5628 integer
and2636 floating leaves, maximum error0. Five synthetic metric fixtures pass.
The public checker recomputes all6882 matched CSV strata,64 map effects,
64 start intervals,140 transition accounts and18 endpoint aggregates.

Original training/cell code, checkpoint and frozen evidence were preserved.
Summary removes only redundant per-map transition rows; first/stable per-map
and exact-distance profiles remain. Full traces and large behavior/evaluation
JSON remain local, with hashes in provenance. Manifest/completion copies remove
machine identifiers. Full saved-trace validation was done locally; exported
arithmetic verification is a narrower check. No model inference or training
was performed to prepare this upload.

## Concrete reviewer questions

1. Does the retained/gained/ever-regressed accounting justify describing this
   selected solution as mostly retaining solved outputs while acquiring regions?
2. Is the map/time/exact-distance matching implemented correctly, and are the
   causal and two-hop limitations of its acquisition association stated clearly?
3. Do censor counts, finite horizon, incomplete distant regions and checkpoint
   selection adequately limit any algorithm or trainability interpretation?

No new architecture or follow-up experiment is proposed in this delivery.
