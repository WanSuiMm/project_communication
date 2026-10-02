# Incremental review: local persistent state and ephemeral interface

- Review base: `cdb00c9c33cf1a04139da627fff36767d636343f`.
- Evidence head: `697cd2f7783fdcf1951c2ba93532ac8e4c6766c2`.
- This following handoff-only commit changes review metadata, not code or evidence.

All12 matched K8 arms completed300 updates in1281.532 seconds (21.36 minutes).
**DEVELOPMENT_NO_GO**: reach+hold counts are baseline2/4, stream1/4,
interface0/4. All8 control final parameter hashes and complete evaluation
payloads reproduce their historical records exactly. This is development
evidence on previously inspected seeds/maps, conditional on one bank/schedule.
Rollup eligible-map counts sum repeated seed evaluations; the independent
unit remains initialization seed, n=4.

## Minimal reading order

1. [Current results](evidence/local_interface_init2345/RESULTS.md) and
   [compact analysis](evidence/local_interface_init2345/analysis.json).
2. [Frozen protocol](new/local_interface/PROTOCOL.md),
   [cell equations and shapes](ARCHITECTURE.md), and
   [exact cell](new/local_interface/interface_cells.py).
3. [Runner and decision](new/local_interface/run.py),
   [validation](evidence/local_interface_init2345/validation.json), and
   [publication bindings](LOCAL_INTERFACE_PUBLICATION_MANIFEST.json).
4. For detailed arithmetic, use [curves](evidence/local_interface_init2345/curves.csv),
   [paired effects](evidence/local_interface_init2345/paired_effects.csv),
   [CPU checks](new/local_interface/check.py), and
   [saved-result audit](new/local_interface/analyze.py).

The12 [raw arm JSONs](evidence/local_interface_init2345/raw/) are secondary.
Original outputs and checkpoints remain local; public manifest/completion/
preflight copies omit private process metadata. Large logs and machine
launch receipts are excluded.

## Decision-relevant change

The candidate retains both W24 and Z8 locally. A shared affine E35->24 emits
four6-channel directional messages before each of two residual phases.
Only these messages cross masked open edges using the existing fixed port
permutation. W'=W+0.1F(W,Z,T(E(W,Z,X)),X); then
Z'=Z+0.5Q(W',Z,T(E(W',Z,X)),X). Messages are newly emitted each phase and are
not a persistent carrier state. F59->31->24 and Q59->21->8 give5033 parameters.

Encoder/readout initial draws match both controls. Complete candidate
initialization differs. Raw Laplacian/neighbor inputs are removed, and hidden
widths/state allocation change with the communication interface. This tests
the complete parameterization, not an isolated effect of role separation.
Only fixed T is lossless; encoding, receive/store/re-emit and learned residual
updates have no losslessness or stability guarantee.

Primary size32/T64 strict16<d<32 pooled paired correctness:

| Seed | Baseline % | Stream % | Interface % |
|---:|---:|---:|---:|
| 2 | 92.97 | 1.47 | 0.00 |
| 3 | 17.27 | 5.62 | 0.00 |
| 4 | 44.76 | 85.92 | 0.00 |
| 5 | 95.68 | 13.02 | 0.00 |

Both interface mean and pooled metrics are zero atT64/T128/T256:0/2918
paired-correct pixels in each seed/horizon. Size64 d>32 also has0/27436 hits
for every interface seed/horizon. Paired correctness requires the original
and flipped cue to be answered correctly at the same changed-region pixel;
these zero values are not whole-grid ordinary accuracy. Interface seed4's
Hold=True preserves a failed endpoint, not successful remote computation.

The frozen gate requires both paired statistics>=80%, both BAs>=85%, hold
atT128 AND T256, at least3 interface reach+hold seeds including2/5, and a
count exceeding both controls. The candidate fails reach in all4 seeds.
Farther or later results do not rescue the endpoint.

Changed: this specific ephemeral-interface recipe has a completed negative
development result. Unchanged: Streaming Carry's overall no-go and sustained
seed4 positive case, Direct Spatial Carry's no-go, Phase-II/Phase-I decisions,
and earlier revision, dynamics and transport findings. The new result does
not identify a failure mechanism or refute every local-state/interface design.
No significance, broad reliability, arbitrary horizon or3D claim follows.

Publication uses saved-result arithmetic/provenance checks and CPU checkpoint
hash verification. No new training or checkpoint inference was performed.

## Questions for review

1. Does the cell implement local identity for both persistent blocks and
   ephemeral messages in both phases, without hidden direct-neighbor inputs?
2. Are initialization matching, all8 exact control reproductions, integer
   denominators, null bands and the frozen reach/hold decision represented
   consistently across summaries, raw records and CSVs?
3. Is the specific negative result kept separate from the broader design
   principle and the previous persistent-stream seed4 positive case?
