# Incremental review: Streaming Carry development screen

- Review base: `e4a8574689ca8f9989d9ef14ce2270416230e7a7`.
- Evidence head: `df4aed4e86f7e4619ec443815ef0544b9bda6847`.
- This following handoff-only commit changes review metadata, not code or evidence.

All8 new matched K8 arms completed300 updates in828.297 seconds (13.80 minutes).
**DEVELOPMENT_NO_GO**: baseline reaches AND holds in2/4 seeds; stream in1/4.
Every baseline final parameter hash and full evaluation payload exactly
reproduces Phase II. Seeds2-5 and these maps were already inspected; this is
development evidence, not fresh confirmation.

## Minimal reading order

1. [Current result and boundary](RESULTS.md) and
   [complete screen report](evidence/streaming_carry_init2345/RESULTS.md).
2. [Compact metrics](evidence/streaming_carry_init2345/analysis.json),
   [distance/rollout curves](evidence/streaming_carry_init2345/curves.csv), and
   [paired contrasts](evidence/streaming_carry_init2345/paired_effects.csv).
3. [Frozen protocol](new/streaming_carry/PROTOCOL.md),
   [streaming cell](new/streaming_carry/stream_cells.py),
   [runner/gates](new/streaming_carry/run.py), and
   [architecture/feature clock](ARCHITECTURE.md).
4. [CPU checks](new/streaming_carry/check.py),
   [saved-result analysis](new/streaming_carry/analyze.py),
   [validation](evidence/streaming_carry_init2345/validation.json), and
   [publication bindings](STREAMING_CARRY_PUBLICATION_MANIFEST.json).

The8 [raw arm JSONs](evidence/streaming_carry_init2345/raw/) are secondary.
Raw arms, shared schedule, aggregate and CSVs are byte-identical copies.
Public manifest/completion/preflight copies omit PID metadata. Original
records, checkpoints, machine receipts and transient logs remain local.

## Decision-relevant change

W24 becomes four6-channel directional lanes; Z8 remains stationary local
state. Fixed T moves through open links and reverses lanes at blocked links;
wall ports stay fixed. It is a permutation, with inverse B T B and T^T T=I.
The candidate uses W'=T(W)+0.1F(T(W),Z,L(W),L(Z),X), then the unchanged
additive Z update. F receives incoming carriers but OLD Laplacian features;
Q sees updated W. This preserves the two-hop macro-step bound. F's first
feature also changes, so this tests the specified recipe, not T's isolated
causal effect. Parameters (5033), initialization, data/schedule and K8 clocks
are matched. Fixed-operator losslessness does not protect the learned cell.

Primary is size32/T64 STRICT16<d<32: mean AND pooled paired correctness>=80%,
both original/flipped BA>=85%. Hold limits declines at BOTH T128 and T256.
Continuation also requires>=3 stream reach+hold seeds, retention of2/5 and
improvement over the concurrent baseline count. Those conditions fail.

| Seed | Baseline primary mean / pooled % | Stream primary mean / pooled % |
|---|---:|---:|
| 2 | 93.70 / 92.97 | 1.08 / 1.47 |
| 3 | 21.85 / 17.27 | 8.28 / 5.62 |
| 4 | 52.40 / 44.76 | 88.89 / 85.92 |
| 5 | 96.45 / 95.68 | 16.05 / 13.02 |

Seed4 is a sustained positive case: primary mean/pooled at size32 rises to
99.64/99.35% atT128 and100/100% atT256. On size64, d>32 mean/pooled rises
26.32/12.48 ->55.78/38.60 ->74.99/59.43%. Size64 is descriptive and cannot
replace the primary gate. Stream loses original successful seeds2/5; seed3
also fails. Seed5's hold=True preserves a weak result, not task success.
Seed2/3 long-rollout deterioration is observed, without a causal diagnosis.

Unlike the prior average-carry seed4 transient, this new seed4 case improves
throughT256. It remains one initialization: useful positive evidence inside
an overall failed development screen, not a robust architecture result.
n=4 is conditional on one training bank/schedule. No significance, broad
distance-scaling, arbitrary routing, warm-editing or3D claim follows.

Verification covers35 source bindings,8 saved checkpoint parameter hashes,
96 BA and672 paired aggregates/denominators,8 decisions,312 effects and936
CSV rows. Publication also checks59 source/evidence index hashes and228
Markdown links. No new training or model inference was used for publication.

Unchanged: previous average-carry DEVELOPMENT_NO_GO, Phase-II/Phase-I gates
and original positive K8 cases, revision/source-switch/dynamics conclusions
and earlier transport failures. No automatic confirmation, architecture
rescue, sweep or monitor is scheduled.

## Questions for review

1. Does the implementation satisfy the port bijection, blocked-link bounce
   and stated incoming-versus-old feature clock without adding a third hop?
2. Are baseline exact reproduction, loss of seeds2/5 and seed4's sustained
   narrow and farther gains all represented with both mean and pooled values?
3. Is the failed continuation decision kept separate from the one positive
   case, without treating pure T isometry as semantic or nonlinear stability?
