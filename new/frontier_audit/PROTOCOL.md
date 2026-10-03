# Original Streaming seed4: zero-training behavior audit v1

Frozen before measuring the new trajectories. Question: does this selected
successful checkpoint preserve solved output regions and extend correctness
through nearby open cells, or does its aggregate improvement conceal turnover?
Only original Streaming seed4 is authorized here. No failed seed, sidecar,
parameter noise, altered schedule, retraining, distillation or interventions.

## Model, input and replay qualification

Use the original stream_K8_seed4.pt (300 updates, K8) from the historical
Streaming Carry run. Verify checkpoint file SHA against its publication
manifest, parameter SHA against its original raw record, and all35 historical
source hashes against current files and the executed source snapshot.
Same32 maps each at size32/64, bank seeds40032/40064; tensor hashes must match
the frozen manifest. Cold initialization for original and source-flipped
inputs, same weights and historical FP32/backend settings. Freeze new source
hashes, snapshot executed code and record provenance in a new run directory.

One CPU metric smoke before one local CUDA audit. Hard cap300 seconds,
watchdog330; stop on source/replay drift, nonfinite state/logits/margins,
parameter/checkpoint mutation or a light-cone violation. No optimization,
backward, tuning, additional training maps or automatic follow-up.

## Trace and measurements

Record every macro step t=0..256, rather than inferring first passage from
eight-step samples. Retain original/flipped Boolean prediction correctness,
their intersection, and the paired signed margin
min((2*y-1)*logit, (2*y_flip-1)*flipped_logit). Boolean classification uses
the frozen >=0 threshold exactly; strict-positive margin is an auxiliary
quantity and does not redefine ties or the historical endpoint.

Analyze only the flipped connected component (changed), with open-mask
four-neighbor edges and source BFS distance. Verify the source at distance0,
no wrap/wall edges, and no correct counterfactual pair outside radius2*t.
At T64/128/256, reproduce the full original historical evaluation payload
at both sizes: six records, exact integers, float absolute tolerance1e-6
(BCE additionally relative1e-5). No behavioral interpretation until replay
passes. Accuracy/margins at t=0 are diagnostics, not training endpoints.

Primary behavioral summaries:

1. First correct time at each changed pixel; never correct is right-censored.
   Terminal-stable time means correct at every recorded step from that time
   THROUGH256; it is not infinite persistence. A temporary correct result
   followed by an error is distinguished from stable acquisition. Record
   first margin>0.1 separately as a fixed descriptive confidence threshold.
   Summarize per-map and exact BFS distance with denominator/censor counts;
   do not regress only acquired pixels as if censoring did not exist.
2. At t=8,16,...,256, count retained/lost/gained correctness for each adjacent
   eight-step pair, plus64->128,64->256,128->256. Report all changed pixels and
   strict16<d<32 separately. Retention and relapse are complementary, not
   independent tests. Also report whether ever-correct pixels later regress
   at ANY one-step observation; coarse endpoint retention can hide recovery.
3. Frontier acquisition: currently wrong pixels with at least one currently
   correct open one-hop neighbor, compared to wrong pixels without one.
   At starts0,8,...,248, acquisition means correct at the immediately next
   macro step t+1. Report opportunity/hit counts per map, time and EXACT BFS distance.
   Only strata containing both groups contribute to a matched descriptive
   difference; weight n_frontier*n_nonfrontier/(n_frontier+n_nonfrontier).
   Give eligible-stratum counts and per-map values. Empty groups are null.
   This association does not identify neighbor-to-pixel causal handoff:
   both may respond to prior latent propagation or common input. The cell
   also has a two-hop macro clock, so one-hop eligibility is not exhaustive.
4. If included, correct-region connectivity is defined by open graph edges
   to the changed component's source. Detached correct islands are behavior,
   not evidence that messages crossed walls or that computation is nonlocal.

Selected checkpoint n=1, inspected maps reused; pixels and time points are
correlated observations. Report per-map fractions plus pooled numerator/
denominator for description, no significance or population reliability.
Margin magnitude, output retention and spatial acquisition do not establish
hidden-state stability, an invariant latent manifold, optimization basin or
short-BPTT training causality. The prior architecture no-go verdicts remain.

## Delivery and interpretation

Save compact summary.json, RESULTS.md and descriptive figures first; raw
per-map/exact-distance behavior and compressed full traces are secondary.
Readability plots use map0 at each size, fixed before measurement. Do not
select a showcase map after seeing success. Preserve the original checkpoint
and all old outputs. Independent CPU saved-trace validation checks counts,
retention/gain accounting, first/stable times, margins, sources and replay
bindings; it does not rerun model inference.

The output is a behavioral diagnosis, not a new architecture GO gate.
A failure of a monotone-frontier story does not invalidate successful
computation; oscillatory or repeated-correction behavior remains possible.
Conversely, high retention/acquisition alone is not a proven flood-fill rule.
No causal message intervention or basin replication is authorized by this run.

From repository root, use new output names:

```
python new/frontier_audit/check_metrics.py
python new/frontier_audit/audit.py --out runs/NEW_SEED4_FRONTIER_AUDIT
python new/frontier_audit/validate.py --run runs/NEW_SEED4_FRONTIER_AUDIT --out analyses/NEW_SEED4_FRONTIER_VALIDATION
```
