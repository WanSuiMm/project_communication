# Original seed4: retained correctness and local acquisition

The zero-training audit supports a concrete behavioral observation: this
selected checkpoint usually retains solved outputs while acquiring new ones.
It does not yet identify the recurrent algorithm or explain why training finds
this solution at seed4 but not reliably at other seeds.

Start with the [frozen report](RESULTS.md),
[compact summary](summary.json),
then [independent CPU validation](validation.json). No original files were
changed. There was one model rollout audit, no new training, and no parameter
or operator intervention. Runtime31.813 seconds, versus the frozen300-second
cap. All six full historical T64/128/256 payloads replayed with error0.

## What the paired trajectories establish

Correctness requires the same pixel to be classified correctly for BOTH the
original input and the source-value-flipped input. A single unchanged guess
cannot satisfy this endpoint on the flipped component.

|Changed-component behavior|size32|size64|
|---|---:|---:|
|Correct at64|6856|19084|
|Of these, still correct at256|6856 (100%)|18827 (98.65%)|
|Wrong at64, correct at256|1446|13275|
|Ever-correct pixels that regress at any step0..256|186/8303 (2.24%)|2768/33328 (8.31%)|
|Never correct through256|392/8695|10860/44188|
|Final wrong at256|393/8695|12086/44188|

Thus endpoint improvement is largely acquisition with retained prior results.
There is still regression, and much of the distant component remains unsolved
at size64. High primary-band accuracy cannot be substituted for full-component
completion.

The paired first-correct and terminal-stable medians mostly overlap in the
distance profiles. Those medians include acquired pixels only; the figure also
shows final-correct coverage at each distance. They are not an estimate of a
resolution scaling law and do not remove right censoring.

## A local acquisition association

At t=0,8,...,248, currently wrong pixels with a correct open one-hop neighbor
are more likely to be correct at t+1. Comparison is within the SAME map,
time and exact source BFS distance. Only strata with both frontier and
nonfrontier opportunities contribute, weighted within maps; maps receive
equal weight in the descriptive mean.

|Size|Eligible maps|Common strata|Mean matched acquisition difference|
|---|---:|---:|---:|
|32|31/32|1081|+30.92 percentage points|
|64|32/32|5801|+22.44 percentage points|

This is more specific than aggregate accuracy growth. It is still an
association: a common latent signal can cause the neighboring output and the
next pixel's output. Each macro step has a two-hop upper bound, so a correct
one-hop output neighbor is neither the only possible input nor a proven cause.
There are no significance tests or independent-pixel claims.

## Endpoint retention can conceal a brief error

[within_interval.json](within_interval.json) is a supplementary descriptive
calculation from the already saved trajectories, with no model inference.
For pixels correct at64, count `any(not correct[t])` over65..256. There is
one such pixel at size32 and265 at size64, compared with endpoint losses0/257.
One size32 pixel therefore briefly regresses and recovers before256; eight
size64 pixels do likewise. This is why “100% endpoint retention” does not mean
the correctness set is monotonically increasing at every step.

Across all changed pixels, there are89 unique pixels with a correct-to-wrong
transition after64 at size32, and1561 at size64. These include pixels acquired
after64. The full-interval regression counts above remain the frozen primary
behavioral summary; this narrower window is a supplement, not a new gate.

## What follows from this audit

The “rare structured positive result” interpretation is better supported now:
the checkpoint exhibits retention and local acquisition on both inspected
spatial sizes. We should seek the causal computation behind this observed
behavior before using it to justify another architecture modification.

This audit cannot establish flood-fill, lossless full updates, invariant latent
closure, bounded state amplitude, causal neighbor handoff, training basin,
or cross-checkpoint replication. Maps are reused and the successful checkpoint
was selected; model replication n=1. Terminal stability ends at256. All earlier
architecture qualification/development decisions remain unchanged. No further
experiment is launched by this interpretation.
