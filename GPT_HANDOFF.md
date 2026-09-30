# Incremental review: A0 after c669e18

- Review base: `c669e18cc6f1747ff99480c1e7a57ebc2866e956`.
- Evidence head: `74d77256daadfcd2bf90741b63fa6ba62a8eb65b`.
- The subsequent commit adding this handoff and its README link changes review
  metadata only. Use the evidence head above when comparing code and results.

## What changed

The registered A0 experiment completed its conditional schedule: 24/24 trials,
with 16 possible 3D RT trials skipped after failed positive-control calibration.
Forty was an upper bound, not the number still required after that failure.

A0 uses four seeds and 480 updates, compared with v1's two seeds/240 updates.
Every A0 arm emits q=c*v. Attention learns query/key and uses the same q as
values, with no separate value/output projection. The four RT arms cross
constant/learned medium with raw/normalized reception. Paired arms share initial
parameters, unit initial conductance and packed numerator/confidence solver
work; only receiver division differs within a pair. Learned conductance is
2*sigmoid of symmetric local features, with a zero-initialized final head.

Numerical/oracle checks cover the added grouped solve, normalization gradients,
attention values and paired initialization. Source/protocol hashes and all
completed trial metrics are included in the new public evidence directory.

## Decision-relevant new evidence

- 2D attention: four of four seeds are perfect in every reported condition.
- 2D RT: all four arms fail training-scale fitting in every seed. This is a
  negative qualification under a reliable control, not just the v1 control issue.
- Normalization's mean d128 effect is +0.977 percentage points for constant
  medium and +0.586 for learned medium, while both remain near chance. Effects
  change sign across conditions; no significance or equivalence claim is made.
- 3D: attention seed1729 fails (47.92% train d16); no RT arms or normalization
  treatment estimate exist for that dimension.
- Large-tau oracle emission remains decodable through d128; tiny-tau d128
  underflows. Oracle recoverability does not establish learned-model success.
- Sampled normalized-model final-update probes retain moderate background
  confidence and do not show denominators below epsilon. This observation does
  not causally identify why training fails or quantify total information flow.

## Unchanged claims and evidence

The original v1 data, model, runner and solver sources remain unchanged, as
verified by the original manifest. Its negative/inconclusive evidence is kept.
No B/C, width sweep, natural-vision, general-3D or optimized-kernel experiment
was added. Fixed axis splitting and narrow straight geometry remain limitations.
A0 raw is confidence-gated, so it is not identical to v1 raw. Multiple changes
between protocols prevent isolating the cause of improved attention calibration.

## Minimum reading order

1. [RESULTS.md](RESULTS.md), current A0 sections only.
2. [A0_PROTOCOL.md](A0_PROTOCOL.md), intervention and stopping rules.
3. [a0_models.py](a0_models.py), `transport_pair`, `SharedValueAttention`, `A0Model`.
4. [summary.json](evidence/a0_v2_1/summary.json), per-seed/paired metrics and probe summaries.
5. Use [aggregate.json](evidence/a0_v2_1/aggregate.json) only for a specific
   axis/history question; [checks.json](evidence/a0_v2_1/checks.json) for numerical
   validation. [A0_PUBLICATION_MANIFEST.json](A0_PUBLICATION_MANIFEST.json) binds
   these to the executed source. Do not begin by rereading large raw evidence.

## Concrete reviewer questions

1. Does any code or endpoint contradict the scoped 2D NOT_QUALIFIED_FIT verdict?
2. What one bounded intervention would distinguish emission learning from
   reaction/readout optimization, without treating oracle success as learned evidence?
3. Are confidence-mass diagnostics being interpreted at their actual final-
   update/group/sample scope, including the possibility of messages re-emitted
   elsewhere after earlier recurrent steps?

The current configuration is not being escalated. This handoff requests review,
not additional runs or a reinterpretation of the frozen negative result.
