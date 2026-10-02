# Post-execution review notes

These notes clarify the completed `switch_audit_20261002_seed0` audit. They
are post-execution metadata, not an amendment to the frozen protocol or run.
The original source snapshot and all generated measurements remain unchanged.

## Exact replay scope

The frozen protocol's phrase about replaying original and fresh-flip W/Z RMS
is broader than the saved reference permits. The original experiment saved
W/Z RMS only for its original-input curve, not for its fresh-flip curve.
Actual verification was:

- Both original and fresh-flip trajectories: BA, BCE and per-map BA at all
  seven original horizons.
- Original-input trajectory only: per-map W/Z RMS at those horizons.
- Warm/cold source-switch branches: BA, BCE, per-map BA and changed/unchanged
  per-map accuracy at the six previously saved intervention anchors.

This gives 116 comparisons per size, 232 total, all with zero maximum error.
No fresh-flip norm replay is claimed. Candidate reconstruction error is zero;
the affine readout identity has maximum error 7.153e-7 in the audit. The prior
CPU algebra check used randomized nonzero output weights and nonzero readout
bias; its step error was zero and affine error 1.192e-7.

## Quantified factorial contrasts are included

The compact `analysis.json` contains 36 contrast records: sizes32/64, anchors
K0/64/128, metrics changed accuracy and signed new-target margin, and three
contrasts. Each record includes the mean and all 16 per-map values. At K64,
the continuous signed-margin contrasts are:

| Size | Replace W, hold warm Z | Replace Z, hold warm W | Joint minus additive contrast |
|---|---:|---:|---:|
| 32 | 1.667010 | 1.546669 | 1.029592 |
| 64 | 1.870396 | 1.810439 | 0.454708 |

The last column is f(new W,new Z)-f(new W,warm Z)-f(warm W,new Z)+f(warm W,warm Z),
with X fixed to the new input. These are finite donor substitutions through
the frozen Q interface, not derivatives, significance estimates, or evidence
that a particular internal feature is the unique source of failure.

## Interpretation boundary retained

The audit supports old-aligned candidate readout and dependence on both W and
Z in these interventions. It does not isolate stale W as the sole cause, prove
that mixed states are on-distribution, or establish that generally both blocks
must be reset. Size32 single-block accuracy37.5% is all-negative prediction;
the apparent partial improvement must not be called successful source revision.

No additional inference or training was needed for these review clarifications.
