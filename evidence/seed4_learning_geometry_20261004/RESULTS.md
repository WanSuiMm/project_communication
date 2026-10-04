# Seed4 learning geometry: mixed initialization evidence, no training-time precursor

Two completed2D diagnostics, with zero new training. Initialization geometry
uses CPU30.31s and four exact historical initializations; formation analysis
uses CPU8.86s and44 existing trace records, without new inference.

1. [Initialization results](../initial_geometry_20261004/RESULTS.md): seed4 has
   the most balanced raw lane RMS and the largest measured available relative
   source-flip separation at endpoint64. Centered balance and spectral rankings
   differ; seed3 has higher K8 paired kernel-label alignment on all four banks.
   All seeds share the same initial state-Jacobian isometry. The first task
   gradient primarily opens Q-out weight; encoder/F/Q-in/readout-weight gradients
   are zero, and Q-out gradient is rank one up to FP32 roundoff.
2. [Formation results](../formation_gate_20261004/RESULTS.md) and
   [interpretation](../formation_gate_20261004/INTERPRETATION.md): early rollout
   retention+progress passes at training updates140/145/190/200, while the long
   screen passes only200 in the dense window. Same-checkpoint comparison has
   TP1/FP3/TN17/FN0. Predicting update u+5 has TP0/FP3/TN16/FN1. The sampled
   early proxies do not identify a training-time precursor.
3. Raw feature matrices and full trajectories remain local; compact summaries,
   every negative seed/checkpoint, per-map denominators and exact code are public.
   Run `python -X utf8 -B tools/export_learning_geometry.py --verify-only`.

Training update u and rollout step t are distinct. The initialization audit runs
untrained parameters through t64; its weights are still at update0. Formation
predictors observe t<=64, while later behavior uses t128/256. An association
within a fixed checkpoint does not forecast future optimizer updates.

These are post hoc diagnostics on already-inspected seeds/maps, not a new
architecture qualification or prospective predictor. Initialization has four
seed units, one historical success; formation has one selected training
trajectory and one successful checkpoint in the dense window. No statistical
anomaly, causal initialization rule or reliability improvement is established.

The continuation-contract idea remains conditional: raw-domain closure, small
abstract transition error epsilon, robust progress margin gamma>epsilon, and
task-output error eta below a correct semantic margin m imply continued valid
execution. These audits have not constructed the required representation pi,
abstract rule F or uniform domain certificate. Therefore they lower the weight
of closure as an identified seed4 mechanism, while leaving that conditional
execution implication intact. Retention metrics alone do not identify a quotient.
Earlier architecture and warm-start negative verdicts remain unchanged.

Source/definition routes:
[initialization protocol](../../new/initial_geometry/PROTOCOL.md),
[formation protocol](../../new/formation_gate/PROTOCOL.md),
[initialization reproduction](../initial_geometry_20261004/REPRODUCTION.md),
[formation reproduction](../formation_gate_20261004/REPRODUCTION.md).
