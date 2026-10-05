# Semantic write: source protection and unprotected continuation

Protocol `semantic_write_source_causal_v1`, frozen before efficacy inference.
Zero training: no backward, optimizer, fitting, parameter editing or checkpoint
selection. This is an exploratory causal screen, not architecture qualification.

Question: does preventing destructive semantic writes at input source cells
improve computation at unprotected distant cells?

## Models, data and interventions

Take the first8 paired blocks (00..07), all four final-u300 factorial arms,
without score selection:32 models. Add the original Streaming seed4 u300 as
one selected historical reference, with its original C24/Z8 architecture.
It is not a width-matched control or an independent replication. Use the same
existing32-map evaluation banks at sizes32/64, seeds99332/99364, for every model.
These are reused cohorts. Original/flip worlds are run separately at the
historical batch32 shape. Natural factorial traces must replay the saved
Boolean arrays at all257 times exactly before accepting that model/size.

All conditions start from each model's natural initialization and execute256
steps. The intervention changes only the Q output, before the existing Z
addition. C and R in that same step remain exactly as computed by the model;
future C/R may change through feedback from the modified Z.

| Condition | Activation | Intervention |
|---|---:|---|
| natural | never | unchanged Q; observational signed-write audit |
| source_parallel_t8 | after stateT8 | remove readout-parallel Q only at cue-correct sources with a negative cue-signed proposal |
| source_parallel_t64 | after stateT64 | identical protection, later activation |
| source_orthogonal_t8 | after stateT8 | same current-state source trigger; equal-norm subtraction in readout nullspace |
| oracle_solved_t8 | after stateT8 | protect currently true-label-correct traversable cells against negative semantic writes |
| nullspace_zero_t8 | after stateT8 | retain Q-parallel only, at every cell |

Source interventions derive the sign from the visible positive/negative input
cue, never distant labels. Orthogonal control subtracts a vector parallel to
Q-perpendicular with norm equal to Q-parallel; if Q-perpendicular is zero,
use a deterministic readout-orthogonal basis vector. This matches perturbation
norm and trigger at each trajectory's current state; states and therefore
triggers can diverge after intervention. It is not a permanently matched event
schedule. Full nullspace removal is a mechanistic ablation, not a harmless
control: it changes future F/Q inputs.
Correctness gates use the actual classifier tie rule: logit>=0 predicts1;
zero is correct for a positive cue/label and wrong for a negative one.

Oracle protection uses ground truth. Retention/monotonic margins in protected
cells follow by construction and are implementation checks, not mechanism
evidence. Oracle gating can itself inject distant-label information; never
interpret its reach as learned source communication. Neither its Full gate nor
the protected source score can qualify the primary hypothesis.

## Outcomes and decision

Primary unit is one paired training block in the native arm (n8). Primary
metric is size32/T256 pixel-pooled paired coverage in16<d<32, averaged equally
over the8 blocks. All of these cells are unprotected by source-only conditions.
Report per-map and per-block values plus the paired deltas:

1. source_parallel_t8 minus natural;
2. source_parallel_t8 minus source_orthogonal_t8.

Predeclared exploratory signal requires both mean deltas>=.05 and positive
source-minus-natural in at least6/8 blocks. This heuristic is a decision for
follow-up, not a population significance test. If source is protected but
unprotected distant computation does not improve, classify the source-driver
hypothesis as unsupported for this intervention. If the orthogonal perturbation
improves equally, the semantic-direction-specific hypothesis is not supported.
All other arms, size64, late activation and auxiliary conditions are secondary;
they cannot rescue the primary decision. Report all negative results.

Also report T64/128/256 paired coverage and natural T64-set retention outside
sources, newly acquired versus lost answers, and every-step regression. Use
each model's natural T64-correct set as a common reference for intervention
retention, in addition to each condition's own T64 set. No pixel/map/time is an
independent training replicate. Source score is a manipulation check only.

Natural write diagnostics at each integer step are conditioned on changed-region
sources, paired-correct cells, frontier paired-wrong cells, and nonfrontier
paired-wrong cells. Frontier means a4-neighbor paired-correct cell. Source and
solved categories overlap intentionally. Record actual margins/increments,
negative proposals, margin-crossing proposals, positive/negative write mass and
parallel/nullspace energy per map; mean signed write alone cannot establish a
destructive event. Negative writes that keep margin positive are allowed in a
successful system. Exact logit accounting identifies the immediate output
channel, not the upstream cause of Q.

The seed4 natural reference is separately qualified for useful size32 continuation
(strictT256>=.80 and T64->256 changed-region retention>=.95). A failed reference
qualification is reported without replacing maps/model. It limits success/failure
interpretation but does not replace the native within-checkpoint primary result.

## Execution and retained evidence

396 model/size/condition units:33 models x2 sizes x6 conditions. Local historical
Torch2.5.1 CUDA backend, FP32, unchanged TF32/cuDNN flags. No runtime cap, watchdog,
recurring monitor, rescue run or adaptive expansion. CPU projection/wiring checks
and one actual-checkpoint short CUDA smoke precede dispatch. Measure inference
cost there for the runtime estimate; smoke efficacy is not a formal endpoint.
Also check the first native model's complete natural256-step trace at both
actual32-map shapes against saved arrays before dispatch. No intervention
efficacy is inspected during qualification.

Save complete packed paired Boolean traces, sampled margins, per-map endpoint
CSV, natural per-step signed-write arrays, intervention counts/energies, compact
aggregates, replay checks, source/data/checkpoint hashes and immutable source
copies. No full floating hidden-state trajectories or new checkpoints. Original
evidence remains read-only. Machine launch metadata stays local.
Sampled margins retain every changed-region pixel at the listed times in FP32;
pixels outside that region are omitted. Natural source margins and proposed
signed increments are retained at every step separately for both worlds/maps.

Repository-root commands:

    python -X utf8 -B new/semantic_write_audit/run.py --check --out analyses/NEW_SEMANTIC_CHECK.json
    pwsh -File tools/launch_semantic_write_audit.ps1 -RunName NEW_SEMANTIC_RUN -Qualification analyses/NEW_SEMANTIC_CHECK.json

Launch completion means verified dispatch with a durable receipt. Continued
status checks require a separate user request.
