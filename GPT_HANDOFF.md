# Incremental review: seed4 schedule, initialization and rollback follow-ups

- Review base: `ba433ecc28a7454db853cac6d6c258076a14f6fc`.
- Evidence head: `ff51756bb8fc20f773a6adcc957c0132a9d62f27`.
- This following handoff-only commit changes review metadata, not code or evidence.

Three separate questions, unchanged original StreamingCell and K8 recipe:
A varies batch schedule, B perturbs starting parameters, C rolls back local
state in the existing selected checkpoint. A/B completed all 13 arms x300
updates in31.11 minutes. C completed with zero training. No new GPU training
or inference was performed to prepare this upload.

## Minimal reading order

1. [Current results](evidence/seed4_followup_20261003/RESULTS.md),
   [interpretation](evidence/seed4_followup_20261003/INTERPRETATION.md), and
   [compact all-arm aggregate](evidence/seed4_followup_20261003/summary.json).
   Inspect reach/hold/dynamics/frontier separately before the full gate.
2. [Frozen protocol](new/seed4_followup/PROTOCOL.md),
   [training/control runner](new/seed4_followup/train.py),
   [phenotype metrics/gate](new/seed4_followup/phenotype.py), and
   [local rollback runner](new/seed4_followup/causal.py).
   [GPT_CONTEXT.md](GPT_CONTEXT.md) maps concepts to exact source symbols.
3. [Compact record of full local training validation](evidence/seed4_followup_20261003/training/validation.json),
   [C validation](evidence/seed4_followup_20261003/causal/validation.json),
   [public arithmetic](evidence/seed4_followup_20261003/publication_validation.json),
   [provenance](evidence/seed4_followup_20261003/provenance.json), and
   [publication bindings](SEED4_FOLLOWUP_PUBLICATION_MANIFEST.json).
4. [Reproduction notes](evidence/seed4_followup_20261003/REPRODUCTION.md).
   Verify hashes and exported arithmetic without a model/GPU using
   `python tools/export_seed4_followup.py --verify-only`.

Individual training summaries retain every gate, denominator, map/censor
profile and transition. Raw arm records, frontier CSVs and C event records
are secondary; do not open them first. Checkpoints and full Boolean traces
remain local with their hashes bound. A/B retraining needs three public
reference inputs restored by the documented helper; exact C replay needs
the original excluded checkpoint/source archives.

## Changed evidence and claim boundaries

The historical training control exactly reproduces initial/final parameter
hashes and all six full historical endpoints. It also passes the new full
phenotype on fresh32-map banks at each size32/64. The gate requires reach,
hold, retention, coverage growth, low every-step regression and matched-frontier
acquisition at their frozen thresholds; endpoint retention alone cannot pass.

| Screen | Independent unit | Full phenotype passes | Frozen criterion | Verdict |
|---|---|---:|---:|---|
| A: same initialization, new schedules | 4 schedules | 0/4 | >=3/4 | NOT_QUALIFIED |
| B: global L2 epsilon=.01 | 4 directions | 0/4 | >=3/4 | NOT_QUALIFIED |
| B: global L2 epsilon=.05 | Same paired directions | 0/4 | >=3/4 | NOT_QUALIFIED |

All 12 A/B arms pass both matched-frontier gates. Their other failures vary.
Schedule20032 passes hold/retention/growth but fails reach and per-step
regression. Direction70003/.05 passes size32 reach+hold but fails parts of
size64 dynamics and regression. These partial behaviors are retained;
full-gate failure does not mean nothing was learned. A/B are conditional on
one starting solution, bank and training recipe. B perturbs originally zero
output blocks too; it does not preserve the zero-output initialization
manifold. Four paired directions are not eight independent replicates.
No certified basin radius, monotone radial boundary or architecture reliability
estimate follows.

C qualifies its implementation/event population:93/257 events across31
eligible maps at size32/64, with full replay and numerical controls passing.
Its frozen primary is **NO_PRIMARY_THRESHOLD_SIGNAL**: step1 native-minus-sender
is+14.52/+7.00pp and wrong-sham-minus-sender is+3.76/+0.44pp. Required joint
thresholds are>=10pp and>=5pp at BOTH sizes. Step4 secondary contrasts are
larger (+33.53/+28.60pp and+20.18/+15.33pp) and do not rescue the primary.
The miss is not a statistical null test. C is one selected checkpoint on
historical maps, equal-map averaged after within-map event averaging.
Whole-cell W/Z temporal rollback and norm-matched sham do not identify an
edge-specific message, unique sender, semantic handoff or flood-fill algorithm.

## Validation and unchanged material

The full local training review passes383/383 checks, including56 current
source hashes and56 run snapshots, all13 CPU checkpoint parameter hashes,
reconstructed initialization/radius identities, train/fresh/historical banks,
five schedules, six exact control replays, Boolean-trace metrics and every
phenotype gate. Floating metric comparisons allow2e-7 for FP32 versus NumPy
arithmetic; pass booleans and thresholds must match. Human-readable reason
strings are excluded from numeric comparison because they embed formatted floats.

The C saved-event review verifies55 source bindings, published-reference and
checkpoint hashes,350 event geometries, paired logits, numerical controls and
equal-map contrasts. The narrower public checker verifies all13 summaries,
125459 matched CSV strata and350 C events, plus source/tool/evidence hashes.
The staged-file privacy/link/binding check passes; frozen scientific records
are preserved, and metadata copies omit machine/process fields.

The compact training validation copy omits duplicate metric profiles and
successful-check detail payloads, retains check names/verdicts, provenance,
checkpoints/traces and gate values, and binds the complete local report SHA.
Public arithmetic does not replace the full local trace/checkpoint review.

All earlier architecture no-go verdicts, original cell/training code, old
checkpoints and frozen evidence remain unchanged. This follow-up establishes
conditional sensitivity and partial behavior, not a general rejection of NCA
or an explanation of short-BPTT credit assignment. No new architecture or
follow-up experiment is proposed in this delivery.

## Concrete reviewer questions

1. Are the fresh phenotype conjunction, denominators/censoring and partial
   outcomes preserved without interpreting0/4 as absence of all learned behavior?
2. Do A's schedule unit and B's paired direction unit, including perturbations
   of zero output blocks, support only the stated conditional conclusions?
3. Are C's event selection, equal-map effects, sham limitations and failed
   one-step primary kept distinct from its larger four-step secondary effects?
