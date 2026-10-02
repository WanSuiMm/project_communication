# Incremental review: Streaming seed4 operator audit

- Review base: `87763540c7e342a8fe278b8c198930f4decc2c1a`.
- Evidence head: `4c424cfe3ce605c3749e1430d7aef32bfc38f04f`.
- This following handoff-only commit changes review metadata, not code or evidence.

One selected existing Streaming seed4 checkpoint, no training. Four cold-rollout
conditions completed in 12.36 seconds. All six historical size32/64 by
T64/T128/T256 evaluation records replay exactly (maximum error 0).
Scientific status: SELECTED_CHECKPOINT_OPERATOR_SENSITIVITY. No new GO gate
or model-seed replication. The prior multi-seed architecture decisions remain.

## Minimal reading order

1. [Current results](evidence/stream_path_audit_seed4/RESULTS.md),
   [interpretation and limits](evidence/stream_path_audit_seed4/INTERPRETATION.md),
   and [compact summary](evidence/stream_path_audit_seed4/summary.json).
2. [Frozen protocol](new/stream_path_audit/PROTOCOL.md),
   [exact pathway switches](new/stream_path_audit/operators.py),
   [clock and shapes](ARCHITECTURE.md), and
   [runner/replay gate](new/stream_path_audit/audit.py).
3. [Saved-result validation](evidence/stream_path_audit_seed4/validation.json),
   [replay validation](evidence/stream_path_audit_seed4/replay_validation.json),
   [public provenance](evidence/stream_path_audit_seed4/provenance.json), and
   [publication hash bindings](STREAM_PATH_AUDIT_PUBLICATION_MANIFEST.json).
4. Detailed [curves](evidence/stream_path_audit_seed4/curves.csv),
   [contrasts](evidence/stream_path_audit_seed4/contrasts.json),
   [CPU reference](new/stream_path_audit/check.py), and
   [arithmetic analyzer](new/stream_path_audit/analyze.py).

The 2.1 MB [raw conditions](evidence/stream_path_audit_seed4/raw_conditions.json)
are secondary. Metrics, curves, contrasts, replay and arithmetic validation
are byte-identical copies. Manifest/completion are sanitized copies; their
public hashes are bound separately. Validation input_sha256 refers to original
run files, including excluded private manifest/completion bytes. Checkpoints,
machine identifiers, runtime commands and source snapshots remain local.

## Intervention and decision-relevant evidence

Use the same W24/Z8 cell, trained parameters, encoder, readout, residual scales
and original/source-flip evaluation maps (32 each at sizes32/64). Starting
from the same cold initialization, switch fixed T_M to identity, and/or replace
both Laplacian slots in BOTH F and Q by zero tensors. F retains pre-stream
L(W)/L(Z); Q retains post-F L(Wplus) and pre-update L(Z). Turning off T also
changes the incoming workspace feature read by F. Equal macro-step counts
do not match communication opportunity: hop bounds are 2/2/1/0.

| Condition | Primary pooled T64 % | T128 % | T256 % | Size64 d>32 T256 % |
|---|---:|---:|---:|---:|
| Full original cell | 85.92 | 99.35 | 100.00 | 59.43 |
| T_M replaced by identity | 0.00 | 0.00 | 0.00 | 0.00 |
| Direct Laplacian inputs zeroed | 0.00 | 0.00 | 0.00 | 0.00 |
| Both interventions | 0.00 | 0.00 | 0.00 | 0.00 |

Primary is size32 strict16<d<32, paired correct under BOTH source alternatives
at the SAME changed-component pixel. Full correct counts at T64/T128/T256
are 2507/2899/2918 out of2918; every primary knockout count is0/2918.
These are not ordinary whole-grid accuracy percentages.

Nearby behavior also collapses. At size32, no_perception has0/1706 paired hits
in d<8 already atT8. No_transport has906/1706 atT8 and1067/1706 atT16, then
zero byT32. AtT64 all knockouts have original/flipped BA50.00%/48.44%.
Thus the loss is not adequately characterized as only slower remote propagation.

Changed: this particular successful frozen solution fails to retain behavior
under either pathway intervention, supporting sensitivity to the original
hybrid update. Unchanged: Streaming's multi-seed DEVELOPMENT_NO_GO (1/4 versus
baseline2/4), the previous sustained seed4 positive case, Persistent Roles0/4,
Local Interface0/4, and all earlier gate conclusions.

Knockouts alter learned feature/state distributions and effective spatial
depth. They do not identify the reason H/C/Z failed, uniquely locate semantic
operations, prove that restoring Laplacians will repair training, or show that
retrained knockout models cannot succeed. Positive factorial interaction is
descriptive arithmetic, not an additive causal decomposition. One selected
model remains n=1, without significance or architecture-superiority claims.

Verification: nonzero independent CPU reference passed; six historical replay
records match in5628 integer and2636 float leaves. Saved CPU analysis checks39
source bindings,672 paired aggregates/denominators,96 BA means,672 curve rows
and600 contrasts. Publication performed no additional inference or training.

## Questions for review

1. Are the carry replacement and both F/Q Laplacian switches implemented
   with the historical clocks and unchanged weights? Are the different light
   cones kept explicit?
2. Do paired denominators, all six exact replay records and knockout effects
   agree across raw evidence, summary, curves and contrasts?
3. Does the interpretation distinguish frozen-solution sensitivity from
   training causality, and retain the near-cue collapse and selected-model limit?
