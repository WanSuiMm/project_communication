# Addressed delta writer development screen

Protocol `addressed_delta_native_k8_development_v1` freezes a four-block development
comparison of the current carrier writer, an addressed-additive writer, and an
addressed-delta writer. The question is whether subtracting the current carrier's
addressed read changes the fixed-endpoint readiness outcome relative to the
matched additive writer. This screen does not establish reliability, a semantic
duplicate proof, or a general benefit of addressed writing.

The standalone proposal archive used as the primitive input is
`addressed_delta_primitive_proposal.zip`; its SHA-256 at protocol freeze is
`41bbd9df9e111a135d1c4d51366f0afe9bb98676a3f8750ad60174e707985e7b`. This is
the proposal archive hash only. The final qualification must snapshot and hash
the integrated source ZIP after all source writes, plus its configuration,
training bank, evaluation banks, and schedule.

## Matched arms

All arms retain the existing 67-to-40 tanh feature transform and the established
encoder `E`, pre-stream `L(C)`, query `Q`, readout, and C24/Z8 state.
Only the carrier writer changes. The existing fixed masked port permutation and
K8 credit schedule remain in force. The addressed candidates replace the old
writer head/modules as specified by the integration code; common tensors and
heads are copied identically. Additive and delta start from identical candidate
tensors within each paired block; only the subtraction term differs. Current
has 5,033 trainable parameters. Each
addressed candidate has 5,689; the additive and delta candidates have identical
parameter counts and differ only in the subtraction term.

For each cell, let `U` be the streamed 4-by-6 carrier, `A` the 4-by-4 raw key,
and `V` the 4-by-6 value. Normalize each cell's key independently:

```text
K = A / sqrt(1 + ||A||_F^2)
```

The additive arm writes `U + 0.1 K^T V`. The delta arm writes
`U + 0.1 K^T (V - K U)`. The key head is zero-initialized, the value head uses
the ordinary random initialization, and the existing zero-initialized Q
bootstrap is left unchanged. This makes the initial writer correction zero;
it does not certify semantic preservation or remove the historical Q bootstrap
behavior.

## Blocks, banks, and training

Run four fresh paired blocks. Block `b` uses initialization seed `140001 + b`
and schedule seed `141001 + b`, for `b=0,1,2,3`; all arms in a block share the
paired initialization and schedule where their parameter shapes permit. The
fixed training bank is seed `10002`, with 512 maps of size 32 and batch size 8.
Each block has 300 super-updates. A super-update reuses one minibatch for four
cold 64-step segments: 256 physical steps total, 32 endpoint balanced-BCE losses
and 32 K8 backward windows. Divide the 32 losses by 32. Parameters stay fixed
across the four segments. Apply one AdamW update after the complete super-update, with learning rate
`0.001`, weight decay `0.0001`, betas `(0.9, 0.999)`, epsilon `1e-8`, and global
gradient clipping at norm 1.

Evaluate the same held-out banks for every arm and block: 32 maps at size 32
from seed `122032`, and 32 maps at size 64 from seed `122064`. Record their
content hashes in the run manifest. The 13 checkpoint updates are
`0,25,...,300`: 4 blocks × 3 arms × 13 updates = 156 unique dense-stage records.
The formal endpoint is update 300 only: 4 × 3 = 12 final records. No other
checkpoint can replace or select the endpoint.

## Readiness and decision

At size 32, readiness is the existing joint predicate: strict paired changed
pixel coverage at T64 for graph distances `16 < d < 32` must be at least 0.80
both pooled and mean-map; all-changed T64-correct retention through T256 must be
at least 0.95; and that retention must have at least 16 contributing maps and
100 contributing T64-correct changed pixels. Missing, nonfinite, or
under-supported numeric metrics fail readiness. A numeric evaluation failure
still writes its scheduled record with readiness false and null metrics.

The primary contrast is delta versus additive on this size-32 joint predicate at
update 300. Count paired wins, losses, ties, and net wins. The predeclared
developmental signal requires both at least 2 of 4 delta blocks ready and at
least one net paired win for delta over additive. Otherwise the verdict is
`NO_DEVELOPMENTAL_SIGNAL`. Four paired blocks cannot support a reliability or
significance claim: even four wins and zero losses yield an exact two-sided
paired sign-test p-value of 0.125. The p-value is descriptive only. Current is a
contextual secondary reference.

For both sizes and every checkpoint, report strict pooled and mean T64 coverage,
T64-to-T256 endpoint retention, continuous survival of the T64-correct set,
retention support, all-changed coverage at T128 and T256, and original/flipped
balanced accuracy and BCE when present in the endpoint score records. Also
report arm-level readiness counts and checkpoint-grid ever-ready counts. Dense
stages are descriptive; do not select a checkpoint or infer a second endpoint.
The old Full phenotype evaluation is intentionally not run, including at u300,
and no Full metric is reported.

Persist the frozen bank tensors in compact form (about 0.5 MB total), compact
metrics, the 64 per-map metric rows per stage (32 maps at each size),
training/checkpoint summaries, checkpoints, and provenance. Keep full per-map
Boolean correctness trajectories transient in memory only; use them to compute
the exact metrics and then discard them. Stage and endpoint records use atomic
completion markers. Preserve recovery state so an interrupted run resumes from
the last complete checkpoint and record without changing the frozen arms,
seeds, banks, or schedule.

The run is `COMPLETE` only with exactly 12 unique update-300 endpoint records
and 156 unique dense-stage records. A global training or engineering error
leaves the run `ERROR` and the scientific verdict `INCOMPLETE`. Any missing,
duplicate, invalid, or incomplete record also leaves the verdict
`INCOMPLETE`; never infer a missing endpoint from another checkpoint. Outputs
include a compact JSON summary, `RESULTS.md`, `metrics.csv`, and
`final_metrics.csv`.
