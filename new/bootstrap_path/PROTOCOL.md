# Seed4 early bootstrap versus general training-path sensitivity

Status: frozen before new efficacy results. Protocol `seed4_bootstrap_path_v1`.
This is a conditional mechanism screen on one selected initialization, not a
new architecture or a reliability gate. Original sources and evidence remain
unchanged. No runtime cap, watchdog, adaptive rescue or continuous monitor.

## Question and estimands

Does the first AdamW write actually retain the rank-one structure of the
initial task gradient? Does the historical training prefix preferentially
determine the eventual continuation phenotype, compared with equally short
late perturbations? A batch substitution changes both order and sample
exposure. Two additional reciprocal row permutations preserve the exact batch
multiset and test order sensitivity alone. Adam moments after a schedule
change are part of the treatment. This screen does not isolate a learned
representation convention from optimizer memory or later parameter changes.

## Fixed recipe

Use the historical seed4 `StreamingCell`, 5033 parameters, W24/Z8, zero F/Q
output heads and zero readout bias. Bank: 512 size32 maps, seed10002. Each arm
starts from identical seed4 parameters and a fresh AdamW; lr=.001,
weight_decay=.0001, betas=(.9,.999), eps=1e-8, one global clip_norm=1.
300 updates, batch8 sampled with replacement. Each update retains the original
64 numerical forward steps, eight equally weighted losses, and K8 detaches.
No optimizer reset at a splice. FP32, local historical Torch2.5.1/CUDA backend,
threads2, cudnn benchmark/deterministic false, cudnn TF32 true, matmul TF32 false.

H is the historical schedule20002. S is schedule20012 or20022, the first two
predeclared previously tested alternate schedules; both previously failed the
full phenotype. This deliberate selection is conditional evidence, not a
random schedule sample. Lengths k={1,3,8} are fixed before execution.

## Arms (23 total)

1. Full H control.
2. For each S: full S control; early replacement S[:k]+H[k:]; early preservation
   H[:k]+S[k:]; and H with updates181-k through180 replaced by the FIRST k rows
   of S. Early and pre-180 replacements thus share the exact donor minibatches.
   There are ten arms per S. Indices in code are zero-based; updates in
   this document are one-based.
3. H with its first8 rows reciprocally swapped with updates173..180.
4. H with updates165..172 reciprocally swapped with updates173..180.

The last two arms change16 update positions each and preserve every training
example's multiplicity exactly. Early and late row swaps have matched size.
The late window is deliberately tied to the previously observed175..180
collapse; it is a targeted pre-180 control, not representative of all late
training. Recipient H batches differ across replacement windows, so matching
the donor does not match the final exposure multiset of those replacements.
Arm order is frozen by `schedules.py`; no endpoint-based choice.

## Actual first-step audit

Independently measure six cloned runs: init2/3/4/5 on H firstbatch, and init4
on each S firstbatch. Accumulate the original task gradient, then apply exactly
one original AdamW update. Record gradients, actual parameter changes,
singular values/tail energy, proportional-gradient and epsilon-stabilized
AdamW formula errors, sign approximation, bias writes, optimizer moments and
the first post-update Z write's centered/uncentered spectrum. These diagnostic
models are not used as training initializations. Gradient rank one alone is
not evidence for rank-one parameter or latent-state writes, and a structure
common to all seeds is not a seed4-specific success explanation.

## Endpoints and qualification

Reuse the fixed 32-map size32/64 banks50032/50064 and the unchanged complete
phenotype evaluator in `new/seed4_followup/phenotype.py`. They were used in the
previous screen and are not fresh independent test sets. Paired source-flip
rollout0..256, reach/hold/retention/progress/regression/frontier thresholds are
exactly unchanged. Empty strata fail. Preserve full Boolean traces and integer
denominators. Primary endpoint: complete phenotype pass at update300.
Secondary: strict paired mean/pooled at T64/T128/T256, T64->256 retention and
coverage gain, ever-regressed fraction, each arm's difference from H and S.
All checkpoints are diagnostic only: save model and Adam state after updates
1/3/8/175/180/300, but do not search them for alternate successful endpoints.

H must reproduce the original initial/final parameter hashes and six full
historical endpoint payloads, then pass the fixed phenotype. If it fails,
stop remaining efficacy arms and report an unqualified comparison. Full S
controls must exactly reproduce their prior final parameter hashes and saved
phenotype metrics; any failure stops execution as a replay error. Do not
replace schedules, seeds, endpoints or maps after seeing an outcome.

## Predeclared interpretation

Report all12 early crosses, six late substitutions and two multiset-matched
swaps separately. For each k, a strong conditional prefix-privilege pattern
requires BOTH S choices to (a) lose H's full phenotype when replacing its
prefix, (b) gain the full phenotype when preserving H's prefix over S's suffix,
and (c) retain H's phenotype under the matched late substitution. Report which
k, if any, satisfies all six comparisons; call this a candidate prefix
signature conditional on the specified donor/recipient batches, not proof
that early updates are generally privileged. The three k values are related
screens, not independent tests or a statistical significance claim.

For pure row permutation, early fail plus pre-180 local-swap pass is a specific
conditional order signal; both fail establishes sensitivity to both specified
reorderings, and both pass fails to support order privilege for these
interventions. Donor blocks differ between these two swaps; do not present
their contrast as a timing-only experiment. Other outcomes
are reported without forcing a bootstrap conclusion. None establishes that
the first low-rank write causes downstream semantic composition; that would
require a separately authorized mediator intervention.

Independent training-path units are the two deliberately selected S choices.
Maps, cells, rollout times, prefix lengths and paired arms are not independent
training replicates. No p-values or architecture GO follows from this screen.

## Execution and evidence

One CPU sanity check verifies exact schedule edits, row-multiset preservation
and first-step AdamW arithmetic. CUDA preflight uses three updates on H and
one S prefix plus one full256-step paired evaluation for cost/memory, without
interpreting its phenotype. Formal launch binds CPU/preflight/source hashes,
saves source snapshots, exact schedules, train/eval hashes and launch receipt,
and verifies the first real optimizer update. Numerical/replay failure stops;
wall-clock duration does not. Canonical commands from repository root:

```text
python -X utf8 -B new/bootstrap_path/check.py --out analyses/NEW_BOOTSTRAP_CPU.json
python -X utf8 -u -B new/bootstrap_path/run.py --preflight --qualification analyses/NEW_BOOTSTRAP_CPU.json --out runs/NEW_BOOTSTRAP_PREFLIGHT
pwsh -File tools/launch_bootstrap_path.ps1 -RunName NEW_BOOTSTRAP_RUN -Preflight runs/NEW_BOOTSTRAP_PREFLIGHT
```

Read `RESULTS.md`, `aggregate.json`, `first_step/summary.json` and per-arm
summaries first. NPZ traces, checkpoints, raw curves and launch logs are
secondary. Launch receipts and checkpoints remain local. Sources reference
existing local archival manifests; this experiment has not yet been published.
