# RRC-v0 recovery entrypoint

This directory adds a CPU-audited continuation path for the interrupted
`rrc_v0_prestream_relation_factorization_v1` run. It does not change or invoke
the original qualification runner. The frozen design remains three arms,
eight paired blocks, K8 credit, `reset64x4`, and 300 super-updates.

## Read-only plan and recovery check

From the project root, choose a child path that does not already exist:

```powershell
python -X utf8 -B new/rrc_recovery/run.py `
  --parent runs/rrc_v0_20261007_01 `
  --out runs/rrc_v0_20261007_01_resume_20261007_02 `
  --plan-only

python -X utf8 -B new/rrc_recovery/check_recovery.py `
  --parent runs/rrc_v0_20261007_01 `
  --out runs/rrc_v0_20261007_01_resume_20261007_02 `
  --report analyses/rrc_recovery_check_20261007_01.json
```

Plan-only mode does not create the child directory or initialize CUDA. It
checks the original passing qualification and its source binding, current
qualified sources, copied source snapshot, fixed schedules, data-bank hashes,
every dense checkpoint file hash, and every recorded evaluation stage. It also
loads each trajectory's latest committed checkpoint on CPU and verifies its
model hash and AdamW step count. The check script adds a small 4×4 CPU
save/restore parity run for all three cell arms; after restore, the next
256-step super-update must produce identical model and optimizer states. The
optional `--report` argument writes a new audit receipt under `analyses/` and
refuses to overwrite an existing file.

For the frozen parent snapshot, the plan inherits 137 fully recorded stages,
including 10 trajectories completed at u300. `block03/factorized` resumes from
its committed u150 checkpoint, so its first update is 151 and its first batch
uses `plans["3"]["batch_indices"][150]`. The orphan u175 checkpoint and partial
u175 evaluation folder are ignored because u175 is absent from `dense.json`.
The remaining 13 trajectories start at u0. This gives 14 trajectories to run
and 175 new checkpoint/evaluation stages.

## Authorized execution

After a separate decision to run the training workload, omit `--plan-only`:

```powershell
python -X utf8 -u -B new/rrc_recovery/run.py `
  --parent runs/rrc_v0_20261007_01 `
  --out runs/rrc_v0_20261007_01_resume_20261007_02
```

The entrypoint creates a new child run and leaves the interrupted parent
untouched. It byte-copies the committed checkpoints and evaluation folders,
copies the original banks, plans, config, and source snapshot, and trims each
inherited training curve to its latest committed update. It records both the
parent's original source hashes and the recovery wrapper's execution hashes.
Completed arms are skipped. A partial arm restores the exact saved model and
AdamW state; an unstarted arm uses the frozen initialization. Schedule lookup
uses index `update - 1`, so resumed update 151 receives the saved batch at index
150. New checkpoints are written to a temporary file and atomically replaced.
The dense row, curve, report, and status are written before a per-checkpoint
`.complete.json` marker is atomically written last.

The checkpoint restores model tensors, AdamW moments, and the completed update
count. It does not restore backend RNG state, and the recorded CUDA backend is
not deterministic, so bitwise equivalence to an uninterrupted run is not
promised. Work after the latest committed 25-update checkpoint is replayed;
for the parent partial arm, updates after u150 are recomputed.

Execution checks the GPU model name before creating the child and records
Python/Torch/NumPy/CUDA versions. This checks the model name, not device
identity. Capture setup timing for the partial arm is marked
`resumed_session_only` because its original setup time is unavailable.

This helper is specific to the original 137-stage interrupted parent. It cannot
resume a recovery child after another interruption. Its local parent checkpoints
and passing qualification are prerequisites and are not bundled on GitHub.

The original `new/rrc_v0/run.py`, qualification, launch script, and parent run
artifacts remain unchanged. Public checks and their exact source bindings are in
[`evidence/execution_protection_20261007_01/`](../../evidence/execution_protection_20261007_01/README.md).
