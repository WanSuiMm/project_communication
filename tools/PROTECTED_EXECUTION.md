# Independent on-demand experiment execution

RRC interruption diagnostic,2026-10-07: last worker progress13:19:09 local.
Windows Application events from `CodexSandboxService.OpenAI.Codex` report
1001 at13:19:14.949 (asked to stop),1002 at13:19:14.950 (stopped),1000 at
13:19:16.873 (running). Empty stderr, no run error.json and an unfinalized
launcher receipt support external service/session teardown as the likely trigger.
They do not provide a Python exit code or establish who requested the service stop.
The model's frozen scientific negative/partial conclusions are unchanged.

`start_protected_job.ps1` registers a unique **on-demand** Windows Task Scheduler
task under the logged-on user's interactive token, limited privilege, no password,
no recurring trigger, no automatic retries and zero execution-time limit.
It returns after task dispatch; `protected_job_worker.ps1` is owned by Task
Scheduler rather than the originating Codex tool process. The worker holds a
single-instance lock, runs Python in a hidden process, atomically writes a
receipt/heartbeat and records native exit code and stderr.

While Python runs, `SetThreadExecutionState(CONTINUOUS | SYSTEM_REQUIRED)`
requests temporary idle-sleep prevention. It is cleared on completion/error;
no persistent power plan or display setting is changed. This does not protect
against reboot, logout, forced shutdown, manual sleep, lid policy or critical
battery. Existing checkpointing and explicit recovery remain necessary.

## RRC entry points

Fresh authorized run, from repository root:

```
pwsh -NoProfile -File tools/launch_rrc_protected.ps1 -RunName NEW_UNIQUE_NAME -Qualification analyses/rrc_v0_qualification_20261007_02.json
```

Recovery uses a **new** output directory and leaves the parent run/evidence intact:

```
pwsh -NoProfile -File tools/launch_rrc_protected.ps1 -RunName NEW_RECOVERY_NAME -Qualification analyses/rrc_v0_qualification_20261007_02.json -ResumeParent runs/rrc_v0_20261007_01
```

See `new/rrc_recovery/README.md` for planning and restored-state limits. The old
qualified RRC sources and old foreground launcher are untouched. The wrapper
checks the original qualification, GPU/disk headroom and nonexistent output.
Independent execution is not a change to model, loss, cadence or gate.
The recovery helper is specific to the original 137-stage snapshot and requires
its retained local checkpoints. It cannot resume a partially completed recovery
child after a second interruption. A fresh checkout needs its own locally
generated passing qualification before launch; private qualification/receipt
paths in these examples are not bundled as runnable checkpoint state.

Each guard directory under runs contains the private job plan, receipt,
heartbeat and stdout/stderr. The receipt has the exact scheduler task name.
After the worker completes, remove that specific completed registration:

```
Unregister-ScheduledTask -TaskName EXACT_TASK_NAME_FROM_RECEIPT -Confirm:$false
```

Task deletion is separate from output deletion. Keep job receipts locally for
diagnosis. Do not publish machine paths, account names or PIDs.

## Validation

`protected_job_smoke.py` is a24-second CPU sentinel using the registered Python
and Torch import; it does not train a model or create a CUDA context. A task is
started, the initiating tool returns, and a separate call verifies ongoing
heartbeat before recording its eventual exit code and power-request cleanup.
The test registration is then removed. No real experiment is launched by this test.
Public sanitized evidence is in
[`evidence/execution_protection_20261007_01/`](../evidence/execution_protection_20261007_01/README.md).
The original local receipt remains in `analyses/protected_execution_20261007_01.json`.
