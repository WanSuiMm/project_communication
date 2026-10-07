# Local execution: full-writer port relations

This is a new experiment under `port_relation_v1_full_writer_native_k8`.
Existing RRC-v0 code, checkpoints and interrupted evidence remain unchanged.
The scientific protocol is in [PROTOCOL.md](PROTOCOL.md).

Run from the standalone repository root with the established Torch 2.5.1 CUDA
environment. The single implementation qualification runs the CPU cell/report
checks, then two matched eager/CUDA Graph optimizer updates per arm at the
actual batch-eight, size-32 shape. Qualification-only nonzero tail/relation
weights exercise the branches; formal training uses the prescribed zero
initialization. One full two-size evaluation per arm checks the recorder and
measures runtime/storage. Qualification requires finite outputs and matching
loss, boundary state, gradients, optimizer moments and action telemetry.

```powershell
python -X utf8 -u -B new/port_relation/run.py --check --out analyses/port_relation_qualification_20261007_01.json
pwsh -NoProfile -File tools/launch_port_relation.ps1 -RunName port_relation_20261007_01 -Qualification analyses/port_relation_qualification_20261007_01.json
```

The launcher checks the passing source binding and GPU/disk headroom, then
dispatches a unique on-demand Task Scheduler task through the previously
tested [independent worker](../../tools/PROTECTED_EXECUTION.md). The worker
outlives the originating tool call, requests temporary idle-sleep prevention,
and records a private heartbeat, child PID, native exit code and stderr under
the new guard directory. No recurring trigger, watchdog or time cap is set.
Dispatch verification is separate from the model's scientific endpoint.

All eight fresh initializations and batch schedules are recorded before the
first training update. All common E/F/Q/readout initialization hashes match
within each block. The source snapshot, bank hashes and fixed minibatch plan
are retained. Native parameters/moments/RNG are saved at u0 and every 25 updates;
packed Boolean traces retain all integer execution times through T256.
Reports, dense rows and checkpoint files are written before the stage's atomic
`.complete.json` marker. An incomplete stage is never inherited as complete.

Explicit recovery, when requested, uses a new output directory:

```powershell
pwsh -NoProfile -File tools/launch_port_relation.ps1 -RunName NEW_CHILD_NAME -Qualification analyses/port_relation_qualification_20261007_01.json -ResumeParent runs/port_relation_20261007_01
```

### Telemetry overflow repair, 2026-10-07

The original run stopped during conditioned u175 evaluation after two completed
trajectories. Its two-size T256 state/logit finite checks and Boolean trace
recording had succeeded. The observer squared float32 carrier values before its
float64 accumulation, producing `inf` that strict JSON rejected. The original
ERROR/INCOMPLETE evidence is retained unchanged.

The repair casts diagnostic operands to float64 before scaling, squaring and
reduction. It does not change model arithmetic, parameters, loss, optimization,
batch plans, banks, formal endpoint or failure criteria. A focused large-value
regression and the actual failed u175 checkpoint are checked; the latter receives
no optimizer update and must preserve both packed Boolean trace banks exactly.
The patched source receives its own actual-shape qualification.

```powershell
python -X utf8 -u -B new/port_relation/run.py --check --out analyses/port_relation_qualification_20261007_02.json --resume-parent runs/port_relation_20261007_01
pwsh -NoProfile -File tools/launch_port_relation.ps1 -RunName port_relation_20261007_02 -Qualification analyses/port_relation_qualification_20261007_02.json -ResumeParent runs/port_relation_20261007_01
```

The new qualification records an explicit parent/current source bridge allowing
only the telemetry implementation, its CPU regression, the cached evaluation
implementation, this execution note and runner recovery/check plumbing to change.
Parent source snapshots,
qualification, model/configuration, committed evaluations, checkpoint hashes and
curve files remain bound. The child inherits33 committed stages and resumes the
conditioned trajectory from u150; the uncommitted u175 stage is not inherited.
New stage markers also bind the corresponding training-curve prefix.
Inherited activity records retain their original diagnostic implementation;
their finite float32-squared energy summaries may differ slightly from new
float64 diagnostics and are not model endpoint definitions.

### Cached evaluation acceleration

Training retains all four cold64 segments and32 K8 windows per super-update;
the repeated segments are not collapsed or assigned a smaller physical budget.
For evaluation, CUDA Graphs cache the fixed-bank two-world T0..T256 recorder once
per model and size. Later checkpoint evaluations replay those kernels with live
parameter storage. Finite-state, source-cone, scoring, packed trace, metrics and
readiness rules remain the frozen evaluator's definitions. All activity buffers
are reset between evaluations, and caches are released between trajectories.
The repair qualification checks repeated replay and loading another checkpoint
into the same cached model against the original saved Boolean traces. Timings
separate the first evaluation including capture from subsequent cached replay.
Dense intermediate reports compute ever-correct and relapse directly with
Boolean reductions instead of computing unused first/stable passage times.
The full u300 report retains its frozen time-profile and frontier routines.
Qualification compares complete report dictionaries, not only trace arrays,
against the original u175/u150 artifacts.

This runner can recover any committed prefix of its own protocol, including a
recovery child after another interruption. It validates source/configuration,
banks, schedules, stage markers and restored parameter/Adam step bindings;
completed arms are inherited and the next saved minibatch is used. The parent
is read-only. Unsaved updates after the last committed checkpoint are replayed.
Torch CPU/CUDA RNG is retained, but the recorded backend is nondeterministic,
so bitwise equivalence to uninterrupted CUDA execution is not promised.

Temporary idle prevention cannot prevent manual sleep, logout, shutdown, lid
policy or critical-battery actions. Private receipts and checkpoints remain
local. The compact report and numerical-failure markers distinguish completed,
failed and interrupted records; no best intermediate checkpoint replaces u300.
