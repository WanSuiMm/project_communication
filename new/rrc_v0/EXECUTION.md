# RRC-v0 local execution

Read [PROTOCOL.md](PROTOCOL.md) first. `cells.py` defines the three transitions,
`activity.py` measures actual relation action, `run.py` reuses the fixed
CUDA Graph K8/runtime and paired evaluation, and `reporting.py` applies the
single frozen RRC-vs-Factorized contrast. No existing code or result is overwritten.

Qualification: `analyses/rrc_v0_qualification_20261007_02.json`.
Formal run: `runs/rrc_v0_20261007_01`; planned24 trajectories and312 evaluations.
Source binding includes all new Python/Markdown and the launcher plus existing
runtime/evaluation dependencies. Private local launch receipts record the worker
PID, GPU, command and owning persistent exec session.

Use the foreground PowerShell launcher in PROTOCOL.md. Its parent tool session
must remain alive; detached tool-shell children are not a reliable launch method
on this host. The one-time `tools/verify_rrc_v0_dispatch.ps1` records progress
after the initial tool yield. No runtime cap or recurring monitor is installed.

All saved checkpoints are retained locally. Outputs use compressed bit-packed
correctness traces rather than dense floating trajectories. Timing and size
estimates come from qualification, not a promise of final wall-clock duration.
u300 is the formal endpoint; intermediate ready checkpoints cannot rescue it.
