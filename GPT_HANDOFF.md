# Incremental review: execution protection and fixed-parent recovery

- Review base: `c316425d885d999829b3bc1a17fdc6721901de94`.
- Engineering evidence head: `67adfa60c70e4b9884412d8073168ffddc050315`.
- Scientific evidence head remains `8c5b11afca26cf6205686f51e79acf495af4e120`.
- RRC publication stays `PARTIAL_INTERRUPTED`; formal verdict stays `INCOMPLETE`.
- No research training was resumed. No RRC-v1 was implemented or launched.
- This later handoff commit changes review metadata only.

## Read first

1. [Engineering validation](evidence/execution_protection_20261007_01/README.md),
   [lifecycle proof](evidence/execution_protection_20261007_01/validation.json),
   and [recovery proof](evidence/execution_protection_20261007_01/recovery_validation.json).
2. [Execution contract](tools/PROTECTED_EXECUTION.md),
   [launcher](tools/start_protected_job.ps1), and
   [worker](tools/protected_job_worker.ps1).
3. [Recovery scope](new/rrc_recovery/README.md),
   [entrypoint](new/rrc_recovery/run.py), and
   [CPU checker](new/rrc_recovery/check_recovery.py).
4. [Publication bindings](EXECUTION_PROTECTION_PUBLICATION_MANIFEST.json).

## Changed engineering evidence

System events show the Codex sandbox service was asked to stop and stopped
seconds after the last RRC progress, then restarted. Service teardown is the
likely trigger; the native Python exit cause and requester remain unproved.

The launcher now dispatches a unique on-demand Task Scheduler task under the
interactive user, with no password, recurring trigger, automatic retry or time
cap. A CPU sentinel continued after the initiating tool exited, returned code
zero, and cleared its temporary idle-sleep request. Its task registration was
removed. No forced service restart or reboot was tested.

Recovery validates the full frozen configuration, source bundle, schedules,
banks, recorded stages and latest committed model/AdamW checkpoints on CPU.
It copies only committed stages into a distinct child. It skips ten completed
trajectories, resumes the partial Factorized trajectory from u150, and can
execute the remaining 175 stages. The orphan u175 stage is ignored.
Tiny CPU save/restore parity passed for all three arms. The public proof binds
that test's source version separately from the final metadata/validation changes.
The final source audit also passed; CUDA remained uninitialized.

## Unchanged claims and limits

The scientific snapshot still has ten of 24 u300 trajectories and 137 of 312
fully recorded checkpoint evaluations. Joint readiness remains Current1/4,
Factorized0/3, RRC0/3, with only three of eight primary pairs. Old Full remains
0/4,0/3,0/3. Do not reinterpret these incomplete denominators as final rates.
The old qualified runner, launcher, cells, protocol and evidence are unchanged.

CUDA recovery is unexecuted. The helper requires the retained local parent
checkpoints and is specific to the original 137-stage snapshot; it cannot resume
a recovery child after a second interruption. Runtime GPU comparison is by
model name, not device identity. Original backend RNG state is unavailable,
so exact uninterrupted CUDA equivalence is not claimed. The resumed arm's
capture setup timing is explicitly scoped to the resumed session.
Temporary idle prevention does not cover manual sleep, logout or power loss.

Review the process ownership and recovery commit boundaries. Do not use this
engineering update as new evidence for any architecture mechanism.
