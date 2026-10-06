# Background-process lifetime repair

The first dispatch was interrupted after its first real optimizer update.
Its Python process was absent; status remained RUNNING, stderr was empty, and
there was no Python error artifact. Preserve that run and its source snapshot.
It is an incomplete execution, not an efficacy result.

A short CPU-only process-lifetime control reproduced the launch problem:
`Start-Process -WindowStyle Hidden` ran while its launching tool shell was alive,
but its heartbeat stopped after that shell returned. The same 30-second helper
run in a live foreground exec session completed all 30 seconds across separate
tool calls. This control concerns process lifetime, not model arithmetic.

The repaired launcher executes Python in the foreground of a persistent tool
session. That shell stays alive until Python finishes. A separate `-Verify`
invocation checks the worker and launcher processes, the manifest, and a real
optimizer update after the launch tool has yielded. It writes the durable
verification receipt. This is bounded dispatch verification, not a monitor.

Restart from the identical initialization and schedules in a new run directory.
No training state is carried across the interrupted dispatch. The architecture,
data, seeds, clock, loss, gradient arithmetic, optimizer, checkpoint grid, and
formal decision rules remain unchanged. Qualification binds the repaired source
snapshot. No runtime cap, watchdog, scheduled task, or recurring poll is added.

Launch in a long-running tool session, then verify from a separate call:

    pwsh -File tools/launch_continuous_coverage.ps1 -RunName NEW_RUN -Qualification analyses/NEW_CHECK.json
    pwsh -File tools/launch_continuous_coverage.ps1 -RunName NEW_RUN -Qualification analyses/NEW_CHECK.json -Verify

The first command must remain running; its returned tool session is the worker's
execution owner. A returned session ID is not an experiment completion.
