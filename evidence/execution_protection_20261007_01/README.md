# Execution protection: bounded engineering validation

This update adds launch protection and an explicit checkpoint recovery path.
It adds no scientific measurements, resumes no research training, and leaves
the RRC partial snapshot and frozen qualification unchanged.

1. [Lifecycle validation](validation.json): a CPU sentinel was still running
   after its initiating tool returned, then exited with code zero. Temporary
   idle-sleep prevention was cleared. The task had no recurring trigger or
   execution-time limit; its test registration was removed.
2. [Recovery validation](recovery_validation.json): CPU validation of the
   original source binding, banks, schedules, committed stages and checkpoints;
   tiny save/restore parity for all three arms, including AdamW state.
3. [Execution wrapper](../../tools/PROTECTED_EXECUTION.md) and
   [recovery entrypoint](../../new/rrc_recovery/README.md).

The original run has 137 committed evaluation stages and ten completed u300
trajectories. Recovery can inherit those stages, restart the partial Factorized
trajectory from u150, and execute the remaining 175 stages in a new output
directory. The orphan u175 checkpoint is not treated as a committed stage.
This recovery helper is specific to the original interrupted snapshot; it
cannot resume a second interruption from a recovery child. It requires the
retained local parent checkpoints, which are excluded from GitHub.

The checks did not force a Codex service restart or reboot and did not exercise
CUDA recovery. The tiny CPU parity check performs synthetic optimizer updates;
it is an engineering test, not a new research training trajectory. Original
checkpoints lack backend RNG state, so exact uninterrupted CUDA equivalence is
not claimed. Idle-sleep prevention cannot prevent manual sleep, shutdown,
logout, lid policy or critical-battery actions.

Private job receipts, machine/account paths, process IDs, logs and checkpoints
are excluded. These small JSON files retain only the engineering checks and
source bindings needed for review.
