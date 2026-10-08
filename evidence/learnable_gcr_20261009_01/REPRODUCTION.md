# Saved-data reproduction

Read RESULTS.md, aggregate.json and final_metrics.csv first. This is a new
ordered-path task, not Region Identity or historical StreamingCell. The frozen
verdict is POSITIVE_CONTROLS_UNQUALIFIED because Full Writer K64 did not reach
held-out R2 >= .5. GCR K64 qualified individually; neither that nor its K8 result
establishes improved short-credit learning. One block is not a reliability study.

From repository root (CPU NumPy and Torch are sufficient; no CUDA is used):

```powershell
python -X utf8 -B tools/export_learnable_gcr.py --verify-only
```

The package retains all750 training updates,15 checkpoint bindings and their
held-out evaluations, all35 final metric rows, all saved per-example predictions,
four exact token/path banks, the teacher specification and the shared schedule.
NPZ archives use ordinary numeric arrays and load with allow_pickle=False.
The diagnostic's predictions are horizon-independent, since it reads offline
raw8D sufficient statistics and performs no recurrent execution.

Weights, optimizer checkpoint contents and machine-specific receipts remain
local. Source hashes bind the exact executed files, which match new/learnable_gcr
and the existing protected launcher/worker in this repository. Export verification
does not train or run a model. Verification of omitted checkpoint contents is
represented by the checkpoint binding hashes and local pre-export validation.

For a new scientific run, follow new/learnable_gcr/PROTOCOL.md using fresh output
and qualification names. It requires the declared Torch2.5.1 CUDA environment.
Use run.py --check, then run.py --out with --qualification, or the protected
on-demand launcher. The sole primary endpoint is u150; u50/u100 are diagnostics.
