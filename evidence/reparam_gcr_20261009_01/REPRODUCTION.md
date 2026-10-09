# Saved-data reproduction

Read RESULTS.md, aggregate.json and final_metrics.csv first. Exactly one new
K8 arm ran150 updates; u150 is primary. Old GCR-K8/K64 references are locked in
the earlier evidence/learnable_gcr_20261009_01 package, not retrained or selected.
Their aggregate hash is bound in manifest.json. Earlier frozen results remain.

From repository root with CPU NumPy/Torch (no CUDA is used for verification):

```powershell
python -X utf8 -B tools/export_reparam_gcr.py --verify-only
```

The package retains exact token/path banks, teacher and shared schedule, all150
training records, three checkpoint bindings and stage evaluations, and all
saved per-example predictions for seven final metric rows. NPZ arrays load
with allow_pickle=False. Weights/optimizer contents/private receipts stay local.

Qualification records per-step forward and full-gradient equivalence at initial
and old trained-reference parameters, plus CUDA short-credit gradient/cut checks.
No training or inference occurs during publication. Checkpoint bindings describe
local pre-export verification; omitted checkpoint contents cannot be rechecked
from hashes alone. Re-running the full prelaunch qualification requires the
retained old GCR-K64/u150 checkpoint. To regenerate it from scratch, first follow
the prior learnable_gcr protocol, then the new/reparam_gcr/PROTOCOL.md commands
using fresh names in the declared CUDA environment.

The verdict is NEAR_FULL_CREDIT_DEVELOPMENTAL on one known paired block. The
intervention changes state21->73 scalars while preserving5921 trainable parameters
and the full forward function. It supports late parameter placement for this
linear degree-two task, not generic learned NCA recurrence, reliability, or
early-input gradient recovery. Known validation cohorts are reused.
