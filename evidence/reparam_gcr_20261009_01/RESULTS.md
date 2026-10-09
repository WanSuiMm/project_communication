# Reparameterized GCR K8: one-block developmental result

Execution: **COMPLETE**. Frozen verdict: **NEAR_FULL_CREDIT_DEVELOPMENTAL**.

| Primary held-out T64 R2 | Value |
|---|---:|
| Old GCR K8 (locked) | 0.260103 |
| Old GCR K64 (locked) | 0.535750 |
| Reparam GCR K8 | 0.535705 |

New-minus-old K8: 0.275601; new-minus-old K64: -0.000046.
Elapsed seconds: 20.144. Exactly one new150-update arm was trained.

Forward/full-gradient equivalence and short-credit checks are bound in qualification.json.
Raw sufficient state is locally produced, with73 instead of21 per-node scalars and the same5921 learned parameters.
Known held-out cohorts are reused; u150 is primary and intermediate checkpoints do not select the model.
This tests late parameter placement for a linear degree-two task, not generic learned recurrence or autonomous NCA stability.
The earlier Full Writer comparison remains control-unqualified. All predictions and metrics are in the saved NPZ and aggregate.json.
