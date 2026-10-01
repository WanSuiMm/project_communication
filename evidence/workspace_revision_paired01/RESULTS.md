# Workspace + revision paired screen

Decision: NO_JOINT_SCREEN_PASS.
Two-seed exploratory screen; not a population-level superiority claim.

| Arm | Seed | Status | Updates | Size | BA64 | Hold min BA | Paired64 | Switch changed K64 | Z-repair BA K64 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ws_additive | 0 | COMPLETE | 600 | 32 | 94.29% | 94.18% | 65.02% | 13.85% | 94.07% |
| ws_additive | 0 | COMPLETE | 600 | 64 | 76.72% | 76.66% | 21.20% | 36.42% | 76.60% |
| ws_revision | 0 | COMPLETE | 600 | 32 | 99.19% | 100.00% | 98.00% | 0.00% | 100.00% |
| ws_revision | 0 | COMPLETE | 600 | 64 | 95.99% | 98.93% | 80.13% | 2.13% | 96.91% |
| ws_revision | 1 | COMPLETE | 600 | 32 | 50.00% | 50.00% | 0.00% | 37.50% | 50.00% |
| ws_revision | 1 | COMPLETE | 600 | 64 | 50.00% | 50.00% | 0.00% | 43.75% | 50.00% |
| ws_additive | 1 | COMPLETE | 600 | 32 | 78.54% | 73.86% | 26.13% | 41.58% | 78.55% |
| ws_additive | 1 | COMPLETE | 600 | 64 | 57.79% | 50.20% | 5.41% | 56.03% | 53.16% |

Mean paired hold effect (revision - additive): -9.019887447357178 pp.
See aggregate.json for every frozen predicate; raw arm JSON retains all endpoints/controls.
Hold = minimum aggregate BA at T128/192/256. Null is unavailable, never success.
Repair uses specified damage only; source switch retains W and Z.
Prior checkpoint scores use different training and are not matched controls.
