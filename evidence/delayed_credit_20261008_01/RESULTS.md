# Delayed-credit cue-once screen

Execution status: COMPLETE.
Recorded trajectories: 12/12.
Verdict: BASELINE_UNQUALIFIED.
Complete: True.

| Arm | Reach pass | Reach fail | Unknown |
| --- | ---: | ---: | ---: |
| dense_k8 | 0/4 | 4/4 | 0/4 |
| terminal_k8 | 0/4 | 4/4 | 0/4 |
| terminal_k64 | 0/4 | 4/4 | 0/4 |

| Block | A dense K8 | B terminal K8 | C terminal K64 |
| ---: | --- | --- | --- |
| 0 | fail | fail | fail |
| 1 | fail | fail | fail |
| 2 | fail | fail | fail |
| 3 | fail | fail | fail |

C-qualified: 0/4; B-qualified: 0/4.
All-block B/C reach contrasts: {'C_pass_B_fail': 0, 'B_pass_C_fail': 0, 'both_pass': 0, 'both_fail': 4, 'unknown': 0}.
C-qualified B/C strict-band map-mean scores: {'wins': 0, 'losses': 0, 'ties': 0, 'unscored': 0}.
T128/T256 are secondary; the primary gate is size 32 at T64.
Independent unit: paired model block; maps and pixels are within-block measurements.
A failure does not invalidate a qualified B/C contrast; it leaves any dense-rescue claim unqualified.
Full statuses, errors, and incomplete records are in summary.json.
