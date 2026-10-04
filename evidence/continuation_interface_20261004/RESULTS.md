# Continuation interface audit

Status: COMPLETE. Common native Full qualification: True.
Gauge control: PASS. No model training or optimizer steps.

| Mode | Producer | Consumer | Transfer proxy |
|---|---|---|---:|
| raw | S1 | S1 | True |
| raw | S1 | S2 | False |
| raw | S1 | F1 | True |
| raw | S1 | F2 | False |
| raw | S1 | F3 | False |
| raw | S2 | S1 | False |
| raw | S2 | S2 | True |
| raw | S2 | F1 | False |
| raw | S2 | F2 | False |
| raw | S2 | F3 | False |
| raw | F1 | S1 | False |
| raw | F1 | S2 | False |
| raw | F1 | F1 | False |
| raw | F1 | F2 | False |
| raw | F1 | F3 | False |
| raw | F2 | S1 | False |
| raw | F2 | S2 | False |
| raw | F2 | F1 | False |
| raw | F2 | F2 | False |
| raw | F2 | F3 | False |
| raw | F3 | S1 | False |
| raw | F3 | S2 | False |
| raw | F3 | F1 | False |
| raw | F3 | F2 | False |
| raw | F3 | F3 | False |
| aligned | S1 | S1 | True |
| aligned | S1 | S2 | False |
| aligned | S1 | F1 | False |
| aligned | S1 | F2 | False |
| aligned | S1 | F3 | False |
| aligned | S2 | S1 | True |
| aligned | S2 | S2 | True |
| aligned | S2 | F1 | False |
| aligned | S2 | F2 | True |
| aligned | S2 | F3 | False |
| aligned | F1 | S1 | False |
| aligned | F1 | S2 | False |
| aligned | F1 | F1 | False |
| aligned | F1 | F2 | False |
| aligned | F1 | F3 | False |
| aligned | F2 | S1 | False |
| aligned | F2 | S2 | False |
| aligned | F2 | F1 | False |
| aligned | F2 | F2 | False |
| aligned | F2 | F3 | False |
| aligned | F3 | S1 | False |
| aligned | F3 | S2 | False |
| aligned | F3 | F1 | False |
| aligned | F3 | F2 | False |
| aligned | F3 | F3 | False |

Read summary.json, selection.json, phase0/comparisons.json, alignment/validation.json,
native/*_phenotype.json, cross/decomposition.json and per-cell JSON before the larger arrays.
Negative alignment qualifies only the tested restricted linear interface family.

## Recorded qualification flags

- Restricted off-diagonal maps qualified: 0/20.
- Matched short-observable flags: 0/6.
- Transfer proxies and restricted-map qualification are separate saved tests.
- F2/F3 are nearest short-output candidates, not guaranteed matched observables.
- This release contains data and code; no new mechanism interpretation.
