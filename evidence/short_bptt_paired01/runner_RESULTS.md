# Matched short-BPTT screen

Execution complete: True. Frozen protocol: new/short_bptt/PROTOCOL.md.

Size32/T64; far means graph distance >16. Percentages; model seed is independent unit.

| Cell | Seed | Full BA | Short BA | Full far paired | Short far paired | Far effect pp | Decision | Hold predicates |
|---|---:|---:|---:|---:|---:|---:|---|---|
| ws_additive | 0 | 87.20 | 91.22 | 15.83 | 23.18 | +7.36 | BASELINE_UNQUALIFIED | False |
| ws_additive | 1 | 87.73 | 98.03 | 0.00 | 80.87 | +80.87 | BASELINE_UNQUALIFIED | True |
| ws_revision | 0 | 78.04 | 83.90 | 21.44 | 0.67 | -20.77 | BASELINE_UNQUALIFIED | False |
| ws_revision | 1 | 86.16 | 64.38 | 0.00 | 17.81 | +17.81 | BASELINE_UNQUALIFIED | False |

Full controls that fail qualification prevent interpreting a truncation failure.
Hold alone cannot rescue failed reach. Size64 and longer rollout are secondary.
See per-arm JSON for all horizons, per-map scores, distance bins, timing and memory.
