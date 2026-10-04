# Output-preserving state factorization

Status: COMPLETE. Zero training/optimizer updates.
Primary episode: LOCALIZATION_EPISODE_UNQUALIFIED. Confirmation: UNQUALIFIED.

| Size | Arm | Keep continuous | Prog immediate | Prog sustained | Prog delayed sustained | Rescue proxy | Neutral output |
|---:|---|---:|---:|---:|---:|---|---|
| 32 | FF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True |
| 32 | F-span | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | visible |
| 32 | F-null | 0.1102 | 0.0000 | 0.0000 | 0.0000 | False | True |
| 32 | FS | 0.0089 | 0.0000 | 0.0000 | 0.0000 | False | visible |
| 32 | SF | 1.0000 | 0.0000 | 0.5152 | 0.5152 | False | True |
| 32 | S-span | 0.9938 | 0.0000 | 0.8009 | 0.8009 | False | visible |
| 32 | S-null | 1.0000 | 0.0000 | 0.5152 | 0.5152 | False | True |
| 32 | SS | 0.9954 | 0.0000 | 0.9567 | 0.9567 | False | visible |
| 64 | FF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True |
| 64 | F-span | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | visible |
| 64 | F-null | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True |
| 64 | FS | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | visible |
| 64 | SF | 1.0000 | 0.0000 | 0.6730 | 0.6730 | False | True |
| 64 | S-span | 0.9898 | 0.0000 | 0.7746 | 0.7746 | False | visible |
| 64 | S-null | 1.0000 | 0.0000 | 0.6805 | 0.6805 | False | True |
| 64 | SS | 0.9924 | 0.0000 | 0.8788 | 0.8788 | False | visible |

Read summary.json and metrics.csv first. Native Full, transfer/localization proxy and
output neutrality are separate qualifications. Cohorts/denominators are fixed before intervention.
Hybrid failures do not establish a W/Z consistency relation; successes are conditional state effects.

## Qualification denominators (publication annotation)

Native Full: S1=True, F1=False. All sixteen intervention rescue proxies are
unqualified by the frozen cohort-support rule. Size32 shared-unsolved covers
6/32 maps and231 cells; size64 shared-solved covers9/32 maps and393 cells.
The frozen minimum is16 nonempty maps and100 cells for BOTH keep/prog.
Thus the gate flags are not a claim that all continuous behaviors failed.
Size64 is confirmation and does not replace the primary eligibility check.

SF keeps the immediate F1 output unchanged. Its continuous keep rate is1.0
at both sizes; shared-unsolved sustained/delayed rates are.5152 (32) and.6730
(64). These selected-checkpoint measurements remain descriptive because the
frozen support qualification failed. No population or mechanism claim follows.

Read summary.json (compact derived aggregate), metrics.csv, validation.json
and config.json first. raw_summary.json (4.6MB) and raw_RESULTS.md preserve the
original run summaries unchanged. Arm JSON, native records and66 NPZ files
retain all per-map cohorts, counts, traces, logits, scales and native states.
