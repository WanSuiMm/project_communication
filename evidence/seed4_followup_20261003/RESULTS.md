# Original seed4: independent schedule, initialization and rollback screens

A/B execution COMPLETE: 13 arms x300 updates, 1866.672s (31.11min).
Historical control: identical initial/final parameters and six full saved endpoints; fresh control phenotype QUALIFIED.

| Experiment | Independent unit | Full phenotype passes | Frozen threshold | Result |
|---|---|---:|---:|---|
| A: fixed init, changed schedule | 4 schedules | 0/4 | >=3/4 | NOT_QUALIFIED |
| B: epsilon=.01, fixed training | 4 directions | 0/4 | >=3/4 | NOT_QUALIFIED |
| B: epsilon=.05, same paired directions | 4 directions | 0/4 | >=3/4 | NOT_QUALIFIED |

All 12 A/B arms pass both matched-frontier gates, but fail the complete phenotype. Frontier association alone is insufficient for retained useful computation.

| Arm | T64 strict mean % | T64 strict pooled % | Reach | Hold | Dynamics | Frontier | Full |
|---|---:|---:|---|---|---|---|---|
| control | 93.41 | 90.42 | True | True | True | True | True |
| A_schedule20012 | 35.06 | 31.48 | False | False | False | True | False |
| B_direction70002_eps01 | 75.81 | 68.00 | False | False | False | True | False |
| B_direction70002_eps05 | 17.74 | 10.30 | False | False | False | True | False |
| A_schedule20022 | 81.67 | 79.47 | False | False | False | True | False |
| B_direction70003_eps05 | 89.41 | 86.08 | True | True | False | True | False |
| B_direction70003_eps01 | 46.30 | 42.65 | False | False | False | True | False |
| A_schedule20032 | 67.43 | 62.00 | False | True | False | True | False |
| B_direction70004_eps01 | 61.52 | 56.15 | False | False | False | True | False |
| B_direction70004_eps05 | 74.62 | 69.14 | False | False | False | True | False |
| A_schedule20042 | 80.18 | 71.65 | False | False | False | True | False |
| B_direction70005_eps05 | 50.50 | 40.68 | False | False | False | True | False |
| B_direction70005_eps01 | 26.58 | 24.97 | False | False | False | True | False |

Reach: size32/T64 strict16<d<32 paired mean and pooled>=.80; both open-grid BA>=.85.
Hold: size32/T128 AND T256 drops from T64 <=.03 BA and <=.05 strict paired mean/pooled.
Dynamics: at each spatial size, T64->T256 retention>=.95, coverage gain>=.05, every-step regression/ever-correct<=.15.
Frontier: each size matched effect>=.05, eligible maps>=16, common strata>=100. Empty groups fail.

Schedule20032 passes hold, retention and coverage-gain checks, but reaches only 67.4% mean/62.0% pooled at T64 and regresses 37.2%/40.4% of ever-correct pixels.
Direction70003/epsilon=.05 passes size32 reach+hold, but fails parts of the size64 dynamics and regresses 26.8%/71.3%. No radius monotonicity or basin boundary follows.

## C: selected checkpoint local temporal-state intervention

Zero training; 93/257 events at size32/64, 31 eligible maps each. Historical six-endpoint replay and numerical controls pass.

| Size | Horizon | Native minus sender pp | Wrong-sham minus sender pp | Endpoint |
|---:|---:|---:|---:|---|
| 32 | 1 | 14.52 | 3.76 | Primary |
| 32 | 4 | 33.53 | 20.18 | Secondary |
| 64 | 1 | 7.00 | 0.44 | Primary |
| 64 | 4 | 28.60 | 15.33 | Secondary |

**NO_PRIMARY_THRESHOLD_SIGNAL**: step1 requires native-minus-sender>=.10 AND sham-minus-sender>=.05 at both sizes. Size32 misses the sham threshold; size64 misses both.
Larger step4 contrasts are secondary and do not rescue this endpoint. The threshold miss is not a statistical null test.
This is a whole-cell W/Z temporal rollback on selected targets, not an edge-specific message knockout. Equal-norm sham does not match phase, direction or activation context. Model n=1, reused historical maps.

## Reading order and reproduction

1. This report, [interpretation](INTERPRETATION.md), and [compact aggregate](summary.json).
2. [Frozen protocol](../../new/seed4_followup/PROTOCOL.md), [compact training validation](training/validation.json), [causal validation](causal/validation.json), and [public arithmetic check](publication_validation.json).
3. Individual training/*_summary.json files retain every gate, denominator, map/censor profile. raw/ retains original arm records. frontier/ CSVs and causal/events.json are secondary; do not open them first.
4. [Bindings](../../SEED4_FOLLOWUP_PUBLICATION_MANIFEST.json) and [provenance](provenance.json) identify byte-identical and sanitized evidence; NPZ/checkpoints remain local.

Public verification and focused CPU checks from repository root:

```
python tools/export_seed4_followup.py --verify-only
python new/seed4_followup/check.py --out analyses/NEW_FOLLOWUP_CPU.json
```

For a fresh clone, `python tools/prepare_seed4_followup_training_references.py` restores three already-public archival inputs without overwriting an existing archive. Then use the frozen train.py preflight/formal commands in [reproduction notes](REPRODUCTION.md). No checkpoint is needed to retrain A/B.
Exact C replay needs the original excluded checkpoint/source archives. CPU publication checks validate saved arithmetic; they do not repeat GPU inference or training.
Earlier architecture no-go decisions remain unchanged. No general NCA rejection, training basin radius, short-BPTT mechanism or unique handoff law follows.
