# StreamingCell Full reevaluation

Status: **COMPLETE**, 6/6 model/cohort units. No training or optimizer updates.

Selected-checkpoint descriptive evaluation. Reused unchanged historical Full predicate.
The addressed cohort selected the primary models; it is not independent validation.

| Checkpoint | Role | Cohort | Full | Failed gates |
|---|---|---|---|---|
| block00_u300 | primary | historical_full | PASS | 0 |
| block00_u300 | primary | addressed_selection | FAIL | 1 |
| block02_u300 | primary | historical_full | FAIL | 3 |
| block02_u300 | primary | addressed_selection | FAIL | 2 |
| block00_u275 | diagnostic | historical_full | PASS | 0 |
| block00_u275 | diagnostic | addressed_selection | PASS | 0 |

## Complete gate values and evidence

- block00_u300 / historical_full: [full summary](block00_u300/historical_full/summary.json), [matched frontier](block00_u300/historical_full/matched_frontier_strata.csv).
- block00_u300 / addressed_selection: [full summary](block00_u300/addressed_selection/summary.json), [matched frontier](block00_u300/addressed_selection/matched_frontier_strata.csv).
  size32_ever_regressed_over_ever_correct: observed 0.16, requires <=0.15
- block02_u300 / historical_full: [full summary](block02_u300/historical_full/summary.json), [matched frontier](block02_u300/historical_full/matched_frontier_strata.csv).
  size32_ever_regressed_over_ever_correct: observed 0.32238959701300374, requires <=0.15
  size64_all_changed_retention64_to256: observed 0.9264164041223901, requires >=0.95
  size64_ever_regressed_over_ever_correct: observed 0.32865052718383697, requires <=0.15
- block02_u300 / addressed_selection: [full summary](block02_u300/addressed_selection/summary.json), [matched frontier](block02_u300/addressed_selection/matched_frontier_strata.csv).
  size32_ever_regressed_over_ever_correct: observed 0.27270327349524814, requires <=0.15
  size64_ever_regressed_over_ever_correct: observed 0.2588533647314555, requires <=0.15
- block00_u275 / historical_full: [full summary](block00_u275/historical_full/summary.json), [matched frontier](block00_u275/historical_full/matched_frontier_strata.csv).
- block00_u275 / addressed_selection: [full summary](block00_u275/addressed_selection/summary.json), [matched frontier](block00_u275/addressed_selection/matched_frontier_strata.csv).

Each unit retains losslessly packed size32/size64 original, flipped and paired
correctness arrays at every integer time0..256, plus per-map summaries and full frontier strata.
No intermediate diagnostic checkpoint replaces u300. Pass counts are not a formation rate.
