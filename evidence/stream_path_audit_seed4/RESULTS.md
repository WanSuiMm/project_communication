# Streaming seed4: fixed-weight operator audit

Completed zero-training single-checkpoint diagnostic. Original replay passed.

| Condition | T | Size32 primary mean / pooled % | Primary correct / pixels | BA original / flipped % | Size64 d>32 pooled % |
|---|---:|---:|---:|---:|---:|
| full | 64 | 88.89 / 85.92 | 2507/2918 | 96.98 / 96.86 | 12.48 |
| full | 128 | 99.64 / 99.35 | 2899/2918 | 99.08 / 99.02 | 38.60 |
| full | 256 | 100.00 / 100.00 | 2918/2918 | 99.44 / 99.56 | 59.43 |
| no_transport | 64 | 0.00 / 0.00 | 0/2918 | 50.00 / 48.44 | 0.00 |
| no_transport | 128 | 0.00 / 0.00 | 0/2918 | 50.00 / 48.44 | 0.00 |
| no_transport | 256 | 0.00 / 0.00 | 0/2918 | 50.00 / 48.44 | 0.00 |
| no_perception | 64 | 0.00 / 0.00 | 0/2918 | 50.00 / 48.44 | 0.00 |
| no_perception | 128 | 0.00 / 0.00 | 0/2918 | 50.00 / 48.44 | 0.00 |
| no_perception | 256 | 0.00 / 0.00 | 0/2918 | 50.00 / 48.44 | 0.00 |
| neither | 64 | 0.00 / 0.00 | 0/2918 | 50.00 / 48.44 | 0.00 |
| neither | 128 | 0.00 / 0.00 | 0/2918 | 50.00 / 48.44 | 0.00 |
| neither | 256 | 0.00 / 0.00 | 0/2918 | 50.00 / 48.44 | 0.00 |

## Descriptive sensitivity

- no_transport: sustained_knockout_loss; primary pooled effects at T64/T128/T256: -85.92 / -99.35 / -100.00 pp.
- no_perception: sustained_knockout_loss; primary pooled effects at T64/T128/T256: -85.92 / -99.35 / -100.00 pp.
- neither: sustained_knockout_loss; primary pooled effects at T64/T128/T256: -85.92 / -99.35 / -100.00 pp.

Paired correctness requires both source alternatives to be correct at the same changed-region pixel.
All four conditions use identical weights, input banks and macro-step counts, but hop bounds per step are 2/2/1/0.
Removing transport also replaces the incoming feature read by F. Removing perception zeros both Laplacian slots in both F and Q.
Knockouts change learned state/input distributions. A loss is a dependency of this frozen solution under this intervention, not proof of training causality, unique semantics or inability to retrain.
This is selected seed4, n=1. Prior multi-seed DEVELOPMENT_NO_GO and seed4 positive evidence are unchanged.

Runtime: 12.36 seconds. Read summary.json, curves.csv and contrasts.json first; raw_conditions.json is secondary.
