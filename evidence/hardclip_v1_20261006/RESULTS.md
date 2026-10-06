# HardClip v1 fixed-tail qualification

The completed run has 16/16 trajectories and 208/208 predeclared evaluations. The formal endpoint is joint readiness at u300; the independent unit is the paired training block (n=8).

Calibration: **CALIBRATED** at four frozen actual-write RMS caps; no task outcomes or training updates entered calibration.
Primary verdict: **NO_HARDCLIP_RELIABILITY_QUALIFICATION**.

| Arm | u300 joint-ready | Blocks | Wilson 95% readiness interval | Old Full pass (secondary) |
|---|---:|---:|---|---:|
| neural | 0 | 8/8 | [0, 0.324408] | 0/8 |
| hardclip | 0 | 8/8 | [0, 0.324408] | 0/8 |

## Sole primary contrast

| Contrast | Complete pairs | HardClip-only wins | Neural-only wins | Net | Exact two-sided p |
|---|---:|---:|---:|---:|---:|
| hardclip − neural | 8 | 0 | 0 | 0 | 1 |

HardClip qualifies only when hardclip-only minus neural-only wins are at least 6 and the exact two-sided discordant-pair binomial p value is at most 0.05. This is the sole primary contrast; no Holm adjustment.

## Observed training dose

Dose sums pool the saved per-update open lane-cell events. Neural reports counterfactual clipping; HardClip reports actual clipping.

| Arm | Zero-trigger blocks | Blocks with any trigger | Aggregate trigger fraction | Aggregate removed fraction |
|---|---|---:|---:|---:|
| neural | 0, 1, 4, 5, 6 | 3 | 0.016323853278267127 | 0.003060503786300287 |
| hardclip | 0, 1, 4, 5, 6 | 3 | 0.014960562089219118 | 0.0028119305546224745 |

HardClip recorded zero trigger events in 5/8 paired blocks. The intervention dose was therefore absent in those blocks; this limits the negative result to the tested fixed intervention and does not refute all large-write mechanisms.

Earlier checkpoints are formation diagnostics only; no peak checkpoint replaces u300. The Old Full gate is a saved u300 evaluator result and remains secondary.

Finite evidence for this fixed K8 recipe and intervention. A negative result does not exclude all small benefits or all large-write mechanisms.

See [REPRODUCTION.md](REPRODUCTION.md), [final_metrics.csv](final_metrics.csv), [metrics.csv](metrics.csv), [training_dose.csv](training_dose.csv), and [validation.json](validation.json). All 208 evaluation summaries and dose records, 416 packed trace banks, and 16 training curves are retained in the package.
