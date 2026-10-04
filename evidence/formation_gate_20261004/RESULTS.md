# Continuation formation: saved-trajectory diagnostic

One selected historical training trajectory; zero training and zero inference.
u = training update; t = rollout step. Predictors read only t<=64.
The long behavioral screen uses T128/T256. All21 dense checkpoints and300 control are retained.

| Update | Survival32–64 | G32–64 | Newly solved survival32–64 | Margin reserve at64 | Early screen | Strict T128 | Survival64–256 | Long screen |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 100 | 0.8889 | 0.0365 | 0.4419 | 0.9476 | False | 0.0022 | 0.9587 | False |
| 105 | 0.8147 | 0.0379 | 0.5060 | 0.9163 | False | 0.0000 | 0.8936 | False |
| 110 | 0.8364 | 0.0722 | 0.4381 | 0.7428 | False | 0.0171 | 0.7428 | False |
| 115 | 0.9865 | 0.1133 | 0.8898 | 0.8963 | False | 0.1123 | 0.8621 | False |
| 120 | 0.9865 | 0.1203 | 0.8667 | 0.9190 | False | 0.1249 | 0.9964 | False |
| 125 | 0.9980 | 0.1925 | 0.9804 | 0.9709 | False | 0.2409 | 0.9970 | False |
| 130 | 0.9370 | 0.2458 | 0.9254 | 0.9542 | False | 0.3874 | 0.8811 | False |
| 135 | 0.9035 | 0.2022 | 0.8509 | 0.9454 | False | 0.4134 | 0.8025 | False |
| 140 | 0.9896 | 0.2598 | 0.9919 | 0.9764 | True | 0.4052 | 0.8611 | False |
| 145 | 0.9760 | 0.2543 | 0.9791 | 0.9760 | True | 0.4714 | 0.9194 | False |
| 150 | 0.9494 | 0.2497 | 0.9258 | 0.9821 | False | 0.4803 | 0.9047 | False |
| 155 | 0.9210 | 0.2044 | 0.8021 | 0.9637 | False | 0.4260 | 0.8901 | False |
| 160 | 0.9395 | 0.2680 | 0.8908 | 0.9699 | False | 0.5933 | 0.9109 | False |
| 165 | 0.9406 | 0.2641 | 0.9568 | 0.9857 | False | 0.6535 | 0.9039 | False |
| 170 | 0.9408 | 0.2778 | 0.9540 | 0.9729 | False | 0.6967 | 0.9102 | False |
| 175 | 0.9291 | 0.2751 | 0.9204 | 0.9768 | False | 0.7152 | 0.9248 | False |
| 180 | 0.4907 | 0.0617 | 0.4071 | 0.8657 | False | 0.1026 | 0.3566 | False |
| 185 | 0.8894 | 0.2168 | 0.7631 | 0.8686 | False | 0.5903 | 0.9117 | False |
| 190 | 0.9753 | 0.2256 | 0.8347 | 0.6986 | True | 0.5747 | 0.9439 | False |
| 195 | 0.9377 | 0.3324 | 0.9520 | 0.9115 | False | 0.8565 | 0.9453 | False |
| 200 | 0.9996 | 0.3762 | 0.9970 | 0.9904 | True | 0.9755 | 0.9997 | True |
| 300 | 1.0000 | 0.4804 | 1.0000 | 0.9785 | True | 0.9978 | 0.9997 | True |

Dense same-checkpoint confusion: {'TP': 1, 'FP': 3, 'TN': 17, 'FN': 0}.
Predicting next saved update (+5) confusion: {'TP': 0, 'FP': 3, 'TN': 16, 'FN': 1}.
Early candidate passes: [140, 145, 190, 200].
Long screen passes: [200].
Single-positive retrospective pairwise ordering: {'early_survival': 1.0, 'early_G': 1.0, 'early_new_survival': 1.0, 'margin_reserve': 1.0}.

These are retrospective diagnostics with one successful dense checkpoint. Thresholds were not fitted,
but the historical outcomes were already known when these candidate definitions were proposed.
No out-of-sample predictor, physical phase transition, abstract quotient, or causal commitment mechanism is established.
Early retention/progress are short-horizon behavior proxies. Margin reserve is a heuristic, not a uniform certificate.
A property of the rule at update175 cannot guarantee it survives the optimizer change to update180.
The failure to discover an early predictor lowers the evidential weight of these particular mechanism candidates;
it does not invalidate the conditional continuation theorem or all possible low-complexity representations.

All counts, strict-band variants and per-map denominators: [summary.json](summary.json).
All sizes/checkpoints: [profiles.csv](profiles.csv).
Frozen definitions: [protocol](../../new/formation_gate/PROTOCOL.md).
