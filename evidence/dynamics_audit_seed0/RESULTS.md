# Generic NCA dynamics audit

Execution status: `COMPLETE`. No training or model modification.
Elapsed seconds: 112.9. One training seed; maps are diagnostic examples, not training replicates.

Read manifest.json for frozen settings. Raw curves and per-map dynamics are separate files.

| Arm | Size | T | BA % | BCE | Paired % | Open K16 median log gain | Converged estimates / maps | H mean-mode energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nca_state_matched | 32 | 16 | 77.45 | 0.4071 | 26.33 | 0.26456 | 3/4 | 0.5876 |
| nca_state_matched | 32 | 32 | 85.86 | 0.2687 | 44.65 | 0.23716 | 4/4 | 0.4918 |
| nca_state_matched | 32 | 48 | 85.40 | 0.2590 | 46.24 | 0.18091 | 3/4 | 0.4638 |
| nca_state_matched | 32 | 64 | 85.17 | 0.2579 | 49.31 | 0.16076 | 4/4 | 0.4388 |
| nca_state_matched | 32 | 96 | 83.71 | 0.2741 | 40.41 | 0.13651 | 4/4 | 0.4074 |
| nca_state_matched | 32 | 128 | 83.81 | 0.2895 | 39.05 | 0.11434 | 3/4 | 0.3913 |
| nca_state_matched | 32 | 192 | 83.03 | 0.3379 | 40.17 | 0.09263 | 3/4 | 0.3741 |
| nca_state_matched | 32 | 256 | 82.85 | 0.4112 | 38.44 | 0.08870 | 1/4 | 0.3645 |
| nca_state_matched | 64 | 16 | 63.01 | 0.5762 | 7.81 | 0.30567 | 2/4 | 0.6551 |
| nca_state_matched | 64 | 32 | 72.43 | 0.6880 | 13.07 | 0.29085 | 4/4 | 0.4756 |
| nca_state_matched | 64 | 48 | 73.98 | 1.8882 | 13.57 | 0.23398 | 4/4 | 0.5210 |
| nca_state_matched | 64 | 64 | 73.68 | 3.3162 | 13.18 | 0.19223 | 4/4 | 0.5385 |
| nca_state_matched | 64 | 96 | 73.16 | 6.0317 | 12.93 | 0.17192 | 4/4 | 0.5395 |
| nca_state_matched | 64 | 128 | 73.27 | 8.7634 | 12.99 | 0.13960 | 4/4 | 0.5365 |
| nca_state_matched | 64 | 192 | 73.16 | 14.5311 | 12.82 | 0.11689 | 3/4 | 0.5359 |
| nca_state_matched | 64 | 256 | 73.03 | 20.3612 | 12.75 | 0.10424 | 4/4 | 0.5356 |
| momentum_nca | 32 | 16 | 75.13 | 0.4692 | 19.00 | 0.31229 | 1/4 | 0.6242 |
| momentum_nca | 32 | 32 | 85.07 | 0.2834 | 39.07 | 0.26866 | 3/4 | 0.4531 |
| momentum_nca | 32 | 48 | 86.49 | 0.2732 | 48.23 | 0.22405 | 4/4 | 0.3191 |
| momentum_nca | 32 | 64 | 84.44 | 0.3217 | 42.75 | 0.19202 | 2/4 | 0.2304 |
| momentum_nca | 32 | 96 | 82.42 | 0.4059 | 38.86 | 0.18040 | 1/4 | 0.1523 |
| momentum_nca | 32 | 128 | 81.69 | 0.4661 | 38.12 | 0.17928 | 2/4 | 0.1197 |
| momentum_nca | 32 | 192 | 76.75 | 0.8113 | 31.57 | 0.17510 | 1/4 | 0.0914 |
| momentum_nca | 32 | 256 | 73.38 | 1.2479 | 28.60 | 0.17396 | 0/4 | 0.0778 |
| momentum_nca | 64 | 16 | 57.30 | 0.6302 | 4.36 | 0.33112 | 0/4 | 0.8283 |
| momentum_nca | 64 | 32 | 68.50 | 0.5325 | 11.61 | 0.35778 | 0/4 | 0.6149 |
| momentum_nca | 64 | 48 | 74.27 | 0.6119 | 15.65 | 0.28016 | 4/4 | 0.4159 |
| momentum_nca | 64 | 64 | 75.94 | 0.4808 | 19.04 | 0.22390 | 1/4 | 0.2502 |
| momentum_nca | 64 | 96 | 77.70 | 0.5693 | 21.13 | 0.19986 | 4/4 | 0.1127 |
| momentum_nca | 64 | 128 | 76.88 | 0.6587 | 20.44 | 0.18519 | 2/4 | 0.0689 |
| momentum_nca | 64 | 192 | 75.51 | 0.9284 | 18.12 | 0.18411 | 1/4 | 0.0394 |
| momentum_nca | 64 | 256 | 74.47 | 1.2127 | 17.29 | 0.17657 | 0/4 | 0.0288 |
| masked_state_nca | 32 | 16 | 84.11 | 0.3048 | 22.73 | 0.68407 | 4/4 | 0.7609 |
| masked_state_nca | 32 | 32 | 89.21 | 0.1784 | 37.66 | 0.28222 | 4/4 | 0.8265 |
| masked_state_nca | 32 | 48 | 92.14 | 0.3622 | 63.22 | 0.15961 | 2/4 | 0.8606 |
| masked_state_nca | 32 | 64 | 90.43 | 0.9871 | 56.37 | 0.14666 | 2/4 | 0.8555 |
| masked_state_nca | 32 | 96 | 80.91 | 3.0290 | 18.39 | 0.13295 | 0/4 | 0.8389 |
| masked_state_nca | 32 | 128 | 67.60 | 7.1276 | 7.14 | 0.12590 | 1/4 | 0.8098 |
| masked_state_nca | 32 | 192 | 19.39 | 29.3116 | 13.81 | 0.10975 | 2/4 | 0.6682 |
| masked_state_nca | 32 | 256 | 11.38 | 63.4853 | 9.38 | 0.10359 | 3/4 | 0.6231 |
| masked_state_nca | 64 | 16 | 59.85 | 0.6475 | 5.65 | 0.85735 | 1/4 | 0.8193 |
| masked_state_nca | 64 | 32 | 69.81 | 0.4578 | 11.54 | 0.90419 | 2/4 | 0.7410 |
| masked_state_nca | 64 | 48 | 80.39 | 1.0986 | 19.88 | 0.33390 | 3/4 | 0.6846 |
| masked_state_nca | 64 | 64 | 75.93 | 2.6384 | 16.14 | 0.20206 | 3/4 | 0.7102 |
| masked_state_nca | 64 | 96 | 73.76 | 5.5418 | 10.22 | 0.20179 | 4/4 | 0.7359 |
| masked_state_nca | 64 | 128 | 47.35 | 11.0186 | 0.15 | 0.18012 | 4/4 | 0.7983 |
| masked_state_nca | 64 | 192 | 35.28 | 25.9444 | 0.26 | 0.16655 | 2/4 | 0.8367 |
| masked_state_nca | 64 | 256 | 25.93 | 40.6141 | 4.35 | 0.15398 | 2/4 | 0.7637 |
| masked_momentum_nca | 32 | 16 | 70.43 | 0.4876 | 16.13 | 0.76828 | 2/4 | 0.1141 |
| masked_momentum_nca | 32 | 32 | 91.47 | 0.1598 | 49.82 | 0.70763 | 3/4 | 0.0466 |
| masked_momentum_nca | 32 | 48 | 96.04 | 0.0734 | 69.09 | 0.53006 | 4/4 | 0.0291 |
| masked_momentum_nca | 32 | 64 | 96.45 | 0.0512 | 77.25 | 0.27742 | 3/4 | 0.0179 |
| masked_momentum_nca | 32 | 96 | 92.56 | 0.2402 | 68.23 | 0.16272 | 0/4 | 0.0082 |
| masked_momentum_nca | 32 | 128 | 86.17 | 0.9152 | 53.28 | 0.16949 | 2/4 | 0.0050 |
| masked_momentum_nca | 32 | 192 | 80.24 | 3.5378 | 45.41 | 0.16165 | 1/4 | 0.0028 |
| masked_momentum_nca | 32 | 256 | 77.68 | 6.8694 | 42.78 | 0.17582 | 4/4 | 0.0020 |
| masked_momentum_nca | 64 | 16 | 57.09 | 0.6632 | 3.83 | 0.78248 | 0/4 | 0.3597 |
| masked_momentum_nca | 64 | 32 | 73.42 | 0.3870 | 15.02 | 0.72968 | 0/4 | 0.1049 |
| masked_momentum_nca | 64 | 48 | 78.84 | 0.2984 | 23.13 | 0.65201 | 1/4 | 0.0546 |
| masked_momentum_nca | 64 | 64 | 84.75 | 0.2421 | 29.24 | 0.58748 | 0/4 | 0.0330 |
| masked_momentum_nca | 64 | 96 | 78.61 | 0.5370 | 23.89 | 0.49394 | 1/4 | 0.0145 |
| masked_momentum_nca | 64 | 128 | 68.70 | 1.8109 | 17.03 | 0.48936 | 2/4 | 0.0078 |
| masked_momentum_nca | 64 | 192 | 65.16 | 6.2904 | 14.89 | 0.20387 | 2/4 | 0.0035 |
| masked_momentum_nca | 64 | 256 | 64.15 | 11.4356 | 14.14 | 0.17611 | 4/4 | 0.0022 |

The performance columns use all 16 frozen maps; Jacobian diagnostics use fixed maps 0..3.
Positive finite-window log gain is not asymptotic instability. Compare each arm to its free-momentum baseline.
Unconverged power estimates remain lower estimates; do not claim an upper bound or near-criticality from them.
Open projection is at product input/output only; whole-grid paths may traverse walls between endpoints.
Source tangents include encoder initialization and repeated source drive, separately and jointly.
Binary source flips are finite interventions; their agreement with infinitesimal tangents is not assumed.
No causal velocity isolation, asymptotic convergence, trained dynamics fix, or systems speedup is established.
