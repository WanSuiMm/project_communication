# Masked inertial trajectory audit

Inference only; exact frozen evaluation maps, independent32-map calibration per size. No model changes.
Reconstruction max absolute error: 0 (tolerance5e-5).

| Size | T | BA % | Raw BCE | Calibrated BCE | Temperature | H open RMS | H wall RMS | H constant-mode energy | V mean RMS | Wrong margin median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 83.73 | 0.6024 | 0.2831 | 0.046 | 0.409 | 1.063 | 0.7638 | 0.0477 | -0.0186 |
| 32 | 32 | 89.25 | 0.3161 | 0.1922 | 0.250 | 2.634 | 6.554 | 0.8193 | 0.2309 | -0.1273 |
| 32 | 64 | 88.79 | 0.2503 | 0.2569 | 1.695 | 14.476 | 22.711 | 0.8910 | 0.4326 | -1.8887 |
| 32 | 128 | 89.96 | 0.6565 | 0.2659 | 5.936 | 46.534 | 60.625 | 0.9322 | 0.5429 | -6.2575 |
| 32 | 256 | 91.55 | 1.4519 | 0.2492 | 15.695 | 120.607 | 141.926 | 0.9704 | 0.6005 | -18.4205 |
| 64 | 16 | 62.06 | 0.6751 | 0.5653 | 0.042 | 0.281 | 1.063 | 0.8968 | 0.0247 | -0.0183 |
| 64 | 32 | 72.48 | 0.5369 | 0.4251 | 0.199 | 1.027 | 6.554 | 0.5559 | 0.0966 | -0.1217 |
| 64 | 64 | 75.39 | 0.6276 | 0.5178 | 1.823 | 10.649 | 22.711 | 0.7931 | 0.3933 | -2.3346 |
| 64 | 128 | 75.71 | 2.2848 | 0.5345 | 7.242 | 42.881 | 60.625 | 0.8053 | 0.5048 | -9.9588 |
| 64 | 256 | 75.69 | 6.3402 | 0.5335 | 20.041 | 117.668 | 141.926 | 0.8464 | 0.5567 | -28.9915 |
| 128 | 16 | 53.30 | 0.6884 | 0.6662 | 0.071 | 0.266 | 1.063 | 0.9695 | 0.0234 | -0.0186 |
| 128 | 32 | 57.62 | 0.6463 | 0.6189 | 0.317 | 0.606 | 6.554 | 0.5383 | 0.0598 | -0.1281 |
| 128 | 64 | 57.47 | 1.0110 | 0.6725 | 5.398 | 9.313 | 22.711 | 0.8217 | 0.3843 | -2.3417 |
| 128 | 128 | 54.12 | 3.8007 | 0.6806 | 27.140 | 38.718 | 60.625 | 0.9256 | 0.5174 | -9.9966 |
| 128 | 256 | 50.00 | 12.9662 | 0.6903 | 169.807 | 114.513 | 141.926 | 0.9951 | 0.6366 | -29.3417 |

Temperature is fit on independent calibration maps; it cannot change BA or repair source revision.
Component cancellation is an identity; nonzero reaction/velocity and projection energies are observations.
The mean dynamics depends on the full field, so this is not a closed autonomous mean model.
Finite-horizon direction similarity is not proof of convergence. Detailed vectors are in components.json.
No architecture changes, extra training seeds, new benchmark or speed comparison are performed.
