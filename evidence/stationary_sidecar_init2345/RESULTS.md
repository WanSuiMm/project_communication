# Stationary sidecar: completed development screen

**DEVELOPMENT_NO_GO.** All12 arms completed300 updates in 1676.281 seconds (27.94 minutes).
Four original Streaming controls reproduce historical final parameter hashes and full evaluations exactly.
Reach+hold: stream1/4; memory0/4; stateless0/4. All three arms reach at seed4; only stream holds.

## Frozen primary: size32, T64, strict16<d<32

| Seed | Variant | BA original / flipped % | Per-map mean % | Pooled correct / pixels (%) | Reach | Hold | Reach+hold |
|---:|---|---:|---:|---:|---|---|---|
| 2 | stream | 57.44 / 55.51 | 1.08 | 43/2918 (1.47%) | False | False | False |
| 2 | memory | 86.25 / 87.46 | 0.61 | 20/2918 (0.69%) | False | True | False |
| 2 | stateless | 95.72 / 96.45 | 73.38 | 1957/2918 (67.07%) | False | True | False |
| 3 | stream | 89.03 / 90.74 | 8.28 | 164/2918 (5.62%) | False | False | False |
| 3 | memory | 88.53 / 84.53 | 16.06 | 410/2918 (14.05%) | False | False | False |
| 3 | stateless | 86.14 / 82.09 | 5.74 | 140/2918 (4.80%) | False | True | False |
| 4 | stream | 96.98 / 96.86 | 88.89 | 2507/2918 (85.92%) | True | True | True |
| 4 | memory | 97.80 / 97.37 | 95.51 | 2750/2918 (94.24%) | True | False | False |
| 4 | stateless | 99.27 / 99.23 | 100.00 | 2918/2918 (100.00%) | True | False | False |
| 5 | stream | 87.44 / 88.19 | 16.05 | 380/2918 (13.02%) | False | True | False |
| 5 | memory | 85.70 / 84.37 | 6.54 | 101/2918 (3.46%) | False | False | False |
| 5 | stateless | 84.80 / 83.46 | 17.18 | 449/2918 (15.39%) | False | False | False |

Reach requires BA>=85% in each source orientation and BOTH paired metrics>=80%.
Hold bounds declines at BOTH T128/T256: BA<=3pp and paired metrics<=5pp relative to T64.
Memory GO also requires>=3/4 reach+hold including seed4 and strictly more successes than both controls.
Hold=True cannot qualify a failed reach endpoint.

## Seed4: early gains do not preserve the longer computation

| Variant | T | BA original / flipped % | Per-map mean % | Pooled correct / pixels (%) |
|---|---:|---:|---:|---:|
| stream | 64 | 96.98 / 96.86 | 88.89 | 2507/2918 (85.92%) |
| stream | 128 | 99.08 / 99.02 | 99.64 | 2899/2918 (99.35%) |
| stream | 256 | 99.44 / 99.56 | 100.00 | 2918/2918 (100.00%) |
| memory | 64 | 97.80 / 97.37 | 95.51 | 2750/2918 (94.24%) |
| memory | 128 | 50.00 / 51.56 | 0.00 | 0/2918 (0.00%) |
| memory | 256 | 50.02 / 51.67 | 0.31 | 11/2918 (0.38%) |
| stateless | 64 | 99.27 / 99.23 | 100.00 | 2918/2918 (100.00%) |
| stateless | 128 | 99.13 / 99.18 | 100.00 | 2918/2918 (100.00%) |
| stateless | 256 | 78.06 / 76.13 | 85.96 | 2582/2918 (88.49%) |

Memory improves the T64 primary but loses it at T128 (0/2918), while stateless remains perfect at T128 then fails hold at T256.
Thus a higher finite-horizon endpoint does not establish reliable beneficial continued rollout.
Stateless seed2 improves through T128/T256, but its T64 endpoint remains below the frozen threshold.
These are descriptive observations on selected previously inspected seeds; later endpoints cannot replace the frozen gate.

## All primary trajectories

| Seed | Variant | Pooled T64 % | Pooled T128 % | Pooled T256 % |
|---:|---|---:|---:|---:|
| 2 | stream | 1.47 | 0.10 | 0.00 |
| 2 | memory | 0.69 | 0.31 | 0.03 |
| 2 | stateless | 67.07 | 80.36 | 86.46 |
| 3 | stream | 5.62 | 1.88 | 0.00 |
| 3 | memory | 14.05 | 46.85 | 53.74 |
| 3 | stateless | 4.80 | 10.52 | 10.66 |
| 4 | stream | 85.92 | 99.35 | 100.00 |
| 4 | memory | 94.24 | 0.00 | 0.38 |
| 4 | stateless | 100.00 | 100.00 | 88.49 |
| 5 | stream | 13.02 | 13.98 | 17.10 |
| 5 | memory | 3.46 | 6.79 | 18.81 |
| 5 | stateless | 15.39 | 15.87 | 18.13 |

## Far-space summaries

Size64 d>32 values pool integer counts over the four seeds for description.
The same32 maps per size are reused;128 map-seed evaluations are not128 independent task maps.
| T | Variant | Correct / pixels | Pooled % |
|---:|---|---:|---:|
| 64 | stream | 3701/109744 | 3.37 |
| 64 | memory | 2814/109744 | 2.56 |
| 64 | stateless | 10690/109744 | 9.74 |
| 128 | stream | 11201/109744 | 10.21 |
| 128 | memory | 1465/109744 | 1.33 |
| 128 | stateless | 14064/109744 | 12.82 |
| 256 | stream | 17001/109744 | 15.49 |
| 256 | memory | 2624/109744 | 2.39 |
| 256 | stateless | 11371/109744 | 10.36 |

## Scope and interpretation

Original W24/Z8, F/Q modules, masked streaming/Laplacians, perception clocks and training are preserved.
G67->32->12 writes an additional H12. Bias-free P_F/P_Q feed H* into F/Q.
Memory H*=H+.1G and stateless H*=.1G are the only structural difference between the7989-parameter side arms; stream has5033.
No direct H/LH input columns are disabled in stateless. Both side arms consume current H* immediately.
Neutral P_F=P_Q=0 nests the original W/Z/logits and raw core derivatives. Extra gradients can change global norm clipping and optimizer updates.
All new gradient paths opened by update3. Two-hop macro/K8 radius16 bounds are retained.
This recipe did not improve reproducible reach+hold. It does not reject stationary memory in general: original Z8 is already stationary.
Carry changes temporal accumulation and activation scale together. No identified Jacobian, saturation, overwrite, optimization, or semantic mechanism follows from these trajectories.
Stateless has no extra H carry but recurrent W/Z still retain information.
Four reused initialization seeds are the independent units, conditional on one shared training bank/schedule; no population reliability, p-value, fresh confirmation, or3D claim.

## Systems and saved-result verification

| Variant | Median training seconds | Median peak allocated MiB | Median clipped % |
|---|---:|---:|---:|
| stream | 120.27 | 97.08 | 6.83 |
| memory | 140.79 | 125.25 | 18.17 |
| stateless | 137.53 | 125.37 | 3.33 |

Verified45 source bindings in current checkout and both snapshots,12 CPU checkpoint state hashes,4 exact historical controls, banks/schedules/preflight,144 BA means,1008 paired metrics with independently regenerated denominators,936 curve rows,624 contrasts and all frozen gate decisions.
These arithmetic and CPU checkpoint checks do not rerun inference or training. Timing/allocation/clipping/state RMS are descriptive; no matched-FLOP or stability claim.
Read [analysis](analysis.json), [validation](validation.json), [curves](curves.csv) and [paired effects](paired_effects.csv) before the12 [raw records](raw/).
Checkpoints, machine metadata and launch receipts remain local.
