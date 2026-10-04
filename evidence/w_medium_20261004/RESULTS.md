# W medium qualification

Status: COMPLETE; 36/36 measured cells; zero training.
Stage1: QUALIFIED_W_REPLICATION_CONTROLS_ENABLED. Native Full: {'S1': True, 'F1': False}.

| Size | Arm | Keep continuous | Prog sustained | Prog delayed | Proxy | Output |
|---:|---|---:|---:|---:|---|---|
| 32 | FF | 0.0005 | 0.0000 | 0.0000 | FAIL | exact FF |
| 32 | SF | 1.0000 | 0.5726 | 0.5726 | PASS | exact FF |
| 32 | FS | 0.0368 | 0.0000 | 0.0000 | FAIL | visible Z |
| 32 | SS | 0.9941 | 0.7576 | 0.7576 | PASS | visible Z |
| 32 | SF_norm_F | 1.0000 | 0.6566 | 0.6566 | PASS | exact FF |
| 32 | FF_norm_S | 0.0244 | 0.0000 | 0.0000 | FAIL | exact FF |
| 32 | SF_payload | 0.8633 | 0.0053 | 0.0053 | FAIL | exact FF |
| 32 | SF_payload_gauge | 1.0000 | 0.5726 | 0.5726 | PASS | exact FF |
| 32 | SF_lane | 0.9999 | 0.5183 | 0.5183 | PASS | exact FF |
| 32 | SF_lane_gauge | 1.0000 | 0.5726 | 0.5726 | PASS | exact FF |
| 32 | SF_spatial | 1.0000 | 0.9966 | 0.9966 | PASS | exact FF |
| 32 | SF_spatial_gauge | 1.0000 | 0.5726 | 0.5726 | PASS | exact FF |
| 32 | SF_cue_swap | 0.0000 | 0.0000 | 0.0000 | FAIL | exact FF |
| 32 | SF_time32 | 0.0020 | 0.3017 | 0.3017 | FAIL | exact FF |
| 32 | SF_time48 | 1.0000 | 0.6513 | 0.6513 | PASS | exact FF |
| 32 | SF_time80 | 1.0000 | 0.5329 | 0.5329 | PASS | exact FF |
| 32 | SF_time96 | 1.0000 | 0.5192 | 0.5192 | PASS | exact FF |
| 32 | SF_norm_F_cue_swap | 0.0000 | 0.0000 | 0.0000 | FAIL | exact FF |
| 64 | FF | 0.0000 | 0.0000 | 0.0000 | FAIL | exact FF |
| 64 | SF | 0.9899 | 0.5925 | 0.5925 | PASS | exact FF |
| 64 | FS | 0.0000 | 0.0000 | 0.0000 | FAIL | visible Z |
| 64 | SS | 0.9813 | 0.7794 | 0.7794 | PASS | visible Z |
| 64 | SF_norm_F | 0.9899 | 0.6494 | 0.6494 | PASS | exact FF |
| 64 | FF_norm_S | 0.0000 | 0.0000 | 0.0000 | FAIL | exact FF |
| 64 | SF_payload | 0.6560 | 0.0134 | 0.0134 | FAIL | exact FF |
| 64 | SF_payload_gauge | 0.9899 | 0.5925 | 0.5925 | PASS | exact FF |
| 64 | SF_lane | 0.9889 | 0.5678 | 0.5678 | PASS | exact FF |
| 64 | SF_lane_gauge | 0.9899 | 0.5925 | 0.5925 | PASS | exact FF |
| 64 | SF_spatial | 0.9029 | 0.9147 | 0.9147 | FAIL | exact FF |
| 64 | SF_spatial_gauge | 0.9899 | 0.5925 | 0.5925 | PASS | exact FF |
| 64 | SF_cue_swap | 0.0000 | 0.0000 | 0.0000 | FAIL | exact FF |
| 64 | SF_time32 | 0.1151 | 0.5573 | 0.5573 | FAIL | exact FF |
| 64 | SF_time48 | 0.9702 | 0.6010 | 0.6010 | PASS | exact FF |
| 64 | SF_time80 | 0.9899 | 0.5885 | 0.5885 | PASS | exact FF |
| 64 | SF_time96 | 0.9899 | 0.5911 | 0.5911 | PASS | exact FF |
| 64 | SF_norm_F_cue_swap | 0.0000 | 0.0000 | 0.0000 | FAIL | exact FF |

Read summary.json, metrics.csv and stage1_gate.json first. Native Full,
map support and conditional rescue are distinct. Gauge arms change consumer
coordinates as well as W; raw-arm losses are compatibility sensitivities.
No mechanism class or learning algorithm is selected automatically.

## Publication notes

Fresh128 maps at each size. Fixed keep/prog support at size32 is118/33 maps
(20,620/3,573 cells), and at size64 is39/123 maps (2,884/103,230 cells).
Both SF and SS pass the unchanged rescue/support thresholds; all six gauge
suffixes exactly reproduce SF. Native Full is a separate recorded qualification.

Read the compact summary, metrics, configuration and validation first. The
37MB raw_summary and per-arm JSON are secondary and preserve original counts,
per-map differences and every gate. Negative results are retained. Spatial
scrambling has high progress but FAILS size64 preservation (.9029<.95).
No exclusive W/Z role, pure phase, task semantics or learning guarantee follows.

Scientific states are losslessly split into48 map/time chunks to keep each file
below50MiB; state_chunks.json records exact reconstruction and tensor hashes.
No activation values, frozen run files, gates or model parameters were changed.
Checkpoints, machine manifests, PIDs, launch receipts and logs remain local.
