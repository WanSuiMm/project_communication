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
