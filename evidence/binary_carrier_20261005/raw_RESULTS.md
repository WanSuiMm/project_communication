# Binary Carrier Causal Compression

Status COMPLETE; 24/24 cells; elapsed 473.469s.
Recurrent updates: 0. Linear decoder fits: 1. Templates: fixed calibration centroids.
Anchor: QUALIFIED_W_REPLICATION_CONTROLS_ENABLED. Combined gates: {'bit_once': {'pass': False, 'status': 'FAIL'}, 'bit_repeat8': {'pass': False, 'status': 'FAIL'}, 'source_once': {'pass': False, 'status': 'FAIL'}}.

| Size | Arm | Keep continuous | Prog sustained | Prog delayed | 90% gate |
|---:|---|---:|---:|---:|---|
| 32 | FF | 0.0006 | 0.0000 | 0.0000 | control |
| 32 | SF | 1.0000 | 0.6793 | 0.6793 | control |
| 32 | SS | 0.9956 | 0.9048 | 0.9048 | control |
| 32 | bit_once | 0.2159 | 0.4327 | 0.4327 | FAIL |
| 32 | bit_repeat8 | 0.0064 | 0.0368 | 0.0368 | FAIL |
| 32 | swap_once | 0.0000 | 0.0038 | 0.0038 | control |
| 32 | swap_repeat8 | 0.0002 | 0.0026 | 0.0026 | control |
| 32 | midpoint_once | 0.0000 | 0.0000 | 0.0000 | control |
| 32 | midpoint_repeat8 | 0.0000 | 0.0000 | 0.0000 | control |
| 32 | oracle_once | 0.1193 | 0.9967 | 0.9967 | control |
| 32 | source_once | 0.0000 | 0.0000 | 0.0000 | FAIL |
| 32 | source_swap_once | 0.0000 | 0.0000 | 0.0000 | control |
| 64 | FF | 0.0000 | 0.0000 | 0.0000 | control |
| 64 | SF | 0.9890 | 0.6404 | 0.6404 | control |
| 64 | SS | 0.9832 | 0.7984 | 0.7984 | control |
| 64 | bit_once | 0.1527 | 0.4614 | 0.4614 | FAIL |
| 64 | bit_repeat8 | 0.0372 | 0.1280 | 0.1280 | FAIL |
| 64 | swap_once | 0.0000 | 0.2366 | 0.2366 | control |
| 64 | swap_repeat8 | 0.0000 | 0.0036 | 0.0036 | control |
| 64 | midpoint_once | 0.0000 | 0.0000 | 0.0000 | control |
| 64 | midpoint_repeat8 | 0.0000 | 0.0000 | 0.0000 | control |
| 64 | oracle_once | 0.0896 | 0.6390 | 0.6390 | control |
| 64 | source_once | 0.0000 | 0.0000 | 0.0000 | FAIL |
| 64 | source_swap_once | 0.0000 | 0.0000 | 0.0000 | control |

Oracle-only outcomes are diagnostic and excluded from primary claims.
Read decoder/probes.json and decoder/fit.json, then summary.json; arrays and per-map records are secondary.
Repeated projections retain Z and apply at T64,72,...,256. Source-only retains Z_F64 and recipient X.
Selected fixed consumer, finite restart/projected continuation; no universal bit insufficiency, quotient, architecture or BPTT guarantee.
