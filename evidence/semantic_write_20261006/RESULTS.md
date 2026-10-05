# Semantic-write source audit

Status: COMPLETE

Zero training; protected-source and oracle retention are manipulation checks.

Primary: NO_PRIMARY_SOURCE_DRIVER_SIGNAL
Source-natural mean=0.0006; source-orthogonal mean=0.0004; positive blocks=2/8.

|Arm|Size|Condition|Strict T256|Off-source T256|Source T256|
|---|---:|---|---:|---:|---:|
|native|32|natural|0.3217|0.4346|0.5820|
|native|32|source_parallel_t8|0.3223|0.4391|1.0000|
|native|32|source_parallel_t64|0.3217|0.4362|0.8438|
|native|32|source_orthogonal_t8|0.3219|0.4349|0.5820|
|native|32|oracle_solved_t8|0.6540|0.7863|1.0000|
|native|32|nullspace_zero_t8|0.1777|0.3833|0.6641|
|native|64|natural|0.1782|0.1220|0.4766|
|native|64|source_parallel_t8|0.1797|0.1238|0.9609|
|native|64|source_parallel_t64|0.1782|0.1221|0.5469|
|native|64|source_orthogonal_t8|0.1785|0.1221|0.4570|
|native|64|oracle_solved_t8|0.5212|0.4781|0.9609|
|native|64|nullspace_zero_t8|0.1034|0.0994|0.4609|
|factorized|32|natural|0.1759|0.2765|0.4453|
|factorized|32|source_parallel_t8|0.1760|0.2790|1.0000|
|factorized|32|source_parallel_t64|0.1760|0.2775|0.9023|
|factorized|32|source_orthogonal_t8|0.1759|0.2769|0.4336|
|factorized|32|oracle_solved_t8|0.7199|0.8247|1.0000|
|factorized|32|nullspace_zero_t8|0.1396|0.2635|0.4883|
|factorized|64|natural|0.1477|0.1137|0.4336|
|factorized|64|source_parallel_t8|0.1480|0.1143|0.9922|
|factorized|64|source_parallel_t64|0.1477|0.1138|0.7930|
|factorized|64|source_orthogonal_t8|0.1477|0.1137|0.3633|
|factorized|64|oracle_solved_t8|0.8058|0.7978|0.9922|
|factorized|64|nullspace_zero_t8|0.1321|0.0929|0.3398|
|native_r2|32|natural|0.1894|0.2946|0.4102|
|native_r2|32|source_parallel_t8|0.1897|0.2966|0.9062|
|native_r2|32|source_parallel_t64|0.1895|0.2953|0.7383|
|native_r2|32|source_orthogonal_t8|0.1895|0.2950|0.4062|
|native_r2|32|oracle_solved_t8|0.5300|0.6582|0.9062|
|native_r2|32|nullspace_zero_t8|0.1572|0.3217|0.5000|
|native_r2|64|natural|0.1448|0.1121|0.4531|
|native_r2|64|source_parallel_t8|0.1455|0.1130|0.8516|
|native_r2|64|source_parallel_t64|0.1448|0.1124|0.5898|
|native_r2|64|source_orthogonal_t8|0.1446|0.1121|0.4531|
|native_r2|64|oracle_solved_t8|0.3991|0.3504|0.8516|
|native_r2|64|nullspace_zero_t8|0.1151|0.0917|0.4141|
|factorized_r2|32|natural|0.1978|0.2595|0.3906|
|factorized_r2|32|source_parallel_t8|0.1982|0.2634|1.0000|
|factorized_r2|32|source_parallel_t64|0.1978|0.2605|0.7891|
|factorized_r2|32|source_orthogonal_t8|0.1982|0.2602|0.3633|
|factorized_r2|32|oracle_solved_t8|0.6715|0.7816|1.0000|
|factorized_r2|32|nullspace_zero_t8|0.0575|0.1302|0.2812|
|factorized_r2|64|natural|0.1620|0.1099|0.4023|
|factorized_r2|64|source_parallel_t8|0.1624|0.1103|0.9844|
|factorized_r2|64|source_parallel_t64|0.1620|0.1100|0.6562|
|factorized_r2|64|source_orthogonal_t8|0.1621|0.1099|0.3516|
|factorized_r2|64|oracle_solved_t8|0.5941|0.5496|0.9844|
|factorized_r2|64|nullspace_zero_t8|0.0667|0.0562|0.2969|
|historical_seed4|32|natural|1.0000|0.9565|1.0000|
|historical_seed4|32|source_parallel_t8|1.0000|0.9565|1.0000|
|historical_seed4|32|source_parallel_t64|1.0000|0.9565|1.0000|
|historical_seed4|32|source_orthogonal_t8|1.0000|0.9565|1.0000|
|historical_seed4|32|oracle_solved_t8|1.0000|0.9566|1.0000|
|historical_seed4|32|nullspace_zero_t8|1.0000|0.9448|1.0000|
|historical_seed4|64|natural|0.9544|0.7579|0.9375|
|historical_seed4|64|source_parallel_t8|0.9544|0.7579|0.9688|
|historical_seed4|64|source_parallel_t64|0.9544|0.7579|0.9375|
|historical_seed4|64|source_orthogonal_t8|0.9544|0.7579|0.9375|
|historical_seed4|64|oracle_solved_t8|0.9982|0.8085|0.9688|
|historical_seed4|64|nullspace_zero_t8|0.9544|0.6758|0.9375|


## Interpretation and publication

The frozen primary requires both mean deltas >=0.05 and positive source-natural
in >=6/8 native blocks. Neither threshold is met. Native size32 source T256
coverage rises from0.5820 to1.0000, while unprotected strict16<d<32 T256 coverage
changes only0.3217 to0.3223. Source preservation alone did not rescue distant
computation for this intervention and these fixed checkpoints/cohorts.

Oracle protection uses distant ground truth and can inject labels via its gate;
its improvements cannot establish learned communication or rescue the primary.
Nullspace ablation changes future dynamics. The selected C24 seed4 reference
qualifies on this common cohort; it is not a matched-width replication. The
independent primary unit is the paired training block(n8), not a pixel/map/time.

All396 packed paired Boolean traces, FP32 sampled margins,66 natural signed-write
diagnostics, source trajectories, intervention arrays and all per-map endpoint
rows are retained. NPZ files are already losslessly compressed and byte-exact;
per-map CSV files use lossless gzip. No precision reduction. Protected retention
is a manipulation check. Checkpoints and machine/launch records remain local.

The1318.938-second timing covers the resumed worker only:75 completed units were
imported with validated hashes,321 were newly computed. It is not total elapsed
time for all396 measurements. Saved-data publication performs no model inference,
training or optimizer update. See validation.json for its exact scope.
