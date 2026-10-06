# HardClip-v1: outcome-blind tail calibration

Status: COMPLETE8/8. Verdict: CALIBRATED.

Fixed target:1% removed raw-write magnitude per lane; maximum20% trigger rate.

|Lane|Frozen actual-write cap|Removed fraction|Trigger fraction|Sparse enough|
|---|---:|---:|---:|---|
|N|0.396451592|0.0100000016|0.10063387|True|
|E|0.482601881|0.0100000018|0.0985681285|True|
|S|0.474225521|0.0100000033|0.0990670246|True|
|W|0.515789509|0.00999999896|0.0982087177|True|

All8 old neural u300 checkpoints, SAME16 fixed training maps, cold64 steps.
No labels, readout, optimizer, losses or task outcomes were used. Zero training updates.

Calibration PASS permits the frozen two-arm experiment; it is not efficacy evidence.

Full indexed FP32 samples and checkpoint/source hashes are retained locally.
Elapsed 12.31 seconds.
