# Fixed-cap training entry points

The frozen [protocol](PROTOCOL.md) remains unchanged after calibration. Stage1
completed in `runs/hardclip_v1_calibration_20261006_01`: all four lanes passed,
with1% removed magnitude and9.82%–10.06% event triggers. This permits Stage2;
it supplies no training efficacy result. The fixed FP32 actual-write caps in
N/E/S/W order are0.3964515924,0.4826018810,0.4742255211,0.5157895088.

From the standalone repository root:

    python -X utf8 -B new/hardclip_v1/check_cells.py
    python -X utf8 -B new/hardclip_v1/reporting.py
    python -X utf8 -u -B new/hardclip_v1/run.py --check --out analyses/hardclip_v1_qualification_20261006_01.json
    pwsh -File tools/launch_hardclip_v1.ps1 -RunName hardclip_v1_20261006_01 -Qualification analyses/hardclip_v1_qualification_20261006_01.json

All output paths are new and fail rather than overwrite evidence. The launcher
requires successful outcome-blind calibration plus actual-shape eager/CUDA Graph
qualification, and runs in a persistent foreground tool session. It sets no
runtime limit, watchdog or recurring monitoring. Its private launch receipt,
source hashes, calibration/sample hashes, batch plans, configurations and
checkpoint hashes bind the dispatch. Machine receipts stay local.

`cells.HardClipCell` retains the canonical E/F/Q/readout and masked transport.
`primitive.FixedLaneHardClip` changes only above-cap lane corrections, with
genuine autodiff and fixed buffers. `dose.TrainingDose` records all open-cell
events in each cold64x4 update; `dose.EvaluationDose` separates steps1..64 from
65..256 in each size/world. Neural dose is a counterfactual under the same caps,
while its actual write remains unchanged. Dose collection has no gradient and
uses no task labels. All qualification statistics are separate from endpoints.

The8 fresh paired blocks produce16 trajectories and208 checkpoint evaluations
at u0,25,...,300. The only primary comparison is fixed-u300 joint readiness;
the two-sided paired test and net gain>=6/8 are frozen. Checkpoint-grid readiness
and historical Full are secondary. `RESULTS.md`/`summary.json` are final only
when `status.json` says COMPLETE; partial runs remain INCOMPLETE. A negative
result with meaningful intervention closes this amplitude repair recipe without
establishing another architecture mechanism.
