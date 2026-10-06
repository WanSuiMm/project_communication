# Incremental review: fixed HardClip-v1 calibration and completed result

- Review base: `64aa8680d23446bd6375259213767e3e6c997274`.
- Result evidence head: `0ce156ff8c5e6ddc307c3a8e665e26d208532f9e`.
- Execution: COMPLETE, 16/16 trajectories, 208/208 predeclared checkpoints,
  3658.782 seconds (about61 minutes), no runtime limit.
- Formal verdict: `NO_HARDCLIP_RELIABILITY_QUALIFICATION`.
- This handoff changes review metadata only; the result evidence head stays stable.

## Read first

1. [Results](evidence/hardclip_v1_20261006/RESULTS.md),
   [aggregate](evidence/hardclip_v1_20261006/summary.json),
   [16-row final metrics](evidence/hardclip_v1_20261006/final_metrics.csv).
2. [Training dose summary](evidence/hardclip_v1_20261006/dose_summary.json),
   [lane/block dose table](evidence/hardclip_v1_20261006/training_dose.csv),
   [calibration result](evidence/hardclip_v1_20261006/calibration/RESULTS.md).
3. [Validation](evidence/hardclip_v1_20261006/validation.json),
   [reproduction](evidence/hardclip_v1_20261006/REPRODUCTION.md),
   [publication manifest](HARDCLIP_V1_PUBLICATION_MANIFEST.json).
4. [Frozen protocol](new/hardclip_v1/PROTOCOL.md),
   [supplied primitive](new/hardclip_v1/primitive.py),
   [cell](new/hardclip_v1/cells.py), [dose observer](new/hardclip_v1/dose.py),
   [runner](new/hardclip_v1/run.py), [reporter](new/hardclip_v1/reporting.py).

All416 packed traces,208 summaries and evaluation dose records,16 training
curves, banks, schedules, frontier CSVs and full calibration RMS samples are
secondary evidence. JSON/CSV compression is lossless. Model/optimizer contents
and private machine records stay local; hashes retain their provenance.

## Changed evidence

Prior Hybrid v0 changed small-signal gain, radial amplitude encoding and
write bound together. This experiment clips only above-cap writes, with a raw
inactive branch, true autodiff and no learned controller. Other scientific
modules and fixed K8/reset64x4/u300 recipe remain unchanged. The8 new paired
seeds were not selected by task performance.

Outcome-blind calibration used ALL8 prior neural u300 models on the SAME16
training maps, cold64 steps. No labels/readout/task outcomes selected caps.
All lanes have1% cumulative RMS removal and9.82%–10.06% triggers, passing
the<=20% gate. Fixed actual-write caps N/E/S/W are0.3964515924,0.4826018810,
0.4742255211,0.5157895088.

At u300 fresh Neural and HardClip are both joint-ready0/8, with
candidate-only/reference-only0/0, net gain0 and exact paired p=1.
Old Full and checkpoint-grid ever-ready are also0/8 in each arm. Readiness-loss
ratios remain null, since no saved point first becomes ready.

The key limitation is training dose: HardClip blocks0,1,4,5,6 never
trigger clipping. Only block2(E),3(N),7(N) are active. Their whole-training
lane trigger fractions are about5.41%,27.32%,15.14%; RMS removal fractions
are about0.30%,4.73%,1.59%. A sparse calibration tail on old checkpoints
does not guarantee a sparse or nonzero intervention on fresh trajectories.
No threshold was retuned to rescue the result.

## Claim boundary and reviewer focus

This recipe did not qualify a reliability gain. The5 zero-dose blocks limit
a mechanistic rejection of unusually large writes. Do not promote the negative
to a general failure of amplitude control, Hybrid/NCA or continuation theory.
Neither a state-construction mechanism nor another architecture is proved.
Earlier audits and claim boundaries stay unchanged.

Scientific source matches the run snapshot. The public launcher only replaces
a private executable path with Python environment discovery; original tested
and sanitized source hashes are distinguished in the publication. The saved-data
check needs no training or inference:

    python -X utf8 -B tools/export_hardclip_v1.py --verify-only

For review, separate formal readiness from continuous reach/retention metrics,
and calibration dose from actual fresh-training/evaluation dose. Any next
hypothesis should address a measured failure rather than reinterpret this
negative as a positive architectural result.
