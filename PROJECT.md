# Reaction-Transport: local 2D / 3D qualification

Current work: A0 protocol `rt_a0_2d3d_v2_1` is implemented in `a0_models.py`
and `a0_runner.py`; see `A0_PROTOCOL.md`. It adds same-emission attention,
oracle transport diagnostics and a paired 2x2 normalization/medium comparison.
The new schedule has four seeds, 480 updates per trial, a 25-minute cap and
per-dimension positive-control stopping. It does not schedule B/C or monitoring.
The original v1 training sources and evidence remain unchanged.
Completed on 2026-09-30: 24/24 trials in the conditional A0 schedule. In 2D,
all four attention seeds pass and every RT arm is NOT_QUALIFIED_FIT. In 3D,
one attention seed fails, so its sixteen RT trials are skipped. The initial
40-trial maximum was conditional, not a remaining-work count.
Public evidence is in `evidence/a0_v2_1/`; see `RESULTS.md` first and
`A0_PUBLICATION_MANIFEST.json` for hashes. Numerical checks passed. No B/C,
further training or recurring monitor is scheduled by this publication.

Historical status: the first frozen local 2D/3D qualification completed. Gate A is
INCONCLUSIVE_POSITIVE_CONTROL in both dimensions: one attention seed solves
the task and one does not fit; learned transport remains near chance in both
seeds. B/C and the width sweep were not launched. See `RESULTS.md` and
`RUN_MANIFEST.md`. That historical run has no ongoing training or monitoring.
This is an architecture screen, not a natural-vision or foundation-model claim.

The question is whether persistent local state plus narrow, symmetric implicit
message transport can learn long-distance dependence, select relevant sources,
and preserve local detail at a useful measured cost.

Read `RESULTS.md` first, `PROTOCOL.md` for frozen gates and scope, and `THEORY.md` for the four short
checks. Source map: `data.py` generates tasks; `transport.py` implements PCR
and its implicit adjoint; `models.py` defines controls; `runner.py` trains,
evaluates and writes per-run `RESULTS.md` plus `aggregate.json`.

Run commands from this standalone repository root (Python with PyTorch and CUDA):

```powershell
python run.py check
python run.py smoke --out runs/NEW_SMOKE_DIRECTORY
python run.py qualify --out runs/NEW_RUN_DIRECTORY --minutes 25
python a0.py check --out runs/NEW_A0_CHECK_DIRECTORY
python a0.py qualify --out runs/NEW_A0_RUN_DIRECTORY --checks runs/NEW_A0_CHECK_DIRECTORY/checks.json
```

Output directories must be new. A timed-out or failed qualification is retained.
The original v1 uses two seeds; A0 uses four. Both use the local GPU and a
25-minute scheduling budget. No external datasets, remote jobs,
DEQ, directed transport, adaptive recurrence or generation tasks are involved.

The 3D task uses genuine Conv3d and three-axis transport on narrow volumes.
It does not establish performance on general 3D object geometry.
