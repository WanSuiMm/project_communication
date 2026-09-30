# Reaction-Transport: local 2D / 3D qualification

Status: the first frozen local 2D/3D qualification completed. Gate A is
INCONCLUSIVE_POSITIVE_CONTROL in both dimensions: one attention seed solves
the task and one does not fit; learned transport remains near chance in both
seeds. B/C and the width sweep were not launched. See `RESULTS.md` and
`RUN_MANIFEST.md`. No ongoing training or monitoring is scheduled.
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
```

Output directories must be new. A timed-out or failed qualification is retained.
The first run uses the local GPU only, two training seeds and a 25-minute
training/evaluation scheduling budget. No external datasets, remote jobs,
DEQ, directed transport, adaptive recurrence or generation tasks are involved.

The 3D task uses genuine Conv3d and three-axis transport on narrow volumes.
It does not establish performance on general 3D object geometry.
