# Reaction-Transport: local 2D / 3D qualification

Current follow-up (2026-10-01): the user authorized the bounded masked-medium
diagnostic in `new/masked_medium/PROTOCOL.md`. Only masked RD and masked inertial
RD are added, paired with the frozen unmasked seed-0 controls. The intervention
is input-mask edge weights m_i*m_j; initialization, data, 800 updates and all
evaluation settings are retained. Original sources and evidence stay unchanged.
The main outcome is the paired change at 32x32/T64; this does not test superiority
over a generic momentum model with the same masked graph.

Entry commands from this repository root (new output directories required):

```powershell
python new/masked_medium/test_masked.py
python new/masked_medium/run_masked.py --preflight --out runs/NEW_MASKED_PREFLIGHT
python new/masked_medium/run_masked.py --out runs/NEW_MASKED_SCREEN
```

Four operator/initialization/isolation checks passed, followed by a three-update
CUDA preflight for each masked arm. The formal run was dispatched on 2026-10-01
at 12:09 +08:00 on the local RTX4060 Laptop GPU; the child manifest was verified.
The formal run finished at 12:12:23 +08:00; both arms completed 800 updates.
At the prespecified 32x32/T64 endpoint, masked inertial RD improved balanced
accuracy from 69.94% to 88.79% and paired-source correctness from 17.81% to
37.22%, passing the descriptive joint +5 percentage-point criterion. Masked RD
declined from 81.27% to 73.21% and from 37.81% to 17.13%, respectively.
At T256, masked inertial RD reached 91.55% BA at 32x32 and 75.69% at 64x64,
but returned to 50.00% at 128x128. No sustained aggregate 95% endpoint was
reached. This single-seed diagnostic supports an operator-dependent improvement
within the inertial recipe; scale generalization and superiority over a masked
generic-momentum control remain unestablished.
Run directory: `runs/masked_medium_20261001_seed0/`. Read its status.json
for execution state and RESULTS.md / paired_comparison.json for completed arms.
No automatic additional seed, architecture change or continuous monitor is scheduled.

Previous screen (2026-10-01): the user-supplied inertial NCA wind tunnel is integrated
under `new/nca_inertial_wind_tunnel/`. Read its `WIND_TUNNEL.md` and
`LOCAL_INTEGRATION.md`. The latter records the original archive hash, limited
metric/output fixes and the frozen local 2D screen. Eight unit tests, the linear
checks and four full-size CUDA preflight arms passed. Seed 0 learning completed
on 2026-10-01 at 00:46:51 +08:00 on the local RTX 4060 Laptop GPU: all four arms
finished 800 updates and all three evaluation sizes without a numerical-failure
or budget-limit status. The
local run directory is `runs/inertial_20261001_seed0/`. Public results are in
`evidence/inertial_seed0/`; read its RESULTS.md and summary.json first.
INERTIAL_PUBLICATION_MANIFEST.json binds source, protocol and evidence hashes.
Private local status files and receipts are not required for GitHub review.

This is a negative exploratory result for the current inertial_rd recipe, not a
rejection of all PDE/NCA architectures. At 32x32 and T=64, balanced accuracy was
85.17% (state-matched NCA), 84.44% (momentum NCA), 81.27% (RD) and 69.94%
(inertial RD). Inertial RD declined to 53.50% at T=256, and to 50.00% at both
64x64 and 128x128 at T=256, where paired-source correctness was zero. No arm
reached sustained aggregate 95% BA. Inertial RD had zero pre-damage eligible
examples at every size, so its conditional repair outcome is unevaluable.
There is no benefit over generic momentum in this single-seed screen. Each size
has 16 held-out maps; no multi-seed superiority or universal impossibility claim
is established. The separately authorized masked follow-up above tests one
operator intervention; it does not revise this completed negative result.

Run from this repository root with CUDA (new output directory required):

```powershell
python new/nca_inertial_wind_tunnel/run_wind_tunnel.py --device cuda --arms all --seed 0 --steps 800 --minutes 25 --out runs/NEW_INERTIAL_SCREEN
```

This tests explicit transport plus inertia against generic momentum and state
capacity controls. It does not test redesigned graph substrates or 3D, and does
not revise the older A0 evidence. No automatic extra seeds or monitor is scheduled.

Previous work: A0 protocol `rt_a0_2d3d_v2_1` is implemented in `a0_models.py`
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
