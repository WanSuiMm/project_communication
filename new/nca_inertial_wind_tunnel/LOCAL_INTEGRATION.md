# Local integration, 2026-10-01

Source: the user-supplied `NCA_Inertial_Theory_WindTunnel.zip`, SHA256
`a125cb898bd680d3af1e73912b1ca7c1f3c1beba4713d6bb92171ac2277160e0`.
Its THEORY.md, WIND_TUNNEL.md and RESULTS.md exactly match the three supplied
Markdown documents supplied alongside the archive. The ZIP, original manifest,
duplicate Markdown and external CPU smoke outputs remain local and are excluded
from GitHub. They are not needed for reproduction. Published execution evidence
is in [the inertial results](../../evidence/inertial_seed0/RESULTS.md) and its
[publication manifest](../../INERTIAL_PUBLICATION_MANIFEST.json).

The four cells, their initialization, parameter counts, optimizer, training data
generator and training schedule are unchanged. This is the 2D seed-0 screen in
WIND_TUNNEL.md, not a continuation or reinterpretation of A0, and not a 3D test.

## Corrections before the first local learning run

- After a source flip, all regions can have the same label. Balanced accuracy
  and balanced BCE now average only over classes present in each image. This
  avoids scoring a perfect one-class image at 50%. Original training maps always
  contain both labels, so their training objective is unchanged.
- Repair controls now also report clean continuation and cold restart on the
  exact same eligible examples as damaged continuation.
- The harness explicitly reports the first tested horizon that reaches 95% BA
  and stays there at every later tested checkpoint. No hit is null. It adds
  paired-source distance bins and full encoder/rollout/readout latency for each
  evaluated size and horizon, so quality and latency can be compared together.
- Output directories must be new and explicit. Smoke no longer redirects to the
  imported evidence directory. Checkpoints are saved before evaluation; partial
  results, source snapshots, source hashes, progress and error statuses survive
  interruption. A whole-run scheduling cap stops new work at the next boundary.
- The full-size preflight uses three updates per arm, batch 8, 32x32 and T=64,
  with abbreviated evaluation. It is labeled SMOKE_NOT_EFFICACY_EVIDENCE.

Original generator details retained for reproducibility: it forces each original
map to contain at least one positive and one negative region. Consequently labels
are random and position-independent but not mutually independent across regions.
Counterfactual evaluation flips the largest component, keeping geometry unchanged.
Walls and doors are one pixel wide as rooms grow with canvas size. The sampler
therefore tests increasing room/path scale, not continuous geometric rescaling.

## Frozen local screen

Four arms, seed 0, 800 AdamW updates each, 512 training maps, batch 8, lr 1e-3,
weight decay 1e-4, clip norm 1. Training rollouts are 32/48/64 with the original
paired RNG schedule and damage recipe. Evaluation uses 16 fixed held-out maps
per size (32/64/128), horizons 16/32/64/128/256, repair/revision at 8/16/32/64
extra steps, and the original gradient probes. FP32 eager CUDA, two CPU threads.
The whole run has a 25-minute scheduling cap; each arm still has 800 updates.
Budget exhaustion, numerical failure and an unevaluated arm are not negative
architecture evidence. All-arms failure leaves task/recipe usability unresolved.

One seed is exploratory. No automatic extra seeds, sweep, 3D extension or monitor.
The most relevant comparison is inertial_rd versus momentum_nca. Merely beating
rd_nca does not establish a benefit beyond generic momentum. Repair claims require
pre-damage correctness; zero eligible samples produce null conditional metrics.

Commands run from the standalone repository root; replace NEW_* with fresh names:

```powershell
python new/nca_inertial_wind_tunnel/math_checks.py --out runs/NEW_CHECKS/linear_checks.json
python new/nca_inertial_wind_tunnel/run_wind_tunnel.py --device cuda --arms all --preflight --minutes 5 --out runs/NEW_PREFLIGHT
python new/nca_inertial_wind_tunnel/run_wind_tunnel.py --device cuda --arms all --seed 0 --steps 800 --minutes 25 --out runs/NEW_SCREEN
```

Local checks: eight unit tests passed; both 20,000-point analytic stability
classifications had zero mismatches. The four full-size CUDA preflight arms each
completed three updates and abbreviated evaluation. These are implementation
checks, not evidence of learning superiority. Public records are
[validation.json](../../evidence/inertial_seed0/validation.json) and
[linear_checks.json](../../evidence/inertial_seed0/linear_checks.json).

## Completed result

All four arms completed the frozen screen. Inertial RD underperformed generic
momentum at the training size (69.94% versus 84.44% BA at T64), then degraded at
longer rollouts. No arm reached sustained aggregate 95% BA. Candidate repair has
zero eligible maps and is unevaluable. The actual code and frozen THEORY.md /
WIND_TUNNEL.md remain byte-identical to the executed snapshot; publication edits
only change reading guidance, evidence export and validation tooling. The
scientific label "negative exploratory screen" is a descriptive interpretation,
not a newly claimed preregistered binary gate. No further training was run.
