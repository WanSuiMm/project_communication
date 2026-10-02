# Incremental review: persistent local, carrier and task roles

- Review base: `891f968d85e84e1bc47e0fd3bcbe83210c55f2f6`.
- Evidence head: `84af24f36c44bac7acfe3f972ef281e0c812557e`.
- This following handoff-only commit changes review metadata, not code or evidence.

All 12 matched K8 arms completed 300 updates in 1204.234 seconds (20.07 minutes).
**DEVELOPMENT_NO_GO**: reach-and-hold counts are baseline 2/4, stream 1/4,
roles 0/4. All eight control final parameter hashes and complete evaluation
payloads reproduce their historical records exactly. This is development
evidence on previously inspected seeds/maps, conditional on one bank/schedule.
The independent comparison unit remains initialization seed, n=4. Counts
pooled across seeds are descriptive; rollup map counts repeat the same maps.

## Minimal reading order

1. [Current results](evidence/persistent_roles_init2345/RESULTS.md) and
   [compact analysis](evidence/persistent_roles_init2345/analysis.json).
2. [Frozen protocol](new/persistent_roles/PROTOCOL.md),
   [cell equations and shapes](ARCHITECTURE.md), and
   [exact cell](new/persistent_roles/role_cells.py).
3. [Runner and decision](new/persistent_roles/run.py),
   [validation](evidence/persistent_roles_init2345/validation.json), and
   [publication bindings](PERSISTENT_ROLES_PUBLICATION_MANIFEST.json).
4. For detailed arithmetic, use [curves](evidence/persistent_roles_init2345/curves.csv),
   [paired effects](evidence/persistent_roles_init2345/paired_effects.csv),
   [CPU checks](new/persistent_roles/check.py), and
   [saved-result audit](new/persistent_roles/analyze.py).

The 12 [raw arm JSONs](evidence/persistent_roles_init2345/raw/) are secondary.
Raw metrics and schedule are copied byte-identically. Public manifest,
completion and preflight copies omit private process/device metadata.
Checkpoints, machine launch receipts and logs remain local and are excluded.

## Decision-relevant change

The candidate has stationary H12 workspace, persistent C12 carriers
(four directions, three channels each), and stationary Z8 task state.
A shared R35 -> 72 -> 32 Tanh rule emits simultaneous residuals for all
three blocks. Each of two phases per macro step computes
`H' = H + 0.1 Delta_H`, `C' = T(C + 0.1 Delta_C)`, and
`Z' = Z + 0.5 Delta_Z`. Only C moves through the existing masked port
permutation, with blocked/exterior ports bouncing back and wall ports fixed.
There are no direct Laplacian or neighbor inputs. Encoder, rule and readout
total 5033 parameters, matching the historical controls.

Encoder/readout initialization draws match the controls. H and C split the
old 24-channel encoder output; Z starts at zero. Complete candidate
initialization differs. With zero residuals and a fixed mask, the base
`I_H ⊕ T_C ⊕ I_Z` is isometric. The full learned recurrence has
no losslessness or stability guarantee. K8 gives 16 carrier phases and
read-before-stream Z radius at most 15; the primary strict d>16 is unchanged.

Historical Streaming already has moving W24 and stationary Z8. This recipe
adds a dedicated stationary workspace while changing carrier width,
perception, rule sharing and readout clock together. It is a test of that
complete parameterization, not an isolated causal test of H or the first
architecture to combine moving and stationary state.

Primary size 32/T64 strict 16<d<32 paired correctness:

| Seed | Baseline pooled % | Stream pooled % | Roles per-map mean % | Roles pooled % | Roles correct / pixels |
|---:|---:|---:|---:|---:|---:|
| 2 | 92.97 | 1.47 | 0.00 | 0.00 | 0/2918 |
| 3 | 17.27 | 5.62 | 0.53 | 0.31 | 9/2918 |
| 4 | 44.76 | 85.92 | 0.00 | 0.00 | 0/2918 |
| 5 | 95.68 | 13.02 | 1.80 | 1.58 | 46/2918 |

Paired correctness requires the original and flipped cue to be answered
correctly at the same changed-region pixel; these values are not whole-grid
ordinary accuracy. All four roles seeds have Hold=True but Reach=False:
holding a failed endpoint does not establish successful remote computation.
The frozen gate requires both primary paired statistics >=80%, both BAs
>=85%, hold at T128 and T256, at least three candidate reach-and-hold seeds
including seeds 2/5, and a success count exceeding both controls.

Longer rollouts do not recover the endpoint. Roles seed 5 has the largest
primary pooled T256 value, 61/2918 = 2.09%. Its size 64/d>32 T256 result is
also the maximum roles far-band pooled value, 65/27436 = 0.24%.
Other horizons and per-map means are available in the compact analysis;
these secondary endpoints do not replace the frozen gate.

Changed: this persistent H/C/Z recipe has a completed negative development
result and fails to make the prior rare Streaming behavior more reproducible.
Unchanged: the earlier Local Interface no-go, Streaming's overall no-go and
sustained seed 4 positive case, Direct Spatial Carry's no-go, Phase-II/Phase-I
decisions, and earlier revision, dynamics and transport findings.
No failure mechanism is isolated. No general impossibility, population
reliability, significance, arbitrary-horizon or 3D claim follows.

The saved-result audit verifies 51 executed source hashes against current
files and both snapshots, 12 CPU checkpoint state hashes, eight historical
control reproductions, data/schedule bindings, 144 BA means, 1008 paired
metric/count aggregates, 12 gates, 936 curve rows and 624 paired contrasts.
No new training or checkpoint inference was performed for publication.

## Questions for review

1. Does the cell implement persistent stationary H/Z and moving C, with
   simultaneous residuals, the stated readout clock and no hidden neighbor
   inputs? Are the fixed zero-residual invariants kept separate from learned
   recurrence stability claims?
2. Are initialization matching, all eight exact control reproductions,
   integer denominators, null bands and the frozen reach/hold decision
   consistent across the summaries, raw records and CSVs?
3. Is this specific bundled negative result kept separate from the broader
   role-separation principle and the previous Streaming seed 4 positive case?
