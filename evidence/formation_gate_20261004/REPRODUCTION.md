# Reproduction and evidence scope

Public arithmetic/hash verification from the repository root (standard library):

    python -X utf8 -B tools/export_learning_geometry.py --verify-only

The complete44-record public summary retains per-map numerators/denominators,
all21 dense checkpoints plus300, both spatial sizes and strict-band variants.
The profiles CSV and figure are reading aids; source truth is summary.json.
Original Boolean/logit NPZs remain local. Public verification recomputes rate,
screen and confusion-table arithmetic; it does not reexecute a model or prove
closure. The original analysis checked raw archive hashes and independently
matched the prior long-horizon rates. There is one selected training trajectory.

Given the original saved dense archive, execute:

    python -X utf8 -u -B new/formation_gate/analyze.py --out runs/NEW_FORMATION_GATE

That command reads runs/transition_20261003_seed4_dense01 and requires its
excluded raw trace archive. It performs no inference or training. Full dense
trajectory reproduction is separately documented in
[the preceding reproduction notes](../transition_20261003/REPRODUCTION.md).
This publication did not launch that training. NumPy and Matplotlib dependencies
are listed in the root requirements.txt. Read the
[frozen diagnostic definitions](../../new/formation_gate/PROTOCOL.md).
