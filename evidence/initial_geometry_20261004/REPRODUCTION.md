# Reproduction and evidence scope

Public arithmetic/hash verification from the repository root (standard library):

    python -X utf8 -B tools/export_learning_geometry.py --verify-only

The public summaries preserve all four seeds, four banks, three instantaneous
feature views, eight K8 and eight full-history tangent views, and their spectra.
Per-map distance-stratum counts and relative separations are in per_map_bands.csv.
Full target-feature NPZs, checkpoints and original input archives remain local.
The export independently recomputed all304 relation Gram spectra and alignments
from saved feature arrays; it did not repeat model inference. Tangent/autograd
sanity was part of the original CPU diagnostic, not this publication.

Exact independent measurement uses the excluded historical update0 checkpoints
and paired map banks at the relative archive locations declared in the protocol:

    python -X utf8 -u -B new/initial_geometry/audit.py --out runs/NEW_INITIAL_GEOMETRY
    python -X utf8 -B new/initial_geometry/summarize.py --run runs/NEW_INITIAL_GEOMETRY

These commands require those archives; a fresh clone can verify the published
evidence without them. Do not mistake public verification for exact trajectory
reproduction. Dependencies are in ../../requirements.txt: Torch2.5.1, NumPy and
Matplotlib. The audit is CPU-only and takes no optimizer step. The existing
initialization hashes in ../streaming_carry_init2345/raw/ provide public identity
anchors. Review the [protocol](../../new/initial_geometry/PROTOCOL.md) first.
