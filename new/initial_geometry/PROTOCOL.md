# Update-zero geometry diagnostic v1

Question: does the historical successful initialization seed4 already differ
from seeds2/3/5 in accessible relation features or task tangent geometry?
Zero training, CPU only. No optimizer step, new architecture, parameter sweep,
or success-selected feature extraction. Existing evidence is read-only.

Use the four exact historical update0 baseline checkpoints from the paired
warm-start replay; compare their parameter hashes to original Streaming records.
Reuse the joint195/200 primary16 and diagnostic confirmation8 maps at sizes32/64.
These banks have already been inspected: confirmation here is a second-bank
consistency check, not an independent scientific replication. The independent
initialization unit is seed (n=4); pixels/maps do not increase that count.

Fixed measurements before execution:

1. Per-direction lane RMS, centered RMS, max/min ratio and centered flattened
   cross-lane cosine at initial state, open cells only. Cosine depends on payload
   coordinates; it is not an invariant notion of useful lane diversity.
2. Q-hidden16 feature covariance spectra at times0/8/64 on fixed targets:
   changed-component cells, max128/map, evenly spaced within distance strata
   0..8,9..16,17..32,>32, with a cap32/stratum and no seed-dependent selection.
3. Source-flip relation separation and relation-feature Gram spectrum. Record
   geometric availability separately: initial cue support travels through the
   exact port permutation; Q can inspect incoming W and one masked neighbor
   shell, plus the fixed local cue. No signal before availability is expected.
   Report per-map means, denominators, and near/far strata. Availability is an
   upper bound: cancellation may leave a reachable feature difference zero.
4. At zero heads, define h_t=tanh(q_in(features(W_t,0,X))), t>=1,
   H_j=.5*sum_{t=8j+1}^{8j+8} h_t. The numerical trajectory is W_t=T^t E(X).
   The actual K8 endpoint tangent uses H_j; a full-history endpoint tangent uses
   cumulative sum H_0+...+H_j. For paired differences, bias terms cancel and
   K_rel=||readout.weight||^2 * DeltaH DeltaH^T. Retain both versions at all eight
   windows, spectral rank/effective rank, trace and normalized spectrum, and
   alignment with the known paired target difference. Nonzero condition numbers
   exclude numerical null directions using one fixed relative threshold1e-8;
   report the discarded/rank dimensions explicitly. A huge full sample Gram has
   rank<=16; its ordinary smallest eigenvalue is therefore not an indicator.
5. Run the original K8 backward helper once per seed on the historical FIRST
   training batch, without optimizer/clip. Report each parameter gradient norm;
   this tests which branches initially receive task loss, not AdamW updates.

One numerical sanity check: compare analytic parameter-to-logit tangents to
autograd at full8/full16/detached16 on the same sampled pixel; also check zero
Z/logits and initial parameters remain unchanged. Stop on source/hash drift,
nonfinite values or mismatch>2e-6. All metrics are exploratory; no significance
test, causal initialization claim, new architecture gate, or training reliability
claim. In particular same seed4 initialization failed the full phenotype under
four changed schedules, so initialization alone is already known insufficient.

Outputs in a new runs directory: summary.json, per-map rows, feature arrays,
RESULTS.md, source/input/checkpoint hashes, completion receipt. No publication is
part of this request. Runtime is recorded; no training or recurring monitor.

Command from repository root:

    python -X utf8 -u -B new/initial_geometry/audit.py --out runs/NEW_INITIAL_GEOMETRY
