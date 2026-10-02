# Streaming seed 4 operator audit v1

Frozen before audit measurement, 2026-10-02. No training. Question: which
fixed transport and direct neighborhood pathways does this already successful
Streaming checkpoint depend on during cold computation? This is a selected
single-model diagnostic, not an architecture comparison or replication.

## Fixed inputs and provenance

- Original `stream_K8_seed4.pt` from the completed Streaming Carry run;
  300 updates, K8. Verify file hash against the publication manifest and
  parameter hash against its raw record. Load on CPU with weights_only=True.
- Original evaluation banks 40032/40064: 32 maps at each size 32/64. Verify
  complete tensor hashes against the original manifest before using CUDA.
- Keep all parameters, W24/Z8 widths, initialization, input injection,
  residual scales, F/Q ordering and readout unchanged. No optimizer, backward,
  hidden-state ground truth, other seeds, tuning or new training data.
- Check every bound historical source against current and saved executed
  copies; snapshot the new audit sources in a new output directory.
- Same FP32/backend settings as the historical run. Record local runtime,
  PID, device, command and provenance in the ignored run directory.

## Four cold-rollout interventions

Use the same loaded model, fresh initial(X), and fresh initial(X_flip) for
each condition. Apply its switches at every step starting at t=0:

| Condition | Workspace carry | F and Q neighborhood inputs |
|---|---|---|
| full | T_M(W) | masked Laplacians retained |
| no_transport | W | masked Laplacians retained |
| no_perception | T_M(W) | zero tensors in the same input slots |
| neither | W | zero tensors in the same input slots |

F consumes incoming carry, Z, pre-stream L(W), L(Z), X. Its residual makes
Wplus. Q consumes Wplus, Z, L(Wplus), L(Z), X, then Z'=Z+0.5Q. Switch off
both Laplacian slots in both networks together; do not change input widths
or silently retain Q neighborhood access. Switching T to I also changes
the incoming feature supplied to F. Fixed carry and residual response to
that feature are therefore tested together, not cleanly separated.

The same macro-step count does not imply equal communication opportunity.
Upper-bound graph hops per step are 2 for full/no_transport, 1 for
no_perception, and 0 for neither. Record and check each actual light cone.
Legacy radius-16 bands retain their original definitions for replay and are
not called beyond-window computation for every intervention.

## Verification and measurements

One CPU smoke: nonzero random residual weights, all four conditions versus
an independent neighbor/scatter/conv reference, all-on versus original cell,
parameter hash and input-state immutability. Then one local CUDA audit with
a 300-second cap, stopping on source/hash drift, failed replay, nonfinite
states/metrics, parameter mutation or a light-cone violation.

Run full first at both sizes. At T64/T128/T256 compare its entire saved
evaluation structure: integer counts exact; accuracy metrics atol1e-6;
continuous metrics atol1e-6/rtol1e-5. Do not proceed to knockouts until all
six size/horizon reference records pass. Also record descriptive T8/16/32.

Primary: size32/T64 strict16<d<32 original-and-flipped paired per-map mean
and pooled correct/count. Retain both BAs/BCEs, every original distance band,
all per-map denominators/hits, actual light-cone source-flip differences,
W/Z RMS and source-flip logit RMS. T128/T256 assess persistence and size64
d>32 is a secondary transfer diagnostic; neither can replace the primary.

Report no_transport/full and no_perception/full differences, and the
factorial descriptive interaction y11-y01-y10+y00 at each size/horizon/band.
Also give per-map paired fraction differences on identical eligible maps.
Pixels/maps are not independent model replications: n=1 selected checkpoint.

## Interpretation limits

Report numeric pathway sensitivity rather than a new GO/NO_GO. For a concise
label, a >=20pp pooled primary drop at BOTH T64 and T256 is a sustained
knockout loss; <=5pp absolute pooled differences at ALL T64/128/256 means
primary retention under that intervention. Other outcomes remain mixed.
These labels are descriptive and are not significance or necessity proofs.

Knockouts alter learned input/state distributions and available communication
depth. Failure means the frozen solution is not robust to that intervention;
it does not establish the reason for failed training, the value of H/C/Z,
which path naturally carries all semantics, or whether a retrained knockout
can succeed. The both-off condition is a no-communication sanity control:
source-dependent paired correctness at d>0 must be zero. The prior Streaming
multi-seed DEVELOPMENT_NO_GO and positive seed4 observation remain unchanged.

Commands from repository root, always with a new output name:

    python new/stream_path_audit/check.py
    python new/stream_path_audit/audit.py --out runs/NEW_STREAM_PATH_AUDIT
