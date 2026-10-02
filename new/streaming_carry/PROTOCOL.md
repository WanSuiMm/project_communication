# Lossless streaming path: matched K8 development screen

Frozen before efficacy training, 2026-10-02. This separately authorized screen
tests a port-permutation carry with a local residual update. The previous
rho0.5 averaging-carry DEVELOPMENT_NO_GO remains unchanged. This is development
on inspected initialization seeds/maps, not fresh confirmation.

## Operator and state

Reuse W24 as four6-channel payload lanes (N,E,S,W), and Z8 as stationary
local state. Total persistent state is32 channels, with the same5033 trainable
parameters, encoder, F/Q/readout dimensions, parameter draws and initialization
as the additive baseline. No extra carrier channels or expanded memory.

For an open cell i and lane d, T sends (i,d) to (i+delta_d,d) if the destination
is open; if blocked by a wall or exterior boundary, it sends (i,d) to
(i,opposite(d)). Wall-cell ports stay fixed. Isolated open cells reverse lane
directions in place. Each destination has exactly one predecessor: T is a
permutation of position/direction registers, independently for each payload
coordinate. No zeroing, averaging, periodic wrap, broadcast, learned routing
or transport normalization. With B reversing lanes, T^-1=B T B. Consequently
T^T T=I and pure transport preserves global Euclidean distances and norms.
These are properties of the fixed operator, not of the full learned cell.
Directional ports identify incoming lanes, not complete source provenance.

## Exact cell and control

```math
\begin{aligned}
U &= T_M W,\\
W' &= U+0.1F_\theta(U,Z,L_MW,L_MZ,X),\\
Z' &= Z+0.5Q_\theta(W',Z,L_MW',L_MZ,X).
\end{aligned}
```

Baseline sets T_M=I and is the unchanged RevisionCell('ws_additive'). The
candidate sets T_M to the permutation above. The F feature width remains67:
24 incoming +8 stationary +24 old L(W) +8 old L(Z) +3 inputs. Old Laplacian
perception remains auxiliary input to the residual branch; the fixed carry
itself never aggregates. Thus this tests a permutation path within the
existing two-phase cell, not a pure stream/collision model with all neighbor
perception removed. It also changes F's first feature to incoming values,
as specified; it is not the prior intervention with an unchanged F input.

T(W), L(W) and L(Z) are computed from the OLD state in parallel, so W' depends
on at most one graph hop. Q's subsequent perception adds at most one hop.
Do not use L(T(W)) in F: that would introduce a third hop and invalidate the
K8 spatial-radius16 comparison. If F and Q are zero, W_t=T^t W_0 and Z_t=Z_0;
the residual branch may otherwise amplify, erase or mix messages. Streaming
does not restore gradients across detach, guarantee useful routing, or retain
all local W information at its old spatial position. Readout stays on Z.

## Matched training, endpoints and gate

- BOTH baseline/stream arms, initialization seeds2,3,4,5:8 runs. Orders by
  seed: baseline/stream, stream/baseline, baseline/stream, stream/baseline.
- Exact Phase-II bank10002 (512 size32 maps), schedule20002 (300x8), eval
  banks40032/40064 (32 maps each). Bind initial/data/schedule/eval identities
  to historical evidence as well as within each matched pair.
- Fixed300 updates per arm. Fresh64-step trajectories, equally weighted
  losses at8,16,...64; K8 detaches BOTH W/Z while keeping values, accumulates
  parameter gradients across windows, clips once at norm1, then one AdamW
  update. lr0.001, weight_decay0.0001, batch8, FP32, same CUDA flags. No state
  pool, normalization, gate, revision, momentum, auxiliary loss or sweep.
- Evaluation is unchanged: fresh original/flipped runs, sizes32/64, T64/128/256.
  Primary size32/T64 STRICT16<d<32, paired mean AND pooled>=80%, both BA>=85%.
  Hold at BOTH T128/T256: each BA drops<=3pp and each primary paired statistic
  drops<=5pp relative to its own T64. Report reach and reach+hold separately.
- DEVELOPMENT_GO requires all8 arms complete, stream reaches AND holds in
  >=3/4 including2/5, and its count exceeds concurrent baseline. Baseline2/5
  must reproduce reach+hold; otherwise BASELINE_REPRODUCTION_DRIFT. A complete
  matched failure is DEVELOPMENT_NO_GO. Any incomplete/error/time cap is
  INCOMPLETE, never a scientific pass. No efficacy-dependent stop or tuning.
- Preserve all old absolute/normalized bands, size64, T128/T256, per-map and
  pooled denominators, every seed and paired effects. Empty bands are null.
  Farther gains or a transient peak cannot replace the fixed primary/hold.
- Independent unit: initialization seed,n=4 conditional on one train bank
  and schedule. No p-value or cross-dataset reliability claim. Reusing the
  chosen seeds/maps makes a positive result developmental only.

## Engineering, budget and provenance

One CPU check: independent explicit port scatter on all binary2x3 masks,
permutation/inverse/norm preservation, adjoint gradient, no component crossing,
walls/exterior/isolation, T=I old-cell forward/gradient equivalence, nonzero
residual reference, zero-residual rollout, two-hop light cone and K8 clock.
No learned efficacy is used for these checks.

One CUDA preflight:3 updates of both arms, seed2, abbreviated evaluation.
Let c be the larger median of the last2 synchronized update times. Require
8*300*c*1.20+120<=1500 seconds, peak allocated<3GiB, finite values, matching
identities and primary coverage>=16 maps/500 pixels. Otherwise stop before
formal training without changing300 updates or selecting settings by scores.
Formal cap25 minutes, emergency watchdog26 minutes. No retries, confirmation,
alternative topology, coefficient sweep or monitoring is started automatically.

Save source snapshots/hashes, metadata including lane layout and boundary rule,
initial/final parameter hashes, data/schedule bindings, checkpoints, all raw
metrics/denominators, timings/clipping/memory, aggregate/status and a private
host/PID/GPU/command/time launch receipt. Outputs use new directories; frozen
earlier sources and evidence are unchanged. Completion of this launch request
means verified dispatch plus a receipt; result inspection is a separate step.

Commands from repository root:

    python new/streaming_carry/check.py
    python new/streaming_carry/run.py --preflight --out runs/NEW_STREAM_PREFLIGHT
    pwsh -File tools/launch_streaming_carry.ps1 -RunName NEW_STREAM_RUN -Preflight runs/NEW_STREAM_PREFLIGHT
