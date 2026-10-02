# Local persistent state with a learned communication interface

Frozen before efficacy training, 2026-10-02. This development screen asks
whether keeping all32 persistent state channels local and exposing a learned
interface makes sustained K8 computation more reproducible than the existing
additive baseline and Streaming Carry. Previously inspected seeds/maps are
reused; this is not fresh confirmation. Earlier scientific decisions remain
unchanged. No mechanism of the previous stream seed4 is assumed.

## Exact architecture and comparison boundary

The controls are unchanged RevisionCell('ws_additive') and StreamingCell().
The candidate keeps W24 and Z8 at their spatial positions through fixed
identity paths. M has24 transient channels, split into four6-channel N/E/S/W
ports. M is newly emitted in each of the two phases; it is not an additional
persistent carrier and does not survive a step as an independent state.

```math
\begin{aligned}
M_1 &= E_\theta(W,Z,X), & U_1 &= T_M M_1,\\
W' &= W+0.1F_\theta(W,Z,U_1,X),\\
M_2 &= E_\theta(W',Z,X), & U_2 &= T_M M_2,\\
Z' &= Z+0.5Q_\theta(W',Z,U_2,X).
\end{aligned}
```

E is one shared pointwise affine35->24 map. F and Q are pointwise Tanh MLPs
with widths59->31->24 and59->21->8. Their output layers are zero initialized.
The encoder3->24 and readout8->1 match the controls' initial draws exactly.
Initialize the old default cell first, retain encoder/readout, then create E
and replace F/Q. Random seeds are shared, but differently shaped F/Q and E
cannot have identical complete parameter sets. Record both full initial hashes
and common encoder/readout hashes; never claim full candidate initialization
identity. There are exactly5033 parameters: encoder96, E864, F2628, Q1436,
readout9. Persistent state remains32 channels and message width remains24.

T is the existing masked port permutation: open ports move to the neighboring
cell in the same direction; blocked/exterior links bounce into the opposite
lane; wall ports stay fixed. No wrapping, dropping or averaging. Only E's
messages cross cell boundaries in the candidate. F/Q do not read L(W), L(Z),
raw neighboring states or component IDs. Two sequential message phases retain
the controls' bound of at most two graph hops per macro-step and radius16
inside one K8 window. X remains available at every update, as in the controls.

If F=Q=0, W'=W and Z'=Z regardless of emitted messages. Pure T is lossless;
E, the residual updates and repeated receive/store/re-emit computation are
not guaranteed to be lossless or stable. This candidate explicitly tests an
ephemeral interface, rather than the previous persistent carrier carry.

This is a comparison of complete parameterizations. Local state allocation,
neighbor representation and hidden-width allocation change together. Matching
5033 parameters,32 persistent channels,24 message channels and two phases does
not isolate the causal effect of interface factorization or match FLOPs,
activation memory or latency. No claim of a unique RelationFirst mechanism,
new primitive, general routing, warm editing, arbitrary horizon or3D success.

## Fixed training, endpoint and decision

- Variants baseline/stream/interface; initialization seeds2,3,4,5;12 arms.
  Orders: seed2 baseline/stream/interface; seed3 stream/interface/baseline;
  seed4 interface/baseline/stream; seed5 interface/stream/baseline.
- Exact historical training bank10002 (512 size32 maps), schedule20002
  (300x8), evaluation banks40032/40064 (32 maps each). Bind bank, schedule,
  evaluation maps and control initial parameters to published prior evidence.
  Candidate common encoder/readout initialization must match both controls.
- Fixed300 updates per arm. Fresh64-step trajectories; losses at8,16,...64
  equally weighted. Unchanged K8 trainer detaches BOTH W/Z without changing
  their values, accumulates gradients across windows, clips once at norm1,
  and performs one AdamW update after the full trajectory. lr0.001,
  weight_decay0.0001, batch8, FP32, same CUDA flags. No pool, normalization,
  gate, momentum, auxiliary loss, optimizer change or efficacy-selected stop.
- Fresh original/flipped evaluations at sizes32/64 and T64/128/256. Primary:
  size32/T64 STRICT16<d<32, paired per-map mean AND pooled accuracy>=80%,
  original AND flipped BA>=85%. Hold at BOTH T128 and T256: each BA declines
  <=3pp, each primary paired statistic declines<=5pp from its own T64.
- Controls must reproduce historical complete evaluation payloads and final
  parameter hashes. Baseline seeds2/5 and stream seed4 must also reproduce
  reach+hold. Any completed control mismatch is CONTROL_REPRODUCTION_DRIFT,
  not a qualified architecture verdict.
- DEVELOPMENT_GO requires all12 arms complete, qualified controls, interface
  reach+hold>=3/4 including seeds2/5, and its count exceeds BOTH concurrent
  controls. Otherwise a complete matched experiment is DEVELOPMENT_NO_GO.
  Any error, incompletion or time cap is INCOMPLETE, never scientific success.
- Keep every seed, size, horizon and distance band, integer hits/denominators,
  per-map mean and pooled metrics. Report interface-minus-baseline AND
  interface-minus-stream effects separately. Empty bands remain null. Farther
  gains or a transient peak cannot replace the frozen primary and hold gate.
- Independent unit is initialization seed (n=4), conditional on one shared
  training bank/schedule. No p-value or population reliability claim; inspected
  data/seeds make any positive result developmental.

## Engineering checks, budget and launch boundary

One CPU suite checks5033 parameters and32-channel persistent state; control
factories and common initialization; explicit outgoing-port/reference updates
and nonzero gradients; zero-residual local identity; two-hop light cone and
component separation; and K8 gradient/optimizer-clock consistency. Verify
decision truth cases before the CUDA preflight. Existing pure-transport proofs
and tests remain unchanged; do not reinterpret them as learned-cell guarantees.

One CUDA preflight runs3 updates of all3 variants at seed2, with abbreviated
evaluation. If c is the largest median of the last2 synchronized update times,
require12*300*c*1.20+180<=2400 seconds, peak allocated<3GiB, finite quantities,
matching identities and primary coverage>=16 maps/500 pixels. Otherwise stop
before formal training without reducing300 updates or selecting by accuracy.
The preflight tests engineering, not scientific efficacy.

Formal hard cap40 minutes; emergency watchdog41 minutes. No automatic retry,
architecture rescue, sweep, fresh-seed confirmation or monitor. Save executed
sources/hashes, parameter/data/schedule bindings, checkpoints, raw integer
metrics, timing/clipping/memory, aggregate/curves/effects, and a private local
host/PID/GPU/command/time receipt. Use new run directories. Prior sources,
evidence, checkpoints and negative decisions remain untouched.
Source bindings retain all35 paths from STREAMING_CARRY_PUBLICATION_MANIFEST.json,
then add that manifest, the four historical stream raw records, the new cell,
CPU check, runner, this protocol and launcher. Formal sources must match the
passed preflight exactly; preserve an executed source snapshot in the run.

For this authorized launch, completion means verified dispatch and a durable
receipt. Inspecting final results is a separate request; do not poll the job.

Commands from repository root:

    python new/local_interface/check.py
    python new/local_interface/run.py --preflight --out runs/NEW_INTERFACE_PREFLIGHT
    pwsh -File tools/launch_local_interface.ps1 -RunName NEW_INTERFACE_RUN -Preflight runs/NEW_INTERFACE_PREFLIGHT
