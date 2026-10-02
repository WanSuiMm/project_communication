# Stationary sidecar: frozen development screen v1

Question: can additional persistent local state improve the reproducibility of
the original StreamingCell's reach-and-hold behavior, compared with both that
core and a same-parameter instantaneous side branch? This is authorized new
training after the selected-seed4 operator audit, not another knockout audit.
Previously inspected seeds/maps make this a development result only.

## Architecture and intervention

Keep original W24 (four six-channel carrier lanes), Z8, encoder, F67->40->24,
Q67->16->8, readout, masked streaming and masked Laplacian clocks unchanged.
The stream arm is the original StreamingCell, 5033 parameters. Two side arms
have 7989 parameters and H12 initialized to zero:

```
U=.1*G([W,Z,L_M(W),L_M(Z),X])          # G67->32(tanh)->12
H*=H+U  (memory); H*=U  (stateless)
I=T_M(W)
W'=I+.1*(F([I,Z,L_M(W),L_M(Z),X])+P_F(H*))
Z'=Z+.5*(Q([W',Z,L_M(W'),L_M(Z),X])+P_Q(H*))
H'=H* (memory); H'=0 (stateless)
logits=readout(Z')
```

G reads only original core features, so no zero H input columns disable
parameters in the stateless arm. Both consume the instantaneous H* before
the stateless reset. The single structural difference is retaining H across
steps. H is not transported directly; original perception/transport remain
the only spatial operators. At most two graph edges per macro step and16
per K8 gradient window, including the side path.

Create old core modules before any new modules, with identical old RNG draws.
G_in uses default initialization; G_out weights N(0,.02), bias0. P_F12->24
and P_Q12->8 are bias-free and initialized to zero. Identical seed produces
identical full side-arm parameter tensors and identical core tensors across
all three arms. At P_F=P_Q=0, W/Z/logits and their core derivatives match
the old model even for arbitrary finite H; full states have different sizes.
Nonzero G features permit first-update P_Q gradients. Original Q_out=0 blocks
the initial P_F task gradient; P_F and all G layer gradients must open by
update3. Neutral forward/core-gradient equality does not imply equal
AdamW updates: extra gradients change joint norm clipping.

This tests a concrete sidecar/carry parameterization, not the necessity of
local memory: the old Z8 is already persistent and stationary. Parameter
capacity is matched between memory/stateless only. Stateless means no extra
H carry: its instantaneous outputs still feed recurrent W/Z. Carry changes
temporal accumulation and activation scale together; log state/injection RMS
and do not infer a scale-independent memory mechanism. No FLOP matching, complete
stability guarantee, seed4 mechanism claim, or universal short-BPTT claim.

## Frozen execution

Seeds2,3,4,5; all models start cold, no checkpoint warm start. Fixed arm order:
seed2 stream/memory/stateless; seed3 memory/stateless/stream;
seed4 stateless/stream/memory; seed5 stateless/memory/stream.
300 AdamW updates/arm, lr.001, weight_decay.0001, batch8, global clip1.
Fresh64-step trajectory each update, losses at8/16/.../64, equal loss weights.
K8: backward eight windows; detach all state tensors without changing values;
one optimizer update after all eight backwards. Same frozen trainer and
FP32/backend settings as historical Streaming (cudnn benchmark=False,
deterministic=False, tf32=True; matmul tf32=False; torch threads2).

Reuse frozen training bank size32/512 maps seed10002, index schedule seed20002,
and32 paired original/flipped maps at each size32/64, seeds40032/40064.
Evaluate T64/128/256 and all13 frozen distance bands. No new loss, pool,
curriculum, topology, gate, adaptive horizon, auxiliary supervision or sweep.

## Qualification and stop rules

Primary: size32/T64 strict16<d<32. Reach requires BOTH paired per-map mean and
pooled accuracy>=.80 AND original/flipped balanced accuracy>=.85. Hold requires
at BOTH T128/T256: each BA loses<=.03, each primary paired metric loses<=.05
relative to T64. Empty primary bands cannot pass. Seed is the independent unit.

All12 arms must complete300 updates. All four stream controls must reproduce
historical final parameter hashes AND full evaluation payloads exactly, with
seed4 reach+hold preserved. Otherwise CONTROL_REPRODUCTION_DRIFT; partial or
error/time-capped execution is INCOMPLETE, never a scientific pass.

DEVELOPMENT_GO requires memory reach+hold>=3/4 including seed4, and STRICTLY
more successes than BOTH stream and stateless. A tie with stateless cannot
support the persistent-memory interpretation. Otherwise DEVELOPMENT_NO_GO.
Show paired memory-minus-stream and memory-minus-stateless differences by seed,
with denominators and empty-band nulls. Far-size gains cannot replace the gate.
Four reused seeds do not establish population reliability/significance; a GO
would justify a separately authorized fresh-seed confirmation only.

## Bounded engineering preflight and launch

One focused CPU check: shared initialization/7989 counts, neutral nonzero-core
forward/derivatives, independent nonzero update reference, component isolation
and two-hop light cone, three-state K8 detach clock, feedback/G gradient entry,
decision truth table. CUDA preflight:3 updates for each arm at seed2, abbreviated
size32/T64 evaluation of two maps,180-second cap. Require finite state/gradients,
all identities, P_Q gradients on update1, P_F/G gradients by update3,
peak allocated<3GiB, full-bank primary coverage>=16 maps/500 pixels.

Let c be the worst arm's median synchronized update duration over updates2/3.
Proceed only if 12*300*c*1.20+180<=2400 seconds. Fixed formal cap40 minutes,
watchdog grace60 seconds. Do not reduce updates or tune architecture if this
fails. All sources, protocol, launcher, historical control records, banks,
schedule and preflight are hash-bound before dispatch. A launch is complete
when PID/manifest and first training update are verified and receipt saved.
No background monitoring or automatic follow-up experiment.

From repository root (new output names required):

```
python new/stationary_sidecar/check.py --out analyses/NEW_SIDECAR_CHECK.json
python new/stationary_sidecar/run.py --preflight --out runs/NEW_SIDECAR_PREFLIGHT
pwsh -File tools/launch_stationary_sidecar.ps1 -RunName NEW_SIDECAR_RUN -Preflight runs/NEW_SIDECAR_PREFLIGHT
```

Read aggregate.json / RESULTS.md first after completion. Raw arm JSON and
curves.csv retain integer denominators, clipping, gradient groups, runtime
and memory; checkpoints/source snapshots/launch receipts remain local.
