# Joint 195--200 causal audit (audit195_200_v1)

Question: in the original StreamingCell, which changes between historical
u195 and u200 account for improved acquisition and preservation, conditional
on this selected training trajectory? This is an exploratory mechanism audit,
not a new architecture gate or a proof of a computational phase transition.
No training, architecture replacement, runtime cap, watchdog or monitor.

## Binding and controls

Load immutable checkpoints and banks from
`runs/transition_20261003_seed4_dense01`. Require exact state-dictionary hashes
and source bindings. Use the historical FP32 PyTorch2.5.1 backend settings.
Before interventions, reproduce both checkpoints on both frozen 16-map banks
(61032/61064), all 257 steps, original and cue-flipped worlds. Boolean traces
must match exactly; paired margins must agree within 2e-7. Nonfinite control
states, binding drift, replay mismatch, failed sham/telescope checks stop
execution. A nonfinite counterfactual is a recorded scientific failure of that
arm; remaining prespecified arms continue without rescue or tuning.
New artifacts never overwrite old evidence.

Primary banks are these selected 16-map banks. Independent diagnostic banks
have 8 maps each, seeds62032/62064. They confirm map generalization only;
there is one selected training trajectory, no new training-seed replication.
Map is the unit of averaging; pooled cell counts are descriptive. All summaries
retain per-map denominators. Empty cohorts and zero perturbations are reported.

## Fixed intervention suite

1. Weight interpolation: lambda=0,.1,...,1 from195 to200 in the existing
   parameter coordinates. Report continuous margins as well as Boolean risk.
   This finite-horizon curve cannot by itself establish a bifurcation.
2. All16 combinations of checkpoint195/200 blocks E(encoder), F(f_in/f_out),
   Q(q_in/q_out), R(readout). This includes forward replacements and reverse
   rollbacks. Compute effects over all24 block orders from this full factorial,
   including Shapley average and order ranges. Necessity/sufficiency is
   conditional on the other blocks and these task maps. Transport is fixed at
   both endpoints; its parameter delta is zero.
3. Cross-continuation at T64: 2 state producers x 2 continuation rules x 2
   readouts. Readouts are evaluated post hoc on the same Z trajectory, never
   fed back into dynamics. Diagonal arms reproduce the endpoint controls.
   Use each arm's own initial-correct cohort and also the shared cohort correct
   under all producer/readout combinations. Preserve the whole interaction cube.
4. Sparse within-map, same-role reciprocal state swaps at T64: at most8
   disjoint pairs/map, same plan in both cue worlds. Eligible changed/open,
   non-source cells are matched by paired label class, local degree, distance
   bands, absolute margin bands and solved-run-age bands. These are explicitly
   correctness-conditioned interventions, not label-blind deployment rules.
   Roles: solved; frontier = wrong with a four-neighbor solved cell. Fixed
   pre-intervention cohorts never change in response to a counterfactual.
   Components W,Z,both, four separate W lanes, each of eight Z channels,
   readout span and readout nullspace. Identity sham must be exact. Nullspace
   swaps must preserve immediate logits within1e-6. Report perturbation norms,
   target outcomes and graph-distance rings0,1--4,5--16,17--32,33+ from targets.
   Same-coordinate local transplants195<-200 and200<-195 on common solved or
   common frontier cells use W,Z,both. These interventions can be off manifold;
   failure is not a general impossibility of compositional representations.
   Confirmation repeats W/Z/both/span/null plus all local transplants; individual
   channels/lanes remain primary-bank exploration.
5. One-step update audit on actual states at64,96,128: original-clock incoming,
   L(W),L(Z),F,Q, learned dW, total dW and dZ. Fixed linear-readout projection
   gives exact signed logit changes. Project every recorded state/update under
   both195/200 readouts while holding its update rule fixed. The
   Q(W,Z)->Q(TW,Z)->Q(TW+etaF,Z)
   telescope is order-dependent mediation accounting. Q-input preactivation
   blocks are W,Z,LW,LZ,X; tanh prevents additive semantic attribution.
   At64, pulse interventions in solved/frontier cells remove F, Q, streaming,
   or one Q feature block for exactly one macro step, then restore the original
   rule and continue to256. X knockout removes mask and both cue channels from
   Q inputs together; the transport mask remains intact. These are conditional
   sensitivity tests, distinct from checkpoint parameter changes.

## Endpoints and interpretation

All interventions continue to256. Primary outcomes are strict changed-region
coverage at128 (16<graph distance<32), acquisition64->128, first-exit64->128
and continuous survival64->256. Also report endpoint destruction, repeated
correct/wrong turnover, signed original/flipped margins, fixed-cohort margins
and spatially localized downstream effects. Sources and walls are excluded.
Common endpoint-correct cohorts prevent denominator drift from masquerading
as preservation. A correct readout does not establish latent completion.

Independent banks repeat interpolation, all factorial blocks, cross-continuation,
the prespecified coarse state swaps/transplants and all pulses without selection
or tuning. Agreement is diagnostic confirmation; disagreement is retained.
Further adaptive probes, if needed, get a separate protocol amendment/output.

## Outputs and execution

`manifest.json` binds source/checkpoint/bank hashes and backend; `status.json`
records phase/case progress; `cases/*.json` and `summary.json` are compact
canonical evidence. Baseline replay raw traces, compact intervention Boolean
traces and endpoint margins, swap plans, selected state tensors and update
summaries remain local. `RESULTS.md` is generated from canonical summaries.
Source snapshots and a local machine-specific launch receipt are retained.

From repository root, always use new output names:

    python -X utf8 -B new/audit_195_200/run.py --preflight --out runs/NEW_AUDIT_PREFLIGHT
    pwsh -File tools/launch_audit_195_200.ps1 -RunName NEW_AUDIT -Preflight runs/NEW_AUDIT_PREFLIGHT

Launch qualification runs the small CPU fixtures, four exact full replay
controls and representative suffix/sham/pulse checks, then estimates runtime
from measured inference. The estimate is descriptive and never a stop criterion.
