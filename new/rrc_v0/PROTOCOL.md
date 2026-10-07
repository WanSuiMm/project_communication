# RRC-v0: direct relation on a factorized carrier writer

Frozen protocol ID: `rrc_v0_prestream_relation_factorization_v1`.
Status: implementation and actual-shape qualification before authorized local dispatch.
No training result is claimed here. No threshold, initialization, or width sweep.

## Question and interventions

Does a directly optimized local lane relation improve formation reliability over
the same factorized content writer under K8 training?

Three paired arms, C24 (N/E/S/W, six payload dimensions each), Z8:

1. `current`: the historical `StreamingCell`, F67→40→24 (5033 parameters).
2. `factorized`: shared F31→96→6, independently applied to four lanes (4983).
3. `rrc`: the identical shared F, plus four relation logits and one scalar (4988).

The PRIMARY contrast is `rrc` minus `factorized`. Current is a secondary
architectural reference. Equal approximate parameter counts do not imply equal
function classes or compute costs.

For the two shared-writer arms:

```
U = T_M(C_t)
H = U                              # factorized, exact fixed identity
H = U + rho * (sum_j pi_j P_j U-U) # rrc
p_d = shared_F([H_d, L_M(C_t)_d, Z_t, L_M(Z_t), X])
C_next = H + 0.1*p
Z_next = Z_t + 0.5*Q(C_next,Z_t,L_M(C_next),L_M(Z_t),X)
y = readout(Z)
```

Input order is 6+6+8+8+3=31. **Pre-stream** L(C_t) is intentional:
post-stream L(H) would increase the carrier writer's spatial radius and change
the effective K8 spatial credit horizon. All three arms retain the historical
two-hop whole-state clock and evaluation causal bound. Q and readout are unchanged.

P0=identity; PL moves N→W→S→E→N; PO moves opposite; PR=PL inverse.
The operator acts only on the lane axis. RRC alpha=zeros(4), pi=uniform;
gamma=logit(0.1), rho=0.1. One outcome-blind initialization, no retuning.
The factorized arm has no trainable relation parameters. Encoder, Q and readout
are copied identically within each block; the two shared F modules start identically.
Current retains its exact historical seeded initialization. F/Q output layers
start at zero as in the historical cell.

## Mathematical limits

Pure masked transport T_M is a permutation with bounce-back on binary fixed masks.
The relation operator A=(1-rho)I+rho*sum pi_j P_j is a convex combination of
permutations: ||A||2<=1, but it generally attenuates directional contrast.
Uniform initial pi gives gain 1 on the constant lane mode and 0.9 on the other
three modes. Residual notation does not make this mixing lossless.
The complete neural recurrence has no stability guarantee. Commutation with a
common payload basis change applies to the relation block alone. Q still reads
all C and can mix lanes through Z; this is direct carrier-writer factorization,
not a proof that all cross-lane communication must use the relation block.
Five scalar parameters describe only three effective mixture degrees of freedom.
Small parameter count is not an optimization theorem.

## Paired training and provenance

Eight independent paired blocks; initialization seeds 120001..120008; schedules
121001..121008. Arm order rotates by block mod 3. SAME minibatch indices and
512-map size32 training bank (seed10002) in each arm of a block. Batch8, FP32,
AdamW lr0.001/wd0.0001/betas(0.9,0.999)/eps1e-8, global gradient clip1.
300 super-updates. Each is four cold64 segments on the SAME minibatch,
32 K8 windows; backward at each window endpoint, detach state after each window,
divide loss by32, one optimizer step at the end. Parameters stay fixed within
the super-update. This is the existing reset64x4 training protocol, not full256 BPTT.

Save u=0,25,...,300, all model and optimizer states. Only u300 is the formal
endpoint. Intermediate checkpoints are formation diagnostics, never model selection.
Fixed held-out paired banks: size32 seed122032 and size64 seed122064,
32 maps each. Same banks across arms/blocks. Rollout to T256 with bit-packed
correctness traces, T64/T128/T256 metrics, retention and regression; old Full
phenotype/frontier measurements at u300 only. 24 trajectories,312 stage records.
No hidden-state ground truth, new regularizer, topology controller, or Q redesign.

## Endpoint and decision

Joint readiness uses the unchanged size32 gate: strict paired changed-pixel
coverage at T64, graph distance16<d<32, BOTH pooled and mean-map >=0.80;
T64-correct retention to T256 >=0.95; >=16 reference maps and >=100 reference
pixels. Missing/support-insufficient metrics do not pass.

For paired u300 readiness, w=#(rrc passes, factorized fails),
l=#(factorized passes, rrc fails). Qualification requires w-l>=6 out of8 AND
exact two-sided discordant-pair sign-test p<=0.05. This is a conservative
finite-seed screening rule, not population or task-general proof. Ties contribute
to counts but not the sign-test sample. Current contrasts, old Full, dense
ever-ready, size64 extrapolation, and relation telemetry are secondary.
Otherwise report NO_RRC_RELIABILITY_QUALIFICATION, preserving all negatives.

## Relation activity and systems measurements

Every update and every checkpoint evaluation records rho, pi, effective
nonidentity weight beta=rho*(1-pi0), cyclic mode gains, relation gradient norm,
and per-lane sqrt(sum||H-U||²/sum||U||²) over open cells. The action ratio is
aggregate energy-based, not a mean of unstable per-cell ratios. Zero denominator
is null. Training measures cold1..64 repeated four times; evaluation separates
steps1..64 and65..256 and original/flipped worlds. Current/factorized action is
exactly zero and their relation parameters are not applicable. These are
descriptive intervention measurements; no post-hoc activity threshold or rescue.

Regular tensor ops, CUDA Graph K8 replay, measured latency and peak memory.
Actual-shape eager/capture checks exercise three optimizer steps in all arms,
nonzero writer/Q outputs and relation gradients; verify identical losses,
gradients, states, Adam moments and observer statistics within frozen tolerances.
CPU checks cover cell plumbing and report pairing. Qualification measures one
32-map two-size rollout per arm to estimate runtime; outcomes do not tune protocol.
No runtime cap, watchdog, recurring monitor, or remote deployment. Dispatch is
verified once after tool yield and recorded in a private local launch receipt.

## Reproduction

From repository root with the existing Torch2.5.1 CUDA environment:

```
python -X utf8 -u -B new/rrc_v0/run.py --check --out analyses/rrc_v0_qualification_20261007_02.json
pwsh -File tools/launch_rrc_v0.ps1 -RunName rrc_v0_20261007_01 -Qualification analyses/rrc_v0_qualification_20261007_02.json
```

Sources, bank hashes, schedules and qualifications are snapshotted. Checkpoints,
launch receipts and machine metadata stay local unless an explicit publication
request triggers a sanitized evidence export. Existing runs are read-only.
