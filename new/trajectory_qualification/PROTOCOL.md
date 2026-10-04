# Serial training-state, reliability and second-task qualification

Protocol: `trajectory_qualification_v1`. Frozen before this suite's efficacy
results. User authorized all three experiments serially on 2026-10-04;
stage2 depends on stage1, and stage3 is an independent task. No runtime cap,
watchdog, periodic monitor, adaptive seed replacement or endpoint selection.
Existing evidence/checkpoints are read-only. All new outputs use a new run.

## Fixed original recipe

Unchanged `StreamingCell`, 5033 parameters, W24 in four directional lanes/Z8,
zero initial F/Q output heads. Original training bank512, size32, seed10002.
AdamW lr=.001, wd=.0001, betas=(.9,.999), eps=1e-8, global clip1, batch8,
300 updates. Every update numerically executes64 steps, with eight equally
weighted task losses8..64 and K8 detaches of both states. FP32 and historical
local Torch2.5.1/CUDA settings are unchanged. K8 is the credit window, not the
training execution horizon. Tests extend execution to128/256 and size64.

## Stage1: update3 parameter × Adam state × suffix (8 arms)

Use saved H and S20022 update3 checkpoints from
`runs/bootstrap_20261004_seed4_01/checkpoints/`. Cross parameter source H/S,
Adam moment source H/S and fixed suffix H/S (updates4..300). Keep parameter
order, all optimizer hyperparameters, and all per-parameter step counters3.
No optimizer reset. A transplant copies donor m/v while parameters are held,
or donor parameters while Adam state is held. No activation-state transplant.

Four native controls run first. They must exactly reproduce prior final
parameter hashes and complete saved phenotype metrics:

| Parameter | Adam | Suffix | Native reference | Prior Full |
|---|---|---|---|---|
| H | H | H | H | PASS |
| H | H | S | S20022_preserve_early3 | PASS |
| S | S | H | S20022_replace_early3 | PASS |
| S | S | S | S20022 | FAIL |

Then run four hybrids: H parameters/S moments and S parameters/H moments,
each on H and S suffixes. Primary outcome is the unchanged original Full gate
on reused32-map size32/64 banks50032/50064. Preserve all Boolean trajectories,
frontier strata and continuous endpoint metrics. These selected suffixes and
one init4 do not estimate a population success rate. A rescue is conditional
on S suffix; H suffix tests preservation, not a component main effect.

### Prespecified stage1 → stage2 selector

1. Select `shadow_moments` if thetaS/mH passes Full on BOTH suffixes.
   The comparison is native thetaS/mS PASS→PASS on H and FAIL→PASS on S.
2. Otherwise select `shadow_parameters` if thetaH/mS passes BOTH suffixes.
   The comparisons use identical recipient moments and fixed suffixes.
3. Otherwise select `shadow_joint`, provided thetaH/mH passes BOTH native
   suffix controls. This selects the whole donor package, not an isolated
   mediator. When both component candidates qualify, prefer moments because
   recipient parameters stay unchanged.

This selector chooses one donor-derived recipe; it does not establish a
general representation convention, main effect, basin, or reset mechanism.
Any native replay mismatch makes stage1 unqualified and skips stage2. Stage3
remains independent. Numerical failure stops execution without a rescue sweep.

## Stage2: one selected recipe versus baseline,16 fresh pairs

Fresh initializations31001..31016, independent schedule seeds32001..32016.
Schedules are300x8 IID draws with replacement from the original512-map bank.
Freeze them before stage1 efficacy runs. Both arms in each pair use identical
initial parameters and the same complete recipient schedule. Fresh primary
evaluation banks:32 maps at size32/64, seeds83032/83064, unchanged Full gate.
None of these initialization/schedule/evaluation seeds is selected by outcome.

For each fresh initialization, compute a three-update SHADOW from that same
initialization using the FIXED historical H first three minibatches. Save its
parameters and Adam state. After recipient update3, baseline retains its
state; treatment imports shadow moments, shadow parameters or both according
to the frozen selector. Adam clock remains3, later batches/optimizer settings
are identical. Shadow preparation is shared once per pair and is not an
efficacy-selected model. Save before/after parameter and optimizer bindings.

The fixed, previously selected H prefix changes the treatment distribution.
An independent IID bootstrap followed by an IID main schedule would have the
same sequence law as ordinary IID sampling; that ineffective contrast is
NOT this experiment. The chosen donor prefix is a conditional candidate, and
this fresh paired test is its first reliability assessment.

Independent units:16 initialization+schedule pairs. Report baseline/treatment
Full counts, paired wins/losses, delta, Wilson intervals and exact two-sided
McNemar/binomial p on discordant pairs. Qualification requires net improvement
>=4/16 AND p<=.05. Otherwise report `NO_RELIABILITY_QUALIFICATION`, including
continuous retention/progress/regression and uncertainty. No cell/map/time
count serves as a training replicate. All32 arms run regardless of scientific
failures. A negative result is retained; no extra recipes/seeds are tried.
The claim concerns this frozen recipe and task distribution, not geometric
basin volume or universal trainability.

## Stage3: independent multi-source geodesic-distance task (8 arms)

See [DISTANCE_PROTOCOL.md](DISTANCE_PROTOCOL.md). Keep the exact cell/K8,
optimizer and300-update budget. Change task semantics to multi-source minimum
distance, fixed output scale32 and masked regression. Eight fresh baseline
initializations41001..41008 and schedules42001..42008. Trainbank81032;
32-map size32/64 evaluation banks82032/82064. Stage3 does NOT consume stage1
selection or stage2 results; it examines the frequency of long-execution
progress with preservation on a different local iterative task.

The algorithmic shortest-path/min-plus control verifies task construction,
not qualification of a learned baseline. If no model reaches the short-task
endpoint, classify the tested recipe/budget as task-unqualified, not a general
architecture impossibility. Report the task-specific gate and continuous
metrics separately from the original paired-source-flip Full gate.

## Execution and evidence

One bounded combined sanity check verifies source/checkpoint schemas, all
selectors/statistical fixtures, the new task oracle, and small finite CUDA
training for original/new loss paths. Source/reference/schedule hashes bind
the dispatch; all source snapshots and final checkpoints remain local.
The persistent controller executes1→2→3 with durable status/manifest/receipt,
per-stage summaries, raw trajectories and training curves. Stage2 selection
is automatic and recorded; stage3 is independent even if stage1 replay is
unqualified or stage2 has a non-numerical runner error. Nonfinite values or
CUDA device/memory failures stop the entire suite; ordinary scientific
qualification failures never suppress stage3. No continuous monitoring is started.

Commands from repository root (new output names required):

```text
python -X utf8 -B new/trajectory_qualification/run.py --check --out analyses/NEW_SERIAL_CHECK.json
pwsh -File tools/launch_trajectory_qualification.ps1 -RunName NEW_SERIAL_RUN -Qualification analyses/NEW_SERIAL_CHECK.json
```

Expected work:8 stage1 continuations,32 stage2 trajectories,8 stage3
trajectories. Historical timing suggests roughly1.5–2 hours on the local GPU;
this is descriptive, not a time limit. Read each stage's RESULTS.md/summary.json
first. Checkpoints, raw arrays, logs and machine launch records are secondary;
machine records must be excluded from any later public export.
