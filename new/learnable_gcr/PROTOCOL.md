# Learnable ordered composition: developmental v0

Status: frozen before scientific training. This is a one-block developmental
screen, not an architecture reliability estimate or a universal NCA claim.

## Question and intervention

Does a structured second-order local state reduce the learning gap between
terminal K8 and terminal K64 on a learned ordered-composition task?

Four primary arms: `full_writer_k8`, `full_writer_k64`, `gcr_k8`, `gcr_k64`.
One diagnostic: fixed raw 8D evidence plus a feed-forward MLP, trained on the
same minibatch schedule. The diagnostic receives offline-computed statistics;
none of the four primary models receives global statistics or intermediate
targets. It is explicitly an evidence/readout reference, not a matched NCA.

## Task and data

Independent Gaussian 8D tokens occupy a random simple directed path in a 2D
grid. The predecessor is a Manhattan-adjacent cell; no path revisits a cell.
There is one endpoint target:

\[
y=\tanh\left(\frac{1}{n}\sum_{i<j}(a^T x_i)(b^T x_j)\right).
\]

One fixed teacher is used in every split: orthonormal random vectors `a,b`,
teacher seed 170001, gamma 1. Teacher vectors are unavailable to models.
Training: 512 examples, grid 8, lengths uniformly 16..32, seed 170301.
Held-out primary: 128 examples, grid 8, lengths 16..32, seed 170401.
Long tests: 128 examples each, grid 16, lengths 64 and 96, seeds 170501/170502.
All token values and geometries are generated independently by split.
Saved banks, path coordinates and hashes make all inputs reconstructible.

## Architectures and controls

Both primary architectures use the same directed predecessor graph, persistent
raw local inputs, `Linear(8,4,bias=False)` encoder, 20D workspace plus one count,
and `Linear(21,256)-ReLU-Linear(256,1)` interpretation network. They use exactly
the same initial encoder/interpreter tensors, seed 170101. Count is deterministic
and identical in both models. An endpoint scalar is refreshed with rho=.5,
and prediction is tanh of this scalar. There are no learned gates or auxiliary
losses. Only the endpoint decoder is evaluated: other decoder states never
affect any supervised future state, so this is exact pruning of unused work.

GCR overwrites each node's state with predecessor statistics plus its current
encoded token: S1'=S1pred+p; S2'=S2pred+outer(S1pred,p). Evidence is not decayed.
The readout receives log-count and workspace/count, hence S1/n and **S2/n**.
This explicitly matches the teacher's length scale; no MLP must learn n*m2.

Full Writer overwrites the workspace with predecessor workspace plus
`.1 * MLP(predecessor workspace, p, log-count)`, using a 25->16->20 MLP.
It has a full unconstrained vector output. The readout also gets workspace/count.
No four-direction StreamingCell transport is imported into this different task.
The Full Writer has 6677 parameters versus GCR 5921 (12.8% more); this is reported
openly rather than introducing dummy parameters or changing shared modules.
Both have the same 21 per-node state scalars and one supervised endpoint scalar.

The diagnostic uses fixed raw-token mean and ordered pair sum/n, with count,
and a 73->64->1 ReLU/tanh MLP. It has no recurrent or trainable evidence formation.

## Training, credit and execution

150 updates, batch 16, AdamW lr=.001, weight decay=.0001, betas .9/.999,
epsilon 1e-8; global gradient clipping at 1. Schedule seed 170201 is shared.
Every primary update cold-starts one 64-step rollout and takes one optimizer
step. The **only** loss is endpoint MSE at T64. K64 differentiates all 64 steps;
K8 runs steps 1..56 without gradients and differentiates steps 57..64. States,
forward outputs and scalar loss must agree between credit settings at identical
parameters. The encoder is active in the final window, not a detached cache.
Parameter-independent local inputs allow exact feature caching inside each
gradient window. No AMP, compilation, CUDA Graph, sweep, new seeds, or monitor.

Checkpoints and held-out T64 metrics at u50/u100/u150; **u150 only** is the primary
endpoint. Intermediate measurements are diagnostics, never model selection.
Secondary evaluations: held-out train lengths T128/T256; long lengths T128/T256.
MSE, R2, predictions and endpoint counts are saved per example. Forward timing,
allocated GPU memory, gradient norms and losses are measured for the actual code.

## Frozen decision rules

R2 is relative to the evaluation-bank constant-mean predictor. Both K64 arms must
have primary held-out R2 >= .5; otherwise `POSITIVE_CONTROLS_UNQUALIFIED`.
If qualified but Full Writer's K64-minus-K8 R2 gap is < .10, the verdict is
`NO_IDENTIFIED_CREDIT_GAP`. A `DEVELOPMENTAL_SIGNAL` requires GCR-K8 R2 >= .5,
GCR-K8 minus Full-Writer-K8 R2 >= .10, and GCR K64-minus-K8 R2 <= .05.
Other qualified outcomes are `NO_DEVELOPMENTAL_SIGNAL`. Report the continuous
architecture-by-credit gap, all four metrics, and diagnostic independently.
Numerical failures and unfinished runs are explicit and never counted as a
qualified scientific negative. One block cannot establish success probability.

A success means shared-parameter training from local suffix uses is useful for
this task family; it does not restore gradient to individual distant early writes.
The task is deliberately matched to second-order evidence. General NCA,
learned routing, cyclic duplicate handling and arbitrary learned write are untested.

## Reproduction and protection

From this standalone repository root in the established CUDA environment:

```powershell
python -X utf8 -u -B new/learnable_gcr/run.py --check --out analyses/NEW_GCR_CHECK
& ./tools/start_protected_job.ps1 -JobName NEW_GCR_RUN -Script new/learnable_gcr/run.py -ScriptArguments @('--out','runs/NEW_GCR_RUN','--qualification','analyses/NEW_GCR_CHECK/qualification.json')
```

Use new names. The existing protected on-demand worker provides an independent
process and temporary idle-sleep prevention, with no runtime cap or recurring
trigger. Checkpoints retain optimizer state and the next schedule position.
Resume is explicit with `--resume`, and requires identical source/config/bank
hashes. Existing unrelated results and historical StreamingCell sources stay read-only.
