# Delayed-credit cue-once StreamingCell screen

Protocol ID: delayed_credit_cue_once_streaming_v0. This is a four-block
developmental comparison with exactly three arms and 12 total trajectories.
It tests temporal credit assignment in the existing 5,033-parameter
StreamingCell; it adds no architecture or hyperparameter sweep.

## Task and shared input contract

Use seeded Region Identity from new/nca_inertial_wind_tunnel/tasks.py. Each
traversable connected component receives an independent binary label and a
signed source cue. The model never receives region IDs, component labels, graph
distances, or targets. The paired source-flipped view changes one component's
cue and label while leaving geometry unchanged.

Keep the StreamingCell C24/Z8 architecture, initialization, and parameter
count unchanged (new/streaming_carry/stream_cells.py and
new/workspace_revision/revision_cells.py). All three arms share the same
cue-once input contract: initial(x) receives the full mask and two source
channels; every recurrent step receives stepgeometry(mask,0,0), with the mask
in channel 0 and zero in both cue channels. The sourceonlyinitial and
stepgeometry(mask,0,0) names describe this shared input contract, not
additional arms. Assert that the task bank resolves to the exact task source
path before training.

Source map: new/nca_inertial_wind_tunnel/tasks.py,
new/streaming_carry/stream_cells.py,
new/workspace_revision/revision_cells.py,
new/delayed_credit/training.py, and new/delayed_credit/evaluation.py.

## Arms and gradient paths

Each optimizer update runs one 64-step trajectory, followed by one update.
Losses use the task's balanced binary cross-entropy normalization.

| Label | Supervision and gradient path |
| --- | --- |
| A: dense_k8 | Mean of the eight losses at steps 8, 16, ..., 64. Detach state after each 8-step window and accumulate gradients over all windows. |
| B: terminal_k8 | L64 only. Detach after steps 8, 16, ..., 56; backpropagate through the final 8 steps. |
| C: terminal_k64 | L64 only, with no gradient cuts through the 64-step trajectory. |

Do not use a 64x4 reset regime or update the optimizer between windows.
Before training, verify that B and C produce identical forward states and L64
from identical parameters and inputs. B's encoder gradient is expected to be
None because the initial-state graph is cut. Record connected-tensor counts
and gradient norms by module (E/F/Q/R); shared F/Q modules remain trainable
in B's final window. This diagnostic makes no gradient-approximation claim.

## Training schedule

Train three arms in each paired block. Block IDs and seeds are:

| Block | Initialization seed | Minibatch schedule seed |
| --- | ---: | ---: |
| 0 | 150001 | 151001 |
| 1 | 150002 | 151002 |
| 2 | 150003 | 151003 |
| 3 | 150004 | 151004 |

Within each block, copy identical initial weights across arms and use the same
minibatch sequence. Rotate the three-arm order by one position each block.
Use the existing frozen 512-map, size-32 training bank from seed 10002.
Run 300 updates per arm, batch size 8, AdamW with learning rate 0.001, weight
decay 0.0001, betas (0.9, 0.999), epsilon 1e-8, and gradient clipping at 1.
Update 300 is the fixed endpoint; impose no timed cap.

Save weights and training curves at updates 50, 100, 150, 200, 250, and 300.
Evaluate only update 300. Keep per-arm status, completed update count, errors,
and available diagnostics explicit so failed and incomplete records remain
visible.

Before the screen, qualify deterministic B/C forwards, including terminal
states and L64, and confirm B's encoder gradient is None. Run one actual-shape
update for each arm at batch 8 and size 32; record elapsed seconds, peak
allocated memory, and the resulting full-screen time estimate. These checks
are implementation qualification, not additional model blocks.

Launch through the existing protected-job path, which supplies temporary idle
prevention for the run. Do not create a recurring monitor.

## Evaluation and gates

Use 32 fresh size-32 maps from seed 152032 and 32 fresh size-64 maps from seed
152064. These held-out sets are new, not the earlier seed-122 or seed-500
sets. At update 300, evaluate original and source-flipped views at T64, T128,
and T256. T64 is primary. T128/T256 hold and regression results are secondary;
a nonfinite secondary horizon does not erase a valid finite-T64 reach result.
Do not apply the historical Full gain-at-least-5 rule.

The T64 reach gate uses size-32 strict distance 16 < d < 32. Require pooled
and mean per-map accuracy in that band to each be at least 0.80, original and
flipped open-region balanced accuracy each at least 0.85, at least 16 eligible
maps, and at least 100 strict-band pixels. Missing or invalid T64 support
fails the reach gate.

The secondary hold gate compares T128 and T256 with T64: each original/flipped
balanced-accuracy drop must be at most 0.03, and strict-band pooled and
mean-map coverage drops at most 0.05; the trace must remain finite through
T256. Report hold separately from reach.

## Reporting and interpretation

The independent unit is the paired model block (four total), not a map, pixel,
or repeated evaluation. Keep every block's A/B/C pass, fail, or unknown
classification and every failure or incomplete record. Report reach counts
out of four and B/C paired reach outcomes among C-qualified blocks and over
all blocks. Descriptive score wins/losses/ties compare size-32 T64 strict-band
mean-map accuracy for C versus B; report reach-gate contrasts separately.
Show dense A counts independently: A failure does not invalidate a qualified
B/C contrast, but leaves any dense-rescue claim unqualified.

Only issue a verdict after all four primary A/B/C records reach update 300:

* C qualifies in fewer than 3 blocks: BASELINE_UNQUALIFIED.
* C qualifies in at least 3 blocks and C passes while B fails in at least 3
  paired blocks: DEVELOPMENTAL_CREDIT_GAP.
* Otherwise, if C qualifies in at least 3 and B passes in at least 3 blocks:
  NO_K8_GAP_ON_THIS_SCREEN.
* Otherwise: MIXED_DEVELOPMENTAL_RESULT.

These four blocks support a development decision only. Do not claim population
reliability, equivalence, or a broad negative result. reporting.py writes the
compact JSON and Markdown summaries.
