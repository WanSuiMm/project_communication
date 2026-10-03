# Seed4 local rollback follow-up

Run status: COMPLETE. Training: none. Checkpoint: seed4 K8, 300 updates.

The intervention restores one neighboring cell’s W/Z to its own preceding macro-step. The sham applies the wrong neighbor’s own rollback direction, scaled separately for W and Z to match the selected neighbor’s rollback norm in each source world.

| Size | Events | Eligible maps | Horizon | Native − sender rollback | Sham − sender rollback | Native acquisition | Sender acquisition | Sham acquisition |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 93 | 31 | 1 | 14.52% | 3.76% | 15.59% | 1.08% | 4.84% |
| 32 | 93 | 31 | 4 | 33.53% | 20.18% | 84.83% | 51.30% | 71.48% |
| 64 | 257 | 31 | 1 | 7.00% | 0.44% | 22.80% | 15.79% | 16.23% |
| 64 | 257 | 31 | 4 | 28.60% | 15.33% | 74.71% | 46.11% | 61.43% |

Primary outcomes are paired correctness at p after one step; four steps are secondary. Effects average events within each eligible map, then average maps. Reused maps, times, and events are correlated; model replication count is one.

This is a selected local temporal-state sensitivity contrast. Whole-cell W/Z rollback is not an edge-message knockout, and equal-norm sham does not match activation context or direction. The result does not establish causal flood fill, unique edge necessity, or neuron semantics. Off-cone invariance and the duplicated no-op trajectory are implementation qualifications only.

Read compact_summary.json and replaychecks.json first; events.json is the event-level record.
