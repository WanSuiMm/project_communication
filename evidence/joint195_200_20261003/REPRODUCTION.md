# Verification and reproduction

Use the repository root and the pinned [requirements](../../requirements.txt). The executed inference sources are bound in [run_metadata.json](run_metadata.json) and [JOINT195_PUBLICATION_MANIFEST.json](../../JOINT195_PUBLICATION_MANIFEST.json); post-run validation/export tools have separate hashes. No original architecture or training helper changed for this audit.

## Public CPU checks

These checks need the compact public evidence and code only:

```powershell
python -X utf8 -B tools/export_joint195_evidence.py --verify-only
python -X utf8 -B new/audit_195_200/check_instrument.py
python -X utf8 -B new/audit_195_200/check_state_interventions.py
python -X utf8 -B new/audit_195_200/check_metrics.py
python -X utf8 -B new/audit_195_200/check_probes.py
python -X utf8 -B new/audit_195_200/check_analyze.py
```

The public verifier checks hashes,492-case inventory, all exported rate totals and equal-map means, and factorial/Shapley identities. It does not rerun inference or reconstruct omitted trajectories. The fixtures check the original-clock step and logit telescope, reciprocal swaps/null projection, hand-calculated risk/cohort metrics, pulse semantics and analyzer arithmetic.

## Offline checks requiring the local raw archive

[Saved-artifact validation](validation/local_validation.json) recomputes saved Boolean endpoints, interval/cohort survival, turnover, endpoint margins and baseline trajectory quantiles, checks source/reference bindings and plan invariants, and verifies all16-coalition/24-order factorial results. It uses CPU only and performs no model inference. Nonbaseline full-trajectory margin quantiles and solved-age matching-bin membership are not independently reconstructed; these limitations are explicit in its report.

```powershell
python -X utf8 -B new/audit_195_200/validate_saved.py --run runs/audit195_200_20261003_joint01 --out analyses/NEW_SAVED_VALIDATION.json
```

Two earlier validator reports remain local. The first had17 failures from applying primary historical replay checks to new confirmation banks and including a trailing comma in streamed factorial parsing. The second's2304 failures came from a numeric comparator receiving nested factorial dictionaries, despite identical saved/recomputed values. The comparator was repaired and checked with positive/negative fixtures before the final report; scientific run artifacts were unchanged.

## Independent GPU reproduction

Checkpoints and raw banks are excluded. A public clone first recreates the dense seed4 reference archive using the [preceding reproduction path](../transition_20261003/REPRODUCTION.md):

```powershell
python -X utf8 -B tools/replay_transition_public.py --check --out analyses/NEW_TRANSITION_CPU.json
python -X utf8 -u -B tools/replay_transition_public.py --preflight --qualification analyses/NEW_TRANSITION_CPU.json --out runs/NEW_TRANSITION_PREFLIGHT
python -X utf8 -u -B tools/replay_transition_public.py --preflight-dir runs/NEW_TRANSITION_PREFLIGHT --out runs/NEW_TRANSITION_REFERENCE
python -X utf8 -B tools/replay_joint195_public.py --reference-run runs/NEW_TRANSITION_REFERENCE --check-bindings
python -X utf8 -u -B tools/replay_joint195_public.py --reference-run runs/NEW_TRANSITION_REFERENCE --preflight --out runs/NEW_JOINT_PREFLIGHT
python -X utf8 -u -B tools/replay_joint195_public.py --reference-run runs/NEW_TRANSITION_REFERENCE --preflight-dir runs/NEW_JOINT_PREFLIGHT --out runs/NEW_JOINT_REPLAY
```

The adapter leaves the frozen runner unchanged, redirects its reference archive, checks published195/200 parameter and primary-bank tensor hashes, and records `original_input_archive_bytes_verified: false`. It cannot verify excluded original checkpoint file bytes. Historical FP32 settings are PyTorch2.5.1, two CPU threads, cudnn benchmark/deterministic false, cudnn TF32 true and matmul TF32 false. Exact reproduction across devices/backend changes is not promised; a hash/replay mismatch is a qualification failure, not permission to silently relax the protocol.

GPU commands above are provided for independent reproduction and were not run during publication. Publication used the completed original local archive for CPU validation and tested the public adapter's binding-only mode against that archive; it did not independently retrain a fresh clone. The [frozen protocol](../../new/audit_195_200/PROTOCOL.md) distinguishes execution controls from exploratory scientific interpretation.
