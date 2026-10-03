# Reproduction and source bindings

The frozen executed run used the source files identified by SHA-256 in [run_metadata.json](run_metadata.json) and in the root publication manifest. The original local run passed historical checkpoint and payload replay checks. The local checkpoints and trace archives are intentionally excluded; their relative names, byte sizes and SHA-256 hashes are listed in the root publication manifest.

A public clone uses the dedicated wrapper below. The wrapper is a reproduction path, not a byte-for-byte verification of the private historical checkpoint files. It checks the published parameter hashes against its public stage payloads and records `original_checkpoint_files_verified: false`; the original local run's stronger private file-binding result remains in the frozen evidence record.

From the repository root, CPU metric and binding qualification is:

```powershell
python -X utf8 -B tools/replay_transition_public.py --check --out analyses/transition_20261003_public_check.json
```

GPU reproduction commands are:

```powershell
python -X utf8 -u -B tools/replay_transition_public.py --preflight --qualification analyses/transition_20261003_public_check.json --out runs/transition_20261003_public_preflight
python -X utf8 -u -B tools/replay_transition_public.py --preflight-dir runs/transition_20261003_public_preflight --out runs/transition_20261003_public_replay
```

These commands are provided for independent reproduction; this publication export did not launch a GPU job. The frozen protocol and metric definitions are in [new/transition_100_200/PROTOCOL.md](../../new/transition_100_200/PROTOCOL.md) and [new/transition_100_200/metrics.py](../../new/transition_100_200/metrics.py).

The excluded raw checkpoint and trace archive names, sizes and hashes are in [input_provenance.json](input_provenance.json).

`tools/export_transition_evidence.py --verify-only` checks published table arithmetic, data hashes, source bindings, and local Markdown links without importing the model runtime.
