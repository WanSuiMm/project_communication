# Reproduction and public verification

From the repository root, run the public CPU arithmetic and hash checks:

```powershell
python tools/export_warmstart.py --verify-only
python new/warmstart/check.py --out analyses/warmstart_cpu01.json
```

The first command uses only the Python standard library. It checks the publication hashes, all eight complete phenotype gate summaries, the matched-frontier count arithmetic, the integer turnover identity, and the sanitized public tree. It performs no training, model inference, CUDA work, or GPU detection.

The second command is the frozen focused CPU qualification. It checks the unchanged historical training path, detached-prefix gradient behavior, sticky prefix finiteness, and the diagnostic arithmetic fixtures. Use a new output path that does not already exist.

## Fresh-clone references and GPU rerun

The required historical references are already public in `evidence/streaming_carry_init2345/`: its manifest, shared schedule, and `stream_K8_seed2.json` through `stream_K8_seed5.json`. The frozen warm-start runner reads these public files directly and verifies them through `STREAMING_CARRY_PUBLICATION_MANIFEST.json`; no private run archive or checkpoint restoration is required.

After the CPU qualification, a new formal screen can be run with fresh output names:

```powershell
python new/warmstart/run.py --preflight --qualification analyses/warmstart_cpu01.json --out runs/warmstart_preflight_new
python new/warmstart/run.py --out runs/warmstart_formal_new --preflight-dir runs/warmstart_preflight_new
```

The preflight estimate is informational under `detached_warmstart_v1_nocap`; the amended protocol has no runtime qualification limit or watchdog. Its gradient, replay, memory, source, and finite-state gates remain fixed. The formal run uses the frozen four initialization pairs and the same fixed training bank, batch schedule, and sampled prefix ages. The in-repo [protocol](../../new/warmstart/PROTOCOL.md) defines the full recipe and stop criteria.

## What this public package reproduces

The public tool verifies saved arithmetic and binding hashes. It does not reconstruct training or inference from excluded model weights and NPZ traces. Full local validator outcomes, when included, are CPU verification of saved records; they do not repeat the historical GPU replay or the 8-arm training. The full sanitized local validation receipt is available at [validation/local_validation.json](validation/local_validation.json).
