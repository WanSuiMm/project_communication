"""Publish/verify the completed raw-state lift with saved arrays only."""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/reparam_gcr_20261009_01"
PUBLIC = ROOT / "evidence/reparam_gcr_20261009_01"
PUBLICATION = ROOT / "REPARAM_GCR_PUBLICATION_MANIFEST.json"
ARM = "reparam_gcr_k8"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


R = module("_reparam_public_runner", ROOT / "new/reparam_gcr/run.py")
A = module("_reparam_public_helpers", ROOT / "tools/export_learnable_gcr.py")


def verify_saved(folder):
    agg, manifest, status, qual = [A.read(folder / n) for n in ("aggregate.json", "manifest.json", "status.json", "qualification.json")]
    assert status["status"] == agg["status"] == "COMPLETE"
    assert status["completed_units"] == status["total_units"] == 1
    assert manifest["config"] == agg["config"] == qual["config"] == R.CONFIG
    assert qual["status"] == "PASS" and qual["source_hashes"] == manifest["source_hashes"]
    for rel, digest in manifest["source_hashes"].items():
        assert A.sha(ROOT / rel) == digest
        if (folder / "source" / rel).exists():
            assert A.sha(folder / "source" / rel) == digest
    for rel, digest in manifest["data_hashes"].items():
        assert A.sha(folder / rel) == digest
    old_agg = A.read(R.REFERENCE / "aggregate.json")
    assert A.sha(R.REFERENCE / "aggregate.json") == manifest["reference_aggregate_sha256"]
    locked = {arm: old_agg["results"][arm]["evaluations"]["heldout"]["horizons"]["T64"]["r2"] for arm in ("gcr_k8", "gcr_k64")}
    assert locked == manifest["locked_reference_r2"] == qual["locked_reference_r2"]
    curve = [__import__("json").loads(line) for line in (folder / ARM / "training.jsonl").read_text().splitlines()]
    assert [row["update"] for row in curve] == list(range(1, 151))
    assert all(np.isfinite(row["loss"]) and np.isfinite(row["gradient_norm_before_clip"]) for row in curve)
    counts = {"training_updates": 150, "final_metric_rows": 0, "final_prediction_values": 0,
              "intermediate_evaluations": 0, "checkpoint_bindings": 0, "source_bindings": len(manifest["source_hashes"])}

    def compare(arrays, horizon, saved):
        y, pred = arrays["y"].astype(np.float64), arrays[f"pred_{horizon}"].astype(np.float64)
        assert np.isfinite(pred).all() and saved["finite"] and saved["n"] == len(y)
        var = np.mean((y - y.mean()) ** 2)
        mse = np.mean((y - pred) ** 2)
        np.testing.assert_allclose([mse, 1 - mse / var, var],
            [saved["mse"], saved["r2"], saved["target_variance"]], rtol=1e-12, atol=1e-12)
        assert np.array_equal(arrays[f"count_{horizon}"], arrays["lengths"])
        assert saved["count_matches_length"]

    for bank in ("heldout", "length64", "length96"):
        arrays = dict(np.load(folder / ARM / f"{bank}_final_predictions.npz", allow_pickle=False))
        bank_y = np.load(folder / "banks" / f"{bank}.npz", allow_pickle=False)["y"]
        assert np.array_equal(arrays["y"], bank_y)
        for t, saved in agg["evaluations"][bank]["horizons"].items():
            compare(arrays, t, saved)
            counts["final_metric_rows"] += 1
            counts["final_prediction_values"] += len(bank_y)
    for u in (50, 100, 150):
        arrays = dict(np.load(folder / ARM / f"heldout_u{u:03d}.npz", allow_pickle=False))
        saved = A.read(folder / ARM / f"heldout_u{u:03d}.json")["horizons"]["T64"]
        compare(arrays, "T64", saved)
        counts["intermediate_evaluations"] += 1
        checkpoint = folder / ARM / "checkpoints" / f"u{u:03d}.pt"
        if checkpoint.exists():
            value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        else:
            value = A.read(folder / ARM / "checkpoints" / f"u{u:03d}.binding.json")
            assert value["update"] == u and value["formal_endpoint"] == (u == 150)
        assert value["config"] == R.CONFIG and value["source_hashes"] == manifest["source_hashes"]
        assert value["next_update"] == u + 1
        counts["checkpoint_bindings"] += 1
    r = agg["evaluations"]["heldout"]["horizons"]["T64"]["r2"]
    assert agg["decision"] == status["decision"] == R.decision(r, locked)
    assert agg["decision"]["verdict"] == "NEAR_FULL_CREDIT_DEVELOPMENTAL"
    assert counts == {"training_updates": 150, "final_metric_rows": 7, "final_prediction_values": 896,
                      "intermediate_evaluations": 3, "checkpoint_bindings": 3, "source_bindings": 9}
    return {"status": "PASS", "method": "Saved arrays and hashes only; no training or model inference", **counts}


def export():
    assert not PUBLIC.exists() and not PUBLICATION.exists(), "Fresh publication destination required"
    validation = verify_saved(RUN)
    receipt = A.read(ROOT / "runs/reparam_gcr_20261009_01_guard_7fc8074e4941/launch_receipt.json")
    assert receipt["status"] == "COMPLETE" and receipt["worker_exit_code"] == 0 and receipt["idle_sleep_request_cleared"]
    PUBLIC.mkdir(parents=True)
    for name in ("RESULTS.md", "aggregate.json", "status.json", "manifest.json", "qualification.json"):
        shutil.copyfile(RUN / name, PUBLIC / name)
    shutil.copytree(RUN / "banks", PUBLIC / "banks")
    (PUBLIC / ARM).mkdir()
    for path in (RUN / ARM).iterdir():
        if path.is_file() and path.suffix in (".json", ".jsonl", ".npz"):
            shutil.copyfile(path, PUBLIC / ARM / path.name)
    for u in (50, 100, 150):
        checkpoint = RUN / ARM / "checkpoints" / f"u{u:03d}.pt"
        value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        binding = {"update": u, "next_update": value["next_update"], "formal_endpoint": u == 150,
                   "checkpoint_sha256": A.sha(checkpoint), "config": value["config"],
                   "source_hashes": value["source_hashes"], "weights_published": False}
        A.write(PUBLIC / ARM / "checkpoints" / f"u{u:03d}.binding.json", binding)
    A.write(PUBLIC / "validation.json", validation)
    A.write(PUBLIC / "execution_provenance.json", {"execution": "COMPLETE", "worker_exit_code": 0,
            "protected_on_demand_job": True, "runtime_cap": None, "recurring": False,
            "idle_sleep_request_cleared": True, "private_receipts_published": False})
    agg = A.read(PUBLIC / "aggregate.json")
    with (PUBLIC / "final_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bank", "horizon", "n", "mse", "r2", "target_variance", "finite", "count_matches_length"])
        writer.writeheader()
        for bank, ev in agg["evaluations"].items():
            for t, met in ev["horizons"].items():
                writer.writerow({"bank": bank, "horizon": t, **met})
    (PUBLIC / "REPRODUCTION.md").write_text("""# Saved-data reproduction

Read RESULTS.md, aggregate.json and final_metrics.csv first. Exactly one new
K8 arm ran150 updates; u150 is primary. Old GCR-K8/K64 references are locked in
the earlier evidence/learnable_gcr_20261009_01 package, not retrained or selected.
Their aggregate hash is bound in manifest.json. Earlier frozen results remain.

From repository root with CPU NumPy/Torch (no CUDA is used for verification):

```powershell
python -X utf8 -B tools/export_reparam_gcr.py --verify-only
```

The package retains exact token/path banks, teacher and shared schedule, all150
training records, three checkpoint bindings and stage evaluations, and all
saved per-example predictions for seven final metric rows. NPZ arrays load
with allow_pickle=False. Weights/optimizer contents/private receipts stay local.

Qualification records per-step forward and full-gradient equivalence at initial
and old trained-reference parameters, plus CUDA short-credit gradient/cut checks.
No training or inference occurs during publication. Checkpoint bindings describe
local pre-export verification; omitted checkpoint contents cannot be rechecked
from hashes alone. Re-running the full prelaunch qualification requires the
retained old GCR-K64/u150 checkpoint. To regenerate it from scratch, first follow
the prior learnable_gcr protocol, then the new/reparam_gcr/PROTOCOL.md commands
using fresh names in the declared CUDA environment.

The verdict is NEAR_FULL_CREDIT_DEVELOPMENTAL on one known paired block. The
intervention changes state21->73 scalars while preserving5921 trainable parameters
and the full forward function. It supports late parameter placement for this
linear degree-two task, not generic learned NCA recurrence, reliability, or
early-input gradient recovery. Known validation cohorts are reused.
""", encoding="utf-8")
    hashes = {p.relative_to(ROOT).as_posix(): A.sha(p) for p in sorted(PUBLIC.rglob("*")) if p.is_file()}
    A.write(PUBLICATION, {"schema": "reparam-gcr-publication-v1", "source_sha256": A.read(RUN / "manifest.json")["source_hashes"],
            "published_artifacts_sha256": hashes, "validation": validation,
            "checkpoint_contents_published": False, "raw_run_preserved": True})


def verify_public():
    publication = A.read(PUBLICATION)
    hashes = {p.relative_to(ROOT).as_posix(): A.sha(p) for p in sorted(PUBLIC.rglob("*")) if p.is_file()}
    assert hashes == publication["published_artifacts_sha256"]
    validation = verify_saved(PUBLIC)
    assert validation == A.read(PUBLIC / "validation.json") == publication["validation"]
    for path in PUBLIC.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".md", ".csv"):
            assert not A.PRIVATE.search(path.read_text(encoding="utf-8")), path
    return {**validation, "files": len(hashes), "bytes": sum(p.stat().st_size for p in PUBLIC.rglob("*") if p.is_file())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        export()
    print(__import__("json").dumps(verify_public(), indent=2))
