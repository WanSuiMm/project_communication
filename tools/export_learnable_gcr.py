"""Export and verify the frozen GCR screen using saved arrays only, on CPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/learnable_gcr_20261009_01"
PUBLIC = ROOT / "evidence/learnable_gcr_20261009_01"
PUBLICATION = ROOT / "LEARNABLE_GCR_PUBLICATION_MANIFEST.json"
sys.path.insert(0, str(ROOT / "new/learnable_gcr"))
from run import CONFIG, verdict  # noqa: E402

ARMS = [a[0] for a in CONFIG["arms"]] + ["fixed_raw_evidence_mlp"]
PRIVATE = re.compile(r"(?i)([A-Z]:[\\/]|/home/|/data/users/|(?:\d{1,3}\.){3}\d{1,3}|GPU-[0-9a-f-]{10,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----)")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy(relative):
    source, target = RUN / relative, PUBLIC / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def verify_saved(folder):
    aggregate, manifest, status = [read(folder / n) for n in ("aggregate.json", "manifest.json", "status.json")]
    assert status["status"] == aggregate["status"] == "COMPLETE"
    assert status["completed_units"] == status["total_units"] == 5
    assert manifest["config"] == aggregate["config"] == CONFIG
    assert aggregate["decision"] == status["decision"] == verdict(aggregate["results"])
    qual = read(folder / "qualification.json")
    assert qual["status"] == "PASS" and qual["config"] == CONFIG
    assert qual["source_hashes"] == manifest["source_hashes"]
    for name, digest in manifest["source_hashes"].items():
        assert sha(ROOT / name) == digest
        if (folder / "source" / name).exists():
            assert sha(folder / "source" / name) == digest
    for rel, digest in manifest["data_hashes"].items():
        assert sha(folder / rel) == digest
    banks = {name: dict(np.load(folder / "banks" / f"{name}.npz", allow_pickle=False)) for name in CONFIG["banks"]}
    schedule = np.load(folder / "banks/schedule.npy", allow_pickle=False)
    expected_schedule = np.random.default_rng(CONFIG["schedule_seed"]).integers(0, 512, (150, 16), dtype=np.int64)
    assert np.array_equal(schedule, expected_schedule)
    for name, (count, _, _, _) in CONFIG["banks"].items():
        assert len(banks[name]["y"]) == count
    counts = {"units": 0, "training_updates": 0, "checkpoint_bindings": 0,
              "intermediate_evaluations": 0, "final_metric_rows": 0, "final_prediction_values": 0}

    def compare_metric(arrays, predicted, saved):
        y, pred = arrays["y"].astype(np.float64), arrays[predicted].astype(np.float64)
        assert np.isfinite(y).all() and np.isfinite(pred).all()
        mse = float(np.mean((y - pred) ** 2))
        var = float(np.mean((y - y.mean()) ** 2))
        assert var > 0 and saved["finite"] and saved["n"] == len(y)
        np.testing.assert_allclose([mse, 1 - mse / var, var],
                                   [saved["mse"], saved["r2"], saved["target_variance"]], rtol=1e-12, atol=1e-12)

    for arm in ARMS:
        unit = folder / arm
        result = read(unit / "result.json")
        assert result == aggregate["results"][arm] and result["status"] == "COMPLETE"
        curve = [json.loads(line) for line in (unit / "training.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [row["update"] for row in curve] == list(range(1, 151))
        assert all(np.isfinite(row["loss"]) and np.isfinite(row["gradient_norm_before_clip"]) for row in curve)
        counts["units"] += 1
        counts["training_updates"] += len(curve)
        for update in (50, 100, 150):
            local_checkpoint = unit / "checkpoints" / f"u{update:03d}.pt"
            binding_path = unit / "checkpoints" / f"u{update:03d}.binding.json"
            if local_checkpoint.exists():
                saved = torch.load(local_checkpoint, map_location="cpu", weights_only=False)
                assert saved["config"] == CONFIG and saved["source_hashes"] == manifest["source_hashes"]
                assert saved["next_update"] == update + 1
            else:
                binding = read(binding_path)
                assert binding["update"] == update and binding["next_update"] == update + 1
                assert binding["config"] == CONFIG and binding["source_hashes"] == manifest["source_hashes"]
                assert binding["formal_endpoint"] == (update == 150)
            counts["checkpoint_bindings"] += 1
            met = read(unit / f"heldout_u{update:03d}.json")["horizons"]["T64"]
            arrays = dict(np.load(unit / f"heldout_u{update:03d}.npz", allow_pickle=False))
            assert np.array_equal(arrays["y"], banks["heldout"]["y"])
            compare_metric(arrays, "pred" if arm == ARMS[-1] else "pred_T64", met)
            if update == 150:
                assert met == result["evaluations"]["heldout"]["horizons"]["T64"]
            counts["intermediate_evaluations"] += 1
        for bank_name in ("heldout", "length64", "length96"):
            arrays = dict(np.load(unit / f"{bank_name}_final_predictions.npz", allow_pickle=False))
            assert np.array_equal(arrays["y"], banks[bank_name]["y"])
            for t, met in result["evaluations"][bank_name]["horizons"].items():
                compare_metric(arrays, "pred" if arm == ARMS[-1] else f"pred_{t}", met)
                if arm != ARMS[-1]:
                    assert np.array_equal(arrays[f"count_{t}"], banks[bank_name]["lengths"])
                    assert met["count_matches_length"]
                counts["final_metric_rows"] += 1
                counts["final_prediction_values"] += len(arrays["y"])
    assert counts == {"units": 5, "training_updates": 750, "checkpoint_bindings": 15,
                      "intermediate_evaluations": 15, "final_metric_rows": 35, "final_prediction_values": 4480}
    return {"status": "PASS", "method": "Saved arrays and checkpoint/source bindings only; no inference or optimizer updates", **counts}


def export():
    assert not PUBLIC.exists() and not PUBLICATION.exists(), "Fresh publication destination required"
    validation = verify_saved(RUN)
    receipt = read(ROOT / "runs/learnable_gcr_20261009_01_guard_79905cb797cc/launch_receipt.json")
    assert receipt["status"] == "COMPLETE" and receipt["worker_exit_code"] == 0
    assert receipt["idle_sleep_request_cleared"]
    manifest = read(RUN / "manifest.json")
    for name in ("RESULTS.md", "aggregate.json", "status.json", "manifest.json", "qualification.json"):
        copy(name)
    for path in (RUN / "banks").iterdir():
        copy(path.relative_to(RUN))
    for arm in ARMS:
        for path in (RUN / arm).iterdir():
            if path.is_file() and path.suffix in (".json", ".jsonl", ".npz"):
                copy(path.relative_to(RUN))
        for update in (50, 100, 150):
            checkpoint = RUN / arm / "checkpoints" / f"u{update:03d}.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            binding = {"update": update, "next_update": saved["next_update"], "formal_endpoint": update == 150,
                       "checkpoint_sha256": sha(checkpoint), "config": saved["config"],
                       "source_hashes": saved["source_hashes"], "weights_published": False}
            write(PUBLIC / arm / "checkpoints" / f"u{update:03d}.binding.json", binding)
    write(PUBLIC / "validation.json", validation)
    write(PUBLIC / "execution_provenance.json", {"execution": "COMPLETE", "worker_exit_code": 0,
        "protected_on_demand_job": True, "runtime_cap": None, "recurring": False,
        "idle_sleep_request_cleared": True, "failed_initial_dispatch": "sandbox task-scheduler access failure before training",
        "recovery": "authorized protected launch; no training restart or configuration change",
        "private_receipts_published": False})
    aggregate = read(PUBLIC / "aggregate.json")
    with (PUBLIC / "final_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["arm", "bank", "horizon", "n", "mse", "r2", "target_variance", "finite", "count_matches_length", "role"])
        writer.writeheader()
        for arm, result in aggregate["results"].items():
            for bank, evaluation in result["evaluations"].items():
                for horizon, met in evaluation["horizons"].items():
                    writer.writerow({"arm": arm, "bank": bank, "horizon": horizon, **met,
                                     "role": "offline diagnostic" if arm == ARMS[-1] else "primary arm"})
    (PUBLIC / "REPRODUCTION.md").write_text("""# Saved-data reproduction

Read RESULTS.md, aggregate.json and final_metrics.csv first. This is a new
ordered-path task, not Region Identity or historical StreamingCell. The frozen
verdict is POSITIVE_CONTROLS_UNQUALIFIED because Full Writer K64 did not reach
held-out R2 >= .5. GCR K64 qualified individually; neither that nor its K8 result
establishes improved short-credit learning. One block is not a reliability study.

From repository root (CPU NumPy and Torch are sufficient; no CUDA is used):

```powershell
python -X utf8 -B tools/export_learnable_gcr.py --verify-only
```

The package retains all750 training updates,15 checkpoint bindings and their
held-out evaluations, all35 final metric rows, all saved per-example predictions,
four exact token/path banks, the teacher specification and the shared schedule.
NPZ archives use ordinary numeric arrays and load with allow_pickle=False.
The diagnostic's predictions are horizon-independent, since it reads offline
raw8D sufficient statistics and performs no recurrent execution.

Weights, optimizer checkpoint contents and machine-specific receipts remain
local. Source hashes bind the exact executed files, which match new/learnable_gcr
and the existing protected launcher/worker in this repository. Export verification
does not train or run a model. Verification of omitted checkpoint contents is
represented by the checkpoint binding hashes and local pre-export validation.

For a new scientific run, follow new/learnable_gcr/PROTOCOL.md using fresh output
and qualification names. It requires the declared Torch2.5.1 CUDA environment.
Use run.py --check, then run.py --out with --qualification, or the protected
on-demand launcher. The sole primary endpoint is u150; u50/u100 are diagnostics.
""", encoding="utf-8")
    public_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(PUBLIC.rglob("*")) if p.is_file()}
    write(PUBLICATION, {"schema": "learnable-gcr-publication-v1", "source_sha256": manifest["source_hashes"],
        "published_artifacts_sha256": public_hashes, "validation": validation,
        "checkpoint_contents_published": False, "raw_run_preserved": True})


def verify_public():
    publication = read(PUBLICATION)
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(PUBLIC.rglob("*")) if p.is_file()}
    assert hashes == publication["published_artifacts_sha256"]
    validation = verify_saved(PUBLIC)
    assert validation == read(PUBLIC / "validation.json") == publication["validation"]
    for path in PUBLIC.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".md", ".csv"):
            assert not PRIVATE.search(path.read_text(encoding="utf-8")), path
    return {**validation, "files": len(hashes), "bytes": sum(p.stat().st_size for p in PUBLIC.rglob("*") if p.is_file())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        export()
    print(json.dumps(verify_public(), indent=2, allow_nan=False))
