"""Losslessly publish or CPU-verify the completed Hybrid Writer experiment."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from export_continuous_coverage import (
    bank_tensor_hash, close, compare_trace_metrics, copy_bound, decode_trace,
    exact_two_sided_p, gzip_json, joint_pass, load_bank, read, safe, sha,
    trace_metrics, write,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/hybrid_writer_20261006_01"
PUBLIC = ROOT / "evidence/hybrid_writer_20261006"
MANIFEST = ROOT / "HYBRID_WRITER_PUBLICATION_MANIFEST.json"
ARMS = ("neural", "budget", "hybrid", "affine_hybrid")
CHECKPOINTS = tuple(range(0, 301, 25))
SIZES = (32, 64)
PROTOCOL = "hybrid_lane_write_budget_v1"
HELPER = ROOT / "tools/export_continuous_coverage.py"


def load_module(relative):
    spec = importlib.util.spec_from_file_location("hybrid_public_" + Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_artifacts():
    for name in ("config.json", "dense.json", "metrics.csv", "perarm.json", "summary.json",
                 "RESULTS.md", "plans.json", "reach_retention_trajectory.png"):
        compressed = name == "plans.json"
        yield RUN / name, name + (".gz" if compressed else ""), compressed
    for name in ("train.npz", "evaluation32.npz", "evaluation64.npz"):
        yield RUN / "banks" / name, "banks/" + name, False
    for block in range(8):
        for arm in ARMS:
            folder = f"block{block:02d}/{arm}"
            yield RUN / folder / "training_curve.json", folder + "/training_curve.json.gz", True
            for update in CHECKPOINTS:
                stage = folder + f"/evaluation/u{update:03d}"
                for name in ("summary.json", "writer_telemetry.json"):
                    yield RUN / stage / name, stage + "/" + name + ".gz", True
                for size in SIZES:
                    name = f"size{size}_traces.npz"
                    yield RUN / stage / name, stage + "/" + name, False
                if update == 300:
                    name = "matched_frontier_strata.csv"
                    yield RUN / stage / name, stage + "/" + name + ".gz", True


def measure():
    """Recompute all checkpoint R/S metrics from saved data, without Torch."""
    summary, dense, final = (read(PUBLIC / name) for name in ("summary.json", "dense.json", "perarm.json"))
    assert summary["status"] == "COMPLETE" and summary["protocol"] == PROTOCOL
    assert summary["completed_arms"] == summary["expected_arms"] == 32
    assert summary["dense_records"] == len(dense) == 416 and len(final) == 32
    expected = {(b, a, u) for b in range(8) for a in ARMS for u in CHECKPOINTS}
    assert {(r["block"], r["arm"], r["update"]) for r in dense} == expected
    assert len(expected) == len(dense)
    metrics_module = load_module("new/continuous_coverage/metrics.py")
    reporting = load_module("new/hybrid_writer/reporting.py")
    banks = {size: load_bank(PUBLIC / f"banks/evaluation{size}.npz") for size in SIZES}
    trace_rows = {}
    for index, row in enumerate(dense):
        block, arm, update = row["block"], row["arm"], row["update"]
        folder = PUBLIC / f"block{block:02d}/{arm}/evaluation/u{update:03d}"
        saved = gzip_json(folder / "summary.json.gz")
        assert metrics_module.compact_pair_metrics(saved) == row["metrics"]
        assert metrics_module.joint_readiness(saved) == row["joint"]
        assert row["formal_endpoint"] == (update == 300) and not row["checkpoint_selection"]
        assert saved["full_evaluated"] == (update == 300)
        for size in SIZES:
            traces = decode_trace(folder / f"size{size}_traces.npz", size)
            assert np.array_equal(traces["correct"], traces["original_correct"] & traces["flipped_correct"])
            measured = trace_metrics(traces["correct"], banks[size])
            compare_trace_metrics(measured, row["metrics"]["sizes"][str(size)],
                                  f"block{block}/{arm}/u{update}/size{size}")
            trace_rows[(block, arm, update, size)] = measured
            if size == 32:
                assert joint_pass(measured) == row["joint"]["pass"]
        telemetry = gzip_json(folder / "writer_telemetry.json.gz")
        assert len(telemetry) == 112
        telemetry_keys = {(x["size"], x["world"], x["step"], x["lane"]) for x in telemetry}
        assert len(telemetry_keys) == 112
        safe(telemetry)
        if (index + 1) % 104 == 0:
            print(json.dumps({"saved_checkpoints_verified": index + 1, "total": 416}), flush=True)
    recomputed = reporting.aggregate(final, dense, expected_blocks=8)
    for key, value in recomputed.items():
        assert summary[key] == value, key
    readiness = {a: sum(joint_pass(trace_rows[(b, a, 300, 32)]) for b in range(8)) for a in ARMS}
    assert readiness == {a: summary["per_arm"][a]["formal_readiness_passes"] for a in ARMS}
    # Independently check paired counts and multiplicity, beyond the original reporter.
    primary = [("budget_minus_neural", "budget", "neural"),
               ("hybrid_minus_budget", "hybrid", "budget"),
               ("hybrid_minus_neural", "hybrid", "neural")]
    raw_p = {}
    for name, candidate, reference in primary:
        pairs = [(joint_pass(trace_rows[(b, candidate, 300, 32)]),
                  joint_pass(trace_rows[(b, reference, 300, 32)])) for b in range(8)]
        wins, losses = sum(a and not b for a, b in pairs), sum(b and not a for a, b in pairs)
        contrast = summary["comparisons"][name]
        assert (wins, losses, wins - losses) == (contrast["wins"], contrast["losses"], contrast["net_gain"])
        raw_p[name] = exact_two_sided_p(wins, losses)
        close(raw_p[name], contrast["exact_two_sided_p"], name)
    running = 0.0
    for index, (name, p) in enumerate(sorted(raw_p.items(), key=lambda x: x[1])):
        running = max(running, min(1.0, (3 - index) * p))
        close(running, summary["comparisons"][name]["holm_adjusted_p"], name + ".holm")
    for block in range(8):
        for arm in ARMS:
            curve = gzip_json(PUBLIC / f"block{block:02d}/{arm}/training_curve.json.gz")
            assert [r["update"] for r in curve] == list(range(1, 301))
            assert all(r["forward_steps"] == 256 and r["backward_calls"] == 32
                       and r["optimizer_steps"] == 1 and r["credit_horizon"] == 8 for r in curve)
    with (PUBLIC / "metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 832
    return {"status": "PASS", "scope": "Saved-data CPU checks; no model inference, training or optimizer updates.",
            "checkpoint_metric_records": 416, "packed_trace_banks": 832,
            "paired_boolean_AND_and_bitpack_roundtrip": 832,
            "training_curves": 32, "writer_telemetry_files": 416,
            "joint_readiness_at_u300": readiness,
            "primary_verdict": summary["primary_verdict"],
            "primary_aggregate_and_checkpoint_grid_match": True,
            "old_Full_and_frontier": "Saved u300 outputs retained; neither regenerated by this check."}


def verify_files(publication):
    safe(publication)
    actual = {p.relative_to(ROOT).as_posix(): sha(p) for p in PUBLIC.rglob("*") if p.is_file()}
    assert actual == publication["published_sha256"]
    assert sum((ROOT / p).stat().st_size for p in actual) == publication["published_bytes"]
    for relative, expected in publication["verification_source_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    for relative, expected in publication["source_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    for relative, binding in publication["raw_artifact_bindings"].items():
        path = ROOT / binding["public_path"]
        raw = gzip.decompress(path.read_bytes()) if binding["encoding"] == "gzip lossless" else path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == binding["sha256"], relative
        if relative.endswith(".json"):
            safe(json.loads(raw))
    for name in ("train", "evaluation32", "evaluation64"):
        assert bank_tensor_hash(load_bank(PUBLIC / f"banks/{name}.npz")) == publication["data_sha256"][name]
    plans_raw = gzip.decompress((PUBLIC / "plans.json.gz").read_bytes())
    assert hashlib.sha256(plans_raw).hexdigest() == publication["schedule_plan_sha256"]
    plans = json.loads(plans_raw)
    for block in range(8):
        plan = plans[str(block)]
        assert plan["initialization_seed"] == 110001 + block
        assert plan["schedule_seed"] == 111001 + block
        assert set(plan["initial_parameter_sha256"]) == set(ARMS)
        assert plan["arm_order"] == list(ARMS[block % 4:] + ARMS[:block % 4])
        batches = np.asarray(plan["batch_indices"])
        assert batches.shape == (300, 8) and batches.min() >= 0 and batches.max() < 512
    expected_counts = {"*.npz": 835, "size*_traces.npz": 832, "summary.json.gz": 416,
                       "writer_telemetry.json.gz": 416, "training_curve.json.gz": 32,
                       "matched_frontier_strata.csv.gz": 32}
    for pattern, count in expected_counts.items():
        assert len(list(PUBLIC.rglob(pattern))) == count, pattern
    assert len(publication["checkpoint_bindings"]) == 416
    assert not list(PUBLIC.rglob("*.pt"))


def build():
    assert not PUBLIC.exists() and not MANIFEST.exists(), "Refusing to overwrite evidence"
    manifest, status = read(RUN / "manifest.json"), read(RUN / "status.json")
    assert status["status"] == "COMPLETE" and manifest["protocol"] == PROTOCOL
    assert sha(ROOT / "analyses/hybrid_writer_qualification_20261006_02.json") == manifest["qualification_sha256"]
    for relative, expected in manifest["source_sha256"].items():
        assert sha(ROOT / relative) == sha(RUN / "source" / relative) == expected, relative
    assert sha(RUN / "plans.json") == manifest["schedule_plan_sha256"]
    checkpoints = {}
    for row in read(RUN / "dense.json"):
        assert sha(RUN / row["checkpoint"]) == row["checkpoint_sha256"]
        checkpoints[row["checkpoint"]] = {"file_sha256": row["checkpoint_sha256"],
                                           "parameter_sha256": row["parameter_sha256"]}
    bindings = {}
    for source, relative, compressed in raw_artifacts():
        if source.suffix == ".json":
            safe(read(source))
        copy_bound(source, PUBLIC / relative, bindings, source.relative_to(RUN).as_posix(),
                   "gzip lossless" if compressed else "byte-exact copy")
    final = read(PUBLIC / "perarm.json")
    with (PUBLIC / "final_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["block", "arm", "update", "joint_ready", "old_full_pass", "R_strict_pooled_T64",
                  "R_strict_mean_T64", "S_retention64_to256", "S_continuous_survival64_to256",
                  "retention_reference_pixels", "retention_reference_maps"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in final:
            values = {"block": row["block"], "arm": row["arm"], "update": row["update"],
                      "joint_ready": row["joint"]["pass"], "old_full_pass": row["metrics"]["full_pass"]}
            values.update({k: row["metrics"]["sizes"]["32"][k] for k in fields[5:]})
            writer.writerow(values)
    reproduction = """# Hybrid Writer result verification and reproduction

Start with [RESULTS.md](RESULTS.md), [summary.json](summary.json), and the32-row
[final_metrics.csv](final_metrics.csv). The independent unit is the paired
training block, n=8; u300 joint readiness is the formal endpoint. Ever-ready
on u0,u25,...,u300 and writer telemetry are secondary diagnostics.

From the repository root, run the CPU-only saved-data verification:

```powershell
python -X utf8 -B tools/export_hybrid_writer.py --verify-only
```

This checks hashes, three data banks, schedules, all832 packed Boolean trace
banks, paired original/flipped AND, and R/S/survival for all416 checkpoints.
The saved reporter aggregate, primary paired counts/Holm correction and
checkpoint grid are checked. No Torch import, model inference or training is
performed. Old Full and frontier are preserved evaluator results, not rebuilt.

The complete scientific output is included. Large raw summaries, writer
telemetry, training curves, schedules and frontier CSVs use lossless gzip.
NPZ banks are unchanged; bitpack order is little-endian. Each trace contains
correct, original_correct and flipped_correct shape[257,32,size,size], with
times0..256. The dense records refer to original local artifact names; use
the root publication manifest's raw_artifact_bindings to resolve gzip paths.
Every checkpoint file SHA was checked locally and its parameter hash retained;
checkpoint/optimizer contents, launch receipts, private manifests and logs are
excluded. Prelaunch validation is in the separate historical evidence package.

The frozen protocol and training commands are in
[PROTOCOL.md](../../new/hybrid_writer/PROTOCOL.md). Training needs the qualified
PyTorch2.5.1 CUDA environment and a GPU. This export is self-contained for
saved-data checks with Python and NumPy and does not need local runs/checkpoints.
The plot and dense table are formation diagnostics; no peak selection replaces
the frozen u300 result. No general semantic-closure or recurrent-stability claim
is made by this recipe.
"""
    (PUBLIC / "REPRODUCTION.md").write_text(reproduction, encoding="utf-8")
    validation = measure()
    validation.update(source_snapshot_current_equality_files=len(manifest["source_sha256"]),
                      locally_verified_checkpoint_file_hashes=416, published_checkpoint_contents=False,
                      original_evidence_changed=False)
    write(PUBLIC / "validation.json", validation)
    published = {p.relative_to(ROOT).as_posix(): sha(p) for p in PUBLIC.rglob("*") if p.is_file()}
    publication = {"protocol": "hybrid_writer_saved_data_publication_v1", "run_id": RUN.name,
                   "experiment_protocol": PROTOCOL, "code_review_base": manifest["git_review_base"],
                   "result_review_base": "283ebe5ae584c53e26f588d8230808349a66472d",
                   "source_sha256": manifest["source_sha256"], "source_binding_status": "PASS",
                   "verification_source_sha256": {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__)),
                                                   HELPER.relative_to(ROOT).as_posix(): sha(HELPER)},
                   "data_sha256": manifest["data_sha256"], "schedule_plan_sha256": manifest["schedule_plan_sha256"],
                   "qualification_sha256": manifest["qualification_sha256"],
                   "checkpoint_bindings": checkpoints, "raw_artifact_bindings": bindings,
                   "published_sha256": published, "published_file_count": len(published),
                   "published_bytes": sum((ROOT / name).stat().st_size for name in published),
                   "excluded": ["checkpoint and optimizer contents", "machine manifest, status/PID and launch receipts",
                                "logs and absolute paths", "snapshot copies (original source hashes retained)"]}
    write(MANIFEST, publication)
    verify_files(publication)
    print(json.dumps({"status": "PASS", "files": len(published),
                      "MiB": round(publication["published_bytes"] / 2**20, 2), **validation}), flush=True)


def verify():
    publication = read(MANIFEST)
    verify_files(publication)
    measured = measure()
    validation = read(PUBLIC / "validation.json")
    assert all(validation[k] == v for k, v in measured.items())
    print(json.dumps({"status": "PASS", "files": publication["published_file_count"], **measured}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify() if args.verify_only else build()
