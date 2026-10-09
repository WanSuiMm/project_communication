"""Verify and publish the saved spatial nonlinear lift experiment."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/spatial_nonlinear_lift_20261009_01"
GUARD_RECEIPT = ROOT / "runs/spatial_nonlinear_lift_20261009_01_guard_3f2b84896f05/launch_receipt.json"
PUBLIC = ROOT / "evidence/spatial_nonlinear_lift_20261009_01"
PUBLICATION = ROOT / "SPATIAL_NONLINEAR_LIFT_PUBLICATION_MANIFEST.json"
RUNNER_PATH = ROOT / "new/spatial_nonlinear_lift/run.py"
SOURCE_ROOT = RUNNER_PATH.parent
BLOCKS = range(3)
UPDATES = (0, 100, 200, 300)
STAGES = (100, 200, 300)
FINAL_BANKS = ("heldout", "long128", "long256", "spatial16")
SOURCE_NAMES = (
    "new/spatial_nonlinear_lift/run.py",
    "new/spatial_nonlinear_lift/cells.py",
    "new/spatial_nonlinear_lift/tasks.py",
    "new/spatial_nonlinear_lift/THEORY.md",
    "new/spatial_nonlinear_lift/PROTOCOL.md",
    "tools/start_protected_job.ps1",
    "tools/protected_job_worker.ps1",
)


def load_runner():
    # Load this experiment's runner first so its bare cells/tasks imports resolve
    # inside its own source directory, without importing an older exporter.
    sys.path.insert(0, str(SOURCE_ROOT))
    spec = importlib.util.spec_from_file_location("_spatial_nonlinear_lift_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load experiment runner: {RUNNER_PATH}")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


R = load_runner()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_normalize(value):
    return json.loads(json.dumps(value, allow_nan=False))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parameter_hash(state) -> str:
    digest = hashlib.sha256()
    for name in ("a", "b", "gamma"):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def finite_metric(value, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise AssertionError(f"Nonfinite {name}")
    return number


def compute_metrics(y, prediction):
    y64 = np.asarray(y, dtype=np.float64)
    p64 = np.asarray(prediction, dtype=np.float64)
    assert y64.shape == p64.shape and y64.ndim == 3
    assert np.isfinite(y64).all() and np.isfinite(p64).all()
    residual = np.square(y64 - p64)
    mse = float(residual.mean())
    target_variance = float(np.square(y64 - y64.mean()).mean())
    assert target_variance > 1e-12
    per_map_mse = residual.mean(axis=(1, 2))
    return {
        "mse": mse,
        "r2": 1.0 - mse / target_variance,
        "target_variance": target_variance,
        "per_map_mse": per_map_mse,
        "n_maps": int(y64.shape[0]),
    }


def compare_saved_metrics(actual, saved, where: str) -> None:
    computed = compute_metrics(actual["y"], actual["prediction"])
    for name in ("mse", "r2", "target_variance"):
        np.testing.assert_allclose(
            computed[name], finite_metric(saved[name], f"{where}.{name}"),
            rtol=1e-12, atol=1e-15,
            err_msg=f"Saved {where}.{name} does not match saved arrays",
        )
    saved_per_map = np.asarray(saved["per_map_mse"], dtype=np.float64)
    assert saved_per_map.shape == computed["per_map_mse"].shape
    np.testing.assert_allclose(
        computed["per_map_mse"], saved_per_map, rtol=1e-12, atol=1e-15,
        err_msg=f"Saved {where}.per_map_mse does not match saved arrays",
    )
    return computed


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as value:
        return {name: value[name] for name in value.files}


def expected_sources():
    value = R.source_hashes()
    assert set(value) == set(SOURCE_NAMES), "The executed source set changed"
    return value


def check_sources(folder: Path, manifest, snapshots: bool) -> None:
    source_hashes = manifest["source_hashes"]
    assert source_hashes == expected_sources(), "Current experiment source hashes changed"
    for rel, digest in source_hashes.items():
        current = ROOT / Path(rel)
        assert sha(current) == digest, f"Current source hash mismatch: {rel}"
        snapshot = folder / "source" / Path(rel)
        if snapshots:
            assert snapshot.is_file() and sha(snapshot) == digest, f"Source snapshot mismatch: {rel}"


def check_curve(path: Path) -> int:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 300, f"Expected 300 records in {path.name}, got {len(records)}"
    updates = [int(row["update"]) for row in records]
    assert updates == list(range(1, 301)), f"Updates are missing, duplicated, or reordered in {path.name}"
    assert len(set(updates)) == 300
    for row in records:
        finite_metric(row["loss"], f"{path.name}.loss")
        finite_metric(row["gradient_norm_before_clip"], f"{path.name}.gradient_norm_before_clip")
    return len(records)


def validate_checkpoint(checkpoint: Path, config, source_hashes, block: int, arm: str, update: int):
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert value["block"] == block and value["arm"] == arm
    assert value["next_update"] == update + 1
    assert json_normalize(value["config"]) == config
    assert value["source_hashes"] == source_hashes
    state = value["model"]
    assert set(state) == {"a", "b", "gamma"}
    parameters = {}
    for name in ("a", "b", "gamma"):
        tensor = state[name]
        assert torch.is_tensor(tensor) and tuple(tensor.shape) == (3,)
        assert tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
        parameters[name] = {"shape": [3], "dtype": str(tensor.dtype).removeprefix("torch."), "finite": True}
    return {
        "block": block,
        "arm": arm,
        "update": update,
        "next_update": update + 1,
        "checkpoint_sha256": sha(checkpoint),
        "model_parameter_sha256": parameter_hash(state),
        "initial_parameter_sha256": parameter_hash(state) if update == 0 else None,
        "model_parameters": parameters,
        "config": config,
        "source_hashes": source_hashes,
    }


def validate_public_binding(path: Path, config, source_hashes, block: int, arm: str, update: int):
    value = read_json(path)
    assert value["block"] == block and value["arm"] == arm
    assert value["update"] == update and value["next_update"] == update + 1
    assert value["config"] == config and value["source_hashes"] == source_hashes
    assert re.fullmatch(r"[0-9a-f]{64}", value["checkpoint_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", value["model_parameter_sha256"])
    assert value["weights_published"] is False and value["optimizer_state_published"] is False
    assert value["model_parameters"] == {
        name: {"shape": [3], "dtype": "float32", "finite": True}
        for name in ("a", "b", "gamma")
    }
    initial = value["initial_parameter_sha256"]
    if update == 0:
        assert initial == value["model_parameter_sha256"]
    else:
        assert initial is None
    return value


def comparable_result(value):
    result = copy.deepcopy(value)
    result.pop("learned_parameter_contents_omitted", None)
    return result


def verify_saved(folder: Path, snapshots: bool = False):
    aggregate = read_json(folder / "aggregate.json")
    manifest = read_json(folder / "manifest.json")
    status = read_json(folder / "status.json")
    qualification = read_json(folder / "qualification.json")
    config = json_normalize(R.CONFIG)
    assert status["status"] == aggregate["status"] == "COMPLETE"
    assert status["completed_units"] == status["total_units"] == 9
    assert aggregate["completed_units"] == aggregate["planned_units"] == 9
    assert aggregate["stopped_at_predeclared_control"] is False
    assert aggregate["decision"] == status["decision"] == R.decision(aggregate["arms"])
    assert manifest["config"] == aggregate["config"] == qualification["config"] == config
    assert qualification["status"] == "PASS"
    assert qualification["source_hashes"] == manifest["source_hashes"]
    assert sha(folder / "qualification.json") == manifest["qualification_sha256"]
    check_sources(folder, manifest, snapshots)

    data_hashes = manifest["data_hashes"]
    required_data = {
        *(f"banks/{name}.npz" for name in config["banks"]),
        *(f"banks/schedule_block{block:02d}.npy" for block in BLOCKS),
        "banks/teacher.json",
    }
    assert set(data_hashes) == required_data
    for rel, digest in data_hashes.items():
        assert sha(folder / Path(rel)) == digest, f"Data hash mismatch: {rel}"
    assert read_json(folder / "banks/teacher.json") == config["teacher"]

    schedule_rebuilds = 0
    for block, seed in enumerate(config["schedule_seeds"]):
        path = folder / f"banks/schedule_block{block:02d}.npy"
        saved_schedule = np.load(path, allow_pickle=False)
        expected = R.make_schedule(
            config["updates"], config["batch_size"], config["banks"]["train"][0], seed
        )
        assert saved_schedule.dtype == expected.dtype and np.array_equal(saved_schedule, expected)
        schedule_rebuilds += 1

    bank_targets = {}
    for name in config["banks"]:
        arrays = load_npz(folder / f"banks/{name}.npz")
        assert "e" in arrays and "y" in arrays
        assert arrays["e"].shape[0] == arrays["y"].shape[0]
        assert np.isfinite(arrays["e"]).all() and np.isfinite(arrays["y"]).all()
        bank_targets[name] = arrays["y"]

    expected_pairs = {(block, spec[0]) for block in BLOCKS for spec in config["arms"]}
    aggregate_arms = {(int(row["block"]), row["arm"]): row for row in aggregate["arms"]}
    assert set(aggregate_arms) == expected_pairs and len(aggregate_arms) == 9
    results_by_pair = {}
    for block, arm in sorted(expected_pairs):
        result_path = folder / f"block{block:02d}/{arm}/result.json"
        result = read_json(result_path)
        assert (int(result["block"]), result["arm"]) == (block, arm)
        assert comparable_result(result) == comparable_result(aggregate_arms[(block, arm)])
        omitted = result.get("learned_parameter_contents_omitted", False)
        if omitted:
            assert "final_parameters" not in result
        else:
            assert "final_parameters" in result
        results_by_pair[(block, arm)] = result

    training_records = 0
    final_rows = []
    stage_rows = 0
    prediction_values = 0
    initial_hashes = {}
    checkpoint_bindings = 0
    public_package = (folder / "block00").is_dir() and not (folder / "block00/original_k64/checkpoints/u000.pt").exists()

    for block, arm in sorted(expected_pairs):
        arm_dir = folder / f"block{block:02d}/{arm}"
        result = results_by_pair[(block, arm)]
        training_records += check_curve(arm_dir / "training.jsonl")
        assert set(result["evaluations"]) == set(FINAL_BANKS)

        for bank in FINAL_BANKS:
            saved = result["evaluations"][bank]["metrics"]
            prediction_file = arm_dir / f"{bank}_predictions.npz"
            arrays = load_npz(prediction_file)
            assert set(arrays) == {"y", "prediction"}
            assert np.array_equal(arrays["y"], bank_targets[bank])
            computed = compare_saved_metrics(arrays, saved, f"{block}/{arm}/{bank}")
            prediction_values += int(arrays["prediction"].size)
            final_rows.append({
                "block": block,
                "arm": arm,
                "bank": bank,
                "horizon": int(config["banks"][bank][2]),
                "n_maps": computed["n_maps"],
                "mse": computed["mse"],
                "r2": computed["r2"],
                "target_variance": computed["target_variance"],
                "per_map_mse": computed["per_map_mse"].tolist(),
            })

        heldout_final = load_npz(arm_dir / "heldout_predictions.npz")
        for update in STAGES:
            stage_arrays = load_npz(arm_dir / f"heldout_u{update:03d}.npz")
            stage_json = read_json(arm_dir / f"heldout_u{update:03d}.json")
            assert set(stage_arrays) == {"y", "prediction"}
            assert np.array_equal(stage_arrays["y"], bank_targets["heldout"])
            compare_saved_metrics(stage_arrays, stage_json["metrics"], f"{block}/{arm}/u{update}")
            stage_rows += 1
        stage_300 = load_npz(arm_dir / "heldout_u300.npz")
        assert np.array_equal(stage_300["y"], heldout_final["y"])
        assert np.array_equal(stage_300["prediction"], heldout_final["prediction"])

        if not result.get("learned_parameter_contents_omitted", False):
            final_checkpoint = torch.load(arm_dir / "checkpoints/u300.pt", map_location="cpu", weights_only=False)
            final_state = final_checkpoint["model"]
            for name in ("a", "b", "gamma"):
                saved_parameter = np.asarray(result["final_parameters"][name], dtype=np.float32)
                assert np.array_equal(saved_parameter, final_state[name].detach().cpu().numpy())

        block_initials = []
        for update in UPDATES:
            if public_package:
                binding_path = arm_dir / f"checkpoints/u{update:03d}.binding.json"
                binding = validate_public_binding(
                    binding_path, config, manifest["source_hashes"], block, arm, update
                )
            else:
                checkpoint = arm_dir / f"checkpoints/u{update:03d}.pt"
                binding = validate_checkpoint(
                    checkpoint, config, manifest["source_hashes"], block, arm, update
                )
            if update == 0:
                block_initials.append(binding["initial_parameter_sha256"])
            checkpoint_bindings += 1
        initial_hashes.setdefault(str(block), []).extend(block_initials)

    assert training_records == 2700
    assert len(final_rows) == 36 and stage_rows == 27
    assert checkpoint_bindings == 36 and schedule_rebuilds == 3
    assert all(len(values) == 3 and len(set(values)) == 1 for values in initial_hashes.values())
    assert set(initial_hashes) == {"0", "1", "2"}
    initial_hashes = {block: values[0] for block, values in initial_hashes.items()}

    return {
        "status": "PASS",
        "method": "Saved arrays, logs, hashes, and checkpoint bindings only; no training or model inference",
        "training_records": training_records,
        "final_metric_rows": len(final_rows),
        "stage_metric_rows": stage_rows,
        "prediction_values": prediction_values,
        "checkpoint_bindings": checkpoint_bindings,
        "source_bindings": len(manifest["source_hashes"]),
        "data_bindings": len(data_hashes),
        "schedule_rebuilds": schedule_rebuilds,
        "initial_parameter_sha256_by_block": initial_hashes,
        "_final_rows": final_rows,
    }


def write_final_metrics(path: Path, rows) -> None:
    fields = ("block", "arm", "bank", "horizon", "n_maps", "mse", "r2", "target_variance", "per_map_mse")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "per_map_mse": json.dumps(row["per_map_mse"], separators=(",", ":"))})


def systems_rows(aggregate):
    rows = []
    for result in aggregate["arms"]:
        systems = result["systems"]
        rows.append({
            "block": int(result["block"]),
            "arm": result["arm"],
            "parameters": int(systems["parameters"]),
            "state_scalars_per_cell": int(systems["state_scalars_per_cell"]),
            "resumed_from_update": int(systems["resumed_from_update"]),
            "training_and_intermediate_eval_seconds": finite_metric(
                systems["training_and_intermediate_eval_seconds"], "systems_seconds"
            ),
            "peak_allocated_mib_including_resident_banks": finite_metric(
                systems["peak_allocated_mib"], "peak_allocated_mib"
            ),
            "incremental_peak_mib_including_resident_banks": finite_metric(
                systems["incremental_peak_mib"], "incremental_peak_mib"
            ),
        })
    assert len(rows) == 9
    return rows


def write_systems(path: Path, rows) -> None:
    fields = tuple(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sanitized_result(value):
    result = copy.deepcopy(value)
    result.pop("final_parameters", None)
    result["learned_parameter_contents_omitted"] = True
    return result


def copy_local_evidence(validation) -> None:
    assert not PUBLIC.exists() and not PUBLICATION.exists(), "Fresh publication destination required"
    PUBLIC.mkdir(parents=True)

    aggregate = read_json(RUN / "aggregate.json")
    public_aggregate = copy.deepcopy(aggregate)
    public_aggregate["arms"] = [sanitized_result(row) for row in aggregate["arms"]]
    public_aggregate["learned_parameter_contents_omitted"] = True
    write_json(PUBLIC / "aggregate.json", public_aggregate)

    for name in ("RESULTS.md", "manifest.json", "qualification.json", "status.json"):
        shutil.copyfile(RUN / name, PUBLIC / name)
    validation_for_file = {key: value for key, value in validation.items() if key != "_final_rows"}
    write_json(PUBLIC / "validation.json", validation_for_file)

    shutil.copytree(RUN / "banks", PUBLIC / "banks")
    config = json_normalize(R.CONFIG)
    source_hashes = read_json(RUN / "manifest.json")["source_hashes"]
    for block in BLOCKS:
        for spec in config["arms"]:
            arm = spec[0]
            source_dir = RUN / f"block{block:02d}/{arm}"
            public_dir = PUBLIC / f"block{block:02d}/{arm}"
            public_dir.mkdir(parents=True)
            for path in sorted(source_dir.iterdir()):
                if path.is_file() and path.suffix in {".json", ".jsonl", ".npz"}:
                    if path.name == "result.json":
                        write_json(public_dir / path.name, sanitized_result(read_json(path)))
                    else:
                        shutil.copyfile(path, public_dir / path.name)
            (public_dir / "checkpoints").mkdir()
            for update in UPDATES:
                local_checkpoint = source_dir / f"checkpoints/u{update:03d}.pt"
                binding = validate_checkpoint(local_checkpoint, config, source_hashes, block, arm, update)
                binding["weights_published"] = False
                binding["optimizer_state_published"] = False
                binding["checkpoint_contents_published"] = False
                write_json(public_dir / f"checkpoints/u{update:03d}.binding.json", binding)

    final_rows = validation["_final_rows"]
    write_final_metrics(PUBLIC / "final_metrics.csv", final_rows)
    write_systems(PUBLIC / "systems.csv", systems_rows(aggregate))
    (PUBLIC / "REPRODUCTION.md").write_text(
        """# Reproduce and review this result

Start with `RESULTS.md`, `aggregate.json`, `final_metrics.csv`, and
`systems.csv`. The aggregate is the decision summary; the nine per-arm
`result.json` files contain the same final metrics and system records. The
complete saved curves and predictions are retained beside them. Public result
copies omit learned a/b/gamma parameter values; checkpoint binding JSON files
publish only hashes, parameter shapes, and update metadata.

From the repository root, verify the exported arrays and artifact hashes on
CPU without training or model inference:

```powershell
python -X utf8 -B tools/export_spatial_nonlinear_lift.py --verify-only
```

The verifier checks all 2,700 finite loss records, all 36 final metric rows
including per-map MSE, all 27 heldout stage evaluations, all 36 checkpoint
bindings, the three regenerated CPU minibatch schedules, and current source
hashes. Predictions and targets are loaded from the saved NPZ files as
float64 for metric recomputation.

The original run reached 300 updates for each of three arms in each of three
matched initialization/schedule blocks. It has 42 versus 2 state scalars per
cell and nine trainable parameters per arm. The result is a three-block
developmental screen on a matched finite-degree teacher task. Long-horizon
rows are separately generated finite episodes; they do not establish
autonomous continuation or stability. Systems timings cover training plus
intermediate evaluation, and reported peak memory includes resident banks;
neither is a standalone kernel benchmark.

The public source and protocol are sufficient to rerun the qualification and
experiment without an older reference run. Any new training run must use a
fresh output directory and the declared CUDA environment; it does not belong
in this saved-array verification step.
""",
        encoding="utf-8",
    )


def public_text_files():
    suffixes = {".json", ".jsonl", ".md", ".csv", ".txt"}
    return [path for path in PUBLIC.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]


def scan_private_text(extra_paths=()) -> None:
    private = re.compile(
        r"(?:\b[A-Z]:[\\/]|\\\\|/(?:home|data/users)/|"
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\bhttps?://|\bfile://)", re.IGNORECASE
    )
    for path in [*public_text_files(), *extra_paths]:
        text = path.read_text(encoding="utf-8")
        match = private.search(text)
        assert match is None, f"Private path, host identity, IP, username, or URL in {path.name}"


def artifact_hashes():
    return {
        path.relative_to(ROOT).as_posix(): sha(path)
        for path in sorted(PUBLIC.rglob("*")) if path.is_file()
    }


def local_source_provenance():
    paths = [RUN / "aggregate.json", RUN / "manifest.json", RUN / "qualification.json"]
    paths += [RUN / f"block{block:02d}/{arm}/result.json" for block in BLOCKS for arm, *_ in json_normalize(R.CONFIG)["arms"]]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def verify_public():
    assert PUBLIC.is_dir() and PUBLICATION.is_file(), "Publication package or root manifest is missing"
    outer = read_json(PUBLICATION)
    current_exporter_hash = sha(Path(__file__).resolve())
    assert outer["exporter_sha256"] == current_exporter_hash
    assert outer["executed_source_hashes"] == expected_sources()
    actual_artifacts = artifact_hashes()
    assert actual_artifacts == outer["published_artifacts_sha256"], "Published artifact hash mismatch"

    validation = verify_saved(PUBLIC, snapshots=False)
    validation_for_file = {key: value for key, value in validation.items() if key != "_final_rows"}
    assert validation_for_file == read_json(PUBLIC / "validation.json") == outer["validation"]
    aggregate = read_json(PUBLIC / "aggregate.json")
    final_rows = validation["_final_rows"]
    expected_csv = PUBLIC / "final_metrics.csv"
    actual_csv_rows = list(csv.DictReader(expected_csv.open("r", encoding="utf-8", newline="")))
    assert len(actual_csv_rows) == 36
    for actual, expected in zip(actual_csv_rows, final_rows):
        assert (int(actual["block"]), actual["arm"], actual["bank"], int(actual["horizon"]), int(actual["n_maps"])) == (
            expected["block"], expected["arm"], expected["bank"], expected["horizon"], expected["n_maps"]
        )
        for name in ("mse", "r2", "target_variance"):
            np.testing.assert_allclose(float(actual[name]), expected[name], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(
            np.asarray(json.loads(actual["per_map_mse"]), dtype=np.float64),
            expected["per_map_mse"], rtol=1e-12, atol=1e-15,
        )
    assert len(list(csv.DictReader((PUBLIC / "systems.csv").open("r", encoding="utf-8", newline="")))) == 9
    scan_private_text([PUBLICATION])
    return {
        **{key: value for key, value in validation.items() if key != "_final_rows"},
        "published_files": len(actual_artifacts),
        "published_bytes": sum(path.stat().st_size for path in PUBLIC.rglob("*") if path.is_file()),
    }


def export():
    assert not PUBLIC.exists() and not PUBLICATION.exists(), "Fresh publication destination required"
    validation = verify_saved(RUN, snapshots=True)
    validation_for_file = {key: value for key, value in validation.items() if key != "_final_rows"}
    receipt = read_json(GUARD_RECEIPT)
    assert receipt["status"] == "COMPLETE"
    assert receipt["worker_exit_code"] == 0
    assert receipt["idle_sleep_request_cleared"] is True

    copy_local_evidence(validation)
    # The local verifier's normalized result rows are returned internally only;
    # recompute the 36 CSV rows from the copied public arrays before hashing.
    public_validation = verify_saved(PUBLIC, snapshots=False)
    assert {key: value for key, value in public_validation.items() if key != "_final_rows"} == validation_for_file
    write_final_metrics(PUBLIC / "final_metrics.csv", public_validation["_final_rows"])
    write_systems(PUBLIC / "systems.csv", systems_rows(read_json(PUBLIC / "aggregate.json")))
    scan_private_text()

    published = artifact_hashes()
    write_json(PUBLICATION, {
        "schema": "spatial-nonlinear-lift-publication-v1",
        "run_id": "spatial_nonlinear_lift_20261009_01",
        "execution": {
            "status": receipt["status"],
            "worker_exit_code": receipt["worker_exit_code"],
            "idle_sleep_request_cleared": receipt["idle_sleep_request_cleared"],
            "private_launch_receipt_published": False,
        },
        "executed_source_hashes": read_json(RUN / "manifest.json")["source_hashes"],
        "exporter_sha256": sha(Path(__file__).resolve()),
        "local_result_source_sha256": local_source_provenance(),
        "published_artifacts_sha256": published,
        "published_artifact_count": len(published),
        "learned_parameter_contents_omitted": True,
        "checkpoint_contents_published": False,
        "raw_run_preserved": True,
        "validation": validation_for_file,
    })
    return verify_public()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_public() if args.verify_only else export()
    print(json.dumps(result, indent=2, allow_nan=False))
