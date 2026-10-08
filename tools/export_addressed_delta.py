"""Publish and verify the completed addressed-delta screen from saved data only."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/addressed_delta_20261008_02"
PARENT = ROOT / "runs/addressed_delta_20261007_01"
QUALIFICATION = ROOT / "analyses/addressed_delta_qualification_20261007_01.json"
PUBLIC = ROOT / "evidence/addressed_delta_20261008_02"
MANIFEST = ROOT / "ADDRESSED_DELTA_PUBLICATION_MANIFEST.json"
UPSTREAM_BANKS = ROOT / "evidence/port_relation_20261007_02/banks"
UPSTREAM_MANIFEST = ROOT / "PORT_RELATION_PUBLICATION_MANIFEST.json"
PROTOCOL = "addressed_delta_native_k8_development_v1"
ARMS = ("current", "additive", "delta")
BLOCKS = tuple(range(4))
CHECKPOINTS = tuple(range(0, 301, 25))
PRIVATE_KEYS = {"pid", "host", "hostname", "username", "gpu_uuid", "executable",
                "command", "launch_command", "cwd"}
PRIVATE_TEXT = re.compile(
    r"\b[A-Za-z]:[\\/]|\\\\[^\\\s]+[\\/]|(?<![\w.])/(?:[^/\s]+/)+[^/\s]+|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b|BEGIN [^\n]*PRIVATE KEY|"
    r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|"
    r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}"
)
MAX_FILE_BYTES = 90 * 1024 * 1024


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha_bytes(raw)


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON value: {value}")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)


def read_json_bytes(raw: bytes):
    return json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def safe(value) -> None:
    if isinstance(value, dict):
        assert not ({str(key).lower() for key in value} & PRIVATE_KEYS), "private metadata key"
        for item in value.values():
            safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            safe(item)
    elif isinstance(value, str):
        assert not PRIVATE_TEXT.search(value), "private text detected"
    elif isinstance(value, float):
        assert math.isfinite(value), "non-finite number"


def safe_relative(value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or not posix.parts or any(part in ("", ".", "..") for part in posix.parts):
        raise ValueError(f"unsafe repository-relative path: {value!r}")
    target = ROOT.joinpath(*posix.parts).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError(f"path escapes repository: {value!r}")
    return target


def module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, safe_relative(relative))
    if spec is None or spec.loader is None:
        raise ImportError(relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def bank_tensor_hash(bank: dict[str, np.ndarray]) -> str:
    """Match the frozen torch tensor hash using only CPU NumPy arrays."""
    result = hashlib.sha256()
    for key, value in sorted(bank.items()):
        array = np.ascontiguousarray(value)
        result.update(key.encode())
        result.update(str((array.shape, array.dtype)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def load_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def public_summary(raw_summary: dict) -> tuple[dict, list[str]]:
    removed: set[str] = set()

    def strip(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if str(key).lower() in PRIVATE_KEYS:
                    removed.add(str(key))
                else:
                    result[key] = strip(item)
            return result
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return strip(raw_summary), sorted(removed)


def qualification_public(raw: dict, qualification_sha256: str) -> dict:
    # Keep the original qualification's list structure while retaining its
    # actual-shape and gradient-equivalence evidence; omit timing/allocator data.
    arm_fields = (
        "arm", "status", "parameter_count",
        "three_update_gradient_max_absolute_error", "key_value_bootstrap_active_by_update3",
    )
    arms = [{key: row[key] for key in arm_fields} for row in raw["arms"]]
    result = {
        "schema": "addressed-delta-public-qualification-v1",
        "status": raw["status"],
        "protocol": raw["protocol"],
        "qualification_sha256": qualification_sha256,
        "source_sha256": dict(sorted(raw["source_sha256"].items())),
        "source_file_count": len(raw["source_sha256"]),
        "arms": arms,
        "actual_training_shape": raw["actual_training_shape"],
        "qualification_optimizer_updates_each_arm": raw["qualification_optimizer_updates_each_arm"],
        "qualification_updates_are_not_formal_training": raw["qualification_updates_are_not_formal_training"],
        "runtime_limit_enforced": raw["runtime_limit_enforced"],
        "source_binding_method": "The qualification bound 71 source files by individual SHA-256 entries; the protocol's integrated source ZIP was not created.",
        "sanitization": "Retains qualification status, per-file source bindings, actual shape, arm checks, and gradient equality; removes timing and allocator telemetry.",
    }
    safe(result)
    return result


def interruption_public(run_manifest: dict, parent_manifest_path: Path, parent_status: dict) -> dict:
    recovery = run_manifest["recovery_compatibility"]
    result = {
        "schema": "addressed-delta-interruption-recovery-v1",
        "protocol": PROTOCOL,
        "parent": {
            "run_id": Path(run_manifest["parent_run"]).name,
            "status": parent_status["status"],
            "parent_manifest_sha256": sha(parent_manifest_path),
            "completed_arms": parent_status["completed_arms"],
            "expected_arms": parent_status["expected_arms"],
            "dense_records": parent_status["dense_records"],
            "expected_dense_records": parent_status["expected_dense_records"],
            "last_observed_arm": parent_status["arm"],
            "last_completed_update": parent_status["completed_updates"],
            "stop_reason": "USER_CANCELLED",
        },
        "recovered_run": {
            "run_id": RUN.name,
            "inherited_committed_stages": recovery["inherited_committed_stages"],
            "usable_prefix": recovery["usable_prefix"],
            "recovery_entrypoint": recovery["entrypoint"],
            "recovery_entrypoint_sha256": recovery["entrypoint_sha256"],
            "configuration_compatibility_fix": recovery["fix"],
            "numerical_training_code_changed": recovery["numerical_training_code_changed"],
        },
        "initial_recovery_check": {
            "status": "FAILED_BEFORE_TRAINING",
            "cause": "JSON tuple/list configuration representation mismatch",
            "resolution": "Compare JSON-canonical configuration values in tools/recover_addressed_delta.py",
        },
        "scope": "The parent cancellation and resume are execution provenance, not scientific endpoints; recovery did not change the numerical training code.",
    }
    assert result["parent"]["status"] == "CANCELLED"
    assert result["parent"]["parent_manifest_sha256"] == run_manifest["parent_manifest_sha256"]
    assert result["recovered_run"]["recovery_entrypoint_sha256"] == recovery["entrypoint_sha256"]
    assert result["recovered_run"]["numerical_training_code_changed"] is False
    safe(result)
    return result


def reproduction_text() -> str:
    return """# Addressed-delta development screen: reproduction and evidence map

Read `RESULTS.md`, `summary.json`, and `final_metrics.csv` first. This is a
four-block development screen: delta and additive each passed joint readiness
in 0/4 blocks; current passed in 2/4. The verdict is
`NO_DEVELOPMENTAL_SIGNAL`. Four paired blocks are descriptive evidence, not a
reliability or significance result. Old Full was not evaluated, and full
Boolean arrays were transient and are not persisted.

From the repository root, the CPU-only saved-data check is:

```powershell
python -X utf8 -B tools/export_addressed_delta.py --verify-only
```

It checks published hashes, the 71 qualified source bindings, frozen bank and
schedule hashes, all 12 final and 156 dense records, record-to-summary gates,
reporter aggregate and CSV equality, all 156 checkpoint completion markers,
and the 12 training curves' 300-update cadence. It performs no inference,
training, or optimizer updates and does not read local runs or checkpoints.

Fresh training requires the existing published bank dependency at
`evidence/port_relation_20261007_02/banks/` and
`PORT_RELATION_PUBLICATION_MANIFEST.json`, plus NumPy and the qualified
CUDA-enabled PyTorch environment. From the repository root:

```powershell
python -X utf8 -B new/addressed_delta/check_cells.py
python -X utf8 -B new/addressed_delta/reporting.py --self-test
python -X utf8 -u -B new/addressed_delta/run.py --check --out analyses/addressed_delta_fresh_qualification.json
python -X utf8 -u -B new/addressed_delta/run.py --out runs/addressed_delta_fresh --qualification analyses/addressed_delta_fresh_qualification.json
```

Use a new run name for each independent reproduction. The check command runs
the small actual-shape CUDA qualification; the final command starts a new
four-block training run. The publication includes the exact source banks but
does not include checkpoints or optimizer state. `interruption_and_recovery.json`
records that the published run inherited 22 committed stages after the parent
was cancelled and that the compatibility runner fixed JSON tuple/list config
comparison without changing numerical training code.
"""


def json_from_zip(archive: zipfile.ZipFile, name: str):
    return read_json_bytes(archive.read(name))


def stage_grid(final_rows: list[dict], dense_rows: list[dict]) -> None:
    reporting = module("new/addressed_delta/reporting.py", "_addressed_delta_public_reporting")
    assert len(final_rows) == 12 and len(dense_rows) == 156
    final_keys = [(row.get("block"), row.get("arm")) for row in final_rows]
    dense_keys = [(row.get("block"), row.get("arm"), row.get("update")) for row in dense_rows]
    assert len(set(final_keys)) == 12 and len(set(dense_keys)) == 156
    assert all(row.get("update") == 300 for row in final_rows)
    aggregate = reporting.aggregate(final_rows, dense_rows)
    assert aggregate["status"] == "COMPLETE"
    assert aggregate["verdict"] == "NO_DEVELOPMENTAL_SIGNAL"
    assert aggregate["final_readiness"]["by_arm"] == {
        "current": {"expected_blocks": 4, "observed_endpoints": 4, "ready": 2, "not_ready": 2, "missing_or_invalid": 0},
        "additive": {"expected_blocks": 4, "observed_endpoints": 4, "ready": 0, "not_ready": 4, "missing_or_invalid": 0},
        "delta": {"expected_blocks": 4, "observed_endpoints": 4, "ready": 0, "not_ready": 4, "missing_or_invalid": 0},
    }


def render_csv(reporting, rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=reporting._csv_header(), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(reporting._csv_row(row))
    return output.getvalue()


def validate_archives(publication: dict, dense_rows: list[dict], public_summary: dict) -> dict:
    reporting = module("new/addressed_delta/reporting.py", "_addressed_delta_verify_reporting")
    metrics = module("new/continuous_coverage/metrics.py", "_addressed_delta_verify_metrics")
    seen_records = set()
    marker_count = map_csv_rows = archive_entries_checked = 0
    expected_curve_paths = {
        f"block{block:02d}/{arm}/training_curve.json"
        for block in BLOCKS for arm in ARMS
    }

    for block in BLOCKS:
        archive_path = PUBLIC / f"raw/block{block:02d}.zip"
        binding_key = archive_path.relative_to(PUBLIC).as_posix()
        entries = publication["archive_entries"][binding_key]
        expected_members = set()
        for arm in ARMS:
            expected_members.add(f"block{block:02d}/{arm}/training_curve.json")
            for update in CHECKPOINTS:
                stage = f"block{block:02d}/{arm}/evaluation/u{update:03d}"
                expected_members.update(f"{stage}/{name}" for name in ("summary.json", "map_metrics.csv", "record.json"))
                expected_members.add(f"block{block:02d}/{arm}/checkpoints/u{update:03d}.pt.complete.json")
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            assert names == set(entries) == expected_members, binding_key
            assert all(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts for name in names)
            archive_entries_checked += len(names)
            for name in sorted(names):
                raw = archive.read(name)
                assert sha_bytes(raw) == entries[name]["sha256"], name
                if name.endswith(".json"):
                    safe(read_json_bytes(raw))
                elif name.endswith(".csv"):
                    safe(raw.decode("utf-8-sig"))
                assert not name.endswith((".pt", ".npz")), name

            curves = {}
            for name in sorted(expected_curve_paths):
                if name.startswith(f"block{block:02d}/"):
                    curve = json_from_zip(archive, name)
                    assert isinstance(curve, list) and len(curve) == 300
                    assert [row["update"] for row in curve] == list(range(1, 301))
                    for row in curve:
                        assert row["forward_steps"] == 256
                        assert row["backward_calls"] == row["loss_windows"] == 32
                        assert row["optimizer_steps"] == 1 and row["credit_horizon"] == 8
                    curves[name] = curve

            for row in (item for item in dense_rows if item["block"] == block):
                key = (row["block"], row["arm"], row["update"])
                assert key not in seen_records
                seen_records.add(key)
                assert row["arm"] in ARMS and row["update"] in CHECKPOINTS
                summary_name = row["evaluation_summary"]
                stage = PurePosixPath(summary_name).parent.as_posix()
                record_name = f"{stage}/record.json"
                map_name = f"{stage}/map_metrics.csv"
                marker_name = f"{row['checkpoint']}.complete.json"
                record = json_from_zip(archive, record_name)
                summary = json_from_zip(archive, summary_name)
                marker = json_from_zip(archive, marker_name)
                assert record == row, (key, "dense/record mismatch")
                assert digest(record) == marker["record_sha256"]
                assert marker["stage_complete"] is True
                assert marker["completed_update"] == row["update"]
                assert marker["checkpoint_sha256"] == row["checkpoint_sha256"]
                assert set(marker["evaluation_sha256"]) == {"summary.json", "map_metrics.csv", "record.json"}
                for filename, expected_hash in marker["evaluation_sha256"].items():
                    archive_name = f"{stage}/{filename}"
                    assert entries[archive_name]["sha256"] == expected_hash
                assert summary["metrics"] == row["metrics"]
                assert metrics.joint_readiness(summary["metrics"]) == row["joint"]
                assert summary["joint"] == row["joint"]
                assert row["evaluation_status"] == "COMPLETE"
                assert row["numerical_failure"] is False and row["trace_complete"] is True
                assert row["full_evaluated"] is False
                curve_name = f"block{block:02d}/{row['arm']}/training_curve.json"
                assert digest(curves[curve_name][:row["update"]]) == marker["curve_prefix_sha256"]

                with io.TextIOWrapper(io.BytesIO(archive.read(map_name)), encoding="utf-8-sig", newline="") as handle:
                    map_rows = list(csv.DictReader(handle))
                assert len(map_rows) == 64 and {row["size"] for row in map_rows} == {"32", "64"}
                for size in ("32", "64"):
                    indices = sorted(int(item["map_index"]) for item in map_rows if item["size"] == size)
                    assert indices == list(range(32))
                    selected = [item for item in map_rows if item["size"] == size]
                    point = row["metrics"]["sizes"][size]
                    total = lambda name: sum(int(item[name]) for item in selected)
                    ratio = lambda n, d: n / d if d else None
                    strict_rates = [int(item["strict_correct_T64"]) / int(item["strict_pixels"])
                                    for item in selected if int(item["strict_pixels"])]
                    ref = total("retention_reference_pixels")
                    actual = {
                        "R_strict_pooled_T64": ratio(total("strict_correct_T64"), total("strict_pixels")),
                        "R_strict_mean_T64": sum(strict_rates) / len(strict_rates) if strict_rates else None,
                        "S_retention64_to256": ratio(total("retained_pixels_T256"), ref),
                        "S_continuous_survival64_to256": ratio(total("continuously_retained_pixels_T64_to256"), ref),
                        "retention_reference_pixels": ref,
                        "retention_reference_maps": sum(int(item["retention_reference_pixels"]) > 0 for item in selected),
                    }
                    for field, measured in actual.items():
                        expected = point[field]
                        assert (measured is None and expected is None) or (
                            measured is not None and expected is not None
                            and math.isclose(measured, expected, rel_tol=1e-12, abs_tol=1e-12)), (key, size, field)
                map_csv_rows += len(map_rows)
                marker_count += 1

    assert len(seen_records) == 156 and marker_count == 156
    assert len(expected_curve_paths) == 12
    assert archive_entries_checked == sum(len(value) for value in publication["archive_entries"].values())

    final_rows = read_gzip_json(PUBLIC / "perarm.json.gz")
    final_by_key = {(row["block"], row["arm"]): row for row in final_rows}
    dense_final = {(row["block"], row["arm"]): row for row in dense_rows if row["update"] == 300}
    assert final_by_key == dense_final
    aggregate_summary = reporting._json_ready_summary(final_rows, dense_rows, public_summary["execution_status"])
    assert aggregate_summary == public_summary, "published reporter aggregate differs"

    ordered_dense = sorted(dense_rows, key=lambda row: (row["block"], row["update"], row["arm"]))
    ordered_final = sorted(final_rows, key=lambda row: (row["block"], row["arm"]))
    with (PUBLIC / "metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        assert handle.read() == render_csv(reporting, ordered_dense)
    with (PUBLIC / "final_metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        assert handle.read() == render_csv(reporting, ordered_final)

    return {
        "stage_records": len(seen_records),
        "checkpoint_markers": marker_count,
        "training_curves": len(expected_curve_paths),
        "per_map_metric_rows": map_csv_rows,
        "archive_entries_hash_checked": archive_entries_checked,
        "reporter_aggregate_equal": True,
        "reporter_csv_exports_equal": True,
    }


def read_gzip_json(path: Path):
    with gzip.open(path, "rb") as handle:
        return read_json_bytes(handle.read())


def validation_result(publication: dict, dense_rows: list[dict], final_rows: list[dict], archive_checks: dict) -> dict:
    summary = read_json(PUBLIC / "summary.json")
    return {
        "status": "PASS",
        "protocol": PROTOCOL,
        "scope": "Saved CPU data, source and artifact hashes, reporter arithmetic, records, markers, CSVs, and training-curve cadence only.",
        "inference_or_optimizer_updates": 0,
        "formal_training_updates_per_arm": 300,
        "final_records": len(final_rows),
        "dense_stage_records": len(dense_rows),
        "source_hashes_checked": publication["source_file_count"],
        "bank_tensor_hashes_checked": 3,
        "schedule_plan_sha256": publication["schedule_plan_sha256"],
        "config_sha256": publication["config_sha256"],
        "qualification_sha256": publication["qualification_sha256"],
        "sanitized_qualification_sha256": publication["sanitized_qualification_sha256"],
        "archive_checks": archive_checks,
        "final_readiness": summary["final_readiness"]["by_arm"],
        "verdict": summary["verdict"],
        "old_full_evaluated": False,
        "full_boolean_arrays_persisted": False,
        "checkpoint_contents_published": False,
        "frozen_evidence_changed": False,
    }


def verify() -> dict:
    publication = read_json(MANIFEST)
    safe(publication)
    assert sha(Path(__file__)) == publication["exporter_sha256"]
    assert publication["run_id"] == RUN.name and publication["protocol"] == PROTOCOL
    assert publication["source_file_count"] == len(publication["source_sha256"]) == 71
    for relative, expected in publication["source_sha256"].items():
        assert sha(safe_relative(relative)) == expected, relative
    assert sha(safe_relative("tools/recover_addressed_delta.py")) == publication["recovery_entrypoint_sha256"]

    actual = {path.relative_to(ROOT).as_posix(): sha(path) for path in PUBLIC.rglob("*") if path.is_file()}
    assert actual == publication["published_sha256"]
    expected_files = {
        "RESULTS.md", "summary.json", "metrics.csv", "final_metrics.csv", "config.json",
        "dense.json.gz", "perarm.json.gz", "plans.json.gz", "REPRODUCTION.md",
        "runtime_qualification.json", "interruption_and_recovery.json", "validation.json",
        "banks/train.npz", "banks/evaluation32.npz", "banks/evaluation64.npz",
        *(f"raw/block{block:02d}.zip" for block in BLOCKS),
    }
    assert {PurePosixPath(relative).relative_to(PurePosixPath(PUBLIC.relative_to(ROOT).as_posix())).as_posix()
            for relative in actual} == expected_files
    assert len(actual) == publication["published_file_count"]
    assert sum((ROOT / relative).stat().st_size for relative in actual) == publication["published_bytes"]
    assert all((ROOT / relative).stat().st_size < MAX_FILE_BYTES for relative in actual)
    assert not any(Path(relative).suffix.lower() in (".pt", ".pth") for relative in actual)

    for name, binding in publication["raw_artifact_bindings"].items():
        public_path = safe_relative(binding["public_path"])
        if binding["encoding"] == "gzip lossless":
            with gzip.open(public_path, "rb") as handle:
                assert sha_bytes(handle.read()) == binding["run_sha256"], name
        elif binding["encoding"] == "byte-exact copy":
            assert sha(public_path) == binding["run_sha256"], name
        else:
            assert binding["encoding"] == "sanitized JSON copy" and name == "summary.json"

    for path in PUBLIC.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            safe(read_json(path))
        elif path.suffix in (".md", ".csv"):
            safe(path.read_text(encoding="utf-8-sig"))

    qualification = read_json(PUBLIC / "runtime_qualification.json")
    assert qualification["status"] == "PASS" and qualification["protocol"] == PROTOCOL
    assert len(qualification["source_sha256"]) == qualification["source_file_count"] == 71
    assert qualification["qualification_sha256"] == publication["qualification_sha256"]
    assert qualification["source_sha256"] == publication["source_sha256"]
    assert isinstance(qualification["arms"], list) and len(qualification["arms"]) == 3
    assert all(row["status"] == "PASS" and row["three_update_gradient_max_absolute_error"] == 0.0
               for row in qualification["arms"])
    interruption = read_json(PUBLIC / "interruption_and_recovery.json")
    assert interruption["recovered_run"]["inherited_committed_stages"] == 22
    assert interruption["recovered_run"]["recovery_entrypoint_sha256"] == publication["recovery_entrypoint_sha256"]
    assert read_json(PUBLIC / "validation.json")["status"] == "PASS"
    assert sha(PUBLIC / "config.json") == publication["config_sha256"]

    for label, filename in (("train", "train.npz"), ("evaluation32", "evaluation32.npz"),
                            ("evaluation64", "evaluation64.npz")):
        bank_path = PUBLIC / "banks" / filename
        assert sha(bank_path) == publication["bank_file_sha256"][label]
        assert bank_tensor_hash(load_bank(bank_path)) == publication["data_sha256"][label]

    plan_raw = gzip.decompress((PUBLIC / "plans.json.gz").read_bytes())
    assert sha_bytes(plan_raw) == publication["schedule_plan_sha256"]
    plans = read_json_bytes(plan_raw)
    safe(plans)
    assert set(plans) == {str(block) for block in BLOCKS}
    for block in BLOCKS:
        plan = plans[str(block)]
        assert plan["initialization_seed"] == 140001 + block
        assert plan["schedule_seed"] == 141001 + block
        indices = np.asarray(plan["batch_indices"])
        assert indices.shape == (300, 8) and indices.min() >= 0 and indices.max() < 512
        assert plan["arm_order"] == [ARMS[(i + block) % 3] for i in range(3)]

    summary = read_json(PUBLIC / "summary.json")
    dense_rows = read_gzip_json(PUBLIC / "dense.json.gz")
    final_rows = read_gzip_json(PUBLIC / "perarm.json.gz")
    safe(dense_rows)
    safe(final_rows)
    stage_grid(final_rows, dense_rows)
    reporting = module("new/addressed_delta/reporting.py", "_addressed_delta_verify_csv_reporting")
    assert reporting._json_ready_summary(final_rows, dense_rows, summary["execution_status"]) == summary
    archive_checks = validate_archives(publication, dense_rows, summary)
    expected_validation = validation_result(publication, dense_rows, final_rows, archive_checks)
    assert read_json(PUBLIC / "validation.json") == expected_validation

    return {
        "status": "PASS",
        "files": len(actual),
        "published_bytes": publication["published_bytes"],
        "final_records": len(final_rows),
        "dense_stage_records": len(dense_rows),
        "bank_tensor_hashes": 3,
        "source_hashes": publication["source_file_count"],
        "verdict": summary["verdict"],
        **archive_checks,
    }


def copy_raw(source: Path, destination: Path, name: str, bindings: dict, encoding: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    if encoding == "gzip lossless":
        destination.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    elif encoding == "byte-exact copy":
        shutil.copyfile(source, destination)
    elif encoding == "sanitized JSON copy":
        sanitized, removed = public_summary(read_json(source))
        if removed:
            write_json(destination, sanitized)
        else:
            shutil.copyfile(source, destination)
    else:
        raise ValueError(encoding)
    assert destination.stat().st_size < MAX_FILE_BYTES
    bindings[name] = {
        "run_sha256": sha(source),
        "public_path": destination.relative_to(ROOT).as_posix(),
        "encoding": encoding,
    }
    if encoding == "sanitized JSON copy":
        bindings[name]["removed_private_fields"] = removed


def create_archives() -> dict:
    (PUBLIC / "raw").mkdir(parents=True, exist_ok=True)
    archive_entries = {}
    for block in BLOCKS:
        archive_relative = f"raw/block{block:02d}.zip"
        archive_path = PUBLIC / archive_relative
        entries = {}
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for arm in ARMS:
                curve_relative = f"block{block:02d}/{arm}/training_curve.json"
                sources = [(curve_relative, RUN / curve_relative)]
                for update in CHECKPOINTS:
                    stage = f"block{block:02d}/{arm}/evaluation/u{update:03d}"
                    checkpoint = f"block{block:02d}/{arm}/checkpoints/u{update:03d}.pt"
                    sources.extend((f"{stage}/{filename}", RUN / stage / filename)
                                   for filename in ("summary.json", "map_metrics.csv", "record.json"))
                    marker = f"{checkpoint}.complete.json"
                    sources.append((marker, RUN / marker))
                for relative, source in sources:
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    raw = source.read_bytes()
                    if relative.endswith(".json"):
                        safe(read_json_bytes(raw))
                    elif relative.endswith(".csv"):
                        safe(raw.decode("utf-8-sig"))
                    archive.writestr(relative, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
                    entries[relative] = {"sha256": sha_bytes(raw), "raw_bytes": len(raw)}
        assert len(entries) == 159
        assert archive_path.stat().st_size < MAX_FILE_BYTES
        archive_entries[archive_relative] = entries
    return archive_entries


def build() -> None:
    assert not PUBLIC.exists() and not MANIFEST.exists(), "Never overwrite a publication snapshot"
    run_manifest = read_json(RUN / "manifest.json")
    run_status = read_json(RUN / "status.json")
    summary = read_json(RUN / "summary.json")
    qualification = read_json(QUALIFICATION)
    parent_status = read_json(PARENT / "status.json")
    parent_manifest_path = PARENT / "manifest.json"
    source_hashes = dict(sorted(run_manifest["source_sha256"].items()))

    assert run_manifest["protocol"] == PROTOCOL and run_status["status"] == "COMPLETE"
    assert run_status["completed_arms"] == 12 and run_status["dense_records"] == 156
    assert summary["status"] == "COMPLETE" and summary["verdict"] == "NO_DEVELOPMENTAL_SIGNAL"
    assert sha(RUN / "config.json") == run_manifest["config_sha256"]
    assert sha(RUN / "plans.json") == run_manifest["schedule_plan_sha256"]
    assert sha(QUALIFICATION) == run_manifest["qualification_sha256"]
    assert qualification["status"] == "PASS" and qualification["protocol"] == PROTOCOL
    assert qualification["source_sha256"] == source_hashes and len(source_hashes) == 71
    assert parent_status["status"] == "CANCELLED"
    assert sha(parent_manifest_path) == run_manifest["parent_manifest_sha256"]
    assert run_manifest["recovery_compatibility"]["inherited_committed_stages"] == 22
    assert run_manifest["recovery_compatibility"]["usable_prefix"] == {
        "block00/current": 300, "block00/additive": 200,
    }

    for relative, expected in source_hashes.items():
        current = safe_relative(relative)
        snapshot = RUN / "source" / Path(*PurePosixPath(relative).parts)
        assert snapshot.is_file() and sha(snapshot) == expected, relative
        assert current.is_file() and sha(current) == expected, relative
    wrapper = safe_relative("tools/recover_addressed_delta.py")
    assert sha(wrapper) == run_manifest["recovery_compatibility"]["entrypoint_sha256"]
    wrapper_snapshot = RUN / "source/tools/recover_addressed_delta.py"
    assert wrapper_snapshot.is_file() and sha(wrapper_snapshot) == sha(wrapper)

    config = read_json(RUN / "config.json")
    assert config["old_full_evaluated"] is False and config["full_boolean_trace_persistence"] is False
    dense_rows, final_rows = read_json(RUN / "dense.json"), read_json(RUN / "perarm.json")
    stage_grid(final_rows, dense_rows)
    assert module("new/addressed_delta/reporting.py", "_addressed_delta_build_reporting")._json_ready_summary(
        final_rows, dense_rows, summary["execution_status"]
    ) == summary

    upstream_manifest = read_json(UPSTREAM_MANIFEST)
    for label, filename in (("train", "train.npz"), ("evaluation32", "evaluation32.npz"),
                            ("evaluation64", "evaluation64.npz")):
        run_bank = RUN / "banks" / filename
        upstream_bank = UPSTREAM_BANKS / filename
        expected_tensor = run_manifest["data_sha256"][label if label == "train" else label]
        assert bank_tensor_hash(load_bank(run_bank)) == expected_tensor
        assert bank_tensor_hash(load_bank(upstream_bank)) == expected_tensor
        assert sha(run_bank) == sha(upstream_bank)
        assert expected_tensor == upstream_manifest["data_sha256"][label]

    PUBLIC.mkdir(parents=True)
    raw_bindings = {}
    for name, encoding in (
        ("RESULTS.md", "byte-exact copy"),
        ("summary.json", "sanitized JSON copy"),
        ("metrics.csv", "byte-exact copy"),
        ("final_metrics.csv", "byte-exact copy"),
        ("config.json", "byte-exact copy"),
        ("dense.json", "gzip lossless"),
        ("perarm.json", "gzip lossless"),
        ("plans.json", "gzip lossless"),
    ):
        copy_raw(RUN / name, PUBLIC / (name + ".gz" if encoding == "gzip lossless" else name), name,
                 raw_bindings, encoding)
    bank_file_sha256 = {}
    for label, filename in (("train", "train.npz"), ("evaluation32", "evaluation32.npz"),
                            ("evaluation64", "evaluation64.npz")):
        source, target = RUN / "banks" / filename, PUBLIC / "banks" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        bank_file_sha256[label] = sha(source)
        raw_bindings[f"banks/{filename}"] = {
            "run_sha256": bank_file_sha256[label],
            "public_path": target.relative_to(ROOT).as_posix(),
            "encoding": "byte-exact copy",
        }

    qual_hash = sha(QUALIFICATION)
    qualification_public_value = qualification_public(qualification, qual_hash)
    write_json(PUBLIC / "runtime_qualification.json", qualification_public_value)
    interruption = interruption_public(run_manifest, parent_manifest_path, parent_status)
    write_json(PUBLIC / "interruption_and_recovery.json", interruption)
    (PUBLIC / "REPRODUCTION.md").write_text(reproduction_text(), encoding="utf-8")
    archive_entries = create_archives()

    sanitized_summary, removed_summary_fields = public_summary(summary)
    if removed_summary_fields:
        assert read_json(PUBLIC / "summary.json") == sanitized_summary
    final_public_summary = read_json(PUBLIC / "summary.json")
    final_public_summary = public_summary(final_public_summary)[0]
    archive_checks = validate_archives(
        {"archive_entries": {f"{key}": value for key, value in archive_entries.items()}},
        read_gzip_json(PUBLIC / "dense.json.gz"), final_public_summary,
    )
    provisional = {
        "status": "PASS", "protocol": PROTOCOL,
        "scope": "Saved CPU data, source and artifact hashes, reporter arithmetic, records, markers, CSVs, and training-curve cadence only.",
        "inference_or_optimizer_updates": 0, "formal_training_updates_per_arm": 300,
        "final_records": 12, "dense_stage_records": 156,
        "source_hashes_checked": len(source_hashes), "bank_tensor_hashes_checked": 3,
        "schedule_plan_sha256": run_manifest["schedule_plan_sha256"],
        "config_sha256": run_manifest["config_sha256"],
        "qualification_sha256": qual_hash,
        "sanitized_qualification_sha256": sha(PUBLIC / "runtime_qualification.json"),
        "archive_checks": archive_checks,
        "final_readiness": final_public_summary["final_readiness"]["by_arm"],
        "verdict": final_public_summary["verdict"],
        "old_full_evaluated": False, "full_boolean_arrays_persisted": False,
        "checkpoint_contents_published": False, "frozen_evidence_changed": False,
    }
    write_json(PUBLIC / "validation.json", provisional)

    published = {path.relative_to(ROOT).as_posix(): sha(path)
                 for path in PUBLIC.rglob("*") if path.is_file()}
    published_bytes = sum(path.stat().st_size for path in PUBLIC.rglob("*") if path.is_file())
    recovery = run_manifest["recovery_compatibility"]
    publication = {
        "schema": "addressed-delta-development-publication-v1",
        "protocol": PROTOCOL,
        "run_id": RUN.name,
        "publication_status": "COMPLETE",
        "verdict": summary["verdict"],
        "source_sha256": source_hashes,
        "source_file_count": len(source_hashes),
        "exporter_sha256": sha(Path(__file__)),
        "config_sha256": run_manifest["config_sha256"],
        "schedule_plan_sha256": run_manifest["schedule_plan_sha256"],
        "qualification_sha256": qual_hash,
        "sanitized_qualification_sha256": sha(PUBLIC / "runtime_qualification.json"),
        "data_sha256": run_manifest["data_sha256"],
        "bank_file_sha256": bank_file_sha256,
        "bank_provider_manifest_sha256": sha(UPSTREAM_MANIFEST),
        "recovery_entrypoint_sha256": recovery["entrypoint_sha256"],
        "parent_manifest_sha256": run_manifest["parent_manifest_sha256"],
        "inherited_committed_stages": recovery["inherited_committed_stages"],
        "sanitized_summary_fields_removed": removed_summary_fields,
        "raw_artifact_bindings": raw_bindings,
        "archive_entries": archive_entries,
        "published_sha256": published,
        "published_file_count": len(published),
        "published_bytes": published_bytes,
        "excluded": [
            "checkpoint and optimizer-state contents",
            "run manifest, status/PID records, qualification timing telemetry, and launch logs",
            "source snapshots (their 71 hashes are bound to the qualified source and current files)",
            "absolute host paths, usernames, server addresses, GPU identifiers, and transient logs",
            "full Boolean arrays and old Full phenotype evaluation",
        ],
    }
    safe(publication)
    write_json(MANIFEST, publication)
    print(json.dumps(verify(), sort_keys=True, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify the public snapshot from CPU saved data without reading runs or checkpoints.")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(), sort_keys=True, allow_nan=False), flush=True)
    else:
        build()


if __name__ == "__main__":
    main()
