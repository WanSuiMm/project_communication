"""CPU-only validation of the saved dense seed4 transition audit.

This reads the frozen run and its source snapshot. It never instantiates a
model, performs inference, trains, or writes into the run directory.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/transition_20261003_seed4_dense01"
REFERENCE = ROOT / "runs/warmstart_20261003_paired01"
OUTPUT = ROOT / "analyses/transition_20261003_validation01.json"
DISPATCH_BINDINGS = ROOT / "analyses/transition_20261003_dispatch_bindings01.json"
PUBLIC_HISTORICAL = ROOT / "evidence/streaming_carry_init2345/raw/stream_K8_seed4.json"
PUBLIC_REPLAY_REFERENCE = ROOT / "evidence/transition_20261003/replay_reference/manifest.json"
PRIOR_FAILED_REPORT = ROOT / "analyses/transition_20261003_validation01_attempt01_failed.json"
TOL = 2e-7
UPDATES = (*range(100, 201, 5), 300)
CHECKPOINTS = (0, *range(100, 201, 5), 300)
SIZES = (32, 64)
ENDPOINTS = (32, 64, 128, 256)


class Validation:
    def __init__(self) -> None:
        self.checks: dict[str, dict[str, Any]] = {}
        self.failures: list[str] = []
        self.max_float_error = 0.0
        self.compare_counts = {"integer_leaves": 0, "float_leaves": 0, "boolean_leaves": 0,
                               "null_leaves": 0, "string_leaves": 0}

    def record(self, name: str, ok: bool, **details: Any) -> None:
        self.checks[name] = {**details, "status": "PASS" if ok else "FAIL"}
        if not ok:
            self.failures.append(name)

    def compare(self, expected: Any, actual: Any, *, tolerance: float = TOL,
                label: str = "value") -> dict[str, Any]:
        counts = {"integer_leaves": 0, "float_leaves": 0, "boolean_leaves": 0,
                  "null_leaves": 0, "string_leaves": 0}
        mismatches: list[dict[str, str]] = []
        maximum = 0.0

        def fail(path: str, reason: str) -> None:
            if len(mismatches) < 12:
                mismatches.append({"path": path, "reason": reason})

        def walk(left: Any, right: Any, path: str) -> None:
            nonlocal maximum
            if isinstance(left, dict) or isinstance(right, dict):
                if not isinstance(left, dict) or not isinstance(right, dict):
                    fail(path, "container type differs")
                    return
                if set(left) != set(right):
                    fail(path, "object keys differ")
                    return
                for key in left:
                    walk(left[key], right[key], f"{path}.{key}")
                return
            if isinstance(left, list) or isinstance(right, list):
                if not isinstance(left, list) or not isinstance(right, list):
                    fail(path, "container type differs")
                    return
                if len(left) != len(right):
                    fail(path, "array lengths differ")
                    return
                for index, (a, b) in enumerate(zip(left, right)):
                    walk(a, b, f"{path}[{index}]")
                return
            if isinstance(left, bool) or isinstance(right, bool):
                counts["boolean_leaves"] += 1
                if type(left) is not bool or type(right) is not bool or left != right:
                    fail(path, "boolean differs")
                return
            if left is None or right is None:
                counts["null_leaves"] += 1
                if left is not None or right is not None:
                    fail(path, "nullability differs")
                return
            if isinstance(left, int) or isinstance(right, int):
                counts["integer_leaves"] += 1
                if type(left) is not int or type(right) is not int or left != right:
                    fail(path, "integer differs")
                return
            if isinstance(left, float) or isinstance(right, float):
                counts["float_leaves"] += 1
                if type(left) is not float or type(right) is not float:
                    fail(path, "numeric leaf type differs")
                    return
                if not np.isfinite(left) or not np.isfinite(right):
                    fail(path, "nonfinite float")
                    return
                error = abs(left - right)
                maximum = max(maximum, error)
                if error > tolerance:
                    fail(path, f"absolute error {error:.9g} exceeds {tolerance:.9g}")
                return
            counts["string_leaves"] += 1
            if type(left) is not str or type(right) is not str or left != right:
                fail(path, "string differs")

        walk(expected, actual, label)
        for key, value in counts.items():
            self.compare_counts[key] += value
        self.max_float_error = max(self.max_float_error, maximum)
        return {"ok": not mismatches, **counts, "maximum_absolute_error": maximum,
                "tolerance": tolerance, "mismatches": mismatches}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def arrays_equal(left: Any, right: Any) -> bool:
    return np.asarray(left).shape == np.asarray(right).shape and np.array_equal(left, right)


def scalar_close(left: float | None, right: float | None, tolerance: float = TOL) -> tuple[bool, float]:
    if left is None or right is None:
        return left is None and right is None, 0.0
    if not np.isfinite(left) or not np.isfinite(right):
        return False, float("inf")
    error = abs(float(left) - float(right))
    return error <= tolerance, error


def balanced_accuracy(correct: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    values: list[float] = []
    for index in range(correct.shape[0]):
        valid = mask[index]
        positive = valid & (labels[index] >= 0.5)
        negative = valid & (labels[index] < 0.5)
        recalls = []
        if positive.any():
            recalls.append(float(np.count_nonzero(correct[index] & positive) / positive.sum()))
        if negative.any():
            recalls.append(float(np.count_nonzero(correct[index] & negative) / negative.sum()))
        values.append(float(np.mean(recalls)) if recalls else 0.0)
    return float(np.mean(values))


def endpoint_payload(npz: dict[str, np.ndarray], bank: dict[str, np.ndarray], step: int) -> dict[str, Any]:
    correct = npz["correct"][step]
    changed = npz["changed"][:, 0].astype(bool)
    distance = npz["distance"][:, 0]
    strict = changed & (distance > 16) & (distance < 32)
    nums = np.count_nonzero(correct & strict, axis=(1, 2)).astype(np.int64)
    dens = np.count_nonzero(strict, axis=(1, 2)).astype(np.int64)
    per_map = [float(n / d) if d else None for n, d in zip(nums, dens)]
    eligible = [value for value in per_map if value is not None]
    denominator = int(dens.sum())
    changed_labels = bank["y"][:, 0]
    flipped_labels = bank["y_flip"][:, 0]
    mask = bank["mask"][:, 0] > 0.5
    margin = npz["margin"][step][npz["changed"][:, 0].astype(bool)]
    return {
        "strict_coverage": {
            "per_map": per_map,
            "per_map_numerator": nums.tolist(),
            "per_map_denominator": dens.tolist(),
            "equal_map_mean": float(np.mean(eligible)) if eligible else None,
            "pooled_rate": float(nums.sum() / denominator) if denominator else None,
        },
        "original_BA_equal_map": balanced_accuracy(npz["original_correct"][step], changed_labels, mask),
        "flipped_BA_equal_map": balanced_accuracy(npz["flipped_correct"][step], flipped_labels, mask),
        "paired_margin_quantiles": np.quantile(margin.astype(np.float64), [0, .1, .5, .9, 1]).tolist(),
    }


def audit_one(v: Validation, metrics: Any, update: int, size: int,
              expected_bank: dict[str, np.ndarray], expected_tensor_sha: str) -> dict[str, Any] | None:
    stem = f"u{update:03}"
    raw_path = RUN / "checkpoints" / f"{stem}_size{size}.npz"
    row_path = RUN / "checkpoints" / f"{stem}_size{size}.json"
    key = f"u{update}_size{size}"
    if not raw_path.is_file() or not row_path.is_file():
        v.record(f"audit_files_{key}", False, raw_exists=raw_path.is_file(), row_exists=row_path.is_file())
        return None
    row = read_json(row_path)
    raw_digest = sha256(raw_path)
    v.record(f"audit_raw_hash_{key}", raw_digest == row.get("trace_sha256"), sha256_matches=True if raw_digest == row.get("trace_sha256") else False)
    v.record(f"audit_bank_binding_{key}", row.get("audit_bank_sha256") == expected_tensor_sha,
             tensor_hash_matches=row.get("audit_bank_sha256") == expected_tensor_sha)
    with np.load(raw_path, allow_pickle=False) as data:
        npz = {name: data[name] for name in data.files}

    expected_names = {"correct", "original_correct", "flipped_correct", "margin", "changed", "distance", "mask", "source"}
    names_ok = set(npz) == expected_names
    expected_shape = (257, 16, size, size)
    shape_ok = names_ok and all(npz[name].shape == expected_shape for name in ("correct", "original_correct", "flipped_correct", "margin"))
    map_shape = (16, 1, size, size)
    shape_ok = shape_ok and all(npz[name].shape == map_shape for name in ("changed", "distance", "mask", "source"))
    dtypes_ok = names_ok and all(npz[name].dtype == np.bool_ for name in ("correct", "original_correct", "flipped_correct", "source"))
    dtypes_ok = dtypes_ok and npz["margin"].dtype == np.float32 and npz["distance"].dtype == np.int32
    v.record(f"audit_schema_{key}", shape_ok and dtypes_ok and names_ok,
             fields=len(npz), shape=expected_shape if shape_ok else "mismatch", boolean_dtypes=dtypes_ok)
    if not (shape_ok and dtypes_ok and names_ok):
        return row

    margins = npz["margin"]
    margin_finite = int(np.count_nonzero(np.isfinite(margins))) == margins.size
    paired_exact = np.array_equal(npz["correct"], npz["original_correct"] & npz["flipped_correct"])
    geometry_fields = ("changed", "distance", "mask")
    geometry_exact = all(arrays_equal(npz[name], expected_bank[name]) for name in geometry_fields)
    source_derived = ((expected_bank["x"][:, 1:2] != 0) | (expected_bank["x"][:, 2:3] != 0))
    source_exact = np.array_equal(npz["source"], source_derived)
    pos_margin_conflicts = int(np.count_nonzero((margins > 0) & ~npz["correct"]))
    neg_margin_conflicts = int(np.count_nonzero((margins < 0) & npz["correct"]))
    margin_sign_ok = pos_margin_conflicts == 0 and neg_margin_conflicts == 0
    v.record(f"audit_trace_invariants_{key}", paired_exact and geometry_exact and source_exact and margin_finite and margin_sign_ok,
             paired_boolean_mismatches=int(np.count_nonzero(npz["correct"] != (npz["original_correct"] & npz["flipped_correct"]))),
             bank_geometry_exact=geometry_exact, source_derived_exact=source_exact,
             finite_margin_values=int(np.count_nonzero(np.isfinite(margins))), margin_values=int(margins.size),
             positive_margin_wrong_pairs=pos_margin_conflicts, negative_margin_right_pairs=neg_margin_conflicts,
             zero_margin_values=int(np.count_nonzero(margins == 0)))

    times = np.arange(257, dtype=np.int32).reshape(257, 1, 1, 1)
    changed = npz["changed"][:, 0].astype(bool)
    distance = npz["distance"][:, 0]
    outside = changed[None] & (distance[None] > 2 * times)
    lightcone_violations = int(np.count_nonzero(outside & npz["correct"]))
    v.record(f"audit_light_cone_{key}", lightcone_violations == 0,
             paired_correct_outside_two_hop_cone=lightcone_violations)

    detail_finite = row.get("finite") is True and row.get("steps") == 256 and row.get("light_cone") is True
    rms_rows = row.get("state_rms", [])
    rms_fields = ("original_W", "original_Z", "flipped_W", "flipped_Z")
    rms_valid = len(rms_rows) == 257 and all(
        item.get("t") == t and all(isinstance(item.get(field), (int, float)) and np.isfinite(item[field]) and item[field] >= 0 for field in rms_fields)
        for t, item in enumerate(rms_rows)
    )
    bce_finite: dict[str, int] = {"checked": 0, "invalid": 0}
    endpoint_max_error = 0.0
    endpoints_ok = True
    for step in ENDPOINTS:
        stored = row.get("endpoints", {}).get(str(step), {})
        derived = endpoint_payload(npz, expected_bank, step)
        for field, value in derived.items():
            result = v.compare(value, stored.get(field), label=f"{key}.endpoints.{step}.{field}")
            endpoints_ok &= result["ok"]
            endpoint_max_error = max(endpoint_max_error, result["maximum_absolute_error"])
        for field in ("original_BCE_balanced", "flipped_BCE_balanced"):
            value = stored.get(field)
            bce_finite["checked"] += 1
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                bce_finite["invalid"] += 1
        for field in ("original_BA_equal_map", "flipped_BA_equal_map"):
            value = stored.get(field)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                endpoints_ok = False
    v.record(f"audit_endpoint_recompute_{key}", endpoints_ok and bce_finite["invalid"] == 0,
             endpoint_count=len(ENDPOINTS), max_absolute_error=endpoint_max_error,
             bce_recorded_finite=bce_finite["invalid"] == 0, bce_scalars_checked=bce_finite["checked"])
    v.record(f"audit_state_rms_finite_{key}", rms_valid,
             recorded_steps=len(rms_rows), expected_steps=257, values_checked=len(rms_rows) * len(rms_fields), raw_states_saved=False)
    v.record(f"audit_row_flags_{key}", detail_finite, finite_flag=row.get("finite"), steps=row.get("steps"), light_cone_flag=row.get("light_cone"))

    recomputed = metrics.summarize(npz["correct"], npz["changed"], npz["distance"], source=npz["source"])
    behavior_compare = v.compare(recomputed, row.get("behavior"), label=f"{key}.behavior")
    v.record(f"audit_behavior_recompute_{key}", behavior_compare["ok"],
             integer_leaves=behavior_compare["integer_leaves"], float_leaves=behavior_compare["float_leaves"],
             null_leaves=behavior_compare["null_leaves"], boolean_leaves=behavior_compare["boolean_leaves"],
             maximum_absolute_error=behavior_compare["maximum_absolute_error"], tolerance=TOL,
             mismatch_count=len(behavior_compare["mismatches"]))
    profile_compare = v.compare(recomputed["intervals"], row.get("behavior_profile"), label=f"{key}.behavior_profile")
    v.record(f"audit_profile_recompute_{key}", profile_compare["ok"],
             maximum_absolute_error=profile_compare["maximum_absolute_error"], mismatch_count=len(profile_compare["mismatches"]))
    # Later summary checks use only the compact profile, endpoint and stage fields.
    row.pop("behavior", None)
    row.pop("state_rms", None)
    return row


def finalize_saved_full_pass() -> int:
    """Repair only the failed reference lookup and rerun cheap bindings.

    The 44 raw-metric recomputations remain those recorded in the preserved
    first full pass. This mode deliberately does not rerun model or metric work.
    """
    prior = read_json(PRIOR_FAILED_REPORT)
    prior_sha = sha256(PRIOR_FAILED_REPORT)
    if prior.get("status") != "FAIL" or prior.get("failed_checks") != ["audit_banks_regenerated"]:
        raise RuntimeError("Expected the preserved first full-pass report with only the known bank-reference schema failure")
    root_snapshot = RUN / "source"
    tasks = load_module("_transition_finalize_tasks", root_snapshot / "new/nca_inertial_wind_tunnel/tasks.py")
    manifest = read_json(RUN / "manifest.json")
    dispatch = read_json(DISPATCH_BINDINGS)
    warm_manifest = read_json(REFERENCE / "manifest.json")
    replay_reference = read_json(PUBLIC_REPLAY_REFERENCE)
    banks: dict[str, Any] = {}
    bank_ok = True
    for size in SIZES:
        generated = tasks.bank(size, 16, 61000 + size, "cpu")
        expected_sha = tensor_hash(generated)
        bank_path = RUN / f"bank_size{size}.npz"
        with np.load(bank_path, allow_pickle=False) as arrays_in:
            arrays = {key: arrays_in[key] for key in arrays_in.files}
        generated_arrays = {key: value.detach().cpu().numpy() for key, value in generated.items()}
        contents_match = set(arrays) == set(generated_arrays) and all(
            arrays_equal(arrays[key], generated_arrays[key]) for key in generated_arrays
        )
        actual_file_sha = sha256(bank_path)
        manifest_item = manifest.get("audit_banks", {}).get(str(size), {})
        dispatch_item = dispatch.get("audit_banks", {}).get(str(size), {})
        item_ok = (
            manifest_item.get("maps") == 16 and manifest_item.get("seed") == 61000 + size
            and manifest_item.get("tensor_sha256") == expected_sha
            and dispatch_item.get("tensor_sha256") == expected_sha
            and dispatch_item.get("file_sha256") == actual_file_sha
            and contents_match
        )
        bank_ok &= item_ok
        banks[str(size)] = {"tensor_hash_matches_manifest_and_regeneration": manifest_item.get("tensor_sha256") == expected_sha,
                            "npz_content_matches_regeneration": contents_match,
                            "file_hash_matches_dispatch_binding": dispatch_item.get("file_sha256") == actual_file_sha}
    size32_hash = manifest.get("audit_banks", {}).get("32", {}).get("tensor_sha256")
    warm_hash_ok = warm_manifest.get("stage_diagnostic_data_sha256") == size32_hash
    public_hash_ok = replay_reference.get("stage_diagnostic_data_sha256") == size32_hash
    bank_ok &= warm_hash_ok and public_hash_ok

    dispatch_validator = ROOT / "tools/validate_transition_dispatch.py"
    dispatcher_hash = sha256(dispatch_validator) if dispatch_validator.is_file() else None
    dispatch_script_hash_ok = dispatcher_hash is not None and dispatcher_hash == dispatch.get("validation_script_sha256")
    historical_data_hash_ok = dispatch.get("historical_evaluation_data_sha256") == warm_manifest.get("historical_evaluation_data_sha256")
    manifest_binding_ok = dispatch.get("status") == "PASS" and dispatch.get("training") is False
    manifest_binding_ok &= dispatch.get("gpu_execution") is False
    manifest_binding_ok &= dispatch.get("manifest_sha256") == sha256(RUN / "manifest.json")
    manifest_binding_ok &= dispatch.get("source_sha256") == manifest.get("source_sha256")
    manifest_binding_ok &= dispatch.get("reference_manifest_sha256") == sha256(REFERENCE / "manifest.json")
    manifest_binding_ok &= dispatch_script_hash_ok and historical_data_hash_ok

    checks = dict(prior.get("checks", {}))
    checks["audit_banks_regenerated"] = {
        "status": "PASS" if bank_ok else "FAIL",
        "sizes_checked": list(SIZES),
        "maps_per_size": 16,
        "generator": "frozen source snapshot CPU bank()",
        "bank_hash_checks": banks,
        "historical_evaluation_bindings_match": checks.get("audit_banks_regenerated", {}).get("historical_evaluation_bindings_match"),
        "size32_stage_bank_matches_warmstart_reference": warm_hash_ok,
        "size32_stage_bank_matches_public_replay_reference": public_hash_ok,
        "repair_note": "The first full pass also queried the older streaming-carry manifest, which has no stage_diagnostic_data_sha256 field. The prior warmstart manifest and public transition replay_reference manifest both carry the matching hash.",
    }
    old_dispatch = dict(checks.get("postdispatch_bindings", {}))
    checks["postdispatch_bindings"] = {
        **old_dispatch,
        "status": "PASS" if manifest_binding_ok else "FAIL",
        "dispatch_validator_source_hash_independently_bound": dispatch_script_hash_ok,
        "dispatch_validator_source_hash_matches": dispatch_script_hash_ok,
        "historical_evaluation_data_bindings_match": historical_data_hash_ok,
    }
    run_status = read_json(RUN / "status.json")
    run_aggregate = read_json(RUN / "aggregate.json")
    run_training = read_json(RUN / "training.json")
    run_summary = read_json(RUN / "summary.json")
    completion_ok = (
        run_status.get("status") == "COMPLETE" and run_status.get("completed_audits") == 44
        and run_aggregate.get("pass") is True and run_aggregate.get("execution_complete") is True
        and run_aggregate.get("completed_updates") == 300 and run_aggregate.get("completed_audits") == 44
        and run_training.get("completed_updates") == 300 and len(run_training.get("checkpoints", {})) == 23
        and run_summary.get("execution_complete") is True and run_summary.get("scientific_endpoint") is True
        and run_summary.get("diagnostic_only") is True
    )
    prior_completion = dict(checks.get("run_completion", {}))
    checks["run_completion"] = {
        **prior_completion,
        "status": "PASS" if completion_ok else "FAIL",
        "run_status": run_status.get("status"),
        "updates": run_training.get("completed_updates"),
        "audit_rows": run_status.get("completed_audits"),
        "elapsed_seconds": run_status.get("elapsed_seconds"),
        "aggregate_complete": run_aggregate.get("execution_complete"),
        "serialization_repair": "The first report used status as both check status and run-status detail; the saved COMPLETE value replaced PASS in the JSON field.",
    }
    failed = [name for name, value in checks.items() if value.get("status") != "PASS"]
    report = dict(prior)
    report["checks"] = checks
    report["status"] = "FAIL" if failed else "PASS"
    report["failed_checks"] = failed
    counters = dict(report.get("counters", {}))
    get_checks = report.get("checks", {})
    counters.update({
        "state_rms_records_checked": sum(item.get("recorded_steps", 0) for name, item in get_checks.items() if name.startswith("audit_state_rms_finite_")),
        "state_rms_scalars_checked": sum(item.get("values_checked", 0) for name, item in get_checks.items() if name.startswith("audit_state_rms_finite_")),
        "endpoint_payloads_recomputed": sum(item.get("endpoint_count", 0) for name, item in get_checks.items() if name.startswith("audit_endpoint_recompute_")),
        "recorded_bce_scalars_checked_finite_only": sum(item.get("bce_scalars_checked", 0) for name, item in get_checks.items() if name.startswith("audit_endpoint_recompute_")),
        "paired_margin_values_scanned": sum(item.get("margin_values", 0) for name, item in get_checks.items() if name.startswith("audit_trace_invariants_")),
        "paired_boolean_mismatches": sum(item.get("paired_boolean_mismatches", 0) for name, item in get_checks.items() if name.startswith("audit_trace_invariants_")),
        "two_hop_cone_violations": sum(item.get("paired_correct_outside_two_hop_cone", 0) for name, item in get_checks.items() if name.startswith("audit_light_cone_")),
        "stage_replay_integer_leaves": sum(item.get("integer_leaves", 0) for item in checks["historical_stage_pure_trace_replay"].get("checkpoints", [])),
        "stage_replay_float_leaves": sum(item.get("float_leaves", 0) for item in checks["historical_stage_pure_trace_replay"].get("checkpoints", [])),
    })
    report["counters"] = counters
    report["prior_attempt"] = {
        "artifact": PRIOR_FAILED_REPORT.relative_to(ROOT).as_posix(),
        "sha256": prior_sha,
        "status": prior.get("status"),
        "failed_checks": prior.get("failed_checks"),
        "reason": "Validator schema assumption: the older streaming-carry evidence manifest omits the stage-bank hash field; bank artifacts themselves matched regeneration.",
    }
    report["finalization"] = {
        "mode": "cheap_reference_and_dispatch_binding_recheck_only",
        "full_44_audit_metric_recomputations_reused": True,
        "full_44_audit_metric_recomputations_rerun": False,
        "first_full_pass_validator_sha256": prior.get("validator_sha256"),
        "finalizer_validator_sha256": sha256(Path(__file__)),
        "rechecked": ["both CPU-regenerated audit banks", "warmstart and public replay-reference stage hash", "dispatch manifest/source/reference/script/evaluation-data bindings"],
        "bank_schema_failure_corrected": bank_ok,
        "dispatch_script_hash_verified": dispatch_script_hash_ok,
    }
    report["limitations"] = [
        item for item in report.get("limitations", [])
        if "dispatch validator source" not in item
    ]
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "failed_checks": failed,
                      "preserved_first_attempt_sha256": prior_sha,
                      "bank_binding_pass": bank_ok, "dispatch_script_hash_pass": dispatch_script_hash_ok,
                      "full_metric_recomputations_rerun": False,
                      "output": OUTPUT.relative_to(ROOT).as_posix()}, indent=2), flush=True)
    return 1 if failed else 0


def main() -> int:
    torch.set_num_threads(1)
    v = Validation()
    status = read_json(RUN / "status.json")
    aggregate = read_json(RUN / "aggregate.json")
    manifest = read_json(RUN / "manifest.json")
    training = read_json(RUN / "training.json")
    summary = read_json(RUN / "summary.json")
    reference_record_path = REFERENCE / "baseline_seed4.json"
    reference_record = read_json(reference_record_path)
    reference_manifest_path = REFERENCE / "manifest.json"
    reference_manifest = read_json(reference_manifest_path)
    old_public = read_json(PUBLIC_HISTORICAL)
    replay_reference_manifest = read_json(PUBLIC_REPLAY_REFERENCE)
    dispatch = read_json(DISPATCH_BINDINGS)

    completed = (
        status.get("status") == "COMPLETE" and status.get("completed_audits") == 44
        and aggregate.get("pass") is True and aggregate.get("execution_complete") is True
        and aggregate.get("completed_updates") == 300 and aggregate.get("completed_audits") == 44
        and training.get("completed_updates") == 300 and len(training.get("checkpoints", {})) == 23
        and summary.get("execution_complete") is True and summary.get("scientific_endpoint") is True
        and summary.get("diagnostic_only") is True
    )
    v.record("run_completion", completed, run_status=status.get("status"), updates=training.get("completed_updates"),
             audit_rows=status.get("completed_audits"), elapsed_seconds=status.get("elapsed_seconds"),
             aggregate_complete=aggregate.get("execution_complete"))

    expected_source = manifest.get("source_sha256", {})
    source_names = set(expected_source)
    snapshot_root = RUN / "source"
    snapshot_names = {p.relative_to(snapshot_root).as_posix() for p in snapshot_root.rglob("*") if p.is_file()}
    source_root_ok = len(expected_source) == 67 and all((ROOT / name).is_file() and sha256(ROOT / name) == digest for name, digest in expected_source.items())
    source_snapshot_ok = source_names == snapshot_names and all(
        (snapshot_root / name).is_file() and sha256(snapshot_root / name) == digest
        for name, digest in expected_source.items()
    )
    dispatch_sources_ok = dispatch.get("source_sha256") == expected_source
    v.record("source_and_snapshot_bindings", source_root_ok and source_snapshot_ok and dispatch_sources_ok,
             manifest_source_bindings=len(expected_source), live_source_hashes_checked=len(expected_source),
             snapshot_hashes_checked=len(expected_source), snapshot_file_count=len(snapshot_names),
             dispatch_source_manifest_exact=dispatch_sources_ok)

    ref_bindings = manifest.get("reference_bindings", {})
    ref_anchor_bindings = ref_bindings.get("anchors", {})
    ref_hash_ok = ref_bindings.get("record_sha256") == sha256(reference_record_path)
    all_param_tensors = 0
    anchor_tensor_equal = True
    checkpoint_file_hashes_ok = True
    checkpoint_parameter_hashes_ok = True
    nonfinite_checkpoint_tensors = 0
    parameter_hashes_recomputed = 0
    checkpoint_updates = sorted(int(key) for key in training.get("checkpoints", {}))
    expected_checkpoint_paths = {f"u{update:03}.pt" for update in CHECKPOINTS}
    saved_checkpoint_paths = {p.name for p in (RUN / "checkpoints").glob("u*.pt")}
    all_states: dict[int, dict[str, torch.Tensor]] = {}
    for update in CHECKPOINTS:
        item = training.get("checkpoints", {}).get(str(update), {})
        path = RUN / item.get("path", f"checkpoints/u{update:03}.pt")
        digest_ok = path.is_file() and sha256(path) == item.get("sha256")
        checkpoint_file_hashes_ok &= digest_ok
        if not digest_ok:
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", {})
        calculated_hash = tensor_hash(state)
        parameter_hashes_recomputed += 1
        hash_ok = calculated_hash == item.get("parameter_sha256")
        checkpoint_parameter_hashes_ok &= hash_ok
        if checkpoint.get("completed_updates") != update or checkpoint.get("seed") != 4:
            checkpoint_parameter_hashes_ok = False
        for value in state.values():
            if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
                nonfinite_checkpoint_tensors += 1
        all_param_tensors += len(state)
        if update in (0, 100, 200, 300):
            all_states[update] = state
            prior = reference_record.get("checkpoints", {}).get(str(update), {})
            reference_path = REFERENCE / prior.get("file", "")
            prior_file_ok = reference_path.is_file() and sha256(reference_path) == prior.get("sha256")
            ref_hash_ok &= prior_file_ok and ref_anchor_bindings.get(str(update)) == prior
            if not prior_file_ok:
                anchor_tensor_equal = False
                continue
            prior_checkpoint = torch.load(reference_path, map_location="cpu", weights_only=True)
            prior_state = prior_checkpoint.get("state_dict", {})
            tensor_keys_equal = set(state) == set(prior_state)
            tensor_values_equal = tensor_keys_equal and all(
                torch.equal(state[key], prior_state[key]) for key in state
            )
            anchor_tensor_equal &= tensor_values_equal and calculated_hash == prior.get("parameter_sha256")
    file_set_ok = saved_checkpoint_paths == expected_checkpoint_paths
    train_checkpoint_set_ok = set(checkpoint_updates) == set(CHECKPOINTS)
    checkpoint_count_ok = len(checkpoint_updates) == 23 and file_set_ok and train_checkpoint_set_ok
    v.record("checkpoint_hashes_and_anchors", checkpoint_count_ok and checkpoint_file_hashes_ok
             and checkpoint_parameter_hashes_ok and ref_hash_ok and anchor_tensor_equal
             and nonfinite_checkpoint_tensors == 0,
             saved_checkpoints=len(checkpoint_updates), checkpoint_files=len(saved_checkpoint_paths),
             file_hashes_checked=sum(bool(training.get("checkpoints", {}).get(str(update), {}).get("path")) for update in CHECKPOINTS),
             parameter_hashes_recomputed=parameter_hashes_recomputed,
             reference_anchor_updates=[0, 100, 200, 300], exact_anchor_tensor_sets=4 if anchor_tensor_equal else 0,
             parameter_tensor_arrays_checked=all_param_tensors, nonfinite_parameter_tensors=nonfinite_checkpoint_tensors,
             reference_record_sha256_matches=ref_hash_ok)

    # Recreate paired audit banks using the exact bank generator in the frozen source snapshot.
    tasks = load_module("_transition_saved_tasks", snapshot_root / "new/nca_inertial_wind_tunnel/tasks.py")
    bank_values: dict[int, dict[str, np.ndarray]] = {}
    bank_hashes: dict[str, Any] = {}
    banks_ok = True
    expected_bank_hashes: dict[str, str] = {}
    for size in SIZES:
        regenerated = tasks.bank(size, 16, 61000 + size, "cpu")
        expected_tensor_sha = tensor_hash(regenerated)
        expected_bank_hashes[str(size)] = expected_tensor_sha
        arrays = {key: value.detach().cpu().numpy() for key, value in regenerated.items()}
        bank_path = RUN / f"bank_size{size}.npz"
        with np.load(bank_path, allow_pickle=False) as stored_data:
            stored = {key: stored_data[key] for key in stored_data.files}
        content_equal = set(stored) == set(arrays) and all(arrays_equal(stored[key], arrays[key]) for key in arrays)
        binding = manifest.get("audit_banks", {}).get(str(size), {})
        binding_ok = binding.get("maps") == 16 and binding.get("seed") == 61000 + size and binding.get("tensor_sha256") == expected_tensor_sha
        file_digest = sha256(bank_path)
        dispatch_binding = dispatch.get("audit_banks", {}).get(str(size), {})
        dispatch_file_ok = dispatch_binding.get("file_sha256") == file_digest and dispatch_binding.get("tensor_sha256") == expected_tensor_sha
        banks_ok &= content_equal and binding_ok and dispatch_file_ok
        bank_values[size] = arrays
        bank_hashes[str(size)] = {"tensor_sha256_matches": expected_tensor_sha == binding.get("tensor_sha256"),
                                  "npz_content_matches_regenerated": content_equal,
                                  "file_sha256_matches_dispatch": dispatch_file_ok}
    history_data_match = reference_manifest.get("historical_evaluation_data_sha256") == dispatch.get("historical_evaluation_data_sha256")
    history_hash_match = reference_manifest.get("stage_diagnostic_data_sha256") == expected_bank_hashes["32"]
    public_history_match = replay_reference_manifest.get("stage_diagnostic_data_sha256") == expected_bank_hashes["32"]
    v.record("audit_banks_regenerated", banks_ok and history_data_match and history_hash_match and public_history_match,
             sizes_checked=list(SIZES), maps_per_size=16, generator="frozen source snapshot CPU bank()",
             bank_hash_checks=bank_hashes, historical_evaluation_bindings_match=history_data_match,
             size32_stage_bank_matches_reference=history_hash_match and public_history_match)

    schedule_path = RUN / "schedule.json"
    schedule_digest = sha256(schedule_path)
    schedule_binding_ok = schedule_digest == manifest.get("schedule_sha256") == ref_bindings.get("schedule_sha256") == reference_record.get("schedule_sha256")
    train_data_binding_ok = manifest.get("train_data_sha256") == ref_bindings.get("train_data_sha256") == reference_record.get("train_data_sha256") == reference_manifest.get("train_data_sha256")
    v.record("training_input_bindings", schedule_binding_ok and train_data_binding_ok,
             schedule_file_hash_matches_all_bindings=schedule_binding_ok,
             generated_training_data_hash_matches_reference=train_data_binding_ok)

    # Verify full historical evaluation against the published original payload, preserving exact zero error.
    history_comparison = v.compare(old_public.get("evaluation"), read_json(RUN / "historical_replay.json").get("evaluation"),
                                   tolerance=0.0, label="historical_replay.evaluation")
    replay_record = read_json(RUN / "historical_replay.json")
    historical_endpoints = {f"{size}:{step}" for size in ("32", "64") for step in ("64", "128", "256")}
    actual_endpoints = {f"{size}:{step}" for size, steps in replay_record.get("evaluation", {}).items() for step in steps}
    stored_counts = replay_record.get("counts", {})
    replay_zero = history_comparison["maximum_absolute_error"] == 0 and history_comparison["ok"]
    replay_counts_match = all(stored_counts.get(key) == history_comparison.get(key) for key in ("integer_leaves", "float_leaves"))
    replay_counts_match &= stored_counts.get("maximum_absolute_error") == history_comparison["maximum_absolute_error"] == 0
    v.record("historical_replay_exact", replay_record.get("status") == "PASS" and replay_zero and replay_counts_match
             and actual_endpoints == historical_endpoints,
             endpoint_payloads=len(actual_endpoints), integer_leaves=history_comparison["integer_leaves"],
             float_leaves=history_comparison["float_leaves"], maximum_absolute_error=history_comparison["maximum_absolute_error"],
             exact_zero_error=replay_zero and replay_counts_match)

    metrics = load_module("_transition_saved_metrics", snapshot_root / "new/transition_100_200/metrics.py")
    raw_npz_paths = sorted((RUN / "checkpoints").glob("u*_size*.npz"))
    audit_json_paths = sorted((RUN / "checkpoints").glob("u*_size*.json"))
    expected_audit_json_names = {f"u{update:03}_size{size}.json" for update in UPDATES for size in SIZES}
    expected_audit_npz_names = {f"u{update:03}_size{size}.npz" for update in UPDATES for size in SIZES}
    artifacts_set_ok = {path.name for path in audit_json_paths} == expected_audit_json_names
    artifacts_set_ok &= {path.name for path in raw_npz_paths} == expected_audit_npz_names
    v.record("audit_artifact_set", artifacts_set_ok, expected_records=44, audit_json_count=len(audit_json_paths), raw_npz_count=len(raw_npz_paths))

    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    audit_count = 0
    for update in UPDATES:
        for size in SIZES:
            row = audit_one(v, metrics, update, size, bank_values[size], expected_bank_hashes[str(size)])
            if row is not None:
                rows_by_key[(update, size)] = row
                audit_count += 1
        print(f"validated saved audits at update {update} ({audit_count}/44)", flush=True)

    summary_rows = summary.get("checkpoints", [])
    profile_rows = summary.get("descriptive_screen_profiles", [])
    summary_rows_ok = len(summary_rows) == 44 and len(profile_rows) == 44
    summary_row_mismatches = 0
    for update in UPDATES:
        for size in SIZES:
            saved = rows_by_key.get((update, size))
            if saved is None:
                summary_rows_ok = False
                continue
            projected = {key: value for key, value in saved.items() if key not in ("state_rms", "behavior")}
            matching = next((item for item in summary_rows if item.get("update") == update and item.get("size") == size), None)
            if matching is None or not v.compare(projected, matching, label=f"summary.checkpoint.{update}.{size}")["ok"]:
                summary_row_mismatches += 1
                summary_rows_ok = False
    v.record("summary_checkpoint_rows", summary_rows_ok,
             checkpoint_rows=len(summary_rows), expected=44, mismatches=summary_row_mismatches)

    derived_profiles = []
    for update in UPDATES:
        for size in SIZES:
            row = rows_by_key.get((update, size))
            if row is None:
                continue
            interval = row.get("behavior_profile", {}).get("64_128", {})
            long = row.get("behavior_profile", {}).get("64_256", {})
            strict128 = row.get("endpoints", {}).get("128", {}).get("strict_coverage", {})
            checks = {
                "strict128_pooled": strict128.get("pooled_rate") is not None and strict128["pooled_rate"] >= .80,
                "strict128_equal_map": strict128.get("equal_map_mean") is not None and strict128["equal_map_mean"] >= .80,
                "g64_128": interval.get("acquisition", {}).get("pooled_rate") is not None and interval["acquisition"]["pooled_rate"] >= .20,
                "first_exit64_128": interval.get("first_exit", {}).get("pooled_rate") is not None and interval["first_exit"]["pooled_rate"] <= .01,
                "survival64_256": long.get("continuous_survival", {}).get("pooled_rate") is not None and long["continuous_survival"]["pooled_rate"] >= .95,
            }
            derived_profiles.append({"update": update, "size": size, "checks": checks,
                                     "descriptive_screen": all(checks.values()),
                                     "prespecified_primary_screen": size == 32,
                                     "size64_is_secondary_projection": size == 64})
    profiles_comp = v.compare(derived_profiles, profile_rows, label="summary.descriptive_screen_profiles")
    primary_dense = sorted((item for item in derived_profiles if item["size"] == 32 and 100 <= item["update"] <= 200),
                           key=lambda item: item["update"])
    onset = None
    for first, second, third in zip(primary_dense, primary_dense[1:], primary_dense[2:]):
        if (second["update"] == first["update"] + 5 and third["update"] == second["update"] + 5
                and first["descriptive_screen"] and second["descriptive_screen"] and third["descriptive_screen"]):
            onset = {"first_saved_update": first["update"], "confirmed_through": third["update"],
                     "resolution_updates": 5, "causal_or_phase_transition_claim": False}
            break
    u300_controls = [item for item in derived_profiles if item["update"] == 300]
    onset_check = summary.get("operational_onset") == onset
    dense_grid_ok = len(primary_dense) == 21 and all(100 <= item["update"] <= 200 for item in primary_dense)
    u300_control_ok = len(u300_controls) == 2 and all(item["update"] == 300 for item in u300_controls)
    none_not_enough = onset is None and len([item for item in primary_dense if item["descriptive_screen"]]) < 3
    v.record("profiles_and_operational_onset", profiles_comp["ok"] and onset_check and dense_grid_ok and u300_control_ok and none_not_enough,
             profile_rows=len(derived_profiles), primary_dense_profiles=len(primary_dense),
             primary_dense_passes=sum(item["descriptive_screen"] for item in primary_dense),
             saved_onset=summary.get("operational_onset"), derived_onset=onset,
             u300_control_rows=len(u300_controls), u300_excluded_from_dense_search=True)

    # Recompute the frozen warmstart pure-correctness T0..128 summaries at its three anchors.
    warmdiag = load_module("_transition_saved_warmdiag", snapshot_root / "new/warmstart/diagnostics.py")
    stage_results = []
    stage_ok = True
    for update in (100, 200, 300):
        row = rows_by_key.get((update, 32))
        diagnostic_path = REFERENCE / f"baseline_seed4_u{update:03}_diagnostic.json"
        reference_diag = read_json(diagnostic_path)
        actual_reference_hash = sha256(diagnostic_path)
        dispatch_stage = dispatch.get("reference_stage_diagnostics", {}).get(str(update), {})
        parameter_sha = training.get("checkpoints", {}).get(str(update), {}).get("parameter_sha256")
        ref_stage_binding_ok = (dispatch_stage.get("sha256") == actual_reference_hash
                                and dispatch_stage.get("checkpoint_parameter_sha256") == parameter_sha
                                and dispatch_stage.get("sha256") == row.get("historical_stage_replay", {}).get("reference_sha256")) if row else False
        stage_comparison: dict[str, Any] = {"ok": False}
        if row is not None:
            with np.load(RUN / "checkpoints" / f"u{update:03}_size32.npz", allow_pickle=False) as data:
                reproduced = warmdiag.summarize_correct_traces(data["correct"][:129], data["changed"])
            fields = ("changed_pixels_per_map", "steps", "survival", "coverage_gain", "relapse")
            comparisons = {field: v.compare(reference_diag[field], reproduced[field], tolerance=0.0,
                                             label=f"stage{update}.{field}") for field in fields}
            stage_comparison = {"ok": all(item["ok"] for item in comparisons.values()),
                                "maximum_absolute_error": max(item["maximum_absolute_error"] for item in comparisons.values()),
                                "integer_leaves": sum(item["integer_leaves"] for item in comparisons.values()),
                                "float_leaves": sum(item["float_leaves"] for item in comparisons.values())}
        wrapper = row.get("historical_stage_replay", {}) if row else {}
        row_wrapper_ok = wrapper.get("status") == "PASS" and wrapper.get("reference_sha256") == actual_reference_hash
        stage_pass = bool(row is not None and stage_comparison["ok"] and ref_stage_binding_ok and row_wrapper_ok
                          and wrapper.get("comparisons", {}).get("maximum_absolute_error") == 0)
        stage_ok &= stage_pass
        stage_results.append({"update": update, "status": "PASS" if stage_pass else "FAIL",
                              "reference_sha256_matches_dispatch": ref_stage_binding_ok,
                              "exact_replay": stage_comparison["ok"],
                              "integer_leaves": stage_comparison.get("integer_leaves", 0),
                              "float_leaves": stage_comparison.get("float_leaves", 0),
                              "maximum_absolute_error": stage_comparison.get("maximum_absolute_error"),
                              "runner_wrapper_matches": row_wrapper_ok})
    v.record("historical_stage_pure_trace_replay", stage_ok, checkpoints=stage_results, trace_steps=129)

    dispatch_validator = ROOT / "tools/validate_transition_dispatch.py"
    dispatch_script_hash_ok = dispatch_validator.is_file() and sha256(dispatch_validator) == dispatch.get("validation_script_sha256")
    historical_data_hash_ok = dispatch.get("historical_evaluation_data_sha256") == reference_manifest.get("historical_evaluation_data_sha256")
    manifest_binding_checks = (
        dispatch.get("status") == "PASS" and dispatch.get("training") is False
        and dispatch.get("gpu_execution") is False
        and dispatch.get("manifest_sha256") == sha256(RUN / "manifest.json")
        and dispatch_sources_ok
        and dispatch.get("historical_evaluation_data_sha256") == reference_manifest.get("historical_evaluation_data_sha256")
        and dispatch.get("reference_manifest_sha256") == sha256(reference_manifest_path)
        and dispatch.get("audit_banks", {}).keys() == manifest.get("audit_banks", {}).keys()
        and dispatch.get("reference_stage_diagnostics", {}).keys() == {"100", "200", "300"}
        and dispatch_script_hash_ok and historical_data_hash_ok
    )
    v.record("postdispatch_bindings", manifest_binding_checks,
             dispatch_status=dispatch.get("status"), training_false=dispatch.get("training") is False,
             gpu_execution_false=dispatch.get("gpu_execution") is False,
             manifest_hash_matches=dispatch.get("manifest_sha256") == sha256(RUN / "manifest.json"),
             reference_manifest_hash_matches=dispatch.get("reference_manifest_sha256") == sha256(reference_manifest_path),
             source_hashes_match=dispatch_sources_ok,
             dispatch_validator_source_hash_independently_bound=dispatch_script_hash_ok,
             dispatch_validator_source_hash_matches=dispatch_script_hash_ok,
             historical_evaluation_data_bindings_match=historical_data_hash_ok)

    limitations = [
        "State-RMS scalars are checked for finite, nonnegative values and all 257 recorded steps; hidden-state tensors were not saved, so RMS values cannot be independently recomputed.",
        "Original and flipped BCE scalars are checked for finite nonnegative values; logits were not saved, so BCE cannot be recomputed.",
        "The saved paired signed margin is checked for finiteness, sign consistency, and endpoint quantiles; separate original/flipped logits were not saved, so the margin construction itself cannot be rebuilt.",
        "The two-hop cone check verifies paired correctness outside the cone; raw logits and hidden states were not saved, so logit-difference bounds outside the cone cannot be independently checked.",
    ]
    report = {
        "schema_version": "transition-dense-saved-validation-v1",
        "status": "FAIL" if v.failures else "PASS",
        "scope": "Saved artifacts only; CPU validation; no model forward, training, GPU execution, or new inference.",
        "run": "runs/transition_20261003_seed4_dense01",
        "validator": "tools/validate_transition_results.py",
        "validator_sha256": sha256(Path(__file__)),
        "checks": v.checks,
        "counters": {
            "audit_records_expected": 44,
            "audit_records_processed": audit_count,
            "raw_npz_sha256_checked": sum(name.startswith("audit_raw_hash_") and item["status"] == "PASS" for name, item in v.checks.items()),
            "source_bindings": len(expected_source),
            "source_snapshots": len(snapshot_names),
            "saved_checkpoints": len(checkpoint_updates),
            "historical_replay_integer_leaves": history_comparison["integer_leaves"],
            "historical_replay_float_leaves": history_comparison["float_leaves"],
            "historical_replay_maximum_absolute_error": history_comparison["maximum_absolute_error"],
            "metric_comparison_leaves": v.compare_counts,
            "metric_comparison_maximum_absolute_error": v.max_float_error,
            "metric_comparison_tolerance": TOL,
        },
        "limitations": limitations,
        "failed_checks": v.failures,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(v.checks), "failed_checks": v.failures,
                      "output": OUTPUT.relative_to(ROOT).as_posix(),
                      "max_metric_error": v.max_float_error}, indent=2), flush=True)
    return 1 if v.failures else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--finalize-from-prior"]:
        raise SystemExit(finalize_saved_full_pass())
    raise SystemExit(main())
