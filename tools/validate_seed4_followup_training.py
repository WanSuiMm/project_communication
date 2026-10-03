"""CPU-only audit of the saved seed4 A/B follow-up training artifacts.

This validator does not instantiate a CUDA device or rerun model inference. It
checks saved provenance/checkpoints and independently derives behavior metrics
from each compressed Boolean trace and the frozen task banks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/seed4_followup_20261003_training01"
DEFAULT_OUT = ROOT / "analyses/seed4_followup_20261003_training_review01"
HISTORICAL_RAW = ROOT / "evidence/streaming_carry_init2345/raw/stream_K8_seed4.json"
HISTORICAL_RUN = ROOT / "runs/streaming_carry_20261002_init2345"
PREFLIGHT_RUN = ROOT / "runs/seed4_followup_20261003_preflight01"
CPU_QUALIFICATION = ROOT / "analyses/seed4_followup_20261003_cpu01.json"
FLOAT_TOL = 2e-7


def _setup_imports() -> None:
    for relative in (
        "new/nca_inertial_wind_tunnel",
        "new/workspace_revision",
        "new/seed4_followup",
    ):
        path = str(ROOT / relative)
        if path not in sys.path:
            sys.path.insert(0, path)


_setup_imports()
from tasks import bank  # noqa: E402
from run_revision import tensor_hash  # noqa: E402
import initialization  # noqa: E402
import phenotype  # noqa: E402


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any = None) -> bool:
        row = {"name": name, "pass": bool(passed)}
        if detail is not None:
            row["detail"] = detail
        self.checks.append(row)
        if not passed:
            self.failures.append(row)
        return bool(passed)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _close(a: Any, b: Any, tol: float = FLOAT_TOL) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    return a == b


def _tree_mismatches(actual: Any, expected: Any, path: str = "", limit: int = 12) -> list[str]:
    mismatch: list[str] = []
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            mismatch.append(f"{path or '<root>'}: keys differ")
        for key in sorted(set(actual) & set(expected), key=str):
            mismatch.extend(_tree_mismatches(actual[key], expected[key], f"{path}/{key}", limit))
            if len(mismatch) >= limit:
                break
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            mismatch.append(f"{path}: list lengths {len(actual)} != {len(expected)}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            mismatch.extend(_tree_mismatches(left, right, f"{path}/{index}", limit))
            if len(mismatch) >= limit:
                break
    elif not _close(actual, expected):
        mismatch.append(f"{path}: {actual!r} != {expected!r}")
    return mismatch[:limit]


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _bank_hash(data: dict[str, Any]) -> str:
    return tensor_hash(data)


def _expected_specs() -> dict[str, dict[str, Any]]:
    expected = {
        "control": {"name": "control", "experiment": "control", "epsilon": 0.0,
                    "direction_seed": None, "schedule_seed": 20002}
    }
    for seed in (20012, 20022, 20032, 20042):
        name = f"A_schedule{seed}"
        expected[name] = {"name": name, "experiment": "A", "epsilon": 0.0,
                          "direction_seed": None, "schedule_seed": seed}
    for direction in range(70002, 70006):
        for epsilon, suffix in ((0.01, "01"), (0.05, "05")):
            name = f"B_direction{direction}_eps{suffix}"
            expected[name] = {"name": name, "experiment": "B", "epsilon": epsilon,
                              "direction_seed": direction, "schedule_seed": 20002}
    return expected


def _trace_arrays(path: Path, size: int) -> dict[str, np.ndarray]:
    names = ("correct", "original_correct", "flipped_correct")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(names):
            raise ValueError(f"{path.name}: trace keys are {sorted(archive.files)}")
        arrays = {name: archive[name] for name in names}
    wanted = (257, 32, size, size)
    for name, array in arrays.items():
        if array.shape != wanted or array.dtype != np.bool_:
            raise ValueError(f"{path.name}:{name} expected bool{wanted}, got {array.dtype}{array.shape}")
    if not np.array_equal(arrays["correct"], arrays["original_correct"] & arrays["flipped_correct"]):
        raise ValueError(f"{path.name}: paired trace differs from original AND flipped correctness")
    return arrays


def _balanced_accuracy(correct: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Match per-example balanced accuracy without using saved endpoint records."""
    target = np.asarray(target, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    positive = mask & target
    negative = mask & ~target
    pos_n = positive.sum(axis=(1, 2), dtype=np.int64)
    neg_n = negative.sum(axis=(1, 2), dtype=np.int64)
    pos_hit = (correct & positive).sum(axis=(1, 2), dtype=np.int64)
    neg_hit = (correct & negative).sum(axis=(1, 2), dtype=np.int64)
    pos_recall = pos_hit / np.maximum(pos_n, 1)
    neg_recall = neg_hit / np.maximum(neg_n, 1)
    present = (pos_n > 0).astype(np.int64) + (neg_n > 0).astype(np.int64)
    per_map = (pos_recall + neg_recall) / np.maximum(present, 1)
    return float(per_map.mean())


def _records_from_traces(traces: dict[str, dict[str, np.ndarray]], data_cpu: dict[int, dict[str, Any]]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for size in (32, 64):
        bank_data = data_cpu[size]
        mask = _numpy(bank_data["mask"])[:, 0] > 0.5
        original_y = _numpy(bank_data["y"])[:, 0] >= 0.5
        flipped_y = _numpy(bank_data["y_flip"])[:, 0] >= 0.5
        trace = traces[str(size)]
        records[str(size)] = {}
        for step in (64, 128, 256):
            records[str(size)][str(step)] = {
                "original": {"balanced_accuracy": _balanced_accuracy(trace["original_correct"][step], original_y, mask)},
                "flipped": {"balanced_accuracy": _balanced_accuracy(trace["flipped_correct"][step], flipped_y, mask)},
            }
    return records


def _matched_csv_audit(path: Path, expected_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    columns = (
        "size", "map_index", "start_step", "end_step", "distance",
        "frontier_opportunities", "frontier_acquired", "nonfrontier_opportunities",
        "nonfrontier_acquired", "matched_weight", "frontier_rate",
        "nonfrontier_rate", "rate_difference",
    )
    mismatches: list[str] = []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            mismatches.append("CSV columns differ from frozen matched-strata schema")
        for line, raw in enumerate(reader, 2):
            row: dict[str, Any] = {}
            for key in columns[:9]:
                row[key] = int(raw[key])
            for key in columns[9:]:
                row[key] = float(raw[key]) if raw[key] else None
            rows.append(row)
    if len(rows) != len(expected_rows):
        mismatches.append(f"CSV row count {len(rows)} != trace-derived {len(expected_rows)}")
    for index, (actual, expected) in enumerate(zip(rows, expected_rows)):
        for key in columns:
            if not _close(actual[key], expected[key], 1e-10):
                mismatches.append(f"row {index} {key}: {actual[key]!r} != {expected[key]!r}")
                if len(mismatches) >= 12:
                    break
        if len(mismatches) >= 12:
            break

    grouped: dict[int, dict[int, list[tuple[float, float]]]] = {
        32: defaultdict(list), 64: defaultdict(list)
    }
    seen = set()
    invalid_count = 0
    for row in rows:
        key = (row["size"], row["map_index"], row["start_step"], row["distance"])
        if key in seen:
            mismatches.append(f"duplicate matched stratum {key}")
        seen.add(key)
        nf, af = row["frontier_opportunities"], row["frontier_acquired"]
        nn, an = row["nonfrontier_opportunities"], row["nonfrontier_acquired"]
        if not (nf > 0 and nn > 0 and 0 <= af <= nf and 0 <= an <= nn):
            invalid_count += 1
            continue
        weight = nf * nn / (nf + nn)
        difference = af / nf - an / nn
        if not _close(row["matched_weight"], weight, 1e-10) or not _close(row["rate_difference"], difference, 1e-10):
            mismatches.append(f"CSV rates/weight inconsistent with integer counts at {key}")
        grouped[row["size"]][row["map_index"]].append((weight, difference))
    if invalid_count:
        mismatches.append(f"{invalid_count} rows have empty or impossible acquisition counts")

    by_size: dict[str, Any] = {}
    for size in (32, 64):
        by_map = []
        for map_index in range(32):
            terms = grouped[size].get(map_index, [])
            denominator = sum(weight for weight, _ in terms)
            effect = sum(weight * difference for weight, difference in terms) / denominator if denominator else None
            by_map.append({"map_index": map_index, "common_strata": len(terms),
                           "weight_sum": denominator if denominator else None,
                           "weighted_difference": effect})
        eligible = [row["weighted_difference"] for row in by_map if row["weighted_difference"] is not None]
        by_size[str(size)] = {
            "csv_rows": len([row for row in rows if row["size"] == size]),
            "common_strata": sum(row["common_strata"] for row in by_map),
            "eligible_maps": len(eligible),
            "mean_map_weighted_difference": float(np.mean(eligible)) if eligible else None,
            "per_map": by_map,
        }
    return by_size, mismatches[:20]


def _trace_counts(summary: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for size in (32, 64):
        row = summary["sizes"][str(size)]
        count_first_observed = sum(item["first_correct"]["observed"] for item in row["per_map"])
        count_first_censored = sum(item["first_correct"]["censored"] for item in row["per_map"])
        count_stable_observed = sum(item["terminal_stable_through256"]["observed"] for item in row["per_map"])
        count_stable_censored = sum(item["terminal_stable_through256"]["censored"] for item in row["per_map"])
        size_result: dict[str, Any] = {
            "maps": row["maps"],
            "changed_pixels": row["changed_pixels"],
            "first_correct_observed_pixels": count_first_observed,
            "first_correct_right_censored_pixels": count_first_censored,
            "terminal_stable_pixels": count_stable_observed,
            "not_terminal_correct_or_stable_pixels": count_stable_censored,
            "ever_correct_pixels": row["ever_regressed_over_ever_correct"]["denominator"],
            "ever_relapsed_pixels": row["ever_regressed_over_ever_correct"]["numerator"],
            "ever_relapsed_over_ever_correct": row["ever_regressed_over_ever_correct"]["value"],
            "endpoints": {},
            "transitions": {
                kind: {
                    pair: {
                        "from_correct": transition["pooled"]["from_correct"],
                        "retained": transition["pooled"]["retained"],
                        "lost": transition["pooled"]["lost"],
                        "gained": transition["pooled"]["gained"],
                        "retention": transition["pooled"]["retention"],
                        "relapse": transition["pooled"]["relapse"],
                        "mean_map_retention": transition["mean_map_retention"],
                        "eligible_maps_for_retention": transition["eligible_maps_for_retention"],
                    }
                    for pair, transition in pairs.items()
                }
                for kind, pairs in row["transitions"].items()
            },
            "distance_profile": [
                {"distance": item["distance"], "pixels": item["pixels"],
                 "ever_correct_pixels": item["ever_correct_pixels"],
                 "never_correct_or_right_censored_pixels": item["never_correct_pixels"],
                 "ever_relapsed_pixels": item["ever_regressed"]["numerator"],
                 "first_correct_observed": item["first_correct"]["observed"],
                 "first_correct_censored": item["first_correct"]["censored"],
                 "terminal_stable": item["terminal_stable_through256"]["observed"],
                 "terminal_stable_censored": item["terminal_stable_through256"]["censored"]}
                for item in row["distance_profile"]
            ],
        }
        for step in (64, 128, 256):
            endpoint = row["endpoints"][str(step)]
            acquisition = row["acquisition"][str(step)]
            size_result["endpoints"][str(step)] = {
                "original_ba": endpoint["original_ba"],
                "flipped_ba": endpoint["flipped_ba"],
                "all_changed_correct": endpoint["all_changed"]["correct_pixels"],
                "all_changed_total": endpoint["all_changed"]["pixels"],
                "all_changed_coverage_pooled": endpoint["all_changed"]["pooled_coverage"]["value"],
                "strict_16_32_correct": endpoint["strict_16_32"]["correct_pixels"],
                "strict_16_32_total": endpoint["strict_16_32"]["pixels"],
                "strict_16_32_coverage_pooled": endpoint["strict_16_32"]["pooled_coverage"]["value"],
                "strict_16_32_coverage_mean_map": endpoint["strict_16_32"]["mean_map_coverage"],
                "newly_correct_vs_step0_pixels": acquisition["newly_correct_vs_step0_pixels"],
                "newly_correct_since64_pixels": acquisition["newly_correct_since64_pixels"],
                "lost_since64_pixels": acquisition["lost_since64_pixels"],
            }
        output[str(size)] = size_result
    return output


def _compare_saved_summary(audit: Audit, name: str, recomputed: dict[str, Any], saved: dict[str, Any], gate: dict[str, Any], record: dict[str, Any]) -> None:
    detail = _tree_mismatches(recomputed["sizes"], saved.get("sizes"))
    audit.check(f"{name}: raw-trace metrics match saved summary", not detail, detail or {"float_tolerance": FLOAT_TOL})
    project_gate = lambda value: {key: value.get(key) for key in ("pass", "checks", "interpretation")}
    gate_detail = _tree_mismatches(project_gate(gate), project_gate(saved.get("phenotype_gate", {})))
    audit.check(f"{name}: independent phenotype gate matches saved summary", not gate_detail,
                gate_detail or {"pass": gate["pass"]})
    record_detail = _tree_mismatches(project_gate(gate), project_gate(record.get("phenotype_gate", {})))
    audit.check(f"{name}: independent phenotype gate matches arm record", not record_detail,
                record_detail or {"pass": gate["pass"]})


def _verify_checkpoint(audit: Audit, run: Path, record: dict[str, Any], expected_spec: dict[str, Any]) -> dict[str, Any]:
    name = record["name"]
    path = run / f"{name}.pt"
    actual_hash = _sha(path)
    file_ok = actual_hash == record.get("checkpoint_file_sha256")
    audit.check(f"{name}: checkpoint file SHA256", file_ok,
                {"actual": actual_hash, "recorded": record.get("checkpoint_file_sha256")})
    # weights_only=True and map_location=cpu ensure only the tensor state is read;
    # no model forward, CUDA context, or checkpoint pickle execution occurs.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"{name}: checkpoint has no tensor-only state_dict")
    state_hash = tensor_hash(state)
    state_ok = state_hash == record.get("final_parameter_sha256")
    audit.check(f"{name}: checkpoint state parameter SHA256", state_ok,
                {"actual": state_hash, "recorded": record.get("final_parameter_sha256")})
    count = sum(int(value.numel()) for value in state.values())
    audit.check(f"{name}: state parameter count", count == 5033,
                {"actual": count, "expected": 5033})
    checkpoint_ok = (checkpoint.get("completed_updates") == 300 and
                     checkpoint.get("experiment") == expected_spec["experiment"] and
                     checkpoint.get("epsilon") == expected_spec["epsilon"] and
                     checkpoint.get("direction_seed") == expected_spec["direction_seed"] and
                     checkpoint.get("schedule_seed") == expected_spec["schedule_seed"] and
                     checkpoint.get("variant") == "stream" and not checkpoint.get("partial", False))
    audit.check(f"{name}: checkpoint metadata", checkpoint_ok,
                {"completed_updates": checkpoint.get("completed_updates"),
                 "variant": checkpoint.get("variant"), "partial": bool(checkpoint.get("partial", False))})
    return {"file_sha256": actual_hash, "state_parameter_sha256": state_hash,
            "file_sha_match": file_ok, "state_sha_match": state_ok,
            "parameter_count": count}


def validate(run: Path, out: Path) -> dict[str, Any]:
    audit = Audit()
    report: dict[str, Any] = {
        "schema": "seed4-followup-training-saved-result-review-v1",
        "run_id": run.name,
        "validation_mode": "CPU saved-artifact review; no model inference or training",
        "training_status": {},
        "provenance": {},
        "arms": [],
        "aggregate": {},
        "checks": audit.checks,
        "failures": audit.failures,
    }
    manifest = _read(run / "manifest.json")
    aggregate = _read(run / "aggregate.json")
    status = _read(run / "status.json")
    report["training_status"] = {key: status.get(key) for key in ("status", "completed_arms", "elapsed_seconds", "finished_utc")}

    expected_status = status.get("status") == "COMPLETE" and status.get("completed_arms") == 13
    audit.check("training run completed 13/13", expected_status, report["training_status"])
    audit.check("training aggregate complete 13/13 at 300 updates",
                aggregate.get("complete") is True and aggregate.get("completed_arms") == 13 and
                aggregate.get("expected_arms") == 13 and aggregate.get("updates_per_arm") == 300,
                {"complete": aggregate.get("complete"), "completed_arms": aggregate.get("completed_arms"),
                 "expected_arms": aggregate.get("expected_arms"), "updates_per_arm": aggregate.get("updates_per_arm")})
    audit.check("training aggregate identity flag", aggregate.get("identities_verified") is True)
    audit.check("protocol and launch metadata", manifest.get("protocol") == "seed4_independent_followup_v1" and
                manifest.get("training") is True and manifest.get("preflight") is False and
                manifest.get("updates_per_arm") == 300 and manifest.get("maximum_seconds") == 2400 and
                manifest.get("maps_per_size") == 32 and manifest.get("fresh_evaluation_seeds") == [50032, 50064],
                {"protocol": manifest.get("protocol"), "updates_per_arm": manifest.get("updates_per_arm"),
                 "maximum_seconds": manifest.get("maximum_seconds"), "fresh_evaluation_seeds": manifest.get("fresh_evaluation_seeds")})

    source_hashes = manifest.get("source_sha256", {})
    snapshot_files = [p for p in (run / "source").rglob("*") if p.is_file()]
    audit.check("frozen source manifest has 56 entries", len(source_hashes) == 56,
                {"source_hashes": len(source_hashes), "source_snapshots": len(snapshot_files)})
    source_results = []
    for relative, expected_hash in sorted(source_hashes.items()):
        current = ROOT / Path(relative)
        snapshot = run / "source" / Path(relative)
        current_hash = _sha(current) if current.is_file() else None
        snapshot_hash = _sha(snapshot) if snapshot.is_file() else None
        current_ok = current_hash == expected_hash
        snapshot_ok = snapshot_hash == expected_hash
        audit.check(f"source current:{relative}", current_ok,
                    {"expected": expected_hash, "actual": current_hash})
        audit.check(f"source snapshot:{relative}", snapshot_ok,
                    {"expected": expected_hash, "actual": snapshot_hash})
        source_results.append({"path": relative, "expected_sha256": expected_hash,
                               "current_matches": current_ok, "snapshot_matches": snapshot_ok})
    audit.check("source snapshot contains exactly the frozen manifest files",
                len(snapshot_files) == len(source_hashes),
                {"snapshot_files": len(snapshot_files), "manifest_files": len(source_hashes)})
    report["provenance"]["sources"] = {"count": len(source_hashes), "snapshot_count": len(snapshot_files),
                                        "all_current_and_snapshot_hashes_match": all(
                                            item["current_matches"] and item["snapshot_matches"] for item in source_results),
                                        "files": source_results}

    # Verify preflight and focused CPU qualification artifacts bound by training.
    q_binding = manifest.get("cpu_qualification", {})
    qpath = ROOT / Path(q_binding.get("path", ""))
    q_actual_hash = _sha(qpath) if qpath.is_file() else None
    qdoc = _read(qpath) if qpath.is_file() else {}
    q_ok = q_actual_hash == q_binding.get("sha256") and qdoc.get("status") == "PASS" and qdoc.get("training") is False and qdoc.get("gpu_execution") is False
    audit.check("CPU qualification hash/status binding", q_ok,
                {"expected_sha256": q_binding.get("sha256"), "actual_sha256": q_actual_hash,
                 "status": qdoc.get("status"), "training": qdoc.get("training"), "gpu_execution": qdoc.get("gpu_execution")})
    audit.check("CPU qualification source bindings equal training run bindings",
                qdoc.get("source_sha256") == source_hashes,
                {"qualification_sources": len(qdoc.get("source_sha256", {})), "training_sources": len(source_hashes)})
    cpu_checks = qdoc.get("checks", {})
    audit.check("CPU training, phenotype, and causal qualification checks passed",
                all(cpu_checks.get(key, {}).get("status") == "PASS" for key in ("training", "phenotype", "causal")),
                {key: cpu_checks.get(key, {}).get("status") for key in ("training", "phenotype", "causal")})

    pf_status_path = PREFLIGHT_RUN / "status.json"
    pf_manifest_path = PREFLIGHT_RUN / "manifest.json"
    pf_aggregate_path = PREFLIGHT_RUN / "aggregate.json"
    pf_files = {"status.json": pf_status_path, "manifest.json": pf_manifest_path, "aggregate.json": pf_aggregate_path}
    pf_hashes = {name: (_sha(path) if path.is_file() else None) for name, path in pf_files.items()}
    recorded_pf_hashes = manifest.get("preflight_sha256") or {}
    audit.check("preflight three-file binding hashes", recorded_pf_hashes == pf_hashes,
                {"recorded": recorded_pf_hashes, "actual": pf_hashes})
    pf_doc = _read(pf_manifest_path) if pf_manifest_path.is_file() else {}
    pf_agg = _read(pf_aggregate_path) if pf_aggregate_path.is_file() else {}
    pf_status = _read(pf_status_path) if pf_status_path.is_file() else {}
    audit.check("preflight passed and is source-bound to formal run",
                pf_status.get("status") == "PREFLIGHT_PASSED" and pf_agg.get("decision") == "PREFLIGHT_PASSED" and
                pf_agg.get("updates_per_arm") == 300 and pf_agg.get("identities_verified") is True and
                pf_doc.get("source_sha256") == source_hashes,
                {"status": pf_status.get("status"), "decision": pf_agg.get("decision"),
                 "source_bindings": len(pf_doc.get("source_sha256", {}))})
    audit.check("preflight and formal run share the same CPU qualification receipt",
                pf_doc.get("cpu_qualification") == q_binding,
                {"formal": q_binding, "preflight": pf_doc.get("cpu_qualification")})
    report["provenance"]["cpu_qualification"] = {"path": q_binding.get("path"), "sha256": q_actual_hash,
                                                  "status": qdoc.get("status"), "checks": {
                                                      key: cpu_checks.get(key, {}).get("status")
                                                      for key in ("training", "phenotype", "causal")}}
    report["provenance"]["preflight"] = {"run_id": PREFLIGHT_RUN.name, "status": pf_status.get("status"),
                                          "decision": pf_agg.get("decision"), "file_sha256": pf_hashes}

    # Historical control reference is bound by the 56-file frozen manifest.
    old = _read(HISTORICAL_RAW)
    old_run_path = HISTORICAL_RUN / "stream_K8_seed4.json"
    old_manifest_path = HISTORICAL_RUN / "manifest.json"
    old_file_hash = _sha(HISTORICAL_RAW)
    old_run_hash = _sha(old_run_path) if old_run_path.is_file() else None
    audit.check("historical reference hash binding", old_file_hash == manifest.get("historical_reference_sha256") and
                old_run_hash == old_file_hash,
                {"expected": manifest.get("historical_reference_sha256"), "raw_sha256": old_file_hash,
                 "historical_run_sha256": old_run_hash})
    old_manifest = _read(old_manifest_path)
    audit.check("historical/fresh data seeds remain disjoint and recorded",
                manifest.get("historical_evaluation_data_sha256") == old_manifest.get("evaluation_data_sha256") and
                manifest.get("fresh_evaluation_seeds") == [50032, 50064],
                {"fresh_seeds": manifest.get("fresh_evaluation_seeds"),
                 "historical_data_hashes": manifest.get("historical_evaluation_data_sha256")})

    # Regenerate all task banks with their frozen seeds; hash canonical tensors.
    data_cpu = {size: bank(size, 32, 50000 + size, "cpu") for size in (32, 64)}
    actual_fresh = {str(size): _bank_hash(data) for size, data in data_cpu.items()}
    audit.check("fresh evaluation bank identities", actual_fresh == manifest.get("fresh_evaluation_data_sha256"),
                {"expected": manifest.get("fresh_evaluation_data_sha256"), "actual": actual_fresh})
    historical_cpu = {size: bank(size, 32, 40000 + size, "cpu") for size in (32, 64)}
    actual_historical = {str(size): _bank_hash(data) for size, data in historical_cpu.items()}
    audit.check("historical evaluation bank identities", actual_historical == manifest.get("historical_evaluation_data_sha256") and
                actual_historical == old_manifest.get("evaluation_data_sha256"),
                {"expected": manifest.get("historical_evaluation_data_sha256"), "actual": actual_historical})
    train_bank = bank(32, 512, 10002, "cpu")
    train_hash = _bank_hash(train_bank)
    audit.check("training bank identity", train_hash == manifest.get("train_data_sha256") == old.get("train_data_sha256"),
                {"actual": train_hash, "manifest": manifest.get("train_data_sha256"),
                 "historical": old.get("train_data_sha256")})
    del historical_cpu, train_bank

    # Recreate each batch schedule and compare the stored schedule table/hash.
    schedule_seeds = (20002, 20012, 20022, 20032, 20042)
    schedule_rows: dict[int, list[list[int]]] = {}
    schedule_hashes: dict[str, str] = {}
    for seed in schedule_seeds:
        rows = np.random.default_rng(seed).integers(0, 512, (300, 8)).tolist()
        path = run / f"schedule{seed}.json"
        stored = _read(path) if path.is_file() else None
        actual_hash = _sha(path) if path.is_file() else None
        expected_hash = manifest.get("schedule_sha256", {}).get(str(seed))
        exact_rows = stored == rows
        hash_ok = actual_hash == expected_hash
        audit.check(f"schedule {seed} regenerated 300x8 rows", exact_rows,
                    {"shape": [len(stored), len(stored[0])] if stored else None, "matches_regeneration": exact_rows})
        audit.check(f"schedule {seed} file hash", hash_ok,
                    {"actual": actual_hash, "expected": expected_hash})
        schedule_rows[seed] = rows
        schedule_hashes[str(seed)] = actual_hash or ""
    historical_schedule = _read(HISTORICAL_RUN / "schedule.json")
    audit.check("control schedule exactly replays historical schedule 20002",
                schedule_rows[20002] == historical_schedule and
                schedule_hashes["20002"] == old.get("schedule_sha256"),
                {"schedule_seed": 20002, "historical_schedule_match": schedule_rows[20002] == historical_schedule})

    expected_specs = _expected_specs()
    manifest_arms = manifest.get("arms", [])
    manifest_by_name = {item.get("name"): item for item in manifest_arms}
    audit.check("formal manifest lists exact 13 unique protocol arms",
                len(manifest_arms) == 13 and set(manifest_by_name) == set(expected_specs) and
                len(manifest_by_name) == len(manifest_arms),
                {"arms": len(manifest_arms), "unique": len(manifest_by_name),
                 "expected": len(expected_specs)})
    for name, spec in expected_specs.items():
        audit.check(f"manifest arm specification:{name}", manifest_by_name.get(name) == spec,
                    {"manifest": manifest_by_name.get(name), "expected": spec})

    records: dict[str, dict[str, Any]] = {}
    for name in expected_specs:
        path = run / f"{name}.json"
        if not path.is_file():
            audit.check(f"arm record present:{name}", False, "missing")
            continue
        record = _read(path)
        records[name] = record
        spec = expected_specs[name]
        spec_ok = all(record.get(key) == spec[key] for key in ("name", "experiment", "epsilon", "direction_seed", "schedule_seed"))
        audit.check(f"arm record specification:{name}", spec_ok,
                    {key: record.get(key) for key in ("name", "experiment", "epsilon", "direction_seed", "schedule_seed")})
        audit.check(f"arm trained exactly 300 updates:{name}", record.get("status") == "COMPLETE" and record.get("completed_updates") == 300,
                    {"status": record.get("status"), "completed_updates": record.get("completed_updates")})
        schedule_seed = spec["schedule_seed"]
        audit.check(f"arm training bank/schedule binding:{name}",
                    record.get("train_data_sha256") == train_hash and
                    record.get("schedule_sha256") == schedule_hashes[str(schedule_seed)],
                    {"train_data_sha256": record.get("train_data_sha256"),
                     "schedule_sha256": record.get("schedule_sha256"), "schedule_seed": schedule_seed})

    audit.check("all 13 expected arm records are present", set(records) == set(expected_specs),
                {"present": len(records), "expected": len(expected_specs)})

    # Verify all initial parameter hashes/rays on CPU against frozen construction.
    initial_metadata: dict[str, Any] = {}
    for name, record in records.items():
        spec = expected_specs[name]
        model, noise = initialization.initial_model(spec["epsilon"], spec["direction_seed"])
        initial_hash = tensor_hash(model.state_dict())
        del model
        initial_match = initial_hash == record.get("initial_parameter_sha256")
        audit.check(f"{name}: reconstructed CPU initial parameter SHA256", initial_match,
                    {"actual": initial_hash, "recorded": record.get("initial_parameter_sha256")})
        noise_detail = _tree_mismatches(noise, record.get("noise"))
        audit.check(f"{name}: initialization ray metadata", not noise_detail,
                    noise_detail or {"relative_l2": noise.get("actual_relative_l2"), "epsilon": spec["epsilon"]})
        if spec["experiment"] == "B":
            ray_ok = abs(noise["actual_relative_l2"] - spec["epsilon"]) <= 1e-5
            all_zero_blocks_shift = any(item["originally_all_zero"] and item["actual_delta_l2"] > 0
                                        for item in noise["blocks"].values())
            audit.check(f"{name}: global L2 radius and originally-zero perturbation", ray_ok and all_zero_blocks_shift,
                        {"epsilon": spec["epsilon"], "actual_relative_l2": noise["actual_relative_l2"],
                         "originally_zero_block_shifted": all_zero_blocks_shift})
        initial_metadata[name] = {"sha256": initial_hash, "matches_record": initial_match,
                                  "epsilon": spec["epsilon"], "direction_seed": spec["direction_seed"],
                                  "actual_relative_l2": noise.get("actual_relative_l2")}
    direction_rays = {}
    for direction in range(70002, 70006):
        small = initial_metadata.get(f"B_direction{direction}_eps01", {})
        large = initial_metadata.get(f"B_direction{direction}_eps05", {})
        direction_rays[str(direction)] = {
            "epsilon01_present": bool(small), "epsilon05_present": bool(large),
            "relative_l2": [small.get("actual_relative_l2"), large.get("actual_relative_l2")],
            "initial_sha256": [small.get("sha256"), large.get("sha256")],
        }
        audit.check(f"paired B direction has both radii:{direction}", bool(small and large) and
                    abs(float(small.get("actual_relative_l2", 0)) - .01) <= 1e-5 and
                    abs(float(large.get("actual_relative_l2", 0)) - .05) <= 1e-5,
                    direction_rays[str(direction)])
    report["provenance"]["directions"] = direction_rays

    # Control hashes and full historical endpoint replay (exact JSON equality).
    control = records.get("control", {})
    audit.check("control starting and final parameter hashes exactly match historical run",
                control.get("initial_parameter_sha256") == old.get("initial_parameter_sha256") and
                control.get("final_parameter_sha256") == old.get("final_parameter_sha256"),
                {"initial_match": control.get("initial_parameter_sha256") == old.get("initial_parameter_sha256"),
                 "final_match": control.get("final_parameter_sha256") == old.get("final_parameter_sha256")})
    historical_eval = control.get("historical_replay", {}).get("evaluation")
    historical_equal = historical_eval == old.get("evaluation")
    audit.check("control full six historical size x horizon records exactly equal reference",
                historical_equal,
                {"records_compared": 6, "exact_tree_equality": historical_equal,
                 "size_horizon_keys": [f"{size}/{step}" for size in ("32", "64") for step in ("64", "128", "256")]})
    audit.check("control historical replay record itself passed exact checks",
                control.get("historical_replay", {}).get("status") == "PASS" and
                control.get("historical_replay", {}).get("size_horizon_records") == 6,
                {"status": control.get("historical_replay", {}).get("status"),
                 "size_horizon_records": control.get("historical_replay", {}).get("size_horizon_records")})

    # Independently recalculate all Boolean endpoint/trajectory and matched
    # frontier results from NPZ traces. The source summaries are only compared
    # after the trace/data-derived values have been rebuilt.
    arm_rows: list[dict[str, Any]] = []
    arm_validation: list[dict[str, Any]] = []
    independent_passes: dict[str, bool] = {}
    for name in expected_specs:
        if name not in records:
            continue
        record = records[name]
        saved_path = run / f"{name}_summary.json"
        summary_sha = _sha(saved_path) if saved_path.is_file() else None
        summary_sha_ok = summary_sha == record.get("phenotype_summary_sha256")
        audit.check(f"{name}: saved phenotype summary SHA256", summary_sha_ok,
                    {"actual": summary_sha, "recorded": record.get("phenotype_summary_sha256")})
        saved_summary = _read(saved_path) if saved_path.is_file() else {}
        trace_sets = {str(size): _trace_arrays(run / f"{name}_size{size}.npz", size) for size in (32, 64)}
        independent_records = _records_from_traces(trace_sets, data_cpu)
        recomputed = phenotype.summarize_from_traces(trace_sets, independent_records, data_cpu)
        matched_rows = recomputed.pop("_matched_frontier_rows")
        gate = phenotype.predicate(recomputed)
        _compare_saved_summary(audit, name, recomputed, saved_summary, gate, record)

        matched_path = run / f"{name}_matched_frontier_strata.csv"
        matched_stats, csv_mismatches = _matched_csv_audit(matched_path, matched_rows)
        audit.check(f"{name}: matched frontier CSV integer counts and weighted recomputation",
                    not csv_mismatches, csv_mismatches or {"per_size": {
                        size: {key: value for key, value in row.items() if key != "per_map"}
                        for size, row in matched_stats.items()}})
        for size in (32, 64):
            key = str(size)
            csv_frontier = matched_stats[key]
            recomputed_frontier = recomputed["sizes"][key]["frontier"]
            expected_map_rows = [
                {field: row[field] for field in ("map_index", "common_strata", "weight_sum", "weighted_difference")}
                for row in recomputed_frontier["per_map"]
            ]
            actual_map_rows = [
                {field: row[field] for field in ("map_index", "common_strata", "weight_sum", "weighted_difference")}
                for row in csv_frontier["per_map"]
            ]
            map_detail = _tree_mismatches(actual_map_rows, expected_map_rows, limit=12)
            effect_detail = _tree_mismatches(
                {field: csv_frontier[field] for field in ("common_strata", "eligible_maps", "mean_map_weighted_difference")},
                {field: recomputed_frontier[field] for field in ("common_strata", "eligible_maps", "mean_map_weighted_difference")},
                limit=12)
            audit.check(f"{name}: size{size} CSV map denominators/effects match Boolean traces",
                        not map_detail and not effect_detail, (map_detail + effect_detail)[:12] or {
                            "common_strata": csv_frontier["common_strata"],
                            "eligible_maps": csv_frontier["eligible_maps"],
                            "mean_map_weighted_difference": csv_frontier["mean_map_weighted_difference"]})
        independent_passes[name] = bool(gate["pass"])

        # Validate Boolean NPZ checks separately so a failed trace never gets
        # hidden by a matching derived summary.
        trace_files = {str(size): {"file_sha256": _sha(run / f"{name}_size{size}.npz"),
                                   "shape": list(trace_sets[str(size)]["correct"].shape),
                                   "dtype": str(trace_sets[str(size)]["correct"].dtype),
                                   "paired_equals_branch_conjunction": True}
                       for size in (32, 64)}
        saved_gate = saved_summary.get("phenotype_gate", {})
        gate_failures = [key for key, value in gate.get("checks", {}).items() if not value["pass"]]
        project_gate = lambda value: {key: value.get(key) for key in ("pass", "checks", "interpretation")}
        gate_matches_saved = not _tree_mismatches(project_gate(gate), project_gate(saved_gate)) and not _tree_mismatches(
            project_gate(gate), project_gate(record.get("phenotype_gate", {})))
        metrics = _trace_counts(recomputed)
        validation_row = {
            "name": name,
            "experiment": expected_specs[name]["experiment"],
            "epsilon": expected_specs[name]["epsilon"],
            "direction_seed": expected_specs[name]["direction_seed"],
            "schedule_seed": expected_specs[name]["schedule_seed"],
            "updates": record.get("completed_updates"),
            "status": record.get("status"),
            "initial_parameter_sha256": initial_metadata.get(name, {}).get("sha256"),
            "final_parameter_sha256": record.get("final_parameter_sha256"),
            "checkpoint": _verify_checkpoint(audit, run, record, expected_specs[name]),
            "traces": trace_files,
            "phenotype_gate": {"pass": gate["pass"], "saved_pass": saved_gate.get("pass"),
                               "matches_saved": gate_matches_saved, "failed_checks": gate_failures,
                               "checks": gate["checks"]},
            "sizes": metrics,
            "matched_frontier_csv": {
                "sha256": _sha(matched_path),
                "sizes": matched_stats,
            },
        }
        arm_validation.append(validation_row)
        row: dict[str, Any] = {
            "name": name, "experiment": expected_specs[name]["experiment"],
            "epsilon": expected_specs[name]["epsilon"], "direction_seed": expected_specs[name]["direction_seed"],
            "schedule_seed": expected_specs[name]["schedule_seed"], "updates": record.get("completed_updates"),
            "status": record.get("status"), "phenotype_pass": gate["pass"],
            "failed_gate_count": len(gate_failures),
            "checkpoint_file_sha_match": validation_row["checkpoint"]["file_sha_match"],
            "checkpoint_state_sha_match": validation_row["checkpoint"]["state_sha_match"],
            "initial_sha_match": initial_metadata.get(name, {}).get("matches_record"),
        }
        for size in (32, 64):
            size_key = str(size)
            srow = recomputed["sizes"][size_key]
            transition = srow["transitions"]["all_changed"]["64_to_256"]["pooled"]
            row.update({
                f"size{size}_T64_original_ba": srow["endpoints"]["64"]["original_ba"],
                f"size{size}_T64_flipped_ba": srow["endpoints"]["64"]["flipped_ba"],
                f"size{size}_T64_all_changed_coverage": srow["endpoints"]["64"]["all_changed"]["pooled_coverage"]["value"],
                f"size{size}_T256_all_changed_coverage": srow["endpoints"]["256"]["all_changed"]["pooled_coverage"]["value"],
                f"size{size}_strict_T64_map_mean": srow["endpoints"]["64"]["strict_16_32"]["mean_map_coverage"],
                f"size{size}_strict_T64_pooled": srow["endpoints"]["64"]["strict_16_32"]["pooled_coverage"]["value"],
                f"size{size}_retention64_256": transition["retention"]["value"],
                f"size{size}_gain64_256": srow["endpoints"]["256"]["all_changed"]["pooled_coverage"]["value"] -
                                              srow["endpoints"]["64"]["all_changed"]["pooled_coverage"]["value"],
                f"size{size}_ever_relapse": srow["ever_regressed_over_ever_correct"]["value"],
                f"size{size}_frontier_effect": srow["frontier"]["mean_map_weighted_difference"],
                f"size{size}_frontier_eligible_maps": srow["frontier"]["eligible_maps"],
                f"size{size}_frontier_common_strata": srow["frontier"]["common_strata"],
            })
        arm_rows.append(row)

    report["arms"] = arm_validation

    # Recompute experiment decisions from independent phenotype passes using
    # the protocol's model-level units, never map/pixel counts.
    a_names = sorted(name for name in expected_specs if expected_specs[name]["experiment"] == "A")
    a_passes = sum(independent_passes.get(name, False) for name in a_names)
    a_decision = "CONDITIONAL_SCHEDULE_ROBUSTNESS" if len(a_names) == 4 and a_passes >= 3 else (
        "NOT_QUALIFIED" if len(a_names) == 4 else "INCOMPLETE")
    b_results = {}
    for epsilon in (0.01, 0.05):
        selected = [name for name in expected_specs if expected_specs[name]["experiment"] == "B" and
                    expected_specs[name]["epsilon"] == epsilon]
        passed = sum(independent_passes.get(name, False) for name in selected)
        b_results[f"{epsilon:g}"] = {
            "completed_directions": len(selected), "passes": passed, "independent_unit": "paired direction",
            "direction_passes": {str(direction): independent_passes.get(
                f"B_direction{direction}_eps{'01' if epsilon == .01 else '05'}", False)
                for direction in range(70002, 70006)},
            "decision": ("SAMPLED_DIRECTIONAL_TOLERANCE" if passed >= 3 else "NOT_QUALIFIED")
            if len(selected) == 4 else "INCOMPLETE",
        }
    aggregate_result = {
        "complete": len(independent_passes) == 13 and all(records.get(name, {}).get("status") == "COMPLETE"
                                                          for name in expected_specs),
        "control_fresh_qualification": "QUALIFIED" if independent_passes.get("control") else "BASELINE_FRESH_PHENOTYPE_UNQUALIFIED",
        "A": {"completed_schedules": len(a_names), "passes": a_passes, "independent_unit": "schedule",
              "decision": a_decision, "schedule_passes": {name: independent_passes.get(name, False) for name in a_names}},
        "B": b_results,
        "scope": "Selected initialization, fixed training bank and recipe; no basin theorem or architecture reliability claim.",
    }
    report["aggregate"] = aggregate_result
    audit.check("independent phenotype gates equal saved aggregate arm outcomes",
                aggregate.get("phenotype_pass") == independent_passes,
                {"saved_pass_count": sum(bool(v) for v in aggregate.get("phenotype_pass", {}).values()),
                 "independently_recomputed_pass_count": sum(independent_passes.values())})
    for key in ("control_fresh_qualification", "A", "B"):
        actual = aggregate.get(key)
        expected = aggregate_result.get(key)
        # Only compare protocol decision fields; per-arm map/pixel observations
        # are not independent experimental units.
        if key == "A":
            actual = {field: actual.get(field) for field in ("completed_schedules", "passes", "unit", "decision")}
            expected = {"completed_schedules": expected["completed_schedules"], "passes": expected["passes"],
                        "unit": "schedule", "decision": expected["decision"]}
        elif key == "B":
            actual = {epsilon: {field: actual.get(epsilon, {}).get(field) for field in
                                ("completed_directions", "passes", "unit", "decision")}
                      for epsilon in ("0.01", "0.05")}
            expected = {epsilon: {"completed_directions": value["completed_directions"],
                                  "passes": value["passes"], "unit": "paired_direction",
                                  "decision": value["decision"]}
                        for epsilon, value in expected.items()}
        audit.check(f"saved aggregate {key} decision matches independent model-unit recount",
                    actual == expected, {"saved": actual, "independent": expected})
    audit.check("saved aggregate decision is complete screen", aggregate.get("decision") == "SCREEN_COMPLETE",
                {"decision": aggregate.get("decision")})

    report["checks"] = audit.checks
    report["failures"] = audit.failures
    report["status"] = "PASS" if not audit.failures else "FAIL"
    report["check_counts"] = {"total": len(audit.checks), "passed": len(audit.checks) - len(audit.failures),
                              "failed": len(audit.failures)}

    # Include a small stable CSV only; detailed strata and trace-derived profiles
    # stay in the existing immutable run and are validated by this JSON receipt.
    csv_path = out / "arm_metrics.csv"
    if arm_rows:
        fields = list(arm_rows[0])
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(arm_rows)
    else:
        csv_path.write_text("name,experiment,phenotype_pass\n", encoding="utf-8")
    report["output_files"] = {"arm_metrics.csv": {"sha256": _sha(csv_path), "rows": len(arm_rows)}}
    _json_write(out / "validation.json", _json_safe(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run = args.run.resolve()
    out = args.out.resolve()
    if not run.is_relative_to(ROOT / "runs"):
        raise SystemExit("--run must be inside this project's runs directory")
    if not out.is_relative_to(ROOT / "analyses"):
        raise SystemExit("--out must be inside this project's analyses directory")
    if out.exists():
        existing = {path.name for path in out.iterdir()}
        allowed = {"validation.json", "arm_metrics.csv"}
        if not existing <= allowed:
            raise SystemExit(f"refusing to overwrite unexpected files in {out.name}")
    else:
        out.mkdir(parents=True, exist_ok=False)
    try:
        report = validate(run, out)
    except Exception as error:
        safe_error = str(error).replace(str(ROOT), "PROJECT_ROOT")
        report = {
            "schema": "seed4-followup-training-saved-result-review-v1",
            "run_id": run.name,
            "status": "FAIL",
            "validation_mode": "CPU saved-artifact review; no model inference or training",
            "fatal_error": f"{type(error).__name__}: {safe_error}",
            "check_counts": {"total": 0, "passed": 0, "failed": 1},
            "failures": [{"name": "validator execution", "pass": False, "detail": safe_error}],
            "arms": [],
        }
        csv_path = out / "arm_metrics.csv"
        csv_path.write_text("name,experiment,phenotype_pass\n", encoding="utf-8")
        report["output_files"] = {"arm_metrics.csv": {"sha256": _sha(csv_path), "rows": 0}}
        _json_write(out / "validation.json", report)
    print(json.dumps({"status": report.get("status"), "check_counts": report.get("check_counts"),
                      "output": out.relative_to(ROOT).as_posix()}, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
