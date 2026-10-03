#!/usr/bin/env python3
"""Export the frozen seed-4 transition audit as compact, sanitized public evidence.

The exporter reads completed local artifacts only. It never imports torch or
replays a model. Use --verify-only to check the exported tables, hashes and
relative Markdown links using CPU and the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = Path("runs/transition_20261003_seed4_dense01")
DEFAULT_VALIDATION = Path("analyses/transition_20261003_validation01.json")
DEFAULT_EVIDENCE = Path("evidence/transition_20261003")
DEFAULT_MANIFEST = Path("TRANSITION_PUBLICATION_MANIFEST.json")
SIZES = (32, 64)
STRICT_STEPS = (64, 128, 256)
INTERVALS = ("64_128", "64_256")
DISTANCE_BANDS = ("source0", "1_16", "17_31", "32_63", "64_inf")
PUBLIC_TOOL_PATHS = (
    "tools/export_transition_evidence.py",
    "tools/validate_transition_results.py",
    "tools/replay_transition_public.py",
)
OMIT_KEY_RE = re.compile(
    r"(?:^|_)(?:pid|host|hostname|user|username|ip|gpu|uuid|machine|command|"
    r"receipt|session|account|path|paths|absolute_path|local_path|remote_path|"
    r"started_utc|finished_utc|elapsed_seconds)(?:_|$)",
    re.IGNORECASE,
)
WINDOWS_PATH_RE = re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\[a-z0-9_.-]+[\\/])")
PRIVATE_IPV4_RE = re.compile(
    r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3}|"
    r"192\.168\.(?:\d{1,3}\.)\d{1,3})\b"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def safe_json_copy(value: Any) -> Any:
    """Copy JSON-safe values while dropping fields that carry machine receipts."""
    if isinstance(value, dict):
        return {
            key: safe_json_copy(item)
            for key, item in value.items()
            if not OMIT_KEY_RE.search(str(key))
        }
    if isinstance(value, list):
        return [safe_json_copy(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number in publication input")
    return value


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def compact_rate(rate: dict[str, Any]) -> dict[str, Any]:
    numerators, denominators = rate_arrays(rate)
    pooled_numerator = rate.get("pooled_numerator", sum(numerators))
    pooled_denominator = rate.get("pooled_denominator", sum(denominators))
    pooled_rate = rate.get("pooled_rate")
    if pooled_rate is None and pooled_denominator:
        pooled_rate = pooled_numerator / pooled_denominator
    eligible_maps = rate.get("equal_map_eligible_maps", sum(denominator > 0 for denominator in denominators))
    return {
        "equal_map_mean": rate.get("equal_map_mean"),
        "equal_map_eligible_maps": eligible_maps,
        "pooled_numerator": pooled_numerator,
        "pooled_denominator": pooled_denominator,
        "pooled_rate": pooled_rate,
        "per_map_numerator": numerators,
        "per_map_denominator": denominators,
    }


def rate_arrays(rate: dict[str, Any]) -> tuple[list[int], list[int]]:
    per_map = rate.get("per_map")
    if per_map and isinstance(per_map[0], dict):
        return (
            [int(item["numerator"]) for item in per_map],
            [int(item["denominator"]) for item in per_map],
        )
    if "per_map_numerator" in rate and "per_map_denominator" in rate:
        return (
            [int(value) for value in rate["per_map_numerator"]],
            [int(value) for value in rate["per_map_denominator"]],
        )
    raise ValueError("rate summary does not expose per-map numerators and denominators")


def metric_columns(prefix: str, rate: dict[str, Any]) -> dict[str, Any]:
    compact = compact_rate(rate)
    return {
        f"{prefix}_equal_map_mean": compact["equal_map_mean"],
        f"{prefix}_equal_map_eligible_maps": compact["equal_map_eligible_maps"],
        f"{prefix}_pooled_numerator": compact["pooled_numerator"],
        f"{prefix}_pooled_denominator": compact["pooled_denominator"],
        f"{prefix}_pooled_rate": compact["pooled_rate"],
        f"{prefix}_per_map_numerator_json": json.dumps(compact["per_map_numerator"], separators=(",", ":")),
        f"{prefix}_per_map_denominator_json": json.dumps(compact["per_map_denominator"], separators=(",", ":")),
    }


def update_size_key(update: int, size: int) -> tuple[int, int]:
    return int(update), int(size)


def public_rate_rows(
    update: int,
    size: int,
    kind: str,
    interval: str,
    metric: str,
    distance_band: str,
    rate: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    numerators, denominators = rate_arrays(rate)
    for map_index, (numerator, denominator) in enumerate(zip(numerators, denominators)):
        rows.append({
            "update": update,
            "size": size,
            "kind": kind,
            "interval": interval,
            "metric": metric,
            "distance_band": distance_band,
            "map_index": map_index,
            "numerator": numerator,
            "denominator": denominator,
            "rate": None if denominator == 0 else numerator / denominator,
        })
    return rows


def primary_profile_row(
    checkpoint: dict[str, Any],
    screen: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "update": int(checkpoint["update"]),
        "size": int(checkpoint["size"]),
        "steps": int(checkpoint["steps"]),
        "finite": bool(checkpoint["finite"]),
        "light_cone": bool(checkpoint["light_cone"]),
        "descriptive_screen": bool(screen["descriptive_screen"]),
        "prespecified_primary_screen": bool(screen["prespecified_primary_screen"]),
        "size64_is_secondary_projection": bool(screen["size64_is_secondary_projection"]),
        "screen_strict128_pooled": bool(screen["checks"]["strict128_pooled"]),
        "screen_strict128_equal_map": bool(screen["checks"]["strict128_equal_map"]),
        "screen_G64_128": bool(screen["checks"]["g64_128"]),
        "screen_first_exit64_128": bool(screen["checks"]["first_exit64_128"]),
        "screen_survival64_256": bool(screen["checks"]["survival64_256"]),
    }
    for step in STRICT_STEPS:
        rate = checkpoint["endpoints"][str(step)]["strict_coverage"]
        row.update(metric_columns(f"strict_T{step}", rate))
    names = {
        "acquisition": "G",
        "endpoint_destruction": "D_endpoint",
        "first_exit": "D_first_exit",
        "continuous_survival": "S_continuous",
        "net_gain": "net_gain",
    }
    for interval in INTERVALS:
        for source_name, public_name in names.items():
            rate = checkpoint["behavior_profile"][interval][source_name]
            row.update(metric_columns(f"{public_name}_{interval}", rate))
    for step in STRICT_STEPS:
        for field in (
            "original_BA_equal_map",
            "flipped_BA_equal_map",
            "original_BCE_balanced",
            "flipped_BCE_balanced",
        ):
            row[f"T{step}_{field}"] = checkpoint["endpoints"][str(step)][field]
    return row


def aggregates_only(value: Any) -> Any:
    """Keep aggregate rates here; exact map counts are stored in the CSV files."""
    if isinstance(value, dict):
        return {
            key: aggregates_only(item)
            for key, item in value.items()
            if key not in {"per_map", "per_map_numerator", "per_map_denominator"}
        }
    if isinstance(value, list):
        return [aggregates_only(item) for item in value]
    return safe_json_copy(value)


def summarize_checkpoint(
    raw: dict[str, Any],
    compact: dict[str, Any],
    screen: dict[str, Any],
) -> dict[str, Any]:
    endpoints = {}
    for step in STRICT_STEPS:
        raw_endpoint = raw["endpoints"][str(step)]
        endpoints[str(step)] = {
            "strict_coverage": aggregates_only(raw_endpoint["strict_coverage"]),
            "original_BA_equal_map": raw_endpoint["original_BA_equal_map"],
            "flipped_BA_equal_map": raw_endpoint["flipped_BA_equal_map"],
            "original_BCE_balanced": raw_endpoint["original_BCE_balanced"],
            "flipped_BCE_balanced": raw_endpoint["flipped_BCE_balanced"],
            "paired_margin_quantiles": safe_json_copy(raw_endpoint["paired_margin_quantiles"]),
        }
    behavior = raw["behavior"]
    return {
        "schema_version": "transition-checkpoint-public-v1",
        "update": int(raw["update"]),
        "size": int(raw["size"]),
        "parameter_sha256": raw["parameter_sha256"],
        "trace_sha256": raw["trace_sha256"],
        "audit_bank_sha256": raw["audit_bank_sha256"],
        "steps": int(raw["steps"]),
        "finite": bool(raw["finite"]),
        "light_cone": bool(raw["light_cone"]),
        "strict_endpoint_profiles": endpoints,
        "screen": safe_json_copy(screen),
        "all_changed_transition_profiles": aggregates_only(compact["behavior_profile"]),
        "primary_cohort_timing_and_retention": {
            "cohort": behavior["cohort_definition"],
            "first_acquisition": aggregates_only(behavior["first_acquisition"]),
            "fixed_lag_first_correct_survival": aggregates_only(
                behavior["fixed_lag_first_correct_survival"]["primary"]
            ),
            "finite_horizon_lags": aggregates_only(behavior["finite_horizon_lags"]["primary"]),
            "age_hazard": aggregates_only(behavior["age_hazard"]["by_distance"]["17_31"]),
            "distance_stratified_map_counts_are_in_csv": True,
        },
        "age_hazard_definition": behavior["age_hazard"]["definition"],
        "fixed_lag_definition": behavior["fixed_lag_first_correct_survival"]["primary"]["definition"],
        "finite_horizon_definition": behavior["finite_horizon_lags"]["primary"]["definition"],
        "age_spell_note": behavior["age_spell_note"],
        "scope_note": behavior["scope"],
        "omitted_redundant_or_stepwise_arrays": [
            "map-level arrays (retained in CSV files)",
            "coverage at each macro step",
            "step accounting at each macro step",
            "per-step state RMS",
        ],
    }


def source_binding_check(source_sha256: dict[str, str]) -> None:
    mismatches = []
    for relpath, expected in source_sha256.items():
        path = ROOT / PurePosixPath(relpath)
        if not path.is_file():
            mismatches.append(f"missing {relpath}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            mismatches.append(f"hash mismatch {relpath}: expected {expected}, got {actual}")
    if mismatches:
        raise ValueError("frozen execution source binding check failed:\n" + "\n".join(mismatches))


def make_profile_rows(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    screen_by_key = {
        update_size_key(item["update"], item["size"]): item
        for item in summary["descriptive_screen_profiles"]
    }
    checkpoints = {
        update_size_key(item["update"], item["size"]): item
        for item in summary["checkpoints"]
    }
    if set(screen_by_key) != set(checkpoints):
        raise ValueError("checkpoint and screen profile keys do not match")
    ordered = sorted(checkpoints)
    return [primary_profile_row(checkpoints[key], screen_by_key[key]) for key in ordered], checkpoints


def run_metadata(
    run_manifest: dict[str, Any],
    aggregate: dict[str, Any],
    validation_hash: str,
) -> dict[str, Any]:
    anchors = run_manifest["reference_bindings"]["anchors"]
    return {
        "schema_version": "transition-run-metadata-v1",
        "protocol": run_manifest["protocol"],
        "execution_status": "COMPLETE",
        "completed_updates": int(aggregate["completed_updates"]),
        "completed_audits": int(aggregate["completed_audits"]),
        "selected_training_runs": 1,
        "seed": int(run_manifest["seed"]),
        "train_seed": int(run_manifest["train_seed"]),
        "schedule_seed": int(run_manifest["schedule_seed"]),
        "gradient_horizon": int(run_manifest["gradient_horizon"]),
        "forward_steps": int(run_manifest["forward_steps"]),
        "audit_banks": safe_json_copy(run_manifest["audit_banks"]),
        "dense_audit_updates": [int(value) for value in run_manifest["audit_updates"] if value <= 200],
        "separate_control_update": 300,
        "profile_sizes": [32, 64],
        "strict_endpoint_steps": list(STRICT_STEPS),
        "reference_validation": {
            "historical_replay_status": "PASS",
            "baseline_anchor_parameter_sha256": {
                str(update): anchors[str(update)]["parameter_sha256"]
                for update in (0, 100, 200, 300)
            },
            "reference_record_sha256": run_manifest["reference_bindings"]["record_sha256"],
            "historical_replay_payload_sha256": "6814f8be22040b63d9dc8e28e293aa3db3ff1d5c63520c35098afb63821c1c2f",
            "public_replay_file_byte_verification": False,
            "public_replay_limitation": (
                "The original local execution verified its private historical checkpoint file bindings. "
                "A public clone cannot verify those unavailable file bytes; the public replay wrapper checks "
                "published parameter hashes against the public stage payloads instead."
            ),
        },
        "preflight": {
            "passed": True,
            "updates": 3,
            "audited_banks": [32, 64],
            "map_count_per_bank": 16,
            "all_six_parameter_groups_finite_and_nonzero_gradient": True,
        },
        "validation_artifact_sha256": validation_hash,
        "source_base_commit": run_manifest.get("git_base"),
        "machine_and_dispatch_receipts": "omitted",
    }


def input_exclusion_records(run_dir: Path, raw_json_hashes: dict[str, tuple[str, int]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relname, (digest, byte_count) in sorted(raw_json_hashes.items()):
        records.append({
            "relative_name": relname,
            "sha256": digest,
            "bytes": byte_count,
            "class": "raw per-checkpoint JSON with stepwise arrays; sanitized summary exported",
            "published": False,
        })
    checkpoints = run_dir / "checkpoints"
    for path in sorted(checkpoints.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".npz", ".pt"}:
            continue
        records.append({
            "relative_name": f"checkpoints/{path.name}",
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "class": "raw trace archive" if path.suffix.lower() == ".npz" else "model checkpoint",
            "published": False,
        })
    for name in ("bank_size32.npz", "bank_size64.npz"):
        path = run_dir / name
        records.append({
            "relative_name": name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "class": "fixed audit bank tensor archive",
            "published": False,
        })
    return records


def make_validation_public(source: Path, target: Path) -> str:
    data = source.read_bytes()
    source_hash = sha256_bytes(data)
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, dict) or parsed.get("status") != "PASS" or parsed.get("failed_checks"):
        raise ValueError("counter-validation does not have a passing status with zero failed checks")
    counters = parsed.get("counters", {})
    comparison_error = counters.get("metric_comparison_maximum_absolute_error")
    tolerance = counters.get("metric_comparison_tolerance")
    finalizer_hash = parsed.get("finalization", {}).get("finalizer_validator_sha256")
    current_validator_hash = sha256_file(ROOT / "tools/validate_transition_results.py")
    if (
        counters.get("audit_records_expected") != 44
        or counters.get("audit_records_processed") != 44
        or counters.get("raw_npz_sha256_checked") != 44
        or comparison_error is None
        or tolerance is None
        or float(comparison_error) > float(tolerance)
        or finalizer_hash != current_validator_hash
    ):
        raise ValueError("counter-validation lacks the expected 44-record and metric-recomputation checks")
    public = safe_json_copy(parsed)
    public_counters = public.get("counters", {})
    if (
        public_counters.get("metric_comparison_maximum_absolute_error") is None
        or float(public_counters["metric_comparison_maximum_absolute_error"]) != float(comparison_error)
        or public_counters.get("metric_comparison_tolerance") is None
        or float(public_counters["metric_comparison_tolerance"]) != float(tolerance)
    ):
        raise ValueError("public counter-validation lost its metric error or tolerance")
    write_json(target, public)
    return source_hash


def write_training_csv(training: dict[str, Any], path: Path) -> None:
    checkpoints = {int(update) for update in training["checkpoints"]}
    rows = []
    for point in training["curve"]:
        rows.append({
            "update": int(point["update"]),
            "loss": float(point["loss"]),
            "gradient_norm": float(point["gradient_norm"]),
            "checkpoint_saved": int(point["update"]) in checkpoints,
        })
    write_csv(path, ["update", "loss", "gradient_norm", "checkpoint_saved"], rows)


def write_transition_counts(
    raw_checkpoints: dict[tuple[int, int], dict[str, Any]],
    compact_checkpoints: dict[tuple[int, int], dict[str, Any]],
    output_dir: Path,
) -> dict[str, int]:
    strict_rows: list[dict[str, Any]] = []
    transition_rows: dict[str, list[dict[str, Any]]] = {"all_changed": [], **{band: [] for band in DISTANCE_BANDS}}
    transition_names = {
        "acquisition": "G",
        "endpoint_destruction": "D_endpoint",
        "first_exit": "D_first_exit",
        "continuous_survival": "S_continuous",
        "net_gain": "net_gain",
    }
    for key in sorted(raw_checkpoints):
        update, size = key
        raw_behavior = raw_checkpoints[key]["behavior"]
        compact = compact_checkpoints[key]
        for step in STRICT_STEPS:
            rate = compact["endpoints"][str(step)]["strict_coverage"]
            strict_rows.extend(public_rate_rows(update, size, "strict_coverage", f"T{step}", "coverage", "17_31", rate))
        for interval in INTERVALS:
            for source_name, public_name in transition_names.items():
                rate = raw_behavior["intervals"][interval][source_name]
                transition_rows["all_changed"].extend(public_rate_rows(update, size, "transition", interval, public_name, "all_changed", rate))
        for band in DISTANCE_BANDS:
            for interval in INTERVALS:
                for source_name, public_name in transition_names.items():
                    rate = raw_behavior["intervals_by_distance"][band][interval][source_name]
                    transition_rows[band].extend(public_rate_rows(update, size, "transition", interval, public_name, band, rate))
    fields = ["update", "size", "kind", "interval", "metric", "distance_band", "map_index", "numerator", "denominator", "rate"]
    output: dict[str, int] = {}
    strict_path = output_dir / "strict_coverage_counts.csv"
    write_csv(strict_path, fields, strict_rows)
    output[strict_path.name] = len(strict_rows)
    for band, rows in transition_rows.items():
        path = output_dir / f"transition_counts_{band}.csv"
        write_csv(path, fields, rows)
        output[path.name] = len(rows)
    return output


def write_age_hazard(
    raw_checkpoints: dict[tuple[int, int], dict[str, Any]],
    output_dir: Path,
) -> dict[str, int]:
    rows_by_band: dict[str, list[dict[str, Any]]] = {band: [] for band in DISTANCE_BANDS}
    for (update, size), raw in sorted(raw_checkpoints.items()):
        for band, ages in raw["behavior"]["age_hazard"]["by_distance"].items():
            for age_bin, rate in ages.items():
                for item in rate["per_map"]:
                    rows_by_band[band].append({
                        "update": update,
                        "size": size,
                        "distance_band": band,
                        "age_bin": age_bin,
                        "map_index": int(item["map_index"]),
                        "loss_events": int(item["numerator"]),
                        "at_risk_cell_steps": int(item["denominator"]),
                        "next_step_loss_hazard": item["value"],
                    })
    fields = ["update", "size", "distance_band", "age_bin", "map_index", "loss_events", "at_risk_cell_steps", "next_step_loss_hazard"]
    output: dict[str, int] = {}
    for band, rows in rows_by_band.items():
        path = output_dir / f"age_hazard_{band}.csv"
        write_csv(path, fields, rows)
        output[path.name] = len(rows)
    return output


def write_fixed_lags(
    raw_checkpoints: dict[tuple[int, int], dict[str, Any]],
    output_dir: Path,
) -> dict[str, int]:
    rows_by_band: dict[str, list[dict[str, Any]]] = {band: [] for band in DISTANCE_BANDS}
    for (update, size), raw in sorted(raw_checkpoints.items()):
        by_distance = raw["behavior"]["fixed_lag_first_correct_survival"]["by_distance"]
        for band, summary in by_distance.items():
            for lag_row in summary["lags"]:
                exclusions = {
                    int(item["map_index"]): item
                    for item in lag_row["exclusions"]["per_map"]
                }
                for item in lag_row["uninterrupted_survival"]["per_map"]:
                    excluded = exclusions[int(item["map_index"])]
                    rows_by_band[band].append({
                        "update": update,
                        "size": size,
                        "distance_band": band,
                        "lag": int(lag_row["lag"]),
                        "map_index": int(item["map_index"]),
                        "survived_first_correct_spell": int(item["numerator"]),
                        "eligible_first_correct_spells": int(item["denominator"]),
                        "never_correct": int(excluded["never_correct"]),
                        "right_censored": int(excluded["right_censored"]),
                        "uninterrupted_survival": item["value"],
                    })
    fields = ["update", "size", "distance_band", "lag", "map_index", "survived_first_correct_spell", "eligible_first_correct_spells", "never_correct", "right_censored", "uninterrupted_survival"]
    output: dict[str, int] = {}
    for band, rows in rows_by_band.items():
        path = output_dir / f"fixed_lag_survival_{band}.csv"
        write_csv(path, fields, rows)
        output[path.name] = len(rows)
    return output


def write_lag_medians(
    raw_checkpoints: dict[tuple[int, int], dict[str, Any]],
    output_dir: Path,
) -> dict[str, int]:
    rows_by_band: dict[str, list[dict[str, Any]]] = {band: [] for band in DISTANCE_BANDS}
    for (update, size), raw in sorted(raw_checkpoints.items()):
        behavior = raw["behavior"]
        finite_by_distance = behavior["finite_horizon_lags"]["by_distance"]
        acquisition_by_distance = behavior["first_acquisition_by_distance"]
        for band in DISTANCE_BANDS:
            finite = finite_by_distance[band]
            acquisition = acquisition_by_distance[band]
            finite_maps = {int(item["map_index"]): item for item in finite["per_map"]}
            lag_maps = {
                int(item["map_index"]): item
                for item in finite["lag_final_correct"]["per_map"]
            }
            acquisition_maps = {int(item["map_index"]): item for item in acquisition["per_map"]}
            for map_index in sorted(finite_maps):
                counts = finite_maps[map_index]
                first = acquisition_maps[map_index]
                lag = lag_maps[map_index]
                measures = (
                    ("first_correct_ever_correct", int(first["ever_correct"]), first["median_first_correct_time"]),
                    ("first_correct_conditioned_final_correct", int(counts["final_correct"]), counts["first_correct_median_final_correct"]),
                    ("stable_time_conditioned_final_correct", int(counts["final_correct"]), counts["stable_time_median_final_correct"]),
                    ("stable_minus_first_conditioned_final_correct", int(lag["n"]), lag["median"]),
                )
                for measure, n, median in measures:
                    rows_by_band[band].append({
                        "update": update,
                        "size": size,
                        "distance_band": band,
                        "map_index": map_index,
                        "measure": measure,
                        "n": n,
                        "median_macro_step": median,
                        "selected_pixels": int(counts["selected_pixels"]),
                        "ever_correct": int(first["ever_correct"]),
                        "never_correct": int(counts["never_correct"]),
                        "final_correct": int(counts["final_correct"]),
                        "final_wrong": int(counts["final_wrong"]),
                        "stable_time_defined_only_for_final_correct": True,
                    })
    fields = ["update", "size", "distance_band", "map_index", "measure", "n", "median_macro_step", "selected_pixels", "ever_correct", "never_correct", "final_correct", "final_wrong", "stable_time_defined_only_for_final_correct"]
    output: dict[str, int] = {}
    for band, rows in rows_by_band.items():
        path = output_dir / f"finite_trace_lag_medians_{band}.csv"
        write_csv(path, fields, rows)
        output[path.name] = len(rows)
    return output


def render_results(summary: dict[str, Any], rows: list[dict[str, Any]], diagnostics: dict[str, Any]) -> str:
    def get_row(update: int, size: int) -> dict[str, Any]:
        return next(row for row in rows if row["update"] == update and row["size"] == size)

    table_updates = ((100, 32), (115, 32), (130, 32), (160, 32), (175, 32), (180, 32), (185, 32), (195, 32), (200, 32), (300, 32))
    table = [
        "| Update | Size | Strict T64 | Strict T128 | Strict T256 | G64→128 | First exit64→128 | Survival64→256 | Net gain64→128 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for update, size in table_updates:
        row = get_row(update, size)
        table.append(
            f"| {update} | {size} | {row['strict_T64_pooled_rate']:.4f} | {row['strict_T128_pooled_rate']:.4f} | "
            f"{row['strict_T256_pooled_rate']:.4f} | {row['G_64_128_pooled_rate']:.4f} | "
            f"{row['D_first_exit_64_128_pooled_rate']:.4f} | {row['S_continuous_64_256_pooled_rate']:.4f} | "
            f"{row['net_gain_64_128_pooled_rate']:.4f} |"
        )
    screen32 = [int(row["update"]) for row in rows if row["size"] == 32 and row["descriptive_screen"]]
    dense_screen32 = [update for update in screen32 if update <= 200]
    control_screen32 = [update for update in screen32 if update > 200]
    first_entry = diagnostics["first_entry64_lag64"]
    finite_lags = diagnostics["final_correct_conditioned_lag"]
    u175 = get_row(175, 32)
    u180 = get_row(180, 32)
    u195 = get_row(195, 32)
    u200 = get_row(200, 32)
    return "\n".join([
        "# Seed 4 dense transition audit evidence",
        "",
        "This package describes one selected historical seed-4 training trajectory. It completed 300 training updates and 44 post-training behavior audits. The dense scan covers updates 100–200 every five updates for fixed paired map banks at sizes 32 and 64; update 300 is a separate control. The independent training unit is one trajectory. Maps, pixels, macro steps and checkpoints are not independent training replications.",
        "",
        "## Result",
        "",
        "The measured behavior improves gradually with substantial reversals, rather than following a smooth monotone curve. At updates 120 and 125, size-32 continuous survival64→256 is 0.9964 and 0.9970 while strict T128 coverage remains 0.1249 and 0.2409; retention is high while reach is low. Update 180 has a large, temporary regression after stronger profiles at 175. The trajectory recovers through 195–200, with a marked reduction in first-exit rate and a rise in finite-horizon first-entry survival from 195 to 200.",
        "",
        f"At update 175, size-32 pooled strict coverage is T128={u175['strict_T128_pooled_rate']:.4f}, G64→128={u175['G_64_128_pooled_rate']:.4f}, first-exit64→128={u175['D_first_exit_64_128_pooled_rate']:.4f}, and survival64→256={u175['S_continuous_64_256_pooled_rate']:.4f}. At update 180 these are {u180['strict_T128_pooled_rate']:.4f}, {u180['G_64_128_pooled_rate']:.4f}, {u180['D_first_exit_64_128_pooled_rate']:.4f}, and {u180['S_continuous_64_256_pooled_rate']:.4f}.",
        "",
        f"At update 195, size-32 first-entry survival through lag 64 is {first_entry[195]['rate']:.4f} over {first_entry[195]['eligible']} eligible first-correct cells ({first_entry[195]['right_censored']} late entries are right-censored). At update 200 it is {first_entry[200]['rate']:.4f} over {first_entry[200]['eligible']} eligible cells ({first_entry[200]['right_censored']} right-censored). This supports a sharper retention profile at the saved 200 checkpoint; it does not identify a latent commitment event.",
        "",
        f"Only update {', '.join(map(str, dense_screen32))} passes the size-32 screen in the dense window. The separate update-300 control {('also passes' if control_screen32 else 'does not pass')}; it is not adjacent to update 200. Dense checkpoints 205 and 210 were not measured, so the persistent three-checkpoint onset is NONE (persistence not established). This is not a failed candidate or failed architecture-gate verdict.",
        "",
        *table,
        "",
        "## Finite-horizon timing caveat",
        "",
        f"For size 32, the final-correct-conditioned median stable-minus-first-correct lag is already {finite_lags[120]['median']:.1f} macro steps at update 120, with {finite_lags[120]['final_correct']} of {finite_lags[120]['selected']} primary-cohort cells final-correct. At update 180 the median remains {finite_lags[180]['median']:.1f} while only {finite_lags[180]['final_correct']} of {finite_lags[180]['selected']} are final-correct on {finite_lags[180]['eligible_maps']} eligible map; at update 200 it remains {finite_lags[200]['median']:.1f} with {finite_lags[200]['final_correct']} of {finite_lags[200]['selected']} final-correct. The conditioned median alone therefore cannot locate the update-200 change; its denominator and selection change sharply and do not establish commitment.",
        "",
        "Map-level counts and denominators are retained in the CSV evidence so the displayed rates can be recomputed. These map and cell-step counts are descriptive; they do not support independent-cell confidence intervals.",
        "",
        "## Evidence files",
        "",
        "- [Canonical 44-row endpoint and transition profile table](profiles.csv)",
        "- [Per-map strict endpoint coverage counts](strict_coverage_counts.csv)",
        "- [Per-map all-changed transition counts](transition_counts_all_changed.csv)",
        "- Per-map transition counts by distance: " + ", ".join(f"[{band}](transition_counts_{band}.csv)" for band in DISTANCE_BANDS),
        "- [Training loss and gradient norm by update](training_curve.csv)",
        "- Distance and run-age loss hazard counts: " + ", ".join(f"[{band}](age_hazard_{band}.csv)" for band in DISTANCE_BANDS),
        "- Censoring-aware fixed-lag first-correct survival: " + ", ".join(f"[{band}](fixed_lag_survival_{band}.csv)" for band in DISTANCE_BANDS),
        "- Finite-horizon lag medians and conditioning counts: " + ", ".join(f"[{band}](finite_trace_lag_medians_{band}.csv)" for band in DISTANCE_BANDS),
        "- [44 compact per-checkpoint JSON summaries](checkpoints/)",
        "- [Transition profile figure](transition_profiles.png)",
        "- [Sanitized counter-validation result](validation/local_validation.json)",
        "- [Reproduction and source binding notes](REPRODUCTION.md)",
        "",
        "## Claim boundary",
        "",
        "This is a descriptive audit of one trajectory, not a population estimate or causal intervention. Saved checkpoints locate observed behavior on this trajectory; they do not establish a physical phase transition, latent commitment, contextual type closure or generalization. Size 64 is a secondary projection. The historical gate is unchanged.",
        "",
    ])


def selected_diagnostics(raw_checkpoints: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    finite = {}
    fixed = {}
    for update in (120, 180, 195, 200):
        raw = raw_checkpoints[(update, 32)]["behavior"]
        primary = raw["finite_horizon_lags"]["by_distance"]["17_31"]
        pooled = primary["pooled"]
        lag_summary = primary["lag_final_correct"]
        finite[update] = {
            "selected": pooled["selected_pixels"],
            "never_correct": pooled["never_correct"],
            "final_correct": pooled["final_correct"],
            "final_wrong": pooled["final_wrong"],
            "median": lag_summary["pooled_median"],
            "pooled_lag_n": lag_summary["pooled_n"],
            "eligible_maps": primary["equal_map"]["final_correct_eligible_maps"],
        }
        lag64 = raw["fixed_lag_first_correct_survival"]["by_distance"]["17_31"]["lags"]
        row = next(item for item in lag64 if int(item["lag"]) == 64)
        rate = row["uninterrupted_survival"]
        fixed[update] = {
            "numerator": int(rate["pooled_numerator"]),
            "eligible": int(rate["pooled_denominator"]),
            "rate": float(rate["pooled_rate"]) if rate["pooled_rate"] is not None else None,
            "never_correct": int(row["exclusions"]["never_correct_pooled"]),
            "right_censored": int(row["exclusions"]["right_censored_pooled"]),
        }
    return {"final_correct_conditioned_lag": finite, "first_entry64_lag64": fixed}


def markdown_local_links(path: Path, text: str) -> list[str]:
    bad = []
    for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)|<img\s+[^>]*src=['\"]([^'\"]+)", text, re.IGNORECASE):
        target = match.group(1) or match.group(2)
        target = target.split("#", 1)[0]
        if not target or re.match(r"^[a-z]+://", target, re.IGNORECASE) or target.startswith("#"):
            continue
        if target.startswith("/") or WINDOWS_PATH_RE.search(target):
            bad.append(f"absolute Markdown link: {target}")
            continue
        resolved = (path.parent / Path(target)).resolve()
        if not resolved.exists():
            bad.append(f"broken Markdown link: {target} in {path.relative_to(ROOT).as_posix()}")
    return bad


def privacy_scan(paths: Iterable[Path]) -> list[str]:
    failures = []
    for path in paths:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".npz", ".pt"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if WINDOWS_PATH_RE.search(content):
            failures.append(f"Windows or UNC path found in {path.relative_to(ROOT).as_posix()}")
        if PRIVATE_IPV4_RE.search(content):
            failures.append(f"private IPv4 address found in {path.relative_to(ROOT).as_posix()}")
    return failures


def evidence_files(evidence_dir: Path, manifest_path: Path) -> list[Path]:
    files = []
    for path in evidence_dir.rglob("*"):
        if path.is_file() and path.resolve() != manifest_path.resolve():
            files.append(path)
    return sorted(files)


def file_hash_map(paths: Iterable[Path]) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(paths)
    }


def build(
    run_rel: Path,
    validation_rel: Path,
    evidence_rel: Path,
    manifest_rel: Path,
) -> dict[str, Any]:
    run_dir = ROOT / run_rel
    validation_path = ROOT / validation_rel
    evidence_dir = ROOT / evidence_rel
    manifest_path = ROOT / manifest_rel
    for required in (run_dir / "manifest.json", run_dir / "aggregate.json", run_dir / "summary.json", run_dir / "training.json", run_dir / "transition_profiles.png", validation_path):
        if not required.is_file():
            raise FileNotFoundError(f"required publication input is missing: {required.relative_to(ROOT).as_posix()}")
    run_manifest = read_json(run_dir / "manifest.json")
    aggregate = read_json(run_dir / "aggregate.json")
    summary = read_json(run_dir / "summary.json")
    training = read_json(run_dir / "training.json")
    source_binding_check(run_manifest["source_sha256"])
    if aggregate.get("pass") is not True or aggregate.get("completed_updates") != 300 or aggregate.get("completed_audits") != 44:
        raise ValueError("frozen run aggregate is not the expected completed 300-update/44-audit record")
    if summary.get("execution_complete") is not True or summary.get("decision") != "DESCRIPTIVE_AUDIT_COMPLETE":
        raise ValueError("frozen result summary is incomplete or has an unexpected decision label")
    if summary.get("protocol") != "seed4_dense_transition_v1" or run_manifest.get("protocol") != summary["protocol"]:
        raise ValueError("frozen protocol labels disagree")

    profile_rows, compact_checkpoints = make_profile_rows(summary)
    expected_keys = {(update, size) for update in run_manifest["audit_updates"] for size in SIZES}
    if set(compact_checkpoints) != expected_keys or len(profile_rows) != 44:
        raise ValueError("expected 22 audited updates x two map-bank sizes")
    screen_primary = [row["update"] for row in profile_rows if row["size"] == 32 and row["descriptive_screen"]]
    if summary.get("operational_onset") is not None:
        raise ValueError("the frozen summary unexpectedly reports a persistent onset")

    evidence_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = evidence_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validation_out = evidence_dir / "validation" / "local_validation.json"
    validation_hash = make_validation_public(validation_path, validation_out)

    profile_fieldnames = list(profile_rows[0])
    write_csv(evidence_dir / "profiles.csv", profile_fieldnames, profile_rows)
    write_training_csv(training, evidence_dir / "training_curve.csv")

    raw_checkpoints: dict[tuple[int, int], dict[str, Any]] = {}
    raw_json_hashes: dict[str, tuple[str, int]] = {}
    screen_by_key = {
        update_size_key(item["update"], item["size"]): item
        for item in summary["descriptive_screen_profiles"]
    }
    for key in sorted(compact_checkpoints):
        update, size = key
        relname = f"checkpoints/u{update:03d}_size{size:02d}.json"
        source = run_dir / relname
        raw_bytes = source.read_bytes()
        raw_json_hashes[relname] = (sha256_bytes(raw_bytes), len(raw_bytes))
        raw = json.loads(raw_bytes.decode("utf-8"))
        if (int(raw["update"]), int(raw["size"])) != key:
            raise ValueError(f"checkpoint identity mismatch: {relname}")
        raw_checkpoints[key] = raw
        public = summarize_checkpoint(raw, compact_checkpoints[key], screen_by_key[key])
        write_json(checkpoint_dir / f"u{update:03d}_size{size:02d}.json", public)

    transition_count_rows = write_transition_counts(raw_checkpoints, compact_checkpoints, evidence_dir)
    age_rows = write_age_hazard(raw_checkpoints, evidence_dir)
    fixed_rows = write_fixed_lags(raw_checkpoints, evidence_dir)
    lag_rows = write_lag_medians(raw_checkpoints, evidence_dir)
    diagnostics = selected_diagnostics(raw_checkpoints)

    preflight_aggregate_path = ROOT / "runs/transition_20261003_preflight01/aggregate.json"
    if not preflight_aggregate_path.is_file():
        raise FileNotFoundError("frozen preflight aggregate is missing")
    preflight_aggregate = read_json(preflight_aggregate_path)
    if preflight_aggregate.get("pass") is not True or preflight_aggregate.get("preflight") is not True:
        raise ValueError("recorded preflight did not pass")
    write_json(evidence_dir / "run_metadata.json", run_metadata(run_manifest, aggregate, validation_hash))
    write_json(evidence_dir / "preflight_qualification.json", {
        "schema_version": "transition-preflight-public-v1",
        "protocol": preflight_aggregate["protocol"],
        "status": "PREFLIGHT_PASSED",
        "passed": True,
        "updates": int(preflight_aggregate["completed_updates"]),
        "audited_map_sizes": [32, 64],
        "map_count_per_size": 16,
        "gradient_groups": safe_json_copy(preflight_aggregate["gradient_groups"]),
        "source_bindings": safe_json_copy(run_manifest["source_sha256"]),
        "machine_and_dispatch_receipts": "omitted",
    })
    shutil.copyfile(run_dir / "transition_profiles.png", evidence_dir / "transition_profiles.png")

    # Publish only a compact, claim-bounded summary of the exact frozen run.
    key_profiles = []
    for update in (100, 120, 175, 180, 185, 195, 200, 300):
        for size in SIZES:
            row = next(item for item in profile_rows if item["update"] == update and item["size"] == size)
            key_profiles.append({
                key: row[key]
                for key in (
                    "update", "size", "strict_T64_pooled_rate", "strict_T128_pooled_rate", "strict_T256_pooled_rate",
                    "G_64_128_pooled_rate", "D_endpoint_64_128_pooled_rate", "D_first_exit_64_128_pooled_rate",
                    "S_continuous_64_128_pooled_rate", "S_continuous_64_256_pooled_rate", "net_gain_64_128_pooled_rate",
                    "prespecified_primary_screen", "descriptive_screen",
                )
            })
    public_summary = {
        "schema_version": "transition-public-summary-v1",
        "protocol": summary["protocol"],
        "execution_status": "COMPLETE",
        "selected_training_runs": 1,
        "training_updates": 300,
        "completed_behavior_audits": 44,
        "dense_updates": [int(value) for value in run_manifest["audit_updates"] if value <= 200],
        "separate_control_update": 300,
        "sizes": [32, 64],
        "primary_size": 32,
        "size64_role": "secondary projection",
        "descriptive_screen": {
            "thresholds": {
                "strict_T128_pooled_and_equal_map": 0.80,
                "G64_128_pooled": 0.20,
                "first_exit64_128_pooled_max": 0.01,
                "continuous_survival64_256_pooled_min": 0.95,
            },
            "size32_passing_updates": [int(value) for value in screen_primary],
            "three_consecutive_dense_checkpoints": summary["operational_onset"],
        "verdict": "NONE (persistence not established); update 200 passes but has no two later consecutive dense saved profiles",
            "u300_is_not_in_dense_sequence": True,
        },
        "diagnostic_caveats": diagnostics,
        "historical_replay": {
            "status": "PASS",
            "integer_leaves": 2481,
            "float_leaves": 822,
            "maximum_absolute_error": 0.0,
            "public_replay_file_byte_verification": False,
        },
        "key_profiles": key_profiles,
        "independent_unit": "one selected training trajectory; maps, cells, cell-times, and checkpoints are not independent training replications",
        "claim_boundary": [
            "descriptive within-training behavior only",
            "no physical phase transition or latent commitment identified",
            "no causal handoff or contextual type closure tested",
            "the historical formal gate is unchanged",
        ],
    }
    write_json(evidence_dir / "summary.json", public_summary)
    (evidence_dir / "RESULTS.md").write_text(render_results(summary, profile_rows, diagnostics), encoding="utf-8", newline="\n")
    (evidence_dir / "REPRODUCTION.md").write_text("\n".join([
        "# Reproduction and source bindings",
        "",
        "The frozen executed run used the source files identified by SHA-256 in [run_metadata.json](run_metadata.json) and in the root publication manifest. The original local run passed historical checkpoint and payload replay checks. The local checkpoints and trace archives are intentionally excluded; their relative names, byte sizes and SHA-256 hashes are listed in the root publication manifest.",
        "",
        "A public clone uses the dedicated wrapper below. The wrapper is a reproduction path, not a byte-for-byte verification of the private historical checkpoint files. It checks the published parameter hashes against its public stage payloads and records `original_checkpoint_files_verified: false`; the original local run's stronger private file-binding result remains in the frozen evidence record.",
        "",
        "From the repository root, CPU metric and binding qualification is:",
        "",
        "```powershell",
        "python -X utf8 -B tools/replay_transition_public.py --check --out analyses/transition_20261003_public_check.json",
        "```",
        "",
        "GPU reproduction commands are:",
        "",
        "```powershell",
        "python -X utf8 -u -B tools/replay_transition_public.py --preflight --qualification analyses/transition_20261003_public_check.json --out runs/transition_20261003_public_preflight",
        "python -X utf8 -u -B tools/replay_transition_public.py --preflight-dir runs/transition_20261003_public_preflight --out runs/transition_20261003_public_replay",
        "```",
        "",
        "These commands are provided for independent reproduction; this publication export did not launch a GPU job. The frozen protocol and metric definitions are in [new/transition_100_200/PROTOCOL.md](../../new/transition_100_200/PROTOCOL.md) and [new/transition_100_200/metrics.py](../../new/transition_100_200/metrics.py).",
        "",
        "The excluded raw checkpoint and trace archive names, sizes and hashes are in [input_provenance.json](input_provenance.json).",
        "",
        "`tools/export_transition_evidence.py --verify-only` checks published table arithmetic, data hashes, source bindings, and local Markdown links without importing the model runtime.",
        "",
    ]), encoding="utf-8", newline="\n")

    # Confirm the wrapper and validator inputs have arrived before sealing the manifest.
    for relpath in PUBLIC_TOOL_PATHS:
        if not (ROOT / relpath).is_file():
            raise FileNotFoundError(f"public reproduction/validation tool is missing: {relpath}")
    if not (evidence_dir / "replay_reference").is_dir():
        raise FileNotFoundError("root-owned public replay_reference evidence is missing")

    exclusions = input_exclusion_records(run_dir, raw_json_hashes)
    run_metadata_public = read_json(evidence_dir / "run_metadata.json")
    write_json(evidence_dir / "input_provenance.json", {
        "schema_version": "transition-input-provenance-v1",
        "run_relative_name": run_rel.as_posix(),
        "source_base_commit": run_metadata_public["source_base_commit"],
        "frozen_execution_source_sha256": run_manifest["source_sha256"],
        "excluded_raw_artifacts": exclusions,
        "private_run_manifest_content": "omitted; its hash is recorded in the root publication manifest",
        "dispatch_receipts_and_machine_identifiers": "omitted",
    })

    all_evidence = evidence_files(evidence_dir, manifest_path)
    oversized_csv = [path.relative_to(ROOT).as_posix() for path in all_evidence if path.suffix.lower() == ".csv" and path.stat().st_size >= 1_000_000]
    if oversized_csv:
        raise ValueError("CSV file exceeds the staged-file size cap:\n" + "\n".join(oversized_csv))
    privacy_failures = privacy_scan(all_evidence)
    if privacy_failures:
        raise ValueError("publication privacy scan failed:\n" + "\n".join(privacy_failures))
    for path in all_evidence:
        if path.suffix.lower() == ".md":
            links = markdown_local_links(path, path.read_text(encoding="utf-8"))
            if links:
                raise ValueError("Markdown link check failed:\n" + "\n".join(links))

    tool_bindings = {}
    for relpath in PUBLIC_TOOL_PATHS:
        path = ROOT / relpath
        tool_bindings[relpath] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    evidence_hashes = file_hash_map(all_evidence)
    source_input_hashes = {
        "run/manifest.json": sha256_file(run_dir / "manifest.json"),
        "run/aggregate.json": sha256_file(run_dir / "aggregate.json"),
        "run/summary.json": sha256_file(run_dir / "summary.json"),
        "run/training.json": sha256_file(run_dir / "training.json"),
        "run/transition_profiles.png": sha256_file(run_dir / "transition_profiles.png"),
        "preflight/aggregate.json": sha256_file(preflight_aggregate_path),
        "validation/source.json": validation_hash,
    }
    publication_manifest = {
        "schema_version": "transition-publication-manifest-v1",
        "study": "seed4 dense transition audit",
        "protocol": "seed4_dense_transition_v1",
        "frozen_execution_base_commit": run_manifest.get("git_base"),
        "evidence_head_commit": None,
        "evidence_head_note": "The containing evidence commit is recorded in GPT_HANDOFF.md after commit; this file cannot self-reference its own commit.",
        "run_identity": {
            "relative_name": run_rel.as_posix(),
            "status": "COMPLETE",
            "updates": 300,
            "audits": 44,
            "profile_sizes": [32, 64],
            "primary_size": 32,
            "dense_checkpoints": [int(value) for value in run_manifest["audit_updates"] if value <= 200],
            "separate_control": 300,
        },
        "source_bindings": {
            "frozen_execution_sha256": run_manifest["source_sha256"],
            "publication_and_validation_tools": tool_bindings,
            "run_input_artifact_sha256": source_input_hashes,
        },
        "excluded_private_artifact_count": len(exclusions),
        "excluded_private_artifact_manifest": "evidence/transition_20261003/input_provenance.json",
        "public_evidence_sha256": evidence_hashes,
        "validation": {
            "source_relative_name": validation_rel.as_posix(),
            "source_sha256": validation_hash,
        "sanitized_copy": "evidence/transition_20261003/validation/local_validation.json",
        },
        "privacy": {
            "absolute_paths": "excluded",
            "machine_identifiers_and_dispatch_receipts": "excluded",
            "checkpoints_and_raw_npz_traces": "excluded; names and hashes retained in input provenance",
        },
    }
    write_json(manifest_path, publication_manifest)
    return publication_manifest


def parse_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def as_int(value: str) -> int:
    return int(value)


def as_float(value: str) -> float | None:
    if value == "" or value.lower() == "none":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite CSV value: {value}")
    return result


def verify_profile_tables(evidence_dir: Path) -> dict[str, Any]:
    _, profiles = parse_csv(evidence_dir / "profiles.csv")
    if len(profiles) != 44:
        raise ValueError(f"profiles.csv has {len(profiles)} rows, expected 44")
    keys = {(as_int(row["update"]), as_int(row["size"])) for row in profiles}
    updates = tuple(range(100, 201, 5)) + (300,)
    expected = {(update, size) for update in updates for size in SIZES}
    if keys != expected:
        raise ValueError("profiles.csv update/size grid differs from the frozen audit")
    grouped_counts: dict[tuple[int, int, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    map_counts: list[dict[str, str]] = []
    strict_path = evidence_dir / "strict_coverage_counts.csv"
    _, strict_rows = parse_csv(strict_path)
    if len(strict_rows) != 44 * 16 * len(STRICT_STEPS):
        raise ValueError("strict_coverage_counts.csv row count differs from the frozen grid")
    map_counts.extend(strict_rows)
    for band in ("all_changed",) + DISTANCE_BANDS:
        path = evidence_dir / f"transition_counts_{band}.csv"
        _, rows = parse_csv(path)
        if len(rows) != 44 * 16 * len(INTERVALS) * 5:
            raise ValueError(f"{path.name} row count differs from the frozen grid")
        map_counts.extend(rows)
    expected_map_rows = 44 * 16 * (len(STRICT_STEPS) + (len(DISTANCE_BANDS) + 1) * len(INTERVALS) * 5)
    if len(map_counts) != expected_map_rows:
        raise ValueError(f"transition count files have {len(map_counts)} rows, expected {expected_map_rows}")
    for item in map_counts:
        update, size, kind = as_int(item["update"]), as_int(item["size"]), item["kind"]
        band, interval, metric = item["distance_band"], item["interval"], item["metric"]
        numerator, denominator = as_int(item["numerator"]), as_int(item["denominator"])
        rate = as_float(item["rate"])
        if denominator < 0 or (metric != "net_gain" and numerator < 0):
            raise ValueError(f"invalid count at {update}/{size}/{band}/{interval}/{metric}")
        expected_rate = None if denominator == 0 else numerator / denominator
        if (rate is None) != (expected_rate is None) or (rate is not None and abs(rate - expected_rate) > 1e-12):
            raise ValueError(f"rate/count mismatch at {update}/{size}/{band}/{interval}/{metric}")
        grouped_counts[(update, size, kind, interval, metric, band)].append(item)

    profile_by_key = {(as_int(row["update"]), as_int(row["size"])): row for row in profiles}
    for (update, size, kind, interval, metric, band), counts in grouped_counts.items():
        if len(counts) != 16 or {as_int(item["map_index"]) for item in counts} != set(range(16)):
            raise ValueError(f"map denominator coverage mismatch for {update}/{size}/{kind}/{interval}/{metric}/{band}")
        if kind == "strict_coverage":
            prefix = f"strict_{interval}"
            public_metric = "coverage"
            if band != "17_31":
                raise ValueError("strict coverage rows must use the primary 17_31 cohort")
        elif band == "all_changed":
            prefix_map = {
                "G": f"G_{interval}",
                "D_endpoint": f"D_endpoint_{interval}",
                "D_first_exit": f"D_first_exit_{interval}",
                "S_continuous": f"S_continuous_{interval}",
                "net_gain": f"net_gain_{interval}",
            }
            prefix = prefix_map[metric]
            public_metric = metric
        else:
            continue
        row = profile_by_key[(update, size)]
        numerators = [as_int(item["numerator"]) for item in counts]
        denominators = [as_int(item["denominator"]) for item in counts]
        eligible = [num / den for num, den in zip(numerators, denominators) if den > 0]
        equal_mean = sum(eligible) / len(eligible) if eligible else None
        pooled_num, pooled_den = sum(numerators), sum(denominators)
        pooled_rate = pooled_num / pooled_den if pooled_den else None
        if prefix == "strict_T64" or prefix == "strict_T128" or prefix == "strict_T256":
            eq_key = f"{prefix}_equal_map_mean"
            pooled_num_key = f"{prefix}_pooled_numerator"
            pooled_den_key = f"{prefix}_pooled_denominator"
            pooled_rate_key = f"{prefix}_pooled_rate"
        else:
            eq_key = f"{prefix}_equal_map_mean"
            pooled_num_key = f"{prefix}_pooled_numerator"
            pooled_den_key = f"{prefix}_pooled_denominator"
            pooled_rate_key = f"{prefix}_pooled_rate"
        for column, calculated in ((eq_key, equal_mean), (pooled_num_key, pooled_num), (pooled_den_key, pooled_den), (pooled_rate_key, pooled_rate)):
            observed = as_float(row[column]) if column not in {pooled_num_key, pooled_den_key} else as_int(row[column])
            if calculated is None:
                if observed is not None:
                    raise ValueError(f"profile aggregate mismatch in {column} at {update}/{size}")
            elif isinstance(calculated, int):
                if observed != calculated:
                    raise ValueError(f"profile count mismatch in {column} at {update}/{size}")
            elif observed is None or abs(observed - calculated) > 1e-12:
                raise ValueError(f"profile rate mismatch in {column} at {update}/{size}")
    for update, size in expected:
        for band in ("all_changed",) + DISTANCE_BANDS:
            for interval in INTERVALS:
                acquired = {as_int(row["map_index"]): as_int(row["numerator"]) for row in grouped_counts[(update, size, "transition", interval, "G", band)]}
                destroyed = {as_int(row["map_index"]): as_int(row["numerator"]) for row in grouped_counts[(update, size, "transition", interval, "D_endpoint", band)]}
                net = {as_int(row["map_index"]): as_int(row["numerator"]) for row in grouped_counts[(update, size, "transition", interval, "net_gain", band)]}
                if any(net[index] != acquired[index] - destroyed[index] for index in range(16)):
                    raise ValueError(f"per-map acquisition/destruction/net identity failed at {update}/{size}/{band}/{interval}")

    summary_row_count = 0
    for band in DISTANCE_BANDS:
        _, age_rows = parse_csv(evidence_dir / f"age_hazard_{band}.csv")
        _, fixed_rows = parse_csv(evidence_dir / f"fixed_lag_survival_{band}.csv")
        _, lag_rows = parse_csv(evidence_dir / f"finite_trace_lag_medians_{band}.csv")
        if len(age_rows) != 44 * 16 * 8:
            raise ValueError(f"age_hazard_{band}.csv row count differs from the frozen grid")
        if len(fixed_rows) != 44 * 16 * 5:
            raise ValueError(f"fixed_lag_survival_{band}.csv row count differs from the frozen grid")
        if len(lag_rows) != 44 * 16 * 4:
            raise ValueError(f"finite_trace_lag_medians_{band}.csv row count differs from the frozen grid")
        for item in age_rows:
            numerator = as_int(item["loss_events"])
            denominator = as_int(item["at_risk_cell_steps"])
            rate = as_float(item["next_step_loss_hazard"])
            expected_rate = numerator / denominator if denominator else None
            if numerator < 0 or denominator < numerator or (rate is None) != (expected_rate is None):
                raise ValueError(f"invalid age-hazard count in age_hazard_{band}.csv")
            if rate is not None and abs(rate - expected_rate) > 1e-12:
                raise ValueError(f"age-hazard rate/count mismatch in age_hazard_{band}.csv")
        for item in fixed_rows:
            numerator = as_int(item["survived_first_correct_spell"])
            denominator = as_int(item["eligible_first_correct_spells"])
            rate = as_float(item["uninterrupted_survival"])
            expected_rate = numerator / denominator if denominator else None
            never, censored = as_int(item["never_correct"]), as_int(item["right_censored"])
            if numerator < 0 or denominator < numerator or never < 0 or censored < 0:
                raise ValueError(f"invalid fixed-lag count in fixed_lag_survival_{band}.csv")
            if (rate is None) != (expected_rate is None) or (rate is not None and abs(rate - expected_rate) > 1e-12):
                raise ValueError(f"fixed-lag rate/count mismatch in fixed_lag_survival_{band}.csv")
        for item in lag_rows:
            selected, ever = as_int(item["selected_pixels"]), as_int(item["ever_correct"])
            never, final_correct = as_int(item["never_correct"]), as_int(item["final_correct"])
            final_wrong, n = as_int(item["final_wrong"]), as_int(item["n"])
            median = as_float(item["median_macro_step"])
            if ever + never != selected or final_correct + final_wrong != selected or n < 0:
                raise ValueError(f"finite-trace lag conditioning counts do not sum in {band}")
            if (median is None) != (n == 0):
                raise ValueError(f"finite-trace lag median eligibility mismatch in {band}")
        summary_row_count += len(age_rows) + len(fixed_rows) + len(lag_rows)
    return {
        "profile_rows": len(profiles),
        "transition_count_rows": len(map_counts),
        "age_hazard_rows": 44 * 16 * len(DISTANCE_BANDS) * 8,
        "fixed_lag_rows": 44 * 16 * len(DISTANCE_BANDS) * 5,
        "lag_median_rows": 44 * 16 * len(DISTANCE_BANDS) * 4,
        "stratified_diagnostic_rows": summary_row_count,
    }


def verify_manifest(manifest_path: Path, evidence_dir: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "transition-publication-manifest-v1":
        raise ValueError("unexpected publication manifest schema")
    mismatches = []
    for relpath, expected in manifest["public_evidence_sha256"].items():
        path = ROOT / PurePosixPath(relpath)
        if not path.is_file():
            mismatches.append(f"missing public evidence file {relpath}")
        elif sha256_file(path) != expected:
            mismatches.append(f"public evidence hash mismatch {relpath}")
    for relpath, expected in manifest["source_bindings"]["frozen_execution_sha256"].items():
        path = ROOT / PurePosixPath(relpath)
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(f"frozen source binding mismatch {relpath}")
    for relpath, binding in manifest["source_bindings"]["publication_and_validation_tools"].items():
        path = ROOT / PurePosixPath(relpath)
        if not path.is_file() or path.stat().st_size != binding["bytes"] or sha256_file(path) != binding["sha256"]:
            mismatches.append(f"publication tool binding mismatch {relpath}")
    if mismatches:
        raise ValueError("manifest hash verification failed:\n" + "\n".join(mismatches))

    # Public reference evidence remains root-owned; verify that it is covered by the manifest.
    for path in (evidence_dir / "replay_reference").rglob("*"):
        if path.is_file():
            relpath = path.relative_to(ROOT).as_posix()
            if relpath not in manifest["public_evidence_sha256"]:
                raise ValueError(f"root-owned public reference file is not hashed: {relpath}")
    return {"evidence_files": len(manifest["public_evidence_sha256"]), "tool_bindings": len(manifest["source_bindings"]["publication_and_validation_tools"])}


def verify_only(manifest_rel: Path, evidence_rel: Path) -> dict[str, Any]:
    manifest_path = ROOT / manifest_rel
    evidence_dir = ROOT / evidence_rel
    manifest_result = verify_manifest(manifest_path, evidence_dir)
    table_result = verify_profile_tables(evidence_dir)
    text_paths = evidence_files(evidence_dir, manifest_path)
    privacy_failures = privacy_scan(text_paths)
    if privacy_failures:
        raise ValueError("publication privacy verification failed:\n" + "\n".join(privacy_failures))
    broken = []
    for path in text_paths:
        if path.suffix.lower() == ".csv" and path.stat().st_size >= 1_000_000:
            raise ValueError(f"CSV file exceeds the staged-file size cap: {path.relative_to(ROOT).as_posix()}")
        if path.suffix.lower() == ".md":
            broken.extend(markdown_local_links(path, path.read_text(encoding="utf-8")))
    if broken:
        raise ValueError("publication Markdown links failed:\n" + "\n".join(broken))
    return {"status": "PASS", **manifest_result, **table_result, "relative_markdown_links": "PASS", "privacy_scan": "PASS"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="completed run relative to the repository root")
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION, help="counter-validation JSON relative to the repository root")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE, help="public evidence directory relative to the repository root")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="root publication manifest path relative to the repository root")
    parser.add_argument("--verify-only", action="store_true", help="check existing public outputs; do not read private run artifacts or rewrite files")
    args = parser.parse_args()
    for label, path in (("run", args.run), ("validation", args.validation), ("evidence", args.evidence), ("manifest", args.manifest)):
        if path.is_absolute() or ".." in path.parts:
            parser.error(f"--{label} must be a repository-relative path contained in the repository")
    try:
        if args.verify_only:
            result = verify_only(args.manifest, args.evidence)
        else:
            result = build(args.run, args.validation, args.evidence, args.manifest)
            result = {"status": "EXPORTED", "evidence_files": len(result["public_evidence_sha256"]), "manifest": args.manifest.as_posix()}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"transition evidence export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
