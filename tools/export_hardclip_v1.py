"""Publish and CPU-verify the completed HardClip v1 study from saved data."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from export_continuous_coverage import (
    MAX_FILE_BYTES,
    bank_tensor_hash,
    close,
    compare_trace_metrics,
    copy_bound,
    decode_trace,
    gzip_json,
    joint_pass,
    load_bank,
    read,
    safe,
    sha,
    trace_metrics,
    write,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/hardclip_v1_20261006_01"
CALIBRATION = ROOT / "runs/hardclip_v1_calibration_20261006_01"
QUALIFICATION = ROOT / "analyses/hardclip_v1_qualification_20261006_01.json"
PUBLIC = ROOT / "evidence/hardclip_v1_20261006"
MANIFEST = ROOT / "HARDCLIP_V1_PUBLICATION_MANIFEST.json"
PROTOCOL = "hardclip_v1_fixed_tail_v1"
CALIBRATION_PROTOCOL = "hardclip_v1_tail_calibration_v1"
ARMS = ("neural", "hardclip")
BLOCKS = 8
CHECKPOINTS = tuple(range(0, 301, 25))
SIZES = (32, 64)
LANES = ("N", "E", "S", "W")
LAUNCHER = "tools/launch_hardclip_v1.ps1"
REPORTING = "new/hardclip_v1/reporting.py"
METRICS = "new/continuous_coverage/metrics.py"
HELPER = "tools/export_continuous_coverage.py"
PRIVATE_KEYS = {"pid", "host", "hostname", "username", "gpu_uuid", "executable",
                "command", "launch_command", "cwd"}
PRIVATE_TEXT = re.compile(
    r"\b[A-Za-z]:[\\/]|\\\\[^\\\s]+[\\/]|(?<![\w.])/(?:[^/\s]+/)+[^/\s]+|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b|BEGIN [^\n]*PRIVATE KEY|"
    r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|"
    r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}"
)

_DOSE_FIELDS = (
    "phase", "lane", "lane_name", "event_count", "trigger_count", "raw_rms_sum",
    "removed_rms_sum", "trigger_fraction", "removed_fraction", "actual_clipping",
    "interpretation",
)
_FINAL_METRIC_KEYS = (
    "R_strict_pooled_T64", "R_strict_mean_T64", "S_retention64_to256",
    "S_retention_numerator", "S_continuous_survival64_to256",
    "retention_reference_pixels", "retention_reference_maps",
)


def load_module(relative: str, name: str):
    if str(ROOT / "new") not in sys.path:
        sys.path.insert(0, str(ROOT / "new"))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AssertionError(f"Boolean is not a numeric dose value: {label}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AssertionError(f"Invalid numeric dose value: {label}={value!r}") from error
    assert math.isfinite(result), (label, value)
    return result


def _integer(value: Any, label: str) -> int:
    assert not isinstance(value, bool), (label, value)
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AssertionError(f"Invalid integer: {label}={value!r}") from error
    assert result == value and result >= 0, (label, value)
    return result


def _launcher_transform_is_limited(snapshot: Path, current: Path) -> bool:
    """Accept only the confirmed local-Python-path sanitization in this launcher."""
    old_lines = snapshot.read_text(encoding="utf-8-sig").splitlines()
    new_lines = current.read_text(encoding="utf-8-sig").splitlines()
    if len(old_lines) != len(new_lines):
        return False
    differences = [i for i, (old, new) in enumerate(zip(old_lines, new_lines)) if old != new]
    if len(differences) != 2:
        return False
    first, second = differences
    old_python = old_lines[first].strip()
    new_python = new_lines[first].strip()
    old_presence = old_lines[second].strip()
    new_presence = new_lines[second].strip()
    old_path_pattern = re.compile(
        r"^\$pythonPath='[A-Za-z]:\\Users\\[^\\']+\\anaconda3\\python\.exe'$",
        re.IGNORECASE,
    )
    return bool(
        old_path_pattern.fullmatch(old_python)
        and new_python == "$pythonPath=(Get-Command python -CommandType Application | Select-Object -First 1).Source"
        and old_presence == "if (-not (Test-Path -LiteralPath $pythonPath)) {throw 'Historical local Python unavailable.'}"
        and new_presence == "if (-not $pythonPath) {throw 'Historical local Python unavailable.'}"
    )


def source_bindings(run_manifest: dict[str, Any], run_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    bindings: dict[str, Any] = {}
    published_hashes: dict[str, str] = {}
    for relative, expected in run_manifest["source_sha256"].items():
        snapshot = run_dir / "source" / relative
        current = ROOT / relative
        assert snapshot.is_file() and current.is_file(), relative
        snapshot_hash = sha(snapshot)
        current_hash = sha(current)
        assert snapshot_hash == expected, ("snapshot hash mismatch", relative)
        if current_hash == expected:
            status = "EXACT"
            transform = None
        elif relative == LAUNCHER and _launcher_transform_is_limited(snapshot, current):
            status = "SANITIZED_LAUNCHER_PYTHON_DISCOVERY"
            transform = (
                "Replaced the local absolute Python executable with Get-Command python; "
                "changed the presence guard to test the resolved string. No scientific code changed."
            )
        else:
            raise RuntimeError(f"Unapproved current-source difference: {relative}")
        bindings[relative] = {
            "original_run_snapshot_sha256": snapshot_hash,
            "current_public_sha256": current_hash,
            "status": status,
            **({"permitted_transform": transform} if transform else {}),
        }
        published_hashes[relative] = current_hash
    return bindings, published_hashes


def _csv_text(fields: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _write_csv(path: Path, fields: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_csv_text(fields, rows).encode("utf-8"))


def _dose_consistent(row: dict[str, Any], label: str, actual_clipping: bool) -> dict[str, Any]:
    lane = _integer(row.get("lane"), label + ".lane")
    assert lane in range(4), (label, lane)
    assert row.get("lane_name") == LANES[lane], (label, row.get("lane_name"))
    events = _integer(row.get("event_count"), label + ".event_count")
    triggers = _integer(row.get("trigger_count"), label + ".trigger_count")
    assert triggers <= events, (label, events, triggers)
    raw = _finite_number(row.get("raw_rms_sum"), label + ".raw_rms_sum")
    removed = _finite_number(row.get("removed_rms_sum"), label + ".removed_rms_sum")
    assert raw is not None and removed is not None and raw >= 0 and removed >= 0
    assert row.get("actual_clipping") is actual_clipping, (label, row.get("actual_clipping"))
    expected_interpretation = "actual" if actual_clipping else "counterfactual_fixed_cap"
    assert row.get("interpretation") == expected_interpretation, label
    trigger_fraction = row.get("trigger_fraction")
    if events:
        close(trigger_fraction, triggers / events, label + ".trigger_fraction")
    else:
        assert trigger_fraction is None, (label, trigger_fraction)
    removed_fraction = row.get("removed_fraction")
    if raw > 0:
        close(removed_fraction, removed / raw, label + ".removed_fraction")
    else:
        assert removed == 0 and removed_fraction is None, (label, removed, removed_fraction)
    for key in _DOSE_FIELDS:
        assert key in row, (label, key)
    return {
        "lane": lane,
        "event_count": events,
        "trigger_count": triggers,
        "raw_rms_sum": raw,
        "removed_rms_sum": removed,
        "trigger_fraction": trigger_fraction,
        "removed_fraction": removed_fraction,
    }


def _training_dose_data(public: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_block: dict[tuple[int, str, int], dict[str, float | int]] = {}
    aggregate: dict[tuple[str, int], dict[str, float | int]] = {}
    curve_count = 0
    for block in range(BLOCKS):
        for arm in ARMS:
            curve_path = public / f"block{block:02d}/{arm}/training_curve.json.gz"
            curve = gzip_json(curve_path)
            assert len(curve) == 300, (curve_path, len(curve))
            assert [entry.get("update") for entry in curve] == list(range(1, 301)), curve_path
            for entry in curve:
                update = entry["update"]
                assert entry.get("forward_steps") == 256
                assert entry.get("backward_calls") == entry.get("loss_count") == 32
                assert entry.get("detach_after_loss") == 32
                assert entry.get("optimizer_steps") == 1 and entry.get("credit_horizon") == 8
                dose_rows = entry.get("clip_dose")
                assert isinstance(dose_rows, list) and len(dose_rows) == 4
                assert {row.get("lane") for row in dose_rows} == set(range(4))
                for dose in dose_rows:
                    lane = dose["lane"]
                    checked = _dose_consistent(
                        dose, f"block{block}/{arm}/u{update}/lane{lane}", arm == "hardclip"
                    )
                    assert dose.get("phase") == "cold_1_64_repeated_four_times"
                    rows.append(
                        {
                            "block": block,
                            "arm": arm,
                            "update": update,
                            **{key: dose.get(key) for key in _DOSE_FIELDS},
                        }
                    )
                    key = (block, arm, lane)
                    bucket = by_block.setdefault(
                        key,
                        {"event_count": 0, "trigger_count": 0, "raw_rms_sum": 0.0, "removed_rms_sum": 0.0},
                    )
                    total = aggregate.setdefault(
                        (arm, lane),
                        {"event_count": 0, "trigger_count": 0, "raw_rms_sum": 0.0, "removed_rms_sum": 0.0},
                    )
                    for target in (bucket, total):
                        target["event_count"] += checked["event_count"]
                        target["trigger_count"] += checked["trigger_count"]
                        target["raw_rms_sum"] += checked["raw_rms_sum"]
                        target["removed_rms_sum"] += checked["removed_rms_sum"]
            curve_count += 1
    assert curve_count == BLOCKS * len(ARMS) == 16
    assert len(rows) == BLOCKS * len(ARMS) * 300 * 4 == 19200

    aggregate_by_arm: dict[str, Any] = {}
    for arm in ARMS:
        lanes = []
        for lane in range(4):
            values = aggregate[(arm, lane)]
            events = int(values["event_count"])
            raw = float(values["raw_rms_sum"])
            removed = float(values["removed_rms_sum"])
            lanes.append(
                {
                    "lane": lane,
                    "lane_name": LANES[lane],
                    "event_count": events,
                    "trigger_count": int(values["trigger_count"]),
                    "trigger_fraction": values["trigger_count"] / events if events else None,
                    "raw_rms_sum": raw,
                    "removed_rms_sum": removed,
                    "removed_fraction": removed / raw if raw else None,
                    "actual_clipping": arm == "hardclip",
                    "interpretation": "actual" if arm == "hardclip" else "counterfactual_fixed_cap",
                }
            )
        aggregate_by_arm[arm] = lanes

    block_rows = []
    no_trigger_blocks: dict[str, list[int]] = {arm: [] for arm in ARMS}
    for block in range(BLOCKS):
        for arm in ARMS:
            values = [by_block[(block, arm, lane)] for lane in range(4)]
            events = sum(int(item["event_count"]) for item in values)
            triggers = sum(int(item["trigger_count"]) for item in values)
            raw = sum(float(item["raw_rms_sum"]) for item in values)
            removed = sum(float(item["removed_rms_sum"]) for item in values)
            triggered_lanes = [
                LANES[lane]
                for lane, item in enumerate(values)
                if int(item["trigger_count"]) > 0
            ]
            if triggers == 0:
                no_trigger_blocks[arm].append(block)
            block_rows.append(
                {
                    "block": block,
                    "arm": arm,
                    "event_count": events,
                    "trigger_count": triggers,
                    "triggered_lanes": triggered_lanes,
                    "raw_rms_sum": raw,
                    "removed_rms_sum": removed,
                    "trigger_fraction": triggers / events if events else None,
                    "removed_fraction": removed / raw if raw else None,
                    "actual_clipping": arm == "hardclip",
                }
            )

    summary = {
        "schema": "hardclip-v1-training-dose-summary-v1",
        "updates_per_trajectory": 300,
        "trajectories": BLOCKS * len(ARMS),
        "training_dose_csv_rows": len(rows),
        "aggregation": "Sum lane-cell event counts and RMS sums over all 300 super-updates; zero denominators remain null.",
        "by_arm_lane": aggregate_by_arm,
        "by_block": block_rows,
        "zero_trigger_blocks": no_trigger_blocks,
        "interpretation_boundary": (
            "Neural rows are counterfactual clipping dose. Zero observed trigger dose in a block "
            "limits this fixed intervention's negative efficacy result; it does not reject all large-write mechanisms."
        ),
    }
    return rows, summary


def _training_dose_csv(public: Path) -> str:
    rows, _ = _training_dose_data(public)
    fields = (
        "block", "arm", "update", *_DOSE_FIELDS,
    )
    return _csv_text(fields, rows)


def _evaluation_dose_summary(public: Path, dense: list[dict[str, Any]], raw_bindings: dict[str, Any]) -> dict[str, Any]:
    expected_groups: set[tuple[int, str, int, int, str, str]] = set()
    lane_rows = 0
    for record in dense:
        block, arm, update = record["block"], record["arm"], record["update"]
        assert arm in ARMS and update in CHECKPOINTS
        run_relative = record["clip_dose"]
        binding = raw_bindings[run_relative]
        dose_rows = gzip_json(ROOT / binding["public_path"])
        assert isinstance(dose_rows, list) and len(dose_rows) == 32, run_relative
        groups: dict[tuple[int, str, str], set[int]] = {}
        for dose in dose_rows:
            size = dose.get("size")
            world = dose.get("world")
            phase = dose.get("phase")
            assert size in SIZES and world in ("original", "flipped") and phase in ("1_64", "65_256")
            checked = _dose_consistent(dose, f"{run_relative}/size{size}/{world}/{phase}", arm == "hardclip")
            group = (size, world, phase)
            groups.setdefault(group, set()).add(checked["lane"])
            lane_rows += 1
        assert len(groups) == 8 and all(lanes == set(range(4)) for lanes in groups.values()), run_relative
        expected_groups.update((block, arm, update, size, world, phase) for size, world, phase in groups)
    expected_count = len(dense) * 8
    assert lane_rows == len(dense) * 32
    assert len(expected_groups) == expected_count
    return {
        "evaluation_dose_files_verified": len(dense),
        "size_world_phase_groups_verified": len(expected_groups),
        "lane_rows_verified": lane_rows,
        "groups_per_evaluation": 8,
        "rows_per_evaluation": 32,
        "all_counts_and_values_finite": True,
        "zero_raw_rms_keeps_removed_fraction_null": True,
        "actual_vs_counterfactual_arm_labels_verified": True,
    }


def _write_final_metrics(public: Path, final_rows: list[dict[str, Any]]) -> str:
    fields = ["block", "arm", "update", "initialization_seed", "schedule_seed", "joint_readiness", "old_full_pass"]
    fields.extend(f"{key}_{size}" for size in SIZES for key in _FINAL_METRIC_KEYS)
    rows = []
    for row in sorted(final_rows, key=lambda item: (item["block"], ARMS.index(item["arm"]))):
        result = {
            "block": row["block"],
            "arm": row["arm"],
            "update": row["update"],
            "initialization_seed": row["initialization_seed"],
            "schedule_seed": row["schedule_seed"],
            "joint_readiness": row["joint"]["pass"],
            "old_full_pass": row["metrics"]["full_pass"],
        }
        for size in SIZES:
            metrics = row["metrics"]["sizes"][str(size)]
            result.update({f"{key}_{size}": metrics.get(key) for key in _FINAL_METRIC_KEYS})
        rows.append(result)
    value = _csv_text(fields, rows)
    (public / "final_metrics.csv").write_bytes(value.encode("utf-8"))
    return value


def raw_artifacts() -> list[tuple[Path, str, str, str]]:
    items: list[tuple[Path, str, str, str]] = []
    for name in ("config.json", "dense.json", "metrics.csv", "perarm.json", "summary.json"):
        items.append((RUN / name, name, "byte-exact copy", name))
    items.append((RUN / "RESULTS.md", "frozen_RESULTS.md", "byte-exact copy", "RESULTS.md"))
    items.append((RUN / "plans.json", "plans.json.gz", "gzip lossless", "plans.json"))
    for name in ("train.npz", "evaluation32.npz", "evaluation64.npz"):
        items.append((RUN / "banks" / name, f"banks/{name}", "byte-exact copy", f"banks/{name}"))
    for block in range(BLOCKS):
        for arm in ARMS:
            folder = f"block{block:02d}/{arm}"
            items.append(
                (RUN / folder / "training_curve.json", f"{folder}/training_curve.json.gz", "gzip lossless", f"{folder}/training_curve.json")
            )
            for update in CHECKPOINTS:
                stage = f"{folder}/evaluation/u{update:03d}"
                for name in ("summary.json", "clip_dose.json"):
                    items.append(
                        (RUN / stage / name, f"{stage}/{name}.gz", "gzip lossless", f"{stage}/{name}")
                    )
                for size in SIZES:
                    name = f"size{size}_traces.npz"
                    items.append((RUN / stage / name, f"{stage}/{name}", "byte-exact copy", f"{stage}/{name}"))
                if update == 300:
                    name = "matched_frontier_strata.csv"
                    items.append((RUN / stage / name, f"{stage}/{name}.gz", "gzip lossless", f"{stage}/{name}"))
    for name, encoding in (
        ("config.json", "byte-exact copy"),
        ("summary.json", "byte-exact copy"),
        ("RESULTS.md", "byte-exact copy"),
        ("calibration_inputs.npz", "byte-exact copy"),
        ("raw_write_rms.npz", "byte-exact copy"),
    ):
        public_name = name
        items.append(
            (CALIBRATION / name, f"calibration/{public_name}", encoding, f"calibration/{name}")
        )
    return items


def _check_source_manifest(manifest: dict[str, Any], base: Path) -> tuple[dict[str, Any], dict[str, str]]:
    return source_bindings(manifest, base)


def _sanitized_qualification(raw: dict[str, Any]) -> dict[str, Any]:
    assert raw.get("status") == "PASS" and raw.get("protocol") == PROTOCOL
    arms = []
    for item in raw.get("arms", []):
        arms.append(
            {
                key: item[key]
                for key in (
                    "arm", "status", "parameter_count", "three_update_gradient_max_absolute_error",
                    "active_branch_exercised_all_lanes", "fixed_caps_unchanged",
                )
                if key in item
            }
        )
    sanitized = {
        "status": raw["status"],
        "protocol": raw["protocol"],
        "calibration_binding": {
            "actual_write_caps": raw["calibration_binding"]["actual_write_caps"],
            "summary_sha256": raw["calibration_binding"]["summary_sha256"],
            "parent_protocol": raw["calibration_binding"]["parent_protocol"],
            "outcome_blind": raw["calibration_binding"]["outcome_blind"],
        },
        "arms": arms,
        "actual_training_shape": raw["actual_training_shape"],
        "credit_horizon": raw["credit_horizon"],
        "eager_graph_updates_each_arm": raw["eager_graph_updates_each_arm"],
        "atol": raw["atol"],
        "rtol": raw["rtol"],
        "sanitization": "Retains CPU/graph equivalence and fixed-cap checks; removes device, memory, timing, and allocation telemetry.",
    }
    safe(sanitized)
    assert all(row.get("status") == "PASS" and row.get("fixed_caps_unchanged") is True for row in arms)
    return sanitized


def _validate_calibration(cal_summary: dict[str, Any], cal_config: dict[str, Any], run_config: dict[str, Any],
                          run_manifest: dict[str, Any], raw_samples: np.ndarray, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
    assert cal_summary["status"] == "CALIBRATED"
    assert cal_summary["protocol"] == CALIBRATION_PROTOCOL
    assert cal_summary["completed_blocks"] == cal_summary["expected_blocks"] == BLOCKS
    assert cal_summary["training_updates"] == 0 and cal_summary["task_outcomes_used"] is False
    assert raw_samples.ndim == 4 and raw_samples.shape[0] == BLOCKS and raw_samples.shape[1] == 64 and raw_samples.shape[-1] == 4
    assert list(raw_samples.shape) == cal_summary["sampling_shape"]
    assert np.isfinite(raw_samples).all() and (raw_samples >= 0).all()
    assert inputs["x"].shape == (16, 3, 32, 32)
    assert inputs["open_mask"].shape == (16, 32, 32)
    assert int(inputs["open_mask"].sum()) == raw_samples.shape[2]
    assert np.array_equal(inputs["map_indices"], np.arange(16, dtype=np.int32))
    caps = np.asarray(cal_summary["frozen_caps"], dtype=np.float32)
    run_caps = np.asarray(run_config["actual_write_caps"], dtype=np.float32)
    assert caps.shape == run_caps.shape == (4,) and np.array_equal(caps, run_caps)
    assert np.array_equal(run_caps, np.asarray(run_manifest["calibration_binding"]["actual_write_caps"], dtype=np.float32))
    assert cal_config["caps_are_actual_write_rms"] is True
    assert cal_config["labels_used"] is False and cal_config["task_readout_called"] is False
    assert cal_config["optimizer_constructed"] is False
    for lane, row in enumerate(cal_summary["lanes"]):
        assert row["lane"] == lane and row["sparse_enough"] is True
        assert abs(row["fp32_removed_fraction"] - 0.01) <= 1e-7
        assert row["fp32_trigger_fraction"] <= 0.20
        close(row["frozen_fp32_write_cap"], float(caps[lane]), f"calibration.cap.{lane}")
        sampled = raw_samples[..., lane].reshape(-1).astype(np.float64)
        removed = float(np.maximum(sampled-float(caps[lane]), 0).sum()/sampled.sum())
        triggered = float(np.mean(sampled > float(caps[lane])))
        assert abs(removed-.01) <= 1e-7
        close(removed, row['fp32_removed_fraction'], f'calibration.sample_dose.{lane}')
        close(triggered, row['fp32_trigger_fraction'], f'calibration.sample_trigger.{lane}')
    return {
        "status": "PASS",
        "calibration_verdict": cal_summary["status"],
        "raw_write_rms_shape": list(raw_samples.shape),
        "raw_write_rms_finite_nonnegative": True,
        "input_maps_and_open_cell_index_binding": "PASS",
        "frozen_caps_match_run_and_qualification": True,
        "source_task_outcomes_or_training_updates_used": False,
    }


def _validate_packed_trace(path: Path, size: int) -> None:
    expected_shape = (257, 32, size, size)
    count = math.prod(expected_shape)
    with np.load(path, allow_pickle=False) as archive:
        for name in ("correct", "original_correct", "flipped_correct"):
            shape = tuple(int(value) for value in archive[f"{name}_shape"])
            packed = archive[f"{name}_packed"]
            assert shape == expected_shape, (path, name, shape)
            assert packed.dtype == np.uint8 and packed.size == (count + 7) // 8


def _verify_metric_csv(path: Path, dense: list[dict[str, Any]]) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == BLOCKS * len(ARMS) * len(CHECKPOINTS) * len(SIZES) == 416
    expected = {(row["block"], row["arm"], row["update"]): row for row in dense}
    observed = set()
    for row in rows:
        key = (int(row["block"]), row["arm"], int(row["update"]))
        assert key in expected and row["size"] in ("32", "64")
        observed.add((*key, int(row["size"])))
    assert len(observed) == 416


def _verify_final_metrics_csv(path: Path, final: list[dict[str, Any]]) -> None:
    fields = ["block", "arm", "update", "initialization_seed", "schedule_seed", "joint_readiness", "old_full_pass"]
    fields.extend(f"{key}_{size}" for size in SIZES for key in _FINAL_METRIC_KEYS)
    rows = []
    for row in sorted(final, key=lambda item: (item["block"], ARMS.index(item["arm"]))):
        result = {
            "block": row["block"], "arm": row["arm"], "update": row["update"],
            "initialization_seed": row["initialization_seed"], "schedule_seed": row["schedule_seed"],
            "joint_readiness": row["joint"]["pass"], "old_full_pass": row["metrics"]["full_pass"],
        }
        for size in SIZES:
            metrics = row["metrics"]["sizes"][str(size)]
            result.update({f"{key}_{size}": metrics.get(key) for key in _FINAL_METRIC_KEYS})
        rows.append(result)
    assert path.read_bytes() == _csv_text(fields, rows).encode("utf-8")


def measure(publication: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recheck the public package from saved NumPy/JSON/CSV data only."""
    publication = read(MANIFEST) if publication is None else publication
    summary = read(PUBLIC / "summary.json")
    dense = read(PUBLIC / "dense.json")
    final = read(PUBLIC / "perarm.json")
    run_config = read(PUBLIC / "config.json")
    cal_summary = read(PUBLIC / "calibration/summary.json")
    cal_config = read(PUBLIC / "calibration/config.json")
    expected_keys = {(block, arm, update) for block in range(BLOCKS) for arm in ARMS for update in CHECKPOINTS}
    observed_keys = {(row["block"], row["arm"], row["update"]) for row in dense}
    assert len(dense) == 208 and observed_keys == expected_keys
    assert len(final) == 16
    assert {(row["block"], row["arm"], row["update"]) for row in final} == {
        (block, arm, 300) for block in range(BLOCKS) for arm in ARMS
    }
    assert summary["status"] == "COMPLETE" and summary["protocol"] == PROTOCOL
    assert summary["completed_arms"] == summary["expected_arms"] == 16 and summary["dense_records"] == 208

    reporting = load_module(REPORTING, "hardclip_public_reporting")
    metrics_module = load_module(METRICS, "hardclip_public_metrics")
    aggregate = reporting.aggregate(final, dense, expected_blocks=BLOCKS)
    for key, value in aggregate.items():
        assert summary[key] == value, key

    banks = {size: load_bank(PUBLIC / f"banks/evaluation{size}.npz") for size in SIZES}
    bank_hashes = {
        name: bank_tensor_hash(load_bank(PUBLIC / f"banks/{filename}"))
        for name, filename in (("train", "train.npz"), ("evaluation32", "evaluation32.npz"),
                               ("evaluation64", "evaluation64.npz"))
    }
    assert bank_hashes == publication["data_sha256"]

    # Validate every saved checkpoint summary and every packed trace's stored shape.
    for row in dense:
        folder = PUBLIC / f"block{row['block']:02d}/{row['arm']}/evaluation/u{row['update']:03d}"
        saved = gzip_json(folder / "summary.json.gz")
        assert metrics_module.compact_pair_metrics(saved) == row["metrics"]
        assert metrics_module.joint_readiness(saved) == row["joint"]
        assert row["formal_endpoint"] is (row["update"] == 300)
        assert row["checkpoint_selection"] is False
        assert saved["full_evaluated"] is (row["update"] == 300)
        for size in SIZES:
            _validate_packed_trace(folder / f"size{size}_traces.npz", size)

    trace_rows = {}
    for row in final:
        block, arm = row["block"], row["arm"]
        folder = PUBLIC / f"block{block:02d}/{arm}/evaluation/u300"
        saved = gzip_json(folder / "summary.json.gz")
        assert saved["phenotype_gate"]["pass"] == row["metrics"]["full_pass"]
        for size in SIZES:
            traces = decode_trace(folder / f"size{size}_traces.npz", size)
            assert np.array_equal(traces["correct"], traces["original_correct"] & traces["flipped_correct"])
            measured = trace_metrics(traces["correct"], banks[size])
            compare_trace_metrics(measured, row["metrics"]["sizes"][str(size)], f"block{block}/{arm}/u300/size{size}")
            trace_rows[(block, arm, size)] = measured
            if size == 32:
                assert joint_pass(measured) == row["joint"]["pass"]

    _verify_metric_csv(PUBLIC / "metrics.csv", dense)
    _verify_final_metrics_csv(PUBLIC / "final_metrics.csv", final)
    training_rows, training_summary = _training_dose_data(PUBLIC)
    assert len(training_rows) == 19200
    assert (PUBLIC / "training_dose.csv").read_bytes() == _csv_text(
        ("block", "arm", "update", *_DOSE_FIELDS), training_rows
    ).encode("utf-8")
    assert read(PUBLIC / "dose_summary.json") == training_summary
    evaluation_dose = _evaluation_dose_summary(PUBLIC, dense, publication["raw_artifact_bindings"])

    with np.load(PUBLIC / "calibration/raw_write_rms.npz", allow_pickle=False) as archive:
        raw_samples = archive["raw_write_rms"]
        assert archive["blocks"].tolist() == list(range(BLOCKS))
        assert archive["times"].tolist() == list(range(1, 65))
        sample_indices = archive["open_cell_indices"]
    with np.load(PUBLIC / "calibration/calibration_inputs.npz", allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    assert np.array_equal(sample_indices, np.argwhere(inputs["open_mask"]))
    calibration_validation = _validate_calibration(
        cal_summary, cal_config, run_config, publication, raw_samples, inputs
    )
    qualification = read(PUBLIC / "runtime_qualification.json")
    safe(qualification)
    assert qualification["status"] == "PASS" and qualification["protocol"] == PROTOCOL
    assert qualification["calibration_binding"]["actual_write_caps"] == run_config["actual_write_caps"]
    assert len(list(PUBLIC.rglob("size*_traces.npz"))) == 416
    assert len(list(PUBLIC.rglob("summary.json.gz"))) == 208
    assert len(list(PUBLIC.rglob("clip_dose.json.gz"))) == 208
    assert len(list(PUBLIC.rglob("training_curve.json.gz"))) == 16
    assert len(list(PUBLIC.rglob("matched_frontier_strata.csv.gz"))) == 16

    return {
        "status": "PASS",
        "scope": "CPU/NumPy/Gzip saved-data checks; no Torch, model inference, training, or checkpoint loading.",
        "record_keys": {"expected": 208, "observed": len(observed_keys), "unique": len(observed_keys)},
        "report_aggregate_matches_saved_summary": True,
        "evaluation_summaries_checked": 208,
        "packed_trace_files_shape_checked": 416,
        "u300_trace_metric_banks_recomputed": 32,
        "u300_original_flipped_and_rechecked": 32,
        "metrics_csv_rows": 416,
        "final_metrics_rows": 16,
        "training_curves": 16,
        "training_dose_rows": len(training_rows),
        "training_update_cadence": "256 forward steps, 32 loss/backward windows, 32 detaches, one optimizer step, horizon 8",
        "evaluation_dose": evaluation_dose,
        "calibration": calibration_validation,
        "data_bank_tensor_hashes_recomputed": bank_hashes,
        "checkpoint_hash_records_retained": len(publication["checkpoint_bindings"]),
        "checkpoint_contents_published": False,
        "old_Full_scope": "Saved u300 evaluator gate reused; not regenerated from traces.",
        "frozen_run_evidence_changed": False,
    }


def _write_results(summary: dict[str, Any], calibration: dict[str, Any], dose_summary: dict[str, Any]) -> None:
    per_arm = summary["per_arm"]
    primary = summary["primary_comparison"]
    lines = [
        "# HardClip v1 fixed-tail qualification",
        "",
        "The completed run has 16/16 trajectories and 208/208 predeclared evaluations. The formal endpoint is joint readiness at u300; the independent unit is the paired training block (n=8).",
        "",
        f"Calibration: **{calibration['status']}** at four frozen actual-write RMS caps; no task outcomes or training updates entered calibration.",
        f"Primary verdict: **{summary['overall_verdict']}**.",
        "",
        "| Arm | u300 joint-ready | Blocks | Wilson 95% readiness interval | Old Full pass (secondary) |",
        "|---|---:|---:|---|---:|",
    ]
    for arm in ARMS:
        ready = per_arm[arm]
        interval = ready["formal_readiness_wilson95"]
        ci = f"[{interval['lower']:.6g}, {interval['upper']:.6g}]" if interval else "—"
        full = summary["old_full_secondary"][arm]
        lines.append(
            f"| {arm} | {ready['formal_readiness_passes']} | {ready['valid_formal_blocks']}/{ready['expected_formal_blocks']} | {ci} | {full['passes']}/{full['observed']} |"
        )
    lines += [
        "",
        "## Sole primary contrast",
        "",
        "| Contrast | Complete pairs | HardClip-only wins | Neural-only wins | Net | Exact two-sided p |",
        "|---|---:|---:|---:|---:|---:|",
        f"| hardclip − neural | {primary['complete_pairs']} | {primary['wins']} | {primary['losses']} | {primary['net_gain']} | {primary['exact_two_sided_p']:.6g} |",
        "",
        summary["qualification_rule"],
        "",
        "## Observed training dose",
        "",
        "Dose sums pool the saved per-update open lane-cell events. Neural reports counterfactual clipping; HardClip reports actual clipping.",
        "",
        "| Arm | Zero-trigger blocks | Blocks with any trigger | Aggregate trigger fraction | Aggregate removed fraction |",
        "|---|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        block_summaries = [row for row in dose_summary["by_block"] if row["arm"] == arm]
        no_trigger = dose_summary["zero_trigger_blocks"][arm]
        values = [row for row in dose_summary["by_arm_lane"][arm]]
        events = sum(row["event_count"] for row in values)
        triggers = sum(row["trigger_count"] for row in values)
        raw = sum(row["raw_rms_sum"] for row in values)
        removed = sum(row["removed_rms_sum"] for row in values)
        lines.append(
            f"| {arm} | {', '.join(map(str, no_trigger)) if no_trigger else '—'} | "
            f"{sum(row['trigger_count'] > 0 for row in block_summaries)} | "
            f"{triggers / events if events else '—'} | {removed / raw if raw else '—'} |"
        )
    lines += [
        "",
        f"HardClip recorded zero trigger events in {len(dose_summary['zero_trigger_blocks']['hardclip'])}/8 paired blocks. The intervention dose was therefore absent in those blocks; this limits the negative result to the tested fixed intervention and does not refute all large-write mechanisms.",
        "",
        "Earlier checkpoints are formation diagnostics only; no peak checkpoint replaces u300. The Old Full gate is a saved u300 evaluator result and remains secondary.",
        "",
        summary["claim_boundary"],
        "",
        "See [REPRODUCTION.md](REPRODUCTION.md), [final_metrics.csv](final_metrics.csv), [metrics.csv](metrics.csv), [training_dose.csv](training_dose.csv), and [validation.json](validation.json). All 208 evaluation summaries and dose records, 416 packed trace banks, and 16 training curves are retained in the package.",
    ]
    (PUBLIC / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reproduction() -> None:
    text = """# HardClip v1 package verification

Start with [RESULTS.md](RESULTS.md), [final_metrics.csv](final_metrics.csv), and [dose_summary.json](dose_summary.json). The formal endpoint is joint readiness at u300 across eight paired blocks. `metrics.csv` retains both evaluated sizes at all 208 checkpoints; `training_dose.csv` retains the four lane summaries for each of 300 updates in each trajectory.

From the repository root, verify this package with saved data only:

```powershell
python -X utf8 -B tools/export_hardclip_v1.py --verify-only
```

Verification does not need the local run directory, qualification analysis, model checkpoints, PyTorch, or a GPU. It checks package and source hashes, schedule and data-bank bindings, all 208 record keys and evaluator summaries, all packed trace file shapes, the 32 u300 trace banks and paired Boolean AND, the reporter aggregate, training cadence and dose accounting, all saved evaluation-dose rows, and calibration sample/cap bindings. It does not load checkpoint files or regenerate the Old Full evaluator gate.

The package includes the frozen HardClip and calibration summaries, all lossless evaluation summaries and dose files, all 416 trace NPZ files, all 16 training curves, the data banks, and the block schedules. `HARDCLIP_V1_PUBLICATION_MANIFEST.json` binds raw-file hashes, published hashes, current source hashes, calibration inputs, and checkpoint file/parameter hashes. Checkpoint and optimizer contents, private run manifests/status/PIDs, and launch receipts are excluded. One launcher source was sanitized to remove its local Python path; the exact permitted source transform and both hashes are recorded in the manifest.

HardClip had zero trigger events across all lanes in five of the eight paired blocks. Neural-arm dose is counterfactual. This weak observed dose limits interpretation of the negative efficacy result to the fixed tested intervention. No broad large-write-mechanism conclusion follows.

The frozen protocol and implementation are in [PROTOCOL.md](../../new/hardclip_v1/PROTOCOL.md) and `new/hardclip_v1/`. Training requires the qualified CUDA environment; package verification is CPU-only and uses Python with NumPy.
"""
    (PUBLIC / "REPRODUCTION.md").write_text(text, encoding="utf-8")


def verify_files(publication: dict[str, Any]) -> None:
    safe(publication)
    actual = {path.relative_to(ROOT).as_posix(): sha(path) for path in PUBLIC.rglob("*") if path.is_file()}
    assert actual == publication["published_sha256"]
    assert len(actual) == publication["published_file_count"]
    assert sum((ROOT / relative).stat().st_size for relative in actual) == publication["published_bytes"]
    assert all((ROOT / relative).stat().st_size < MAX_FILE_BYTES for relative in actual)
    assert not any((ROOT / relative).suffix in {".pt", ".pth"} for relative in actual)
    assert len(publication["checkpoint_bindings"]) == 208

    for relative, expected in publication["source_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
    for relative, expected in publication["verification_source_sha256"].items():
        assert sha(ROOT / relative) == expected, relative

    for run_relative, binding in publication["raw_artifact_bindings"].items():
        path = ROOT / binding["public_path"]
        if binding["encoding"] == "gzip lossless":
            raw = gzip.decompress(path.read_bytes())
            assert hashlib.sha256(raw).hexdigest() == binding["sha256"], run_relative
        else:
            assert binding["encoding"] == "byte-exact copy"
            assert sha(path) == binding["sha256"], run_relative

    for relative in actual:
        path = ROOT / relative
        if path.suffix == ".json":
            safe(read(path))
        elif path.suffix in (".md", ".csv"):
            assert not PRIVATE_TEXT.search(path.read_text(encoding="utf-8-sig")), relative
    for pattern, expected in (
        ("size*_traces.npz", 416),
        ("summary.json.gz", 208),
        ("clip_dose.json.gz", 208),
        ("training_curve.json.gz", 16),
        ("matched_frontier_strata.csv.gz", 16),
    ):
        assert len(list(PUBLIC.rglob(pattern))) == expected, pattern
    assert len(list((PUBLIC / "banks").glob("*.npz"))) == 3
    assert (PUBLIC / "calibration/raw_write_rms.npz").is_file()
    assert (PUBLIC / "calibration/calibration_inputs.npz").is_file()

    plans = gzip.decompress((PUBLIC / "plans.json.gz").read_bytes())
    assert hashlib.sha256(plans).hexdigest() == publication["schedule_plan_sha256"]
    plan_data = json.loads(plans)
    assert set(plan_data) == {str(block) for block in range(BLOCKS)}
    for block in range(BLOCKS):
        plan = plan_data[str(block)]
        assert plan["initialization_seed"] == 116001 + block
        assert plan["schedule_seed"] == 117001 + block
        assert set(plan["initial_parameter_sha256"]) == set(ARMS)
        batches = np.asarray(plan["batch_indices"])
        assert batches.shape == (300, 8) and batches.min() >= 0 and batches.max() < 512


def build() -> None:
    assert not PUBLIC.exists() and not MANIFEST.exists(), "Refusing to overwrite published evidence"
    run_manifest = read(RUN / "manifest.json")
    run_status = read(RUN / "status.json")
    run_summary = read(RUN / "summary.json")
    run_config = read(RUN / "config.json")
    calibration_manifest = read(CALIBRATION / "manifest.json")
    calibration_status = read(CALIBRATION / "status.json")
    calibration_summary = read(CALIBRATION / "summary.json")
    calibration_config = read(CALIBRATION / "config.json")
    qualification = read(QUALIFICATION)

    assert run_status["status"] == run_summary["status"] == "COMPLETE"
    assert run_manifest["protocol"] == run_summary["protocol"] == run_config["protocol"] == PROTOCOL
    assert run_summary["completed_arms"] == run_summary["expected_arms"] == 16
    assert run_summary["dense_records"] == run_manifest["expected_dense_records"] == 208
    assert calibration_status["status"] == "COMPLETE"
    assert calibration_summary["status"] == "CALIBRATED"
    assert calibration_manifest["protocol"] == calibration_summary["protocol"] == CALIBRATION_PROTOCOL
    assert sha(QUALIFICATION) == run_manifest["qualification_sha256"]
    assert qualification["status"] == "PASS" and qualification["protocol"] == PROTOCOL

    run_source_bindings, run_source_hashes = _check_source_manifest(run_manifest, RUN)
    cal_source_bindings, cal_source_hashes = _check_source_manifest(calibration_manifest, CALIBRATION)
    for relative, digest in cal_source_hashes.items():
        if relative in run_source_hashes:
            assert run_source_hashes[relative] == digest, relative
        else:
            run_source_hashes[relative] = digest
    source_bindings = {"experiment": run_source_bindings, "calibration": cal_source_bindings}

    run_dense = read(RUN / "dense.json")
    run_final = read(RUN / "perarm.json")
    run_keys = {(row["block"], row["arm"], row["update"]) for row in run_dense}
    expected_keys = {(block, arm, update) for block in range(BLOCKS) for arm in ARMS for update in CHECKPOINTS}
    assert len(run_dense) == 208 and run_keys == expected_keys
    assert len(run_final) == 16
    checkpoint_bindings = {}
    for row in run_dense:
        relative = row["checkpoint"]
        assert relative not in checkpoint_bindings
        checkpoint_path = RUN / relative
        digest = sha(checkpoint_path)
        assert digest == row["checkpoint_sha256"], relative
        checkpoint_bindings[relative.replace("\\", "/")] = {
            "file_sha256": digest,
            "parameter_sha256": row["parameter_sha256"],
        }
    assert len(checkpoint_bindings) == 208

    with np.load(CALIBRATION / "raw_write_rms.npz", allow_pickle=False) as archive:
        raw_samples = archive["raw_write_rms"]
        input_arrays = None
    with np.load(CALIBRATION / "calibration_inputs.npz", allow_pickle=False) as archive:
        input_arrays = {key: archive[key] for key in archive.files}
    calibration_validation = _validate_calibration(
        calibration_summary, calibration_config, run_config, run_manifest, raw_samples, input_arrays
    )
    assert sha(CALIBRATION / "raw_write_rms.npz") == calibration_summary["raw_samples_sha256"]
    binding = run_manifest["calibration_binding"]
    assert binding["summary_sha256"] == sha(CALIBRATION / "summary.json")
    assert binding["samples_sha256"] == sha(CALIBRATION / "raw_write_rms.npz")
    assert binding["inputs_sha256"] == sha(CALIBRATION / "calibration_inputs.npz")

    PUBLIC.mkdir(parents=True)
    raw_bindings: dict[str, Any] = {}
    for source, public_relative, encoding, raw_key in raw_artifacts():
        assert source.is_file(), source.name
        if source.suffix == ".json":
            safe(read(source))
        elif source.suffix in (".md", ".csv"):
            assert not PRIVATE_TEXT.search(source.read_text(encoding="utf-8-sig")), raw_key
        copy_bound(source, PUBLIC / public_relative, raw_bindings, raw_key, encoding)

    public_qualification = _sanitized_qualification(qualification)
    write(PUBLIC / "runtime_qualification.json", public_qualification)
    training_rows, training_dose = _training_dose_data(PUBLIC)
    _write_csv(PUBLIC / "training_dose.csv", ("block", "arm", "update", *_DOSE_FIELDS), training_rows)
    write(PUBLIC / "dose_summary.json", training_dose)
    _write_final_metrics(PUBLIC, run_final)

    evaluation_dose = _evaluation_dose_summary(PUBLIC, run_dense, raw_bindings)
    measured = measure({"data_sha256": run_manifest["data_sha256"],
                        "raw_artifact_bindings": raw_bindings,
                        "checkpoint_bindings": checkpoint_bindings,
                        "calibration_binding": run_manifest["calibration_binding"]})
    measured.update(
        source_binding_status="PASS",
        source_binding_files={"experiment": len(run_source_bindings), "calibration": len(cal_source_bindings)},
        permitted_launcher_transform_count=1,
        checkpoint_file_hashes_verified_locally=208,
        calibration_validation=calibration_validation,
        evaluation_dose=evaluation_dose,
    )
    safe(measured)
    write(PUBLIC / "validation.json", measured)
    _write_results(run_summary, calibration_summary, training_dose)
    _write_reproduction()

    published = {path.relative_to(ROOT).as_posix(): sha(path) for path in PUBLIC.rglob("*") if path.is_file()}
    source_sha256 = dict(sorted(run_source_hashes.items()))
    verification_sources = (Path(__file__), ROOT / HELPER, ROOT / REPORTING, ROOT / METRICS)
    verification_hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in verification_sources}
    publication = {
        "protocol": "hardclip_v1_saved_data_publication_v1",
        "run_id": RUN.name,
        "calibration_run_id": CALIBRATION.name,
        "experiment_protocol": PROTOCOL,
        "calibration_protocol": CALIBRATION_PROTOCOL,
        "git_review_base": run_manifest["git_review_base"],
        "source_sha256": source_sha256,
        "source_bindings": source_bindings,
        "source_binding_status": "PASS",
        "data_sha256": run_manifest["data_sha256"],
        "schedule_plan_sha256": run_manifest["schedule_plan_sha256"],
        "qualification_sha256": run_manifest["qualification_sha256"],
        "sanitized_qualification_sha256": sha(PUBLIC / "runtime_qualification.json"),
        "calibration_summary_sha256": sha(CALIBRATION / "summary.json"),
        "calibration_raw_samples_sha256": sha(CALIBRATION / "raw_write_rms.npz"),
        "calibration_inputs_sha256": sha(CALIBRATION / "calibration_inputs.npz"),
        "calibration_binding": run_manifest["calibration_binding"],
        "verification_source_sha256": verification_hashes,
        "checkpoint_bindings": checkpoint_bindings,
        "raw_artifact_bindings": raw_bindings,
        "published_sha256": published,
        "published_file_count": len(published),
        "published_bytes": sum((ROOT / name).stat().st_size for name in published),
        "excluded": [
            "checkpoint and optimizer contents",
            "private run/calibration manifests and status/PID files",
            "launch receipts, logs, host/account/absolute-path metadata",
            "source snapshots (original and current source hashes are retained)",
        ],
    }
    safe(publication)
    write(MANIFEST, publication)
    verify()


def verify() -> None:
    """Verify the public package without accessing runs, analyses, or checkpoints."""
    publication = read(MANIFEST)
    verify_files(publication)
    measured = measure()
    validation = read(PUBLIC / "validation.json")
    for key, value in measured.items():
        if key in validation:
            assert validation[key] == value, key
    assert validation["status"] == "PASS"
    assert validation["record_keys"] == measured["record_keys"]
    print(json.dumps({
        "status": "PASS",
        "published_files": publication["published_file_count"],
        "packed_traces": 416,
        "evaluation_summaries": 208,
        "training_dose_rows": 19200,
        "u300_trace_metric_banks": 32,
        "primary_verdict": read(PUBLIC / "summary.json")["overall_verdict"],
    }, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the public package without accessing local run, analysis, or checkpoint files.",
    )
    arguments = parser.parse_args()
    verify() if arguments.verify_only else build()
