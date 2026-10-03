"""Offline CPU validation of a completed 195/200 saved-case audit.

The validator checks hashes, saved Boolean traces, endpoint/interval metrics,
fixed-cohort summaries, checkpoint controls and plan invariants. It performs no
model inference and never modifies run artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "runs" / "transition_20261003_seed4_dense01"
EXPECTED_CASES = 492
PRIMARY_GROUP_COUNTS = {
    "baseline": 2, "factorial": 16, "interpolation": 9,
    "cross_continuation": 8, "state_swap": 68,
    "update_pulse": 32, "local_transplant": 12,
}
CONFIRMATION_GROUP_COUNTS = {
    "baseline": 2, "factorial": 16, "interpolation": 9,
    "cross_continuation": 8, "state_swap": 20,
    "update_pulse": 32, "local_transplant": 12,
}
BLOCKS = "EFQR"
COALITIONS = tuple(
    "".join(block for block in BLOCKS if block in selected) or "none"
    for n in range(5) for selected in itertools.combinations(BLOCKS, n)
)
ORDERS = tuple("".join(order) for order in itertools.permutations(BLOCKS))
ENDPOINTS = ((64, 0), (128, 64), (256, 192))
INTERVALS = ((64, 128, 0, 64), (64, 256, 0, 192))
DOMAINS = ("primary", "strict")
RATE_FIELDS = ("acquisition", "endpoint_destruction", "first_exit",
               "continuous_survival", "all_steps_correct",
               "any_wrong_including_start", "net_gain")
FLOAT_TOL = 2e-7


class Audit:
    def __init__(self) -> None:
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.errors: list[dict[str, Any]] = []
        self.max_float_error = 0.0
        self.nonfinite_cases = 0
        self.finite_cases = 0
        self.quantile_trajectory_cases = 0
        self.cohort_records = 0

    def check(self, category: str, ok: bool, detail: str | None = None) -> None:
        self.counts[category]["checked"] += 1
        self.counts[category]["passed" if ok else "failed"] += 1
        if not ok and len(self.errors) < 250:
            self.errors.append({"category": category, "detail": detail or "check failed"})

    def exact(self, category: str, actual: Any, expected: Any, where: str) -> None:
        self.check(category, actual == expected, f"{where}: expected {expected!r}, observed {actual!r}")

    def approx(self, category: str, actual: Any, expected: Any, where: str,
               tolerance: float = FLOAT_TOL) -> None:
        if expected is None:
            self.check(category, actual is None, f"{where}: expected null, observed {actual!r}")
            return
        try:
            observed = float(actual)
            reference = float(expected)
            error = abs(observed - reference)
            if math.isfinite(error):
                self.max_float_error = max(self.max_float_error, error)
            ok = math.isfinite(observed) and math.isfinite(reference) and error <= tolerance
        except (TypeError, ValueError, OverflowError):
            ok = False
            observed = actual
            error = None
        self.check(category, ok, f"{where}: expected {expected!r}, observed {observed!r}, abs_error={error!r}")

    @property
    def passed(self) -> bool:
        return all(counts.get("failed", 0) == 0 for counts in self.counts.values())

    def report_counts(self) -> dict[str, dict[str, int]]:
        return {key: dict(value) for key, value in sorted(self.counts.items())}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(tensors: dict[str, Any]) -> str:
    """Match the frozen run's tensor_hash without importing execution code."""
    import torch

    digest = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _numpy_tensor_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _safe_file(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    path = (root / Path(relative.replace("/", "\\"))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _as_map(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr


def _finite_array(array: np.ndarray) -> bool:
    return bool(np.issubdtype(array.dtype, np.bool_) or np.all(np.isfinite(array)))


def _domain_masks(bank_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    changed = _as_map(bank_arrays["changed"]).astype(bool)
    opened = _as_map(bank_arrays["mask"]).astype(bool)
    distance = _as_map(bank_arrays["distance"])
    primary = changed & opened & (distance > 0)
    strict = changed & opened & (distance > 16) & (distance < 32)
    return {"primary": primary, "strict": strict}


def _rate(nums: np.ndarray, dens: np.ndarray) -> dict[str, Any]:
    nums = np.asarray(nums, dtype=np.int64).reshape(-1)
    dens = np.asarray(dens, dtype=np.int64).reshape(-1)
    per_map = []
    rates = []
    for index, (num, den) in enumerate(zip(nums, dens)):
        num, den = int(num), int(den)
        value = float(num / den) if den else None
        per_map.append({"map_index": index, "numerator": num, "denominator": den, "rate": value})
        if value is not None:
            rates.append(value)
    total_num, total_den = int(nums.sum()), int(dens.sum())
    return {
        "pooled": {"numerator": total_num, "denominator": total_den,
                   "rate": float(total_num / total_den) if total_den else None},
        "per_map": per_map,
        "equal_map_mean": float(np.mean(rates)) if rates else None,
        "equal_map_eligible_maps": len(rates),
    }


def _counts(mask: np.ndarray) -> np.ndarray:
    return np.count_nonzero(mask, axis=(1, 2)).astype(np.int64, copy=False)


def _recompute_boolean(correct: np.ndarray, domain: np.ndarray) -> dict[str, Any]:
    pixels = _counts(domain)
    endpoints = {}
    for step, index in ENDPOINTS:
        endpoints[str(step)] = _rate(_counts(domain & correct[index]), pixels)
    intervals = {}
    for start, end, first, last in INTERVALS:
        start_mask = correct[first]
        end_mask = correct[last]
        start_correct = _counts(domain & start_mask)
        wrong_start = pixels - start_correct
        acquired = _counts(domain & ~start_mask & end_mask)
        destroyed = _counts(domain & start_mask & ~end_mask)
        exited = _counts(domain & start_mask & np.any(~correct[first + 1:last + 1], axis=0))
        continuous = _counts(domain & start_mask & np.all(correct[first:last + 1], axis=0))
        all_steps = _counts(domain & np.all(correct[first:last + 1], axis=0))
        any_wrong = _counts(domain & np.any(~correct[first:last + 1], axis=0))
        net = _counts(domain & end_mask) - start_correct
        c2w = np.count_nonzero(domain[None] & correct[first:last] & ~correct[first + 1:last + 1], axis=(0, 2, 3))
        w2c = np.count_nonzero(domain[None] & ~correct[first:last] & correct[first + 1:last + 1], axis=(0, 2, 3))
        opportunity = pixels * (last - first)
        errors = net - (acquired - destroyed)
        intervals[f"{start}_{end}"] = {
            "acquisition": _rate(acquired, wrong_start),
            "endpoint_destruction": _rate(destroyed, start_correct),
            "first_exit": _rate(exited, start_correct),
            "continuous_survival": _rate(continuous, start_correct),
            "all_steps_correct": _rate(all_steps, pixels),
            "any_wrong_including_start": _rate(any_wrong, pixels),
            "net_gain": _rate(net, pixels),
            "turnover": {
                "per_map": [{
                    "map_index": i,
                    "selected_pixels": int(pixels[i]),
                    "transition_opportunities": int(opportunity[i]),
                    "correct_to_wrong": int(c2w[i]),
                    "wrong_to_correct": int(w2c[i]),
                    "total_transitions": int(c2w[i] + w2c[i]),
                } for i in range(len(pixels))],
                "pooled": {
                    "selected_pixels": int(pixels.sum()),
                    "transition_opportunities": int(opportunity.sum()),
                    "correct_to_wrong": int(c2w.sum()),
                    "wrong_to_correct": int(w2c.sum()),
                    "total_transitions": int(c2w.sum() + w2c.sum()),
                },
            },
            "identity": {
                "per_map_accounting_error": [int(value) for value in errors],
                "pooled_net_change": int(net.sum()),
                "pooled_acquired_minus_endpoint_destroyed": int(acquired.sum() - destroyed.sum()),
                "pooled_accounting_error": int(errors.sum()),
            },
        }
    return {
        "domains": {"pixels_per_map": [int(v) for v in pixels], "pooled_pixels": int(pixels.sum())},
        "endpoints": endpoints,
        "intervals": intervals,
    }


def _mean_change(values: np.ndarray, selected: np.ndarray,
                 start: int, end: int) -> dict[str, Any]:
    change = values[end] - values[start]
    per_map = []
    means = []
    pooled = []
    for map_index in range(selected.shape[0]):
        chosen = change[map_index][selected[map_index]].astype(np.float64, copy=False)
        mean = float(chosen.mean()) if chosen.size else None
        per_map.append({"map_index": map_index, "n": int(chosen.size), "mean": mean})
        if chosen.size:
            means.append(mean)
            pooled.append(chosen)
    flat = np.concatenate(pooled) if pooled else np.empty(0, dtype=np.float64)
    return {
        "per_map": per_map,
        "equal_map_mean": float(np.mean(means)) if means else None,
        "equal_map_eligible_maps": len(means),
        "pooled_descriptive": {"n": int(flat.size), "mean": float(flat.mean()) if flat.size else None},
    }


def _recompute_cohort(correct: np.ndarray, margins: dict[str, np.ndarray],
                      selected: np.ndarray) -> dict[str, Any]:
    boolean = _recompute_boolean(correct, selected)
    return {
        "selected_pixels_per_map": boolean["domains"]["pixels_per_map"],
        "endpoints": boolean["endpoints"],
        "first_exit": {key: value["first_exit"] for key, value in boolean["intervals"].items()},
        "all_steps_correct": {key: value["all_steps_correct"] for key, value in boolean["intervals"].items()},
        "any_wrong_including_start": {key: value["any_wrong_including_start"] for key, value in boolean["intervals"].items()},
        "net_margin_change": {
            "64_128": {
                name: _mean_change(values, selected, 0, 1)
                for name, values in margins.items()
            },
            "64_256": {
                name: _mean_change(values, selected, 0, 2)
                for name, values in margins.items()
            },
        },
    }


def _compare_rate(audit: Audit, saved: Any, expected: dict[str, Any], where: str) -> None:
    if not isinstance(saved, dict):
        audit.check("saved_metric_structure", False, f"{where}: rate summary is missing/not an object")
        return
    for field in ("pooled",):
        node, target = saved.get(field), expected[field]
        if not isinstance(node, dict):
            audit.check("saved_metric_structure", False, f"{where}.{field}: missing")
            continue
        for key in ("numerator", "denominator"):
            audit.exact("metric_integer_counts", node.get(key), target[key], f"{where}.{field}.{key}")
        audit.approx("metric_rates", node.get("rate"), target["rate"], f"{where}.{field}.rate")
    saved_rows = saved.get("per_map")
    expected_rows = expected.get("per_map", [])
    if not isinstance(saved_rows, list):
        audit.check("saved_metric_structure", False, f"{where}.per_map: missing")
    else:
        audit.exact("metric_integer_counts", len(saved_rows), len(expected_rows), f"{where}.per_map.length")
        for index, (actual, target) in enumerate(zip(saved_rows, expected_rows)):
            if not isinstance(actual, dict):
                audit.check("saved_metric_structure", False, f"{where}.per_map[{index}]: invalid")
                continue
            for key in ("map_index", "numerator", "denominator"):
                audit.exact("metric_integer_counts", actual.get(key), target[key], f"{where}.per_map[{index}].{key}")
            audit.approx("metric_rates", actual.get("rate"), target["rate"], f"{where}.per_map[{index}].rate")
    audit.exact("metric_integer_counts", saved.get("equal_map_eligible_maps"),
                expected.get("equal_map_eligible_maps"), f"{where}.equal_map_eligible_maps")
    audit.approx("metric_rates", saved.get("equal_map_mean"), expected.get("equal_map_mean"),
                 f"{where}.equal_map_mean")


def _compare_boolean_metrics(audit: Audit, summary: Any,
                             correct: np.ndarray,
                             domains: dict[str, np.ndarray],
                             where: str) -> None:
    if not isinstance(summary, dict):
        audit.check("saved_metric_structure", False, f"{where}: summary absent")
        return
    if summary.get("schema_version") != "audit-195-200-metrics-v1":
        audit.check("case_schema", False, f"{where}: unexpected metrics schema {summary.get('schema_version')!r}")
    expected_steps = {"first": 64, "last": 256, "count": 193}
    audit.exact("case_schema", summary.get("trace_steps"), expected_steps, f"{where}.trace_steps")
    for domain_name, selected in domains.items():
        recomputed = _recompute_boolean(correct, selected)
        saved_domain = summary.get("domains", {}).get(domain_name, {})
        audit.exact("metric_integer_counts", saved_domain.get("pixels_per_map"),
                    recomputed["domains"]["pixels_per_map"], f"{where}.domains.{domain_name}.pixels_per_map")
        audit.exact("metric_integer_counts", saved_domain.get("pooled_pixels"),
                    recomputed["domains"]["pooled_pixels"], f"{where}.domains.{domain_name}.pooled_pixels")
        for step, _ in ENDPOINTS:
            saved_rate = summary.get("endpoints", {}).get(str(step), {}).get(domain_name)
            _compare_rate(audit, saved_rate, recomputed["endpoints"][str(step)],
                          f"{where}.endpoints.{step}.{domain_name}")
        for interval_key, target_interval in recomputed["intervals"].items():
            saved_interval = summary.get("intervals", {}).get(interval_key, {}).get(domain_name, {})
            for metric_name in RATE_FIELDS:
                _compare_rate(audit, saved_interval.get(metric_name), target_interval[metric_name],
                              f"{where}.intervals.{interval_key}.{domain_name}.{metric_name}")
            expected_turnover = target_interval["turnover"]
            saved_turnover = saved_interval.get("turnover", {})
            for scope in ("pooled",):
                actual = saved_turnover.get(scope, {})
                for key, value in expected_turnover[scope].items():
                    audit.exact("metric_integer_counts", actual.get(key), value,
                                f"{where}.intervals.{interval_key}.{domain_name}.turnover.{scope}.{key}")
            actual_rows = saved_turnover.get("per_map", [])
            expected_rows = expected_turnover["per_map"]
            audit.exact("metric_integer_counts", len(actual_rows), len(expected_rows),
                        f"{where}.intervals.{interval_key}.{domain_name}.turnover.per_map.length")
            for index, (actual, target) in enumerate(zip(actual_rows, expected_rows)):
                for key, value in target.items():
                    audit.exact("metric_integer_counts", actual.get(key), value,
                                f"{where}.intervals.{interval_key}.{domain_name}.turnover.per_map[{index}].{key}")
            saved_identity = saved_interval.get("identity", {})
            for key, value in target_interval["identity"].items():
                audit.exact("metric_integer_counts", saved_identity.get(key), value,
                            f"{where}.intervals.{interval_key}.{domain_name}.identity.{key}")


QUANTILE_LEVELS = (0.0, 0.1, 0.5, 0.9, 1.0)


def _quantile_record(values: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    if values.ndim == 3:
        per_map_values = [values[index][selected[index]].astype(np.float64, copy=False)
                          for index in range(selected.shape[0])]
    else:
        per_map_values = [values[:, index][:, selected[index]].reshape(-1).astype(np.float64, copy=False)
                          for index in range(selected.shape[0])]
    per_map = []
    map_quantiles = []
    pooled_parts = []
    for map_index, chosen in enumerate(per_map_values):
        if chosen.size:
            q = np.quantile(chosen, QUANTILE_LEVELS)
            q_values = [float(value) for value in q]
            map_quantiles.append(q)
            pooled_parts.append(chosen)
        else:
            q_values = None
        per_map.append({"map_index": map_index, "observations": int(chosen.size), "quantiles": q_values})
    if pooled_parts:
        pooled = np.concatenate(pooled_parts)
        pooled_q = [float(value) for value in np.quantile(pooled, QUANTILE_LEVELS)]
        equal_map_q = [float(value) for value in np.mean(np.stack(map_quantiles), axis=0)]
    else:
        pooled_q, equal_map_q = None, None
        pooled = np.empty(0, dtype=np.float64)
    return {
        "quantile_levels": list(QUANTILE_LEVELS),
        "per_map": per_map,
        "equal_map_mean_quantiles": equal_map_q,
        "equal_map_eligible_maps": len(map_quantiles),
        "pooled": {"observations": int(pooled.size), "quantiles": pooled_q},
    }


def _compare_quantile_record(audit: Audit, saved: Any, expected: dict[str, Any], where: str) -> None:
    if not isinstance(saved, dict):
        audit.check("margin_quantiles", False, f"{where}: missing summary")
        return
    audit.exact("margin_quantiles", saved.get("quantile_levels"), expected["quantile_levels"], f"{where}.quantile_levels")
    audit.exact("margin_quantiles", saved.get("equal_map_eligible_maps"),
                expected["equal_map_eligible_maps"], f"{where}.equal_map_eligible_maps")
    _compare_float_tree(audit, saved.get("equal_map_mean_quantiles"),
                        expected["equal_map_mean_quantiles"], f"{where}.equal_map_mean_quantiles")
    saved_pooled, expected_pooled = saved.get("pooled", {}), expected["pooled"]
    audit.exact("margin_quantiles", saved_pooled.get("observations"), expected_pooled["observations"],
                f"{where}.pooled.observations")
    _compare_float_tree(audit, saved_pooled.get("quantiles"), expected_pooled["quantiles"],
                        f"{where}.pooled.quantiles")
    saved_rows, expected_rows = saved.get("per_map", []), expected["per_map"]
    audit.exact("margin_quantiles", len(saved_rows), len(expected_rows), f"{where}.per_map.length")
    for index, (actual, target) in enumerate(zip(saved_rows, expected_rows)):
        audit.exact("margin_quantiles", actual.get("map_index"), target["map_index"], f"{where}.per_map[{index}].map_index")
        audit.exact("margin_quantiles", actual.get("observations"), target["observations"], f"{where}.per_map[{index}].observations")
        _compare_float_tree(audit, actual.get("quantiles"), target["quantiles"],
                            f"{where}.per_map[{index}].quantiles")


def _compare_margin_summaries(audit: Audit, summary: dict[str, Any],
                              margins_endpoints: dict[str, np.ndarray],
                              initial_correct: np.ndarray,
                              full_margins: dict[str, np.ndarray] | None,
                              where: str) -> None:
    paired = summary.get("paired_margin", {})
    sources = {"paired_min": "margin_endpoints", "original": "original_margin_endpoints",
               "flipped": "flipped_margin_endpoints"}
    for public_name, array_key in sources.items():
        saved_source = paired.get("sources", {}).get(public_name, {})
        if array_key not in margins_endpoints:
            audit.check("margin_quantiles", False, f"{where}: missing {array_key}")
            continue
        values = margins_endpoints[array_key]
        for endpoint_index, (step, _) in enumerate(ENDPOINTS):
            expected = _quantile_record(values[endpoint_index], initial_correct)
            actual = saved_source.get("endpoints", {}).get(str(step))
            _compare_quantile_record(audit, actual, expected, f"{where}.paired_margin.{public_name}.endpoints.{step}")
        if full_margins is not None and public_name in full_margins:
            expected = _quantile_record(full_margins[public_name], initial_correct)
            actual = saved_source.get("trajectory_64_256")
            _compare_quantile_record(audit, actual, expected,
                                     f"{where}.paired_margin.{public_name}.trajectory_64_256")
        else:
            audit.check("trajectory_margin_quantiles_unavailable", True,
                        f"{where}.{public_name}: full trajectory margin not saved for this counterfactual")


def _compare_float_tree(audit: Audit, actual: Any, expected: Any, where: str) -> None:
    if expected is None:
        audit.check("margin_quantiles", actual is None, f"{where}: expected null, observed {actual!r}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict):
            audit.check("margin_quantiles", False, f"{where}: expected dict, observed {type(actual).__name__}")
        else:
            audit.exact("margin_quantiles", set(actual), set(expected), f"{where}.keys")
            for key, value in expected.items():
                _compare_float_tree(audit, actual.get(key), value, f"{where}.{key}")
    elif isinstance(expected, str):
        audit.exact("margin_quantiles", actual, expected, where)
    elif isinstance(expected, list):
        if not isinstance(actual, list):
            audit.check("margin_quantiles", False, f"{where}: expected list, observed {type(actual).__name__}")
        else:
            audit.exact("margin_quantiles", len(actual), len(expected), f"{where}.length")
            for index, (a, e) in enumerate(zip(actual, expected)):
                _compare_float_tree(audit, a, e, f"{where}[{index}]")
    else:
        audit.approx("margin_quantiles", actual, expected, where)


def _hash_binding_group(audit: Audit, run: Path, bindings: dict[str, Any]) -> dict[str, Any]:
    source_expected = bindings.get("source_sha256", {})
    source_count = 0
    live_mismatches = []
    snapshot_mismatches = []
    for relative, expected in source_expected.items():
        source_count += 1
        live = _safe_file(ROOT, relative)
        snapshot = _safe_file(run / "source", relative)
        try:
            live_hash = _sha256(live) if live else None
            snapshot_hash = _sha256(snapshot) if snapshot else None
        except OSError:
            live_hash, snapshot_hash = None, None
        audit.check("source_live_hash", live_hash == expected,
                    f"{relative}: expected {expected}, observed {live_hash}")
        audit.check("source_snapshot_hash", snapshot_hash == expected,
                    f"{relative}: expected {expected}, observed {snapshot_hash}")
        if live_hash != expected:
            live_mismatches.append(relative)
        if snapshot_hash != expected:
            snapshot_mismatches.append(relative)

    input_expected = bindings.get("input_sha256", {})
    input_mismatches = []
    for relative, expected in input_expected.items():
        path = _safe_file(REFERENCE, relative)
        try:
            observed = _sha256(path) if path else None
        except OSError:
            observed = None
        audit.check("reference_input_hash", observed == expected,
                    f"{relative}: expected {expected}, observed {observed}")
        if observed != expected:
            input_mismatches.append(relative)
    required = {"bank_size32.npz", "bank_size64.npz"}
    for update in (195, 200):
        required.add(f"checkpoints/u{update}.pt")
        for size in (32, 64):
            required.add(f"checkpoints/u{update}_size{size}.json")
            required.add(f"checkpoints/u{update}_size{size}.npz")
    audit.check("reference_input_inventory", required.issubset(set(input_expected)),
                f"missing required reference inputs: {sorted(required - set(input_expected))}")
    return {
        "source_hashes_checked": source_count,
        "live_source_mismatches": live_mismatches,
        "snapshot_source_mismatches": snapshot_mismatches,
        "reference_input_hashes_checked": len(input_expected),
        "reference_input_mismatches": input_mismatches,
    }


def _validate_reference_checkpoints(audit: Audit, bindings: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        audit.check("checkpoint_tensor_hash", False, f"PyTorch unavailable for CPU tensor hash check: {exc}")
        return {"updates_checked": []}
    try:
        reference_manifest = _read_json(REFERENCE / "manifest.json")
    except Exception as exc:
        audit.check("reference_manifest", False, f"Could not read reference manifest: {exc}")
        reference_manifest = {}
    reports = []
    for update in (195, 200):
        path = REFERENCE / "checkpoints" / f"u{update}.pt"
        record = {"update": update, "file": f"checkpoints/u{update}.pt"}
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            state = payload.get("state_dict")
            observed_hash = _tensor_hash(state) if isinstance(state, dict) else None
            expected_hash = bindings.get("parameter_sha256", {}).get(str(update))
            audit.check("checkpoint_tensor_hash", observed_hash == expected_hash,
                        f"u{update}: expected {expected_hash}, observed {observed_hash}")
            audit.exact("checkpoint_metadata", payload.get("completed_updates"), update,
                        f"u{update}.completed_updates")
            audit.exact("checkpoint_metadata", payload.get("seed"), 4, f"u{update}.seed")
            record["parameter_sha256_matches"] = observed_hash == expected_hash
            record["metadata_matches"] = payload.get("completed_updates") == update and payload.get("seed") == 4
        except Exception as exc:
            audit.check("checkpoint_tensor_hash", False, f"u{update}: {type(exc).__name__}: {exc}")
            record["error"] = f"{type(exc).__name__}: {exc}"
        per_size = {}
        for size in (32, 64):
            check_path = REFERENCE / "checkpoints" / f"u{update}_size{size}.json"
            trace_path = REFERENCE / "checkpoints" / f"u{update}_size{size}.npz"
            try:
                check = _read_json(check_path)
                trace_hash = _sha256(trace_path)
                expected_trace = check.get("trace_sha256")
                expected_parameter = check.get("parameter_sha256")
                tensor_expected = bindings.get("parameter_sha256", {}).get(str(update))
                audit.check("reference_checkpoint_trace_hash", trace_hash == expected_trace,
                            f"u{update}/size{size}: expected {expected_trace}, observed {trace_hash}")
                audit.check("reference_checkpoint_parameter_binding", expected_parameter == tensor_expected,
                            f"u{update}/size{size}: row parameter hash differs from bindings.parameter_sha256")
                per_size[str(size)] = {"trace_sha256_matches": trace_hash == expected_trace,
                                       "parameter_sha256_matches": expected_parameter == tensor_expected}
            except Exception as exc:
                audit.check("reference_checkpoint_trace_hash", False,
                            f"u{update}/size{size}: {type(exc).__name__}: {exc}")
                per_size[str(size)] = {"error": f"{type(exc).__name__}: {exc}"}
        record["reference_sizes"] = per_size
        reports.append(record)

    expected_primary = reference_manifest.get("audit_banks", {})
    return {
        "updates_checked": reports,
        "reference_bank_tensor_hashes": {
            str(size): expected_primary.get(str(size), {}).get("tensor_sha256") for size in (32, 64)
        },
    }


def _load_saved_banks(audit: Audit, run: Path, manifest: dict[str, Any],
                      reference_checkpoint_report: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    bank_rows = manifest.get("audit_banks", {})
    expected_names = {"primary32", "primary64", "confirmation32", "confirmation64"}
    audit.exact("bank_inventory", set(bank_rows), expected_names, "manifest.audit_banks names")
    reference_hashes = reference_checkpoint_report.get("reference_bank_tensor_hashes", {})
    saved: dict[str, dict[str, np.ndarray]] = {}
    for bank_name in sorted(expected_names):
        meta = bank_rows.get(bank_name, {})
        path = _safe_file(run, meta.get("file"))
        expected_bytes = meta.get("sha256")
        observed_bytes = _sha256(path) if path else None
        audit.check("saved_bank_byte_hash", observed_bytes == expected_bytes,
                    f"{bank_name}: expected {expected_bytes}, observed {observed_bytes}")
        arrays = {}
        try:
            if path is None:
                raise FileNotFoundError("saved bank file missing or unsafe")
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]) for key in data.files}
            tensor_hash = _numpy_tensor_hash(arrays)
            audit.check("saved_bank_tensor_hash", tensor_hash == meta.get("tensor_sha256"),
                        f"{bank_name}: saved tensor digest differs from manifest")
            size = 32 if bank_name.endswith("32") else 64
            confirmation = bank_name.startswith("confirmation")
            map_count = 8 if confirmation else 16
            audit.exact("saved_bank_metadata", int(meta.get("maps", -1)), map_count, f"{bank_name}.maps")
            audit.exact("saved_bank_metadata", int(meta.get("seed", -1)),
                        (62000 if confirmation else 61000) + size, f"{bank_name}.seed")
            audit.check("saved_bank_shapes", arrays.get("x") is not None
                        and arrays["x"].shape == (map_count, 3, size, size),
                        f"{bank_name}.x shape expected {(map_count, 3, size, size)}")
            required = {"x", "y", "y_flip", "mask", "changed", "distance", "x_flip"}
            audit.check("saved_bank_schema", required.issubset(arrays),
                        f"{bank_name}: missing {sorted(required - set(arrays))}")
            if not confirmation:
                old_tensor = reference_hashes.get(str(size))
                audit.check("primary_bank_reference_tensor_hash", tensor_hash == old_tensor,
                            f"{bank_name}: tensor digest differs from historical bank size {size}")
            else:
                # Confirmation maps are deterministically generated from the
                # frozen task source. Recreate only the CPU bank, never a model.
                try:
                    import torch
                    task_dir = ROOT / "new" / "nca_inertial_wind_tunnel"
                    if str(task_dir) not in sys.path:
                        sys.path.insert(0, str(task_dir))
                    from tasks import bank as make_bank
                    generated = make_bank(size, map_count, 62000 + size, "cpu")
                    generated_hash = _tensor_hash(generated)
                    audit.check("confirmation_bank_generator_hash", generated_hash == tensor_hash,
                                f"{bank_name}: regenerated CPU tensor hash differs")
                except Exception as exc:
                    audit.check("confirmation_bank_generator_hash", False,
                                f"{bank_name}: CPU bank regeneration failed ({type(exc).__name__})")
            saved[bank_name] = arrays
        except Exception as exc:
            audit.check("saved_bank_load", False, f"{bank_name}: {type(exc).__name__}")
            saved[bank_name] = {}
    return saved


def _validate_run_controls(audit: Audit, run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        status = _read_json(run / "status.json")
    except Exception as exc:
        status = {}
        audit.check("run_status", False, f"status.json cannot be read ({type(exc).__name__})")
    audit.exact("run_status", status.get("status"), "COMPLETE", "status.json.status")
    audit.exact("run_status", status.get("completed_cases"), EXPECTED_CASES, "status.json.completed_cases")
    try:
        aggregate = _read_json(run / "aggregate.json")
    except Exception as exc:
        aggregate = {}
        audit.check("aggregate", False, f"aggregate.json cannot be read ({type(exc).__name__})")
    audit.exact("aggregate", aggregate.get("pass"), True, "aggregate.json.pass")
    audit.exact("aggregate", aggregate.get("cases"), EXPECTED_CASES, "aggregate.json.cases")
    controls = aggregate.get("replay_controls", [])
    expected_controls = {(u, s) for u in (195, 200) for s in (32, 64)}
    observed_controls = set()
    for row in controls:
        try:
            observed_controls.add((int(row.get("update")), int(row.get("size"))))
        except (TypeError, ValueError):
            continue
        audit.exact("replay_control", row.get("Boolean_exact"), True,
                    f"aggregate replay u{row.get('update')} size{row.get('size')} Boolean_exact")
        error = row.get("margin_max_error")
        audit.check("replay_control", error is not None and math.isfinite(float(error)) and float(error) <= FLOAT_TOL,
                    f"aggregate replay u{row.get('update')} size{row.get('size')} margin_max_error={error!r}")
    audit.exact("replay_control_inventory", observed_controls, expected_controls,
                "aggregate replay control (update,size) set")
    audit.exact("manifest_protocol", manifest.get("protocol"), "audit195_200_v1", "manifest.protocol")
    audit.exact("manifest_protocol", manifest.get("preflight"), False, "manifest.preflight")
    audit.exact("manifest_protocol", manifest.get("zero_training"), True, "manifest.zero_training")
    audit.exact("manifest_protocol", manifest.get("training_trajectories"), 1, "manifest.training_trajectories")
    return {"status": status.get("status"), "completed_cases": status.get("completed_cases"),
            "aggregate_pass": aggregate.get("pass"), "aggregate_cases": aggregate.get("cases"),
            "replay_controls": len(observed_controls)}


def _compare_historical_replay(audit: Audit, run: Path, row: dict[str, Any],
                               saved: dict[str, np.ndarray], bank_name: str) -> float | None:
    match = re.search(r"_baseline(195|200)$", str(row.get("case", "")))
    if not match:
        audit.check("historical_replay", False, f"{row.get('case')}: baseline update suffix missing")
        return None
    update = int(match.group(1))
    size = 32 if bank_name.endswith("32") else 64
    reference_trace_path = REFERENCE / "checkpoints" / f"u{update}_size{size}.npz"
    try:
        with np.load(reference_trace_path, allow_pickle=False) as reference:
            for key in ("correct", "original_correct", "flipped_correct"):
                expected = np.asarray(reference[key][64:257], dtype=bool)
                actual = saved.get(key)
                audit.check("historical_boolean_replay", actual is not None and np.array_equal(actual, expected),
                            f"{row.get('case')}.{key}: saved suffix must exactly equal reference steps 64..256")
            expected_margin = np.asarray(reference["margin"][64:257])
            actual_margin = saved.get("margin")
            if actual_margin is None or actual_margin.shape != expected_margin.shape:
                audit.check("historical_margin_replay", False,
                            f"{row.get('case')}: saved paired-margin trajectory shape differs")
                return None
            error = float(np.max(np.abs(actual_margin.astype(np.float64) - expected_margin.astype(np.float64))))
            audit.check("historical_margin_replay", math.isfinite(error) and error <= FLOAT_TOL,
                        f"{row.get('case')}: max absolute paired-margin error {error:.9g}")
            return error
    except Exception as exc:
        audit.check("historical_replay", False,
                    f"{row.get('case')}: reference trace read failed ({type(exc).__name__})")
        return None


def _load_case_arrays(audit: Audit, run: Path, row: dict[str, Any],
                      bank_arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    where = str(row.get("case", "unknown"))
    path = _safe_file(run, row.get("trace_file"))
    expected_hash = row.get("trace_sha256")
    observed_hash = _sha256(path) if path else None
    audit.check("case_trace_hash", observed_hash == expected_hash,
                f"{where}: expected {expected_hash}, observed {observed_hash}")
    if path is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except Exception as exc:
        audit.check("case_trace_npz", False, f"{where}: NPZ read failed ({type(exc).__name__})")
        return None
    required_bool = ("correct", "original_correct", "flipped_correct")
    if bank_arrays.get("x") is None:
        audit.check("case_trace_shape", False, f"{where}: corresponding saved bank unavailable")
        return None
    batch, _, height, width = bank_arrays["x"].shape
    trace_shape = (193, batch, height, width)
    for key in required_bool:
        value = arrays.get(key)
        audit.check("case_boolean_trace_shape", value is not None and value.shape == trace_shape
                    and value.dtype == np.bool_, f"{where}.{key}: expected boolean {trace_shape}")
    if not all(key in arrays and arrays[key].shape == trace_shape and arrays[key].dtype == np.bool_
               for key in required_bool):
        return None
    correct = arrays["correct"]
    audit.check("paired_boolean_definition", np.array_equal(correct, arrays["original_correct"] & arrays["flipped_correct"]),
                f"{where}: paired correct must equal original_correct & flipped_correct")

    endpoint_names = ("margin_endpoints", "original_margin_endpoints", "flipped_margin_endpoints")
    endpoint_shape = (3, batch, height, width)
    for key in endpoint_names:
        value = arrays.get(key)
        audit.check("case_margin_endpoint_shape", value is not None and value.shape == endpoint_shape
                    and np.issubdtype(value.dtype, np.number),
                    f"{where}.{key}: expected numeric {endpoint_shape}")
        if value is not None and value.shape == endpoint_shape:
            audit.check("case_margin_endpoint_finite", bool(np.all(np.isfinite(value))),
                        f"{where}.{key}: contains nonfinite values")
    if all(key in arrays and arrays[key].shape == endpoint_shape for key in endpoint_names):
        paired = arrays["margin_endpoints"]
        reconstructed = np.minimum(arrays["original_margin_endpoints"], arrays["flipped_margin_endpoints"])
        error = float(np.max(np.abs(paired.astype(np.float64) - reconstructed.astype(np.float64))))
        audit.check("paired_margin_endpoint_definition", error <= FLOAT_TOL,
                    f"{where}: paired margin endpoints differ from min(original,flipped) by {error:.9g}")

    summary = row.get("summary")
    if not isinstance(summary, dict):
        audit.check("case_schema", False, f"{where}: finite trace has no summary object")
        return arrays, {}
    domains = _domain_masks(bank_arrays)
    _compare_boolean_metrics(audit, summary, correct, domains, where)
    masks = {}
    summary_cohorts = summary.get("cohorts", {})
    if not isinstance(summary_cohorts, dict):
        audit.check("cohort_inventory", False, f"{where}: cohorts must be a JSON object")
        summary_cohorts = {}
    cohort_keys = {key[len("cohort_"):] for key in arrays if key.startswith("cohort_")}
    audit.exact("cohort_inventory", cohort_keys, set(summary_cohorts), f"{where}: NPZ masks vs summary cohort names")
    for cohort_name, cohort_summary in summary_cohorts.items():
        mask = arrays.get(f"cohort_{cohort_name}")
        if mask is None:
            audit.check("cohort_inventory", False, f"{where}: missing saved mask cohort_{cohort_name}")
            continue
        audit.check("cohort_mask_shape", mask.shape == (batch, height, width) and mask.dtype == np.bool_,
                    f"{where}.cohort_{cohort_name}: expected boolean {(batch, height, width)}")
        if mask.shape != (batch, height, width):
            continue
        masks[cohort_name] = mask.astype(bool, copy=False)
        selected = domains["primary"] & masks[cohort_name]
        expected = _recompute_cohort(correct, {
            "paired_min": arrays["margin_endpoints"],
            "original": arrays["original_margin_endpoints"],
            "flipped": arrays["flipped_margin_endpoints"],
        }, selected)
        audit.exact("cohort_integer_counts", cohort_summary.get("selected_pixels_per_map"),
                    expected["selected_pixels_per_map"], f"{where}.cohorts.{cohort_name}.selected_pixels_per_map")
        for step, _ in ENDPOINTS:
            _compare_rate(audit, cohort_summary.get("endpoints", {}).get(str(step)),
                          expected["endpoints"][str(step)], f"{where}.cohorts.{cohort_name}.endpoints.{step}")
        for interval, metric in (("64_128", "first_exit"), ("64_256", "first_exit"),
                                 ("64_128", "all_steps_correct"), ("64_256", "all_steps_correct"),
                                 ("64_128", "any_wrong_including_start"), ("64_256", "any_wrong_including_start")):
            _compare_rate(audit, cohort_summary.get(metric, {}).get(interval),
                          expected[metric][interval], f"{where}.cohorts.{cohort_name}.{metric}.{interval}")
        for interval in ("64_128", "64_256"):
            for source, target in expected["net_margin_change"][interval].items():
                actual = cohort_summary.get("net_margin_change", {}).get(interval, {}).get(source)
                if not isinstance(actual, dict):
                    audit.check("cohort_margin_means", False,
                                f"{where}.cohorts.{cohort_name}.net_margin_change.{interval}.{source}: missing")
                    continue
                audit.exact("cohort_margin_counts", len(actual.get("per_map", [])), len(target["per_map"]),
                            f"{where}.cohorts.{cohort_name}.{interval}.{source}.per_map.length")
                for index, (a_row, e_row) in enumerate(zip(actual.get("per_map", []), target["per_map"])):
                    audit.exact("cohort_margin_counts", a_row.get("map_index"), e_row["map_index"],
                                f"{where}.{cohort_name}.{interval}.{source}.per_map[{index}].map_index")
                    audit.exact("cohort_margin_counts", a_row.get("n"), e_row["n"],
                                f"{where}.{cohort_name}.{interval}.{source}.per_map[{index}].n")
                    audit.approx("cohort_margin_means", a_row.get("mean"), e_row["mean"],
                                 f"{where}.{cohort_name}.{interval}.{source}.per_map[{index}].mean")
                audit.approx("cohort_margin_means", actual.get("equal_map_mean"), target["equal_map_mean"],
                             f"{where}.{cohort_name}.{interval}.{source}.equal_map_mean")
                audit.exact("cohort_margin_counts", actual.get("equal_map_eligible_maps"),
                            target["equal_map_eligible_maps"],
                            f"{where}.{cohort_name}.{interval}.{source}.equal_map_eligible_maps")
                pooled_a, pooled_e = actual.get("pooled_descriptive", {}), target["pooled_descriptive"]
                audit.exact("cohort_margin_counts", pooled_a.get("n"), pooled_e["n"],
                            f"{where}.{cohort_name}.{interval}.{source}.pooled.n")
                audit.approx("cohort_margin_means", pooled_a.get("mean"), pooled_e["mean"],
                             f"{where}.{cohort_name}.{interval}.{source}.pooled.mean")
        audit.cohort_records += 1

    if not np.all(np.isfinite(arrays["margin_endpoints"])):
        # Already counted above; keep this branch useful for load failures only.
        pass
    full_margins = None
    if row.get("group") == "baseline":
        full_margins = {}
        for key, source in (("margin", "paired_min"), ("original_margin", "original"),
                            ("flipped_margin", "flipped")):
            value = arrays.get(key)
            audit.check("baseline_full_margin_shape", value is not None and value.shape == trace_shape,
                        f"{where}.{key}: baseline requires saved [193,B,H,W] trajectory")
            if value is not None and value.shape == trace_shape:
                audit.check("baseline_full_margin_finite", bool(np.all(np.isfinite(value))),
                            f"{where}.{key}: full baseline margin contains nonfinite values")
                full_margins[source] = value
        if len(full_margins) == 3:
            audit.quantile_trajectory_cases += 1
    margins_endpoints = {"margin_endpoints": arrays.get("margin_endpoints"),
                         "original_margin_endpoints": arrays.get("original_margin_endpoints"),
                         "flipped_margin_endpoints": arrays.get("flipped_margin_endpoints")}
    if all(value is not None for value in margins_endpoints.values()):
        initial = domains["primary"] & correct[0]
        _compare_margin_summaries(audit, summary, margins_endpoints, initial, full_margins, where)
    else:
        audit.check("margin_quantiles", False, f"{where}: endpoint margins unavailable")
    return arrays, masks


def _validate_nonfinite_case(audit: Audit, run: Path, row: dict[str, Any],
                             bank_arrays: dict[str, np.ndarray], bank_sha: str) -> None:
    case = str(row.get("case", "unknown"))
    audit.exact("nonfinite_case_marker", row.get("scientific_outcome"),
                "NONFINITE_INTERVENTION", f"{case}.scientific_outcome")
    audit.exact("nonfinite_case_summary", row.get("summary"), None, f"{case}.summary")
    audit.check("nonfinite_case_marker", "trace_file" not in row,
                f"{case}: nonfinite case unexpectedly references a complete trace")
    audit.exact("nonfinite_case_marker", row.get("bank_sha256"), bank_sha, f"{case}.bank_sha256")
    try:
        step = int(row["nonfinite_at_or_before"])
        start = int(row["trace_start_step"])
        expected_time = step - start + 1
        audit.check("nonfinite_case_marker", start in (0, 64) and expected_time > 0,
                    f"{case}: invalid trace start/failed step metadata")
    except (KeyError, TypeError, ValueError):
        audit.check("nonfinite_case_marker", False, f"{case}: nonfinite timing metadata missing")
        return
    path = _safe_file(run, row.get("failed_trace_file"))
    observed_hash = _sha256(path) if path else None
    audit.check("nonfinite_trace_hash", observed_hash == row.get("failed_trace_sha256"),
                f"{case}: failed trace hash does not match")
    if path is None:
        return
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except Exception as exc:
        audit.check("nonfinite_trace_shape", False, f"{case}: failed NPZ read failed ({type(exc).__name__})")
        return
    batch, _, height, width = bank_arrays["x"].shape
    time_shape = (expected_time, batch, height, width)
    for key in ("correct", "original_correct", "flipped_correct"):
        value = arrays.get(key)
        audit.check("nonfinite_trace_shape", value is not None and value.shape == time_shape
                    and value.dtype == np.bool_, f"{case}.{key}: expected truncated boolean shape {time_shape}")
    for key in ("margin", "original_margin", "flipped_margin"):
        value = arrays.get(key)
        audit.check("nonfinite_trace_shape", value is not None and value.shape == time_shape
                    and np.issubdtype(value.dtype, np.number),
                    f"{case}.{key}: expected truncated numeric shape {time_shape}")


def _metric_scalars(summary: Any) -> dict[str, float | None]:
    paths = {
        "strict128_equal_map": ("endpoints", "128", "strict"),
        "acquisition64_128_primary_equal_map": ("intervals", "64_128", "primary", "acquisition"),
        "first_exit64_128_primary_equal_map": ("intervals", "64_128", "primary", "first_exit"),
        "survival64_256_primary_equal_map": ("intervals", "64_256", "primary", "continuous_survival"),
    }
    out = {}
    for name, path in paths.items():
        node = summary
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        value = node.get("equal_map_mean") if isinstance(node, dict) else None
        try:
            value = float(value)
            out[name] = value if math.isfinite(value) else None
        except (TypeError, ValueError, OverflowError):
            out[name] = None
    return out


def _validate_case_inventory(audit: Audit, run: Path,
                             bank_arrays: dict[str, dict[str, np.ndarray]],
                             bank_manifest: dict[str, Any]) -> tuple[
                                 dict[tuple[str, int], dict[str, Any]],
                                 dict[str, dict[str, dict[str, float | None]]],
                                 dict[str, Any]]:
    case_dir = run / "cases"
    paths = sorted(case_dir.glob("*.json")) if case_dir.is_dir() else []
    audit.exact("case_inventory", len(paths), EXPECTED_CASES, "cases/*.json count")
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    identities: set[tuple[str, str]] = set()
    baseline_rows: dict[tuple[str, int], dict[str, Any]] = {}
    factorial_values: dict[str, dict[str, dict[str, float | None]]] = defaultdict(dict)
    case_errors = []

    for path in paths:
        try:
            row = _read_json(path)
        except Exception as exc:
            audit.check("case_schema", False, f"{path.name}: invalid JSON ({type(exc).__name__})")
            case_errors.append(path.name)
            continue
        case = str(row.get("case", ""))
        bank = str(row.get("bank", ""))
        group = str(row.get("group", ""))
        if not case or bank not in bank_arrays or group not in {
            "baseline", "factorial", "interpolation", "cross_continuation",
            "state_swap", "update_pulse", "local_transplant",
        }:
            audit.check("case_schema", False, f"{path.name}: missing/unknown case, bank or group")
            case_errors.append(path.name)
            continue
        audit.exact("case_identity", path.stem, case, f"{path.name}.case vs filename")
        identity = (bank, case)
        audit.check("case_identity", identity not in identities, f"{bank}/{case}: duplicate identity")
        identities.add(identity)
        group_counts[bank][group] += 1
        meta = bank_manifest.get(bank, {})
        size = int(meta.get("size", 32 if bank.endswith("32") else 64))
        confirmation = bank.startswith("confirmation")
        audit.exact("case_identity", row.get("size"), size, f"{case}.size")
        audit.exact("case_identity", row.get("confirmation"), confirmation, f"{case}.confirmation")
        audit.exact("case_bank_binding", row.get("bank_sha256"), meta.get("sha256"), f"{case}.bank_sha256")

        if row.get("summary") is None:
            audit.nonfinite_cases += 1
            _validate_nonfinite_case(audit, run, row, bank_arrays[bank], meta.get("sha256"))
            continue

        audit.finite_cases += 1
        audit.exact("case_schema", row.get("zero_training"), True, f"{case}.zero_training")
        audit.exact("case_schema", row.get("steps"), 256, f"{case}.steps")
        loaded = _load_case_arrays(audit, run, row, bank_arrays[bank])
        if loaded is None:
            case_errors.append(case)
            continue
        arrays, _cohorts = loaded
        if group == "baseline":
            match = re.search(r"_baseline(195|200)$", case)
            if not match:
                audit.check("baseline_inventory", False, f"{case}: baseline checkpoint suffix missing")
            else:
                update = int(match.group(1))
                key = (bank, update)
                audit.check("baseline_inventory", key not in baseline_rows, f"{case}: duplicate checkpoint baseline")
                baseline_rows[key] = row
                # Only primary banks descend from the historical 16-map
                # checkpoint traces. Confirmation banks are a separate
                # deterministic 8-map generator and have no matched trace.
                if bank.startswith("primary"):
                    _compare_historical_replay(audit, run, row, arrays, bank)
        if group == "factorial":
            coalition = row.get("extra", {}).get("blocks_from200")
            audit.check("factorial_identity", coalition in COALITIONS, f"{case}: invalid coalition {coalition!r}")
            if isinstance(coalition, str):
                if coalition in factorial_values[bank]:
                    audit.check("factorial_identity", False, f"{bank}/{coalition}: duplicate coalition")
                factorial_values[bank][coalition] = _metric_scalars(row.get("summary"))

    expected_banks = {
        "primary32": PRIMARY_GROUP_COUNTS,
        "primary64": PRIMARY_GROUP_COUNTS,
        "confirmation32": CONFIRMATION_GROUP_COUNTS,
        "confirmation64": CONFIRMATION_GROUP_COUNTS,
    }
    for bank, expected_groups in expected_banks.items():
        actual = dict(group_counts.get(bank, {}))
        audit.exact("case_group_inventory", actual, expected_groups, f"{bank} group counts")
    audit.exact("finite_nonfinite_inventory", audit.finite_cases + audit.nonfinite_cases,
                len(paths), "finite plus nonfinite case total")
    for bank in expected_banks:
        for update in (195, 200):
            audit.check("baseline_inventory", (bank, update) in baseline_rows,
                        f"missing {bank} baseline u{update}")
        audit.exact("factorial_inventory", set(factorial_values.get(bank, {})), set(COALITIONS),
                    f"{bank} factorial coalition set")
    return baseline_rows, factorial_values, {
        "case_files": len(paths),
        "finite_cases": audit.finite_cases,
        "nonfinite_intervention_cases": audit.nonfinite_cases,
        "cohort_records_recomputed": audit.cohort_records,
        "full_baseline_trajectory_quantile_sets_recomputed": audit.quantile_trajectory_cases,
        "group_counts": {bank: dict(counts) for bank, counts in sorted(group_counts.items())},
        "case_file_errors": case_errors,
        "factorial_cases_by_bank": {bank: len(values) for bank, values in factorial_values.items()},
    }


FACTORIAL_METRICS = (
    "strict128_equal_map",
    "acquisition64_128_primary_equal_map",
    "first_exit64_128_primary_equal_map",
    "survival64_256_primary_equal_map",
)


def _read_json_object_from_stream(stream: Any, initial: str) -> dict[str, Any]:
    """Read one brace-delimited JSON object without loading the giant parent file."""
    chars = []
    depth = 0
    in_string = False
    escaped = False
    started = False
    first = True
    while True:
        if first:
            line = initial
            first = False
        else:
            line = stream.readline()
            if not line:
                break
        for char in line:
            chars.append(char)
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
                started = True
            elif char == "}":
                depth -= 1
                if started and depth == 0:
                    # Stop before a possible sibling-field comma on this line.
                    return json.loads("".join(chars))
    raise ValueError("unterminated JSON object in analysis.json")


def _extract_factorial_sections(path: Path) -> dict[str, dict[str, Any]]:
    banks = {"primary32", "primary64", "confirmation32", "confirmation64"}
    current_bank: str | None = None
    found: dict[str, dict[str, Any]] = {}
    bank_re = re.compile(r'^\s{4}"(primary32|primary64|confirmation32|confirmation64)"\s*:\s*\{\s*$')
    factorial_re = re.compile(r'^\s{6}"factorial"\s*:\s*(\{.*)$')
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            bank_match = bank_re.match(line)
            if bank_match:
                current_bank = bank_match.group(1)
                continue
            if current_bank in banks:
                match = factorial_re.match(line)
                if match:
                    found[current_bank] = _read_json_object_from_stream(stream, match.group(1))
                    current_bank = None
                    if found.keys() >= banks:
                        break
    return found


def _expected_shapley(values: dict[str, float | None], metric: str) -> dict[str, Any]:
    finite = all(values.get(coalition) is not None and math.isfinite(float(values[coalition]))
                 for coalition in COALITIONS)
    if not finite:
        missing = [coalition for coalition in COALITIONS
                   if values.get(coalition) is None or not math.isfinite(float(values[coalition]))]
        return {"complete": False, "missing": missing, "orders": []}
    order_rows = []
    by_block: dict[str, list[float]] = {block: [] for block in BLOCKS}
    for order in ORDERS:
        prefix = ""
        path_values = [{"coalition": "none", "value": float(values["none"])}]
        effects = {}
        for block in order:
            before = "".join(x for x in BLOCKS if x in prefix) or "none"
            prefix += block
            after = "".join(x for x in BLOCKS if x in prefix)
            effect = float(values[after]) - float(values[before])
            effects[block] = effect
            by_block[block].append(effect)
            path_values.append({"coalition": after, "value": float(values[after])})
        order_rows.append({"order": order, "path_values": path_values, "marginal_effects": effects})
    return {
        "complete": True,
        "missing": [],
        "orders": order_rows,
        "by_block": {
            block: {
                "mean": float(np.mean(effects)),
                "minimum": float(np.min(effects)),
                "maximum": float(np.max(effects)),
            }
            for block, effects in by_block.items()
        },
    }


def _validate_factorial_analysis(audit: Audit, run: Path,
                                 factorial_values: dict[str, dict[str, dict[str, float | None]]]) -> dict[str, Any]:
    path = run / "analysis.json"
    try:
        analysis_banks = _extract_factorial_sections(path)
        audit.exact("analysis_factorial_inventory", set(analysis_banks),
                    {"primary32", "primary64", "confirmation32", "confirmation64"},
                    "analysis.json factorial sections")
    except Exception as exc:
        audit.check("analysis_factorial_inventory", False,
                    f"analysis.json factorial extraction failed ({type(exc).__name__})")
        return {"factorial_sections_checked": 0, "shapley_metrics_checked": 0}

    checked_metrics = 0
    for bank in ("primary32", "primary64", "confirmation32", "confirmation64"):
        section = analysis_banks.get(bank, {})
        vals = factorial_values.get(bank, {})
        saved_values = section.get("values_by_metric", {})
        saved_shapley = section.get("shapley", {})
        audit.exact("factorial_analysis", section.get("coalition_count_expected"), 16,
                    f"{bank}.coalition_count_expected")
        for metric in FACTORIAL_METRICS:
            actual_values = saved_values.get(metric, {}) if isinstance(saved_values, dict) else {}
            for coalition in COALITIONS:
                actual = actual_values.get(coalition)
                expected = vals.get(coalition, {}).get(metric)
                audit.approx("factorial_analysis", actual, expected,
                             f"{bank}.{metric}.{coalition}")
            expected_shapley = _expected_shapley(
                {coalition: vals.get(coalition, {}).get(metric) for coalition in COALITIONS}, metric
            )
            actual_shapley = saved_shapley.get(metric, {}) if isinstance(saved_shapley, dict) else {}
            audit.exact("factorial_shapley", actual_shapley.get("complete"),
                        expected_shapley["complete"], f"{bank}.{metric}.complete")
            audit.exact("factorial_shapley", actual_shapley.get("all_24_orders_finite"),
                        expected_shapley["complete"], f"{bank}.{metric}.all_24_orders_finite")
            if expected_shapley["complete"]:
                orders = actual_shapley.get("permutation_orders", [])
                audit.exact("factorial_shapley", len(orders), 24, f"{bank}.{metric}.order_count")
                for index, (actual_order, expected_order) in enumerate(zip(orders, expected_shapley["orders"])):
                    audit.exact("factorial_shapley", actual_order.get("order"), expected_order["order"],
                                f"{bank}.{metric}.orders[{index}].order")
                    _compare_float_tree(audit, actual_order.get("path_values"), expected_order["path_values"],
                                        f"{bank}.{metric}.orders[{index}].path_values")
                    _compare_float_tree(audit, actual_order.get("marginal_effects"), expected_order["marginal_effects"],
                                        f"{bank}.{metric}.orders[{index}].marginal_effects")
                actual_blocks = actual_shapley.get("by_block", {})
                for block, expected_block in expected_shapley["by_block"].items():
                    actual_block = actual_blocks.get(block, {}) if isinstance(actual_blocks, dict) else {}
                    for key, value in expected_block.items():
                        audit.approx("factorial_shapley", actual_block.get(key), value,
                                     f"{bank}.{metric}.by_block.{block}.{key}")
                    audit.exact("factorial_shapley", actual_block.get("order_count"), 24,
                                f"{bank}.{metric}.by_block.{block}.order_count")
            else:
                audit.exact("factorial_shapley", actual_shapley.get("permutation_orders"), [],
                            f"{bank}.{metric}.incomplete.permutation_orders")
                audit.exact("factorial_shapley", actual_shapley.get("by_block"), {},
                            f"{bank}.{metric}.incomplete.by_block")
                audit.exact("factorial_shapley", actual_shapley.get("missing_or_nonfinite_coalitions"),
                            expected_shapley["missing"], f"{bank}.{metric}.missing")
            checked_metrics += 1
    return {"factorial_sections_checked": len(analysis_banks), "shapley_metrics_checked": checked_metrics}


def _bank_label_map(value: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.shape == (shape[0],):
        arr = np.broadcast_to(arr[:, None, None], shape)
    return np.broadcast_to(arr, shape) >= 0.5


def _plan_features(bank: dict[str, np.ndarray], correct64: np.ndarray,
                   margin64: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    changed = _as_map(bank["changed"]).astype(bool)
    opened = _as_map(bank["mask"]).astype(bool)
    distance = _as_map(bank["distance"])
    domain = changed & opened & (distance > 0)
    solved = domain & correct64
    adjacent = np.zeros_like(solved)
    adjacent[:, 1:, :] |= solved[:, :-1, :]
    adjacent[:, :-1, :] |= solved[:, 1:, :]
    adjacent[:, :, 1:] |= solved[:, :, :-1]
    adjacent[:, :, :-1] |= solved[:, :, 1:]
    frontier = domain & ~correct64 & adjacent
    batch, height, width = domain.shape
    degree = np.zeros_like(domain, dtype=np.uint8)
    degree[:, 1:, :] += opened[:, :-1, :]
    degree[:, :-1, :] += opened[:, 1:, :]
    degree[:, :, 1:] += opened[:, :, :-1]
    degree[:, :, :-1] += opened[:, :, 1:]
    y = _bank_label_map(bank["y"], domain.shape).astype(np.int8)
    y_flip = _bank_label_map(bank["y_flip"], domain.shape).astype(np.int8)
    dist_bin = np.full(domain.shape, -1, dtype=np.int8)
    dist_bin[(distance >= 1) & (distance <= 16)] = 0
    dist_bin[(distance >= 17) & (distance <= 31)] = 1
    dist_bin[(distance >= 32) & (distance <= 63)] = 2
    dist_bin[distance >= 64] = 3
    abs_margin = np.abs(np.asarray(margin64, dtype=np.float64))
    margin_bin = np.where(abs_margin < 0.25, 0, np.where(abs_margin < 1.0, 1, 2)).astype(np.int8)
    features = {"map": np.broadcast_to(np.arange(batch)[:, None, None], domain.shape),
                "class_y": y, "class_y_flip": y_flip, "four_neighbor_open_degree": degree,
                "distance_band": dist_bin, "absolute_margin_bin": margin_bin}
    return solved, frontier, features


def _validate_swap_plans(audit: Audit, run: Path,
                         baseline_rows: dict[tuple[str, int], dict[str, Any]],
                         banks: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    plan_dir = run / "plans"
    paths = sorted(plan_dir.glob("*.json")) if plan_dir.is_dir() else []
    expected_names = {f"{bank}_u{update}_{role}.json"
                      for bank in banks for update in (195, 200) for role in ("solved", "frontier")}
    audit.exact("swap_plan_inventory", {path.name for path in paths}, expected_names,
                "plans directory entries")
    checked = 0
    for name in sorted(expected_names):
        path = plan_dir / name
        match = re.fullmatch(r"(primary32|primary64|confirmation32|confirmation64)_u(195|200)_(solved|frontier)\.json", name)
        if not match or not path.is_file():
            audit.check("swap_plan_schema", False, f"{name}: malformed or absent plan")
            continue
        bank_name, update_s, role = match.groups()
        update = int(update_s)
        row = baseline_rows.get((bank_name, update))
        bank = banks.get(bank_name, {})
        if row is None or "x" not in bank:
            audit.check("swap_plan_binding", False, f"{name}: baseline or bank missing")
            continue
        trace_path = _safe_file(run, row.get("trace_file"))
        try:
            with np.load(trace_path, allow_pickle=False) as archive:
                correct64 = np.asarray(archive["correct"][0], dtype=bool)
                margin64 = np.asarray(archive["margin_endpoints"][0], dtype=np.float64)
            saved = _read_json(path)
            targets = np.asarray(saved.get("target_flat", []), dtype=np.int64)
            donors = np.asarray(saved.get("donor_flat", []), dtype=np.int64)
            counts = saved.get("counts", {})
            metadata = saved.get("metadata", {})
            solved, frontier, features = _plan_features(bank, correct64, margin64)
            eligible = solved if role == "solved" else frontier
            shape = eligible.shape
            total = int(np.prod(shape))
            audit.check("swap_plan_shape", targets.ndim == donors.ndim == 1 and targets.shape == donors.shape,
                        f"{name}: target/donor vectors must have matching rank-1 shape")
            audit.check("swap_plan_range", bool(np.all((targets >= 0) & (targets < total))
                        and np.all((donors >= 0) & (donors < total))), f"{name}: index out of range")
            audit.exact("swap_plan_involution", len(np.unique(targets)), len(targets), f"{name}: target uniqueness")
            audit.exact("swap_plan_involution", len(np.unique(donors)), len(donors), f"{name}: donor uniqueness")
            audit.exact("swap_plan_involution", set(targets.tolist()), set(donors.tolist()), f"{name}: reciprocal set")
            audit.check("swap_plan_involution", bool(np.all(targets != donors)), f"{name}: fixed point exists")
            mapping = dict(zip(targets.tolist(), donors.tolist()))
            audit.check("swap_plan_involution", all(mapping.get(mapping.get(t, -1), -2) == t for t in mapping),
                        f"{name}: mapping is not an involution")
            audit.exact("swap_plan_counts", int(counts.get("swap_targets", -1)), len(targets), f"{name}: swap_targets")
            audit.exact("swap_plan_counts", int(counts.get("pairs", -1)), len(targets) // 2, f"{name}: pair count")
            audit.check("swap_plan_counts", len(targets) % 2 == 0, f"{name}: odd number of targets")
            audit.exact("swap_plan_metadata", metadata.get("role"), role, f"{name}: role")
            audit.exact("swap_plan_metadata", metadata.get("donor_mode"), "within_map", f"{name}: donor mode")
            audit.exact("swap_plan_metadata", metadata.get("max_pairs_per_map"), 8, f"{name}: per-map pair cap")
            audit.exact("swap_plan_metadata", metadata.get("teacher_or_diagnostic_role_mask_exposed_to_model"),
                        False, f"{name}: role mask exposure")
            matches = metadata.get("matching", [])
            required = {"map", "class_y", "class_y_flip", "four_neighbor_open_degree",
                        "distance_band", "absolute_margin_bin"}
            if role == "solved":
                required.add("current_correct_run_age_bin")
            audit.check("swap_plan_metadata", required.issubset(set(matches)),
                        f"{name}: metadata matching keys incomplete")
            valid_targets = set(np.flatnonzero(eligible.reshape(-1)).tolist())
            audit.exact("swap_plan_eligibility", set(targets.tolist()),
                        set(targets.tolist()) & valid_targets, f"{name}: target role eligibility")
            per_map_targets = np.bincount(targets // (shape[1] * shape[2]), minlength=shape[0]) if len(targets) else np.zeros(shape[0], dtype=int)
            audit.check("swap_plan_counts", bool(np.all(per_map_targets <= 16)),
                        f"{name}: exceeded eight pair slots on a map")
            mismatched_pairs = 0
            flat_features = {key: value.reshape(-1) for key, value in features.items()}
            for target, donor in mapping.items():
                if any(flat_features[key][target] != flat_features[key][donor]
                       for key in required - {"current_correct_run_age_bin"}):
                    mismatched_pairs += 1
            audit.exact("swap_plan_stratum_matching", mismatched_pairs, 0, f"{name}: matching stratum mismatches")
            audit.exact("swap_plan_counts", int(counts.get("eligible_cells", -1)), int(eligible.sum()),
                        f"{name}: eligible cell count")
            if isinstance(counts.get("per_map"), list):
                audit.exact("swap_plan_counts", len(counts["per_map"]), shape[0], f"{name}: per-map count rows")
                for map_index, map_row in enumerate(counts["per_map"]):
                    audit.exact("swap_plan_counts", map_row.get("map_index"), map_index,
                                f"{name}.per_map[{map_index}].map_index")
                    audit.exact("swap_plan_counts", map_row.get("eligible_cells"), int(eligible[map_index].sum()),
                                f"{name}.per_map[{map_index}].eligible_cells")
                    audit.exact("swap_plan_counts", map_row.get("swap_targets"), int(per_map_targets[map_index]),
                                f"{name}.per_map[{map_index}].swap_targets")
            else:
                audit.check("swap_plan_counts", False, f"{name}: per-map counts missing")
            checked += 1
        except Exception as exc:
            audit.check("swap_plan_read", False, f"{name}: {type(exc).__name__}")
    return {"plans_expected": len(expected_names), "plans_checked": checked,
            "age_bin_matching_independently_recomputed": False}


def validate_run(run: Path) -> dict[str, Any]:
    audit = Audit()
    try:
        manifest = _read_json(run / "manifest.json")
    except Exception as exc:
        audit.check("manifest", False, f"manifest.json cannot be read ({type(exc).__name__})")
        manifest = {}
    run_status = _validate_run_controls(audit, run, manifest)
    binding_report = _hash_binding_group(audit, run, manifest.get("bindings", {}))
    checkpoint_report = _validate_reference_checkpoints(audit, manifest.get("bindings", {}))
    banks = _load_saved_banks(audit, run, manifest, checkpoint_report)
    baseline_rows, factorial_values, case_report = _validate_case_inventory(audit, run, banks, manifest.get("audit_banks", {}))
    plan_report = _validate_swap_plans(audit, run, baseline_rows, banks)
    factorial_report = _validate_factorial_analysis(audit, run, factorial_values)
    return {
        "schema_version": "audit-195-200-saved-validation-v1",
        "validation_status": "PASS" if audit.passed else "FAIL",
        "run_directory": run.name,
        "validation_scope": "saved-artifact, offline CPU validation; no inference or training",
        "run_controls": run_status,
        "artifacts": {
            "source_and_reference_bindings": binding_report,
            "reference_checkpoints": checkpoint_report,
            "cases": case_report,
            "swap_plans": plan_report,
            "factorial_analysis": factorial_report,
        },
        "checks": audit.report_counts(),
        "check_totals": {
            "checked": sum(value.get("checked", 0) for value in audit.counts.values()),
            "passed": sum(value.get("passed", 0) for value in audit.counts.values()),
            "failed": sum(value.get("failed", 0) for value in audit.counts.values()),
        },
        "numerical_tolerance": FLOAT_TOL,
        "maximum_absolute_float_error": audit.max_float_error,
        "limitations": [
            "The audit verifies saved artifacts and deterministic summaries; it does not independently establish a global causal mechanism.",
            "For nonbaseline counterfactuals, only the three saved endpoint margins are available; full-trajectory margin quantiles are not recomputed.",
            "Solved-plan age-bin membership is declared by frozen metadata but not independently reconstructed; role, other matching keys, reciprocity, and map caps are checked.",
            "Case-level raw hidden states are not retained here, so no new hidden-state inference or intervention is performed.",
        ],
        "errors": audit.errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="completed run directory, relative to project root or absolute")
    parser.add_argument("--out", required=True, help="new JSON report path, relative to project root or absolute")
    args = parser.parse_args(argv)
    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    run = run.resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()
    try:
        out.relative_to(ROOT)
    except ValueError:
        parser.error("--out must remain inside the project root")
    if out.exists():
        parser.error("--out must name a new file; refusing to overwrite existing output")
    report = validate_run(run)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"validation_status": report["validation_status"],
                      "run_directory": report["run_directory"],
                      "case_files": report["artifacts"]["cases"]["case_files"],
                      "checked": report["check_totals"]["checked"],
                      "failed": report["check_totals"]["failed"],
                      "out": str(out.relative_to(ROOT)).replace("\\", "/")},
                     sort_keys=True))
    return 0 if report["validation_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
