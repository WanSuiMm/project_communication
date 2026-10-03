"""Summarize saved cases for the bounded historical 195/200 mechanism audit.

This module reads case JSON, saved NPZ traces and banks only. It never creates
or evaluates a model. Call :func:`analyze` after the run has finished.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


BLOCKS = "EFQR"
ORDERS = tuple("".join(order) for order in itertools.permutations(BLOCKS))
COALITIONS = tuple(
    "".join(block for block in BLOCKS if block in selected) or "none"
    for size in range(len(BLOCKS) + 1)
    for selected in itertools.combinations(BLOCKS, size)
)
METRIC_SPECS: dict[str, tuple[str, ...]] = {
    "strict128_equal_map": ("endpoints", "128", "strict"),
    "acquisition64_128_primary_equal_map": ("intervals", "64_128", "primary", "acquisition"),
    "first_exit64_128_primary_equal_map": ("intervals", "64_128", "primary", "first_exit"),
    "survival64_256_primary_equal_map": ("intervals", "64_256", "primary", "continuous_survival"),
}
PRIMARY_CASES_PER_BANK = 147
CONFIRMATION_CASES_PER_BANK = 99
EXPECTED_TOTAL_CASES = 2 * PRIMARY_CASES_PER_BANK + 2 * CONFIRMATION_CASES_PER_BANK
ENDPOINT_STEPS = (64, 128, 256)
INTERVALS = ((64, 128), (64, 256))
MARGIN_SOURCES = ("paired_min", "original", "flipped")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    """Convert nested scalar metadata to strict JSON values; nonfinite=>null."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path.name}")
    return value


def _case_metric_node(summary: Any, metric: str) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    path = METRIC_SPECS[metric]
    node: Any = summary
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


def _metric_record(summary: Any) -> dict[str, Any]:
    result = {}
    for metric in METRIC_SPECS:
        node = _case_metric_node(summary, metric)
        pooled = node.get("pooled") if node else None
        eligible = node.get("equal_map_eligible_maps") if node else None
        result[metric] = {
            "equal_map_mean": _finite(node.get("equal_map_mean")) if node else None,
            "equal_map_eligible_maps": int(eligible) if isinstance(eligible, (int, np.integer)) else None,
            "pooled": _json_safe(pooled) if isinstance(pooled, dict) else None,
        }
    return result


def _metric_value(record: dict[str, Any], metric: str) -> float | None:
    return _finite(record.get(metric, {}).get("equal_map_mean"))


def _case_status(row: dict[str, Any]) -> str:
    if row.get("scientific_outcome"):
        return str(row["scientific_outcome"])
    if row.get("summary") is None:
        return "SUMMARY_MISSING"
    return "RECORDED"


def _load_cases(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    directory = out / "cases"
    if not directory.is_dir():
        raise FileNotFoundError(f"Saved case directory not found: {directory}")
    rows = []
    errors = []
    for path in sorted(directory.glob("*.json")):
        try:
            row = _read_json(path)
            row.setdefault("_source_file", path.name)
            if not row.get("case") or not row.get("bank"):
                errors.append(f"{path.name}: missing case or bank identity")
                continue
            rows.append(row)
        except Exception as exc:  # Keep inventory visible; do not silently drop malformed evidence.
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("bank")), str(row["case"]))
        if key in seen:
            errors.append(f"duplicate case identity: {key[0]}/{key[1]}")
        seen.add(key)
    return rows, errors


def _bank_inventory(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        bank = str(row.get("bank", "unknown"))
        item = result.setdefault(bank, {
            "size": row.get("size"),
            "confirmation": bool(row.get("confirmation", False)),
            "groups": {},
            "rows": [],
        })
        item["rows"].append(row)
        group = str(row.get("group", "unknown"))
        item["groups"][group] = item["groups"].get(group, 0) + 1
    for item in result.values():
        item["rows"].sort(key=lambda row: (str(row.get("group", "")), str(row["case"])))
    return result


def _find_baselines(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    errors = []
    for row in rows:
        if row.get("group") != "baseline":
            continue
        case = str(row.get("case", ""))
        update = next((u for u in (195, 200) if case.endswith(f"_baseline{u}")), None)
        if update is None:
            errors.append(f"{case}: baseline identity does not end in _baseline195/_baseline200")
            continue
        key = (str(row["bank"]), update)
        if key in result:
            errors.append(f"duplicate baseline: {key[0]}/u{key[1]}")
        result[key] = row
    return result, errors


def _factorial_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_coalition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        coal = row.get("extra", {}).get("blocks_from200")
        if isinstance(coal, str):
            by_coalition.setdefault(coal, []).append(row)
    value_by_metric: dict[str, dict[str, float | None]] = {m: {} for m in METRIC_SPECS}
    status_by_coalition = {}
    duplicate_coalitions = []
    for coalition in COALITIONS:
        records = by_coalition.get(coalition, [])
        if len(records) > 1:
            duplicate_coalitions.append(coalition)
        row = records[0] if len(records) == 1 else None
        status_by_coalition[coalition] = {
            "record_count": len(records),
            "case": row.get("case") if row else None,
            "status": _case_status(row) if row else "MISSING_OR_DUPLICATE",
        }
        metric_record = _metric_record(row.get("summary") if row else None)
        for metric in METRIC_SPECS:
            value_by_metric[metric][coalition] = _metric_value(metric_record, metric)

    shapley: dict[str, Any] = {}
    for metric, coalition_values in value_by_metric.items():
        missing = [coalition for coalition in COALITIONS if coalition_values[coalition] is None]
        if duplicate_coalitions:
            complete = False
        else:
            complete = not missing and all(len(by_coalition.get(c, [])) == 1 for c in COALITIONS)
        if not complete:
            shapley[metric] = {
                "complete": False,
                "missing_or_nonfinite_coalitions": missing,
                "duplicate_coalitions": duplicate_coalitions,
                "all_24_orders_finite": False,
                "reason": "Shapley effects are omitted unless all 16 unique coalition outcomes are finite.",
                "permutation_orders": [],
                "by_block": {},
            }
            continue

        order_rows = []
        block_effects: dict[str, list[dict[str, Any]]] = {block: [] for block in BLOCKS}
        for order in ORDERS:
            prefix = ""
            path_values = [{"coalition": "none", "value": coalition_values["none"]}]
            marginals = {}
            for block in order:
                prefix += block
                before_prefix = prefix[:-1]
                before = "".join(letter for letter in BLOCKS if letter in before_prefix) or "none"
                after = "".join(letter for letter in BLOCKS if letter in prefix)
                effect = float(coalition_values[after] - coalition_values[before])
                marginals[block] = effect
                block_effects[block].append({"order": order, "marginal_effect": effect})
                path_values.append({"coalition": after, "value": coalition_values[after]})
            order_rows.append({"order": order, "path_values": path_values, "marginal_effects": marginals})
        by_block = {}
        for block, effects in block_effects.items():
            values = [float(item["marginal_effect"]) for item in effects]
            by_block[block] = {
                "mean": float(np.mean(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "order_count": len(values),
                "by_order": effects,
            }
        shapley[metric] = {
            "complete": True,
            "missing_or_nonfinite_coalitions": [],
            "duplicate_coalitions": [],
            "all_24_orders_finite": True,
            "permutation_orders": order_rows,
            "by_block": by_block,
        }

    return {
        "coalition_count_expected": len(COALITIONS),
        "coalitions": status_by_coalition,
        "values_by_metric": value_by_metric,
        "shapley": shapley,
    }


def check_additive_fixture() -> None:
    """Cheap deterministic check of all-order effects and incomplete handling."""
    coefficients = {
        "strict128_equal_map": {"E": 0.03, "F": -0.02, "Q": 0.01, "R": 0.04},
        "acquisition64_128_primary_equal_map": {"E": -0.01, "F": 0.02, "Q": 0.03, "R": -0.02},
        "first_exit64_128_primary_equal_map": {"E": 0.02, "F": 0.01, "Q": -0.03, "R": 0.01},
        "survival64_256_primary_equal_map": {"E": 0.04, "F": -0.01, "Q": -0.02, "R": 0.03},
    }
    bases = {metric: 0.4 for metric in METRIC_SPECS}
    rows = []
    for coalition in COALITIONS:
        summary: dict[str, Any] = {}
        for metric, path in METRIC_SPECS.items():
            value = bases[metric] + sum(coefficients[metric][block] for block in coalition if block in BLOCKS)
            node = summary
            for part in path:
                node = node.setdefault(part, {})
            node["equal_map_mean"] = value
        rows.append({"case": f"fixture_{coalition}", "extra": {"blocks_from200": coalition}, "summary": summary})

    result = _factorial_summary(rows)
    if len(ORDERS) != 24 or any(not result["shapley"][metric]["complete"] for metric in METRIC_SPECS):
        raise AssertionError("Additive fixture did not yield 24 complete order marginals")
    for metric, block_coefficients in coefficients.items():
        for block, expected in block_coefficients.items():
            record = result["shapley"][metric]["by_block"][block]
            if record["order_count"] != 24 or not np.isclose(record["mean"], expected, rtol=0, atol=1e-12):
                raise AssertionError(f"Additive Shapley mismatch for {metric}/{block}: {record}")
            if not np.isclose(record["minimum"], expected, rtol=0, atol=1e-12):
                raise AssertionError(f"Additive minimum mismatch for {metric}/{block}")
            if not np.isclose(record["maximum"], expected, rtol=0, atol=1e-12):
                raise AssertionError(f"Additive maximum mismatch for {metric}/{block}")
    incomplete = _factorial_summary([row for row in rows if row["extra"]["blocks_from200"] != "Q"])
    for metric in METRIC_SPECS:
        record = incomplete["shapley"][metric]
        if record["complete"] or record["by_block"] or record["all_24_orders_finite"]:
            raise AssertionError(f"Incomplete coalition incorrectly emitted Shapley values: {metric}")
        if "Q" not in record["missing_or_nonfinite_coalitions"]:
            raise AssertionError(f"Missing coalition not identified for {metric}")


def check_baseline_suffix_fixture() -> None:
    """Verify baseline NPZs saved as suffixes are not sliced a second time."""
    with tempfile.TemporaryDirectory(prefix="audit195_analyze_fixture_") as temporary:
        out = Path(temporary)
        correct = np.zeros((193, 2, 1, 3), dtype=bool)
        correct[0, 0, 0, 0] = True
        margins = np.arange(correct.size, dtype=np.float32).reshape(correct.shape)
        np.savez_compressed(
            out / "baseline.npz",
            correct=correct,
            margin=margins,
            original_margin=margins + 1,
            flipped_margin=margins - 1,
        )
        loaded = _load_baseline_for_cohorts(out, {"trace_file": "baseline.npz"})
        if loaded is None:
            raise AssertionError("193-step suffix baseline NPZ was not accepted")
        loaded_correct, loaded_margins = loaded
        if loaded_correct.shape != correct.shape or not np.array_equal(loaded_correct, correct):
            raise AssertionError("193-step baseline Boolean suffix changed")
        if not np.array_equal(loaded_margins["paired_min"], margins):
            raise AssertionError("193-step baseline margin suffix changed")


def _interpolation_summary(rows: list[dict[str, Any]],
                           baselines: dict[tuple[str, int], dict[str, Any]],
                           bank: str) -> dict[str, Any]:
    actual_by_lambda: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        value = _finite(row.get("extra", {}).get("lambda"))
        if value is not None:
            actual_by_lambda.setdefault(value, []).append(row)
    points = []
    for tenth in range(11):
        lam = tenth / 10
        if tenth == 0 or tenth == 10:
            update = 195 if tenth == 0 else 200
            row = baselines.get((bank, update))
            points.append({
                "lambda": lam,
                "source": "saved_endpoint_baseline",
                "case": row.get("case") if row else None,
                "status": _case_status(row) if row else "MISSING_BASELINE",
                "metrics": _metric_record(row.get("summary") if row else None),
            })
            continue
        matches = actual_by_lambda.get(lam, [])
        row = matches[0] if len(matches) == 1 else None
        points.append({
            "lambda": lam,
            "source": "saved_interpolation_case",
            "case": row.get("case") if row else None,
            "record_count": len(matches),
            "status": _case_status(row) if row else "MISSING_OR_DUPLICATE",
            "metrics": _metric_record(row.get("summary") if row else None),
        })
    expected_lambdas = {round(i / 10, 1) for i in range(1, 10)}
    found_lambdas = set(actual_by_lambda)
    return {
        "endpoint_note": "lambda=0 and lambda=1 use the saved u195/u200 baseline cases; only lambda=.1..9 are interpolated cases.",
        "complete": all(point["status"] in ("RECORDED",) for point in points)
                    and expected_lambdas.issubset(found_lambdas)
                    and all(len(actual_by_lambda.get(lam, [])) == 1 for lam in expected_lambdas),
        "missing_or_duplicate_interpolation_lambdas": [
            round(lam, 1) for lam in sorted(expected_lambdas)
            if len(actual_by_lambda.get(lam, [])) != 1
        ],
        "points": points,
    }


def _cohort_view(cohort: Any) -> dict[str, Any] | None:
    if not isinstance(cohort, dict):
        return None
    endpoints = cohort.get("endpoints", {})
    first_exit = cohort.get("first_exit", {})
    all_steps = cohort.get("all_steps_correct", {})
    any_wrong = cohort.get("any_wrong_including_start", {})
    margin = cohort.get("net_margin_change", {})
    return {
        "selected_pixels_per_map": cohort.get("selected_pixels_per_map"),
        "endpoints": {str(k): _json_safe(v) for k, v in endpoints.items()},
        "first_exit": {str(k): _json_safe(v) for k, v in first_exit.items()},
        "all_steps_correct": {str(k): _json_safe(v) for k, v in all_steps.items()},
        "any_wrong_including_start": {str(k): _json_safe(v) for k, v in any_wrong.items()},
        "net_margin_change": {str(k): _json_safe(v) for k, v in margin.items()},
    }


def _cohort_effect(baseline: dict[str, Any] | None,
                   intervention: dict[str, Any] | None) -> dict[str, Any]:
    if baseline is None or intervention is None:
        return {"baseline": baseline, "intervention": intervention, "deltas": None}
    deltas: dict[str, Any] = {"endpoints": {}, "first_exit": {},
                              "all_steps_correct": {}, "any_wrong_including_start": {},
                              "net_margin_change": {}}

    def rate_delta(section: str, key: str) -> dict[str, Any]:
        base = baseline.get(section, {}).get(key)
        current = intervention.get(section, {}).get(key)
        base_mean = _finite(base.get("equal_map_mean")) if isinstance(base, dict) else None
        current_mean = _finite(current.get("equal_map_mean")) if isinstance(current, dict) else None
        delta = current_mean - base_mean if base_mean is not None and current_mean is not None else None
        return {"baseline": base, "intervention": current, "delta_equal_map_mean": delta}

    for step in map(str, ENDPOINT_STEPS):
        deltas["endpoints"][step] = rate_delta("endpoints", step)
    for section in ("first_exit", "all_steps_correct", "any_wrong_including_start"):
        for interval in ("64_128", "64_256"):
            deltas[section][interval] = rate_delta(section, interval)
    for interval in ("64_128", "64_256"):
        deltas["net_margin_change"][interval] = {}
        for source in MARGIN_SOURCES:
            base = baseline.get("net_margin_change", {}).get(interval, {}).get(source)
            current = intervention.get("net_margin_change", {}).get(interval, {}).get(source)
            base_mean = _finite(base.get("equal_map_mean")) if isinstance(base, dict) else None
            current_mean = _finite(current.get("equal_map_mean")) if isinstance(current, dict) else None
            per_map_deltas = []
            b_rows = base.get("per_map", []) if isinstance(base, dict) else []
            i_rows = current.get("per_map", []) if isinstance(current, dict) else []
            for b_row, i_row in zip(b_rows, i_rows):
                b_value, i_value = _finite(b_row.get("mean")), _finite(i_row.get("mean"))
                per_map_deltas.append({
                    "map_index": b_row.get("map_index"),
                    "baseline": b_value,
                    "intervention": i_value,
                    "delta": i_value - b_value if b_value is not None and i_value is not None else None,
                    "n_baseline": b_row.get("n"),
                    "n_intervention": i_row.get("n"),
                })
            deltas["net_margin_change"][interval][source] = {
                "baseline": base,
                "intervention": current,
                "delta_equal_map_mean": current_mean - base_mean
                    if base_mean is not None and current_mean is not None else None,
                "per_map_delta": per_map_deltas,
            }
    return {"baseline": baseline, "intervention": intervention, "deltas": deltas}


def _rate_summary(nums: np.ndarray, dens: np.ndarray) -> dict[str, Any]:
    per_map = []
    rates = []
    for map_index, (num, den) in enumerate(zip(nums, dens)):
        num, den = int(num), int(den)
        rate = num / den if den else None
        per_map.append({"map_index": map_index, "numerator": num, "denominator": den, "rate": rate})
        if rate is not None:
            rates.append(rate)
    n_total, d_total = int(nums.sum()), int(dens.sum())
    return {
        "pooled": {"numerator": n_total, "denominator": d_total,
                   "rate": n_total / d_total if d_total else None},
        "per_map": per_map,
        "equal_map_mean": float(np.mean(rates)) if rates else None,
        "equal_map_eligible_maps": len(rates),
    }


def _baseline_cohort_summary(correct: np.ndarray, margin_arrays: dict[str, np.ndarray],
                             bank_domain: np.ndarray,
                             cohort_mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(cohort_mask, dtype=bool) & bank_domain
    pixels = np.count_nonzero(selected, axis=(1, 2)).astype(np.int64)
    endpoints = {}
    for step, index in ((64, 0), (128, 64), (256, 192)):
        nums = np.count_nonzero(selected & correct[index], axis=(1, 2)).astype(np.int64)
        endpoints[str(step)] = _rate_summary(nums, pixels)

    first_exit = {}
    all_steps_correct = {}
    any_wrong_including_start = {}
    for start, end, a, b in ((64, 128, 0, 64), (64, 256, 0, 192)):
        start_correct = selected & correct[a]
        ever_wrong_after = np.any(~correct[a + 1:b + 1], axis=0)
        exited = np.count_nonzero(start_correct & ever_wrong_after, axis=(1, 2)).astype(np.int64)
        first_exit[f"{start}_{end}"] = _rate_summary(exited, np.count_nonzero(start_correct, axis=(1, 2)))
        every_step = np.all(correct[a:b + 1], axis=0)
        survived = np.count_nonzero(selected & every_step, axis=(1, 2)).astype(np.int64)
        all_steps_correct[f"{start}_{end}"] = _rate_summary(survived, pixels)
        has_wrong = np.any(~correct[a:b + 1], axis=0)
        wrong = np.count_nonzero(selected & has_wrong, axis=(1, 2)).astype(np.int64)
        any_wrong_including_start[f"{start}_{end}"] = _rate_summary(wrong, pixels)

    net_margin_change = {}
    for start, end, a, b in ((64, 128, 0, 64), (64, 256, 0, 192)):
        interval_key = f"{start}_{end}"
        net_margin_change[interval_key] = {}
        for source, values in margin_arrays.items():
            delta = values[b] - values[a]
            per_map = []
            map_means = []
            pool_parts = []
            for map_index in range(selected.shape[0]):
                chosen = delta[map_index][selected[map_index]].astype(np.float64, copy=False)
                mean = float(chosen.mean()) if chosen.size else None
                per_map.append({"map_index": map_index, "n": int(chosen.size), "mean": mean})
                if chosen.size:
                    map_means.append(mean)
                    pool_parts.append(chosen)
            pooled = np.concatenate(pool_parts) if pool_parts else np.empty(0, dtype=np.float64)
            net_margin_change[interval_key][source] = {
                "per_map": per_map,
                "equal_map_mean": float(np.mean(map_means)) if map_means else None,
                "equal_map_eligible_maps": len(map_means),
                "pooled_descriptive": {"n": int(pooled.size),
                                       "mean": float(pooled.mean()) if pooled.size else None},
            }
    return {
        "selected_pixels_per_map": [int(value) for value in pixels],
        "endpoints": endpoints,
        "first_exit": first_exit,
        "all_steps_correct": all_steps_correct,
        "any_wrong_including_start": any_wrong_including_start,
        "net_margin_change": net_margin_change,
    }


def _safe_relative_file(out: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    path = (out / relative).resolve()
    try:
        path.relative_to(out.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _load_npz_cohorts(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as saved:
        for name in names:
            key = f"cohort_{name}"
            if key in saved.files:
                masks[name] = np.asarray(saved[key], dtype=bool)
    return masks


def _load_baseline_for_cohorts(out: Path, baseline_row: dict[str, Any]
                               ) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    path = _safe_relative_file(out, baseline_row.get("trace_file"))
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as saved:
        if "correct" not in saved.files or "margin" not in saved.files:
            return None
        correct_saved = np.asarray(saved["correct"])
        margin_saved = np.asarray(saved["margin"])
        if correct_saved.shape[0] == 257 and margin_saved.shape[0] == 257:
            correct_saved = correct_saved[64:257]
            margin_saved = margin_saved[64:257]
        elif correct_saved.shape[0] != 193 or margin_saved.shape[0] != 193:
            return None
        correct = np.asarray(correct_saved, dtype=bool)
        margins = {"paired_min": margin_saved}
        for key, source in (("original_margin", "original"), ("flipped_margin", "flipped")):
            if key in saved.files:
                values = np.asarray(saved[key])
                if values.shape[0] == 257:
                    values = values[64:257]
                if values.shape[0] != 193:
                    return None
                margins[source] = values
    if correct.shape[0] != 193 or any(values.shape != correct.shape for values in margins.values()):
        return None
    return correct, margins


def _load_bank_domain(out: Path, bank: str) -> np.ndarray | None:
    path = out / f"bank_{bank}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        required = ("changed", "mask", "distance")
        if not all(key in saved.files for key in required):
            return None
        values = {}
        for key in required:
            arr = np.asarray(saved[key])
            if arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0]
            values[key] = arr
    if not (values["changed"].shape == values["mask"].shape == values["distance"].shape):
        return None
    return (values["changed"].astype(bool) & values["mask"].astype(bool)
            & (values["distance"] > 0))


def _global_effect(baseline_summary: Any, intervention_summary: Any) -> dict[str, Any]:
    base = _metric_record(baseline_summary)
    current = _metric_record(intervention_summary)
    result = {}
    for metric in METRIC_SPECS:
        b = _metric_value(base, metric)
        i = _metric_value(current, metric)
        result[metric] = {
            "baseline": base[metric],
            "intervention": current[metric],
            "delta_equal_map_mean": i - b if b is not None and i is not None else None,
        }
    return result


def _state_pulse_effects(out: Path, rows: list[dict[str, Any]],
                         baselines: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    effects = []
    groups = {"state_swap", "local_transplant", "update_pulse"}
    rows_by_baseline: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("group") not in groups:
            continue
        extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
        checkpoint = extra.get("checkpoint")
        if checkpoint is None:
            checkpoint = extra.get("receiver")
        try:
            checkpoint = int(checkpoint)
        except (TypeError, ValueError):
            checkpoint = -1
        rows_by_baseline.setdefault((str(row["bank"]), checkpoint), []).append(row)

    bank_domain_cache: dict[str, np.ndarray | None] = {}
    base_trace_cache: dict[tuple[str, int], tuple[np.ndarray, dict[str, np.ndarray]] | None] = {}
    for key in sorted(rows_by_baseline):
        bank, checkpoint = key
        baseline_row = baselines.get(key)
        if bank not in bank_domain_cache:
            bank_domain_cache[bank] = _load_bank_domain(out, bank)
        if key not in base_trace_cache and baseline_row is not None:
            base_trace_cache[key] = _load_baseline_for_cohorts(out, baseline_row)
        baseline_trace = base_trace_cache.get(key)
        domain = bank_domain_cache[bank]

        for row in sorted(rows_by_baseline[key], key=lambda item: str(item["case"])):
            extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
            summary = row.get("summary")
            trace_path = _safe_relative_file(out, row.get("trace_file"))
            cohort_names = sorted(summary.get("cohorts", {})) if isinstance(summary, dict) else []
            masks = _load_npz_cohorts(trace_path, cohort_names) if trace_path else {}
            baseline_cohort_summaries = {}
            intervention_cohorts = summary.get("cohorts", {}) if isinstance(summary, dict) else {}
            if baseline_trace is not None and domain is not None:
                correct, margins = baseline_trace
                for name, mask in masks.items():
                    baseline_cohort_summaries[name] = _baseline_cohort_summary(correct, margins, domain, mask)

            cohort_effects = {}
            for name in sorted(set(baseline_cohort_summaries) | set(intervention_cohorts)):
                base_view = _cohort_view(baseline_cohort_summaries.get(name))
                intervention_view = _cohort_view(intervention_cohorts.get(name))
                cohort_effects[name] = _cohort_effect(base_view, intervention_view)

            baseline_summary = baseline_row.get("summary") if baseline_row else None
            metadata_fields = {
                "state_swap": ("checkpoint", "role", "component", "eligible_targets", "interpretability", "perturbation"),
                "local_transplant": ("receiver", "donor", "role", "component", "eligible_cells", "targets", "perturbation"),
                "update_pulse": ("checkpoint", "role", "kind", "pulse_transition", "selected_cells"),
            }[str(row["group"])]
            metadata = {field: extra[field] for field in metadata_fields if field in extra}
            effects.append({
                "case": row["case"],
                "bank": bank,
                "group": row["group"],
                "checkpoint": checkpoint if checkpoint in (195, 200) else None,
                "status": _case_status(row),
                "metadata": _json_safe(metadata),
                "baseline_case": baseline_row.get("case") if baseline_row else None,
                "global_metric_effects": _global_effect(baseline_summary, summary),
                "cohort_effects": cohort_effects,
                "cohort_comparison_note": (
                    "Cohort masks are held fixed from the saved intervention plan; baseline outcomes are replayed "
                    "on those exact masks. all_steps_correct/any_wrong_including_start use every selected cohort "
                    "cell in the denominator, while first_exit conditions on cells correct at step 64."
                ),
                "baseline_cohort_replay_available": baseline_trace is not None and domain is not None,
            })
    return {
        "case_count": len(effects),
        "groups": {group: sum(row["group"] == group for row in effects) for group in sorted(groups)},
        "effects": effects,
        "interpretation": (
            "Finite-horizon, fixed-mask counterfactual effects conditional on the selected checkpoint and saved map bank. "
            "State swaps/transplants can be off-trajectory; an empty cohort remains in the report with null rates."
        ),
    }


def _cross_cube(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cube_rows = []
    seen = set()
    duplicate = []
    for row in rows:
        extra = row.get("extra", {})
        try:
            key = (int(extra["producer"]), int(extra["continuation"]), int(extra["readout"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key in seen:
            duplicate.append(list(key))
        seen.add(key)
        cohorts = row.get("summary", {}).get("cohorts", {}) if isinstance(row.get("summary"), dict) else {}
        cube_rows.append({
            "producer": key[0],
            "continuation": key[1],
            "readout": key[2],
            "case": row.get("case"),
            "status": _case_status(row),
            "metrics": _metric_record(row.get("summary")),
            "common_solved64": _cohort_view(cohorts.get("common_solved64")),
        })
    expected = {(p, d, r) for p in (195, 200) for d in (195, 200) for r in (195, 200)}
    return {
        "expected_combinations": 8,
        "observed_unique_combinations": len(seen),
        "complete": seen == expected and len(rows) == 8 and not duplicate
                    and all(item["status"] == "RECORDED" for item in cube_rows),
        "missing_combinations": [list(key) for key in sorted(expected - seen)],
        "duplicate_combinations": duplicate,
        "cases": sorted(cube_rows, key=lambda row: (row["producer"], row["continuation"], row["readout"])),
    }


def _confirmatory_agreement(banks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparisons = []
    for size in (32, 64):
        primary = banks.get(f"primary{size}")
        confirmation = banks.get(f"confirmation{size}")
        if primary is None or confirmation is None:
            comparisons.append({"size": size, "complete": False, "reason": "primary or confirmation bank missing"})
            continue
        rows = []
        base195 = primary.get("baselines", {}).get("195", {}).get("metrics", {})
        base200 = primary.get("baselines", {}).get("200", {}).get("metrics", {})
        conf195 = confirmation.get("baselines", {}).get("195", {}).get("metrics", {})
        conf200 = confirmation.get("baselines", {}).get("200", {}).get("metrics", {})
        for metric in METRIC_SPECS:
            changes = []
            for pair in ((base195, base200), (conf195, conf200)):
                left, right = (_metric_value(pair[0], metric), _metric_value(pair[1], metric))
                changes.append(right - left if left is not None and right is not None else None)
            p, c = changes
            rows.append({
                "contrast": "u200_minus_u195",
                "metric": metric,
                "primary_delta": p,
                "confirmation_delta": c,
                "direction_agrees": bool((p > 0 and c > 0) or (p < 0 and c < 0) or (p == 0 and c == 0))
                    if p is not None and c is not None else None,
            })
        for metric in METRIC_SPECS:
            psh = primary.get("factorial", {}).get("shapley", {}).get(metric, {})
            csh = confirmation.get("factorial", {}).get("shapley", {}).get(metric, {})
            for block in BLOCKS:
                pv = psh.get("by_block", {}).get(block, {}).get("mean") if psh.get("complete") else None
                cv = csh.get("by_block", {}).get(block, {}).get("mean") if csh.get("complete") else None
                p, c = _finite(pv), _finite(cv)
                rows.append({
                    "contrast": "factorial_block_shapley_sign",
                    "metric": metric,
                    "block": block,
                    "primary_delta": p,
                    "confirmation_delta": c,
                    "direction_agrees": bool((p > 0 and c > 0) or (p < 0 and c < 0) or (p == 0 and c == 0))
                        if p is not None and c is not None else None,
                })
        comparisons.append({
            "size": size,
            "complete": True,
            "agreement_rows": rows,
            "interpretation": "Agreement across diagnostic map banks only; this is not training-trajectory replication or treatment-threshold qualification.",
        })
    return {
        "by_size": comparisons,
        "scope": "Independent map banks provide map-set confirmation for one selected training trajectory; no training seeds are replicated.",
    }


def _format_value(value: Any, digits: int = 3) -> str:
    x = _finite(value)
    return "NA" if x is None else f"{x:.{digits}f}"


def _write_results(out: Path, analysis: dict[str, Any]) -> None:
    lines = [
        "# Joint 195/200 Mechanism Audit",
        "",
        f"Execution status: **{analysis.get('execution_status', 'UNKNOWN')}**; "
        f"saved case records: **{analysis.get('observed_case_count', 0)} / {EXPECTED_TOTAL_CASES} expected**.",
        "",
        "This is a finite-horizon, zero-training intervention audit conditional on one selected training trajectory. "
        "The independent confirmation banks add map-set checks, not training-seed replication. Maps are the reporting units; "
        "pooled cells are descriptive.",
        "",
        "## Saved endpoint metrics",
        "",
        "Equal-map means are shown. See `analysis.json` for pooled counts, denominators, per-map values, all cases and uncertainty boundaries.",
        "",
        "| Bank | Update | Strict coverage at 128 | Acquisition 64–128 | First exit 64–128 | Survival 64–256 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bank_name in sorted(analysis.get("banks", {})):
        bank = analysis["banks"][bank_name]
        for update in (195, 200):
            record = bank.get("baselines", {}).get(str(update))
            metrics = record.get("metrics", {}) if record else {}
            values = [_metric_value(metrics, key) for key in METRIC_SPECS]
            lines.append(f"| {bank_name} | {update} | " + " | ".join(_format_value(v) for v in values) + " |")

    lines.extend([
        "",
        "## Factorial block-order effects",
        "",
        "Each block effect averages its marginal change over all 24 replacement orders. A metric is omitted from Shapley summaries unless all 16 unique coalitions are finite.",
        "",
        "| Bank | Metric | E mean [min,max] | F mean [min,max] | Q mean [min,max] | R mean [min,max] |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for bank_name in sorted(analysis.get("banks", {})):
        factorial = analysis["banks"][bank_name].get("factorial", {})
        for metric in METRIC_SPECS:
            shapley = factorial.get("shapley", {}).get(metric, {})
            if not shapley.get("complete"):
                missing = ", ".join(shapley.get("missing_or_nonfinite_coalitions", [])) or "duplicates/incomplete"
                lines.append(f"| {bank_name} | {metric} | incomplete: {missing} | — | — | — |")
                continue
            cells = []
            for block in BLOCKS:
                entry = shapley["by_block"][block]
                cells.append(f"{entry['mean']:.3f} [{entry['minimum']:.3f}, {entry['maximum']:.3f}]")
            lines.append(f"| {bank_name} | {metric} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Reading and claim boundary",
        "",
        "The interpolation curves are finite-horizon comparisons in the historical shared parameter coordinates; they do not establish a bifurcation or a treatment threshold. Block Shapley effects are conditional on this checkpoint pair and selected maps. Cross-continuation preserves all producer × continuation × readout combinations. State swaps and local transplants may be off the learned trajectory; their effects support only conditional sensitivity statements.",
        "",
        "For fixed solved/reference cohorts, `all_steps_correct` and `any_wrong_including_start` use all selected cells as the denominator, so an immediate break is retained. `first_exit` instead conditions on correctness at step 64. Localized target/downstream comparisons replay the baseline on each intervention's exact saved mask; empty cohorts remain present with null rates.",
        "",
        "The confirmation banks test agreement across task maps only, not an independent training trajectory or treatment-threshold qualification. A correct output under a readout does not establish latent completion. No phase transition, universal mechanism, or general architectural claim is established.",
        "",
        "`analysis.png` is a descriptive view; canonical values and all 24-order marginals are in `analysis.json`. Saved per-case JSON remains the source for full per-map denominators and intervention-specific records.",
        "",
    ])
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _make_figure(out: Path, analysis: dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate analysis.png") from exc

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for bank_name in sorted(analysis.get("banks", {})):
        points = analysis["banks"][bank_name].get("interpolation", {}).get("points", [])
        xs, ys = [], []
        for point in points:
            value = _metric_value(point.get("metrics", {}), "strict128_equal_map")
            if value is not None:
                xs.append(point["lambda"])
                ys.append(value)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.3, markersize=3, label=bank_name)
    ax.set(title="Strict changed-region coverage at step 128", xlabel="Interpolation λ (endpoints are saved checkpoints)", ylabel="Equal-map mean")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    bank = analysis.get("banks", {}).get("primary64", {})
    shapley = bank.get("factorial", {}).get("shapley", {})
    metrics = list(METRIC_SPECS)
    x = np.arange(len(BLOCKS), dtype=float)
    width = 0.18
    any_bar = False
    for j, metric in enumerate(metrics):
        entry = shapley.get(metric, {})
        if entry.get("complete"):
            vals = [entry["by_block"][block]["mean"] for block in BLOCKS]
            ax.bar(x + (j - (len(metrics) - 1) / 2) * width, vals, width, label=metric.replace("_equal_map", ""))
            any_bar = True
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x, list(BLOCKS))
    ax.set(title="Primary size-64 factorial Shapley means", xlabel="Block replaced from u200", ylabel="Mean marginal effect")
    ax.grid(axis="y", alpha=0.25)
    if any_bar:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No complete 16-coalition metric", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 0]
    cross = bank.get("cross_continuation", {}).get("cases", [])
    matrix = np.full((4, 2), np.nan, dtype=float)
    labels = []
    for i, (producer, continuation) in enumerate(itertools.product((195, 200), repeat=2)):
        labels.append(f"P{producer}/D{continuation}")
        for j, readout in enumerate((195, 200)):
            row = next((r for r in cross if r["producer"] == producer and r["continuation"] == continuation and r["readout"] == readout), None)
            if row:
                matrix[i, j] = _metric_value(row.get("metrics", {}), "strict128_equal_map")
    image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks((0, 1), ("R195", "R200"))
    ax.set_yticks(np.arange(4), labels)
    ax.set_title("Primary size-64 cross-continuation: strict coverage")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, "NA" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center", color="white" if np.isfinite(value) and value < 0.55 else "black", fontsize=8)

    ax = axes[1, 1]
    groups = ("state_swap", "local_transplant", "update_pulse")
    positions = {name: index for index, name in enumerate(groups)}
    grouped_values: dict[str, list[float]] = {name: [] for name in groups}
    for row in bank.get("state_pulse_effects", {}).get("effects", []):
        metric_effect = row.get("global_metric_effects", {}).get("strict128_equal_map", {})
        delta = _finite(metric_effect.get("delta_equal_map_mean"))
        group = row.get("group")
        if delta is not None and group in grouped_values:
            grouped_values[group].append(delta)
    for group in groups:
        values = grouped_values[group]
        if values:
            jitter = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(positions[group] + jitter, values, s=12, alpha=0.65, label=f"{group} (n={len(values)})")
        else:
            ax.text(positions[group], 0, "no finite rows", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(range(len(groups)), ["swap", "transplant", "pulse"])
    ax.set(title="Primary size-64 global strict-coverage deltas", ylabel="Intervention − same-checkpoint baseline")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("195/200 saved-case audit — descriptive, one training trajectory", fontsize=12)
    fig.savefig(out / "analysis.png", dpi=170)
    plt.close(fig)


def analyze(out: Path) -> dict[str, Any]:
    """Write ``analysis.json``, ``RESULTS.md`` and ``analysis.png`` from cases.

    ``out`` is the completed audit run directory containing ``cases/``,
    ``bank_*.npz``, baseline trace NPZ files and ``status.json``. No inference
    or GPU operation is performed.
    """
    out = Path(out).resolve()
    check_additive_fixture()
    check_baseline_suffix_fixture()
    rows, load_errors = _load_cases(out)
    bank_inventory = _bank_inventory(rows)
    baselines, baseline_errors = _find_baselines(rows)
    try:
        status = _read_json(out / "status.json").get("status", "UNKNOWN")
    except (FileNotFoundError, ValueError):
        status = "UNKNOWN"
    try:
        canonical_summary = _read_json(out / "summary.json")
    except (FileNotFoundError, ValueError):
        canonical_summary = {}
    data_complete = (
        canonical_summary.get("execution_complete") is True
        and len(rows) == EXPECTED_TOTAL_CASES
    )
    if status != "COMPLETE" and data_complete:
        # run.py builds canonical summary.json immediately before setting the
        # final status file; describe the data state without rewriting status.
        status = "DATA_COMPLETE"

    banks: dict[str, Any] = {}
    for bank_name, inventory in sorted(bank_inventory.items()):
        bank_rows = inventory["rows"]
        baseline_rows = {}
        for update in (195, 200):
            row = baselines.get((bank_name, update))
            baseline_rows[str(update)] = {
                "case": row.get("case") if row else None,
                "status": _case_status(row) if row else "MISSING_BASELINE",
                "metrics": _metric_record(row.get("summary") if row else None),
            }
        factorial = _factorial_summary([row for row in bank_rows if row.get("group") == "factorial"])
        interpolation = _interpolation_summary(
            [row for row in bank_rows if row.get("group") == "interpolation"], baselines, bank_name
        )
        cross = _cross_cube([row for row in bank_rows if row.get("group") == "cross_continuation"])
        state_effects = _state_pulse_effects(out, bank_rows, baselines)
        banks[bank_name] = {
            "size": inventory["size"],
            "confirmation": inventory["confirmation"],
            "case_counts_by_group": dict(sorted(inventory["groups"].items())),
            "baselines": baseline_rows,
            "interpolation": interpolation,
            "factorial": factorial,
            "cross_continuation": cross,
            "state_pulse_effects": state_effects,
        }

    expected_by_bank = {
        bank_name: (CONFIRMATION_CASES_PER_BANK if info["confirmation"] else PRIMARY_CASES_PER_BANK)
        for bank_name, info in bank_inventory.items()
    }
    count_checks = {
        bank_name: {
            "observed": len(info["rows"]),
            "expected": expected_by_bank[bank_name],
            "pass": len(info["rows"]) == expected_by_bank[bank_name],
        }
        for bank_name, info in bank_inventory.items()
    }
    analysis = {
        "schema_version": "audit-195-200-analysis-v1",
        "execution_status": status,
        "observed_case_count": len(rows),
        "expected_case_count": EXPECTED_TOTAL_CASES,
        "case_count_check": {
            "by_bank": count_checks,
            "total_pass": len(rows) == EXPECTED_TOTAL_CASES
                and len(bank_inventory) == 4
                and all(item["pass"] for item in count_checks.values()),
            "inventory_errors": load_errors + baseline_errors,
        },
        "canonical_summary_execution_complete": canonical_summary.get("execution_complete") is True,
        "reporting_unit": "map; pooled cells and repeated transitions are descriptive within-map quantities",
        "training_trajectory_count": 1,
        "new_training": False,
        "metric_definitions": {
            metric: {"case_summary_path": list(path), "reported_scalar": "equal_map_mean"}
            for metric, path in METRIC_SPECS.items()
        },
        "banks": banks,
        "confirmation_agreement": _confirmatory_agreement(banks),
        "claim_boundary": [
            "Finite-horizon sensitivity in this u195/u200 checkpoint pair and saved map banks only.",
            "Independent confirmation banks replicate maps, not the selected training trajectory.",
            "No phase transition, treatment threshold, latent completion, or universal mechanism is established.",
            "Interpolation is a finite-horizon curve in historical shared parameter coordinates.",
            "State swaps and transplants may be off trajectory; empty cohorts remain recorded with null outcomes.",
        ],
    }
    analysis = _json_safe(analysis)
    with (out / "analysis.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(analysis, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    _write_results(out, analysis)
    _make_figure(out, analysis)
    return analysis
