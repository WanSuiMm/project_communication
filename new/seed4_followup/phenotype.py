"""Compact fresh-map phenotype evaluator for independent seed4 follow-ups.

The model trace and matched-frontier calculations reuse the frozen frontier
audit implementation. Saved traces contain only paired/original/flipped
Boolean correctness; margins and confidence-threshold summaries are omitted.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRACE_STEPS = 256
ENDPOINTS = (64, 128, 256)
TRANSITIONS = ((64, 128), (64, 256), (128, 256))
_TRACE_LOCK = threading.RLock()
_AUDIT = None


def _load_audit():
    """Load the frozen audit from its source path without editing it."""
    global _AUDIT
    with _TRACE_LOCK:
        if _AUDIT is not None:
            return _AUDIT
        relative_paths = (
            "new/nca_inertial_wind_tunnel",
            "new/workspace_revision",
            "new/short_bptt_phase2",
            "new/streaming_carry",
            "new/frontier_audit",
        )
        for relative in reversed(relative_paths):
            value = str(ROOT / relative)
            if value not in sys.path:
                sys.path.insert(0, value)

        # audit.py uses a top-level `from metrics import ...`; temporarily make
        # that name resolve to the frozen sibling module even if another tool
        # has already imported an unrelated module called `metrics`.
        metrics_path = ROOT / "new/frontier_audit/metrics.py"
        previous_metrics = sys.modules.pop("metrics", None)
        audit_path = ROOT / "new/frontier_audit/audit.py"
        spec = importlib.util.spec_from_file_location(
            "_reactiontransport_seed4_frontier_audit", audit_path
        )
        if spec is None or spec.loader is None:
            if previous_metrics is not None:
                sys.modules["metrics"] = previous_metrics
            raise ImportError(f"Cannot load frozen audit from {audit_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            loaded_metrics = sys.modules.get("metrics")
            loaded_path = Path(getattr(loaded_metrics, "__file__", "")).resolve()
            if loaded_path != metrics_path.resolve():
                raise ImportError("Frozen frontier metrics module did not bind")
        except Exception:
            sys.modules.pop(spec.name, None)
            raise
        finally:
            if previous_metrics is not None:
                sys.modules["metrics"] = previous_metrics
            else:
                sys.modules.pop("metrics", None)
        _AUDIT = module
        return module


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _size_value(mapping: dict, size: int):
    if str(size) in mapping:
        return mapping[str(size)]
    return mapping[size]


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    numerator, denominator = int(numerator), int(denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": float(numerator / denominator) if denominator else None,
    }


def _timing(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values).reshape(-1)
    observed = values[values >= 0]
    return {
        "pixels": int(values.size),
        "observed": int(observed.size),
        "censored": int(values.size - observed.size),
        "sum_observed_steps": int(observed.sum()) if observed.size else 0,
        "median_observed_step": float(np.median(observed)) if observed.size else None,
        "min_observed_step": int(observed.min()) if observed.size else None,
        "max_observed_step": int(observed.max()) if observed.size else None,
    }


def _first_and_stable(correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return first passage, terminal-stable start, and any later relapse."""
    ever = np.any(correct, axis=0)
    first = np.argmax(correct, axis=0).astype(np.int16, copy=False)
    first[~ever] = -1
    wrong = ~correct
    has_wrong = np.any(wrong, axis=0)
    last_wrong = TRACE_STEPS - np.argmax(wrong[::-1], axis=0)
    stable = np.full(correct.shape[1:], -1, dtype=np.int16)
    terminal = correct[-1]
    stable[terminal & has_wrong] = (last_wrong[terminal & has_wrong] + 1).astype(np.int16)
    stable[terminal & ~has_wrong] = 0
    relapse = np.any(correct[:-1] & ~correct[1:], axis=0)
    return first, stable, relapse


def _coverage(correct_map: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    map_pixels = selected.sum(axis=(1, 2), dtype=np.int64)
    map_correct = (correct_map & selected).sum(axis=(1, 2), dtype=np.int64)
    eligible = map_pixels > 0
    rates = map_correct[eligible] / map_pixels[eligible]
    return {
        "correct_pixels": int(map_correct.sum()),
        "pixels": int(map_pixels.sum()),
        "pooled_coverage": _ratio(int(map_correct.sum()), int(map_pixels.sum())),
        "eligible_maps": int(eligible.sum()),
        "mean_map_coverage": float(rates.mean()) if rates.size else None,
    }


def _transition(correct: np.ndarray, selected: np.ndarray, start: int, end: int) -> dict[str, Any]:
    retained_total = from_correct_total = lost_total = gained_total = 0
    map_retention = []
    for index in range(selected.shape[0]):
        mask = selected[index]
        before, after = correct[start, index] & mask, correct[end, index] & mask
        from_correct = int(before.sum())
        retained = int((before & after).sum())
        lost = from_correct - retained
        gained = int((~correct[start, index] & correct[end, index] & mask).sum())
        retained_total += retained
        from_correct_total += from_correct
        lost_total += lost
        gained_total += gained
        if from_correct:
            map_retention.append(retained / from_correct)
    return {
        "from_step": int(start),
        "to_step": int(end),
        "pooled": {
            "from_correct": int(from_correct_total),
            "retained": int(retained_total),
            "lost": int(lost_total),
            "gained": int(gained_total),
            "retention": _ratio(retained_total, from_correct_total),
            "relapse": _ratio(lost_total, from_correct_total),
        },
        "mean_map_retention": float(np.mean(map_retention)) if map_retention else None,
        "eligible_maps_for_retention": int(len(map_retention)),
    }


def _frontier_summary(correct: np.ndarray, data: dict[str, np.ndarray], size: int):
    audit = _load_audit()
    # The frozen analyzer also emits confidence-threshold fields. A fixed zero
    # margin is supplied only to reach its independent Boolean/frontier path;
    # none of those margin-derived outputs are copied into our summary.
    behavior = audit.analyze_trace(
        correct,
        np.zeros(correct.shape, dtype=np.float32),
        data,
    )
    csv_rows = []
    by_map = []
    all_common = 0
    for map_row in behavior["maps"]:
        map_index = int(map_row["map_index"])
        terms: list[tuple[float, float]] = []
        intervals = set()
        for interval in map_row["frontier"]:
            start = int(interval["from_step"])
            for stratum in interval["exact_distance"]:
                weight = stratum["matched_weight"]
                difference = stratum["matched_rate_difference"]
                if weight is None or difference is None:
                    continue
                weight, difference = float(weight), float(difference)
                terms.append((weight, difference))
                intervals.add(start)
                all_common += 1
                csv_rows.append({
                    "size": int(size),
                    "map_index": map_index,
                    "start_step": start,
                    "end_step": int(interval["to_step"]),
                    "distance": int(stratum["distance"]),
                    "frontier_opportunities": int(stratum["frontier_opportunities"]),
                    "frontier_acquired": int(stratum["frontier_acquired"]),
                    "nonfrontier_opportunities": int(stratum["nonfrontier_opportunities"]),
                    "nonfrontier_acquired": int(stratum["nonfrontier_acquired"]),
                    "matched_weight": weight,
                    "frontier_rate": float(stratum["frontier_acquired"] / stratum["frontier_opportunities"]),
                    "nonfrontier_rate": float(stratum["nonfrontier_acquired"] / stratum["nonfrontier_opportunities"]),
                    "rate_difference": difference,
                })
        weight_sum = float(sum(weight for weight, _ in terms))
        effect = (float(sum(weight * value for weight, value in terms) / weight_sum)
                  if weight_sum else None)
        by_map.append({
            "map_index": map_index,
            "changed_pixels": int(map_row["changed_pixels"]),
            "common_strata": int(len(terms)),
            "eligible_intervals": int(len(intervals)),
            "weight_sum": weight_sum if weight_sum else None,
            "weighted_difference": effect,
        })
    eligible = [row["weighted_difference"] for row in by_map
                if row["weighted_difference"] is not None]
    return {
        "acquisition_definition": "t to t+1 at starts 0,8,...,248",
        "within_map_weighting": "nf*nn/(nf+nn) over common exact-distance strata pooled across starts",
        "between_map_weighting": "equal mean over maps with at least one common stratum",
        "eligible_maps": int(len(eligible)),
        "common_strata": int(all_common),
        "mean_map_weighted_difference": float(np.mean(eligible)) if eligible else None,
        "per_map": by_map,
    }, csv_rows


def summarize_from_traces(
    traces: dict,
    records: dict,
    data_cpu: dict,
) -> dict[str, Any]:
    """Build a compact CPU summary from boolean traces, endpoint records, and banks.

    ``traces`` is keyed by integer or string sizes. Each trace has only
    ``correct``, ``original_correct`` and ``flipped_correct`` arrays shaped
    [257, maps, height, width]. ``records`` holds the frozen audit endpoint
    evaluations at 64/128/256.
    """
    result: dict[str, Any] = {
        "schema": "seed4-independent-phenotype-v1",
        "training": False,
        "steps": {"first_observed": 0, "last_observed": TRACE_STEPS},
        "sizes": {},
        "claim_boundary": (
            "A selected checkpoint's descriptive output phenotype on the fresh fixed maps; "
            "not an architecture GO, reliability result, causal handoff law, or formal basin claim."
        ),
        "confidence_threshold_quantities": "omitted; no confidence margins are retained",
    }
    csv_rows = []
    for size in (32, 64):
        values = _size_value(traces, size)
        bank = _size_value(data_cpu, size)
        records_for_size = _size_value(records, size)
        correct = _array(values["correct"]).astype(bool, copy=False)
        original_correct = _array(values["original_correct"]).astype(bool, copy=False)
        flipped_correct = _array(values["flipped_correct"]).astype(bool, copy=False)
        if correct.ndim != 4 or correct.shape[0] != TRACE_STEPS + 1:
            raise ValueError(f"size {size}: correct must be [257,maps,height,width]")
        if original_correct.shape != correct.shape or flipped_correct.shape != correct.shape:
            raise ValueError(f"size {size}: all three correctness traces must have identical shapes")
        maps, height, width = correct.shape[1:]
        changed4 = _array(bank["changed"])
        open4 = _array(bank["mask"])
        distance4 = _array(bank["distance"])
        expected = (maps, 1, height, width)
        for label, value in (("changed", changed4), ("mask", open4), ("distance", distance4)):
            if value.shape != expected:
                raise ValueError(f"size {size}: data[{label!r}] must have shape {expected}")
        changed = changed4[:, 0].astype(bool, copy=False)
        opened = open4[:, 0].astype(bool, copy=False)
        distance = distance4[:, 0].astype(np.int64, copy=False)
        if np.any(changed & ~opened) or np.any(changed & (distance < 0)):
            raise ValueError(f"size {size}: changed component must be open with nonnegative BFS distances")
        strict = changed & (distance > 16) & (distance < 32)
        first, stable, relapse = _first_and_stable(correct)

        per_map = []
        for map_index in range(maps):
            selected = changed[map_index]
            ever = selected & (first[map_index] >= 0)
            relapsed = selected & relapse[map_index]
            per_map.append({
                "map_index": int(map_index),
                "changed_pixels": int(selected.sum()),
                "first_correct": _timing(np.where(selected, first[map_index], -1)[selected]),
                "terminal_stable_through256": _timing(np.where(selected, stable[map_index], -1)[selected]),
                "ever_correct_pixels": int(ever.sum()),
                "never_correct_pixels": int(selected.sum() - ever.sum()),
                "ever_regressed": _ratio(int(relapsed.sum()), int(ever.sum())),
            })

        exact_profile = []
        for d in sorted(int(v) for v in np.unique(distance[changed])):
            selected = changed & (distance == d)
            ever = selected & (first >= 0)
            relapsed = selected & relapse
            exact_profile.append({
                "distance": int(d),
                "pixels": int(selected.sum()),
                "first_correct": _timing(first[selected]),
                "terminal_stable_through256": _timing(stable[selected]),
                "ever_correct_pixels": int(ever.sum()),
                "never_correct_pixels": int(selected.sum() - ever.sum()),
                "ever_regressed": _ratio(int(relapsed.sum()), int(ever.sum())),
            })

        size_row: dict[str, Any] = {
            "maps": int(maps),
            "changed_pixels": int(changed.sum()),
            "per_map": per_map,
            "distance_profile": exact_profile,
            "endpoints": {},
            "transitions": {"all_changed": {}, "strict_16_32": {}},
            "acquisition": {},
        }
        for t in ENDPOINTS:
            key = str(t)
            rec = records_for_size[key] if key in records_for_size else records_for_size[t]
            open_row = {
                "original_ba": float(rec["original"]["balanced_accuracy"]),
                "flipped_ba": float(rec["flipped"]["balanced_accuracy"]),
            }
            size_row["endpoints"][key] = {
                **open_row,
                "all_changed": _coverage(correct[t], changed),
                "strict_16_32": _coverage(correct[t], strict),
            }
            changed_at_t = changed & correct[t]
            new_vs_start = changed & ~correct[0] & correct[t]
            new_since_64 = changed & ~correct[64] & correct[t]
            lost_since_64 = changed & correct[64] & ~correct[t]
            size_row["acquisition"][key] = {
                "newly_correct_vs_step0_pixels": int(new_vs_start.sum()),
                "newly_correct_since64_pixels": int(new_since_64.sum()),
                "lost_since64_pixels": int(lost_since_64.sum()),
                "first_passage_after_step0_by_t_pixels": int((changed & (first > 0) & (first <= t)).sum()),
                "all_changed_coverage": size_row["endpoints"][key]["all_changed"],
            }

        for start, end in TRANSITIONS:
            pair_key = f"{start}_to_{end}"
            size_row["transitions"]["all_changed"][pair_key] = _transition(correct, changed, start, end)
            size_row["transitions"]["strict_16_32"][pair_key] = _transition(correct, strict, start, end)

        ever_correct = changed & (first >= 0)
        ever_relapsed = changed & relapse
        size_row["ever_regressed_over_ever_correct"] = _ratio(
            int(ever_relapsed.sum()), int(ever_correct.sum())
        )
        frontier, rows = _frontier_summary(
            correct,
            {"changed": changed4, "mask": open4, "distance": distance4},
            size,
        )
        size_row["frontier"] = frontier
        csv_rows.extend(rows)
        result["sizes"][str(size)] = size_row

    # Matched rows are kept separately so independent recurrence can recompute
    # each map/time/exact-distance weighted difference from integer counts.
    result["_matched_frontier_rows"] = csv_rows
    return result


def predicate(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen exploratory phenotype gates and return reasons."""
    sizes = summary["sizes"]
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, value: Any, threshold: str, passed: bool):
        checks[name] = {"value": value, "threshold": threshold, "pass": bool(passed)}

    s32 = sizes["32"]
    reach = s32["endpoints"]["64"]
    strict_reach = reach["strict_16_32"]
    add("reach_size32_T64_strict_map_mean", strict_reach["mean_map_coverage"], ">=0.80",
        strict_reach["mean_map_coverage"] is not None and strict_reach["mean_map_coverage"] >= 0.80)
    strict_pooled = strict_reach["pooled_coverage"]["value"]
    add("reach_size32_T64_strict_pooled", strict_pooled, ">=0.80",
        strict_pooled is not None and strict_pooled >= 0.80)
    for branch in ("original_ba", "flipped_ba"):
        value = reach[branch]
        add(f"reach_size32_T64_{branch}", value, ">=0.85", value >= 0.85)

    for t in (128, 256):
        endpoint = s32["endpoints"][str(t)]
        for branch in ("original_ba", "flipped_ba"):
            drop = reach[branch] - endpoint[branch]
            add(f"hold_size32_T{t}_{branch}_drop", drop, "<=0.03", drop <= 0.03)
        for kind in ("mean_map_coverage", "pooled_coverage"):
            baseline = (reach["strict_16_32"][kind] if kind == "mean_map_coverage"
                        else reach["strict_16_32"][kind]["value"])
            current = (endpoint["strict_16_32"][kind] if kind == "mean_map_coverage"
                       else endpoint["strict_16_32"][kind]["value"])
            drop = None if baseline is None or current is None else baseline - current
            add(f"hold_size32_T{t}_strict_{kind}_drop", drop, "<=0.05",
                drop is not None and drop <= 0.05)

    for size in (32, 64):
        row = sizes[str(size)]
        retention = row["transitions"]["all_changed"]["64_to_256"]
        pooled_retention = retention["pooled"]["retention"]["value"]
        add(f"size{size}_all_changed_retention64_to256", pooled_retention, ">=0.95",
            pooled_retention is not None and pooled_retention >= 0.95)
        coverage64 = row["endpoints"]["64"]["all_changed"]["pooled_coverage"]["value"]
        coverage256 = row["endpoints"]["256"]["all_changed"]["pooled_coverage"]["value"]
        gain = None if coverage64 is None or coverage256 is None else coverage256 - coverage64
        add(f"size{size}_all_changed_coverage_gain64_to256", gain, ">=0.05",
            gain is not None and gain >= 0.05)
        relapse = row["ever_regressed_over_ever_correct"]["value"]
        add(f"size{size}_ever_regressed_over_ever_correct", relapse, "<=0.15",
            relapse is not None and relapse <= 0.15)
        frontier = row["frontier"]
        effect = frontier["mean_map_weighted_difference"]
        add(f"size{size}_matched_frontier_effect", effect, ">=0.05",
            effect is not None and effect >= 0.05)
        add(f"size{size}_eligible_frontier_maps", frontier["eligible_maps"], ">=16",
            frontier["eligible_maps"] >= 16)
        add(f"size{size}_common_frontier_strata", frontier["common_strata"], ">=100",
            frontier["common_strata"] >= 100)

    failures = [name for name, row in checks.items() if not row["pass"]]
    return {
        "pass": not failures,
        "reasons": [f"{name}: observed {checks[name]['value']!r}, requires {checks[name]['threshold']}"
                    for name in failures],
        "checks": checks,
        "interpretation": "Descriptive selected-checkpoint phenotype only; not an architecture GO or formal basin result.",
    }


def _json_ready(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items() if key != "_matched_frontier_rows"}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_frontier_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "size", "map_index", "start_step", "end_step", "distance",
        "frontier_opportunities", "frontier_acquired", "nonfrontier_opportunities",
        "nonfrontier_acquired", "matched_weight", "frontier_rate",
        "nonfrontier_rate", "rate_difference",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    model,
    data_cpu: dict[int, dict[str, Any]],
    out: Path,
    name: str,
    budget: Callable[[], Any],
) -> dict[str, Any]:
    """Evaluate one selected model on the frozen 32-map size32/64 fresh banks."""
    if not callable(budget):
        raise TypeError("budget must be callable")
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("name must be a nonempty filename-safe component")
    for size in (32, 64):
        bank = _size_value(data_cpu, size)
        if _array(bank["x"]).shape[0] != 32:
            raise ValueError(f"size {size} must use exactly 32 fresh evaluation maps")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    audit = _load_audit()
    model.eval()
    model.cuda()

    traces: dict[str, dict[str, np.ndarray]] = {}
    records: dict[str, dict[str, Any]] = {}
    with _TRACE_LOCK:
        previous_budget = audit.budget
        audit.budget = budget
        try:
            for size in (32, 64):
                budget()
                data_gpu = {key: value.cuda() for key, value in _size_value(data_cpu, size).items()}
                full_trace, endpoint_records = audit.trace(model, data_gpu, size)
                values = {
                    key: np.asarray(full_trace[key], dtype=bool)
                    for key in ("correct", "original_correct", "flipped_correct")
                }
                # Do not retain or serialize the unneeded full margin tensor.
                del full_trace
                traces[str(size)] = values
                records[str(size)] = endpoint_records
                np.savez_compressed(out / f"{name}_size{size}.npz", **values)
                del data_gpu
        finally:
            audit.budget = previous_budget

    budget()
    summary = summarize_from_traces(traces, records, data_cpu)
    frontier_rows = summary.pop("_matched_frontier_rows")
    summary["model_name"] = name
    summary["phenotype_gate"] = predicate(summary)
    summary = _json_ready(summary)
    _write_frontier_csv(out / f"{name}_matched_frontier_strata.csv", frontier_rows)
    (out / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    budget()
    return summary

