"""Compact CPU summaries for the saved 195/200 paired audit traces.

The reporting unit is a map. Cell and cell-time observations are descriptive
within-map measurements and are never treated as independent replicates.
"""
from __future__ import annotations

from typing import Any

import numpy as np


TRACE_STEPS = 193
FIRST_STEP = 64
LAST_STEP = 256
ENDPOINTS = (64, 128, 256)
INTERVALS = ((64, 128), (64, 256))
QUANTILE_LEVELS = (0.0, 0.1, 0.5, 0.9, 1.0)
MARGIN_NAMES = ("margin", "original_margin", "flipped_margin")


def _binary_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_:
        return array
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise ValueError(f"{name} must be boolean or contain only 0/1")
    if not np.all(np.isfinite(array)) or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be boolean or contain only 0/1")
    return array.astype(bool, copy=False)


def _map_array(name: str, value: Any, shape: tuple[int, int, int]) -> np.ndarray:
    array = _binary_array(name, value)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.shape != shape:
        raise ValueError(f"{name} must have shape [B,H,W] or [B,1,H,W]")
    return array


def _distance_array(value: Any, shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.shape != shape:
        raise ValueError("distance must have shape [B,H,W] or [B,1,H,W]")
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise ValueError("distance must contain finite integer BFS distances")
    if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
        raise ValueError("distance must contain finite integer BFS distances")
    return array.astype(np.int64, copy=False)


def _margin_array(name: str, value: Any, shape: tuple[int, int, int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape [time,B,H,W] matching correct")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _rate_summary(numerators: np.ndarray, denominators: np.ndarray) -> dict[str, Any]:
    """Return pooled counts/rate and descriptive per-map/equal-map rates."""
    nums = np.asarray(numerators, dtype=np.int64).reshape(-1)
    dens = np.asarray(denominators, dtype=np.int64).reshape(-1)
    if nums.shape != dens.shape:
        raise ValueError("numerators and denominators must have the same map count")
    if np.any(dens < 0):
        raise ValueError("denominators must be nonnegative")
    per_map = []
    eligible_rates = []
    for index, (numerator, denominator) in enumerate(zip(nums, dens)):
        numerator, denominator = int(numerator), int(denominator)
        rate = float(numerator / denominator) if denominator else None
        per_map.append({
            "map_index": int(index),
            "numerator": numerator,
            "denominator": denominator,
            "rate": rate,
        })
        if rate is not None:
            eligible_rates.append(rate)
    pooled_num = int(nums.sum())
    pooled_den = int(dens.sum())
    return {
        "pooled": {
            "numerator": pooled_num,
            "denominator": pooled_den,
            "rate": float(pooled_num / pooled_den) if pooled_den else None,
        },
        "per_map": per_map,
        "equal_map_mean": float(np.mean(eligible_rates)) if eligible_rates else None,
        "equal_map_eligible_maps": int(len(eligible_rates)),
    }


def _counts_by_map(mask: np.ndarray) -> np.ndarray:
    return np.count_nonzero(mask, axis=(1, 2)).astype(np.int64, copy=False)


def _endpoint(correct: np.ndarray, selected: np.ndarray, trace_index: int) -> dict[str, Any]:
    denominator = _counts_by_map(selected)
    numerator = _counts_by_map(selected & correct[trace_index])
    return _rate_summary(numerator, denominator)


def _turnover(correct: np.ndarray, selected: np.ndarray,
              start_index: int, end_index: int) -> dict[str, Any]:
    """Count every adjacent correct/wrong transition, including repeat flips."""
    left = correct[start_index:end_index]
    right = correct[start_index + 1:end_index + 1]
    correct_to_wrong = np.count_nonzero(selected[None] & left & ~right, axis=(0, 2, 3))
    wrong_to_correct = np.count_nonzero(selected[None] & ~left & right, axis=(0, 2, 3))
    pixels = _counts_by_map(selected)
    steps = end_index - start_index
    per_map = []
    for map_index in range(selected.shape[0]):
        c2w = int(correct_to_wrong[map_index])
        w2c = int(wrong_to_correct[map_index])
        per_map.append({
            "map_index": int(map_index),
            "selected_pixels": int(pixels[map_index]),
            "transition_opportunities": int(pixels[map_index] * steps),
            "correct_to_wrong": c2w,
            "wrong_to_correct": w2c,
            "total_transitions": c2w + w2c,
        })
    c2w_total = int(correct_to_wrong.sum())
    w2c_total = int(wrong_to_correct.sum())
    return {
        "per_map": per_map,
        "pooled": {
            "selected_pixels": int(pixels.sum()),
            "transition_opportunities": int(pixels.sum() * steps),
            "correct_to_wrong": c2w_total,
            "wrong_to_correct": w2c_total,
            "total_transitions": c2w_total + w2c_total,
        },
        "counting_note": "Counts include repeated flips of a cell; transition opportunities are selected pixels times adjacent-step pairs.",
    }


def _interval(correct: np.ndarray, selected: np.ndarray, start_index: int, end_index: int) -> dict[str, Any]:
    start = correct[start_index]
    end = correct[end_index]
    pixels = _counts_by_map(selected)
    correct_start = _counts_by_map(selected & start)
    wrong_start = pixels - correct_start
    acquired = _counts_by_map(selected & ~start & end)
    endpoint_destroyed = _counts_by_map(selected & start & ~end)
    exited = _counts_by_map(selected & start & np.any(~correct[start_index + 1:end_index + 1], axis=0))
    survived = _counts_by_map(selected & np.all(correct[start_index:end_index + 1], axis=0))
    all_selected_survived = _counts_by_map(selected & np.all(correct[start_index:end_index + 1], axis=0))
    any_wrong_including_start = _counts_by_map(
        selected & np.any(~correct[start_index:end_index + 1], axis=0)
    )
    net = _counts_by_map(selected & end) - correct_start

    accounting_error = net - (acquired - endpoint_destroyed)
    if np.any(accounting_error != 0):
        raise AssertionError(f"net-gain identity failed: {accounting_error.tolist()}")

    return {
        "acquisition": _rate_summary(acquired, wrong_start),
        "endpoint_destruction": _rate_summary(endpoint_destroyed, correct_start),
        "first_exit": _rate_summary(exited, correct_start),
        "continuous_survival": _rate_summary(survived, correct_start),
        "all_steps_correct": _rate_summary(all_selected_survived, pixels),
        "any_wrong_including_start": _rate_summary(any_wrong_including_start, pixels),
        "net_gain": _rate_summary(net, pixels),
        "turnover": _turnover(correct, selected, start_index, end_index),
        "identity": {
            "per_map_accounting_error": [int(value) for value in accounting_error],
            "pooled_net_change": int(net.sum()),
            "pooled_acquired_minus_endpoint_destroyed": int(acquired.sum() - endpoint_destroyed.sum()),
            "pooled_accounting_error": int(accounting_error.sum()),
        },
    }


def _quantiles_by_map(values: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    """Quantiles per map and pooled; no cell/time observations are replicates."""
    if values.ndim == 3:
        values = values[None]
    per_map = []
    map_quantiles: list[np.ndarray] = []
    pooled_values: list[np.ndarray] = []
    for map_index in range(selected.shape[0]):
        chosen = values[:, map_index][:, selected[map_index]].reshape(-1)
        if chosen.size:
            quantiles = np.quantile(chosen.astype(np.float64, copy=False), QUANTILE_LEVELS)
            map_quantiles.append(quantiles)
            pooled_values.append(chosen.astype(np.float64, copy=False))
            quantile_values: list[float] | None = [float(value) for value in quantiles]
        else:
            quantile_values = None
        per_map.append({
            "map_index": int(map_index),
            "observations": int(chosen.size),
            "quantiles": quantile_values,
        })
    if pooled_values:
        pooled = np.concatenate(pooled_values)
        pooled_quantiles: list[float] | None = [
            float(value) for value in np.quantile(pooled, QUANTILE_LEVELS)
        ]
    else:
        pooled_quantiles = None
    if map_quantiles:
        equal_map = np.mean(np.stack(map_quantiles), axis=0)
        equal_map_quantiles: list[float] | None = [float(value) for value in equal_map]
    else:
        equal_map_quantiles = None
    return {
        "quantile_levels": list(QUANTILE_LEVELS),
        "per_map": per_map,
        "equal_map_mean_quantiles": equal_map_quantiles,
        "equal_map_eligible_maps": int(len(map_quantiles)),
        "pooled": {"observations": int(sum(a.size for a in pooled_values)), "quantiles": pooled_quantiles},
        "observation_note": "Pooled cells and cell-times are descriptive; maps are the reporting units.",
    }


def _mean_change_by_map(values: np.ndarray, selected: np.ndarray,
                        start_index: int, end_index: int) -> dict[str, Any]:
    change = values[end_index] - values[start_index]
    per_map = []
    means = []
    pooled_values = []
    for map_index in range(selected.shape[0]):
        chosen = change[map_index][selected[map_index]].astype(np.float64, copy=False)
        mean = float(chosen.mean()) if chosen.size else None
        per_map.append({"map_index": int(map_index), "n": int(chosen.size), "mean": mean})
        if chosen.size:
            means.append(mean)
            pooled_values.append(chosen)
    pooled = np.concatenate(pooled_values) if pooled_values else np.empty(0, dtype=np.float64)
    return {
        "per_map": per_map,
        "equal_map_mean": float(np.mean(means)) if means else None,
        "equal_map_eligible_maps": int(len(means)),
        "pooled_descriptive": {
            "n": int(pooled.size),
            "mean": float(pooled.mean()) if pooled.size else None,
        },
    }


def _cohort_summary(name: str, selected: np.ndarray, correct: np.ndarray,
                    margins: dict[str, np.ndarray]) -> dict[str, Any]:
    endpoint_summary = {
        str(step): _endpoint(correct, selected, step - FIRST_STEP)
        for step in ENDPOINTS
    }
    first_exit = {}
    all_steps_correct = {}
    any_wrong_including_start = {}
    margin_change = {}
    for start, end in INTERVALS:
        interval = _interval(correct, selected, start - FIRST_STEP, end - FIRST_STEP)
        key = f"{start}_{end}"
        first_exit[key] = interval["first_exit"]
        all_steps_correct[key] = interval["all_steps_correct"]
        any_wrong_including_start[key] = interval["any_wrong_including_start"]
        margin_change[key] = {
            margin_name: _mean_change_by_map(values, selected, start - FIRST_STEP, end - FIRST_STEP)
            for margin_name, values in margins.items()
        }
    return {
        "population": "named fixed mask intersected with changed, valid, non-source primary cells",
        "selected_pixels_per_map": [int(value) for value in _counts_by_map(selected)],
        "endpoints": endpoint_summary,
        "first_exit": first_exit,
        "all_steps_correct": all_steps_correct,
        "any_wrong_including_start": any_wrong_including_start,
        "net_margin_change": margin_change,
    }


def summarize(trace: dict[str, Any], bank: dict[str, Any],
              cohorts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize a 193-step paired suffix trace for macro-steps 64 through 256.

    Required trace keys are ``correct`` and ``margin`` with shape
    ``[193,B,H,W]``. ``margin`` is the caller-computed continuous paired
    minimum margin. Optional ``original_margin`` and ``flipped_margin`` use
    the same shape. The function never infers correctness from a zero margin;
    tie semantics belong to the caller's saved Boolean trace.

    Required bank fields are ``changed``, ``mask`` and integer ``distance``.
    An optional ``source`` mask is shape-checked for provenance; distance zero
    is excluded by the primary domain either way. Each optional named cohort
    is intersected with the primary domain and must be fixed before evaluating
    the counterfactuals.
    """
    if not isinstance(trace, dict):
        raise TypeError("trace must be a dictionary")
    if not isinstance(bank, dict):
        raise TypeError("bank must be a dictionary")
    if "correct" not in trace or "margin" not in trace:
        raise ValueError("trace must contain correct and margin")
    if not all(name in bank for name in ("changed", "mask", "distance")):
        raise ValueError("bank must contain changed, mask and distance")

    correct = _binary_array("correct", trace["correct"])
    if correct.ndim != 4 or correct.shape[0] != TRACE_STEPS or any(size <= 0 for size in correct.shape[1:]):
        raise ValueError("correct must have nonempty shape [193,B,H,W] for macro-steps 64..256")
    _, maps, height, width = correct.shape
    shape = (maps, height, width)
    changed = _map_array("changed", bank["changed"], shape)
    valid = _map_array("mask", bank["mask"], shape)
    distance = _distance_array(bank["distance"], shape)
    if "source" in bank:
        _map_array("source", bank["source"], shape)

    margins = {"paired_min": _margin_array("margin", trace["margin"], correct.shape)}
    for name, public_name in (("original_margin", "original"), ("flipped_margin", "flipped")):
        if name in trace and trace[name] is not None:
            margins[public_name] = _margin_array(name, trace[name], correct.shape)

    primary = changed & valid & (distance > 0)
    strict = changed & valid & (distance > 16) & (distance < 32)
    domains = {"primary": primary, "strict": strict}
    endpoints = {
        str(step): {
            name: _endpoint(correct, selected, step - FIRST_STEP)
            for name, selected in domains.items()
        }
        for step in ENDPOINTS
    }
    intervals = {}
    for start, end in INTERVALS:
        interval = {"from_step": int(start), "to_step": int(end)}
        interval.update({
            name: _interval(correct, selected, start - FIRST_STEP, end - FIRST_STEP)
            for name, selected in domains.items()
        })
        intervals[f"{start}_{end}"] = interval

    initial_correct = primary & correct[0]
    margin_summary = {
        "fixed_cohort": "primary cells correct at macro-step 64",
        "initial_correct_per_map": [int(value) for value in _counts_by_map(initial_correct)],
        "sources": {
            name: {
                "endpoints": {
                    str(step): _quantiles_by_map(values[step - FIRST_STEP], initial_correct)
                    for step in ENDPOINTS
                },
                "trajectory_64_256": _quantiles_by_map(values, initial_correct),
            }
            for name, values in margins.items()
        },
    }

    cohort_results: dict[str, Any] = {}
    if cohorts is not None:
        if not isinstance(cohorts, dict):
            raise TypeError("cohorts must be a mapping from names to masks")
        for name, mask in cohorts.items():
            if not isinstance(name, str) or not name:
                raise ValueError("cohort names must be nonempty strings")
            fixed_mask = _map_array(f"cohort {name}", mask, shape)
            cohort_results[name] = _cohort_summary(name, primary & fixed_mask, correct, margins)

    return {
        "schema_version": "audit-195-200-metrics-v1",
        "scope": (
            "Paired correctness and continuous margin summaries on saved maps. Maps are the reporting units; "
            "cells, cell-times, checkpoints, interpolations and permutations are not independent replicates."
        ),
        "trace_steps": {"first": FIRST_STEP, "last": LAST_STEP, "count": TRACE_STEPS},
        "domains": {
            name: {"pixels_per_map": [int(value) for value in _counts_by_map(selected)],
                  "pooled_pixels": int(_counts_by_map(selected).sum())}
            for name, selected in domains.items()
        },
        "endpoints": endpoints,
        "intervals": intervals,
        "paired_margin": margin_summary,
        "cohorts": cohort_results,
    }
