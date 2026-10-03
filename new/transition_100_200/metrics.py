"""CPU-only summaries for paired correctness traces in the 100-200 replay.

The map is the reporting unit. Pixel and cell-step counts describe each map;
they are not independent statistical observations. All returned values are
plain Python scalars and containers so the result can be written as JSON.
"""
from __future__ import annotations

from typing import Any

import numpy as np


DISTANCE_BINS = (
    ("source0", "source cells or BFS distance 0"),
    ("1_16", "1 <= d <= 16, excluding source"),
    ("17_31", "16 < d < 32, excluding source"),
    ("32_63", "32 <= d < 64, excluding source"),
    ("64_inf", "d >= 64, excluding source"),
)
AGE_BINS = (
    ("1", 1, 1),
    ("2_4", 2, 4),
    ("5_8", 5, 8),
    ("9_16", 9, 16),
    ("17_32", 17, 32),
    ("33_64", 33, 64),
    ("65_128", 65, 128),
    ("129_inf", 129, None),
)
FIXED_LAGS = (8, 16, 32, 64, 128)
INTERVALS = ((64, 128), (64, 256))


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
        raise ValueError(f"{name} must have shape [B,H,W] or [B,1,H,W] matching trace")
    return array


def _distance_array(value: Any, shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.shape != shape:
        raise ValueError("distance must have shape [B,H,W] or [B,1,H,W] matching trace")
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise ValueError("distance must contain finite nonnegative integer BFS distances")
    if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
        raise ValueError("distance must contain finite nonnegative integer BFS distances")
    return array.astype(np.int64, copy=False)


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _rate_summary(numerators: Any, denominators: Any) -> dict[str, Any]:
    """Return per-map, equal-map, and pooled rate summaries with integer counts."""
    nums = [int(value) for value in np.asarray(numerators).reshape(-1)]
    dens = [int(value) for value in np.asarray(denominators).reshape(-1)]
    if len(nums) != len(dens):
        raise ValueError("numerators and denominators must have the same map count")
    per_map = []
    eligible_values = []
    for index, (numerator, denominator) in enumerate(zip(nums, dens)):
        value = _rate(numerator, denominator)
        per_map.append({
            "map_index": int(index),
            "numerator": numerator,
            "denominator": denominator,
            "value": value,
        })
        if denominator:
            eligible_values.append(value)
    pooled_numerator = int(sum(nums))
    pooled_denominator = int(sum(dens))
    return {
        "per_map": per_map,
        "equal_map_mean": float(np.mean(eligible_values)) if eligible_values else None,
        "equal_map_eligible_maps": int(len(eligible_values)),
        "pooled_numerator": pooled_numerator,
        "pooled_denominator": pooled_denominator,
        "pooled_rate": _rate(pooled_numerator, pooled_denominator),
    }


def _first_time(trace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seen = np.any(trace, axis=0)
    first = np.argmax(trace, axis=0).astype(np.int32, copy=False)
    first[~seen] = -1
    return first, seen


def _stable_time(trace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Earliest t correct for every observed step t..T; -1 when final-wrong."""
    horizon = trace.shape[0] - 1
    final_correct = trace[-1]
    any_wrong = np.any(~trace, axis=0)
    stable = np.full(final_correct.shape, -1, dtype=np.int32)
    stable[final_correct & ~any_wrong] = 0
    if np.any(final_correct & any_wrong):
        last_wrong = horizon - np.argmax((~trace)[::-1], axis=0)
        selected = final_correct & any_wrong
        stable[selected] = (last_wrong[selected] + 1).astype(np.int32, copy=False)
    return stable, final_correct


def _first_spell_exit(trace: np.ndarray, first: np.ndarray) -> np.ndarray:
    """First regression after the first correct step, or T+1 if not observed."""
    horizon = trace.shape[0] - 1
    exit_time = np.full(first.shape, horizon + 1, dtype=np.int32)
    for step in range(1, horizon + 1):
        exits = (first >= 0) & (first < step) & (~trace[step]) & (exit_time == horizon + 1)
        exit_time[exits] = step
    return exit_time


def _time_summary(values: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    """Summarize observed times for selected cells without inventing empty values."""
    per_map: list[dict[str, Any]] = []
    map_medians: list[float] = []
    all_values: list[np.ndarray] = []
    for index in range(selected.shape[0]):
        values_here = values[index][selected[index]]
        n = int(values_here.size)
        median = float(np.median(values_here)) if n else None
        per_map.append({"map_index": int(index), "n": n, "median": median})
        if n:
            map_medians.append(median)
            all_values.append(values_here.astype(np.int64, copy=False))
    pooled_values = np.concatenate(all_values) if all_values else np.empty(0, dtype=np.int64)
    return {
        "per_map": per_map,
        "equal_map_mean_of_map_medians": float(np.mean(map_medians)) if map_medians else None,
        "equal_map_eligible_maps": int(len(map_medians)),
        "pooled_n": int(pooled_values.size),
        "pooled_median": float(np.median(pooled_values)) if pooled_values.size else None,
    }


def _first_acquisition_summary(first: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    per_map = []
    medians: list[float] = []
    all_first: list[np.ndarray] = []
    for index in range(selected.shape[0]):
        size = int(np.count_nonzero(selected[index]))
        times = first[index][selected[index] & (first[index] >= 0)]
        n = int(times.size)
        median = float(np.median(times)) if n else None
        per_map.append({
            "map_index": int(index),
            "selected_pixels": size,
            "ever_correct": n,
            "never_correct": int(size - n),
            "median_first_correct_time": median,
        })
        if n:
            medians.append(median)
            all_first.append(times.astype(np.int64, copy=False))
    pooled = np.concatenate(all_first) if all_first else np.empty(0, dtype=np.int64)
    return {
        "definition": "first observed correct step within the finite trace",
        "per_map": per_map,
        "equal_map_mean_of_map_medians": float(np.mean(medians)) if medians else None,
        "equal_map_eligible_maps": int(len(medians)),
        "pooled_ever_correct": int(pooled.size),
        "pooled_median_first_correct_time": float(np.median(pooled)) if pooled.size else None,
        "pooled_never_correct": int(sum(row["never_correct"] for row in per_map)),
    }


def _finite_horizon_summary(
    trace: np.ndarray,
    selected: np.ndarray,
    first: np.ndarray,
    stable: np.ndarray,
) -> dict[str, Any]:
    final_correct = trace[-1]
    per_map = []
    first_medians: list[float] = []
    first_final_medians: list[float] = []
    stable_medians: list[float] = []
    pooled_first: list[np.ndarray] = []
    pooled_stable: list[np.ndarray] = []
    total_selected = total_never = total_final_wrong = total_final_correct = 0
    for index in range(selected.shape[0]):
        chosen = selected[index]
        first_values = first[index][chosen & (first[index] >= 0)]
        terminal_good = chosen & final_correct[index]
        terminal_bad = chosen & ~final_correct[index]
        stable_values = stable[index][terminal_good]
        selected_n = int(np.count_nonzero(chosen))
        never_n = int(np.count_nonzero(chosen & (first[index] < 0)))
        final_wrong_n = int(np.count_nonzero(terminal_bad))
        final_correct_n = int(np.count_nonzero(terminal_good))
        first_median = float(np.median(first_values)) if first_values.size else None
        first_final_values = first[index][terminal_good]
        first_final_median = float(np.median(first_final_values)) if first_final_values.size else None
        stable_median = float(np.median(stable_values)) if stable_values.size else None
        per_map.append({
            "map_index": int(index),
            "selected_pixels": selected_n,
            "never_correct": never_n,
            "final_wrong": final_wrong_n,
            "final_correct": final_correct_n,
            "final_correct_conditioned_ineligible": int(final_correct_n == 0),
            "ineligible_stable_time_final_wrong": final_wrong_n,
            "first_correct_median_ever_correct": first_median,
            "first_correct_median_final_correct": first_final_median,
            "stable_time_median_final_correct": stable_median,
        })
        if first_values.size:
            first_medians.append(first_median)
            pooled_first.append(first_values.astype(np.int64, copy=False))
        if final_correct_n:
            first_final_medians.append(first_final_median)
            pooled_stable.append(stable_values.astype(np.int64, copy=False))
            stable_medians.append(stable_median)
        total_selected += selected_n
        total_never += never_n
        total_final_wrong += final_wrong_n
        total_final_correct += final_correct_n

    first_concat = np.concatenate(pooled_first) if pooled_first else np.empty(0, dtype=np.int64)
    stable_concat = np.concatenate(pooled_stable) if pooled_stable else np.empty(0, dtype=np.int64)
    final_first_values = [first[index][selected[index] & final_correct[index]] for index in range(selected.shape[0])]
    final_first_concat = (
        np.concatenate([values for values in final_first_values if values.size])
        if any(values.size for values in final_first_values)
        else np.empty(0, dtype=np.int64)
    )
    return {
        "definition": "finite-trace description only; stable time is undefined for final-wrong cells",
        "per_map": per_map,
        "lag_final_correct": {
            "definition": "cellwise stable_time - first_correct_time, conditioned on correctness at T",
            **_time_summary(stable - first, selected & final_correct),
        },
        "equal_map": {
            "first_correct_median_ever_correct_mean": float(np.mean(first_medians)) if first_medians else None,
            "first_correct_eligible_maps": int(len(first_medians)),
            "first_correct_median_final_correct_mean": float(np.mean(first_final_medians)) if first_final_medians else None,
            "stable_time_median_final_correct_mean": float(np.mean(stable_medians)) if stable_medians else None,
            "final_correct_eligible_maps": int(len(stable_medians)),
        },
        "pooled": {
            "selected_pixels": int(total_selected),
            "never_correct": int(total_never),
            "final_wrong": int(total_final_wrong),
            "final_correct": int(total_final_correct),
            "final_correct_conditioned_ineligible": int(total_final_correct == 0),
            "ineligible_stable_time_final_wrong": int(total_final_wrong),
            "first_correct_median_ever_correct": float(np.median(first_concat)) if first_concat.size else None,
            "first_correct_median_final_correct": float(np.median(final_first_concat)) if final_first_concat.size else None,
            "stable_time_median_final_correct": float(np.median(stable_concat)) if stable_concat.size else None,
        },
    }


def _fixed_lag_summary(
    trace: np.ndarray,
    selected: np.ndarray,
    first: np.ndarray,
    first_exit: np.ndarray,
) -> dict[str, Any]:
    horizon = trace.shape[0] - 1
    rows = []
    for lag in FIXED_LAGS:
        survival_n = np.zeros(selected.shape[0], dtype=np.int64)
        eligible_n = np.zeros(selected.shape[0], dtype=np.int64)
        never_n = np.zeros(selected.shape[0], dtype=np.int64)
        right_censored_n = np.zeros(selected.shape[0], dtype=np.int64)
        for index in range(selected.shape[0]):
            chosen = selected[index]
            first_here = first[index]
            seen = chosen & (first_here >= 0)
            eligible = seen & (first_here <= horizon - lag)
            right_censored = seen & (first_here > horizon - lag)
            survived = eligible & (first_exit[index] > first_here + lag)
            survival_n[index] = int(np.count_nonzero(survived))
            eligible_n[index] = int(np.count_nonzero(eligible))
            never_n[index] = int(np.count_nonzero(chosen & (first_here < 0)))
            right_censored_n[index] = int(np.count_nonzero(right_censored))
        rows.append({
            "lag": int(lag),
            "uninterrupted_survival": _rate_summary(survival_n, eligible_n),
            "exclusions": {
                "per_map": [
                    {
                        "map_index": int(index),
                        "never_correct": int(never_n[index]),
                        "right_censored": int(right_censored_n[index]),
                        "eligible": int(eligible_n[index]),
                    }
                    for index in range(selected.shape[0])
                ],
                "never_correct_pooled": int(never_n.sum()),
                "right_censored_pooled": int(right_censored_n.sum()),
                "eligible_pooled": int(eligible_n.sum()),
            },
        })
    return {
        "definition": (
            "first-correct spell remains continuously correct through lag k; only first_time <= T-k is eligible. "
            "Never-correct cells are excluded and late first-correct cells are right-censored. This is not a latent commitment claim."
        ),
        "lags": rows,
    }


def _interval_summary(
    trace: np.ndarray,
    selected: np.ndarray,
    start: int,
    end: int,
) -> dict[str, Any]:
    horizon = trace.shape[0] - 1
    result: dict[str, Any] = {"from_step": int(start), "to_step": int(end)}
    if end > horizon:
        result.update({"available": False, "trace_last_step": int(horizon)})
        return result
    before = trace[start] & selected
    after = trace[end] & selected
    pixels = selected.reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64)
    correct_start = before.reshape(before.shape[0], -1).sum(axis=1, dtype=np.int64)
    wrong_start = (selected & ~trace[start]).reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64)
    acquired_mask = selected & ~trace[start] & trace[end]
    endpoint_lost_mask = selected & trace[start] & ~trace[end]
    exited_mask = selected & trace[start] & np.any(~trace[start + 1:end + 1], axis=0)
    survived_mask = selected & np.all(trace[start:end + 1], axis=0)
    gained = acquired_mask.reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64)
    endpoint_lost = endpoint_lost_mask.reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64)
    first_exit = exited_mask.reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64)
    continuous = survived_mask.reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64)
    net = after.reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64) - correct_start
    error = net - (gained - endpoint_lost)
    if np.any(error != 0):
        raise AssertionError((start, end, error.tolist()))
    result.update({
        "available": True,
        "acquisition": _rate_summary(gained, wrong_start),
        "endpoint_destruction": _rate_summary(endpoint_lost, correct_start),
        "first_exit": _rate_summary(first_exit, correct_start),
        "continuous_survival": _rate_summary(continuous, correct_start),
        "net_gain": _rate_summary(net, pixels),
        "net_gain_weighting": "pooled_rate is weighted by each map's selected pixel count; equal_map_mean averages map rates",
        "identity": {
            "pooled_net_change": int(net.sum()),
            "pooled_acquired_minus_endpoint_destroyed": int(gained.sum() - endpoint_lost.sum()),
            "accounting_error": int(error.sum()),
        },
    })
    return result


def _step_accounting(trace: np.ndarray, selected: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for step in range(1, trace.shape[0]):
        previous = trace[step - 1] & selected
        current = trace[step] & selected
        acquired = ((~previous) & current).reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64)
        destroyed = (previous & (~current)).reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64)
        delta = current.reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64) - previous.reshape(selected.shape[0], -1).sum(axis=1, dtype=np.int64)
        error = delta - (acquired - destroyed)
        if np.any(error != 0):
            raise AssertionError((step, error.tolist()))
        rows.append({
            "t": int(step),
            "per_map": [
                {
                    "map_index": int(index),
                    "acquired": int(acquired[index]),
                    "destroyed": int(destroyed[index]),
                    "delta_correct": int(delta[index]),
                    "accounting_error": int(error[index]),
                }
                for index in range(selected.shape[0])
            ],
            "pooled": {
                "acquired": int(acquired.sum()),
                "destroyed": int(destroyed.sum()),
                "delta_correct": int(delta.sum()),
                "accounting_error": int(error.sum()),
            },
        })
    return rows


def _age_hazard(trace: np.ndarray, bands: dict[str, np.ndarray]) -> dict[str, Any]:
    """Next-step loss hazard by current consecutive-correct-run age."""
    maps = trace.shape[1]
    totals = {
        name: {
            age_name: [np.zeros(maps, dtype=np.int64), np.zeros(maps, dtype=np.int64)]
            for age_name, _, _ in AGE_BINS
        }
        for name in bands
    }
    age = np.zeros(trace.shape[1:], dtype=np.int32)
    for step in range(trace.shape[0] - 1):
        age = np.where(trace[step], age + 1, 0)
        lost_next = ~trace[step + 1]
        for name, selected in bands.items():
            for age_name, low, high in AGE_BINS:
                at_risk = selected & trace[step] & (age >= low)
                if high is not None:
                    at_risk &= age <= high
                denominator = at_risk.reshape(maps, -1).sum(axis=1, dtype=np.int64)
                numerator = (at_risk & lost_next).reshape(maps, -1).sum(axis=1, dtype=np.int64)
                totals[name][age_name][0] += numerator
                totals[name][age_name][1] += denominator
    return {
        "definition": "cell-step next-update loss hazard; age is the current consecutive correct-run length and resets after regression",
        "age_bins": {name: (f"age == {low}" if high == low else f"{low} <= age <= {high}" if high else f"age >= {low}") for name, low, high in AGE_BINS},
        "by_distance": {
            band: {
                age_name: _rate_summary(*totals[band][age_name])
                for age_name, _, _ in AGE_BINS
            }
            for band in bands
        },
    }


def summarize(trace: Any, changed: Any, distance: Any, source: Any = None) -> dict[str, Any]:
    """Summarize a paired boolean trace of shape ``[T+1,B,H,W]``.

    ``changed`` and ``source`` accept ``[B,H,W]`` or ``[B,1,H,W]`` masks;
    ``distance`` accepts the same shapes with integer BFS distances. When no
    source mask is supplied, distance-zero cells define the source. The
    primary cohort is changed, non-source cells with ``16 < distance < 32``.
    Endpoint intervals use all changed cells, including source cells.
    """
    correct = _binary_array("trace", trace)
    if correct.ndim != 4 or any(size <= 0 for size in correct.shape):
        raise ValueError("trace must have nonempty shape [T+1,B,H,W]")
    steps, maps, height, width = correct.shape
    horizon = steps - 1
    shape = (maps, height, width)
    changed_array = _map_array("changed", changed, shape)
    distances = _distance_array(distance, shape)
    if np.any(changed_array & (distances < 0)):
        raise ValueError("changed cells must have nonnegative BFS distances")
    if source is None:
        source_array = distances == 0
    else:
        source_array = _map_array("source", source, shape)
        source_array = source_array | (distances == 0)
    selected = changed_array
    source_selected = selected & source_array
    bands: dict[str, np.ndarray] = {
        "source0": source_selected,
        "1_16": selected & ~source_array & (distances >= 1) & (distances <= 16),
        "17_31": selected & ~source_array & (distances > 16) & (distances < 32),
        "32_63": selected & ~source_array & (distances >= 32) & (distances < 64),
        "64_inf": selected & ~source_array & (distances >= 64),
    }
    primary = bands["17_31"]
    first, ever = _first_time(correct)
    stable, final_correct = _stable_time(correct)
    first_exit = _first_spell_exit(correct, first)

    coverage = []
    for step in range(steps):
        coverage.append({
            "t": int(step),
            "q": _rate_summary(
                (correct[step] & primary).reshape(maps, -1).sum(axis=1, dtype=np.int64),
                primary.reshape(maps, -1).sum(axis=1, dtype=np.int64),
            ),
            "q_by_distance": {
                name: _rate_summary(
                    (correct[step] & band).reshape(maps, -1).sum(axis=1, dtype=np.int64),
                    band.reshape(maps, -1).sum(axis=1, dtype=np.int64),
                )
                for name, band in bands.items()
            },
        })

    intervals = {
        f"{start}_{end}": _interval_summary(correct, selected, start, end)
        for start, end in INTERVALS
    }
    intervals_by_distance = {
        band: {
            f"{start}_{end}": _interval_summary(correct, band_mask, start, end)
            for start, end in INTERVALS
        }
        for band, band_mask in bands.items()
    }
    first_acquisition_by_distance = {
        name: _first_acquisition_summary(first, band)
        for name, band in bands.items()
    }
    finite_horizon_by_distance = {
        name: _finite_horizon_summary(correct, band, first, stable)
        for name, band in bands.items()
    }
    fixed_lag_by_distance = {
        name: _fixed_lag_summary(correct, band, first, first_exit)
        for name, band in bands.items()
    }
    # Check the per-step count identity for every reported distance band as well
    # as for all changed cells; expose the all-changed counts for auditability.
    for band in tuple(bands.values()) + (selected,):
        _step_accounting(correct, band)

    return {
        "schema_version": "transition-100-200-metrics-v1",
        "scope": (
            "Descriptive map-level summaries. Maps are the reporting units; cells and cell-times are not independent. "
            "Finite-trace stability is descriptive and does not establish latent commitment."
        ),
        "trace_steps": {"first": 0, "last": int(horizon)},
        "cohort_definition": "changed, non-source cells with 16 < BFS distance < 32",
        "interval_population": "all changed cells, including source cells",
        "distance_bins": {name: definition for name, definition in DISTANCE_BINS},
        "coverage": coverage,
        "step_accounting": _step_accounting(correct, selected),
        "intervals": intervals,
        "intervals_by_distance": intervals_by_distance,
        "first_acquisition": _first_acquisition_summary(first, primary),
        "first_acquisition_by_distance": first_acquisition_by_distance,
        "fixed_lag_first_correct_survival": {
            "primary": _fixed_lag_summary(correct, primary, first, first_exit),
            "by_distance": fixed_lag_by_distance,
        },
        "finite_horizon_lags": {
            "primary": _finite_horizon_summary(correct, primary, first, stable),
            "by_distance": finite_horizon_by_distance,
        },
        "age_hazard": _age_hazard(correct, bands),
        "age_spell_note": "Age hazard uses each current correct run, including reacquisitions; fixed-lag survival separately follows the first-correct spell.",
        "counts": {
            "maps": int(maps),
            "changed_pixels_per_map": [int(value) for value in selected.reshape(maps, -1).sum(axis=1)],
            "primary_pixels_per_map": [int(value) for value in primary.reshape(maps, -1).sum(axis=1)],
        },
    }
