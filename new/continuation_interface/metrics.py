"""Gauge-aware handoff summaries for fixed producer cohorts.

Inputs are Boolean traces for absolute steps 64 through 256. The reporting
unit is a map; cells are descriptive observations within each map. Cohorts
are selected only from the producer's native T64 correctness and the supplied
source-flip mask, never from the continuation consumer's outcomes.
"""
from __future__ import annotations

from typing import Any

import numpy as np


TRACE_LENGTH = 193
TIME_INDEX = {64: 0, 128: 64, 256: 192}


def _binary_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_:
        return array
    if not np.issubdtype(array.dtype, np.number):
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
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError("distance must contain finite integer graph distances")
    if (
        not np.all(np.isfinite(array))
        or not np.all(array == np.floor(array))
    ):
        raise ValueError("distance must contain finite integer graph distances")
    return array.astype(np.int64, copy=False)


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _rate_summary(numerators: Any, denominators: Any) -> dict[str, Any]:
    """Return per-map and pooled integer counts with null rates when empty."""
    nums = [int(value) for value in np.asarray(numerators).reshape(-1)]
    dens = [int(value) for value in np.asarray(denominators).reshape(-1)]
    if len(nums) != len(dens):
        raise ValueError("numerators and denominators must have the same map count")
    per_map = []
    eligible = []
    for index, (numerator, denominator) in enumerate(zip(nums, dens)):
        value = _rate(numerator, denominator)
        per_map.append({
            "map_index": int(index),
            "numerator": numerator,
            "denominator": denominator,
            "value": value,
        })
        if value is not None:
            eligible.append(value)
    pooled_numerator = int(sum(nums))
    pooled_denominator = int(sum(dens))
    return {
        "per_map": per_map,
        "map_mean": float(np.mean(eligible)) if eligible else None,
        "valid_maps": int(len(eligible)),
        "pooled": {
            "numerator": pooled_numerator,
            "denominator": pooled_denominator,
            "value": _rate(pooled_numerator, pooled_denominator),
        },
    }


def _counts_by_map(mask: np.ndarray) -> np.ndarray:
    return np.count_nonzero(mask, axis=(1, 2)).astype(np.int64, copy=False)


def _mask_rate(numerator: np.ndarray, denominator: np.ndarray) -> dict[str, Any]:
    if numerator.shape != denominator.shape or numerator.ndim != 3:
        raise ValueError("rate masks must have the same [B,H,W] shape")
    if np.any(numerator & ~denominator):
        raise ValueError("rate numerator mask must be a subset of its denominator")
    return _rate_summary(_counts_by_map(numerator), _counts_by_map(denominator))


def _frontier_mask(producer_correct: np.ndarray, changed: np.ndarray) -> np.ndarray:
    """Unsolved changed cells with a four-neighbor solved changed cell.

    Shifts are clipped to each map's borders. Batch/map boundaries never
    connect, and rows or columns do not wrap.
    """
    solved_changed = producer_correct & changed
    adjacent_solved = np.zeros_like(changed, dtype=bool)
    if changed.shape[1] > 1:
        adjacent_solved[:, 1:, :] |= solved_changed[:, :-1, :]
        adjacent_solved[:, :-1, :] |= solved_changed[:, 1:, :]
    if changed.shape[2] > 1:
        adjacent_solved[:, :, 1:] |= solved_changed[:, :, :-1]
        adjacent_solved[:, :, :-1] |= solved_changed[:, :, 1:]
    return changed & ~producer_correct & adjacent_solved


def _group_summary(
    trace: np.ndarray,
    producer_correct: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    producer_solved = selected & producer_correct
    producer_unsolved = selected & ~producer_correct
    all_future = np.all(trace, axis=0)
    sustained = trace[-1] & np.all(trace[-16:], axis=0)

    metrics = {
        "preservation": _mask_rate(selected & producer_solved & all_future, producer_solved),
        "immediate_retention": _mask_rate(selected & producer_solved & trace[0], producer_solved),
        "endpoint_retention128": _mask_rate(
            selected & producer_solved & trace[TIME_INDEX[128]], producer_solved
        ),
        "endpoint_retention256": _mask_rate(
            selected & producer_solved & trace[TIME_INDEX[256]], producer_solved
        ),
        "progress": _mask_rate(
            selected & producer_unsolved & np.any(trace[1:], axis=0), producer_unsolved
        ),
        "sustained_progress": _mask_rate(
            selected & producer_unsolved & sustained, producer_unsolved
        ),
        "coverage64": _mask_rate(selected & trace[TIME_INDEX[64]], selected),
        "coverage128": _mask_rate(selected & trace[TIME_INDEX[128]], selected),
        "coverage256": _mask_rate(selected & trace[TIME_INDEX[256]], selected),
    }

    per_map_cells = [int(value) for value in _counts_by_map(selected)]
    return {
        "selected_cells": {
            "per_map": [
                {"map_index": index, "count": count}
                for index, count in enumerate(per_map_cells)
            ],
            "pooled": {"count": int(sum(per_map_cells))},
        },
        "empty": not any(per_map_cells),
        **metrics,
    }


def summarize_handoff(
    correct: Any,
    producer_correct64: Any,
    changed: Any,
    distance: Any,
) -> dict[str, Any]:
    """Summarize continuation by frozen producer-defined map cohorts.

    Args:
        correct: Boolean ``[193,B,H,W]`` trace at absolute times 64..256.
        producer_correct64: Producer native-T64 correctness ``[B,H,W]``.
        changed: Largest source-flipped region ``[B,H,W]``.
        distance: Integer source-graph distance ``[B,H,W]``; ``-1`` may mark
            walls outside ``changed``.

    Every rate includes map-level numerator, denominator and value plus pooled
    counts. Empty cohorts remain present and have ``value: None``. The
    consumer trace is measured only after cohorts have been fixed by the
    producer and source-flip inputs.
    """
    trace = _binary_array("correct", correct)
    if trace.ndim != 4 or trace.shape[0] != TRACE_LENGTH:
        raise ValueError("correct must have shape [193,B,H,W] for absolute t64..256")
    map_shape = tuple(int(value) for value in trace.shape[1:])
    producer_correct = _map_array("producer_correct64", producer_correct64, map_shape)
    changed_mask = _map_array("changed", changed, map_shape)
    distances = _distance_array(distance, map_shape)
    if np.any(changed_mask & (distances < 0)):
        raise ValueError("changed cells must have nonnegative source-graph distance")

    unsolved = changed_mask & ~producer_correct
    groups = {
        "all_changed": changed_mask,
        "strict_16_32": changed_mask & (distances > 16) & (distances < 32),
        "far_gt32": changed_mask & (distances > 32),
        "frontier": _frontier_mask(producer_correct, changed_mask),
        "all_unsolved": unsolved,
        "far_unsolved": unsolved & (distances > 32),
    }
    return {
        "time_axis": {
            "absolute_start": 64,
            "absolute_end": 256,
            "length": TRACE_LENGTH,
            "endpoint_indices": {"t64": 0, "t128": 64, "t256": 192},
            "sustained_progress_window": "absolute t241..256 inclusive",
        },
        "cohort_basis": "producer_correct64 and source-flipped changed/distance masks only",
        "bands": {
            name: _group_summary(trace, producer_correct, selected)
            for name, selected in groups.items()
        },
    }


def decompose(matrix: Any) -> dict[str, Any]:
    """Describe a complete square model-by-model matrix additively.

    This is a descriptive row/column/interaction decomposition for one
    selected model and bank. It performs no inferential or significance test.
    A missing or nonfinite entry leaves the decomposition unqualified.
    """
    try:
        rows = [list(row) for row in matrix]
    except TypeError as exc:
        raise ValueError("matrix must be a nonempty square sequence") from exc
    if not rows:
        raise ValueError("matrix must be a nonempty square sequence")
    if any(len(row) != len(rows) for row in rows):
        raise ValueError("matrix must be square")
    try:
        array = np.asarray(
        [[np.nan if value is None else float(value) for value in row] for row in rows],
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("matrix entries must be numeric or None") from exc
    if array.shape != (len(rows), len(rows)):
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(array)):
        return {
            "status": "unqualified_missing_or_nonfinite",
            "grand_mean": None,
            "row_effect": None,
            "column_effect": None,
            "residual_interaction": None,
        }

    grand = float(np.mean(array))
    row_effect = np.mean(array, axis=1) - grand
    column_effect = np.mean(array, axis=0) - grand
    interaction = array - grand - row_effect[:, None] - column_effect[None, :]
    reconstruction = grand + row_effect[:, None] + column_effect[None, :] + interaction
    return {
        "status": "descriptive_complete_matrix",
        "grand_mean": grand,
        "row_effect": [float(value) for value in row_effect],
        "column_effect": [float(value) for value in column_effect],
        "residual_interaction": [
            [float(value) for value in row] for row in interaction
        ],
        "identity": {
            "reconstruction_max_abs_error": float(np.max(np.abs(reconstruction - array))),
            "row_effect_sum": float(np.sum(row_effect)),
            "column_effect_sum": float(np.sum(column_effect)),
            "interaction_row_sums": [float(value) for value in np.sum(interaction, axis=1)],
            "interaction_column_sums": [float(value) for value in np.sum(interaction, axis=0)],
        },
        "interpretation": "descriptive selected-model decomposition; no statistical test",
    }


def sanity_check() -> dict[str, Any]:
    """Run a tiny CPU fixture for stability, no progress, regression and frontier."""
    trace = np.zeros((TRACE_LENGTH, 1, 3, 4), dtype=bool)
    producer = np.zeros((1, 3, 4), dtype=bool)
    changed = np.zeros_like(producer)
    distance = np.zeros_like(producer, dtype=np.int64)

    stable, frozen, regressed, progressed, isolated = (0, 0), (0, 1), (0, 3), (1, 1), (2, 2)
    changed[0, stable[0], stable[1]] = True
    changed[0, frozen[0], frozen[1]] = True
    changed[0, regressed[0], regressed[1]] = True
    changed[0, progressed[0], progressed[1]] = True
    changed[0, isolated[0], isolated[1]] = True
    producer[0, stable[0], stable[1]] = True
    producer[0, regressed[0], regressed[1]] = True
    producer[0, 2, 3] = True  # Unchanged solved cell must not create frontier.
    distance[0, 2, 3] = -1  # Wall sentinel is valid outside the changed region.
    distance[0, frozen[0], frozen[1]] = 17
    distance[0, regressed[0], regressed[1]] = 33
    distance[0, progressed[0], progressed[1]] = 32
    distance[0, isolated[0], isolated[1]] = 40

    trace[:, 0, stable[0], stable[1]] = True
    trace[:, 0, regressed[0], regressed[1]] = True
    trace[64:, 0, regressed[0], regressed[1]] = False
    trace[1:, 0, progressed[0], progressed[1]] = True

    result = summarize_handoff(trace, producer, changed, distance)
    bands = result["bands"]
    solved = bands["all_changed"]
    unsolved = bands["all_unsolved"]
    frontier = bands["frontier"]
    assert solved["preservation"]["pooled"] == {
        "numerator": 1, "denominator": 2, "value": 0.5
    }
    assert solved["endpoint_retention128"]["pooled"]["value"] == 0.5
    assert unsolved["progress"]["pooled"] == {
        "numerator": 1, "denominator": 3, "value": 1 / 3
    }
    assert unsolved["sustained_progress"]["pooled"]["value"] == 1 / 3
    assert frontier["selected_cells"]["pooled"]["count"] == 1
    assert frontier["progress"]["pooled"]["value"] == 0.0
    assert bands["strict_16_32"]["selected_cells"]["pooled"]["count"] == 1
    assert bands["far_gt32"]["selected_cells"]["pooled"]["count"] == 2

    # An edge cell does not neighbor-connect to the opposite edge.
    edge_changed = np.asarray([[[True, False, False, True]]])
    edge_producer = np.asarray([[[False, False, False, True]]])
    assert not np.any(_frontier_mask(edge_producer, edge_changed))

    matrix = decompose([[1, 2], [3, 4]])
    assert abs(matrix["grand_mean"] - 2.5) < 1e-12
    assert matrix["identity"]["reconstruction_max_abs_error"] < 1e-12
    missing = decompose([[1, None], [3, 4]])
    assert missing["status"] == "unqualified_missing_or_nonfinite"
    empty = summarize_handoff(trace, producer, np.zeros_like(changed), distance)
    empty_all = empty["bands"]["all_changed"]
    assert empty_all["empty"] is True
    assert empty_all["preservation"]["pooled"] == {
        "numerator": 0, "denominator": 0, "value": None
    }
    invalid_distance = distance.copy()
    invalid_distance[0, stable[0], stable[1]] = -1
    try:
        summarize_handoff(trace, producer, changed, invalid_distance)
    except ValueError:
        pass
    else:
        raise AssertionError("changed cells must reject negative graph distances")
    return {"status": "pass", "checks": 13}
