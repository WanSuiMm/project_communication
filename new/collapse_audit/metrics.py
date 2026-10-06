"""Pure-NumPy summaries for the bounded collapse audit.

Inputs use ``[time, map, height, width]`` for suffix trajectories and
``[map, height, width]`` for single steps. The returned objects contain only
Python scalars, lists, dictionaries, and ``None``, so they can be serialized
directly as JSON.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_SUFFIX_LENGTH = 193  # T64 through T256, inclusive.
_ENDPOINTS = (("T64", 0), ("T128", 64), ("T256", 192))


def _validate_masks(
    correct_or_before: np.ndarray,
    baseline_or_after: np.ndarray,
    changed: np.ndarray,
    distance: np.ndarray,
    *,
    trajectory: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(correct_or_before)
    second = np.asarray(baseline_or_after)
    changed_array = np.asarray(changed)
    distance_array = np.asarray(distance)

    expected_ndim = 4 if trajectory else 3
    if first.ndim != expected_ndim or second.shape != changed_array.shape:
        raise ValueError("correct/before, baseline/after, and changed have incompatible shapes")
    if changed_array.ndim != 3 or distance_array.shape != changed_array.shape:
        raise ValueError("changed and distance must have the same [map, height, width] shape")
    if trajectory:
        if first.shape[0] != _SUFFIX_LENGTH or first.shape[1:] != changed_array.shape:
            raise ValueError("correct must have shape [193, map, height, width]")
    elif first.shape != changed_array.shape:
        raise ValueError("before must have the same [map, height, width] shape as changed")
    if any(array.dtype.kind != "b" for array in (first, second, changed_array)):
        raise TypeError("correct/before, baseline/after, and changed must be boolean arrays")
    if distance_array.dtype.kind not in "iu":
        raise TypeError("distance must be an integer array")

    return first, second, changed_array, distance_array


def _ratio(numerator: int | float, denominator: int | float) -> dict[str, Any]:
    numerator_value = int(numerator) if isinstance(numerator, (int, np.integer)) else float(numerator)
    denominator_value = int(denominator) if isinstance(denominator, (int, np.integer)) else float(denominator)
    value = numerator_value / denominator_value if denominator_value else None
    return {
        "numerator": numerator_value,
        "denominator": denominator_value,
        "value": float(value) if value is not None else None,
    }


def _counts(mask: np.ndarray) -> dict[str, Any]:
    per_map = [
        {"map_index": map_index, "count": int(mask[map_index].sum())}
        for map_index in range(mask.shape[0])
    ]
    return {"count": int(mask.sum()), "maps": per_map}


def _stat(success: np.ndarray, opportunities: np.ndarray) -> dict[str, Any]:
    """Return a pooled rate and the same numerator/denominator per map."""
    maps = [
        _ratio(
            int((success[map_index] & opportunities[map_index]).sum()),
            int(opportunities[map_index].sum()),
        )
        for map_index in range(opportunities.shape[0])
    ]
    for map_index, row in enumerate(maps):
        row["map_index"] = map_index
    return {
        "pooled": _ratio(int((success & opportunities).sum()), int(opportunities.sum())),
        "maps": maps,
    }


def _coverage(success: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    stat = _stat(success, support)
    map_values = [row["value"] for row in stat["maps"] if row["value"] is not None]
    stat["mapmean"] = _ratio(float(sum(map_values)), len(map_values))
    return stat


def _cohorts(changed: np.ndarray, distance: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all_changed": changed,
        "strict_16_32": changed & (distance > 16) & (distance < 32),
    }


def _suffix_group(
    correct: np.ndarray,
    baseline: np.ndarray,
    cohort: np.ndarray,
) -> dict[str, Any]:
    baseline_correct = cohort & baseline
    baseline_wrong = cohort & ~baseline
    endpoints = {
        name: _coverage(correct[index], cohort)
        for name, index in _ENDPOINTS
    }
    same_view_first_step_lost = correct[0] & ~correct[1]
    return {
        "support": _counts(cohort),
        "producer_baseline_correct_reference": _counts(baseline_correct),
        "endpoint_coverage": endpoints,
        "terminal_retention": _stat(correct[-1], baseline_correct),
        "continuous_preservation": _stat(correct.all(axis=0), baseline_correct),
        "progress_terminal": _stat(correct[-1], baseline_wrong),
        "progress_ever": _stat(correct[1:].any(axis=0), baseline_wrong),
        "sustained_progress_last16": _stat(correct[-16:].all(axis=0), baseline_wrong),
        "handoff_destruction": _stat(~correct[0], baseline_correct),
        "handoff_acquisition": _stat(correct[0], baseline_wrong),
        "same_view_first_step_destruction": _stat(
            same_view_first_step_lost,
            cohort & correct[0],
        ),
    }


def summarize_suffix(
    correct: np.ndarray,
    baseline: np.ndarray,
    changed: np.ndarray,
    distance: np.ndarray,
) -> dict[str, Any]:
    """Summarize T64..T256 correctness under both required changed cohorts.

    ``correct`` must be boolean with shape ``[193, B, H, W]``. ``baseline``
    is the producer-native paired-correct mask at T64 and must have shape
    ``[B, H, W]``; it is held fixed across the supplied readout views.
    ``changed`` is a boolean mask and ``distance`` an integer array of shape
    ``[B, H, W]``. The two returned groups are ``all_changed`` and the strict
    distance band ``strict_16_32`` (changed and 16 < distance < 32).

    Endpoint coverage includes pooled pixel coverage, the mean of nonempty
    per-map coverage rates, and each map's numerator/denominator. Every other
    rate also carries pooled and per-map numerator/denominator/value fields;
    an empty denominator yields ``value: None``.
    """
    correct_array, baseline_array, changed_array, distance_array = _validate_masks(
        correct, baseline, changed, distance, trajectory=True
    )
    return {
        name: _suffix_group(correct_array, baseline_array, cohort)
        for name, cohort in _cohorts(changed_array, distance_array).items()
    }


def _delta_stat(after: np.ndarray, before: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    def row(after_map: np.ndarray, before_map: np.ndarray, support_map: np.ndarray) -> dict[str, Any]:
        return _ratio(
            int((after_map & support_map).sum()) - int((before_map & support_map).sum()),
            int(support_map.sum()),
        )

    maps = [
        {"map_index": map_index, **row(after[map_index], before[map_index], support[map_index])}
        for map_index in range(support.shape[0])
    ]
    return {
        "pooled": row(after, before, support),
        "maps": maps,
    }


def summarize_step(
    before: np.ndarray,
    after: np.ndarray,
    changed: np.ndarray,
    distance: np.ndarray,
) -> dict[str, Any]:
    """Summarize a same-state single step for the two changed cohorts.

    All masks must be boolean with shape ``[B, H, W]``; ``distance`` must be
    integer with the same shape. Net coverage change is the signed fraction
    ``(after_correct - before_correct) / cohort_support``. Acquisition and
    destruction rates use the before-wrong and before-correct cells as their
    respective denominators.
    """
    before_array, after_array, changed_array, distance_array = _validate_masks(
        before, after, changed, distance, trajectory=False
    )
    output: dict[str, Any] = {}
    for name, cohort in _cohorts(changed_array, distance_array).items():
        output[name] = {
            "support": _counts(cohort),
            "net_coverage_change": _delta_stat(after_array, before_array, cohort),
            "acquisition": _stat(after_array, cohort & ~before_array),
            "destruction": _stat(~after_array, cohort & before_array),
        }
    return output
