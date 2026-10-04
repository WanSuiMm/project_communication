"""CPU metrics for fixed S1/F1 T64 cohorts and F1-consumer traces.

The cohorts are selected once from native T64 correctness and ``changed``.
Every later statistic uses those same masks, regardless of the intervention
trace's correctness. These are finite conditional behavior proxies, not
mechanism proofs.
"""
from __future__ import annotations

from typing import Any

import numpy as np


TRACE_LENGTH = 193
FIRST_STEP = 64
LAST_STEP = 256
COHORT_NAMES = ("keep", "rescue", "prog", "F_only")
SUMMARY_NAMES = COHORT_NAMES + ("frontier",)
METRIC_NAMES = (
    "immediate64",
    "continuous64_256",
    "endpoint128",
    "endpoint256",
    "ever_after64",
    "sustained241_256",
    "delayed_sustained",
)


def _bool_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must have boolean dtype")
    return array


def _four_neighbor_mask(source: np.ndarray) -> np.ndarray:
    """Return cells with an in-map four-neighbor in ``source``; edges do not wrap."""
    if source.ndim != 3:
        raise ValueError("source must have shape [B,H,W]")
    adjacent = np.zeros_like(source, dtype=bool)
    if source.shape[1] > 1:
        adjacent[:, 1:, :] |= source[:, :-1, :]
        adjacent[:, :-1, :] |= source[:, 1:, :]
    if source.shape[2] > 1:
        adjacent[:, :, 1:] |= source[:, :, :-1]
        adjacent[:, :, :-1] |= source[:, :, 1:]
    return adjacent


def cohorts(s_correct: Any, f_correct: Any, changed: Any) -> dict[str, np.ndarray]:
    """Build fixed, changed-masked cohorts from native S1/F1 correctness.

    Inputs must be Boolean ``[B,H,W]`` arrays with identical nonempty shapes.
    ``frontier`` is the shared ``prog`` cohort with a four-neighbor ``keep``
    cell. Since both endpoint cells belong to their changed-masked cohorts,
    only mask-valid edges contribute.
    """
    s = _bool_array("s_correct", s_correct)
    f = _bool_array("f_correct", f_correct)
    mask = _bool_array("changed", changed)
    if s.ndim != 3 or s.shape != f.shape or s.shape != mask.shape:
        raise ValueError("s_correct, f_correct, and changed must share shape [B,H,W]")
    if any(size == 0 for size in s.shape):
        raise ValueError("B, H, and W must be nonzero")

    result = {
        "keep": mask & s & f,
        "rescue": mask & s & ~f,
        "prog": mask & ~s & ~f,
        "F_only": mask & ~s & f,
        "all_changed": mask.copy(),
    }
    result["frontier"] = result["prog"] & _four_neighbor_mask(result["keep"])
    return result


def _rate_summary(numerators: np.ndarray, denominators: np.ndarray) -> dict[str, Any]:
    nums = [int(value) for value in np.asarray(numerators).reshape(-1)]
    dens = [int(value) for value in np.asarray(denominators).reshape(-1)]
    if len(nums) != len(dens):
        raise ValueError("numerators and denominators must have the same map count")

    per_map = []
    eligible = []
    for index, (numerator, denominator) in enumerate(zip(nums, dens)):
        value = float(numerator / denominator) if denominator else None
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
        "pooled": {
            "numerator": pooled_numerator,
            "denominator": pooled_denominator,
            "value": (float(pooled_numerator / pooled_denominator)
                      if pooled_denominator else None),
        },
        "map_mean": float(np.mean(eligible)) if eligible else None,
        "valid_maps": int(len(eligible)),
        "per_map": per_map,
    }


def _counts(mask: np.ndarray) -> np.ndarray:
    return np.count_nonzero(mask, axis=(1, 2)).astype(np.int64, copy=False)


def _mask_rate(selected: np.ndarray, hit: np.ndarray) -> dict[str, Any]:
    if selected.ndim != 3 or selected.shape != hit.shape:
        raise ValueError("selected and hit must have the same [B,H,W] shape")
    return _rate_summary(_counts(selected & hit), _counts(selected))


def _validate_cohort_dict(cohort_dict: Any, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    if not isinstance(cohort_dict, dict):
        raise ValueError("cohort_dict must be the mapping returned by cohorts()")
    required = set(COHORT_NAMES) | {"frontier", "all_changed"}
    if not required.issubset(cohort_dict):
        raise ValueError(f"cohort_dict must include {sorted(required)}")
    result = {}
    for name in sorted(required):
        array = _bool_array(f"cohort_dict[{name!r}]", cohort_dict[name])
        if array.shape != shape:
            raise ValueError("all cohort masks must share trace shape [B,H,W]")
        result[name] = array

    changed = result["all_changed"]
    union = np.zeros(shape, dtype=bool)
    for name in COHORT_NAMES:
        selected = result[name]
        if np.any(selected & ~changed):
            raise ValueError(f"cohort {name!r} extends outside all_changed")
        if np.any(union & selected):
            raise ValueError("the four T64 cohorts must be disjoint")
        union |= selected
    if not np.array_equal(union, changed):
        raise ValueError("the four T64 cohorts must partition all_changed")
    if np.any(result["frontier"] & ~result["prog"]):
        raise ValueError("frontier must be a subset of prog")
    return result


def summarize(trace: Any, cohort_dict: Any) -> dict[str, Any]:
    """Summarize a Boolean trace for absolute times 64 through 256 inclusive."""
    correct = _bool_array("trace", trace)
    if correct.ndim != 4 or correct.shape[0] != TRACE_LENGTH:
        raise ValueError("trace must have shape [193,B,H,W] for absolute t64..t256")
    if any(size == 0 for size in correct.shape[1:]):
        raise ValueError("B, H, and W must be nonzero")
    masks = _validate_cohort_dict(cohort_dict, correct.shape[1:])

    outcomes = {
        "immediate64": correct[0],
        "continuous64_256": np.all(correct, axis=0),
        "endpoint128": correct[128 - FIRST_STEP],
        "endpoint256": correct[LAST_STEP - FIRST_STEP],
        "ever_after64": np.any(correct[1:], axis=0),
        "sustained241_256": np.all(correct[241 - FIRST_STEP:], axis=0),
        "delayed_sustained": (~correct[0]) & np.all(correct[241 - FIRST_STEP:], axis=0),
    }

    by_cohort = {}
    for name in SUMMARY_NAMES:
        selected = masks[name]
        by_cohort[name] = {
            metric: _mask_rate(selected, outcome)
            for metric, outcome in outcomes.items()
        }
    changed = masks["all_changed"]
    return {
        "cohorts": by_cohort,
        "all_changed": {
            "coverage64": _mask_rate(changed, correct[0]),
            "coverage128": _mask_rate(changed, correct[128 - FIRST_STEP]),
            "coverage256": _mask_rate(changed, correct[LAST_STEP - FIRST_STEP]),
        },
    }


def _pooled(summary: Any, cohort: str, metric: str) -> tuple[float | None, int, int]:
    try:
        item = summary["cohorts"][cohort][metric]
        value = item["pooled"]["value"]
        denominator = int(item["pooled"]["denominator"])
        valid_maps = int(item["valid_maps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"summary is missing {cohort}.{metric} pooled fields") from exc
    if value is not None:
        value = float(value)
    return value, denominator, valid_maps


def rescue_gate(summary: Any, ff_summary: Any) -> dict[str, Any]:
    """Apply the frozen keep/prog behavior thresholds against an FF summary."""
    keep, keep_cells, keep_maps = _pooled(summary, "keep", "continuous64_256")
    prog, prog_cells, prog_maps = _pooled(summary, "prog", "sustained241_256")
    ff_prog, _, _ = _pooled(ff_summary, "prog", "sustained241_256")
    improvement = (prog - ff_prog) if prog is not None and ff_prog is not None else None

    checks = {
        "keep_continuous_ge_0_95": keep is not None and keep >= 0.95,
        "prog_sustained_ge_0_10": prog is not None and prog >= 0.10,
        "prog_sustained_gain_vs_ff_ge_0_10": improvement is not None and improvement >= 0.10,
        "keep_support_ge_16_maps_and_100_cells": keep_maps >= 16 and keep_cells >= 100,
        "prog_support_ge_16_maps_and_100_cells": prog_maps >= 16 and prog_cells >= 100,
    }
    support_qualified = (
        checks["keep_support_ge_16_maps_and_100_cells"]
        and checks["prog_support_ge_16_maps_and_100_cells"]
    )
    passed = bool(all(checks.values()))
    return {
        "pass": passed,
        "status": "PASS" if passed else ("FAIL" if support_qualified else "UNQUALIFIED"),
        "checks": {name: bool(value) for name, value in checks.items()},
        "observed": {
            "keep_continuous64_256": keep,
            "keep_valid_maps": keep_maps,
            "keep_cells": keep_cells,
            "prog_sustained241_256": prog,
            "prog_valid_maps": prog_maps,
            "prog_cells": prog_cells,
            "ff_prog_sustained241_256": ff_prog,
            "prog_gain_vs_ff": improvement,
        },
        "thresholds": {
            "keep_continuous64_256_min": 0.95,
            "prog_sustained241_256_min": 0.10,
            "prog_sustained_gain_vs_ff_min": 0.10,
            "keep_and_prog_min_valid_maps": 16,
            "keep_and_prog_min_cells": 100,
        },
        "interpretation": "finite conditional proxy, not mechanism proof",
    }
