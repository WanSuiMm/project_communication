"""Pure NumPy summaries for zero-training continuation-credit diagnostics.

``summarize_pair`` consumes already-saved, flattened mean-gradient vectors. It
does not load models, read artifacts, or execute training code.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_PREFIXES = {
    "E": ("encoder.",),
    "F": ("f_in.", "f_out."),
    "Q": ("q_in.", "q_out."),
    "R": ("readout.",),
}
_UPDATES = (175, 180, 195, 200)
_PHENOTYPE_THRESHOLDS = {
    "collapse_preservation_drop_min": 0.10,
    "collapse_sustained_progress_drop_min": 0.10,
    "recovery_preservation_rise_min": 0.10,
    "recovery_sustained_progress_rise_min": 0.10,
    "recovery_no_regression_max": 0.02,
}
_CREDIT_THRESHOLDS = {
    "collapse_C_parallel_drop_min": 0.05,
    "collapse_cosine_drop_min": 0.10,
    "recovery_C_parallel_rise_min": 0.05,
    "recovery_cosine_rise_min": 0.10,
    "recovery_C_parallel_regression_max": 0.02,
    "recovery_cosine_regression_max": 0.05,
}


def _parameter_layout(
    parameter_names: Sequence[str], parameter_shapes: Sequence[Sequence[int]]
) -> tuple[list[str], list[tuple[int, ...]], list[tuple[int, int]]]:
    names = list(parameter_names)
    if any(not isinstance(name, str) for name in names):
        raise TypeError("parameter_names must contain strings")
    if names != sorted(names) or len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique and lexicographically sorted")
    raw_shapes = list(parameter_shapes)
    if len(names) != len(raw_shapes):
        raise ValueError("parameter_names and parameter_shapes must have equal length")

    shapes: list[tuple[int, ...]] = []
    offsets: list[tuple[int, int]] = []
    start = 0
    for shape in raw_shapes:
        try:
            dims = tuple(shape)
        except TypeError as exc:
            raise TypeError("each parameter shape must be a sequence of dimensions") from exc
        if any(isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) for dim in dims):
            raise TypeError("parameter dimensions must be integers")
        normalized = tuple(int(dim) for dim in dims)
        if any(dim < 0 for dim in normalized):
            raise ValueError("parameter dimensions must be nonnegative")
        size = math.prod(normalized)  # An empty shape denotes one scalar parameter.
        end = start + size
        shapes.append(normalized)
        offsets.append((start, end))
        start = end
    return names, shapes, offsets


def _vector(value: Any, expected_size: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be convertible to a float64 vector") from exc
    if vector.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if vector.size != expected_size:
        raise ValueError(f"{label} has {vector.size} elements; expected {expected_size}")
    return vector


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _stable_norm(vector: np.ndarray) -> float | None:
    if vector.size == 0:
        return 0.0
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        scaled_norm = float(np.sqrt(np.dot(vector / scale, vector / scale)))
        return _finite_float(scale * scaled_norm)


def _pair_summary(g8: np.ndarray, g64: np.ndarray) -> dict[str, Any]:
    available = bool(g8.size)
    if not available:
        return {
            "available": False,
            "norm8": None,
            "norm64": None,
            "cosine": None,
            "C_parallel": None,
            "C_miss": None,
            "norm_ratio": None,
            "dot": None,
            "finite": None,
            "finite8": None,
            "finite64": None,
            "finite_difference": None,
            "zero8": None,
            "zero64": None,
            "zero_difference": None,
            "near_equal_gradient": None,
        }

    finite8 = bool(np.all(np.isfinite(g8)))
    finite64 = bool(np.all(np.isfinite(g64)))
    zero8 = bool(np.all(g8 == 0)) if finite8 else None
    zero64 = bool(np.all(g64 == 0)) if finite64 else None

    norm8 = _stable_norm(g8) if finite8 else None
    norm64 = _stable_norm(g64) if finite64 else None
    difference = None
    finite_difference = False
    zero_difference = None
    if finite8 and finite64:
        with np.errstate(over="ignore", invalid="ignore"):
            difference = g64 - g8
        finite_difference = bool(np.all(np.isfinite(difference)))
        zero_difference = bool(np.all(difference == 0)) if finite_difference else None

    cosine = c_parallel = c_miss = norm_ratio = dot = None
    if finite8 and finite64:
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            dot = _finite_float(np.dot(g8, g64))
            scale8 = float(np.max(np.abs(g8)))
            scale64 = float(np.max(np.abs(g64)))
            if scale8 > 0.0 and scale64 > 0.0:
                a = g8 / scale8
                b = g64 / scale64
                norm_a = float(np.sqrt(np.dot(a, a)))
                norm_b = float(np.sqrt(np.dot(b, b)))
                cosine = _finite_float(np.dot(a / norm_a, b / norm_b))
                cosine = min(1.0, max(-1.0, cosine)) if cosine is not None else None
                scale_ratio = _finite_float(scale8 / scale64)
                if scale_ratio is not None:
                    c_parallel = _finite_float(
                        scale_ratio * np.dot(a, b) / np.dot(b, b)
                    )
                    norm_ratio = _finite_float(scale_ratio * norm_a / norm_b)
                    # ||g64-g8|| / ||g64||, evaluated after scaling by max(|g64|).
                    scaled_difference = b - scale_ratio * a
                    c_miss = _finite_float(
                        np.sqrt(np.dot(scaled_difference, scaled_difference)) / norm_b
                    )
            elif scale64 > 0.0:
                # The short-horizon vector is exactly zero.
                c_parallel = 0.0
                norm_ratio = 0.0
                c_miss = 1.0
            # If g64 is zero, cosine, C_parallel, C_miss and norm_ratio are
            # undefined by their denominators and remain null.

    return {
        "available": True,
        "norm8": norm8,
        "norm64": norm64,
        "cosine": cosine,
        "C_parallel": c_parallel,
        "C_miss": c_miss,
        "norm_ratio": norm_ratio,
        "dot": dot,
        "finite": bool(finite8 and finite64),
        "finite8": finite8,
        "finite64": finite64,
        "finite_difference": finite_difference,
        "zero8": zero8,
        "zero64": zero64,
        "zero_difference": zero_difference,
        "near_equal_gradient": (None if cosine is None or c_miss is None
                                else bool(cosine >= 0.90 and c_miss <= 0.10)),
    }


def summarize_pair(
    g8: Any,
    g64: Any,
    parameter_names: Sequence[str],
    parameter_shapes: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Summarize the supplied mean-gradient vectors overall and by module.

    The supplied names and shapes must use the same sorted parameter order as
    the existing continuation gradient helper. ``g8`` and ``g64`` are flattened
    vectors in that order. All outputs are JSON-safe; undefined or nonfinite
    numeric results are represented by ``None``.
    """
    names, shapes, offsets = _parameter_layout(parameter_names, parameter_shapes)
    size = sum(end - start for start, end in offsets)
    vector8 = _vector(g8, size, "g8")
    vector64 = _vector(g64, size, "g64")

    slices = {name: (vector8[start:end], vector64[start:end])
              for name, (start, end) in zip(names, offsets)}

    def select(predicate) -> tuple[np.ndarray, np.ndarray]:
        selected = [slices[name] for name in names if predicate(name)]
        if not selected:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        return (np.concatenate([part[0] for part in selected]),
                np.concatenate([part[1] for part in selected]))

    groups: dict[str, dict[str, Any]] = {}
    for key, prefixes in _PREFIXES.items():
        left, right = select(lambda name, p=prefixes: name.startswith(p))
        groups[key] = _pair_summary(left, right)
    recurrent8, recurrent64 = select(lambda name: not name.startswith("readout."))
    groups["recurrent"] = _pair_summary(recurrent8, recurrent64)

    return {
        "overall": _pair_summary(vector8, vector64),
        "groups": groups,
        "parameter_names": names,
        "parameter_shapes": [list(shape) for shape in shapes],
        "aggregation_basis": "metrics computed from each supplied mean-gradient vector pair",
    }


def _as_update_map(rows: Mapping[Any, Any] | None) -> dict[int, Any]:
    if not isinstance(rows, Mapping):
        return {}
    normalized: dict[int, Any] = {}
    for key, value in rows.items():
        if isinstance(key, bool):
            continue
        try:
            update = int(key)
        except (TypeError, ValueError, OverflowError):
            continue
        if str(key).strip() not in (str(update), f"+{update}"):
            continue
        if update in normalized:
            raise ValueError(f"duplicate row for update {update}")
        normalized[update] = value
    return normalized


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    return _finite_float(value)


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return _finite_float(left - right)


def _ge(value: float | None, threshold: float) -> bool | None:
    return None if value is None else bool(value >= threshold)


def _le(value: float | None, threshold: float) -> bool | None:
    return None if value is None else bool(value <= threshold)


def _conjunction(checks: Mapping[str, bool | None]) -> bool | None:
    if any(value is False for value in checks.values()):
        return False
    if all(value is True for value in checks.values()):
        return True
    return None


def _check_lists(checks: Mapping[str, bool | None]) -> tuple[list[str], list[str]]:
    failed = [name for name, value in checks.items() if value is False]
    undefined = [name for name, value in checks.items() if value is None]
    return failed, undefined


def _phenotype_size_summary(
    phenotype_rows: Mapping[int, Any], size_key: str, *, role: str
) -> dict[str, Any]:
    values: dict[int, dict[str, float | None]] = {}
    for update in _UPDATES:
        row = phenotype_rows.get(update)
        size_row = row.get(size_key) if isinstance(row, Mapping) else None
        if not isinstance(size_row, Mapping):
            size_row = {}
        values[update] = {
            "preservation": _number(size_row.get("preservation")),
            "sustained_progress": _number(size_row.get("sustained_progress")),
        }

    collapse = {
        metric: _difference(values[175][metric], values[180][metric])
        for metric in ("preservation", "sustained_progress")
    }
    recovery = {
        metric: _difference(values[195][metric], values[180][metric])
        for metric in ("preservation", "sustained_progress")
    }
    late_change = {
        metric: _difference(values[200][metric], values[195][metric])
        for metric in ("preservation", "sustained_progress")
    }
    checks: dict[str, bool | None] = {
        "collapse_preservation_drop_ge_0_10": _ge(collapse["preservation"], 0.10),
        "collapse_sustained_progress_drop_ge_0_10": _ge(collapse["sustained_progress"], 0.10),
        "recovery_preservation_rise_ge_0_10": _ge(recovery["preservation"], 0.10),
        "recovery_sustained_progress_rise_ge_0_10": _ge(recovery["sustained_progress"], 0.10),
        "update200_preservation_no_regression_gt_0_02": _ge(late_change["preservation"], -0.02),
        "update200_sustained_progress_no_regression_gt_0_02": _ge(
            late_change["sustained_progress"], -0.02
        ),
    }
    failed, undefined = _check_lists(checks)
    return {
        "role": role,
        "values_by_update": {str(update): values[update] for update in _UPDATES},
        "deltas": {
            "collapse_drop_175_to_180": collapse,
            "recovery_rise_180_to_195": recovery,
            "change_195_to_200": late_change,
        },
        "thresholds": dict(_PHENOTYPE_THRESHOLDS),
        "checks": checks,
        "qualified": _conjunction(checks),
        "failed_thresholds": failed,
        "undefined_thresholds": undefined,
    }


def _scope_summary(credit_rows: Mapping[int, Any], scope_key: str) -> dict[str, Any]:
    values: dict[int, dict[str, Any]] = {}
    for update in _UPDATES:
        row = credit_rows.get(update)
        if scope_key == "overall":
            metric_row = row.get("overall") if isinstance(row, Mapping) else None
        else:
            groups = row.get("groups") if isinstance(row, Mapping) else None
            metric_row = groups.get(scope_key) if isinstance(groups, Mapping) else None
        if not isinstance(metric_row, Mapping):
            metric_row = {}
        row_finite = metric_row.get("finite")
        valid_metrics = row_finite is not False
        values[update] = {
            "C_parallel": _number(metric_row.get("C_parallel")),
            "cosine": _number(metric_row.get("cosine")),
            "C_miss": _number(metric_row.get("C_miss")),
            "norm_ratio": _number(metric_row.get("norm_ratio")),
            "norm8": _number(metric_row.get("norm8")),
            "norm64": _number(metric_row.get("norm64")),
            "dot": _number(metric_row.get("dot")),
            "finite": row_finite if isinstance(row_finite, bool) else None,
            "finite8": metric_row.get("finite8") if isinstance(metric_row.get("finite8"), bool) else None,
            "finite64": metric_row.get("finite64") if isinstance(metric_row.get("finite64"), bool) else None,
        }
        if not valid_metrics:
            values[update]["C_parallel"] = None
            values[update]["cosine"] = None

    collapse = {
        metric: _difference(values[175][metric], values[180][metric])
        for metric in ("C_parallel", "cosine")
    }
    recovery = {
        metric: _difference(values[195][metric], values[180][metric])
        for metric in ("C_parallel", "cosine")
    }
    late_change = {
        metric: _difference(values[200][metric], values[195][metric])
        for metric in ("C_parallel", "cosine")
    }
    checks: dict[str, bool | None] = {
        "collapse_C_parallel_drop_ge_0_05": _ge(collapse["C_parallel"], 0.05),
        "collapse_cosine_drop_ge_0_10": _ge(collapse["cosine"], 0.10),
        "recovery_C_parallel_rise_ge_0_05": _ge(recovery["C_parallel"], 0.05),
        "recovery_cosine_rise_ge_0_10": _ge(recovery["cosine"], 0.10),
        "update200_C_parallel_no_regression_gt_0_02": _ge(late_change["C_parallel"], -0.02),
        "update200_cosine_no_regression_gt_0_05": _ge(late_change["cosine"], -0.05),
    }
    failed, undefined = _check_lists(checks)
    return {
        "values_by_update": {str(update): values[update] for update in _UPDATES},
        "deltas": {
            "collapse_drop_175_to_180": collapse,
            "recovery_rise_180_to_195": recovery,
            "change_195_to_200": late_change,
        },
        "thresholds": dict(_CREDIT_THRESHOLDS),
        "checks": checks,
        "qualified": _conjunction(checks),
        "failed_thresholds": failed,
        "undefined_thresholds": undefined,
    }


def _descriptive_scope(credit_rows: Mapping[int, Any], scope_key: str) -> dict[str, Any]:
    values: dict[int, dict[str, float | None]] = {}
    for update in _UPDATES:
        row = credit_rows.get(update)
        groups = row.get("groups") if isinstance(row, Mapping) else None
        metric_row = groups.get(scope_key) if isinstance(groups, Mapping) else None
        if not isinstance(metric_row, Mapping):
            metric_row = {}
        row_finite = metric_row.get("finite")
        values[update] = {
            "C_parallel": _number(metric_row.get("C_parallel")),
            "cosine": _number(metric_row.get("cosine")),
            "C_miss": _number(metric_row.get("C_miss")),
            "norm_ratio": _number(metric_row.get("norm_ratio")),
            "norm8": _number(metric_row.get("norm8")),
            "norm64": _number(metric_row.get("norm64")),
            "dot": _number(metric_row.get("dot")),
            "finite": row_finite if isinstance(row_finite, bool) else None,
        }
        if row_finite is False:
            values[update]["C_parallel"] = None
            values[update]["cosine"] = None
    return {
        "qualification_evaluated": False,
        "values_by_update": {str(update): values[update] for update in _UPDATES},
    }


def formation_verdict(
    credit_rows: Mapping[Any, Any] | None,
    phenotype_rows: Mapping[Any, Any] | None,
) -> dict[str, Any]:
    """Evaluate the predeclared size-32 phenotype and gradient-synchrony gates.

    The result is a finite-context diagnostic association. It does not establish
    causality, literal gradient faithfulness, or population-level generalization.
    """
    credits = _as_update_map(credit_rows)
    phenotypes = _as_update_map(phenotype_rows)
    primary = _phenotype_size_summary(phenotypes, "32", role="primary")
    secondary64 = _phenotype_size_summary(
        phenotypes, "64", role="secondary_scale_check"
    )
    scopes = {
        "overall": _scope_summary(credits, "overall"),
        "F": _scope_summary(credits, "F"),
        "Q": _scope_summary(credits, "Q"),
    }
    descriptive = {
        "E": _descriptive_scope(credits, "E"),
        "R": _descriptive_scope(credits, "R"),
        "recurrent": _descriptive_scope(credits, "recurrent"),
    }
    qualified_scopes = [name for name, summary in scopes.items()
                        if summary["qualified"] is True]
    phenotype_qualified = primary["qualified"] is True

    if not phenotype_qualified:
        verdict = "PHENOTYPE_UNQUALIFIED"
    elif scopes["overall"]["qualified"] is True:
        verdict = "STRONG_SYNCHRONY"
    elif scopes["F"]["qualified"] is True or scopes["Q"]["qualified"] is True:
        verdict = "MODULE_SYNCHRONY"
    else:
        verdict = "CREDIT_SYNCHRONY_NOT_SUPPORTED"

    return {
        "schema": "credit_formation_verdict_v1",
        "verdict": verdict,
        "primary_size": "32",
        "sequence": list(_UPDATES),
        "phenotype": {
            "primary_size": "32",
            "primary": primary,
            "secondary_size64": secondary64,
            "qualified": primary["qualified"],
        },
        "thresholds": {
            "phenotype_primary": dict(_PHENOTYPE_THRESHOLDS),
            "gradient_scope": dict(_CREDIT_THRESHOLDS),
        },
        "scopes": scopes,
        "descriptive_scopes": descriptive,
        "qualified_scopes": qualified_scopes,
        "interpretation": (
            "Diagnostic synchronous association on the specified finite sequence; "
            "does not establish causality, literal gradient faithfulness, or a population claim."
        ),
    }


def check() -> dict[str, Any]:
    """Run a tiny pure-vector sanity check; no model or device code is imported."""
    names = ["encoder.weight"]
    shapes = [[2]]
    aligned = summarize_pair([1.0, 10.0], [1.0, 0.0], names, shapes)
    metrics = aligned["overall"]
    if metrics["C_parallel"] != 1.0 or metrics["C_miss"] != 10.0:
        raise AssertionError("C_parallel must not hide the orthogonal residual")
    zero = summarize_pair([0.0, 0.0], [0.0, 0.0], names, shapes)["overall"]
    if zero["zero8"] is not True or zero["zero64"] is not True:
        raise AssertionError("zero-vector flags are incorrect")
    if any(zero[key] is not None for key in ("cosine", "C_parallel", "C_miss", "norm_ratio")):
        raise AssertionError("zero-denominator metrics must be undefined")
    if zero["dot"] != 0.0 or zero["finite"] is not True:
        raise AssertionError("zero-vector finite and dot diagnostics are incorrect")
    return {"status": "pass", "orthogonal_residual_case": metrics, "zero_case": zero}
