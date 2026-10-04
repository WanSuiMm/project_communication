"""Small NumPy-only CPU fixtures for state_factorization.metrics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "state_factorization_metrics", Path(__file__).with_name("metrics.py")
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load sibling metrics.py")
metrics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(metrics)


def _raises_value_error(function) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def run_checks() -> dict[str, Any]:
    checks = 0

    # Cohort denominators are fixed when every arm's correctness changes.
    s = np.array([[[1, 1, 0, 0]], [[1, 1, 0, 0]]], dtype=bool)
    f = np.array([[[1, 0, 0, 1]], [[1, 0, 0, 1]]], dtype=bool)
    changed = np.ones_like(s)
    fixed = metrics.cohorts(s, f, changed)
    all_wrong = np.zeros((193, *s.shape), dtype=bool)
    all_right = np.ones_like(all_wrong)
    wrong_summary = metrics.summarize(all_wrong, fixed)
    right_summary = metrics.summarize(all_right, fixed)
    for name in ("keep", "rescue", "prog", "F_only", "frontier"):
        for metric_name in metrics.METRIC_NAMES:
            left = wrong_summary["cohorts"][name][metric_name]
            right = right_summary["cohorts"][name][metric_name]
            assert left["pooled"]["denominator"] == right["pooled"]["denominator"]
            assert [row["denominator"] for row in left["per_map"]] == [
                row["denominator"] for row in right["per_map"]
            ]
    assert wrong_summary["all_changed"]["coverage64"]["pooled"]["denominator"] == 8
    checks += 1

    # Absolute boundaries: t64 is index 0; the sustained suffix includes t241,
    # while a t240-only pulse is excluded. Delayed success also requires t64 wrong.
    s = np.array([[[1, 1, 0]]], dtype=bool)
    f = np.array([[[1, 0, 0]]], dtype=bool)
    fixed = metrics.cohorts(s, f, np.ones_like(s))
    trace = np.zeros((193, 1, 1, 3), dtype=bool)
    trace[0, 0, 0, 0] = True
    trace[177:, 0, 0, 1] = True  # absolute t241..256 inclusive
    trace[176, 0, 0, 2] = True  # absolute t240 only
    result = metrics.summarize(trace, fixed)["cohorts"]
    assert result["keep"]["immediate64"]["pooled"]["numerator"] == 1
    assert result["keep"]["continuous64_256"]["pooled"]["numerator"] == 0
    assert result["rescue"]["delayed_sustained"]["pooled"] == {
        "numerator": 1, "denominator": 1, "value": 1.0,
    }
    assert result["rescue"]["ever_after64"]["pooled"]["numerator"] == 1
    assert result["prog"]["sustained241_256"]["pooled"]["numerator"] == 0
    assert result["prog"]["ever_after64"]["pooled"]["numerator"] == 1
    checks += 1

    # Empty cohorts serialize with null rates and cannot satisfy qualification.
    empty = np.zeros((1, 1, 1), dtype=bool)
    empty_masks = metrics.cohorts(empty, empty, empty)
    empty_trace = np.zeros((193, 1, 1, 1), dtype=bool)
    empty_summary = metrics.summarize(empty_trace, empty_masks)
    empty_gate = metrics.rescue_gate(empty_summary, empty_summary)
    assert empty_summary["cohorts"]["keep"]["continuous64_256"]["pooled"]["value"] is None
    assert not empty_gate["pass"]
    assert empty_gate["status"] == "UNQUALIFIED"
    assert not empty_gate["checks"]["keep_support_ge_16_maps_and_100_cells"]
    assert not empty_gate["checks"]["prog_support_ge_16_maps_and_100_cells"]
    checks += 1

    # Frontier edges are clipped at map borders and require both cells in changed.
    s = np.array([[[1, 1, 1, 0]]], dtype=bool)
    f = np.array([[[1, 0, 1, 0]]], dtype=bool)
    changed = np.array([[[1, 1, 0, 1]]], dtype=bool)
    fixed = metrics.cohorts(s, f, changed)
    assert fixed["keep"][0, 0, 0]
    assert fixed["prog"][0, 0, 3]
    assert not fixed["keep"][0, 0, 2]  # correct but outside changed
    assert not fixed["frontier"][0, 0, 3]  # no wrapping from column 0
    changed[0, 0, 2] = True
    fixed = metrics.cohorts(s, f, changed)
    assert fixed["frontier"][0, 0, 3]  # now column 2 -> 3 is a valid edge
    checks += 1

    # Frozen support and FF contrast pass at the exact 16-map/100-cell minimum.
    s = np.zeros((16, 1, 14), dtype=bool)
    f = np.zeros_like(s)
    s[:, 0, :7] = True
    f[:, 0, :7] = True
    changed = np.ones_like(s)
    fixed = metrics.cohorts(s, f, changed)
    primary = np.zeros((193, 16, 1, 14), dtype=bool)
    primary[:, :, 0, :7] = True  # keep continuously correct
    primary[177:, :, 0, 7:9] = True  # 2/7 prog cells sustained on every map
    ff = np.zeros_like(primary)
    summary = metrics.summarize(primary, fixed)
    ff_summary = metrics.summarize(ff, fixed)
    gate = metrics.rescue_gate(summary, ff_summary)
    assert gate["pass"], gate
    assert gate["observed"]["keep_cells"] == 112
    assert gate["observed"]["prog_cells"] == 112
    checks += 1

    # Shape disagreements and non-Boolean inputs fail at the boundary.
    _raises_value_error(lambda: metrics.cohorts(
        np.zeros((1, 2, 2), dtype=bool),
        np.zeros((1, 2, 1), dtype=bool),
        np.zeros((1, 2, 2), dtype=bool),
    ))
    valid = metrics.cohorts(
        np.zeros((1, 2, 2), dtype=bool),
        np.zeros((1, 2, 2), dtype=bool),
        np.ones((1, 2, 2), dtype=bool),
    )
    mixed_shape = dict(valid)
    mixed_shape["prog"] = np.zeros((1, 2, 1), dtype=bool)
    _raises_value_error(lambda: metrics.summarize(
        np.zeros((193, 1, 2, 2), dtype=bool), mixed_shape
    ))
    _raises_value_error(lambda: metrics.cohorts(
        np.zeros((1, 2, 2), dtype=np.uint8),
        np.zeros((1, 2, 2), dtype=bool),
        np.ones((1, 2, 2), dtype=bool),
    ))
    checks += 1

    # All reported structures are strict JSON values, including empty ratios.
    json.dumps({"summary": wrong_summary, "gate": empty_gate}, allow_nan=False)
    return {"checks": int(checks), "status": "PASS"}


if __name__ == "__main__":
    print(json.dumps(run_checks(), sort_keys=True))
