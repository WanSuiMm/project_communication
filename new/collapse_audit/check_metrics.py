"""Small CPU-only assertions for ``metrics.py``; run this file directly."""

import numpy as np

try:  # Support both ``python check_metrics.py`` and package execution.
    from .metrics import summarize_step, summarize_suffix
except ImportError:  # pragma: no cover - exercised by direct script execution.
    from metrics import summarize_step, summarize_suffix


def _test_suffix_metrics() -> None:
    shape = (2, 1, 4)
    changed = np.zeros(shape, dtype=bool)
    distance = np.zeros(shape, dtype=np.int64)
    baseline = np.zeros(shape, dtype=bool)
    changed[0, 0, :3] = True
    changed[1, 0, 0] = True
    distance[0, 0, :3] = (16, 17, 20)
    distance[1, 0, 0] = 31
    baseline[0, 0, :2] = True

    correct = np.zeros((193, *shape), dtype=bool)
    # A producer-correct cell can retain terminal correctness while failing
    # continuity at an intermediate suffix state.
    correct[0, 0, 0, 0] = True
    correct[1, 0, 0, 0] = True
    correct[192, 0, 0, 0] = True
    correct[10, 0, 0, 0] = False
    # The second producer-correct cell loses correctness at the handoff, then
    # returns briefly, but is wrong at the terminal endpoint.
    correct[1, 0, 0, 1] = True
    # A producer-wrong cell loses same-view correctness at the first step,
    # then becomes correct and stays correct through the final sixteen states.
    correct[0, 0, 0, 2] = True
    correct[2:, 0, 0, 2] = True
    correct[-16:, 0, 0, 2] = True
    # Another producer-wrong cell is acquired after the handoff, but its T64
    # endpoint remains wrong.
    correct[1:, 1, 0, 0] = True
    correct[-16:, 1, 0, 0] = True

    summary = summarize_suffix(correct, baseline, changed, distance)
    all_changed = summary["all_changed"]
    assert all_changed["support"]["count"] == 4
    assert [row["count"] for row in all_changed["support"]["maps"]] == [3, 1]
    assert all_changed["producer_baseline_correct_reference"]["count"] == 2
    assert [row["count"] for row in all_changed["producer_baseline_correct_reference"]["maps"]] == [2, 0]

    # T64 coverage reports both pooled-pixel and equal-map means: map 0 has 2/3
    # correct, map 1 has 0/1, hence pooled=1/2 and mapmean=1/3.
    t64 = all_changed["endpoint_coverage"]["T64"]
    assert t64["pooled"] == {"numerator": 2, "denominator": 4, "value": 0.5}
    assert t64["mapmean"]["denominator"] == 2
    assert np.isclose(t64["mapmean"]["value"], 1 / 3)

    # Endpoint success is not the same as continuous preservation.
    assert all_changed["terminal_retention"]["pooled"]["value"] == 0.5
    assert all_changed["continuous_preservation"]["pooled"]["value"] == 0.0
    # baseline is the producer reference, even where a supplied readout's T64
    # correctness disagrees with it.
    assert all_changed["handoff_destruction"]["pooled"]["numerator"] == 1
    assert all_changed["handoff_acquisition"]["pooled"]["numerator"] == 1
    assert all_changed["same_view_first_step_destruction"]["pooled"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    assert all_changed["progress_terminal"]["pooled"]["value"] == 1.0
    assert all_changed["progress_ever"]["pooled"]["value"] == 1.0
    assert all_changed["sustained_progress_last16"]["pooled"]["value"] == 1.0

    strict = summary["strict_16_32"]
    assert strict["support"]["count"] == 3
    assert [row["count"] for row in strict["support"]["maps"]] == [2, 1]
    assert strict["producer_baseline_correct_reference"]["count"] == 1


def _test_empty_denominators_and_step_metrics() -> None:
    before = np.asarray([[[True, False]], [[True, False]]], dtype=bool)
    after = np.asarray([[[False, True]], [[True, False]]], dtype=bool)
    changed = np.asarray([[[True, True]], [[True, False]]], dtype=bool)
    distance = np.asarray([[[16, 16]], [[32, 32]]], dtype=np.int64)

    summary = summarize_step(before, after, changed, distance)
    all_changed = summary["all_changed"]
    assert all_changed["support"]["count"] == 3
    assert all_changed["net_coverage_change"]["pooled"] == {
        "numerator": 0,
        "denominator": 3,
        "value": 0.0,
    }
    assert all_changed["acquisition"]["pooled"] == {
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
    }
    assert all_changed["destruction"]["pooled"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }

    # Strict distance endpoints 16 and 32 are excluded, leaving no support;
    # every undefined rate keeps its numerator and denominator and uses None.
    empty_strict = summary["strict_16_32"]
    assert empty_strict["support"]["count"] == 0
    assert empty_strict["net_coverage_change"]["pooled"] == {
        "numerator": 0,
        "denominator": 0,
        "value": None,
    }
    assert empty_strict["acquisition"]["pooled"] == {
        "numerator": 0,
        "denominator": 0,
        "value": None,
    }
    assert empty_strict["destruction"]["pooled"]["value"] is None


if __name__ == "__main__":
    _test_suffix_metrics()
    _test_empty_denominators_and_step_metrics()
    print("collapse audit metrics: checks passed")
