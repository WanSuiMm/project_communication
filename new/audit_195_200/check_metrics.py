"""Hand-calculated CPU fixture for :mod:`metrics`.

Run from this directory with ``python check_metrics.py``. This is a small
contract check, not a test of experiment artifacts or inferential validity.
"""
from __future__ import annotations

import numpy as np

from metrics import summarize


def _equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def _close(actual, expected, message: str, atol: float = 1e-12) -> None:
    if actual is None or not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def main() -> None:
    # Time indices 0, 64, and 192 correspond to macro-steps 64, 128, and 256.
    correct = np.zeros((193, 2, 1, 4), dtype=bool)
    correct[:, 0, 0, 0] = True  # Source pixel: must never enter the domain.
    correct[1:, 0, 0, 1] = True  # Wrong at 64, acquired by 128 and retained.
    correct[:, 0, 0, 2] = True
    correct[10, 0, 0, 2] = False  # Temporary exit, then return before 128.
    correct[:192, 1, 0, 1] = True
    correct[21:30, 1, 0, 1] = False  # Another temporary exit, then return.
    correct[192, 1, 0, 1] = False  # Endpoint destruction at 256.

    changed = np.ones((2, 1, 4), dtype=bool)
    changed[1, 0, 2] = False  # Unchanged despite its positive distance.
    valid = np.ones_like(changed)
    valid[0, 0, 3] = False  # Changed but invalid.
    source = np.zeros_like(changed)
    source[:, 0, 0] = True
    distance = np.array([[[0, 1, 20, 5]], [[0, 33, 20, -1]]], dtype=np.int64)

    margin = np.broadcast_to(0.1 * np.arange(193)[:, None, None, None], correct.shape).copy()
    margin[0, 0, 0, 1] = 0.0  # Correctness comes from the saved Boolean, not this tie.
    trace = {
        "correct": correct,
        "margin": margin,
        "original_margin": margin + 1.0,
        "flipped_margin": margin - 0.25,
    }
    bank = {"changed": changed, "mask": valid, "distance": distance, "source": source}
    cohorts = {
        "core": np.zeros_like(changed),
        "empty": np.zeros_like(changed),
    }
    cohorts["core"][0, 0, 1] = True

    result = summarize(trace, bank, cohorts)
    _equal(result["domains"]["primary"]["pixels_per_map"], [2, 1], "primary valid non-source counts")
    _equal(result["domains"]["strict"]["pixels_per_map"], [1, 0], "strict counts")

    endpoint64 = result["endpoints"]["64"]["primary"]
    _equal(endpoint64["pooled"]["numerator"], 2, "step-64 correct count")
    _equal(endpoint64["pooled"]["denominator"], 3, "step-64 denominator")
    _close(endpoint64["pooled"]["rate"], 2 / 3, "step-64 rate")
    endpoint128 = result["endpoints"]["128"]["primary"]
    _equal(endpoint128["pooled"]["numerator"], 3, "step-128 correct count")
    _equal(endpoint128["pooled"]["denominator"], 3, "step-128 denominator")
    endpoint256 = result["endpoints"]["256"]["primary"]
    _equal(endpoint256["pooled"]["numerator"], 2, "step-256 correct count")
    _equal(endpoint256["pooled"]["denominator"], 3, "step-256 denominator")

    strict64 = result["endpoints"]["64"]["strict"]
    _equal(strict64["per_map"][1]["rate"], None, "empty strict map is undefined")
    _equal(strict64["equal_map_eligible_maps"], 1, "empty strict map excluded from equal-map mean")
    _close(strict64["equal_map_mean"], 1.0, "strict equal-map rate")

    interval_64_128 = result["intervals"]["64_128"]["primary"]
    _equal(interval_64_128["acquisition"]["pooled"]["numerator"], 1, "acquisition count")
    _equal(interval_64_128["acquisition"]["pooled"]["denominator"], 1, "acquisition risk set")
    _equal(interval_64_128["endpoint_destruction"]["pooled"]["numerator"], 0, "128 destruction")
    _equal(interval_64_128["first_exit"]["pooled"]["numerator"], 2, "temporary exits count")
    _equal(interval_64_128["first_exit"]["pooled"]["denominator"], 2, "first-exit risk set")
    _equal(interval_64_128["continuous_survival"]["pooled"]["numerator"], 0, "temporary exits break survival")
    _equal(interval_64_128["all_steps_correct"]["pooled"]["denominator"], 3,
           "unconditional survival keeps all selected pixels in denominator")
    _equal(interval_64_128["all_steps_correct"]["pooled"]["numerator"], 0,
           "unconditional survival includes initially wrong acquired cell")
    _equal(interval_64_128["any_wrong_including_start"]["pooled"]["numerator"], 3,
           "unconditional any-wrong includes wrong-at-start")
    _equal(interval_64_128["turnover"]["pooled"]["correct_to_wrong"], 2, "64-128 repeated correct-to-wrong flips")
    _equal(interval_64_128["turnover"]["pooled"]["wrong_to_correct"], 3, "64-128 repeated wrong-to-correct flips")
    _equal(interval_64_128["turnover"]["pooled"]["total_transitions"], 5, "64-128 total transitions")
    _equal(interval_64_128["identity"]["pooled_accounting_error"], 0, "64-128 net identity")

    interval_64_256 = result["intervals"]["64_256"]["primary"]
    _equal(interval_64_256["endpoint_destruction"]["pooled"]["numerator"], 1, "256 destruction count")
    _equal(interval_64_256["first_exit"]["pooled"]["numerator"], 2, "256 first exits")
    _equal(interval_64_256["continuous_survival"]["pooled"]["numerator"], 0, "256 survival count")
    _equal(interval_64_256["net_gain"]["pooled"]["numerator"], 0, "256 net gain")
    _equal(interval_64_256["turnover"]["pooled"]["correct_to_wrong"], 3, "64-256 repeated correct-to-wrong flips")
    _equal(interval_64_256["turnover"]["pooled"]["wrong_to_correct"], 3, "64-256 repeated wrong-to-correct flips")
    _equal(interval_64_256["turnover"]["per_map"][0]["total_transitions"], 3,
           "64-256 map-zero transition count")
    _equal(interval_64_256["identity"]["pooled_accounting_error"], 0, "64-256 net identity")

    paired = result["paired_margin"]
    _equal(paired["initial_correct_per_map"], [1, 1], "fixed initial-correct paired-margin cohort")
    _equal(paired["sources"]["paired_min"]["endpoints"]["64"]["pooled"]["observations"], 2,
           "endpoint margin cohort size")
    _equal(set(paired["sources"]), {"paired_min", "original", "flipped"}, "paired margin sources")

    core = result["cohorts"]["core"]
    _equal(core["selected_pixels_per_map"], [1, 0], "fixed named cohort population")
    _equal(core["first_exit"]["64_128"]["pooled"]["denominator"], 0, "empty initial-correct cohort risk set")
    _equal(core["first_exit"]["64_128"]["pooled"]["rate"], None, "empty risk set rate is null")
    _equal(core["all_steps_correct"]["64_128"]["pooled"]["denominator"], 1,
           "fixed named cohort unconditional denominator")
    _equal(core["all_steps_correct"]["64_128"]["pooled"]["numerator"], 0,
           "fixed named cohort immediate wrong state is a survival failure")
    _equal(core["any_wrong_including_start"]["64_128"]["pooled"]["numerator"], 1,
           "fixed named cohort includes initial wrong state")
    _close(core["net_margin_change"]["64_128"]["paired_min"]["per_map"][0]["mean"], 6.4,
           "cohort margin change to 128")
    _close(core["net_margin_change"]["64_256"]["paired_min"]["per_map"][0]["mean"], 19.2,
           "cohort margin change to 256")
    empty = result["cohorts"]["empty"]
    _equal(empty["endpoints"]["64"]["pooled"]["rate"], None, "empty cohort endpoint is null")
    _equal(empty["net_margin_change"]["64_128"]["paired_min"]["pooled_descriptive"]["mean"], None,
           "empty cohort margin mean is null")

    print("metrics hand-calculated fixture: PASS")


if __name__ == "__main__":
    main()
