"""Small synthetic checks for frontier_audit.metrics (NumPy only)."""
from __future__ import annotations

import numpy as np

from metrics import analyze_trace, open_correct_neighbor


TIMES = tuple(range(8, 257, 8))


def _trace(changed: np.ndarray, opened: np.ndarray, distances: np.ndarray):
    correct = np.zeros((257, 1, *changed.shape), dtype=bool)
    margin = np.zeros(correct.shape, dtype=np.float32)
    data = {
        "changed": changed[None, None].astype(np.uint8),
        "mask": opened[None, None].astype(np.uint8),
        "distance": distances[None, None].astype(np.int32),
    }
    return correct, margin, data


def _exact(rows, distance: int):
    return next(row for row in rows if row["distance"] == distance)


def run_checks() -> dict[str, int]:
    checked = 0

    # Monotone arrival has no correctness-loss event or post-arrival relapse.
    changed = np.array([[1, 1, 1]], dtype=bool)
    opened = np.ones_like(changed)
    distances = np.array([[0, 1, 2]], dtype=np.int32)
    correct, margin, data = _trace(changed, opened, distances)
    correct[:, 0, 0, 0] = True
    correct[8:, 0, 0, 1] = True
    correct[16:, 0, 0, 2] = True
    result = analyze_trace(correct, margin, data, TIMES)
    row = result["maps"][0]
    assert row["loss_events"] == 0 and row["ever_relapse_pixels"] == 0
    assert _exact(row["correctness"]["exact_distance"], 1)["first_correct_time"]["median"] == 8.0
    checked += 1

    # First arrival and terminal-stable arrival differ after a temporary loss;
    # the never-correct pixel is right-censored and is not assigned time 257.
    changed = np.array([[1, 1, 1, 0]], dtype=bool)
    opened = np.array([[1, 1, 1, 0]], dtype=bool)
    distances = np.array([[0, 1, 2, -1]], dtype=np.int32)
    correct, margin, data = _trace(changed, opened, distances)
    correct[5, 0, 0, 1] = True
    correct[10:, 0, 0, 1] = True
    margin[5, 0, 0, 1] = np.float32(0.05)
    margin[11:, 0, 0, 1] = np.float32(0.2)
    # A correct wall/outside pixel is excluded from changed-component counts.
    correct[:, 0, 0, 3] = True
    result = analyze_trace(correct, margin, data, TIMES)
    row = result["maps"][0]
    d1 = _exact(row["correctness"]["exact_distance"], 1)
    d2 = _exact(row["correctness"]["exact_distance"], 2)
    assert d1["first_correct_time"] == {"n": 1, "sum": 5, "median": 5.0}
    assert d1["terminal_stable_time"] == {"n": 1, "sum": 10, "median": 10.0}
    assert d1["confident_first_correct"]["median"] == 11.0
    assert d2["never_correct_pixels"] == 1 and d2["first_correct_time"]["n"] == 0
    assert d2["first_correct_time"]["sum"] is None
    assert row["loss_events"] == 1 and row["ever_relapse_pixels"] == 1
    assert row["changed_pixels"] == 3
    at_zero = next(r for r in row["frontier"] if r["from_step"] == 0)
    d1_frontier = _exact(at_zero["exact_distance"], 1)
    assert d1_frontier["frontier_opportunities"] == 0
    assert d1_frontier["nonfrontier_opportunities"] == 1
    assert max(d1["first_correct_time"]["sum"], d1["terminal_stable_time"]["sum"]) <= 256
    checked += 1

    # Open-neighbor logic blocks walls and never wraps across image borders.
    good = np.zeros((2, 3), dtype=bool)
    mask = np.ones_like(good)
    good[0, 1] = True
    mask[0, 1] = False
    adjacent = open_correct_neighbor(good, mask)
    assert not adjacent[0, 0]
    good[:] = False
    mask[:] = True
    good[0, 0] = True
    adjacent = open_correct_neighbor(good, mask)
    assert not adjacent[0, 2]
    checked += 1

    # Zero denominators and empty acquisition sets stay null in JSON output.
    changed = np.array([[1]], dtype=bool)
    correct, margin, data = _trace(changed, changed, np.array([[0]], dtype=np.int32))
    result = analyze_trace(correct, margin, data, TIMES)
    row = result["maps"][0]
    first_pair = row["transitions"][0]
    assert first_pair["from_step"] == 0 and first_pair["to_step"] == 8
    assert first_pair["retention"] == {"numerator": 0, "denominator": 0, "value": None}
    assert first_pair["relapse"]["value"] is None
    frontier_at_zero = row["frontier"][0]["distance_matched_weighted_difference"]
    assert frontier_at_zero["value"] is None and frontier_at_zero["strata_used"] == 0
    empty = row["correctness"]["all_changed"]
    assert empty["first_correct_time"]["sum"] is None
    assert empty["terminal_stable_time"]["median"] is None
    assert empty["never_correct_pixels"] == 1
    checked += 1

    # Same-distance matching keeps frontier and nonfrontier denominators
    # together; unmatched distance strata cannot enter the weighted contrast.
    opened = np.zeros((5, 5), dtype=bool)
    opened[2, 2] = True  # source, distance 0
    opened[2, 3] = True  # correct one-hop neighbor, distance 1
    opened[2, 4] = True  # frontier opportunity, distance 2
    opened[3, 2] = True  # wrong one-hop neighbor, distance 1
    opened[4, 2] = True  # nonfrontier opportunity, distance 2
    distances = np.full((5, 5), -1, dtype=np.int32)
    distances[2, 2], distances[2, 3], distances[2, 4] = 0, 1, 2
    distances[3, 2], distances[4, 2] = 1, 2
    correct, margin, data = _trace(opened.copy(), opened, distances)
    correct[:, 0, 2, 2] = True
    correct[:, 0, 2, 3] = True
    correct[1:, 0, 2, 4] = True
    result = analyze_trace(correct, margin, data, TIMES)
    interval = result["maps"][0]["frontier"][0]
    d2 = _exact(interval["exact_distance"], 2)
    matched = interval["distance_matched_weighted_difference"]
    assert (d2["frontier_opportunities"], d2["frontier_acquired"]) == (1, 1)
    assert (d2["nonfrontier_opportunities"], d2["nonfrontier_acquired"]) == (1, 0)
    assert d2["matched_weight"] == 0.5 and d2["matched_rate_difference"] == 1.0
    assert matched == {"value": 1.0, "weight_sum": 0.5, "strata_used": 1}
    d1 = _exact(interval["exact_distance"], 1)
    assert d1["matched_rate_difference"] is None
    checked += 1

    return {"checks": checked, "status": "PASS"}


if __name__ == "__main__":
    print(run_checks())

