"""Small hand-computed CPU fixtures for warmstart.diagnostics."""
from __future__ import annotations

import numpy as np

from diagnostics import summarize_correct_traces


def run_checks() -> dict[str, int | str]:
    checked = 0

    # Growth, overwrite, recovery, and later endpoint retention are distinct.
    # Map 0 has a temporary loss/recovery; map 1 is overwritten twice; map 2
    # has an empty changed component and must not enter equal-map means.
    changed = np.zeros((3, 1, 4), dtype=bool)
    changed[0, 0, :] = True
    changed[1, 0, 0] = True
    correct = np.zeros((5, 3, 1, 4), dtype=bool)
    correct[:, 0, 0, :] = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 1, 0],
    ], dtype=bool)
    correct[:, 1, 0, 0] = np.array([0, 1, 0, 1, 0], dtype=bool)

    result = summarize_correct_traces(correct, changed, survival_start=1, survival_end=4)
    rows = result["steps"]
    assert len(rows) == 5 and [row["t"] for row in rows] == list(range(5))
    assert (rows[1]["acquired"], rows[1]["destroyed"], rows[1]["delta_correct"]) == (2, 0, 2)
    assert (rows[2]["acquired"], rows[2]["destroyed"], rows[2]["delta_correct"]) == (1, 2, -1)
    assert (rows[3]["acquired"], rows[3]["destroyed"], rows[3]["delta_correct"]) == (2, 1, 1)
    assert (rows[4]["acquired"], rows[4]["destroyed"], rows[4]["delta_correct"]) == (1, 1, 0)
    assert all(row["accounting_error"] == 0 for row in rows[1:])
    assert rows[1]["q"]["pooled_correct"] == 3 and rows[1]["q"]["pooled_pixels"] == 5
    assert rows[1]["q"]["equal_map_eligible_maps"] == 2
    assert rows[1]["g_acquisition_rate"]["pooled_numerator"] == 2
    assert rows[1]["g_acquisition_rate"]["pooled_denominator"] == 4
    assert rows[1]["g_acquisition_rate"]["equal_map_eligible_maps"] == 2
    assert rows[2]["d_destruction_rate"]["pooled_numerator"] == 2
    assert rows[2]["d_destruction_rate"]["pooled_denominator"] == 3

    survival = result["survival"]
    assert survival["multi_step_survival"] == {
        "equal_map_mean": 0.0,
        "equal_map_eligible_maps": 2,
        "pooled_correct_at_start_and_every_step": 0,
        "pooled_correct_at_start": 3,
        "pooled_rate": 0.0,
    }
    assert survival["endpoint_retention"]["pooled_correct_at_start_and_endpoint"] == 2
    assert survival["endpoint_retention"]["pooled_correct_at_start"] == 3
    assert survival["endpoint_retention"]["pooled_rate"] == 2 / 3
    assert survival["per_map"][0]["multi_step_survival_rate"] == 0.0
    assert survival["per_map"][0]["endpoint_retention_rate"] == 1.0
    assert survival["per_map"][2]["multi_step_survival_rate"] is None

    coverage = result["coverage_gain"]
    assert (coverage["acquired"], coverage["destroyed"], coverage["delta_correct"]) == (1, 1, 0)
    assert coverage["accounting_error"] == 0
    assert result["relapse"]["ever_correct"] == 4
    assert result["relapse"]["relapsed"] == 3
    assert result["relapse"]["equal_map_rate"]["equal_map_eligible_maps"] == 2
    checked += 1

    # Empty masks preserve zero denominators as JSON nulls rather than NaN or 0.
    empty = np.zeros((2, 1, 2), dtype=bool)
    all_wrong = np.zeros((3, 2, 1, 2), dtype=bool)
    empty_result = summarize_correct_traces(all_wrong, empty, survival_start=0, survival_end=2)
    assert empty_result["steps"][0]["q"] == {
        "equal_map_mean": None,
        "equal_map_eligible_maps": 0,
        "pooled_correct": 0,
        "pooled_pixels": 0,
        "pooled_accuracy": None,
    }
    step = empty_result["steps"][1]
    assert step["g_acquisition_rate"]["equal_map_mean"] is None
    assert step["g_acquisition_rate"]["pooled_denominator"] == 0
    assert step["g_acquisition_rate"]["pooled_rate"] is None
    assert step["d_destruction_rate"]["pooled_rate"] is None
    assert empty_result["survival"]["multi_step_survival"]["pooled_rate"] is None
    assert empty_result["coverage_gain"]["g_acquisition_rate"]["pooled_rate"] is None
    assert empty_result["relapse"]["equal_map_rate"]["pooled_rate"] is None
    checked += 1

    return {"checks": checked, "status": "PASS"}


if __name__ == "__main__":
    print(run_checks())
