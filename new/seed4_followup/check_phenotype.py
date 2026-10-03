"""Small CPU-only accounting check for seed4 follow-up phenotype summaries."""
from __future__ import annotations

import numpy as np

from phenotype import predicate, summarize_from_traces


def _synthetic_bank():
    changed = np.zeros((2, 1, 3, 3), dtype=np.uint8)
    distance = np.full((2, 1, 3, 3), -1, dtype=np.int32)

    # Map 0 has a source, a one-step frontier acquisition and a same-distance
    # nonfrontier control. It also contains one temporary arrival and one late
    # stable arrival so the passage, relapse, and coverage counts are distinct.
    cells = {
        (1, 1): 0,   # source, correct from step 0
        (1, 2): 17,  # frontier at start 0, correct at step 1
        (0, 0): 17,  # same-distance nonfrontier, never correct
        (0, 2): 18,  # temporary correct interval, then relapse
        (1, 0): 19,  # arrives after T64
    }
    for (y, x), value in cells.items():
        changed[0, 0, y, x] = 1
        distance[0, 0, y, x] = value

    # Map 1 deliberately has no correct pixel and therefore no frontier group.
    changed[1, 0, 2, 2] = 1
    distance[1, 0, 2, 2] = 0
    return {
        "changed": changed,
        "mask": changed.copy(),
        "distance": distance,
    }


def _synthetic_trace():
    correct = np.zeros((257, 2, 3, 3), dtype=bool)
    original = np.zeros_like(correct)
    flipped = np.zeros_like(correct)
    correct[:, 0, 1, 1] = True
    correct[1:, 0, 1, 2] = True
    correct[5:30, 0, 0, 2] = True
    correct[200:, 0, 1, 0] = True
    # The all-wrong second map gives an explicit empty frontier/denominator case.
    original[:] = correct
    flipped[:] = correct
    return {"correct": correct, "original_correct": original, "flipped_correct": flipped}


def _records():
    result = {}
    for size in (32, 64):
        result[str(size)] = {
            str(t): {
                "original": {"balanced_accuracy": 0.9},
                "flipped": {"balanced_accuracy": 0.9},
            }
            for t in (64, 128, 256)
        }
    return result


def run_checks() -> dict[str, int | str]:
    bank = _synthetic_bank()
    summary = summarize_from_traces(
        {"32": _synthetic_trace(), "64": _synthetic_trace()},
        _records(),
        {32: bank, 64: bank},
    )

    size32 = summary["sizes"]["32"]
    # First passage at 5 is distinct from terminal stability: this pixel rolls
    # back by step 30, while the source, step-1 cell, and late cell persist.
    assert size32["per_map"][0]["first_correct"]["observed"] == 4
    assert size32["per_map"][0]["first_correct"]["censored"] == 1
    assert size32["per_map"][0]["ever_regressed"] == {
        "numerator": 1, "denominator": 4, "value": 0.25,
    }
    assert size32["endpoints"]["64"]["all_changed"]["pooled_coverage"] == {
        "numerator": 2, "denominator": 6, "value": 1 / 3,
    }
    assert size32["endpoints"]["256"]["all_changed"]["pooled_coverage"] == {
        "numerator": 3, "denominator": 6, "value": 0.5,
    }
    assert size32["acquisition"]["256"]["newly_correct_vs_step0_pixels"] == 2
    transition = size32["transitions"]["all_changed"]["64_to_256"]["pooled"]
    assert (transition["from_correct"], transition["retained"], transition["gained"]) == (2, 2, 1)

    frontier = size32["frontier"]
    assert frontier["common_strata"] == 1
    assert frontier["eligible_maps"] == 1
    assert frontier["per_map"][0]["weighted_difference"] == 1.0
    assert frontier["per_map"][1]["weighted_difference"] is None
    assert len(summary["_matched_frontier_rows"]) == 2  # one matched row per size

    # Zero-denominator groups cannot pass the frozen gates, and no margin or
    # confidence-threshold statistic enters the independently replayable file.
    gates = predicate(summary)
    assert not gates["pass"]
    assert any("eligible_frontier_maps" in reason for reason in gates["reasons"])
    assert "margin" not in repr(summary["sizes"])

    return {"checks": 5, "status": "PASS"}


if __name__ == "__main__":
    import json
    print(json.dumps(run_checks(), sort_keys=True))

