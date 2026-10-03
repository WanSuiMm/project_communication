"""Map-level, CPU-only behavior summaries for a frozen Streaming trace.

The analyzer intentionally returns per-map summaries. Pixels are descriptive
within-map observations, not independent statistical units.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


TRACE_STEPS = 256
SAMPLE_TIMES = tuple(range(8, TRACE_STEPS + 1, 8))
EXTRA_TRANSITION_PAIRS = ((64, 128), (64, 256), (128, 256))

DISTANCE_BANDS = (
    ("0_8", lambda d: (d >= 0) & (d < 8)),
    ("8_16", lambda d: (d >= 8) & (d < 16)),
    ("equal16", lambda d: d == 16),
    ("strict16_32", lambda d: (d > 16) & (d < 32)),
    ("32_64", lambda d: (d >= 32) & (d < 64)),
    ("64_128", lambda d: (d >= 64) & (d < 128)),
    ("128_inf", lambda d: d >= 128),
)


def _as_binary(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.bool_:
        return arr
    if not np.issubdtype(arr.dtype, np.number) or not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"{name} must be boolean or contain only 0/1")
    return arr.astype(bool, copy=False)


def _map_field(data: dict[str, Any], name: str, shape: tuple[int, int, int]) -> np.ndarray:
    """Read a required [B,1,H,W] field and return [B,H,W]."""
    if name not in data:
        raise ValueError(f"data is missing required field {name!r}")
    arr = np.asarray(data[name])
    if arr.ndim != 4 or arr.shape[1] != 1 or arr.shape[0] != shape[0] or arr.shape[2:] != shape[1:]:
        raise ValueError(f"data[{name!r}] must have shape [B,1,H,W] matching correct")
    arr = arr[:, 0]
    if name in ("changed", "mask"):
        return _as_binary(f"data[{name!r}]", arr)
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.issubdtype(arr.dtype, np.number) or not np.all(np.isfinite(arr)) or not np.all(arr == np.floor(arr)):
            raise ValueError("data['distance'] must contain integer BFS distances")
    return arr.astype(np.int64, copy=False)


def _time_stats(values: np.ndarray) -> dict[str, Any]:
    """Describe observed integer times; an empty set has null sum/median."""
    flat = np.asarray(values).reshape(-1)
    n = int(flat.size)
    if n == 0:
        return {"n": 0, "sum": None, "median": None}
    return {"n": n, "sum": int(flat.sum()), "median": float(np.median(flat))}


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    n, d = int(numerator), int(denominator)
    return {"numerator": n, "denominator": d, "value": (n / d if d else None)}


def _first_time(values: np.ndarray) -> np.ndarray:
    """First true index along time, with -1 for right-censored pixels."""
    seen = np.any(values, axis=0)
    first = np.argmax(values, axis=0).astype(np.int16, copy=False)
    first[~seen] = -1
    return first


def _terminal_stable_time(values: np.ndarray) -> np.ndarray:
    """First step in the final uninterrupted correct suffix, or -1."""
    terminal = values[-1]
    out = np.full(terminal.shape, -1, dtype=np.int16)
    has_error = np.any(~values, axis=0)
    if np.any(terminal & has_error):
        last_error = TRACE_STEPS - np.argmax((~values)[::-1], axis=0)
        out[terminal & has_error] = (last_error[terminal & has_error] + 1).astype(np.int16)
    out[terminal & ~has_error] = 0
    return out


def _correctness_summary(
    selected: np.ndarray,
    first: np.ndarray,
    stable: np.ndarray,
    first_confident: np.ndarray,
    terminal_correct: np.ndarray,
) -> dict[str, Any]:
    n_pixels = int(np.count_nonzero(selected))
    ever = selected & (first >= 0)
    terminal = selected & terminal_correct
    confident = selected & (first_confident >= 0)
    return {
        "pixels": n_pixels,
        "ever_correct_pixels": int(np.count_nonzero(ever)),
        "never_correct_pixels": int(n_pixels - np.count_nonzero(ever)),
        "terminal_correct_pixels": int(np.count_nonzero(terminal)),
        "first_correct_time": _time_stats(first[ever]),
        "terminal_stable_time": _time_stats(stable[terminal]),
        "confident_first_correct": {
            **_time_stats(first_confident[confident]),
            "never_confident_pixels": int(n_pixels - np.count_nonzero(confident)),
        },
    }


def open_correct_neighbor(correct: Any, open_mask: Any) -> np.ndarray:
    """Return cells with a correct, open four-neighbor; edges never wrap.

    Both inputs are 2-D arrays. The returned value is gated by the neighbors'
    open mask; callers decide which current cells are eligible for analysis.
    """
    good = _as_binary("correct", correct)
    opened = _as_binary("open_mask", open_mask)
    if good.ndim != 2 or opened.shape != good.shape:
        raise ValueError("correct and open_mask must be same-shaped 2-D arrays")
    good_open = good & opened
    adjacent = np.zeros(good.shape, dtype=bool)
    if good.shape[0] > 1:
        adjacent[:-1, :] |= good_open[1:, :]
        adjacent[1:, :] |= good_open[:-1, :]
    if good.shape[1] > 1:
        adjacent[:, :-1] |= good_open[:, 1:]
        adjacent[:, 1:] |= good_open[:, :-1]
    return adjacent


def _distance_rows(
    distances: np.ndarray,
    selected: np.ndarray,
    first: np.ndarray,
    stable: np.ndarray,
    first_confident: np.ndarray,
    terminal_correct: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for value in np.unique(distances[selected]):
        band = selected & (distances == value)
        rows.append({
            "distance": int(value),
            **_correctness_summary(band, first, stable, first_confident, terminal_correct),
        })
    return rows


def _band_masks(distances: np.ndarray, changed: np.ndarray) -> dict[str, np.ndarray]:
    masks = {name: changed & predicate(distances) for name, predicate in DISTANCE_BANDS}
    masks["all_changed"] = changed.copy()
    return masks


def _transition(
    correct_map: np.ndarray,
    changed: np.ndarray,
    distances: np.ndarray,
    start: int,
    end: int,
    band_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    before, after = correct_map[start], correct_map[end]
    fromcorrect_mask = changed & before
    fromwrong_mask = changed & ~before
    retained_mask = changed & before & after
    lost_mask = changed & before & ~after
    gained_mask = changed & ~before & after
    fromcorrect = int(np.count_nonzero(fromcorrect_mask))
    fromwrong = int(np.count_nonzero(fromwrong_mask))
    retained = int(np.count_nonzero(retained_mask))
    lost = int(np.count_nonzero(lost_mask))
    gained = int(np.count_nonzero(gained_mask))
    tocorrect = int(np.count_nonzero(changed & after))
    return {
        "from_step": int(start),
        "to_step": int(end),
        "fromcorrect": fromcorrect,
        "retained": retained,
        "lost": lost,
        "fromwrong": fromwrong,
        "gained": gained,
        "net_change": int(tocorrect - fromcorrect),
        "retention": _ratio(retained, fromcorrect),
        "relapse": _ratio(lost, fromcorrect),
        "loss_by_distance": [
            {"band": name, "pixels": int(np.count_nonzero(lost_mask & band_masks[name]))}
            for name, _ in DISTANCE_BANDS
        ] + [{"band": "all_changed", "pixels": lost}],
        "gain_by_distance": [
            {"band": name, "pixels": int(np.count_nonzero(gained_mask & band_masks[name]))}
            for name, _ in DISTANCE_BANDS
        ] + [{"band": "all_changed", "pixels": gained}],
    }


def _frontier_interval(
    correct_map: np.ndarray,
    changed: np.ndarray,
    open_component: np.ndarray,
    distances: np.ndarray,
    start: int,
    end: int,
) -> dict[str, Any]:
    before, after = correct_map[start], correct_map[end]
    wrong = changed & ~before
    acquired = wrong & after
    has_correct_neighbor = open_correct_neighbor(before & open_component, open_component)
    frontier = wrong & has_correct_neighbor
    nonfrontier = wrong & ~has_correct_neighbor
    exact_rows: list[dict[str, Any]] = []
    weighted_terms: list[tuple[float, float]] = []
    for value in np.unique(distances[changed]):
        at_distance = changed & (distances == value)
        f = frontier & at_distance
        n = nonfrontier & at_distance
        nf, nn = int(np.count_nonzero(f)), int(np.count_nonzero(n))
        af, an = int(np.count_nonzero(f & acquired)), int(np.count_nonzero(n & acquired))
        f_rate = _ratio(af, nf)
        n_rate = _ratio(an, nn)
        matched = nf > 0 and nn > 0
        weight = (nf * nn / (nf + nn)) if matched else None
        difference = f_rate["value"] - n_rate["value"] if matched else None
        if matched:
            weighted_terms.append((float(weight), float(difference)))
        exact_rows.append({
            "distance": int(value),
            "frontier_opportunities": nf,
            "frontier_acquired": af,
            "frontier_acquisition_rate": f_rate,
            "nonfrontier_opportunities": nn,
            "nonfrontier_acquired": an,
            "nonfrontier_acquisition_rate": n_rate,
            "matched_weight": (float(weight) if matched else None),
            "matched_rate_difference": (float(difference) if matched else None),
        })
    weight_sum = float(sum(weight for weight, _ in weighted_terms))
    difference = (
        float(sum(weight * value for weight, value in weighted_terms) / weight_sum)
        if weight_sum else None
    )
    return {
        "from_step": int(start),
        "to_step": int(end),
        "exact_distance": exact_rows,
        "distance_matched_weighted_difference": {
            "value": difference,
            "weight_sum": (weight_sum if weight_sum else None),
            "strata_used": int(len(weighted_terms)),
        },
    }


def _overall_frontier(maps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Equal-map descriptive means; never pool pixel opportunities across maps."""
    if not maps:
        return []
    by_pair: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for map_row in maps:
        for row in map_row["frontier"]:
            key = (row["from_step"], row["to_step"])
            by_pair.setdefault(key, []).append({
                "map_index": map_row["map_index"],
                **row["distance_matched_weighted_difference"],
            })
    result = []
    for (start, end), rows in by_pair.items():
        values = [row["value"] for row in rows if row["value"] is not None]
        result.append({
            "from_step": start,
            "to_step": end,
            "map_values": rows,
            "maps_with_matched_strata": int(len(values)),
            "mean_across_maps": (float(np.mean(values)) if values else None),
            "median_across_maps": (float(np.median(values)) if values else None),
        })
    return result


def analyze_trace(
    correct: Any,
    margin: Any,
    data: dict[str, Any],
    sample_times: Iterable[int] = SAMPLE_TIMES,
) -> dict[str, Any]:
    """Summarize a T=256 paired-correctness trace on changed pixels only.

    ``correct`` and ``margin`` are [T+1,B,H,W]. ``data`` must contain binary
    ``changed`` and traversable ``mask`` plus integer BFS ``distance``, each
    [B,1,H,W]. The fixed samples are 8,16,...,256; interval summaries include
    0->8, every adjacent sample pair, and the three frozen long transitions.
    """
    correct_array = _as_binary("correct", correct)
    if correct_array.ndim != 4 or correct_array.shape[0] != TRACE_STEPS + 1:
        raise ValueError("correct must have shape [257,B,H,W] for steps 0..256")
    if not isinstance(data, dict):
        raise ValueError("data must be a dict containing changed, mask, and distance")
    _, maps, height, width = correct_array.shape
    shape = (maps, height, width)
    margins = np.asarray(margin)
    if margins.shape != correct_array.shape or not np.issubdtype(margins.dtype, np.floating):
        raise ValueError("margin must be a floating array with the same shape as correct")
    if not np.all(np.isfinite(margins)):
        raise ValueError("margin contains non-finite values")
    changed = _map_field(data, "changed", shape)
    opened = _map_field(data, "mask", shape)
    distances = _map_field(data, "distance", shape)
    if np.any(changed & ~opened):
        raise ValueError("changed pixels must be traversable in data['mask']")
    if np.any(changed & (distances < 0)):
        raise ValueError("changed pixels must have nonnegative BFS distances")
    samples = tuple(int(v) for v in sample_times)
    if samples != SAMPLE_TIMES:
        raise ValueError("sample_times must be exactly 8,16,...,256 for this frozen audit")

    interval_pairs = tuple(zip((0,) + samples[:-1], samples))
    transition_pairs = tuple(dict.fromkeys(interval_pairs + EXTRA_TRANSITION_PAIRS))
    # A one-hop frontier is matched to the immediately following macro step;
    # eight-step transitions remain a separate retention/acquisition summary.
    frontier_pairs = tuple((t, t + 1) for t in range(0, TRACE_STEPS, 8))
    map_rows: list[dict[str, Any]] = []
    for b in range(maps):
        selected = changed[b]
        correct_map = correct_array[:, b]
        first = _first_time(correct_map)
        stable = _terminal_stable_time(correct_map)
        confident_mask = correct_map & (margins[:, b] > np.float32(0.1))
        first_confident = _first_time(confident_mask)
        terminal = correct_map[-1]
        correctness = {
            "all_changed": _correctness_summary(selected, first, stable, first_confident, terminal),
            "exact_distance": _distance_rows(
                distances[b], selected, first, stable, first_confident, terminal,
            ),
        }
        bands = _band_masks(distances[b], selected)
        transitions = [
            _transition(correct_map, selected, distances[b], start, end, bands)
            for start, end in transition_pairs
        ]
        frontier = [
            _frontier_interval(
                correct_map, selected, opened[b] & selected, distances[b], start, end,
            )
            for start, end in frontier_pairs
        ]
        losses = selected[None, ...] & correct_map[:-1] & ~correct_map[1:]
        loss_events = int(np.count_nonzero(losses))
        has_relapse = np.zeros((height, width), dtype=bool)
        for t in range(TRACE_STEPS + 1):
            has_relapse |= (first >= 0) & (first < t) & ~correct_map[t]
        map_rows.append({
            "map_index": int(b),
            "changed_pixels": int(np.count_nonzero(selected)),
            "correctness": correctness,
            "loss_events": loss_events,
            "ever_relapse_pixels": int(np.count_nonzero(selected & has_relapse)),
            "transitions": transitions,
            "frontier": frontier,
        })

    return {
        "schema_version": "frontier-audit-metrics-v1",
        "scope": "descriptive map-level summaries on the changed component",
        "steps": {"first_observed": 0, "last_observed": TRACE_STEPS, "terminal_stability_horizon": TRACE_STEPS},
        "sample_times": list(samples),
        "transition_pairs": [list(pair) for pair in transition_pairs],
        "frontier_pairs": [list(pair) for pair in frontier_pairs],
        "distance_band_definitions": {
            "0_8": "0 <= d < 8",
            "8_16": "8 <= d < 16",
            "equal16": "d == 16",
            "strict16_32": "16 < d < 32",
            "32_64": "32 <= d < 64",
            "64_128": "64 <= d < 128",
            "128_inf": "d >= 128",
            "all_changed": "all pixels in the changed component",
        },
        "frontier_aggregation": "weighted difference within exact distance, then equal-map descriptive mean; pixels are not pooled across maps",
        "maps": map_rows,
        "frontier_overall_by_interval": _overall_frontier(map_rows),
    }

