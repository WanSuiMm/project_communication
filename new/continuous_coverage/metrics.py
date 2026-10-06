"""Compact dense phenotype summaries and aggregate paired training blocks."""
from __future__ import annotations

import math
from typing import Any


SIZES = ("32", "64")
ENDPOINTS = ("64", "128", "256")
MIN_REFERENCE_MAPS = 16
MIN_REFERENCE_PIXELS = 100
EXPECTED_BLOCKS = 8
STRICT_THRESHOLD = 0.80
RETENTION_THRESHOLD = 0.95
WILSON_Z_95 = 1.959963984540054


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _count(value: Any) -> int | None:
    result = _number(value)
    if result is None or result < 0 or result != math.floor(result):
        return None
    return int(result)


def _ratio(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"numerator": None, "denominator": None, "value": _number(value)}
    numerator = _count(value.get("numerator"))
    denominator = _count(value.get("denominator"))
    rate = _number(value.get("value"))
    if rate is None and numerator is not None and denominator not in (None, 0):
        rate = numerator / denominator
    return {"numerator": numerator, "denominator": denominator, "value": rate}


def compact_pair_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """Extract both sizes' fixed-time metrics and any available full frontier.

    Undefined rates, denominators, and frontier records stay ``None``. The
    unchanged phenotype gate is reported separately from the joint endpoint.
    """
    if summary.get("schema") == "continuous-coverage-metrics-v1":
        return summary
    sizes: dict[str, Any] = {}
    for size in SIZES:
        row = summary["sizes"][size]
        endpoints = row["endpoints"]
        strict = endpoints["64"]["strict_16_32"]
        pooled_strict = _ratio(strict["pooled_coverage"])
        strict_mean = _number(strict.get("mean_map_coverage"))
        transition = row["transitions"]["all_changed"]["64_to_256"]
        pooled_transition = transition["pooled"]
        retention = _ratio(pooled_transition["retention"])
        reference_pixels = _count(pooled_transition.get("from_correct"))
        reference_maps = _count(transition.get("eligible_maps_for_retention"))

        coverage: dict[str, Any] = {}
        for time in ENDPOINTS:
            measured = endpoints[time]["all_changed"]
            pooled = _ratio(measured["pooled_coverage"])
            coverage[time] = {"mean_map": _number(measured.get("mean_map_coverage")),
                "pooled": pooled["value"], "correct_pixels": _count(measured.get("correct_pixels")),
                "pixels": _count(measured.get("pixels")), "eligible_maps": _count(measured.get("eligible_maps"))}
        coverage_gain = None
        if coverage["64"]["pooled"] is not None and coverage["256"]["pooled"] is not None:
            coverage_gain = coverage["256"]["pooled"] - coverage["64"]["pooled"]

        frontier = row.get("frontier") if isinstance(row.get("frontier"), dict) else None
        survival = row.get("continuous_survival64_to256")
        regression = _ratio(row.get("ever_regressed_over_ever_correct"))
        sizes[size] = {
            "R_strict_pooled_T64": pooled_strict["value"],
            "R_strict_mean_T64": strict_mean,
            "S_retention64_to256": retention["value"],
            "S_retention_numerator": retention["numerator"],
            "S_continuous_survival64_to256": _ratio(survival)["value"] if survival is not None else None,
            "retention_reference_pixels": reference_pixels,
            "retention_reference_maps": reference_maps,
            "ever_regressed_fraction": regression["value"],
            "ever_regressed_numerator": regression["numerator"],
            "ever_regressed_denominator": regression["denominator"],
            "coverage_gain64_to256": coverage_gain,
            "coverage": coverage,
            "frontier_full": frontier,
            "frontier_status": "full" if frontier is not None else "not_evaluated",
            "frontier_effect": _number(frontier.get("mean_map_weighted_difference")) if frontier is not None else None,
        }

    gate = summary.get("phenotype_gate")
    full_pass = gate.get("pass") if isinstance(gate, dict) else None
    return {
        "schema": "continuous-coverage-metrics-v1",
        "full_pass": full_pass if isinstance(full_pass, bool) else None,
        "sizes": sizes,
    }


def joint_readiness(summary: dict[str, Any], size: int = 32) -> dict[str, Any]:
    """Apply J at one size; missing or under-supported data is not ready."""
    metrics = compact_pair_metrics(summary)
    row = metrics["sizes"][str(size)]
    support_maps = row["retention_reference_maps"]
    support_pixels = row["retention_reference_pixels"]
    checks = {
        "strict_T64_mean_ge_0_80": row["R_strict_mean_T64"] is not None and row["R_strict_mean_T64"] >= STRICT_THRESHOLD,
        "strict_T64_pooled_ge_0_80": row["R_strict_pooled_T64"] is not None and row["R_strict_pooled_T64"] >= STRICT_THRESHOLD,
        "retention64_to256_ge_0_95": row["S_retention64_to256"] is not None and row["S_retention64_to256"] >= RETENTION_THRESHOLD,
        "reference_maps_at_least_16": support_maps is not None and support_maps >= MIN_REFERENCE_MAPS,
        "reference_pixels_at_least_100": support_pixels is not None and support_pixels >= MIN_REFERENCE_PIXELS,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "size": int(size),
        "pass": all(checks.values()),
        "checks": checks,
        "support": {"T64_all_changed_reference_maps": support_maps,
            "T64_all_changed_paired_correct_cells": support_pixels},
        "reasons": reasons,
        "criterion_boundary": "J does not require coverage gain, frontier, or the unchanged Full gate.",
    }


def wilson_interval(successes: int, trials: int, z: float = WILSON_Z_95) -> dict[str, float | int]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Require 0 <= successes <= trials")
    if trials == 0:
        return {"successes": successes, "trials": trials, "lower": 0.0, "upper": 1.0}
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denom
    return {"successes": successes, "trials": trials, "lower": max(0.0, center - radius),
            "upper": min(1.0, center + radius)}


def exact_two_sided_binomial_p(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p value over non-tied paired blocks."""
    discordant = wins + losses
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses cannot be negative")
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant)


def _delta_summary(values: list[float | None]) -> dict[str, Any]:
    observed = [value for value in values if value is not None]
    wins = sum(value > 0 for value in observed)
    losses = sum(value < 0 for value in observed)
    return {
        "n_defined": len(observed), "n_undefined": len(values) - len(observed),
        "mean_continuous_minus_reset": sum(observed) / len(observed) if observed else None,
        "wins": wins, "losses": losses, "ties": len(observed) - wins - losses,
        "exact_two_sided_sign_test_p": exact_two_sided_binomial_p(wins, losses),
    }


def aggregate_pairs(final_rows: list[dict[str, Any]], expected_blocks: int = EXPECTED_BLOCKS) -> dict[str, Any]:
    """Compare reset64x4 with continuous256 at fixed u300 by paired block."""
    by_block: dict[Any, dict[str, dict[str, Any]]] = {}
    for record in final_rows:
        if record.get("update") != 300:
            continue
        arm = record.get("arm")
        if arm not in ("reset64x4", "continuous256"):
            raise ValueError(f"unexpected final arm: {arm!r}")
        block = record["block"]
        arms = by_block.setdefault(block, {})
        if arm in arms:
            raise ValueError(f"duplicate u300 row for block {block}, arm {arm}")
        arms[arm] = record

    blocks, per_block = list(range(expected_blocks)), []
    deltas = {key: [] for key in ("R_strict_mean_T64", "R_strict_pooled_T64",
                                  "S_retention64_to256", "S_continuous_survival64_to256")}
    wins = losses = 0
    reset_passes = continuous_passes = complete_pairs = 0
    for block in blocks:
        arms = by_block.get(block, {})
        reset_row = arms.get("reset64x4")
        continuous_row = arms.get("continuous256")
        if reset_row is None or continuous_row is None:
            per_block.append({"block": block, "status": "missing_arm"})
            continue
        complete_pairs += 1
        reset = reset_row["metrics"]["sizes"]["32"]
        continuous = continuous_row["metrics"]["sizes"]["32"]
        supported = all(row.get("retention_reference_maps") is not None
            and row["retention_reference_maps"] >= MIN_REFERENCE_MAPS
            and row.get("retention_reference_pixels") is not None
            and row["retention_reference_pixels"] >= MIN_REFERENCE_PIXELS for row in (reset, continuous))
        block_delta = {}
        for key in deltas:
            before = _number(reset.get(key))
            after = _number(continuous.get(key))
            needs_support = key in ("S_retention64_to256", "S_continuous_survival64_to256")
            delta = after - before if before is not None and after is not None and (supported or not needs_support) else None
            deltas[key].append(delta)
            block_delta[key] = delta

        reset_pass, continuous_pass = bool(reset_row["joint"]["pass"]), bool(continuous_row["joint"]["pass"])
        reset_passes += reset_pass
        continuous_passes += continuous_pass
        if continuous_pass and not reset_pass:
            wins += 1
            comparison = "continuous_only"
        elif reset_pass and not continuous_pass:
            losses += 1
            comparison = "reset_only"
        elif reset_pass:
            comparison = "both_pass"
        else:
            comparison = "neither_pass"
        detail_keys = (*deltas, "retention_reference_pixels", "retention_reference_maps")
        per_block.append({"block": block, "status": "paired", "both_arms_have_reference_support": supported,
            "reset": {key: reset.get(key) for key in detail_keys},
            "continuous": {key: continuous.get(key) for key in detail_keys},
            "delta_continuous_minus_reset": block_delta, "joint_comparison": comparison})

    exact_p = exact_two_sided_binomial_p(wins, losses)
    complete = complete_pairs == expected_blocks
    six_wins_zero_losses = complete and wins >= 6 and losses == 0
    net_six_significant = complete and wins - losses >= 6 and exact_p <= 0.05
    qualified = bool(six_wins_zero_losses or net_six_significant) if complete else False
    primary = {"wins": wins, "losses": losses, "net_gain": wins - losses,
        "exact_two_sided_p": exact_p, "complete_pairs": complete_pairs, "expected_blocks": expected_blocks,
        "six_wins_zero_losses": six_wins_zero_losses, "net_at_least_6_and_p_at_most_0_05": net_six_significant,
        "frozen_improvement_qualified": qualified, "reset_passes": reset_passes,
        "continuous_passes": continuous_passes,
        "reset_pass_rate_wilson95": wilson_interval(reset_passes, complete_pairs) if complete_pairs else None,
        "continuous_pass_rate_wilson95": wilson_interval(continuous_passes, complete_pairs) if complete_pairs else None}
    primary_verdict = (
        "INCOMPLETE" if not complete else
        "CONTINUOUS_COVERAGE_RELIABILITY_QUALIFIED" if qualified else
        "NO_CONTINUOUS_COVERAGE_RELIABILITY_QUALIFICATION"
    )
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "primary_verdict": primary_verdict,
        "primary": primary,
        "paired_metrics": {
            "R_strict_mean_T64": _delta_summary(deltas["R_strict_mean_T64"]),
            "R_strict_pooled_T64": _delta_summary(deltas["R_strict_pooled_T64"]),
            "S_retention64_to256": _delta_summary(deltas["S_retention64_to256"]),
            "S_continuous_survival64_to256": _delta_summary(deltas["S_continuous_survival64_to256"]),
        },
        "per_block": per_block,
        "undefined_support_policy": "S and survival deltas require both arms to meet the 16-map and 100-cell T64 reference floors; R is reported whenever its own rates are defined.",
    }
