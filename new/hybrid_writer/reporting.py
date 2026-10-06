"""Paired fixed-endpoint and checkpoint-grid summaries for the hybrid-writer study."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


ARMS = ("neural", "budget", "hybrid", "affine_hybrid")
EXPECTED_BLOCKS = 8
PRIMARY_COMPARISONS = (
    ("budget_minus_neural", "budget", "neural"),
    ("hybrid_minus_budget", "hybrid", "budget"),
    ("hybrid_minus_neural", "hybrid", "neural"),
)
SECONDARY_COMPARISON = ("affine_minus_hybrid", "affine_hybrid", "hybrid")
CHECKPOINTS = tuple(range(0, 301, 25))
FORMAL_UPDATE = 300
NET_GAIN_THRESHOLD = 6
PRIMARY_ALPHA = 0.05
WILSON_Z_95 = 1.959963984540054
SIZE_COLUMNS = ("32", "64")
MIN_RETENTION_MAPS = 16
MIN_RETENTION_PIXELS = 100

_METRIC_COLUMNS = (
    "R_strict_pooled_T64",
    "R_strict_mean_T64",
    "S_retention64_to256",
    "S_retention_numerator",
    "S_continuous_survival64_to256",
    "retention_reference_pixels",
    "retention_reference_maps",
    "ever_regressed_fraction",
    "ever_regressed_numerator",
    "ever_regressed_denominator",
    "coverage_gain64_to256",
    "frontier_status",
    "frontier_effect",
)


def _number(value: Any) -> float | None:
    """Return a finite number, preserving missing and Boolean values as undefined."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or number != math.floor(number):
        return None
    return int(number)


def _readiness(row: dict[str, Any] | None) -> bool | None:
    if not isinstance(row, dict):
        return None
    joint = row.get("joint")
    if not isinstance(joint, dict):
        return None
    value = joint.get("pass")
    return value if isinstance(value, bool) else None


def exact_two_sided_binomial_p(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p value over discordant paired blocks."""
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses cannot be negative")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    return min(
        1.0,
        2.0 * sum(math.comb(discordant, k) for k in range(tail + 1))
        / 2**discordant,
    )


def wilson_interval(
    successes: int, trials: int, z: float = WILSON_Z_95
) -> dict[str, float | int]:
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
    return {
        "successes": successes,
        "trials": trials,
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p values for the supplied family."""
    count = len(p_values)
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_value))
        adjusted[name] = running
    return adjusted


def _formal_rows(
    final_rows: list[dict[str, Any]], expected_blocks: int
) -> tuple[dict[tuple[int, str], dict[str, Any]], int]:
    by_arm_block: dict[tuple[int, str], dict[str, Any]] = {}
    invalid = 0
    for row in final_rows:
        if not isinstance(row, dict):
            invalid += 1
            continue
        arm = row.get("arm")
        block = _integer(row.get("block"))
        update = _integer(row.get("update"))
        if arm not in ARMS:
            raise ValueError(f"unexpected formal arm: {arm!r}")
        if (
            block is None
            or block not in range(expected_blocks)
            or update != FORMAL_UPDATE
            or row.get("formal_endpoint") is not True
            or _readiness(row) is None
        ):
            invalid += 1
            continue
        key = (block, arm)
        if key in by_arm_block:
            raise ValueError(f"duplicate formal u300 row for block {block}, arm {arm}")
        by_arm_block[key] = row
    return by_arm_block, invalid


def _comparison(
    name: str,
    candidate: str,
    reference: str,
    formal: dict[tuple[int, str], dict[str, Any]],
    expected_blocks: int,
    inferentially_complete: bool,
    primary: bool,
) -> dict[str, Any]:
    wins = losses = both_pass = neither_pass = 0
    per_block: list[dict[str, Any]] = []
    complete_pairs = 0
    for block in range(expected_blocks):
        candidate_row = formal.get((block, candidate))
        reference_row = formal.get((block, reference))
        candidate_pass = _readiness(candidate_row)
        reference_pass = _readiness(reference_row)
        if candidate_pass is None or reference_pass is None:
            per_block.append({"block": block, "status": "missing_or_invalid_formal_row"})
            continue
        complete_pairs += 1
        if candidate_pass and not reference_pass:
            wins += 1
            result = "candidate_only"
        elif reference_pass and not candidate_pass:
            losses += 1
            result = "reference_only"
        elif candidate_pass:
            both_pass += 1
            result = "both_pass"
        else:
            neither_pass += 1
            result = "neither_pass"
        per_block.append(
            {
                "block": block,
                "status": "paired",
                "candidate_ready": candidate_pass,
                "reference_ready": reference_pass,
                "comparison": result,
            }
        )

    complete = inferentially_complete and complete_pairs == expected_blocks
    p_value = exact_two_sided_binomial_p(wins, losses) if complete else None
    return {
        "name": name,
        "role": "primary" if primary else "secondary_capacity",
        "candidate": candidate,
        "reference": reference,
        "expected_pairs": expected_blocks,
        "complete_pairs": complete_pairs,
        "wins": wins,
        "losses": losses,
        "both_pass": both_pass,
        "neither_pass": neither_pass,
        "discordant_pairs": wins + losses,
        "net_gain": wins - losses,
        "exact_two_sided_p": p_value,
        "holm_adjusted_p": None,
        "net_gain_threshold": NET_GAIN_THRESHOLD,
        "net_gain_threshold_met": (wins - losses >= NET_GAIN_THRESHOLD) if complete else None,
        "reliability_qualified": None,
        "verdict": "PENDING",
        "per_block": per_block,
    }


def _checkpoint_index(records: list[dict[str, Any]], expected_blocks: int | None) -> tuple[
    dict[tuple[int, str, int], dict[str, Any]], dict[str, int]
]:
    indexed: dict[tuple[int, str, int], dict[str, Any]] = {}
    counts = {
        "observed_predeclared_records": 0,
        "ignored_non_grid_records": 0,
        "ignored_out_of_range_records": 0,
        "invalid_records": 0,
    }
    checkpoint_set = set(CHECKPOINTS)
    for row in records:
        if not isinstance(row, dict):
            counts["invalid_records"] += 1
            continue
        arm = row.get("arm")
        if arm not in ARMS:
            raise ValueError(f"unexpected checkpoint arm: {arm!r}")
        block = _integer(row.get("block"))
        update = _integer(row.get("update"))
        if (
            block is None
            or block < 0
            or (expected_blocks is not None and block not in range(expected_blocks))
        ):
            counts["ignored_out_of_range_records"] += 1
            continue
        if update is None:
            counts["invalid_records"] += 1
            continue
        if update not in checkpoint_set:
            counts["ignored_non_grid_records"] += 1
            continue
        key = (block, arm, update)
        if key in indexed:
            raise ValueError(
                f"duplicate predeclared checkpoint row for block {block}, arm {arm}, u{update}"
            )
        indexed[key] = row
        counts["observed_predeclared_records"] += 1
    return indexed, counts


def _checkpoint_summary(
    indexed: dict[tuple[int, str, int], dict[str, Any]], expected_blocks: int
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    expected_per_arm = expected_blocks * len(CHECKPOINTS)
    for arm in ARMS:
        complete_paths = []
        ever_ready = 0
        known_nonready = 0
        indeterminate_everready = 0
        first_ready_paths = 0
        readiness_losses = 0
        observed_path_count = 0
        for block in range(expected_blocks):
            path = [
                indexed.get((block, arm, update))
                for update in CHECKPOINTS
            ]
            observed_path_count += sum(row is not None for row in path)
            readiness = [_readiness(row) for row in path]
            has_unknown = any(value is None for value in readiness)
            if any(value is True for value in readiness):
                ever_ready += 1
            elif has_unknown:
                indeterminate_everready += 1
            else:
                known_nonready += 1
            if any(value is None for value in readiness):
                continue
            complete_paths.append(block)
            if any(readiness):
                first = readiness.index(True)
                if first < len(CHECKPOINTS) - 1:
                    first_ready_paths += 1
                    if not all(readiness[first + 1 :]):
                        readiness_losses += 1
        complete_trajectories = len(complete_paths)
        everready_denominator = ever_ready + known_nonready
        summaries[arm] = {
            "expected_checkpoint_records": expected_per_arm,
            "observed_checkpoint_records": observed_path_count,
            "complete_trajectories": complete_trajectories,
            "incomplete_trajectories": expected_blocks - complete_trajectories,
            "everready": {
                "successes": ever_ready,
                "known_nonready": known_nonready,
                "indeterminate_no_pass_trajectories": indeterminate_everready,
                "denominator": everready_denominator,
                "expected_trajectories": expected_blocks,
                "value": (
                    ever_ready / everready_denominator
                    if everready_denominator
                    else None
                ),
                "estimand": (
                    "P(any observed predeclared checkpoint is joint-ready); "
                    "incomplete no-pass trajectories are unresolved and excluded."
                ),
            },
            "readiness_loss_after_first": {
                "losses": readiness_losses,
                "denominator": first_ready_paths,
                "value": (
                    readiness_losses / first_ready_paths
                    if first_ready_paths
                    else None
                ),
                "definition": (
                    "Among complete trajectories first joint-ready before u300, "
                    "fraction with any later predeclared checkpoint not joint-ready."
                ),
            },
        }
    return summaries


def aggregate(
    final_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    expected_blocks: int = 8,
) -> dict[str, Any]:
    """Aggregate fixed-u300 paired reliability and checkpoint-grid diagnostics.

    Formal p values and qualification verdicts are withheld unless every arm has
    a valid formal row for every expected block. Checkpoint diagnostics require
    complete 13-point trajectories; a missing readiness outcome is never coded
    as a failure.
    """
    if expected_blocks <= 0:
        raise ValueError("expected_blocks must be positive")
    formal, invalid_formal_rows = _formal_rows(final_rows, expected_blocks)
    complete_blocks_by_arm = {
        arm: sum((block, arm) in formal for block in range(expected_blocks))
        for arm in ARMS
    }
    complete = all(
        complete_blocks_by_arm[arm] == expected_blocks
        for arm in ARMS
    )
    complete_blocks = sum(
        all((block, arm) in formal for arm in ARMS)
        for block in range(expected_blocks)
    )

    comparisons = {
        name: _comparison(
            name,
            candidate,
            reference,
            formal,
            expected_blocks,
            complete,
            primary=True,
        )
        for name, candidate, reference in PRIMARY_COMPARISONS
    }
    secondary_name, secondary_candidate, secondary_reference = SECONDARY_COMPARISON
    comparisons[secondary_name] = _comparison(
        secondary_name,
        secondary_candidate,
        secondary_reference,
        formal,
        expected_blocks,
        complete,
        primary=False,
    )

    if complete:
        adjusted = _holm_adjust(
            {
                name: comparisons[name]["exact_two_sided_p"]
                for name, _, _ in PRIMARY_COMPARISONS
            }
        )
        for name, _, _ in PRIMARY_COMPARISONS:
            comparison = comparisons[name]
            comparison["holm_adjusted_p"] = adjusted[name]
            qualifies = bool(
                comparison["net_gain"] >= NET_GAIN_THRESHOLD
                and adjusted[name] <= PRIMARY_ALPHA
            )
            comparison["reliability_qualified"] = qualifies
            comparison["verdict"] = (
                "RELIABILITY_QUALIFIED"
                if qualifies
                else "NO_RELIABILITY_QUALIFICATION"
            )
        secondary = comparisons[secondary_name]
        secondary["net_gain_threshold_met"] = None
        secondary["verdict"] = "SECONDARY_DESCRIPTIVE_ONLY"
        overall_verdict = (
            "ANY_PRIMARY_RELIABILITY_QUALIFIED"
            if any(
                comparisons[name]["reliability_qualified"]
                for name, _, _ in PRIMARY_COMPARISONS
            )
            else "NO_PRIMARY_RELIABILITY_QUALIFICATION"
        )
    else:
        overall_verdict = "PENDING"

    per_arm = {}
    for arm in ARMS:
        rows = [
            formal[(block, arm)]
            for block in range(expected_blocks)
            if (block, arm) in formal
        ]
        ready = sum(_readiness(row) is True for row in rows)
        complete_arm = len(rows) == expected_blocks
        per_arm[arm] = {
            "valid_formal_blocks": len(rows),
            "expected_formal_blocks": expected_blocks,
            "formal_readiness_passes": ready,
            "formal_readiness_rate": ready / len(rows) if rows else None,
            "formal_readiness_wilson95": (
                wilson_interval(ready, len(rows)) if complete_arm else None
            ),
            "formal_readiness_ci_status": (
                "COMPLETE_BLOCKS" if complete_arm else "INCOMPLETE_NO_CI"
            ),
        }

    indexed_records, record_counts = _checkpoint_index(records, expected_blocks)
    checkpoint_summaries = _checkpoint_summary(indexed_records, expected_blocks)
    expected_grid_records = expected_blocks * len(ARMS) * len(CHECKPOINTS)
    complete_grid_trajectories = sum(
        checkpoint_summaries[arm]["complete_trajectories"] for arm in ARMS
    )
    expected_grid_trajectories = expected_blocks * len(ARMS)
    return {
        "schema": "hybrid-writer-paired-report-v1",
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "expected_blocks": expected_blocks,
        "arms": list(ARMS),
        "predeclared_checkpoints": list(CHECKPOINTS),
        "formal_update": FORMAL_UPDATE,
        "formal_rows": {
            "valid_rows": len(formal),
            "invalid_rows": invalid_formal_rows,
            "complete_blocks": complete_blocks,
            "expected_complete_blocks": expected_blocks,
            "complete_blocks_by_arm": complete_blocks_by_arm,
            "all_arms_complete": complete,
        },
        "overall_verdict": overall_verdict,
        "primary_verdict": overall_verdict,
        "primary_comparisons": [name for name, _, _ in PRIMARY_COMPARISONS],
        "holm_family_size": len(PRIMARY_COMPARISONS),
        "qualification_rule": (
            "For each primary contrast, candidate-only minus reference-only wins "
            "must be at least 6/8 and its Holm-adjusted exact two-sided discordant "
            "binomial p value must be at most 0.05 across the three primary contrasts."
        ),
        "comparisons": comparisons,
        "per_arm": per_arm,
        "checkpoint_grid": {
            **record_counts,
            "expected_records": expected_grid_records,
            "complete_trajectories": complete_grid_trajectories,
            "expected_trajectories": expected_grid_trajectories,
            "complete": complete_grid_trajectories == expected_grid_trajectories,
            "by_arm": checkpoint_summaries,
        },
        "claim_boundary": (
            "Finite evidence for this K8 recipe only; it does not establish "
            "guaranteed closure, whole-NCA behavior, or 3D transport. The "
            "budget comparison bundles bounds and gates and does not isolate "
            "a mechanism."
        ),
    }


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_metrics_csv(
    path: Path, records: list[dict[str, Any]], expected_blocks: int = EXPECTED_BLOCKS
) -> None:
    fields = (
        "block",
        "initialization_seed",
        "schedule_seed",
        "arm",
        "update",
        "size",
        *_METRIC_COLUMNS,
        "joint_readiness",
        "full_pass",
        "formal_endpoint",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        indexed, _ = _checkpoint_index(records, expected_blocks=expected_blocks)
        for block, arm, update in sorted(
            indexed,
            key=lambda key: (key[0], ARMS.index(key[1]), key[2]),
        ):
            record = indexed[(block, arm, update)]
            metrics = record.get("metrics")
            metrics = metrics if isinstance(metrics, dict) else {}
            sizes = metrics.get("sizes")
            sizes = sizes if isinstance(sizes, dict) else {}
            joint_value = _readiness(record)
            for size in SIZE_COLUMNS:
                size_metrics = sizes.get(size, {})
                size_metrics = size_metrics if isinstance(size_metrics, dict) else {}
                row = {
                    key: record.get(key)
                    for key in (
                        "block",
                        "initialization_seed",
                        "schedule_seed",
                        "arm",
                        "update",
                    )
                }
                row["size"] = size
                row.update({key: size_metrics.get(key) for key in _METRIC_COLUMNS})
                row["joint_readiness"] = joint_value if size == "32" else None
                row["full_pass"] = metrics.get("full_pass")
                row["formal_endpoint"] = record.get("formal_endpoint")
                writer.writerow(row)


def report(
    out: str | Path,
    state: dict[str, Any],
    records: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the human-readable report and compact checkpoint-grid CSV."""
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    summary = aggregate(final_rows, records)
    if state.get("status") == "ERROR":
        summary["overall_verdict"] = summary["primary_verdict"] = "INCOMPLETE"

    lines = [
        "# Hybrid writer paired reliability",
        "",
        f"Execution status: {state.get('status', 'UNKNOWN')}",
        f"Formal aggregation: {summary['status']}",
        f"Overall verdict: {summary['overall_verdict']}",
        "",
        "The formal endpoint is joint readiness at u300. Formation diagnostics use only the predeclared "
        "checkpoint grid u0, u25, ..., u300; no other update or peak checkpoint enters the comparisons.",
        "",
        "| Arm | Valid formal blocks | Expected blocks | Joint-ready at u300 | Wilson 95% readiness interval |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        row = summary["per_arm"][arm]
        interval = row["formal_readiness_wilson95"]
        interval_text = (
            f"[{_cell(interval['lower'])}, {_cell(interval['upper'])}]"
            if interval is not None
            else "— (incomplete; no CI)"
        )
        lines.append(
            f"| {arm} | {row['valid_formal_blocks']} | {row['expected_formal_blocks']} "
            f"| {row['formal_readiness_passes']} | {interval_text} |"
        )

    lines += [
        "",
        "## Paired contrasts",
        "",
        "| Contrast (candidate − reference) | Complete pairs | Wins | Losses | Net gain | Exact two-sided p | Holm-adjusted p | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, candidate, reference in PRIMARY_COMPARISONS:
        comparison = summary["comparisons"][name]
        lines.append(
            f"| {candidate} − {reference} | {comparison['complete_pairs']} | "
            f"{comparison['wins']} | {comparison['losses']} | {_cell(comparison['net_gain'])} | "
            f"{_cell(comparison['exact_two_sided_p'])} | {_cell(comparison['holm_adjusted_p'])} | "
            f"{comparison['verdict']} |"
        )
    name, candidate, reference = SECONDARY_COMPARISON
    comparison = summary["comparisons"][name]
    lines += [
        "",
        f"Secondary affine-capacity contrast, {candidate} − {reference}: "
        f"{comparison['wins']} wins, {comparison['losses']} losses, net "
        f"{comparison['net_gain']}; exact two-sided p={_cell(comparison['exact_two_sided_p'])}; "
        f"{comparison['verdict']}. This contrast is not part of the primary Holm family.",
        "",
        summary["qualification_rule"],
        "",
        "## Predeclared checkpoint diagnostics",
        "",
        "| Arm | Complete trajectories | Ever-ready trajectories | Ever-ready fraction | First-ready trajectories before u300 | Lost readiness later | Loss fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = summary["checkpoint_grid"]["by_arm"][arm]
        ever = row["everready"]
        loss = row["readiness_loss_after_first"]
        lines.append(
            f"| {arm} | {row['complete_trajectories']}/{summary['expected_blocks']} | "
            f"{ever['successes']}/{ever['denominator']} | {_cell(ever['value'])} | "
            f"{loss['denominator']} | {loss['losses']} | {_cell(loss['value'])} |"
        )
    lines += [
        "",
        "Ever-ready counts any observed joint-ready checkpoint; incomplete trajectories with no observed "
        "pass are unresolved and excluded from its denominator. Readiness loss is the fraction "
        "of complete trajectories that first become joint-ready before u300 and fail at any later "
        "predeclared checkpoint. Missing checkpoint outcomes remain undefined.",
        "",
        f"Predeclared records present: {summary['checkpoint_grid']['observed_predeclared_records']}/"
        f"{summary['checkpoint_grid']['expected_records']}; complete trajectories: "
        f"{summary['checkpoint_grid']['complete_trajectories']}/"
        f"{summary['checkpoint_grid']['expected_trajectories']}.",
        "",
        "The budget contrast bundles bounds and gates, so it does not isolate a mechanism. "
        "The verdict is finite evidence for this K8 recipe; it does not establish guaranteed closure, "
        "whole-NCA behavior, or 3D transport.",
        "",
        "metrics.csv contains the compact measurements on the predeclared checkpoint grid. "
        "The runner retains the raw metric records.",
    ]
    (destination / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_metrics_csv(
        destination / "metrics.csv",
        records,
        expected_blocks=summary["expected_blocks"],
    )
    return summary


def plot(out: str | Path, records: list[dict[str, Any]]) -> Path | None:
    """Plot supported R/S checkpoint trajectories at sizes 32 and 64."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    indexed, _ = _checkpoint_index(records, expected_blocks=EXPECTED_BLOCKS)
    colors = {
        "neural": "#2878B5",
        "budget": "#E68632",
        "hybrid": "#31965D",
        "affine_hybrid": "#8B65A9",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharex=True, sharey=True)
    for ax, size in zip(axes, SIZE_COLUMNS):
        labels_seen: set[str] = set()
        panel_supported = False
        for block in range(EXPECTED_BLOCKS):
            for arm in ARMS:
                previous: tuple[int, float, float] | None = None
                for update in CHECKPOINTS:
                    record = indexed.get((block, arm, update))
                    metrics = record.get("metrics") if isinstance(record, dict) else None
                    metrics = metrics if isinstance(metrics, dict) else {}
                    size_map = metrics.get("sizes")
                    size_map = size_map if isinstance(size_map, dict) else {}
                    point = size_map.get(size, {})
                    point = point if isinstance(point, dict) else {}
                    x = _number(point.get("R_strict_pooled_T64"))
                    y = _number(point.get("S_retention64_to256"))
                    maps = _integer(point.get("retention_reference_maps"))
                    pixels = _integer(point.get("retention_reference_pixels"))
                    supported = (
                        x is not None
                        and y is not None
                        and maps is not None
                        and maps >= MIN_RETENTION_MAPS
                        and pixels is not None
                        and pixels >= MIN_RETENTION_PIXELS
                    )
                    if not supported:
                        previous = None
                        continue
                    panel_supported = True
                    label = arm if arm not in labels_seen else None
                    ax.scatter(
                        [x],
                        [y],
                        color=colors[arm],
                        s=19,
                        alpha=0.54,
                        label=label,
                        zorder=3,
                    )
                    if label is not None:
                        labels_seen.add(arm)
                    if previous is not None and update - previous[0] == 25:
                        ax.plot(
                            [previous[1], x],
                            [previous[2], y],
                            color=colors[arm],
                            alpha=0.24,
                            linewidth=0.8,
                            zorder=2,
                        )
                    if update == FORMAL_UPDATE:
                        ax.scatter(
                            [x],
                            [y],
                            marker="*",
                            s=90,
                            color=colors[arm],
                            edgecolors="black",
                            linewidths=0.35,
                            zorder=4,
                        )
                    previous = (update, x, y)
        ax.axvline(0.80, color="0.45", linestyle="--", linewidth=0.8)
        ax.axhline(0.95, color="0.45", linestyle="--", linewidth=0.8)
        ax.set(
            xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02),
            xlabel="R: strict paired reach at T64",
            ylabel="S: T64-to-T256 retention",
            title=f"Size {size}; stars mark formal u300",
        )
        if not panel_supported:
            ax.text(
                0.5,
                0.5,
                "No supported S points on the predeclared grid",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )
        if labels_seen:
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Hybrid-writer checkpoint trajectories (unsupported S values omitted)"
    )
    fig.tight_layout()
    path = destination / "reach_retention_trajectory.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

