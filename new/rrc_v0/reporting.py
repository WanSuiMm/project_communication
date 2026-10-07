"""Paired fixed-endpoint and checkpoint summaries for the RRC v0 screen."""
from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from hybrid_writer.reporting import exact_two_sided_binomial_p, wilson_interval
except ModuleNotFoundError as exc:
    # Direct execution starts with rrc_v0/ on sys.path; the helper is a sibling
    # package under new/.
    if exc.name != "hybrid_writer":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_writer.reporting import exact_two_sided_binomial_p, wilson_interval


ARMS = ("current", "factorized", "rrc")
PRIMARY_CANDIDATE = "rrc"
PRIMARY_REFERENCE = "factorized"
CHECKPOINTS = tuple(range(0, 301, 25))
FORMAL_UPDATE = 300
EXPECTED_BLOCKS = 8
NET_GAIN_THRESHOLD = 6
PRIMARY_ALPHA = 0.05
SIZE_COLUMNS = ("32", "64")

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


def _record_update(row: dict[str, Any]) -> int | None:
    update = _integer(row.get("update"))
    if update is not None:
        return update
    # Some writers use an integer checkpoint field; path-valued checkpoint
    # metadata is deliberately not interpreted as an update number.
    return _integer(row.get("checkpoint"))


def _readiness(row: Any) -> bool | None:
    if not isinstance(row, dict):
        return None
    for key in ("readiness", "joint"):
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, dict) and isinstance(value.get("pass"), bool):
            return value["pass"]
    return None


def _formal_rows(
    final_rows: list[dict[str, Any]], expected_blocks: int
) -> tuple[dict[tuple[int, str], dict[str, Any]], int, int]:
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    seen: set[tuple[int, str]] = set()
    invalid = duplicates = 0
    for row in final_rows:
        if not isinstance(row, dict):
            invalid += 1
            continue
        arm = row.get("arm")
        block = _integer(row.get("block"))
        update_value = row.get("update")
        update = FORMAL_UPDATE if update_value is None else _integer(update_value)
        valid = (
            arm in ARMS
            and block is not None
            and block in range(expected_blocks)
            and update == FORMAL_UPDATE
            and _readiness(row) is not None
            and ("formal_endpoint" not in row or row.get("formal_endpoint") is True)
        )
        if not valid:
            invalid += 1
            continue
        key = (block, arm)
        if key in seen:
            duplicates += 1
            indexed.pop(key, None)
            continue
        seen.add(key)
        indexed[key] = row
    return indexed, invalid, duplicates


def _checkpoint_index(
    records: list[dict[str, Any]], expected_blocks: int
) -> tuple[dict[tuple[int, str, int], dict[str, Any]], dict[str, int]]:
    indexed: dict[tuple[int, str, int], dict[str, Any]] = {}
    seen: set[tuple[int, str, int]] = set()
    counts = {
        "observed_predeclared_records": 0,
        "ignored_non_grid_records": 0,
        "ignored_out_of_range_records": 0,
        "invalid_records": 0,
        "duplicate_predeclared_records": 0,
    }
    grid = set(CHECKPOINTS)
    for row in records:
        if not isinstance(row, dict):
            counts["invalid_records"] += 1
            continue
        arm = row.get("arm")
        block = _integer(row.get("block"))
        if (
            arm not in ARMS
            or block is None
            or block < 0
            or block not in range(expected_blocks)
        ):
            counts["ignored_out_of_range_records"] += 1
            continue
        update = _record_update(row)
        if update is None:
            counts["invalid_records"] += 1
            continue
        if update not in grid:
            counts["ignored_non_grid_records"] += 1
            continue
        key = (block, arm, update)
        if key in seen:
            counts["duplicate_predeclared_records"] += 1
            indexed.pop(key, None)
            continue
        seen.add(key)
        if _readiness(row) is None:
            counts["invalid_records"] += 1
            continue
        indexed[key] = row
        counts["observed_predeclared_records"] += 1
    return indexed, counts


def _checkpoint_summary(
    indexed: dict[tuple[int, str, int], dict[str, Any]], expected_blocks: int
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    expected_per_arm = expected_blocks * len(CHECKPOINTS)
    for arm in ARMS:
        ever_ready = known_nonready = indeterminate = 0
        first_ready_paths = readiness_losses = complete_paths = observed = 0
        for block in range(expected_blocks):
            path = [indexed.get((block, arm, update)) for update in CHECKPOINTS]
            observed += sum(row is not None for row in path)
            readiness = [_readiness(row) for row in path]
            has_unknown = any(value is None for value in readiness)
            if any(value is True for value in readiness):
                ever_ready += 1
            elif has_unknown:
                indeterminate += 1
            else:
                known_nonready += 1
            if has_unknown:
                continue
            complete_paths += 1
            if any(readiness):
                first = readiness.index(True)
                if first < len(CHECKPOINTS) - 1:
                    first_ready_paths += 1
                    if not all(readiness[first + 1 :]):
                        readiness_losses += 1

        ever_denominator = ever_ready + known_nonready
        summaries[arm] = {
            "expected_checkpoint_records": expected_per_arm,
            "observed_checkpoint_records": observed,
            "complete_trajectories": complete_paths,
            "incomplete_trajectories": expected_blocks - complete_paths,
            "everready": {
                "successes": ever_ready,
                "known_nonready": known_nonready,
                "indeterminate_no_pass_trajectories": indeterminate,
                "denominator": ever_denominator,
                "expected_trajectories": expected_blocks,
                "value": ever_ready / ever_denominator if ever_denominator else None,
            },
            "readiness_loss_after_first": {
                "losses": readiness_losses,
                "denominator": first_ready_paths,
                "value": (
                    readiness_losses / first_ready_paths if first_ready_paths else None
                ),
            },
        }
    return summaries


def _full_secondary(
    formal: dict[tuple[int, str], dict[str, Any]], expected_blocks: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        values = []
        for block in range(expected_blocks):
            row = formal.get((block, arm))
            metrics = row.get("metrics") if isinstance(row, dict) else None
            value = metrics.get("full_pass") if isinstance(metrics, dict) else None
            values.append(value if isinstance(value, bool) else None)
        observed = [value for value in values if value is not None]
        successes = sum(observed)
        result[arm] = {
            "passes": successes,
            "observed": len(observed),
            "unknown": expected_blocks - len(observed),
            "rate": successes / len(observed) if observed else None,
        }
    return result


def _final_metric_summary(
    formal: dict[tuple[int, str], dict[str, Any]], expected_blocks: int
) -> dict[str, Any]:
    fields = (
        "R_strict_pooled_T64",
        "R_strict_mean_T64",
        "S_retention64_to256",
    )
    result: dict[str, Any] = {}
    for arm in ARMS:
        result[arm] = {}
        for size in SIZE_COLUMNS:
            collected: dict[str, list[float]] = {field: [] for field in fields}
            for block in range(expected_blocks):
                row = formal.get((block, arm))
                metrics = row.get("metrics") if isinstance(row, dict) else None
                sizes = metrics.get("sizes") if isinstance(metrics, dict) else None
                point = sizes.get(size) if isinstance(sizes, dict) else None
                if point is None and isinstance(sizes, dict):
                    point = sizes.get(int(size))
                point = point if isinstance(point, dict) else {}
                for field in fields:
                    value = _number(point.get(field))
                    if value is not None:
                        collected[field].append(value)
            result[arm][size] = {
                field: {
                    "mean": sum(values) / len(values) if values else None,
                    "observed": len(values),
                    "expected": expected_blocks,
                }
                for field, values in collected.items()
            }
    return result


def aggregate(
    final_rows: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
    expected_blocks: int = EXPECTED_BLOCKS,
) -> dict[str, Any]:
    """Aggregate RRC-vs-Factorized at u300 and secondary checkpoint diagnostics."""
    if expected_blocks != EXPECTED_BLOCKS:
        raise ValueError("RRC v0 is frozen at exactly 8 paired blocks")

    formal, invalid_formal_rows, duplicate_formal_rows = _formal_rows(
        final_rows, expected_blocks
    )
    complete_by_arm = {
        arm: sum((block, arm) in formal for block in range(expected_blocks))
        for arm in ARMS
    }
    complete_pairs = wins = losses = both_pass = neither_pass = 0
    per_block: list[dict[str, Any]] = []
    for block in range(expected_blocks):
        rrc_pass = _readiness(formal.get((block, PRIMARY_CANDIDATE)))
        factorized_pass = _readiness(formal.get((block, PRIMARY_REFERENCE)))
        current_pass = _readiness(formal.get((block, "current")))
        if rrc_pass is None or factorized_pass is None:
            comparison = "missing_or_invalid_formal_pair"
        else:
            complete_pairs += 1
            if rrc_pass and not factorized_pass:
                wins += 1
                comparison = "rrc_only"
            elif factorized_pass and not rrc_pass:
                losses += 1
                comparison = "factorized_only"
            elif rrc_pass:
                both_pass += 1
                comparison = "both_ready"
            else:
                neither_pass += 1
                comparison = "neither_ready"
        per_block.append(
            {
                "block": block,
                "status": "paired" if comparison != "missing_or_invalid_formal_pair" else comparison,
                "current_ready": current_pass,
                "factorized_ready": factorized_pass,
                "rrc_ready": rrc_pass,
                "comparison": comparison,
            }
        )

    formal_complete = complete_pairs == expected_blocks and all(
        complete_by_arm[arm] == expected_blocks for arm in ARMS
    )
    p_value = exact_two_sided_binomial_p(wins, losses) if complete_pairs == expected_blocks else None
    net_gain = wins - losses

    indexed, record_counts = _checkpoint_index(dense_records, expected_blocks)
    checkpoint_by_arm = _checkpoint_summary(indexed, expected_blocks)
    expected_records = expected_blocks * len(ARMS) * len(CHECKPOINTS)
    expected_trajectories = expected_blocks * len(ARMS)
    complete_trajectories = sum(
        checkpoint_by_arm[arm]["complete_trajectories"] for arm in ARMS
    )
    grid_complete = (
        record_counts["observed_predeclared_records"] == expected_records
        and record_counts["duplicate_predeclared_records"] == 0
        and record_counts["invalid_records"] == 0
        and complete_trajectories == expected_trajectories
    )
    content_complete = formal_complete and grid_complete
    primary_qualified = (
        bool(net_gain >= NET_GAIN_THRESHOLD and p_value <= PRIMARY_ALPHA)
        if content_complete and p_value is not None
        else None
    )
    verdict = (
        "INCOMPLETE"
        if not content_complete
        else "RRC_RELIABILITY_QUALIFIED"
        if primary_qualified
        else "NO_RRC_RELIABILITY_QUALIFICATION"
    )

    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        rows = [
            formal[(block, arm)]
            for block in range(expected_blocks)
            if (block, arm) in formal
        ]
        ready = sum(_readiness(row) is True for row in rows)
        arm_complete = len(rows) == expected_blocks
        per_arm[arm] = {
            "valid_formal_blocks": len(rows),
            "expected_formal_blocks": expected_blocks,
            "formal_readiness_passes": ready,
            "formal_readiness_rate": ready / len(rows) if rows else None,
            "formal_readiness_wilson95": (
                wilson_interval(ready, len(rows)) if arm_complete else None
            ),
            "formal_readiness_ci_status": (
                "COMPLETE_BLOCKS" if arm_complete else "INCOMPLETE_NO_CI"
            ),
        }

    return {
        "schema": "rrc-v0-paired-report-v1",
        "status": "COMPLETE" if content_complete else "INCOMPLETE",
        "expected_blocks": expected_blocks,
        "arms": list(ARMS),
        "formal_update": FORMAL_UPDATE,
        "predeclared_checkpoints": list(CHECKPOINTS),
        "formal_rows": {
            "valid_rows": len(formal),
            "invalid_rows": invalid_formal_rows,
            "duplicate_rows": duplicate_formal_rows,
            "complete_pairs": complete_pairs,
            "expected_pairs": expected_blocks,
            "complete_blocks_by_arm": complete_by_arm,
            "complete": formal_complete,
        },
        "checkpoint_grid": {
            **record_counts,
            "expected_records": expected_records,
            "expected_size_rows": expected_records * len(SIZE_COLUMNS),
            "exported_size_rows": len(indexed) * len(SIZE_COLUMNS),
            "complete_trajectories": complete_trajectories,
            "expected_trajectories": expected_trajectories,
            "complete": grid_complete,
            "by_arm": checkpoint_by_arm,
        },
        "overall_verdict": verdict,
        "primary_comparison": {
            "candidate": PRIMARY_CANDIDATE,
            "reference": PRIMARY_REFERENCE,
            "endpoint": "joint readiness at u300",
            "complete_pairs": complete_pairs,
            "expected_pairs": expected_blocks,
            "wins": wins,
            "losses": losses,
            "both_ready": both_pass,
            "neither_ready": neither_pass,
            "discordant_pairs": wins + losses,
            "net_gain": net_gain,
            "exact_two_sided_p": p_value,
            "net_gain_threshold": NET_GAIN_THRESHOLD,
            "net_gain_threshold_met": (
                net_gain >= NET_GAIN_THRESHOLD if formal_complete else None
            ),
            "alpha": PRIMARY_ALPHA,
            "reliability_qualified": primary_qualified,
            "verdict": verdict,
            "per_block": per_block,
        },
        "per_arm": per_arm,
        "old_full_secondary": _full_secondary(formal, expected_blocks),
        "final_metrics": _final_metric_summary(formal, expected_blocks),
        "qualification_rule": (
            "The sole primary screen is RRC-only minus Factorized-only joint-ready "
            "blocks at u300: net gain must be at least 6/8 and the exact two-sided "
            "binomial sign-test p value over discordant pairs must be at most 0.05. "
            "Current and all intermediate checkpoints are secondary."
        ),
        "claim_boundary": (
            "Finite evidence for this frozen RRC v0 recipe and eight paired blocks. "
            "The u300 RRC-versus-Factorized contrast is the sole reliability screen; "
            "Current, old Full, and ever-ready checkpoint counts are descriptive "
            "secondary outcomes. No peak checkpoint replaces u300."
        ),
    }


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _bool_cell(value: bool | None) -> str:
    if value is True:
        return "ready"
    if value is False:
        return "not ready"
    return "unknown"


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    update = _record_update(record)
    if update is None and record.get("formal_endpoint") is True:
        update = FORMAL_UPDATE
    return {
        "block": record.get("block"),
        "init_seed": record.get("init_seed", record.get("initialization_seed")),
        "initialization_seed": record.get("initialization_seed", record.get("init_seed")),
        "schedule_seed": record.get("schedule_seed"),
        "arm": record.get("arm"),
        "update": update,
        "checkpoint": record.get("checkpoint", record.get("checkpoint_path")),
        "model_sha": record.get(
            "model_sha", record.get("model_sha256", record.get("parameter_sha256"))
        ),
        "checkpoint_sha256": record.get("checkpoint_sha256"),
        "last_loss": record.get("last_loss"),
        "seconds": record.get("seconds"),
        "evaluation_seconds": record.get("evaluation_seconds"),
        "formal_endpoint": record.get("formal_endpoint"),
    }


def _size_metrics(record: dict[str, Any], size: str) -> dict[str, Any]:
    metrics = record.get("metrics")
    sizes = metrics.get("sizes") if isinstance(metrics, dict) else None
    point = sizes.get(size) if isinstance(sizes, dict) else None
    if point is None and isinstance(sizes, dict):
        point = sizes.get(int(size))
    return point if isinstance(point, dict) else {}


def _write_metrics_csv(
    path: Path, records: list[dict[str, Any]], expected_blocks: int
) -> None:
    fields = (
        "block",
        "init_seed",
        "initialization_seed",
        "schedule_seed",
        "arm",
        "update",
        "checkpoint",
        "model_sha",
        "checkpoint_sha256",
        "last_loss",
        "seconds",
        "evaluation_seconds",
        "size",
        *_METRIC_COLUMNS,
        "joint_readiness",
        "full_pass",
        "formal_endpoint",
    )
    indexed, _ = _checkpoint_index(records, expected_blocks)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for block, arm, update in sorted(
            indexed, key=lambda key: (key[0], ARMS.index(key[1]), key[2])
        ):
            record = indexed[(block, arm, update)]
            metrics = record.get("metrics")
            metrics = metrics if isinstance(metrics, dict) else {}
            readiness = _readiness(record)
            for size in SIZE_COLUMNS:
                point = _size_metrics(record, size)
                writer.writerow(
                    {
                        **_metadata(record),
                        "block": block,
                        "arm": arm,
                        "update": update,
                        "size": size,
                        **{key: point.get(key) for key in _METRIC_COLUMNS},
                        "joint_readiness": readiness if size == "32" else None,
                        "full_pass": metrics.get("full_pass"),
                    }
                )


def _write_final_metrics_csv(
    path: Path,
    final_rows: list[dict[str, Any]],
    expected_blocks: int,
) -> None:
    fields = (
        "block",
        "init_seed",
        "initialization_seed",
        "schedule_seed",
        "arm",
        "update",
        "checkpoint",
        "model_sha",
        "checkpoint_sha256",
        "last_loss",
        "seconds",
        "evaluation_seconds",
        "size",
        *_METRIC_COLUMNS,
        "joint_readiness",
        "full_pass",
        "formal_endpoint",
    )
    formal, _, _ = _formal_rows(final_rows, expected_blocks)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for block, arm in sorted(
            formal, key=lambda key: (key[0], ARMS.index(key[1]))
        ):
            record = formal[(block, arm)]
            metrics = record.get("metrics")
            metrics = metrics if isinstance(metrics, dict) else {}
            readiness = _readiness(record)
            metadata = _metadata(record)
            for size in SIZE_COLUMNS:
                point = _size_metrics(record, size)
                writer.writerow(
                    {
                        **metadata,
                        "block": block,
                        "arm": arm,
                        "update": FORMAL_UPDATE,
                        "size": size,
                        **{key: point.get(key) for key in _METRIC_COLUMNS},
                        "joint_readiness": readiness if size == "32" else None,
                        "full_pass": metrics.get("full_pass"),
                    }
                )


def _status_fields(status: Any) -> tuple[str, bool | None, list[str]]:
    if status is None:
        return "UNSPECIFIED", None, []
    if isinstance(status, dict):
        label = status.get("status", status.get("run_status", "UNKNOWN"))
        error = status.get("error")
        errors = status.get("errors")
        messages = [str(error)] if error else []
        if isinstance(errors, list):
            messages.extend(str(item) for item in errors if item)
        elif errors:
            messages.append(str(errors))
    else:
        label = status
        messages = []
    normalized = str(label).upper()
    execution_complete = normalized in {
        "COMPLETE",
        "COMPLETED",
        "SUCCESS",
        "SUCCEEDED",
    }
    return str(label), execution_complete, messages


def report(
    folder: str | Path,
    final_records: list[dict[str, Any]],
    dense_records: list[dict[str, Any]],
    status: Any = None,
) -> dict[str, Any]:
    """Write the RRC report and CSVs, returning the compact summary."""
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    summary = aggregate(final_records, dense_records)
    execution_label, execution_complete, execution_errors = _status_fields(status)
    summary["execution_status"] = execution_label
    summary["execution_errors"] = execution_errors
    if execution_complete is False:
        summary["status"] = "INCOMPLETE"
        summary["overall_verdict"] = "INCOMPLETE"
        summary["primary_comparison"]["reliability_qualified"] = None
        summary["primary_comparison"]["verdict"] = "INCOMPLETE"

    primary = summary["primary_comparison"]
    formal = summary["formal_rows"]
    grid = summary["checkpoint_grid"]
    lines = [
        "# RRC v0 paired reliability screen",
        "",
        f"Execution status: {execution_label}",
        f"Aggregation status: {summary['status']}",
        f"Overall verdict: {summary['overall_verdict']}",
    ]
    if execution_errors:
        lines += ["", "Execution errors:", *(f"- {item}" for item in execution_errors)]
    lines += [
        "",
        "The formal endpoint is joint readiness at u300. The sole primary contrast is RRC versus Factorized; Current, old Full, and intermediate checkpoint outcomes are secondary.",
        "",
        "| Arm | Valid u300 blocks | Expected | Joint-ready | Wilson 95% readiness interval | Old Full passes | Full observed |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for arm in ARMS:
        row = summary["per_arm"][arm]
        interval = row["formal_readiness_wilson95"]
        interval_text = (
            f"[{_cell(interval['lower'])}, {_cell(interval['upper'])}]"
            if interval is not None
            else "— (incomplete; no CI)"
        )
        full = summary["old_full_secondary"][arm]
        lines.append(
            f"| {arm} | {row['valid_formal_blocks']} | {row['expected_formal_blocks']} "
            f"| {row['formal_readiness_passes']} | {interval_text} "
            f"| {full['passes']} | {full['observed']}/{row['expected_formal_blocks']} |"
        )

    primary_result = (
        "INCOMPLETE"
        if primary["reliability_qualified"] is None
        else "QUALIFIED" if primary["reliability_qualified"] else "NOT QUALIFIED"
    )
    lines += [
        "",
        "## Sole primary contrast",
        "",
        "| Contrast | Complete pairs | RRC-only wins | Factorized-only losses | Net gain | Exact two-sided p | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| RRC − Factorized at u300 | {primary['complete_pairs']}/{primary['expected_pairs']} | "
        f"{primary['wins']} | {primary['losses']} | {_cell(primary['net_gain'])} | "
        f"{_cell(primary['exact_two_sided_p'])} | {primary_result} |",
        "",
        summary["qualification_rule"],
        "",
        "Current is a secondary descriptive arm and does not enter the primary test.",
        "",
        "## Paired block outcomes",
        "",
        "| Block | Current | Factorized | RRC | Primary RRC − Factorized |",
        "|---:|---|---|---|---|",
    ]
    for row in primary["per_block"]:
        lines.append(
            f"| {row['block']} | {_bool_cell(row['current_ready'])} "
            f"| {_bool_cell(row['factorized_ready'])} | {_bool_cell(row['rrc_ready'])} "
            f"| {row['comparison']} |"
        )

    lines += [
        "",
        "## Final u300 reach and retention metrics",
        "",
        "Values are arithmetic means across blocks with defined values; n reports the number of defined blocks.",
        "",
        "| Arm | Size | R strict pooled T64 | R strict map mean T64 | S retention T64→T256 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for size in SIZE_COLUMNS:
            metrics = summary["final_metrics"][arm][size]
            rendered = []
            for key in (
                "R_strict_pooled_T64",
                "R_strict_mean_T64",
                "S_retention64_to256",
            ):
                item = metrics[key]
                rendered.append(
                    f"{_cell(item['mean'])} (n={item['observed']}/{item['expected']})"
                )
            lines.append(
                f"| {arm} | {size} | {rendered[0]} | {rendered[1]} | {rendered[2]} |"
            )

    lines += [
        "",
        "## Dense checkpoint diagnostics (secondary)",
        "",
        "| Arm | Complete trajectories | Ever-ready trajectories | Ever-ready fraction | Readiness losses after first ready |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = grid["by_arm"][arm]
        ever = row["everready"]
        loss = row["readiness_loss_after_first"]
        lines.append(
            f"| {arm} | {row['complete_trajectories']}/{summary['expected_blocks']} "
            f"| {ever['successes']}/{ever['denominator']} | {_cell(ever['value'])} "
            f"| {loss['losses']}/{loss['denominator']} |"
        )
    lines += [
        "",
        f"Valid u300 rows: {formal['valid_rows']}/24; paired primary blocks: "
        f"{primary['complete_pairs']}/8. Predeclared dense records: "
        f"{grid['observed_predeclared_records']}/{grid['expected_records']}; complete trajectories: "
        f"{grid['complete_trajectories']}/{grid['expected_trajectories']}.",
        "",
        "Ever-ready is the fraction of trajectories with at least one observed joint-ready checkpoint; unresolved no-pass trajectories are excluded from its denominator. Intermediate checkpoints never replace the fixed u300 endpoint.",
        "",
        summary["claim_boundary"],
        "",
        f"final_metrics.csv contains u300 metrics by block, arm, and size. metrics.csv contains one row per dense checkpoint and size ({grid['expected_size_rows']} rows when all {grid['expected_records']} expected records are present).",
    ]
    (destination / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_final_metrics_csv(
        destination / "final_metrics.csv", final_records, EXPECTED_BLOCKS
    )
    _write_metrics_csv(destination / "metrics.csv", dense_records, EXPECTED_BLOCKS)
    return summary


def _synthetic(
    rrc_ready_blocks: set[int],
    factorized_ready_blocks: set[int] | None = None,
    current_ready_blocks: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    factorized_ready_blocks = factorized_ready_blocks or set()
    current_ready_blocks = current_ready_blocks or set()
    final_rows: list[dict[str, Any]] = []
    dense_records: list[dict[str, Any]] = []
    for block in range(EXPECTED_BLOCKS):
        block_ready = {
            "current": block in current_ready_blocks,
            "factorized": block in factorized_ready_blocks,
            "rrc": block in rrc_ready_blocks,
        }
        for arm in ARMS:
            ready = block_ready[arm]
            metrics = {
                "full_pass": bool(block == 0 and arm == "current"),
                "sizes": {
                    size: {
                        "R_strict_pooled_T64": 0.82,
                        "R_strict_mean_T64": 0.81,
                        "S_retention64_to256": 0.96,
                        "S_retention_numerator": 960,
                        "S_continuous_survival64_to256": 0.94,
                        "retention_reference_pixels": 1000,
                        "retention_reference_maps": 16,
                        "ever_regressed_fraction": 0.01,
                        "ever_regressed_numerator": 1,
                        "ever_regressed_denominator": 100,
                        "coverage_gain64_to256": 0.02,
                        "frontier_status": "not_evaluated",
                        "frontier_effect": None,
                    }
                    for size in SIZE_COLUMNS
                },
            }
            final_rows.append(
                {
                    "block": block,
                    "arm": arm,
                    "init_seed": 1000 + block,
                    "schedule_seed": 2000 + block,
                    "metrics": metrics,
                    "readiness": {"pass": ready},
                    "last_loss": 1.0,
                    "seconds": 2.0,
                    "formal_endpoint": True,
                }
            )
            for update in CHECKPOINTS:
                checkpoint_ready = ready
                # A non-final pass is intentionally present to verify that the
                # diagnostic never changes the fixed u300 primary comparison.
                if arm == "rrc" and block == 6 and update == 25:
                    checkpoint_ready = True
                if arm == "rrc" and block == 6 and update == FORMAL_UPDATE:
                    checkpoint_ready = False
                dense_records.append(
                    {
                        "block": block,
                        "arm": arm,
                        "init_seed": 1000 + block,
                        "schedule_seed": 2000 + block,
                        "update": update,
                        "checkpoint": f"u{update:03d}.pt",
                        "model_sha": f"synthetic-{block}-{arm}-{update}",
                        "metrics": metrics,
                        "readiness": {"pass": checkpoint_ready},
                        "last_loss": 1.0,
                        "seconds": 2.0,
                        "formal_endpoint": update == FORMAL_UPDATE,
                    }
                )
    return final_rows, dense_records


def _self_test() -> None:
    final_rows, dense_records = _synthetic(
        set(range(6)), current_ready_blocks={7}
    )
    summary = aggregate(final_rows, dense_records)
    primary = summary["primary_comparison"]
    assert summary["status"] == "COMPLETE"
    assert summary["overall_verdict"] == "RRC_RELIABILITY_QUALIFIED"
    assert primary["wins"] == 6 and primary["losses"] == 0
    assert primary["net_gain"] == 6
    assert primary["exact_two_sided_p"] == 0.03125
    assert primary["reliability_qualified"] is True
    assert summary["per_arm"]["current"]["formal_readiness_passes"] == 1
    # Block 6 is ever-ready at an intermediate point but not ready at u300.
    assert summary["checkpoint_grid"]["by_arm"]["rrc"]["everready"]["successes"] == 7
    assert summary["per_arm"]["rrc"]["formal_readiness_passes"] == 6
    assert summary["checkpoint_grid"]["observed_predeclared_records"] == 312

    failing_finals, failing_dense = _synthetic(set(range(5)))
    failing = aggregate(failing_finals, failing_dense)
    assert failing["status"] == "COMPLETE"
    assert failing["overall_verdict"] == "NO_RRC_RELIABILITY_QUALIFICATION"
    assert failing["primary_comparison"]["net_gain"] == 5

    partial = aggregate(final_rows, dense_records[:-1])
    assert partial["status"] == "INCOMPLETE"
    assert partial["overall_verdict"] == "INCOMPLETE"
    assert partial["primary_comparison"]["reliability_qualified"] is None

    with tempfile.TemporaryDirectory(
        prefix=".rrc-v0-report-selftest-",
        dir=Path(__file__).resolve().parent,
    ) as temp_dir:
        written = report(temp_dir, final_rows, dense_records)
        with (Path(temp_dir) / "metrics.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            dense_csv_rows = sum(1 for _ in csv.DictReader(handle))
        with (Path(temp_dir) / "final_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            final_csv_rows = sum(1 for _ in csv.DictReader(handle))
        assert dense_csv_rows == 624
        assert final_csv_rows == 48
        assert written["checkpoint_grid"]["exported_size_rows"] == 624
    print("RRC v0 reporting self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    parser.error("this module exposes report(); use --self-test for the CPU-only check")


if __name__ == "__main__":
    main()
